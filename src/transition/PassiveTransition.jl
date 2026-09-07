"""
Passive-sheet transition evaluation under `exp(+i*omega*t)`.

For a passive wavenumber and nonnegative real geometrical factor, the
transition argument lies in `real(x)>=0`, `imag(x)<=0`. Mathematical square
roots remain principal; this file never uses the material-wave branch policy.
"""

const PASSIVE_SMALL_RADIUS = 0.35
const PASSIVE_LARGE_RADIUS = 48.0
const PASSIVE_FM1_RADIUS = 40.0
const PASSIVE_MAX_SMALL_TERMS = 180
const PASSIVE_MAX_ASYMPTOTIC_TERMS = 80
const PASSIVE_AXIS_ULPS = 64

"""
    PassiveSheetError

Error raised when a passive-only transition operation receives an argument
outside its verified sheet.
"""
struct PassiveSheetError <: Exception
    message::String
end

Base.showerror(io::IO, error::PassiveSheetError) =
    print(io, "PassiveSheetError: ", error.message)

@inline function _passive_base_type(x::Number)
    return promote_type(
        typeof(float(_primal_value(real(x)))),
        typeof(float(_primal_value(imag(x)))),
    )
end

@inline _passive_supported_type(x::Number) = _passive_base_type(x) in (Float32, Float64)

@inline function _passive_complex(x::Number)
    return complex(float(real(x)), float(imag(x)))
end

@inline function _passive_default_axis_rtol(x::Number)
    T = _passive_base_type(x)
    return PASSIVE_AXIS_ULPS * eps(T)
end

"""
    is_passive_transition_argument(x; axis_rtol=nothing) -> Bool

Return whether finite `x` lies in the passive transition sector
`real(x)>=0`, `imag(x)<=0`, allowing a scale-aware roundoff band of 64 ulps on
the two axes by default. `axis_rtol` may override that nonnegative relative
band. The predicate does not clamp or conjugate `x`.
"""
function is_passive_transition_argument(x::Number; axis_rtol=nothing)
    tolerance = axis_rtol === nothing ? _passive_default_axis_rtol(x) : axis_rtol
    tolerance_primal = _primal_value(tolerance)
    (isfinite(tolerance_primal) && tolerance_primal >= zero(tolerance_primal)) ||
        throw(ArgumentError("axis_rtol must be finite and nonnegative"))

    real_primal = _primal_value(real(x))
    imag_primal = _primal_value(imag(x))
    isfinite(real_primal) && isfinite(imag_primal) || return false
    scale = max(one(real_primal), hypot(real_primal, imag_primal))
    return real_primal / scale >= -tolerance_primal &&
           imag_primal / scale <= tolerance_primal
end

@inline function _require_passive_transition(x::Number)
    _number_isfinite(x) || throw(PassiveSheetError(
        "transition argument must be finite on the passive sheet",
    ))
    is_passive_transition_argument(x) || throw(PassiveSheetError(
        "transition argument must satisfy real(x)>=0 and imag(x)<=0",
    ))
    return _passive_complex(x)
end

@inline function _require_passive_supported(x::Number)
    _passive_supported_type(x) || throw(ArgumentError(
        "passive transition evaluation supports Float32, Float64, and " *
        "automatic-differentiation values based on those types; got $(typeof(x))",
    ))
    return x
end

@inline function _passive_constants(x::Number)
    scalar = real(x)
    pi_value = zero(scalar) + _typed_pi(scalar)
    phase = cis(pi_value / 4)
    return pi_value, phase, sqrt(pi_value) * phase
end

@inline function _passive_zero_bundle(x)
    scalar = real(x)
    infinity = zero(scalar) + oftype(_primal_value(scalar), Inf)
    singular = complex(infinity, infinity)
    value = zero(x)
    return (; F=value, Fm1=-one(x), Fp=singular, Fpp=singular)
end

function _passive_small_bundle(x)
    root = safe_sqrt(x)
    _, _, leading = _passive_constants(x)
    imaginary_unit = complex(zero(real(x)), one(real(x)))
    odd_coefficient = leading
    even_coefficient = -2imaginary_unit

    value = zero(leading * root)
    derivative_y = zero(value)
    second_derivative_y = zero(value)
    power_minus_one = one(root)
    power_minus_two = one(root)
    previous_magnitude = oftype(_primal_value(abs(root)), Inf)
    tolerance = eps(_passive_base_type(x)) / 8
    converged = false

    for order in 1:PASSIVE_MAX_SMALL_TERMS
        coefficient = if order == 1
            odd_coefficient
        elseif order == 2
            even_coefficient
        elseif isodd(order)
            odd_coefficient = 2imaginary_unit * odd_coefficient / (order - 1)
        else
            even_coefficient = 2imaginary_unit * even_coefficient / (order - 1)
        end

        power = power_minus_one * root
        term = coefficient * power
        value += term
        derivative_y += order * coefficient * power_minus_one
        if order >= 2
            second_derivative_y += order * (order - 1) * coefficient * power_minus_two
        end

        magnitude = _primal_value(abs(term))
        value_scale = max(one(magnitude), _primal_value(abs(value)))
        if order >= 10 && magnitude <= tolerance * value_scale &&
           magnitude <= previous_magnitude
            converged = true
            break
        end
        previous_magnitude = magnitude
        power_minus_two = power_minus_one
        power_minus_one = power
    end
    converged || throw(DomainError(x, "passive small-argument series did not converge"))

    first = derivative_y / (2root)
    second = second_derivative_y / (4root^2) - derivative_y / (4root^3)
    return (; F=value, Fm1=value - one(value), Fp=first, Fpp=second)
end

function _passive_asymptotic_bundle(x)
    inverse_x = inv(x)
    power = one(x)
    coefficient = one(x)
    residual = zero(x)
    first = zero(x)
    second = zero(x)
    previous_magnitude = oftype(_primal_value(abs(x)), Inf)
    tolerance = eps(_passive_base_type(x)) / 8
    imaginary_unit = complex(zero(real(x)), one(real(x)))

    for order in 1:PASSIVE_MAX_ASYMPTOTIC_TERMS
        coefficient *= imaginary_unit * (order - one(real(x)) / 2)
        power *= inverse_x
        term = coefficient * power
        magnitude = _primal_value(abs(term))
        magnitude > previous_magnitude && break
        residual += term
        first -= order * term * inverse_x
        second += order * (order + 1) * (term * inverse_x) * inverse_x
        magnitude <= tolerance * max(one(magnitude), _primal_value(abs(residual))) && break
        previous_magnitude = magnitude
    end
    value = one(x) + residual
    return (; F=value, Fm1=residual, Fp=first, Fpp=second)
end

function _passive_direct_bundle(x)
    _, phase, leading = _passive_constants(x)
    root = safe_sqrt(x)
    value = leading * root * _faddeeva_erfcx(phase * root)
    recurrence = complex(zero(real(x)), one(real(x))) + inv(2x)
    first = recurrence * value - complex(zero(real(x)), one(real(x)))
    second = recurrence * first - value / (2x^2)
    _number_isfinite(value) && _number_isfinite(first) && _number_isfinite(second) ||
        throw(DomainError(x, "passive direct transition evaluation is non-finite"))
    return (; F=value, Fm1=value - one(value), Fp=first, Fpp=second)
end

"""Internal passive bundle returning `F`, `F-1`, `F'`, and `F''`."""
function _passive_transition_all(x::Number)
    argument = _require_passive_supported(_require_passive_transition(x))
    radius = _primal_value(abs(argument))
    iszero(radius) && return _passive_zero_bundle(argument)
    radius <= PASSIVE_SMALL_RADIUS && return _passive_small_bundle(argument)
    radius >= PASSIVE_LARGE_RADIUS && return _passive_asymptotic_bundle(argument)

    direct = _passive_direct_bundle(argument)
    if radius >= PASSIVE_FM1_RADIUS
        tail = _passive_asymptotic_bundle(argument)
        return (; F=direct.F, Fm1=tail.Fm1, Fp=direct.Fp, Fpp=direct.Fpp)
    end
    return direct
end

@inline function _checked_passive_transition_output(value, x, quantity::AbstractString)
    _number_isfinite(value) || throw(DomainError(
        x,
        "$quantity is non-finite at the requested passive transition argument",
    ))
    return value
end

# More-specific passive complex value method. Nonpassive and unsupported
# complex types retain the pre-existing general principal-root erfcx path.
function F_utd(x::Complex)
    if _passive_supported_type(x) && is_passive_transition_argument(x)
        return _checked_passive_transition_output(
            _passive_transition_all(x).F, x, "transition function",
        )
    end
    return _F_utd_erfcx(x)
end

function F_utd_minus_one(x::Complex)
    return _checked_passive_transition_output(
        _passive_transition_all(x).Fm1, x, "transition residual F(x)-1",
    )
end

function F_utd_prime(x::Complex)
    iszero(_primal_value(abs(x))) &&
        throw(DomainError(x, "F_utd_prime requires nonzero passive x"))
    return _checked_passive_transition_output(
        _passive_transition_all(x).Fp, x, "transition derivative F'(x)",
    )
end

"""
    F_utd_second(x)

Evaluate the second derivative of the UTD transition function. Real inputs
must be finite and positive. Complex inputs must be finite, nonzero, and lie
in the passive transition sector; wrong-sheet requests raise
[`PassiveSheetError`](@ref).
"""
function F_utd_second(x::Real)
    primal = _primal_value(x)
    (isfinite(primal) && primal > zero(primal)) ||
        throw(DomainError(x, "F_utd_second requires finite x > 0"))
    return _checked_passive_transition_output(
        _passive_transition_all(complex(float(x))).Fpp,
        x,
        "transition second derivative F''(x)",
    )
end

function F_utd_second(x::Complex)
    iszero(_primal_value(abs(x))) &&
        throw(DomainError(x, "F_utd_second requires nonzero passive x"))
    return _checked_passive_transition_output(
        _passive_transition_all(x).Fpp, x, "transition second derivative F''(x)",
    )
end

"""
    passive_wavenumber(k0, attenuation)

Construct `k0*(1-im*attenuation)` for the package's `exp(+i*omega*t)`
convention. Both inputs must be finite, with `k0>0` and `attenuation>=0`.
Throws `DomainError` if their product cannot be represented finitely.
"""
function passive_wavenumber(k0::Real, attenuation::Real)
    k_value, attenuation_value = promote(float(k0), float(attenuation))
    k_primal = _primal_value(k_value)
    attenuation_primal = _primal_value(attenuation_value)
    (isfinite(k_primal) && k_primal > zero(k_primal)) ||
        throw(DomainError(k0, "k0 must be finite and positive"))
    (isfinite(attenuation_primal) && attenuation_primal >= zero(attenuation_primal)) ||
        throw(DomainError(attenuation, "attenuation must be finite and nonnegative"))
    result = complex(k_value, -k_value * attenuation_value)
    _number_isfinite(result) || throw(DomainError(
        (k0, attenuation),
        "passive wavenumber must be finite; k0*attenuation overflowed",
    ))
    return result
end
