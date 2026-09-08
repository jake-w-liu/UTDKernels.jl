"""
Canonical shadow-boundary switch, derivative kernel, Fourier multiplier, and
nonlinear-coordinate pullback coefficients for the `exp(+i*omega*t)` phasor
convention.
"""

using SpecialFunctions: AmosException, erfc, loggamma

const _SHADOW_MAX_MULTIPLIER_TERMS = 256

@inline function _shadow_binary_type(values::Real...)
    primal_type = promote_type(
        (typeof(float(_primal_value(value))) for value in values)...,
    )
    primal_type === BigFloat && throw(ArgumentError(
        "shadow-sensitivity production APIs do not expose a BigFloat path; " *
        "use an independent high-precision oracle",
    ))
    primal_type === Float16 && return Float32
    primal_type in (Float32, Float64) || throw(ArgumentError(
        "shadow-sensitivity production APIs support binary16, binary32, and " *
        "binary64 inputs",
    ))
    return primal_type
end

@inline function _shadow_complex_value(value::Number, name::AbstractString)
    _validate_finite_number(value, name)
    R = _shadow_binary_type(
        _primal_value(real(value)), _primal_value(imag(value)),
    )
    scalar_type = promote_type(
        R, typeof(float(real(value))), typeof(float(imag(value))),
    )
    coordinate = complex(
        convert(scalar_type, float(real(value))),
        convert(scalar_type, float(imag(value))),
    )
    _number_isfinite(coordinate) || throw(DomainError(
        value, "$name is outside the active numeric range",
    ))
    return coordinate, R
end

@inline function _shadow_real_values(first::Real, second::Real)
    R = _shadow_binary_type(first, second)
    scalar_type = promote_type(R, typeof(float(first)), typeof(float(second)))
    first_value = convert(scalar_type, float(first))
    second_value = convert(scalar_type, float(second))
    _number_isfinite(first_value) || throw(DomainError(first, "input must be finite"))
    _number_isfinite(second_value) || throw(DomainError(second, "input must be finite"))
    return first_value, second_value, R
end

@inline function _shadow_real_values(
    first::Real,
    second::Real,
    third::Real,
)
    R = _shadow_binary_type(first, second, third)
    scalar_type = promote_type(
        R, typeof(float(first)), typeof(float(second)), typeof(float(third)),
    )
    first_value = convert(scalar_type, float(first))
    second_value = convert(scalar_type, float(second))
    third_value = convert(scalar_type, float(third))
    _number_isfinite(first_value) || throw(DomainError(first, "input must be finite"))
    _number_isfinite(second_value) || throw(DomainError(second, "input must be finite"))
    _number_isfinite(third_value) || throw(DomainError(third, "input must be finite"))
    return first_value, second_value, third_value, R
end

@inline function _shadow_positive(value::Real, original, name::AbstractString)
    primal = _primal_value(value)
    (isfinite(primal) && primal > zero(primal)) || throw(DomainError(
        original, "$name must be finite and positive",
    ))
    return value
end

@inline function _shadow_quarter_phase(value::Real)
    quarter = one(value) / 4
    return complex(cospi(quarter), sinpi(quarter))
end

@inline _shadow_work_value(value, ::Type{Float32}) =
    _faddeeva_widen_float64(value)
@inline _shadow_work_value(value, ::Type{Float64}) = value

@inline function _shadow_quadratic_phase(value::Number, name::AbstractString)
    phase = value * value
    _number_isfinite(phase) || throw(DomainError(
        phase, "$name quadratic phase is outside the active numeric range",
    ))
    return phase
end

"""
    shadow_switch(q)

Evaluate the canonical Fresnel switch
`erfc(-exp(im*pi/4)*q)/2` for a finite binary real or complex signed
coordinate. The convention is `exp(+i*omega*t)`, and `shadow_switch(0)` is
exactly one half. Array inputs are evaluated elementwise.
"""
function shadow_switch(q::Number)
    coordinate, R = _shadow_complex_value(q, "shadow coordinate q")
    work_coordinate = _shadow_work_value(coordinate, R)
    _shadow_quadratic_phase(work_coordinate, "shadow switch")
    argument = -_shadow_quarter_phase(real(work_coordinate)) * work_coordinate
    value = try
        convert(typeof(coordinate), erfc(argument) / 2)
    catch error
        error isa AmosException || rethrow()
        throw(DomainError(q, "shadow switch exceeds the backend range"))
    end
    _number_isfinite(value) || throw(DomainError(
        value, "shadow switch is non-finite",
    ))
    return value
end

shadow_switch(q::AbstractArray{<:Number}) = map(shadow_switch, q)

"""
    shadow_sensitivity_kernel(q)
    shadow_sensitivity_kernel(s, kappa)

With one argument, evaluate the exact canonical derivative
`exp(im*pi/4)*exp(-im*q^2)/sqrt(pi)`. With finite real `s` and positive
`kappa`, evaluate the scaled physical-coordinate kernel
`sqrt(kappa)*K(sqrt(kappa)*s)`. Binary16 inputs widen to binary32; binary32,
binary64, and supported ForwardDiff arithmetic preserve their promoted type.
"""
function shadow_sensitivity_kernel(q::Number)
    coordinate, R = _shadow_complex_value(q, "shadow coordinate q")
    work_coordinate = _shadow_work_value(coordinate, R)
    phase = _shadow_quadratic_phase(work_coordinate, "shadow kernel")
    scalar = real(work_coordinate)
    pi_value = zero(scalar) + _typed_pi(scalar)
    value = convert(
        typeof(coordinate),
        _shadow_quarter_phase(scalar) * exp(-im * phase) / sqrt(pi_value),
    )
    _number_isfinite(value) || throw(DomainError(
        value, "shadow sensitivity kernel is non-finite",
    ))
    return value
end


shadow_sensitivity_kernel(q::AbstractArray{<:Number}) =
    map(shadow_sensitivity_kernel, q)

function shadow_sensitivity_kernel(s::Real, kappa::Real)
    coordinate, parameter, R = _shadow_real_values(s, kappa)
    _shadow_positive(parameter, kappa, "kappa")
    work_coordinate = _shadow_work_value(coordinate, R)
    work_parameter = _shadow_work_value(parameter, R)
    root = sqrt(work_parameter)
    canonical_coordinate = root * work_coordinate
    _number_isfinite(canonical_coordinate) || throw(DomainError(
        canonical_coordinate,
        "scaled shadow coordinate is outside the active numeric range",
    ))
    output_type = typeof(complex(zero(coordinate)))
    value = convert(
        output_type, root * shadow_sensitivity_kernel(canonical_coordinate),
    )
    _number_isfinite(value) || throw(DomainError(
        value, "scaled shadow sensitivity kernel is non-finite",
    ))
    return value
end

shadow_sensitivity_kernel(s::AbstractArray{<:Real}, kappa::Real) =
    map(value -> shadow_sensitivity_kernel(value, kappa), s)

@inline function _shadow_multiplier_phase(xi::Real, kappa::Real)
    frequency, parameter, R = _shadow_real_values(xi, kappa)
    _shadow_positive(parameter, kappa, "kappa")
    work_frequency = _shadow_work_value(frequency, R)
    work_parameter = _shadow_work_value(parameter, R)
    scaled_frequency = (work_frequency / sqrt(work_parameter)) / 2
    _number_isfinite(scaled_frequency) || throw(DomainError(
        scaled_frequency,
        "scaled Fourier frequency is outside the active numeric range",
    ))
    phase = scaled_frequency * scaled_frequency
    _number_isfinite(phase) || throw(DomainError(
        phase, "quadratic Fourier phase is outside the active numeric range",
    ))
    return phase, typeof(complex(zero(frequency)))
end

@inline function _shadow_multiplier_terms(terms::Integer)
    value = try
        Int(terms)
    catch error
        (error isa InexactError || error isa OverflowError) || rethrow()
        throw(DomainError(terms, "multiplier term count is outside Int range"))
    end
    1 <= value <= _SHADOW_MAX_MULTIPLIER_TERMS || throw(DomainError(
        terms,
        "multiplier term count must lie in 1:$_SHADOW_MAX_MULTIPLIER_TERMS",
    ))
    return value
end

"""
    shadow_sensitivity_multiplier(xi, kappa; terms=nothing)

Evaluate the exact Fourier multiplier `exp(im*xi^2/(4*kappa))` for finite real
`xi` and positive finite `kappa`. Set `terms` to an integer in `1:256` to
evaluate that many terms of its exponential series, beginning with the Dirac
term. Array inputs are evaluated elementwise.
"""
function shadow_sensitivity_multiplier(
    xi::Real,
    kappa::Real;
    terms::Union{Nothing,Integer}=nothing,
)
    phase, output_type = _shadow_multiplier_phase(xi, kappa)
    if terms === nothing
        value = convert(output_type, complex(cos(phase), sin(phase)))
        _number_isfinite(value) || throw(DomainError(
            value, "shadow-sensitivity multiplier is non-finite",
        ))
        return value
    end

    retained_terms = _shadow_multiplier_terms(terms)
    z = im * phase
    total = one(complex(phase))
    compensation = zero(total)
    term = total
    @inbounds for order in 1:(retained_terms - 1)
        term *= z / order
        _number_isfinite(term) || throw(DomainError(
            term, "truncated shadow multiplier exceeds the active range",
        ))
        total, compensation = _multipole_compensated_add(
            total, compensation, term,
        )
    end
    _number_isfinite(total) || throw(DomainError(
        total, "truncated shadow multiplier is non-finite",
    ))
    return convert(output_type, total)
end


shadow_sensitivity_multiplier(xi::Real, kappa::Real, terms::Integer) =
    shadow_sensitivity_multiplier(xi, kappa; terms)

function shadow_sensitivity_multiplier(
    xi::AbstractArray{<:Real},
    kappa::Real;
    terms::Union{Nothing,Integer}=nothing,
)
    return map(
        value -> shadow_sensitivity_multiplier(value, kappa; terms), xi,
    )
end

function _shadow_numeric_type(values::Number...)
    primal_type = promote_type(
        (typeof(float(_primal_value(component)))
         for value in values for component in (real(value), imag(value)))...,
    )
    primal_type === BigFloat && throw(ArgumentError(
        "shadow-sensitivity pullbacks do not expose a BigFloat production path",
    ))
    primal_type === Float16 && (primal_type = Float32)
    primal_type in (Float32, Float64) || throw(ArgumentError(
        "shadow-sensitivity pullbacks support binary16, binary32, and " *
        "binary64 inputs",
    ))
    scalar_type = promote_type(
        primal_type,
        (typeof(float(component))
         for value in values for component in (real(value), imag(value)))...,
    )
    return scalar_type, primal_type
end

"""
    shadow_sensitivity_pullback(
        f0, f1, f2, f3, f4, a, b, c, d, kappa,
    )

Return `(leading, first, second)` for the nonlinear-coordinate distributional
pullback through order `kappa^-2`. `f0` through `f4` are local probe
derivatives, while `a=g'(s0)>0`, `b=g''(s0)`, `c=g'''(s0)`, and `d=g''''(s0)`
describe the real coordinate map. Every input must be finite and `kappa` must
be positive.
"""
function shadow_sensitivity_pullback(
    f0::Number,
    f1::Number,
    f2::Number,
    f3::Number,
    f4::Number,
    a::Real,
    b::Real,
    c::Real,
    d::Real,
    kappa::Real,
)
    scalar_type, _ = _shadow_numeric_type(
        f0, f1, f2, f3, f4, a, b, c, d, kappa,
    )
    C = typeof(complex(zero(scalar_type)))
    values = (
        convert(C, complex(float(f0))),
        convert(C, complex(float(f1))),
        convert(C, complex(float(f2))),
        convert(C, complex(float(f3))),
        convert(C, complex(float(f4))),
    )
    all(_number_isfinite, values) || throw(DomainError(
        values, "probe derivatives must be finite",
    ))
    derivative = convert(scalar_type, float(a))
    second_map = convert(scalar_type, float(b))
    third_map = convert(scalar_type, float(c))
    fourth_map = convert(scalar_type, float(d))
    parameter = convert(scalar_type, float(kappa))
    _shadow_positive(derivative, a, "coordinate derivative a")
    _shadow_positive(parameter, kappa, "kappa")
    all(_number_isfinite, (second_map, third_map, fourth_map)) ||
        throw(DomainError((b, c, d), "higher coordinate derivatives must be finite"))

    inverse = inv(derivative)
    inverse2 = inverse * inverse
    inverse3 = inverse2 * inverse
    inverse4 = inverse2 * inverse2
    inverse5 = inverse4 * inverse
    inverse6 = inverse3 * inverse3
    inverse7 = inverse6 * inverse
    first_bracket = values[3] * inverse2 -
                    second_map * values[2] * inverse3
    leading = values[1]
    first = leading - im * (first_bracket / parameter) / 4
    second_bracket = values[5] * inverse4 -
                     6second_map * values[4] * inverse5 +
                     (15second_map^2 - 4derivative * third_map) *
                     values[3] * inverse6 +
                     (-15second_map^3 +
                      10derivative * second_map * third_map -
                      derivative^2 * fourth_map) * values[2] * inverse7
    second = first - ((second_bracket / parameter) / parameter) / 32
    _number_isfinite(first) && _number_isfinite(second) || throw(DomainError(
        (first, second), "shadow-sensitivity pullback exceeds the active range",
    ))
    return (leading=leading, first=first, second=second)
end

@inline function _shadow_bound_order(order::Integer)
    value = try
        Int(order)
    catch error
        (error isa InexactError || error isa OverflowError) || rethrow()
        throw(DomainError(order, "remainder order is outside Int range"))
    end
    1 <= value <= _SHADOW_MAX_MULTIPLIER_TERMS || throw(DomainError(
        order,
        "remainder order must lie in 1:$_SHADOW_MAX_MULTIPLIER_TERMS",
    ))
    return value
end

function _shadow_sensitivity_remainder_bound(
    omega::Real,
    kappa::Real,
    order::Integer,
    spectral_l1::Real,
)
    bandwidth, parameter, mass, R =
        _shadow_real_values(omega, kappa, spectral_l1)
    bandwidth_primal = _primal_value(bandwidth)
    mass_primal = _primal_value(mass)
    bandwidth_primal >= zero(bandwidth_primal) || throw(DomainError(
        omega, "bandwidth omega must be finite and nonnegative",
    ))
    _shadow_positive(parameter, kappa, "kappa")
    mass_primal >= zero(mass_primal) || throw(DomainError(
        spectral_l1, "spectral_l1 must be finite and nonnegative",
    ))
    retained_order = _shadow_bound_order(order)
    (iszero(bandwidth_primal) || iszero(mass_primal)) && return zero(bandwidth)

    log_bound = 2retained_order * log(bandwidth) -
                retained_order * (log(4one(parameter)) + log(parameter)) -
                loggamma(R(retained_order + 1)) + log(mass)
    primal_log_bound = _primal_value(log_bound)
    primal_log_bound <= log(floatmax(R)) || throw(DomainError(
        primal_log_bound, "shadow-sensitivity remainder bound overflows",
    ))
    primal_log_bound < log(nextfloat(zero(R))) &&
        return zero(bandwidth) + nextfloat(zero(R))
    value = exp(log_bound)
    _number_isfinite(value) || throw(DomainError(
        value, "shadow-sensitivity remainder bound is non-finite",
    ))
    return value
end

@inline function _shadow_halfplane_values(s::Real, kappa::Real)
    coordinate, parameter, _ = _shadow_real_values(s, kappa)
    _shadow_positive(parameter, kappa, "kappa")
    return coordinate, parameter
end

function _shadow_halfplane_coordinate(s::Real, kappa::Real)
    coordinate, parameter = _shadow_halfplane_values(s, kappa)
    doubled = 2parameter
    scale = _number_isfinite(doubled) ? sqrt(doubled) :
            sqrt(2one(parameter)) * sqrt(parameter)
    return scale * sin(coordinate / 2)
end

function _shadow_halfplane_coordinate_derivative(s::Real, kappa::Real)
    coordinate, parameter = _shadow_halfplane_values(s, kappa)
    halved = parameter / 2
    scale = iszero(_primal_value(halved)) ?
            sqrt(parameter) / sqrt(2one(parameter)) : sqrt(halved)
    return scale * cos(coordinate / 2)
end

function _shadow_halfplane_coordinate_second(s::Real, kappa::Real)
    coordinate, parameter = _shadow_halfplane_values(s, kappa)
    eighth = parameter / 8
    scale = iszero(_primal_value(eighth)) ?
            sqrt(parameter) / (2sqrt(2one(parameter))) : sqrt(eighth)
    return -scale * sin(coordinate / 2)
end

function _shadow_halfplane_switch(s::Real, kappa::Real)
    return shadow_switch(_shadow_halfplane_coordinate(s, kappa))
end

function _shadow_halfplane_first(s::Real, kappa::Real)
    coordinate = _shadow_halfplane_coordinate(s, kappa)
    derivative = _shadow_halfplane_coordinate_derivative(s, kappa)
    value = shadow_sensitivity_kernel(coordinate) * derivative
    _number_isfinite(value) || throw(DomainError(
        value, "half-plane first shadow derivative is non-finite",
    ))
    return value
end

function _shadow_halfplane_second(s::Real, kappa::Real)
    coordinate = _shadow_halfplane_coordinate(s, kappa)
    first_coordinate = _shadow_halfplane_coordinate_derivative(s, kappa)
    second_coordinate = _shadow_halfplane_coordinate_second(s, kappa)
    kernel = shadow_sensitivity_kernel(coordinate)
    value = kernel * (
        second_coordinate - 2im * coordinate * first_coordinate^2
    )
    _number_isfinite(value) || throw(DomainError(
        value, "half-plane second shadow derivative is non-finite",
    ))
    return value
end

for function_name in (
    :_shadow_halfplane_coordinate,
    :_shadow_halfplane_coordinate_derivative,
    :_shadow_halfplane_coordinate_second,
    :_shadow_halfplane_switch,
    :_shadow_halfplane_first,
    :_shadow_halfplane_second,
)
    @eval $function_name(values::AbstractArray{<:Real}, kappa::Real) =
        map(value -> $function_name(value, kappa), values)
end
