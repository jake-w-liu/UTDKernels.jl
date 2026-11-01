"""
Derivative of the UTD transition function.

Convention: F(x) = √(πx) exp(iπ/4) erfcx(exp(iπ/4) √x), x ≥ 0.
The DE F' = (i + 1/(2x)) F − i follows from d(erfcx)/dz = 2z erfcx − 2/√π
with z = exp(iπ/4) √x and z² = i x.
"""

function _asymptotic_F_coefficients(count::Int)
    coeffs = ComplexF64[]
    c = 1.0 + 0.0im
    for m in 1:count
        c = im * (m - 0.5) * c
        push!(coeffs, c)
    end
    return coeffs
end

const ASYM_F_COEFFS = _asymptotic_F_coefficients(32)
const MIN_F_MINUS_ONE_ASYMPTOTIC_THRESHOLD = 60.0
const MIN_F_PRIME_ASYMPTOTIC_THRESHOLD = 64.0
# The released keyword accepted 35; it remains an input alias for the current
# binary policy. The legacy BigFloat fixed-series route also starts here.
const _LEGACY_F_PRIME_THRESHOLD = 35.0
include("RealTransitionDerivatives.jl")

"""
    F_utd_minus_one(x; threshold=60)

Evaluate F(x) − 1 without subtracting two values near 1 at large x.
The calibrated crossover is fixed at `threshold=60`: an earlier switch uses
the asymptotic series outside its validated range, while a later switch can
lose `F(x)-1` to floating-point subtraction.

The complex overload is restricted to the passive transition sector and uses
its separately validated sector-wide regime policy; see
[`is_passive_transition_argument`](@ref).
"""
function F_utd_minus_one(x::Real; threshold::Real=60.0)
    x_primal = _primal_value(x)
    threshold_primal = _primal_value(threshold)
    (_number_isfinite(x) && x_primal >= zero(x_primal)) ||
        throw(DomainError(x, "F_utd_minus_one requires finite x ≥ 0"))
    threshold_primal == MIN_F_MINUS_ONE_ASYMPTOTIC_THRESHOLD ||
        throw(DomainError(
            threshold,
            "threshold must equal $(MIN_F_MINUS_ONE_ASYMPTOTIC_THRESHOLD); " *
            "earlier series use and later subtraction are both inaccurate",
        ))
    if x_primal < threshold_primal
        return F_utd(x) - 1
    end
    if _passive_supported_type(x) && _number_contains_ad(x)
        risk, _ = _real_transition_work_parameters((x,), _passive_base_type(x); native_power=2)
        if risk
            native = F_utd_minus_one(x_primal; threshold)
            return _transition_component_wide(
                x, :Fm1, :real, typeof(complex(x + zero(Float64))); native,
            )
        end
    end
    # `one(x) / x` promotes integer inputs while preserving AD number types.
    invx = one(x) / x
    power = one(invx)
    series = zero(Complex{typeof(invx)})
    for c in ASYM_F_COEFFS
        power *= invx
        series += c * power
    end
    return series
end

"""
    F_utd_prime(x; asymptotic_threshold=64)

Return dF/dx for finite positive real `x`. Binary inputs use the differential
identity below 4, Gaussian rational quadrature on `[4,64)`, and the 32-term
inverse-power series from 64. This avoids cancellation in the moderate
argument differential identity and premature asymptotic truncation.

`asymptotic_threshold` accepts the current calibration `64` and the historical
value `35`; both select this same binary policy. Other values raise
`DomainError`. Float32 real inputs retain the existing Float64 working type.
The existing BigFloat route remains limited to the fixed series for `x≥35`;
it does not provide arbitrary-precision accuracy, and smaller finite BigFloat
inputs require unavailable complex-BigFloat erfcx.

The complex overload is restricted to the passive transition sector and uses
its separately validated sector-wide regime policy.
"""
function F_utd_prime(x::Real; asymptotic_threshold::Real=64.0)
    x_primal = _primal_value(x)
    threshold_primal = _primal_value(asymptotic_threshold)
    (_number_isfinite(x) && x_primal > zero(x_primal)) ||
        throw(DomainError(x, "F_utd_prime requires finite x > 0"))
    threshold_primal in (MIN_F_PRIME_ASYMPTOTIC_THRESHOLD, _LEGACY_F_PRIME_THRESHOLD) ||
        throw(DomainError(
            asymptotic_threshold,
            "asymptotic_threshold must equal 64 or the historical alias 35; " *
            "both select the calibrated binary policy",
        ))
    if _passive_supported_type(x)
        x_primal >= _REAL_TRANSITION_GAUSSIAN_MINIMUM &&
            return _real_transition_derivative(x, 1)
        risk, _ = _real_transition_work_parameters((x,); native_power=2)
        risk && return _real_transition_derivative_wide(x, 1)
        argument = x + zero(Float64)
        value = F_utd(argument)
        # Divide the finite F value first: inv(2x) can overflow even though
        # F/(2x), and therefore F', remains representable.
        result = (value / argument) / 2 + im * (value - one(value))
        _number_isfinite(result) && return result
        return _real_transition_derivative_wide(x, 1)
    elseif x_primal < _LEGACY_F_PRIME_THRESHOLD
        argument = x + zero(Float64)
        value = F_utd(argument)
        return (value / argument) / 2 + im * (value - one(value))
    end
    return _real_transition_asymptotic_derivatives(x)[1]
end

# One real derivative owner supplies the compensated wedge difference and
# the public/AD derivatives. The primal retains the ordinary erfcx evaluator.
function _F_utd_derivatives3(x::Real)
    x_primal = _primal_value(x)
    (_number_isfinite(x) && x_primal > zero(x_primal)) ||
        throw(DomainError(x, "transition derivatives require finite x > 0"))
    value = F_utd(x)
    if _passive_supported_type(x) && x_primal >= _REAL_TRANSITION_GAUSSIAN_MINIMUM
        first, second, third = _real_transition_derivatives(x)
        return value, first, second, third
    end
    inverse_x = inv(x)
    recurrence = im + inverse_x / 2
    first = recurrence * value - im
    second = recurrence * first - value * inverse_x^2 / 2
    third = recurrence * second - first * inverse_x^2 + value * inverse_x^3
    return value, first, second, third
end
