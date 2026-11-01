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
const MIN_F_PRIME_ASYMPTOTIC_THRESHOLD = 35.0
const MIN_F_PRIME_CONTINUED_FRACTION_THRESHOLD = 16.0

# The fixed-depth fraction is qualified on the positive binary64 interval
# [16,35), including Dual inputs with binary64 primal values. Other real
# types retain their existing evaluator and precision contract.
@inline function _use_real_fprime_fraction(x::Real)
    primal = _primal_value(x)
    return primal isa Float64 &&
        MIN_F_PRIME_CONTINUED_FRACTION_THRESHOLD <= primal <
        MIN_F_PRIME_ASYMPTOTIC_THRESHOLD
end

function _F_utd_prime_fraction(x::Real)
    root = sqrt(x / 2)
    z = complex(root, root)
    tail = zero(z)
    # DLMF 7.9.1: sqrt(pi)*erfcx(z) has all-z denominators and
    # numerators 1/2,1,3/2,... . Factor its first two levels before
    # applying F'=(i+1/(2x))*F-i, so no nearly equal terms are subtracted.
    for j in 64:-1:3
        tail = (j / 2) / (z + tail)
    end
    w = inv(z + tail)
    denominator = z * (z + w) + 0.5
    return z * w / (2x * denominator)
end

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
    (x_primal >= zero(x_primal) && isfinite(x_primal)) ||
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
    F_utd_prime(x; asymptotic_threshold=35)

Return dF/dx. Positive binary64 arguments in `[16,35)` use a factored
continued fraction to avoid cancellation in the differential equation (DE).
Smaller arguments use the DE; large x uses the inverse-power series.
Other real scalar types retain the DE below the asymptotic crossover.
The calibrated crossover is fixed at `asymptotic_threshold=35`: an earlier
switch uses the series outside its validated range, while a later switch
exposes the differential equation to large-argument cancellation.

The complex overload is restricted to the passive transition sector and uses
its separately validated sector-wide regime policy.
"""
function F_utd_prime(x::Real; asymptotic_threshold::Real=35.0)
    x_primal = _primal_value(x)
    threshold_primal = _primal_value(asymptotic_threshold)
    (_number_isfinite(x) && x_primal > zero(x_primal)) ||
        throw(DomainError(x, "F_utd_prime requires finite x > 0"))
    threshold_primal == MIN_F_PRIME_ASYMPTOTIC_THRESHOLD ||
        throw(DomainError(
            asymptotic_threshold,
            "asymptotic_threshold must equal $(MIN_F_PRIME_ASYMPTOTIC_THRESHOLD); " *
            "earlier series use and later differential-equation evaluation are " *
            "both inaccurate",
        ))
    if _use_real_fprime_fraction(x)
        return _F_utd_prime_fraction(x)
    elseif x_primal < threshold_primal
        return (im + inv(2 * x)) * F_utd(x) - im
    end
    # The inverse-power branch is algebraic, so it also propagates Dual partials
    # without exposing their value component to cancellation in the DE.
    invx = one(x) / x
    power = invx
    series = zero(Complex{typeof(invx)})
    for (m, c) in enumerate(ASYM_F_COEFFS)
        series += -m * c * power * invx
        power *= invx
    end
    return series
end

# Internal real-axis derivative bundle used by cancellation-free transition
# differences. The large-x branch differentiates the same inverse-power series
# as `F_utd_prime`; applying the ODE recurrence there would subtract nearly
# equal terms and lose the derivatives that the compensation is meant to keep.
function _F_utd_derivatives3(x::Real)
    x_primal = _primal_value(x)
    (_number_isfinite(x) && x_primal > zero(x_primal)) ||
        throw(DomainError(x, "transition derivatives require finite x > 0"))

    value = F_utd(x)
    if x_primal < MIN_F_PRIME_ASYMPTOTIC_THRESHOLD
        inverse_x = inv(x)
        recurrence = im + inverse_x / 2
        first = _use_real_fprime_fraction(x) ? _F_utd_prime_fraction(x) :
                recurrence * value - im
        second = recurrence * first - value * inverse_x^2 / 2
        third = recurrence * second - first * inverse_x^2 + value * inverse_x^3
        return value, first, second, third
    end

    inverse_x = inv(x)
    power = inverse_x
    first = zero(value)
    second = zero(value)
    third = zero(value)
    for (order, coefficient) in enumerate(ASYM_F_COEFFS)
        next_power = power * inverse_x
        next_next_power = next_power * inverse_x
        first -= order * coefficient * next_power
        second += order * (order + 1) * coefficient * next_next_power
        third -= order * (order + 1) * (order + 2) * coefficient *
                 next_next_power * inverse_x
        power = next_power
    end
    return value, first, second, third
end
