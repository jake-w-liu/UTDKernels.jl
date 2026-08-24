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

"""
    F_utd_minus_one(x; threshold=60)

Evaluate F(x) − 1 without subtracting two values near 1 at large x.
`threshold` may delay the asymptotic branch but must be at least 60, the
lowest crossover validated for this 32-term series.
"""
function F_utd_minus_one(x::Real; threshold::Real=60.0)
    (x >= 0 && isfinite(x)) || throw(DomainError(x, "F_utd_minus_one requires finite x ≥ 0"))
    (threshold >= MIN_F_MINUS_ONE_ASYMPTOTIC_THRESHOLD && isfinite(threshold)) ||
        throw(DomainError(
            threshold,
            "threshold must be finite and at least " *
            "$(MIN_F_MINUS_ONE_ASYMPTOTIC_THRESHOLD); smaller values select the " *
            "asymptotic series outside its validated range",
        ))
    if x < threshold
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

Return dF/dx. Moderate x uses the DE. Large x uses the inverse-power series
so the two leading terms of the DE do not cancel.
`asymptotic_threshold` may delay the series branch but must be at least 35,
the lowest validated crossover.
"""
function F_utd_prime(x::Real; asymptotic_threshold::Real=35.0)
    (x > 0 && isfinite(x)) || throw(DomainError(x, "F_utd_prime requires finite x > 0"))
    (asymptotic_threshold >= MIN_F_PRIME_ASYMPTOTIC_THRESHOLD &&
     isfinite(asymptotic_threshold)) ||
        throw(DomainError(
            asymptotic_threshold,
            "asymptotic_threshold must be finite and at least " *
            "$(MIN_F_PRIME_ASYMPTOTIC_THRESHOLD); smaller values select the " *
            "asymptotic series outside its validated range",
        ))
    if x < asymptotic_threshold
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
