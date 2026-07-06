"""
The Maliuzhinets function ψ_Φ(w) for impedance-wedge diffraction.

Integral definition (Kotelnikov et al., Phys. Rev. A 87, 023828, 2013, Eq. 21):

    ψ_Φ(w) = exp[−(1/2) ∫₀^∞ (cosh(wη)−1) / (η cosh(πη/2) sinh(2Φη)) dη]

where 2Φ is the exterior wedge angle.

Properties:
  - ψ_Φ(0) = 1
  - ψ_Φ(−w) = ψ_Φ(w)  (even function)
  - Functional relation: ψ_Φ(w+2Φ)/ψ_Φ(w−2Φ) = cot(w/2 + π/4)
  - Regular in the strip |Re(w)| < π/2 + 2Φ
"""

using QuadGK

# Integration-variable floor below which the integrand
# (cosh(wη)−1)/(η cosh(πη/2) sinh(2Φη)) is replaced by its L'Hôpital limit
# w²/(4Φ). The numerator cosh(wη)−1 ≈ (wη)²/2 suffers catastrophic cancellation
# once (wη)² < eps, i.e. η ≲ √eps, so the floor must be √eps (not eps): below it
# the direct form has lost all significant digits, while the L'Hôpital limit is
# itself accurate to O(η²) = O(eps). AD-safe primal-type eps.
const MALIUZHINETS_ETA_FLOOR = sqrt(eps(Float64))

# Asymptotic crossover argument. The asymptotic integrand replaces cosh(a),
# cosh(b=πη/2) and sinh(c) by exp(·)/2; each drops an e^{−2·arg} term, so the
# approximation is valid exactly when the SMALLEST of the three arguments
# {Re(a), b, c} satisfies e^{−2·arg} < eps, i.e. arg > −½ log eps. Testing the
# binding (smallest) argument — not c alone — is what makes this correct: when Φ
# is large b = πη/2 ≪ c, so a c-only threshold drops the still-significant
# cosh(b) tail and breaks reciprocity. Below the crossover the full integrand is
# used, where the smallest argument is < arg so η is small and the remaining
# arguments stay well under the e^709 overflow.
const MALIUZHINETS_ASYMPTOTIC_ARG = -0.5 * log(eps(Float64))

# Underflow floor (joint exponent) for the asymptotic integrand 2 exp(a−b−c)/η.
# Its magnitude is exp(real(a)−b−c), NOT exp(−c): cosh(a) grows as e^{Re(a)η}, so
# the binding decay is the JOINT exponent real(a)−b−c ≈ (Re(w)−π/2−2Φ)η. Dropping
# on a c-only criterion truncates the still-significant tail near the strip edge
# (Re(w) → π/2+2Φ) and corrupts ψ_Φ. The term is negligible only when the joint
# exponent underflows below log floatmin(Float64) ≈ −708.4.
const MALIUZHINETS_UNDERFLOW_LOG = log(floatmin(Float64))

"""
    _log_psi_Phi_strip(w, Phi; rtol=1e-12)

Compute log ψ_Φ(w) via adaptive Gauss-Kronrod quadrature.
Only valid for |Re(w)| < π/2 + 2Φ (the convergence strip).

The integrand (cosh(wη)−1)/(η cosh(πη/2) sinh(2Φη)) has a removable singularity
at η=0 with limit w²/(4Φ). For large η, the integrand decays as
2 exp((|Re(w)| − π/2 − 2Φ)η)/η when |Re(w)| < π/2 + 2Φ.
"""
function _log_psi_Phi_strip(w::Number, Phi::Real; rtol::Real = 1e-12)
    wc = Complex(w)

    function integrand(eta::Real)
        if eta < MALIUZHINETS_ETA_FLOOR
            # L'Hôpital limit: (w²η²/2)/(η · 1 · 2Φη) = w²/(4Φ)
            return wc^2 / (4Phi)
        end

        a = wc * eta           # argument of cosh (complex for complex w)
        b = π * eta / 2        # argument of cosh in denominator (real)
        c = 2Phi * eta          # argument of sinh (real)

        if real(a) - b - c < MALIUZHINETS_UNDERFLOW_LOG   # joint exponent underflows → integrand ≈ 0
            return zero(wc)     # match the (possibly Dual) integrand element type
        elseif min(real(a), b, c) > MALIUZHINETS_ASYMPTOTIC_ARG   # all three exp(·) tails negligible
            # cosh(a)−1 ≈ exp(a)/2 (for Re(a)≥0, ensured by even symmetry)
            # cosh(b) ≈ exp(b)/2, sinh(c) ≈ exp(c)/2
            return 2.0 * exp(a - b - c) / eta
        else
            num = cosh(a) - 1
            den = eta * cosh(b) * sinh(c)
            return num / den
        end
    end

    val, _ = quadgk(integrand, 0.0, Inf; rtol = rtol)
    return -val / 2
end

"""
    psi_Phi(w, Phi; rtol=1e-12)

Evaluate the Maliuzhinets function ψ_Φ(w) for any complex w.

Uses quadrature within the convergence strip |Re(w)| < π/2 + 2Φ, and the
functional relation ψ_Φ(w+2Φ)/ψ_Φ(w−2Φ) = cot(w/2 + π/4) to extend
beyond the strip. Also exploits the even symmetry ψ_Φ(−w) = ψ_Φ(w).
"""
function psi_Phi(w::Number, Phi::Real; rtol::Real = 1e-12)
    wc = Complex(w)
    strip = π / 2 + 2Phi

    # Use even symmetry to ensure Re(w) ≥ 0
    if real(wc) < 0
        wc = -wc
    end

    # Reduce w into the convergence strip using the functional relation:
    # ψ(w) = cot((w-2Φ)/2 + π/4) · ψ(w - 4Φ)
    # Applied backwards: to evaluate ψ(w) with large Re(w), reduce by 4Φ steps.
    cot_factors = typeof(wc)[]   # parametric: holds Complex{Dual} under AD
    while real(wc) >= strip - eps(Float64)
        # ψ(wc) = cot((wc-2Φ)/2 + π/4) · ψ(wc - 4Φ)
        push!(cot_factors, cot((wc - 2Phi) / 2 + π / 4))
        wc -= 4Phi
    end

    # Now |Re(wc)| < strip, evaluate via quadrature
    result = exp(_log_psi_Phi_strip(wc, Phi; rtol = rtol))

    # Multiply back the cotangent factors
    for c in reverse(cot_factors)
        result *= c
    end

    return result
end
