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
# w²/(4Φ). This is now purely a removable-singularity / underflow guard: the
# numerator is evaluated as the cancellation-free identity cosh(wη)−1 =
# 2 sinh(wη/2)² (in the full-integrand else branch below), so no catastrophic
# cancellation occurs at any η and
# the floor no longer needs to depend on |w| (the earlier √eps derivation from
# "(wη)² < eps" silently assumed |w|≈1 and under-protected the |w|<1 case). The
# √eps value is retained because the L'Hôpital limit w²/(4Φ) is accurate to
# O(η²)=O(eps) over [0,√eps], so replacing the (now well-conditioned) integrand
# there costs ≲ √eps·O(eps) in the integral while still avoiding the 0/0 form at
# η→0 (numerator ~(wη)²/2 and denominator ~2Φη² both vanish). AD-safe primal eps.
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
function _log_psi_Phi_strip(
    w::Number,
    Phi::Real;
    rtol::Real=1e-12,
    segbuf=nothing,
)
    return _log_psi_Phi_strip(w, Phi, rtol, segbuf)
end

@inline function _log_psi_Phi_strip(w::Number, Phi::Real, rtol::Real, segbuf)
    # Normalize direct internal calls to the same floating computation type used
    # by the public recurrence.
    T = promote_type(typeof(float(real(w))), typeof(float(Phi)), Float64)
    wc = Complex{T}(w)

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
            # cosh(a)−1 = 2 sinh(a/2)² — the cancellation-free identity (valid for
            # complex a, AD-safe). The direct cosh(a)−1 loses all significant
            # digits once |a|²<eps (η ≲ √eps/|w|, ABOVE the floor when |w|<1),
            # capping psi_Phi at ~1e-10 for small |w|; the sinh form is exact.
            num = 2 * sinh(a / 2)^2
            den = eta * cosh(b) * sinh(c)
            return num / den
        end
    end

    val, estimated_error = if segbuf === nothing
        quadgk(integrand, 0.0, Inf; rtol = rtol)
    else
        quadgk(integrand, 0.0, Inf; rtol = rtol, segbuf = segbuf)
    end
    error_value = _primal_value(estimated_error)
    value_scale = max(one(error_value), _primal_value(abs(val)))
    requested = _primal_value(rtol) * value_scale
    error_value <= requested || throw(DomainError(
        (w, Phi, rtol),
        "Maliuzhinets quadrature could not meet the requested rtol; " *
        "estimated error $(error_value) exceeds $(requested). Use a larger " *
        "rtol or higher-precision inputs.",
    ))
    return -val / 2
end

const _MALIUZHINETS_MAX_RECURRENCE_STEPS = 100_000

function _new_maliuzhinets_segbuf(values::Number...)
    component_types = map(values) do value
        promote_type(typeof(float(real(value))), typeof(float(imag(value))))
    end
    T = promote_type(Float64, component_types...)
    CT = Complex{T}
    ET = typeof(abs(zero(CT)))
    # Sixteen slots cover the adaptive depth of the common exact-wedge calls and
    # avoid Vector growth during their first quadrature. Harder integrands can
    # still grow the buffer normally.
    return QuadGK.alloc_segbuf(Float64, CT, ET; size = 16)
end

"""
    psi_Phi(w, Phi; rtol=1e-12)

Evaluate the Maliuzhinets function ψ_Φ(w) for any complex w.

Uses quadrature within the convergence strip |Re(w)| < π/2 + 2Φ, and the
functional relation ψ_Φ(w+2Φ)/ψ_Φ(w−2Φ) = cot(w/2 + π/4) to extend
beyond the strip and to avoid the slowly decaying quadrature tail near its open
edge. Also exploits the even symmetry ψ_Φ(−w) = ψ_Φ(w).

Throws `DomainError` when `Phi` is too small for a `4Phi` recurrence step to
change `w` at the working precision or when reduction would exceed the bounded
recurrence budget. It also throws when the quadrature error estimate cannot meet
the requested `rtol`; use a larger tolerance or higher-precision inputs then.
"""
function psi_Phi(w::Number, Phi::Real; rtol::Real = 1e-12)
    return _psi_Phi(w, Phi, rtol, nothing)
end

@inline function _psi_Phi(w::Number, Phi::Real, rtol::Real, segbuf)
    _number_isfinite(w) ||
        throw(DomainError(w, "Maliuzhinets argument w must be finite"))
    isfinite(Phi) && Phi > zero(Phi) ||
        throw(DomainError(Phi, "Maliuzhinets half-angle Phi must be finite and positive"))
    isfinite(rtol) && rtol > zero(rtol) ||
        throw(DomainError(rtol, "quadrature tolerance rtol must be finite and positive"))

    # The recurrence multiplies by generally noninteger cotangent factors, so
    # integer and rational inputs need a floating computation type.
    T = promote_type(typeof(float(real(w))), typeof(float(Phi)), Float64)
    wc = Complex{T}(w)
    strip = π / 2 + 2Phi

    # Use even symmetry to ensure Re(w) ≥ 0
    if real(wc) < 0
        wc = -wc
    end

    # Each recurrence step evaluates one complex cotangent factor. Reject inputs
    # whose reduction would require unbounded time. The exact wedge solver needs
    # only a small number of steps; this high ceiling preserves that domain while
    # keeping the standalone public function finite.
    primal_type = typeof(float(_primal_value(real(wc))))
    precision_eps = eps(primal_type)
    strip_primal = _primal_value(strip)
    rtol_primal = _primal_value(rtol)
    # Direct integration becomes ill-conditioned when the tail-decay exponent
    # approaches zero at the open strip edge. Enter the exact recurrence early
    # enough that the expected eps/distance amplification stays below rtol.
    strip_margin = min(
        strip_primal / 2,
        max(32 * sqrt(precision_eps), 8 * precision_eps / rtol_primal),
    )
    reduction_boundary = strip_primal - strip_margin
    if _primal_value(real(wc)) >= reduction_boundary
        estimated_offset_steps = (real(wc) - reduction_boundary) / (4Phi)
        if !isfinite(estimated_offset_steps) ||
           estimated_offset_steps >= _MALIUZHINETS_MAX_RECURRENCE_STEPS
            throw(DomainError(
                (w, Phi),
                "Maliuzhinets reduction requires more than " *
                "$(_MALIUZHINETS_MAX_RECURRENCE_STEPS) recurrence steps",
            ))
        end
    end

    # Reduce w into the convergence strip using the functional relation:
    # ψ(w) = cot((w-2Φ)/2 + π/4) · ψ(w - 4Φ)
    # Applied backwards: to evaluate ψ(w) with large Re(w), reduce by 4Φ steps.
    # Left multiplication preserves the required reverse evaluation order while
    # keeping the recurrence workspace scalar.
    cot_product = one(wc)
    recurrence_steps = 0
    while _primal_value(real(wc)) >= reduction_boundary
        recurrence_steps >= _MALIUZHINETS_MAX_RECURRENCE_STEPS &&
            throw(DomainError(
                (w, Phi),
                "Maliuzhinets reduction exceeded the recurrence-step budget",
            ))
        # ψ(wc) = cot((wc-2Φ)/2 + π/4) · ψ(wc - 4Φ)
        next_wc = wc - 4Phi
        next_wc == wc && throw(DomainError(
            Phi,
            "Maliuzhinets half-angle Phi is too small to reduce w at this precision",
        ))
        cot_product = cot((wc - 2Phi) / 2 + π / 4) * cot_product
        recurrence_steps += 1
        wc = next_wc
    end

    # Now |Re(wc)| < strip, evaluate via quadrature
    result = exp(_log_psi_Phi_strip(wc, Phi, rtol, segbuf))
    return result * cot_product
end
