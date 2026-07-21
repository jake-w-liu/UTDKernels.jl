"""
PEC wedge diffraction coefficients (Kouyoumjian–Pathak form).

Convention: exp(+iωt), outgoing ~ exp(−ikr).
"""

# PEC sign factors: σ_j for j=1..4
# Soft (Dirichlet): +1, +1, -1, -1
# Hard (Neumann):   +1, +1, +1, +1
const PEC_SIGMA_SOFT = (+1, +1, -1, -1)
const PEC_SIGMA_HARD = (+1, +1, +1, +1)

@inline function _complex_finite(z::Complex)
    return isfinite(real(z)) && isfinite(imag(z))
end

@inline function _transition_tolerances(::Type{T}) where {T<:Real}
    # Use machine-precision-derived windows; avoid hand-tuned absolute values.
    # a_j is quadratic in angular detuning, so sqrt(eps) is a practical
    # threshold for "numerically at transition" in Float64 arithmetic.
    τ = sqrt(eps(T))
    return (τa = τ, τs = τ)
end

@inline function _validate_effective_L(L::Number)
    if L isa Real
        (L > 0 && (isfinite(L) || isinf(L))) ||
            throw(DomainError(L, "L must be positive and finite (or +Inf for far-field limit)"))
        return
    end

    Lc = Complex(L)
    isfinite(real(Lc)) && isfinite(imag(Lc)) && abs(Lc) > 0 ||
        throw(DomainError(L, "Complex L must be finite and nonzero"))
    return
end

function _wrap_angle_centered(phi::Real, alpha::Real)
    # Use the AD-safe wrap (floor form) instead of `mod` so a Dual argument keeps
    # a finite derivative on wrap boundaries; value is unchanged.
    w = wrap_angle(phi + alpha / 2, alpha) - alpha / 2
    return w == -alpha / 2 ? alpha / 2 : w
end

@inline function _kp_term_sign_pm(j::Int)
    # KP four-term ordering:
    # j=1,3 -> (π + β)/(2n)
    # j=2,4 -> (π - β)/(2n)
    return (j == 1 || j == 3) ? +1 : -1
end

@inline function _kp_transition_detuning(j::Int, terms, n::Real)
    s = _kp_term_sign_pm(j)
    β = terms.beta[j]
    Nj = terms.Nj[j]
    u = 2 * n * π * Nj - β
    target = s == +1 ? π : -π
    # Return Δψ_j, the signed offset of ψ_j = (π + s·β)/(2n) from its nearest
    # cotangent pole s·N_jπ: Δψ_j = (π - s·u)/(2n) = -s·(u - target)/(2n).
    # The −s and 1/(2n) encode the term-dependent side and the ψ-scale, so the
    # exact-transition surrogate lands on the physically correct side of the
    # pole for all four terms (previously only the s=−1 terms were consistent).
    return -s * (u - target) / (2 * n)
end

function _effective_angles_for_kp(wedge::Wedge, ang::RayAngles)
    alpha = wedge.alpha
    phi  = wrap_angle(ang.phi, alpha)
    phip = wrap_angle(ang.phip, alpha)

    # wrap_angle(α, α) = 0 aliases EXACT on-n-face observation (φ = α) onto the
    # o-face; disambiguate from the raw input azimuth, mirroring the φ' handling
    # below. Adding α back restores the raw value and keeps the unit AD
    # derivative (no constant snap). Runs before the grazing block so the mirror
    # `alpha - phi` also sees the corrected φ.
    if phi <= DEFAULT_TRANSITION_TOL && abs(ang.phi - alpha) <= DEFAULT_TRANSITION_TOL
        phi = phi + alpha
    end

    # At grazing incidence (φ' = 0 or φ' = α), collapse β⁻ = β⁺ = φ by
    # setting φ' exactly to zero.  Keep φ in the standard [0, α) range so
    # that the sign of sin(ψ₂) flips as φ crosses the ISB at π.  Centered
    # wrapping [-α/2, α/2) would place its branch cut at ±α/2, which for a
    # half-plane (α = 2π) coincides with the ISB and destroys the
    # compensating discontinuity that makes the total field continuous.
    #
    # n-face grazing (φ' = α) must additionally MIRROR the geometry about the
    # wedge bisector — (φ, φ') → (α − φ, 0) — so the collapsed problem is
    # measured from the grazed face. Snapping φ' → 0 alone evaluated o-face
    # geometry with an n-face-grazing source, an O(1) error in D_h at exact
    # grazing (the mirrored value matches the φ' → α⁻ limit to machine
    # precision). The third return value reports the mirror so callers can
    # swap face-specific quantities (Lro ↔ Lrn, face materials).
    if min(abs(phip), abs(alpha - phip)) <= DEFAULT_TRANSITION_TOL
        near_n = abs(alpha - phip) < abs(phip)
        if !near_n && abs(phip) <= DEFAULT_TRANSITION_TOL
            # wrap_angle(α, α) = 0 aliases EXACT n-face grazing onto the o-face;
            # disambiguate from the unwrapped input azimuth.
            near_n = abs(ang.phip - alpha) <= DEFAULT_TRANSITION_TOL
        end
        if near_n
            return (alpha - phi, zero(typeof(phi)), true)
        end
        return (phi, zero(typeof(phi)), false)
    end
    return (phi, phip, false)
end

"""
    pec_wedge_prefactor(k, n)

Common amplitude prefactor C(k,n) = -exp(-iπ/4) / (2n√(2πk)).
"""
function pec_wedge_prefactor(k::Number, n::Real)
    -exp(-im * π/4) / (2 * n * sqrt(2π) * safe_sqrt(_validate_wavenumber(k)))
end

"""
    _cot_F_regularized(psi, a, k, L)

Compute cot(ψ)·F(kLa), regularized at shadow boundaries where
cot(ψ) → ±∞ and a → 0 simultaneously.

At shadow/reflection boundaries the cotangent pole in ψ_j coincides
with the zero of the distance parameter a_j.  The product cot(ψ)·F(X)
is finite in the limit, but the naïve `cot(ψ) * F(X)` overflows in
IEEE 754 arithmetic.  This helper reformulates the computation by
factoring out the cancellation:

    cot(ψ)·F(X) = cos(ψ) · [√(πX)/sin(ψ)] · e^{+iπ/4} · erfcx(z)

where the bracketed ratio √X/sin(ψ) → n·√(2kL) at the boundary.
"""
function _cot_F_regularized(
    psi::Real,
    a::Real,
    k::Number,
    L::Number;
    n::Real = 1.0,
    detuning::Real = 0.0,
)
    T = promote_type(Float64, typeof(float(real(psi))), typeof(float(real(a))),
                     typeof(float(real(k))), typeof(float(real(L))))
    CT = Complex{T}
    τ = _transition_tolerances(T)
    sin_psi = sin(psi)

    if isinf(L)
        # Exact infinite-distance (far-field) limit: F(kLa) -> 1, so
        # cot(ψ)·F -> cot(ψ). Zero only the single genuinely degenerate sample
        # where the cotangent pole and the distance-parameter zero coincide, using
        # the SAME AND gate as the finite-L branch below (not OR). The gate must be
        # AND because a ≈ 2n²(Δψ)² is QUADRATIC in the detuning while sin ψ ≈ Δψ is
        # linear: |a| ≤ √eps alone spans |Δψ| ≲ √(2√eps) ≈ 1.7e-4 rad, an annulus
        # around every GO boundary over which cot(ψ) is finite and exactly
        # representable. Gating on |a| alone (OR) wrongly zeroed that whole annulus,
        # dropping the dominant O(10³–10⁴) cot term and creating a discontinuity.
        if abs(a) <= τ.τa && abs(sin_psi) <= τ.τs
            return zero(CT)
        end
        return CT(cot(psi))
    end

    a_eval = a
    psi_eval = psi
    sin_eval = sin_psi

    if abs(a) <= τ.τa && abs(sin_psi) <= τ.τs
        # Exact transition sample: enforce a matched one-sided surrogate
        # (a ~ 2 n^2 (Δψ)^2) instead of midpoint-zero convention.
        # This keeps cot(ψ)·F(kLa) on a physically consistent finite branch.
        # `detuning` is Δψ, the signed offset of ψ from its nearest cot pole, so
        # (psi - detuning) is the pole and sgn·dψ steps to the correct side; this
        # matches sin_eval ≈ ±dψ to a_eval = 2n²dψ² and removes both the
        # wrong-side flip (s=+1 terms) and the additive-offset magnitude bias.
        sgn = detuning == 0 ? one(T) : sign(detuning)
        dψ = max(τ.τs, 10 * eps(T))
        psi_eval = (psi - detuning) + sgn * dψ
        sin_eval = sin(psi_eval)
        a_eval = 2 * n^2 * dψ^2
    end

    X = k * L * a_eval

    if !(isfinite(real(X)) && isfinite(imag(X)))
        # A positive-real overflow is the finite-distance representation of the
        # exact GTD limit F(X)->1. Other non-finite values are invalid and must
        # fail closed rather than being converted to a zero diffraction term.
        if k isa Real && L isa Real && k > 0 && L > 0 && a_eval > 0
            return CT(cot(psi_eval))
        end
        throw(DomainError(X, "non-finite transition argument k*L*a"))
    end

    # Numerically stable form:
    # cot(ψ)F(X) = cos(ψ) * [√(πX)/sin(ψ)] * e^{+iπ/4} * erfcx(e^{+iπ/4}√X)
    # This avoids overflow/underflow from multiplying cot(ψ) and F(X) separately.
    sqrtX = safe_sqrt(X)
    z = exp(+im * π/4) * sqrtX
    ratio = sqrt(π) * sqrtX / sin_eval
    v = cos(psi_eval) * exp(+im * π/4) * erfcx(z) * ratio
    if _complex_finite(v)
        return v
    end

    # Fallback: direct form away from exact transition.
    vd = cot(psi_eval) * F_utd(X)
    if _complex_finite(vd)
        return vd
    end

    throw(DomainError((psi, a, k, L), "cotangent-transition product is non-finite"))
end

"""
    pec_wedge_DsDh(wedge, ang, k, L; convention=EXP_IWT) -> (Ds, Dh)

Compute the soft and hard scalar UTD diffraction coefficients for a
PEC wedge using the Kouyoumjian–Pathak four-term form.

# Arguments
- `wedge::Wedge`: wedge geometry
- `ang::RayAngles`: observation and incident azimuths
- `k::Number`: wavenumber (real positive for lossless media)
- `L::Real`: effective distance parameter s·s'/(s+s')

# Returns
- `(Ds, Dh)`: tuple of complex diffraction coefficients
"""
function pec_wedge_DsDh(
    wedge::Wedge,
    ang::RayAngles,
    k::Number,
    L::Number;
    convention::PhasorConvention = EXP_IWT,
)
    convention.sgn == +1 || error("Only exp(+iωt) convention is supported")
    _validate_effective_L(L)

    n = wedge_n(wedge)
    # The mirror flag is irrelevant here: all four terms share one L and the
    # PEC σ-signs are face-symmetric.
    phi, phip, _ = _effective_angles_for_kp(wedge, ang)

    terms = kp_four_terms(phi, phip, n)
    C = pec_wedge_prefactor(k, n)
    CT = typeof(C)

    Ds = zero(CT)
    Dh = zero(CT)

    for j in 1:4
        psi_j = terms.psi[j]
        a_j   = terms.aj[j]
        detuning_j = _kp_transition_detuning(j, terms, n)

        contrib = _cot_F_regularized(psi_j, a_j, k, L; n = n, detuning = detuning_j)
        Ds += PEC_SIGMA_SOFT[j] * contrib
        Dh += PEC_SIGMA_HARD[j] * contrib
    end

    return (C * Ds, C * Dh)
end

"""
    pec_wedge_DsDh(wedge, ang, k, Li, Lro, Lrn; Rs=-1, Rh=+1, convention=EXP_IWT)

Generalized PEC wedge coefficient with separate transition distances:
- `Li`  for incident-shadow terms (`D1`,`D2`)
- `Lrn` for reflection from face `n` (`D3`)
- `Lro` for reflection from face `o` (`D4`)

For PEC, use `Rs=-1`, `Rh=+1` (soft/hard reflection signs).
"""
function pec_wedge_DsDh(
    wedge::Wedge,
    ang::RayAngles,
    k::Number,
    Li::Real,
    Lro::Real,
    Lrn::Real;
    Rs::Number = -1,
    Rh::Number = +1,
    convention::PhasorConvention = EXP_IWT,
)
    convention.sgn == +1 || error("Only exp(+iωt) convention is supported")
    _validate_effective_L(Li)
    _validate_effective_L(Lro)
    _validate_effective_L(Lrn)

    n = wedge_n(wedge)
    phi, phip, mirrored = _effective_angles_for_kp(wedge, ang)
    if mirrored
        # The n-face grazing mirror swaps the face roles, so the per-face
        # reflection transition distances swap with them.
        Lro, Lrn = Lrn, Lro
    end

    terms = kp_four_terms(phi, phip, n)
    C = pec_wedge_prefactor(k, n)

    c1 = _cot_F_regularized(
        terms.psi[1],
        terms.aj[1],
        k,
        Li;
        n = n,
        detuning = _kp_transition_detuning(1, terms, n),
    )
    c2 = _cot_F_regularized(
        terms.psi[2],
        terms.aj[2],
        k,
        Li;
        n = n,
        detuning = _kp_transition_detuning(2, terms, n),
    )
    c3 = _cot_F_regularized(
        terms.psi[3],
        terms.aj[3],
        k,
        Lrn;
        n = n,
        detuning = _kp_transition_detuning(3, terms, n),
    )
    c4 = _cot_F_regularized(
        terms.psi[4],
        terms.aj[4],
        k,
        Lro;
        n = n,
        detuning = _kp_transition_detuning(4, terms, n),
    )

    common = c1 + c2
    refl = c3 + c4
    Ds = common + Rs * refl
    Dh = common + Rh * refl

    return (C * Ds, C * Dh)
end
