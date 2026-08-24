"""
PEC wedge diffraction coefficients (Kouyoumjian–Pathak form).

Convention: exp(+iωt), outgoing ~ exp(−ikr).
"""

# PEC sign factors: σ_j for j=1..4
# Soft (Dirichlet): +1, +1, -1, -1
# Hard (Neumann):   +1, +1, +1, +1
const PEC_SIGMA_SOFT = (+1, +1, -1, -1)
const PEC_SIGMA_HARD = (+1, +1, +1, +1)

@inline function _checked_coefficients(Ds::Number, Dh::Number)
    _number_isfinite(Ds) && _number_isfinite(Dh) || throw(DomainError(
        (Ds, Dh),
        "diffraction coefficient is non-finite at the requested parameters",
    ))
    return (Ds, Dh)
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
    _number_isfinite(Lc) && abs(Lc) > 0 ||
        throw(DomainError(L, "Complex L must be finite and nonzero"))
    return
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
    T = typeof(float(_primal_value(u)))
    πT = zero(u) + T(π)
    target = s == +1 ? πT : -πT
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
    if phi <= DEFAULT_TRANSITION_TOL && _primal_iszero(ang.phi - alpha)
        phi = phi + alpha
    end

    # At grazing incidence (φ' = 0 or φ' = α), β⁻ = β⁺ = φ in value. Keep the
    # signed local offset from the grazed face, however, so ForwardDiff carries
    # the physically relevant one-sided face-angle derivative instead of a
    # snapped zero derivative. Keep φ in the standard [0, α) range so
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
            # disambiguate only the exact raw seam. A tolerance here would map
            # α+δ to the n-face while the periodic input δ maps to the o-face.
            near_n = _primal_iszero(ang.phip - alpha)
        end
        if near_n
            # At the exact raw α seam, wrapping aliases `phip` to zero. Recover
            # the n-face offset from the unwrapped input; for a nearby interior
            # point `alpha - phip` is the same quantity.
            phip_from_n = abs(phip) <= DEFAULT_TRANSITION_TOL &&
                          _primal_iszero(ang.phip - alpha) ?
                          alpha - ang.phip : alpha - phip
            return (alpha - phi, phip_from_n, true)
        end
        return (phi, phip, false)
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
    near_transition = abs(a) <= τ.τa && abs(sin_psi) <= τ.τs

    if isinf(L)
        # Exact infinite-distance (far-field) limit: F(kLa) -> 1, so
        # cot(ψ)·F -> cot(ψ). The AND gate identifies the neighborhood that
        # needs local detuning arithmetic; only an exact coincident-pole sample
        # receives the documented midpoint value. The gate must be AND because a is
        # QUADRATIC in the angular detuning while sin ψ ≈ Δψ is linear. In ψ-space
        # a ≈ 2n²(Δψ)², so |a| ≤ √eps alone spans |Δψ| ≲ 8.6e-5/n rad; the same
        # window expressed in azimuthal (φ) space has a ≈ (δφ)²/2, so |a| ≤ √eps
        # spans a half-width |δφ| ≲ √(2√eps) ≈ 1.73e-4 rad. Either way |a| ≤ √eps
        # covers an annulus around every GO boundary over which cot(ψ) is finite and
        # exactly representable. Gating on |a| alone (OR) wrongly zeroed that whole
        # annulus, dropping the dominant O(10³–10⁴) cot term and creating a
        # discontinuity. Nonzero samples inside the AND gate remain physical and
        # are evaluated from their exact local detuning.
        if near_transition
            _primal_iszero(detuning) && return zero(CT)
            # At a nonzero offset from the nearest pole, cot(psi) equals
            # cot(detuning). The local form avoids subtracting nearly equal
            # O(π) angles and must not be replaced by the exact-pole midpoint.
            return CT(cot(detuning))
        end
        return CT(cot(psi))
    end

    a_eval = a
    angular_eval = psi

    if near_transition
        if _primal_iszero(detuning)
            # The exact transition has two one-sided limits. Retain the package's
            # documented lit-side convention with a matched local offset.
            angular_eval = max(τ.τs, 10 * eps(T))
        else
            # For a nonzero detuning, use the exact local identities
            # cot(psi)=cot(Δψ) and a=2sin²(nΔψ). Clamping the whole tolerance
            # window to one surrogate destroys valid large-kL near-boundary values.
            angular_eval = detuning
        end
        a_eval = 2 * sin(n * angular_eval)^2
    end

    sin_eval = sin(angular_eval)

    X = k * L * a_eval

    if !_number_isfinite(X)
        # A positive-real overflow is the finite-distance representation of the
        # exact GTD limit F(X)->1. Other non-finite values are invalid and must
        # fail closed rather than being converted to a zero diffraction term.
        if k isa Real && L isa Real && k > 0 && L > 0 && a_eval > 0
            return CT(cot(angular_eval))
        end
        throw(DomainError(X, "non-finite transition argument k*L*a"))
    end

    # Numerically stable form:
    # cot(ψ)F(X) = cos(ψ) * [√(πX)/sin(ψ)] * e^{+iπ/4} * erfcx(e^{+iπ/4}√X)
    # This avoids overflow/underflow from multiplying cot(ψ) and F(X) separately.
    sqrtX = safe_sqrt(X)
    z = exp(+im * π/4) * sqrtX
    ratio = sqrt(π) * sqrtX / sin_eval
    scaled_erfcx = try
        erfcx(z)
    catch err
        err isa MethodError || rethrow()
        throw(ArgumentError(
            "wedge transition product does not support $(typeof(X)): " *
            "erfcx is unavailable for $(typeof(z))",
        ))
    end
    v = cos(angular_eval) * exp(+im * π/4) * scaled_erfcx * ratio
    if _number_isfinite(v)
        return v
    end

    # Fallback: direct form away from exact transition.
    vd = cot(angular_eval) * F_utd(X)
    if _number_isfinite(vd)
        return vd
    end

    throw(DomainError((psi, a, k, L), "cotangent-transition product is non-finite"))
end

"""
    pec_wedge_DsDh(wedge, ang, k, L; convention=EXP_IWT) -> (Ds, Dh)

Compute the soft and hard scalar UTD diffraction coefficients for a
PEC wedge using the Kouyoumjian–Pathak four-term form.

This is the traditional pairing and remains the comparison baseline.
Near face-grazing incidence the soft coefficient can lose significance;
use the recommended [`wedge_DsDh`](@ref) router for automatic continuation.

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

    return _checked_coefficients(C * Ds, C * Dh)
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
    _validate_finite_number(Rs, "soft reflection coefficient Rs")
    _validate_finite_number(Rh, "hard reflection coefficient Rh")

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

    return _checked_coefficients(C * Ds, C * Dh)
end
