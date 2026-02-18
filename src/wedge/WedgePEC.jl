"""
PEC wedge diffraction coefficients (Kouyoumjian–Pathak form).

Convention: exp(+iωt), outgoing ~ exp(−ikr).
"""

# PEC sign factors: σ_j for j=1..4
# Soft (Dirichlet): +1, +1, -1, -1
# Hard (Neumann):   +1, +1, +1, +1
const PEC_SIGMA_SOFT = (+1, +1, -1, -1)
const PEC_SIGMA_HARD = (+1, +1, +1, +1)

function _wrap_angle_centered(phi::Real, alpha::Real)
    w = mod(phi + alpha / 2, alpha) - alpha / 2
    return w == -alpha / 2 ? alpha / 2 : w
end

function _effective_angles_for_kp(wedge::Wedge, ang::RayAngles)
    alpha = wedge.alpha
    phi  = wrap_angle(ang.phi, alpha)
    phip = wrap_angle(ang.phip, alpha)

    # At grazing incidence (φ' = 0 or φ' = α), ξ- and ξ+ should be treated
    # as the same continuous angular variable. Wrapping into [0, α) can create
    # a branch-cut alias (0 ↔ α) that appears as a nonphysical jump in Di/Dr.
    if min(abs(phip), abs(alpha - phip)) <= DEFAULT_TRANSITION_TOL
        return (_wrap_angle_centered(phi - phip, alpha), 0.0)
    end
    return (phi, phip)
end

"""
    pec_wedge_prefactor(k, n)

Common amplitude prefactor C(k,n) = -exp(-iπ/4) / (2n√(2πk)).
"""
function pec_wedge_prefactor(k::Number, n::Real)
    -exp(-im * π/4) / (2 * n * sqrt(2π * k))
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
function _cot_F_regularized(psi::Real, a::Real, k::Number, L::Real)
    if isinf(L)
        # Exact infinite-distance (far-field) limit: F(kLa) -> 1.
        # At exact shadow/reflection boundaries, cotangent terms are singular;
        # use the same symmetric midpoint convention as finite-L regularization.
        if abs(sin(psi)) <= DEFAULT_TRANSITION_TOL
            return zero(ComplexF64)
        end
        return cot(psi)
    end

    X = k * L * a
    sin_psi = sin(psi)

    # Away from cotangent poles: direct evaluation is safe
    if abs(sin_psi) > DEFAULT_TRANSITION_TOL
        return cot(psi) * F_utd(X)
    end

    # Near or at cotangent pole.
    cos_psi = cos(psi)

    if abs(a) < 1e-28
        # Exact boundary point: one-sided limits are finite but opposite-signed.
        # Return the symmetric midpoint (0) as a deterministic convention.
        return zero(ComplexF64)
    end

    # sin(ψ) is small but a > 0: compute ratio √(πX)/sin(ψ) directly.
    # Both numerator and denominator are small, but their ratio is O(n√(kL)).
    sqrtX = safe_sqrt(X)
    z = exp(+im * π/4) * sqrtX
    ratio = sqrt(π * Complex(X)) / sin_psi
    return cos_psi * exp(+im * π/4) * erfcx(z) * ratio
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
    L::Real;
    convention::PhasorConvention = EXP_IWT,
)
    convention.sgn == +1 || error("Only exp(+iωt) convention is supported")

    n = wedge_n(wedge)
    phi, phip = _effective_angles_for_kp(wedge, ang)

    terms = kp_four_terms(phi, phip, n)
    C = pec_wedge_prefactor(k, n)

    Ds = zero(ComplexF64)
    Dh = zero(ComplexF64)

    for j in 1:4
        psi_j = terms.psi[j]
        a_j   = terms.aj[j]

        contrib = _cot_F_regularized(psi_j, a_j, k, L)
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
    Li > 0 || throw(DomainError(Li, "Li must be positive"))
    Lro > 0 || throw(DomainError(Lro, "Lro must be positive"))
    Lrn > 0 || throw(DomainError(Lrn, "Lrn must be positive"))

    n = wedge_n(wedge)
    phi, phip = _effective_angles_for_kp(wedge, ang)

    terms = kp_four_terms(phi, phip, n)
    C = pec_wedge_prefactor(k, n)

    c1 = _cot_F_regularized(terms.psi[1], terms.aj[1], k, Li)
    c2 = _cot_F_regularized(terms.psi[2], terms.aj[2], k, Li)
    c3 = _cot_F_regularized(terms.psi[3], terms.aj[3], k, Lrn)
    c4 = _cot_F_regularized(terms.psi[4], terms.aj[4], k, Lro)

    common = c1 + c2
    refl = c3 + c4
    Ds = common + Rs * refl
    Dh = common + Rh * refl

    return (C * Ds, C * Dh)
end
