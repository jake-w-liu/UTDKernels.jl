"""
Impedance-wedge UTD diffraction coefficients (Holm 2000 heuristic).

Extends the PEC four-term Kouyoumjian–Pathak structure with Holm's
face-specific Fresnel reflection coefficients and incident-term weights.

Convention: exp(+iωt), outgoing ~ exp(−ikr).

Reference:
    P. D. Holm, "A new heuristic UTD diffraction coefficient for
    nonperfectly conducting wedges," IEEE Trans. Antennas Propag.,
    vol. 48, no. 8, pp. 1211–1219, Aug. 2000.
"""

"""
    impedance_wedge_DsDh(iw, ang, k, L; convention=EXP_IWT)

Compute soft and hard UTD diffraction coefficients for an impedance wedge
using the Holm (2000) heuristic with a single effective distance L.

# Arguments
- `iw::ImpedanceWedge`: wedge geometry + face materials
- `ang::RayAngles`: observation (φ) and incident (φ') azimuths
- `k::Number`: wavenumber (real positive for lossless background)
- `L::Number`: effective distance parameter s·s'/(s+s')

# Returns
- `(Ds, Dh)`: tuple of complex diffraction coefficients

# Algorithm
The four KP terms are computed identically to the PEC case. Terms 3–4
(reflection shadow boundaries for faces n and o, respectively) are weighted
by the Fresnel reflection coefficients of the corresponding face material.
Terms 1–2 use Holm's incident weights ``W_n`` and ``W_0``:

    Ds = G · C · (W_TE_n·c1 + W_TE_0·c2 + R_TE_n·c3 + R_TE_0·c4)
    Dh = G · C · (W_TM_n·c1 + W_TM_0·c2 + R_TM_n·c3 + R_TM_0·c4)

For each polarization, ``W_n = R_0 R_n`` and ``W_0 = 1`` when
``φ' < α/2``; the two weights exchange when ``φ' ≥ α/2``. The grazing
factor ``G`` is ``1/2`` at exact face-grazing incidence and ``1`` otherwise.

where the Fresnel coefficients are evaluated at the source's grazing angle
on each face:
  - Face o (φ=0):  ψ_o = φ'
  - Face n (φ=α):  ψ_n = α − φ'
"""
# Physical grazing angle of a ray with azimuth ψ measured from a face: the
# angle between a line and a PLANE lies in [0, π/2], so fold mod π and reflect.
# Clamping to π/2 instead froze the Fresnel coefficient at its normal-incidence
# value for ψ ∈ (π/2, π), breaking total-field continuity at the reflection
# shadow boundary (≈20% of |u| for φ' ∈ (π/2, π), probe-verified).
@inline function _face_grazing_angle(psi::Real)
    # `mod` gives a NaN ForwardDiff partial exactly at multiples of π.  Those
    # points are the two supported face-grazing seams, so use the package's
    # value-equivalent AD-safe floor-form wrap before folding into [0, π/2].
    m = wrap_angle(psi, oftype(psi, π))
    return m > oftype(psi, π) / 2 ? oftype(psi, π) - m : m
end

@inline function _holm_incident_weights(R_o::Number, R_n::Number, phip::Real, alpha::Real)
    product = R_o * R_n
    return phip < alpha / 2 ? (product, one(product)) : (one(product), product)
end

# The effective-angle map sends either exact face-grazing incidence to phip=0.
# For a Dual input at the seam, `iszero` is false when its derivative seed is
# nonzero. This differentiates the continuous one-sided limit of the standard
# single-distance coefficient; its scalar value still receives Holm's factor.
@inline _holm_grazing_factor(phip::Real) = iszero(phip) ? one(phip) / 2 : one(phip)

function impedance_wedge_DsDh(
    iw::ImpedanceWedge,
    ang::RayAngles,
    k::Number,
    L::Number;
    convention::PhasorConvention = EXP_IWT,
)
    convention.sgn == +1 || error("Only exp(+iωt) convention is supported")
    _validate_effective_L(L)

    alpha = iw.alpha
    n = alpha / π
    w = _to_wedge(iw)

    phi, phip, mirrored = _effective_angles_for_kp(w, ang)
    terms = kp_four_terms(phi, phip, n)
    C = pec_wedge_prefactor(k, n)

    # Compute the four cot·F products
    c = ntuple(4) do j
        _cot_F_regularized(
            terms.psi[j], terms.aj[j], k, L;
            n = n, detuning = _kp_transition_detuning(j, terms, n),
        )
    end

    # Fresnel angles: source grazing angle at each face (face roles swap under
    # the n-face grazing mirror).
    psi_o = _face_grazing_angle(phip)               # grazing angle at o-face (φ=0)
    psi_n = _face_grazing_angle(alpha - phip)       # grazing angle at n-face (φ=α)

    eps_r_o = mirrored ? iw.face_n.eps_r : iw.face_o.eps_r
    eps_r_n = mirrored ? iw.face_o.eps_r : iw.face_n.eps_r

    # Fresnel reflection coefficients at each face
    R_te_o = fresnel_te(psi_o, eps_r_o)
    R_tm_o = fresnel_tm(psi_o, eps_r_o)
    R_te_n = fresnel_te(psi_n, eps_r_n)
    R_tm_n = fresnel_tm(psi_n, eps_r_n)

    W_te_n, W_te_o = _holm_incident_weights(R_te_o, R_te_n, phip, alpha)
    W_tm_n, W_tm_o = _holm_incident_weights(R_tm_o, R_tm_n, phip, alpha)
    G = _holm_grazing_factor(phip)

    Ds = G * (W_te_n * c[1] + W_te_o * c[2] + R_te_n * c[3] + R_te_o * c[4])
    Dh = G * (W_tm_n * c[1] + W_tm_o * c[2] + R_tm_n * c[3] + R_tm_o * c[4])

    return (C * Ds, C * Dh)
end

"""
    impedance_wedge_DsDh(iw, ang, k, Li, Lro, Lrn; convention=EXP_IWT)

Generalized impedance-wedge coefficient with separate transition distances:
- `Li`  for incident-shadow terms (c1, c2)
- `Lrn` for reflection from face n (c3)
- `Lro` for reflection from face o (c4)

At exact face-grazing incidence, this separate-distance extension uses the
one-sided continuous value (`G=1`) rather than the standard single-distance
Holm factor.
"""
function impedance_wedge_DsDh(
    iw::ImpedanceWedge,
    ang::RayAngles,
    k::Number,
    Li::Real,
    Lro::Real,
    Lrn::Real;
    convention::PhasorConvention = EXP_IWT,
)
    convention.sgn == +1 || error("Only exp(+iωt) convention is supported")
    _validate_effective_L(Li)
    _validate_effective_L(Lro)
    _validate_effective_L(Lrn)

    alpha = iw.alpha
    n = alpha / π
    w = _to_wedge(iw)

    phi, phip, mirrored = _effective_angles_for_kp(w, ang)
    if mirrored
        # n-face grazing mirror: face roles (and their transition distances) swap.
        Lro, Lrn = Lrn, Lro
    end
    terms = kp_four_terms(phi, phip, n)
    C = pec_wedge_prefactor(k, n)

    # Each term gets its own L
    L_per_term = (Li, Li, Lrn, Lro)
    c = ntuple(4) do j
        _cot_F_regularized(
            terms.psi[j], terms.aj[j], k, L_per_term[j];
            n = n, detuning = _kp_transition_detuning(j, terms, n),
        )
    end

    # Fresnel angles: physical grazing angle at each face (see _face_grazing_angle).
    psi_o = _face_grazing_angle(phip)
    psi_n = _face_grazing_angle(alpha - phip)

    eps_r_o = mirrored ? iw.face_n.eps_r : iw.face_o.eps_r
    eps_r_n = mirrored ? iw.face_o.eps_r : iw.face_n.eps_r

    R_te_o = fresnel_te(psi_o, eps_r_o)
    R_tm_o = fresnel_tm(psi_o, eps_r_o)
    R_te_n = fresnel_te(psi_n, eps_r_n)
    R_tm_n = fresnel_tm(psi_n, eps_r_n)

    W_te_n, W_te_o = _holm_incident_weights(R_te_o, R_te_n, phip, alpha)
    W_tm_n, W_tm_o = _holm_incident_weights(R_tm_o, R_tm_n, phip, alpha)
    # Holm's isolated G=1/2 factor is defined for the published single-distance
    # coefficient. With unequal term distances it would introduce a jump at
    # exact face grazing, so this generalized API uses the one-sided continuous
    # extension (G=1).
    Ds = W_te_n * c[1] + W_te_o * c[2] + R_te_n * c[3] + R_te_o * c[4]
    Dh = W_tm_n * c[1] + W_tm_o * c[2] + R_tm_n * c[3] + R_tm_o * c[4]

    return (C * Ds, C * Dh)
end
