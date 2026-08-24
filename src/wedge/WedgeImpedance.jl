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
Terms 1–2 use Holm's fixed incident weights ``M_1`` and ``M_2``:

    Ds = G · C · (R_TE_0 R_TE_n·c1 + c2 + R_TE_n·c3 + R_TE_0·c4)
    Dh = G · C · (R_TM_0 R_TM_n·c1 + c2 + R_TM_n·c3 + R_TM_0·c4)

For each polarization, ``M_1 = R_0 R_n`` and ``M_2 = 1`` throughout the
wedge. The grazing factor ``G`` is ``1/2`` at exact face-grazing incidence
and ``1`` otherwise.

The face-specific Fresnel angles depend on both incident and observation
directions, as in Holm's definition:
  - Face o (φ=0):  θ_o = min(φ', φ)
  - Face n (φ=α):  θ_n = min(α − φ', α − φ)
"""
# Holm, IEEE TAP 48(8), 2000, uses observation/source minimum angles and fixed
# M1=R0*Rn, M2=1 weights. The source-half exchange belongs to later reciprocal
# modifications of Holm's heuristic and is not part of the cited coefficient.
@inline function _holm_fresnel_angles(phi::Real, phip::Real, alpha::Real)
    return (min(phi, phip), min(alpha - phi, alpha - phip))
end

function _effective_angles_for_holm(wedge::Wedge, ang::RayAngles)
    alpha = wedge.alpha
    phi = wrap_angle(ang.phi, alpha)
    phip = wrap_angle(ang.phip, alpha)

    # Preserve the two physically distinct face values at the exact raw alpha
    # seam. Unlike PEC, Holm's fixed M1/M2 weighting is not mirror-reciprocal,
    # so mapping n-face incidence to the o-face would change the cited model.
    if phi <= DEFAULT_TRANSITION_TOL && _primal_iszero(ang.phi - alpha)
        phi += alpha
    end
    if phip <= DEFAULT_TRANSITION_TOL && _primal_iszero(ang.phip - alpha)
        phip += alpha
    end
    return phi, phip
end

@inline function _holm_incident_weights(R_o::Number, R_n::Number)
    product = R_o * R_n
    return (product, one(product))
end

# For a Dual input at a seam, `iszero` is false when its derivative seed is
# nonzero. This differentiates the continuous one-sided limit; scalar calls at
# either exact face receive Holm's isolated factor.
@inline function _holm_grazing_factor(phip::Real, alpha::Real)
    return iszero(phip) || iszero(alpha - phip) ? one(phip) / 2 : one(phip)
end

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

    phi, phip = _effective_angles_for_holm(w, ang)
    terms = kp_four_terms(phi, phip, n)
    C = pec_wedge_prefactor(k, n)

    # Compute the four cot·F products
    c = ntuple(4) do j
        _cot_F_regularized(
            terms.psi[j], terms.aj[j], k, L;
            n = n, detuning = _kp_transition_detuning(j, terms, n),
        )
    end

    # Holm's Fresnel angles use the nearer of the incident and diffracted rays
    # at each face.
    theta_o, theta_n = _holm_fresnel_angles(phi, phip, alpha)

    eps_r_o = iw.face_o.eps_r
    eps_r_n = iw.face_n.eps_r

    # Fresnel reflection coefficients at each face
    R_te_o = fresnel_te(theta_o, eps_r_o)
    R_tm_o = fresnel_tm(theta_o, eps_r_o)
    R_te_n = fresnel_te(theta_n, eps_r_n)
    R_tm_n = fresnel_tm(theta_n, eps_r_n)

    W_te_n, W_te_o = _holm_incident_weights(R_te_o, R_te_n)
    W_tm_n, W_tm_o = _holm_incident_weights(R_tm_o, R_tm_n)
    G = _holm_grazing_factor(phip, alpha)

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

    phi, phip = _effective_angles_for_holm(w, ang)
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

    # Apply Holm's observation/source minimum-angle prescription to this
    # separate-distance extension as well.
    theta_o, theta_n = _holm_fresnel_angles(phi, phip, alpha)

    eps_r_o = iw.face_o.eps_r
    eps_r_n = iw.face_n.eps_r

    R_te_o = fresnel_te(theta_o, eps_r_o)
    R_tm_o = fresnel_tm(theta_o, eps_r_o)
    R_te_n = fresnel_te(theta_n, eps_r_n)
    R_tm_n = fresnel_tm(theta_n, eps_r_n)

    W_te_n, W_te_o = _holm_incident_weights(R_te_o, R_te_n)
    W_tm_n, W_tm_o = _holm_incident_weights(R_tm_o, R_tm_n)
    # Holm's isolated G=1/2 factor is defined for the published single-distance
    # coefficient. With unequal term distances it would introduce a jump at
    # exact face grazing, so this generalized API uses the one-sided continuous
    # extension (G=1).
    Ds = W_te_n * c[1] + W_te_o * c[2] + R_te_n * c[3] + R_te_o * c[4]
    Dh = W_tm_n * c[1] + W_tm_o * c[2] + R_tm_n * c[3] + R_tm_o * c[4]

    return (C * Ds, C * Dh)
end
