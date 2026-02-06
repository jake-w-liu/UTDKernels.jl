"""
Regime detection: classify observation points as lit, shadow, or transition.
"""

"""
    wedge_transition_args(wedge, ang, k, L; tol=1e-10) -> NamedTuple

Compute the transition arguments and regime classification for each of the
four KP terms.

Returns a NamedTuple with:
- `gj`:     NTuple{4,Float64} – signed quantities cos((2nπN_j - β_j)/2)
- `Xj`:     NTuple{4,Float64} – transition arguments kL·a_j
- `regime`:  NTuple{4,Symbol} – :lit, :shadow, or :transition
"""
function wedge_transition_args(
    wedge::Wedge,
    ang::RayAngles,
    k::Number,
    L::Real;
    tol::Real = DEFAULT_TRANSITION_TOL,
)
    n = wedge_n(wedge)
    phi  = wrap_angle(ang.phi, wedge.alpha)
    phip = wrap_angle(ang.phip, wedge.alpha)

    terms = kp_four_terms(phi, phip, n)

    gj_vals = ntuple(4) do j
        Nj = terms.Nj[j]
        bj = terms.beta[j]
        cos((2 * n * π * Nj - bj) / 2)
    end

    Xj_vals = ntuple(4) do j
        real(k) * L * terms.aj[j]
    end

    regimes = ntuple(4) do j
        g = abs(gj_vals[j])
        if g < tol
            :transition
        elseif Xj_vals[j] > 10.0  # well into lit region
            :lit
        else
            :shadow
        end
    end

    return (gj = gj_vals, Xj = Xj_vals, regime = regimes)
end
