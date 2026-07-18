"""
Regime detection: classify observation points as lit, shadow, or transition.
"""

"""
    wedge_transition_args(wedge, ang, k, L; tol=DEFAULT_TRANSITION_TOL) -> NamedTuple

Compute the transition arguments and regime classification for each of the
four KP terms. The default `tol` is the package transition tolerance
`DEFAULT_TRANSITION_TOL = √eps(Float64) ≈ 1.5e-8`; `:transition` flags a
numerically degenerate boundary sample (cot pole coincident with a_j → 0),
not the physical kLa transition zone.

Returns a NamedTuple with:
- `gj`:     NTuple{4,Float64} – signed quantities cos((2nπN_j - β_j)/2)
- `Xj`:     NTuple{4,<:Number} – transition arguments kL·a_j
- `regime`:  NTuple{4,Symbol} – :lit, :shadow, or :transition
"""
function wedge_transition_args(
    wedge::Wedge,
    ang::RayAngles,
    k::Number,
    L::Number;
    tol::Real = DEFAULT_TRANSITION_TOL,
)
    k = _validate_wavenumber(k)
    _validate_effective_L(L)
    isfinite(tol) && tol >= zero(tol) ||
        throw(DomainError(tol, "transition tolerance must be finite and nonnegative"))
    n = wedge_n(wedge)
    phi, phip, _ = _effective_angles_for_kp(wedge, ang)

    terms = kp_four_terms(phi, phip, n)

    gj_vals = ntuple(4) do j
        Nj = terms.Nj[j]
        bj = terms.beta[j]
        cos((2 * n * π * Nj - bj) / 2)
    end

    Xj_vals = ntuple(4) do j
        k * L * terms.aj[j]
    end

    regimes = ntuple(4) do j
        g = gj_vals[j]
        if abs(g) < tol
            :transition
        elseif g > 0
            :lit
        else
            :shadow
        end
    end

    return (gj = gj_vals, Xj = Xj_vals, regime = regimes)
end
