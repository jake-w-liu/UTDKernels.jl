"""
Regime detection: classify observation points as lit, shadow, or transition.
"""

"""
    wedge_transition_args(wedge, ang, k, L; tol=nothing) -> NamedTuple

Compute the transition arguments and regime classification for each of the
four KP terms. With `tol=nothing`, the tolerance is `√eps(T)` for the promoted
primal angle type `T`. A numeric `tol` overrides it. `:transition` flags a
numerically degenerate boundary sample (cot pole coincident with a_j → 0), not
the physical kLa transition zone.

Returns a NamedTuple with:
- `gj`:     `NTuple{4,<:Real}` – signed quantities cos((2nπN_j - β_j)/2)
- `Xj`:     NTuple{4,<:Number} – transition arguments kL·a_j
- `regime`:  NTuple{4,Symbol} – :lit, :shadow, or :transition
"""
function wedge_transition_args(
    wedge::Wedge,
    ang::RayAngles,
    k::Number,
    L::Number;
    tol::Union{Nothing,Real} = nothing,
)
    k = _validate_wavenumber(k)
    _validate_effective_L(L)
    effective_tol = if tol === nothing
        _transition_tolerance(wedge.alpha, ang.phi, ang.phip)
    else
        requested_tol_primal = _primal_value(tol)
        isfinite(requested_tol_primal) &&
            requested_tol_primal >= zero(requested_tol_primal) ||
            throw(DomainError(tol, "transition tolerance must be finite and nonnegative"))
        tol
    end
    effective_tol_primal = _primal_value(effective_tol)
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
        g_primal = _primal_value(g)
        if abs(g_primal) < effective_tol_primal
            :transition
        elseif g_primal > zero(g_primal)
            :lit
        else
            :shadow
        end
    end

    return (gj = gj_vals, Xj = Xj_vals, regime = regimes)
end
