"""
Canonical reflection-boundary face–edge models and field assembly.

Compensated scalar evaluation delegates to the parent kernels. This module owns
the general-angle coefficient assembly, polygon fields and selection policies;
independent direct and high-precision references support validation.
"""
module FaceEdge

using SpecialFunctions: erfcx
import ..UTDKernels
using ..UTDKernels: FaceEdgeDomainError, Wedge, F_utd, pec_wedge_face_edge, pec_wedge_intrinsic_score

export BoundaryGeometry, BranchDomainError
export transition_F, transition_derivatives
export face_transition
export edge_residual_symmetric, edge_residual_direct
export edge_residual_compensated, edge_residual_first_order
export coefficients_decomposed, coefficient_four_term
export normalized_intrinsic_score, normalized_raw_score, geometric_defect_score
export condition_number, angle_factor
export edge_residual_hp, transition_F_hp, wedge_pairs_hp
export wedge_go_visibility, wedge_go_terms, wedge_pair_terms, wedge_coefficients, wedge_total_field
export incident_shadow_active, INCIDENT_KLA_GUARD, INCIDENT_FIELD_TOL
export cylinder_scattered, cylinder_total, cylinder_radial_derivative
export PolygonScatterer, regular_polygon, warped_polygon
export vertex_wedge, vertex_defect, vertex_probe_L, vertex_lit, vertex_visible
export orientation2, signed_angle2, crossing_fraction2, unit_direction2, ray_segment_intersection
export wedge_cartesian_detunings, polygon_transition_detunings, polygon_go_rays, polygon_go, polygon_total, incident_blocked
export edge_scores, select_modes

const BranchDomainError = FaceEdgeDomainError
const EIP4 = cis(pi / 4)
const EIM4 = cis(-pi / 4)

function transition_F(x::Real)
    (x >= 0 && (isfinite(x) || (isinf(x) && x > 0))) ||
        throw(DomainError(x, "transition_F requires x >= 0 and finite or +Inf"))
    return F_utd(x)
end

"""
    BoundaryGeometry(eps, Delta, k, L)

Released paper geometry. `eps=(alpha-pi)/pi` and `Delta=phi-phip` on
`phi+phip=alpha`. The separate `eps` field preserves defects too small to be
recovered accurately after rounding `alpha` into a binary64 `Wedge`.
"""
struct BoundaryGeometry{T<:Real}
    eps::T
    Delta::T
    k::T
    L::T
    function BoundaryGeometry(eps::Real, Delta::Real, k::Real, L::Real)
        e, d, kv, lv = promote(float(eps), float(Delta), float(k), float(L))
        UTDKernels._number_isfinite(e) && e >= 0 || throw(ArgumentError("eps must be finite and nonnegative"))
        UTDKernels._number_isfinite(d) && 0 <= d < UTDKernels._typed_pi(d) || throw(ArgumentError("Delta must lie in [0, pi)"))
        UTDKernels._number_isfinite(kv) && kv > 0 || throw(ArgumentError("k must be finite and positive"))
        UTDKernels._number_isfinite(lv) && lv > 0 || throw(ArgumentError("L must be finite and positive"))
        isfinite(kv * lv) && kv * lv > 0 || throw(ArgumentError(
            "k*L must be finite and positive in the active numeric type"))
        new{typeof(e)}(e, d, kv, lv)
    end
end

wedge_n(g::BoundaryGeometry) = one(g.eps) + g.eps
alpha(g::BoundaryGeometry) = pi * wedge_n(g)
phi(g::BoundaryGeometry) = (alpha(g) + g.Delta) / 2
phip(g::BoundaryGeometry) = (alpha(g) - g.Delta) / 2
kL(g::BoundaryGeometry) = g.k * g.L
branch_domain_ok(g::BoundaryGeometry) = iszero(g.Delta) || g.Delta > pi * g.eps

function require_branch(g::BoundaryGeometry)
    branch_domain_ok(g) || throw(BranchDomainError(
        "near-coplanar branch requires Delta > pi eps",
    ))
    nothing
end

@inline _central_wedge(g::BoundaryGeometry) = Wedge(pi * (one(g.eps) + g.eps))

@inline function _central_split(g::BoundaryGeometry)
    return pec_wedge_face_edge(_central_wedge(g), g.Delta, g.k, g.L, g.eps)
end

# ---------------------------------------------------------------- production adapters

face_transition(g::BoundaryGeometry) =
    pec_wedge_face_edge(_central_wedge(g), 0.0, g.k, g.L, g.eps).face

function edge_residual_symmetric(g::BoundaryGeometry)
    iszero(g.Delta) || throw(ArgumentError("symmetric formula requires Delta = 0"))
    return pec_wedge_face_edge(_central_wedge(g), 0.0, g.k, g.L, g.eps).edge
end

edge_residual_compensated(g::BoundaryGeometry) = _central_split(g).edge

function coefficients_decomposed(g::BoundaryGeometry)
    split = _central_split(g)
    return split.Ds, split.Dh
end

normalized_intrinsic_score(g::BoundaryGeometry) =
    pec_wedge_intrinsic_score(_central_wedge(g), g.Delta, g.k, g.L, g.eps)

# ----------------------------------------------------------- paper-only oracles

"""Value and first three derivatives of the parent transition evaluator."""
function transition_derivatives(x::Real)
    return UTDKernels._F_utd_derivatives3(x)
end

function sinc_rad(x::Real)
    if abs(x) < 1e-4
        x2 = x * x
        return 1.0 - x2 / 6.0 + x2^2 / 120.0 - x2^3 / 5040.0
    end
    return sin(x) / x
end

"""Nearest integer with half-away-from-zero ties, matching the paper `nint`."""
nint_kp(x::Real) = x >= 0 ? Int(floor(x + oftype(x, 0.5))) :
                            Int(ceil(x - oftype(x, 0.5)))

_B(a::Real, k::Real, L::Real) = transition_F(k * L * a) / sqrt(k)

"""Unrearranged incident-pair reference; intentionally ill-conditioned."""
function edge_residual_direct(g::BoundaryGeometry)
    iszero(g.Delta) && return edge_residual_symmetric(g)
    require_branch(g)
    n = wedge_n(g)
    psi1 = (pi + g.Delta) / (2n)
    psi2 = (pi - g.Delta) / (2n)
    a0 = 2cos(g.Delta / 2)^2
    a1 = 2cos(pi * g.eps - g.Delta / 2)^2
    incident = cot(psi1) * _B(a1, g.k, g.L)
    incident += cot(psi2) * _B(a0, g.k, g.L)
    return -EIM4 * incident / (2n * sqrt(2pi))
end

"""First-order coplanar asymptotic retained as a paper comparison."""
function edge_residual_first_order(g::BoundaryGeometry)
    b = g.Delta / 2
    x0 = 2kL(g) * cos(b)^2
    value, first, _, _ = transition_derivatives(x0)
    bracket = value / cos(b)^2 - 4kL(g) * sin(b)^2 * first
    return -EIM4 * pi * g.eps * bracket / (2sqrt(2pi * g.k))
end

function cot_B_reference(psi::Real, a::Real, k::Real, L::Real)
    sine = sin(psi)
    if abs(sine) > 1e-7
        return cot(psi) * _B(a, k, L)
    end
    iszero(sine) && throw(DivideError())
    x = k * L * a
    return cos(psi) * sqrt(pi * L * a) / sine * EIP4 *
           erfcx(EIP4 * sqrt(x))
end

"""Independently arranged KP four-term reference for moderate defects."""
function coefficient_four_term(g::BoundaryGeometry, polarization::Symbol)
    polarization === :soft || polarization === :hard ||
        throw(ArgumentError("polarization must be :soft or :hard"))
    if iszero(g.eps)
        return polarization === :soft ? complex(-sqrt(g.L)) : complex(sqrt(g.L))
    end
    n = wedge_n(g)
    wedge_angle = alpha(g)
    betas = (g.Delta, g.Delta, wedge_angle, wedge_angle)
    signs = (1, -1, 1, -1)
    weights = polarization === :soft ? (1.0, 1.0, -1.0, -1.0) :
                                        (1.0, 1.0, 1.0, 1.0)
    total = 0.0 + 0.0im
    for (beta, sign, weight) in zip(betas, signs, weights)
        psi = (pi + sign * beta) / (2n)
        branch = nint_kp((beta + sign * pi) / (2n * pi))
        a = 2cos((2n * pi * branch - beta) / 2)^2
        total += weight * cot_B_reference(psi, a, g.k, g.L)
    end
    return -EIM4 * total / (2n * sqrt(2pi))
end

function normalized_raw_score(g::BoundaryGeometry)
    split = _central_split(g)
    return sqrt(2pi) * (sqrt(g.k) * max(abs(split.Ds), abs(split.Dh)))
end

# Stable form of (1-cos(pi*eps))/2 = sin^2(pi*eps/2): for eps = 2/N this is
# sin^2(pi/N), which avoids the rounding loss of the difference form at
# large N.
geometric_defect_score(g::BoundaryGeometry) = sin(pi * g.eps / 2)^2

"""Normwise condition number of the unrearranged incident pair."""
function condition_number(g::BoundaryGeometry)
    iszero(g.Delta) && return 1.0
    require_branch(g)
    n = wedge_n(g)
    psi1 = (pi + g.Delta) / (2n)
    psi2 = (pi - g.Delta) / (2n)
    a0 = 2cos(g.Delta / 2)^2
    a1 = 2cos(pi * g.eps - g.Delta / 2)^2
    term1 = cot(psi1) * _B(a1, g.k, g.L)
    term2 = cot(psi2) * _B(a0, g.k, g.L)
    scale = -EIM4 / (2n * sqrt(2pi))
    denominator = abs(edge_residual_compensated(g) / scale)
    iszero(denominator) && return Inf
    return (abs(term1) + abs(term2)) / denominator
end

"""
Symmetric angular factor `g(n)=cot(pi/(2n))/n`. Near `n = 1` the identity
`cot(pi/(2n)) = tan(pi*eps/(2n))` is evaluated through the separately
supplied coplanar defect `eps = n - 1`, so the small angle is not
recovered by subtracting two rounded angles near pi/2. The one-argument
form derives `eps = n - 1` itself and is intended for `n` away from 1.
"""
function angle_factor(n::Real, eps::Real)
    (isfinite(n) && n > 1 && isfinite(eps) && eps >= 0) ||
        throw(DomainError((n, eps),
            "angle factor requires finite n > 1 and finite eps >= 0"))
    return tan(pi * eps / (2n)) / n
end
angle_factor(n::Real) = angle_factor(n, n - 1)

include("../../validation/FaceEdgeReferences.jl")
include("cartesian.jl")
include("fields.jl")

end
