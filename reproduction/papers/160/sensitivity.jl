if !isdefined(@__MODULE__, :PaperReproductionPaths)
    include(joinpath(@__DIR__, "..", "..", "common.jl"))
end
using .PaperReproductionPaths: output_directory, selected_backend

# Use-case evidence for the grazing continuation (R6): the normalized
# coefficient Ds/h -> -2C G'(phi) and the observation-angle sensitivity
# dDs/dphi. All differentiation routes are scored against the SAME
# high-precision five-point stencil of the BigFloat reference:
#
#   ad_continuation : forward-mode AD through the public adaptive evaluator
#   ad_fourterm     : forward-mode AD through the four-term baseline
#   ad_linear       : forward-mode AD through the O(h) linear limit
#   ad_hybrid       : forward-mode AD through the linear/four-term hybrid
#   fd4             : centered finite difference (delta = 1e-6) of the baseline
#
# The linear/hybrid route is differentiable: its sensitivity path is exact at
# the level of its own O(h^2) truncation, which is visible at moderate offsets.
# AD through the four-term baseline differentiates the SAME coalescing
# subtraction one level down (G'(phi-h) - G'(phi+h)), so it fails at deep
# offsets just like the coefficient itself. Complex outputs AND references
# are archived so every scored error can be independently recomputed.
#
# Run with:  julia --project=paper paper/scripts/sensitivity_sweep.jl

using UTDKernels
using ForwardDiff
using Printf

include(joinpath(@__DIR__, "hp_oracle.jl"))

const DATA = joinpath(output_directory(160), "data")
mkpath(DATA)

const G = Geom(1.5 * pi, 1.7, 100.0, 1.0)
const W = _wedge(G)
const H_SWITCH = 1e-3   # hybrid switching offset (same as cost_accuracy.jl)

# High-precision dDs/dphi by a five-point stencil of the reference Ds.
function dsens_hp(g::Geom, h::Real; dps::Int=60)
    prec = _prec(dps)
    setprecision(BigFloat, prec) do
        delta = BigFloat("1e-8")
        phib = BigFloat(g.phi)
        f(d) = soft_hp_terms(g, h; dps, phi_at=phib + d)[1]
        return (-f(2delta) + 8f(delta) - 8f(-delta) + f(-2delta)) / (12delta)
    end
end

hybrid_route(g::Geom, x, h::Real) =
    h <= H_SWITCH ? Production.pec_wedge_Ds_linear(W, RayAngles(x, h), g.k, g.L; face=:o) :
                    Production.pec_wedge_DsDh(W, RayAngles(x, h), g.k, g.L)[1]

hs = exp10.(range(-16, -1; length=16))
rows = Vector{Vector{Float64}}()
for h in hs
    ref_ds = soft_hp(G, h; dps=60)
    ref_sens = dsens_hp(G, h; dps=60)

    # Normalized coefficient Ds/h for each route.
    d_direct = direct_soft(G, h) / h
    d_stable = fixed_order_soft(G, h; order=8) / h
    r_direct = rel(d_direct, ref_ds / h)
    r_stable = rel(d_stable, ref_ds / h)

    # Sensitivity dDs/dphi: four differentiation routes plus the centered
    # finite-difference baseline, all scored against the same reference.
    ad = ForwardDiff.derivative(
        x -> Production.pec_wedge_DsDh_grazing(W, RayAngles(x, h), G.k, G.L)[1],
        G.phi,
    )
    ad4 = ForwardDiff.derivative(
        x -> Production.pec_wedge_DsDh(W, RayAngles(x, h), G.k, G.L)[1],
        G.phi,
    )
    adl = ForwardDiff.derivative(
        x -> Production.pec_wedge_Ds_linear(W, RayAngles(x, h), G.k, G.L; face=:o),
        G.phi,
    )
    adh = ForwardDiff.derivative(x -> hybrid_route(G, x, h), G.phi)
    d4 = 1e-6
    fd = (Production.pec_wedge_DsDh(W, RayAngles(G.phi + d4, h), G.k, G.L)[1] -
          Production.pec_wedge_DsDh(W, RayAngles(G.phi - d4, h), G.k, G.L)[1]) / (2d4)

    push!(rows, Float64[
        h, abs(ref_ds / h), r_direct, r_stable,
        abs(ref_sens), real(ref_sens), imag(ref_sens),
        real(ad), imag(ad), rel(ad, ref_sens),
        real(ad4), imag(ad4), rel(ad4, ref_sens),
        real(adl), imag(adl), rel(adl, ref_sens),
        real(adh), imag(adh), rel(adh, ref_sens),
        real(fd), imag(fd), rel(fd, ref_sens),
    ])
end

open(joinpath(DATA, "sensitivity_sweep.csv"), "w") do io
    println(io, "# generator=paper/scripts/sensitivity_sweep.jl")
    println(io, "# method=ForwardDiff through pec_wedge_DsDh_grazing / pec_wedge_DsDh / pec_wedge_Ds_linear / hybrid, plus centered FD of pec_wedge_DsDh")
    println(io, "# reference=60-digit BigFloat five-point stencil of the reference Ds (hp_oracle.jl); complex outputs and references archived")
    println(io, "# geometry=alpha=1.5pi,phi=1.7,k=100,L=1")
    println(io, "# hybrid_switch_h=", H_SWITCH)
    println(io, "# utdkernels_version=", pkgversion(UTDKernels))
    println(io, "# backend=", selected_backend())
    println(io, join([
        "h_rad", "abs_Ds_over_h_ref", "direct_Ds_over_h_rel_error",
        "stable_Ds_over_h_rel_error",
        "abs_dDs_dphi_ref", "ref_sens_re", "ref_sens_im",
        "ad_continuation_re", "ad_continuation_im", "ad_continuation_rel_error",
        "ad_fourterm_re", "ad_fourterm_im", "ad_fourterm_rel_error",
        "ad_linear_re", "ad_linear_im", "ad_linear_rel_error",
        "ad_hybrid_re", "ad_hybrid_im", "ad_hybrid_rel_error",
        "fd4_sens_re", "fd4_sens_im", "fd4_sens_rel_error",
    ], ","))
    for r in rows
        println(io, join(r, ","))
    end
end

# Guards: AD through the continuation must stay accurate at every h shown;
# AD through the linear route must stay accurate at deep offsets AND show its
# O(h^2) truncation at moderate offsets; the four-term routes must visibly
# degrade at small h under either differentiation method.
i16 = findfirst(==(1e-16), [r[1] for r in rows])
i1 = findfirst(==(0.1), [r[1] for r in rows])
ad_max = maximum(r[10] for r in rows)
ad4_tiny = rows[i16][13]
adl_tiny = rows[i16][16]
adl_p1 = rows[i1][16]
fd_tiny = rows[i16][22]
ad_max < 1e-8 || error("AD sensitivity error too large: $ad_max")
ad4_tiny > 1e-2 || error("AD through four-term did not degrade at h=1e-16: $ad4_tiny")
fd_tiny > 1e-3 || error("four-term FD did not degrade at h=1e-16: $fd_tiny")
adl_tiny < 1e-10 || error("AD through linear route inaccurate at deep offset: $adl_tiny")
adl_p1 > 1e-4 || error("linear-route O(h^2) truncation not visible at h=0.1: $adl_p1")
println("sensitivity sweep: AD-continuation max rel err = $ad_max;",
    " AD-four-term at h=1e-16: $ad4_tiny; AD-linear: $adl_tiny (h=1e-16)",
    " -> $adl_p1 (h=0.1); four-term FD at h=1e-16: $fd_tiny")
