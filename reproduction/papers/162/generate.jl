# Regenerates paper/data CSVs and results_macros.tex.
# evidence_min_rows: 120
# evidence_density_justification: coplanar/accuracy/condition sweeps use >=120
#   log-spaced defects (about 10 per decade over 1e-15..1e-3) so the linear
#   coplanar slope, 1/eps conditioning, and cancellation floor resolve before
#   binary64; electrical sweep 220 points over 11 decades; diagnostic map
#   120 x 140; HP ensemble 600; four-term ensemble 1500; polygon 150 N values.

include(joinpath(@__DIR__, "..", "..", "common.jl"))
using .PaperReproductionPaths: output_directory, selected_backend

using UTDKernels
const PAPER_BACKEND = string(selected_backend())
PAPER_BACKEND in ("kernels", "diffutd") || error("unknown paper backend: $PAPER_BACKEND")
if PAPER_BACKEND == "diffutd"
    using DiffUTD
    include("backend_diffutd.jl")
    const Study = DiffFaceEdgeStudy
else
    const Study = UTDKernels.FaceEdge
end
const Reference = UTDKernels.FaceEdge # independent references and geometry
const BoundaryGeometry = Study.BoundaryGeometry
const edge_residual_hp = Study.edge_residual_hp
const edge_residual_direct = Study.edge_residual_direct
const edge_residual_first_order = Study.edge_residual_first_order
const coefficient_four_term = Study.coefficient_four_term
const geometric_defect_score = Study.geometric_defect_score
const condition_number = Study.condition_number
const angle_factor = Study.angle_factor
const wedge_total_field = Study.wedge_total_field
const wedge_coefficients = Study.wedge_coefficients
const wedge_go_terms = Study.wedge_go_terms
const wedge_pair_terms = Study.wedge_pair_terms
const cylinder_total = Study.cylinder_total
const cylinder_radial_derivative = Study.cylinder_radial_derivative
const regular_polygon = Study.regular_polygon
const warped_polygon = Study.warped_polygon
const vertex_wedge = Study.vertex_wedge
const vertex_defect = Study.vertex_defect
const vertex_probe_L = Study.vertex_probe_L
const vertex_lit = Study.vertex_lit
const polygon_total = Study.polygon_total
const polygon_go = Study.polygon_go
const edge_scores = Study.edge_scores
const select_modes = Study.select_modes
const incident_shadow_active = Study.incident_shadow_active
const INCIDENT_KLA_GUARD = Study.INCIDENT_KLA_GUARD
const INCIDENT_FIELD_TOL = Study.INCIDENT_FIELD_TOL
using Statistics
using Random
using Printf

const PAPER = output_directory(162)
const ROOT = dirname(PAPER)
const DATA = joinpath(PAPER, "data")
const MACRO_PATH = joinpath(DATA, "results_macros.tex")
# binary64 machine epsilon 2^-52 = eps(Float64): the roundoff scale of a
# stored sum; distinct from unit roundoff u = 2^-53 used in bounds.
const MACH_EPS = eps(Float64)
mkpath(DATA)

central_wedge(g::BoundaryGeometry) = Wedge(pi * (1 + g.eps))
central_split(g::BoundaryGeometry) = Study === UTDKernels.FaceEdge ? UTDKernels.pec_wedge_face_edge(central_wedge(g), g.Delta, g.k, g.L, g.eps) : Study.central_split(g)
edge_residual_compensated(g::BoundaryGeometry) = central_split(g).edge
face_transition(g::BoundaryGeometry) = central_split(BoundaryGeometry(g.eps, 0.0, g.k, g.L)).face
function coefficients_decomposed(g::BoundaryGeometry)
    split=central_split(g)
    return split.Ds,split.Dh
end
normalized_intrinsic_score(g::BoundaryGeometry) = Study.normalized_intrinsic_score(g)
function normalized_raw_score(g::BoundaryGeometry)
    split=central_split(g)
    return sqrt(2pi*g.k)*max(abs(split.Ds),abs(split.Dh))
end

logspace(a, b, n) = 10 .^ range(a, b; length=n)

function write_csv(path, headers::Vector{String}, rows;
                   preamble::Union{Nothing,String}=nothing)
    open(path, "w") do io
        println(io, "# generator=UTDKernels/reproduction/papers/162/generate.jl")
        println(io, "# reference=independent BigFloat unrearranged incident pair")
        preamble !== nothing && print(io, preamble)
        println(io, join(headers, ","))
        for r in rows
            parts = String[]
            for v in r
                if v isa AbstractFloat
                    push!(parts, isnan(v) ? "NaN" : @sprintf("%.17g", Float64(v)))
                else
                    push!(parts, string(v))
                end
            end
            println(io, join(parts, ","))
        end
    end
    return path
end

function log_slope(x, y)
    lx = log10.(Float64.(x))
    ly = log10.(Float64.(y))
    n = length(lx)
    xbar = sum(lx) / n
    ybar = sum(ly) / n
    return sum((lx .- xbar) .* (ly .- ybar)) / sum((lx .- xbar) .^ 2)
end

# NumPy linear quantile (method='linear'): h = 1 + p*(n-1)
function quantile_linear(v, p)
    xs = sort(Float64.(v))
    n = length(xs)
    h = 1 + p * (n - 1)
    lo = clamp(Int(floor(h)), 1, n)
    hi = clamp(Int(ceil(h)), 1, n)
    w = h - lo
    return (1 - w) * xs[lo] + w * xs[hi]
end

function tex_sci(x; digits=2)
    x = Float64(x)
    iszero(x) && return "0"
    ax = abs(x)
    e = floor(Int, log10(ax))
    m = x / 10.0^e
    ms = @sprintf("%.*f", digits, m)
    # Re-round if sprintf rounded to 10.
    if abs(parse(Float64, ms)) >= 10
        m /= 10
        e += 1
        ms = @sprintf("%.*f", digits, m)
    end
    return "$(ms)\\times 10^{$(e)}"
end

function tex_fixed(x; digits=3)
    return @sprintf("%.*f", digits, Float64(x))
end

function ensemble_hp()
    rng = MersenneTwister(2026081401)
    rows = Vector{Vector{Any}}()
    naive = Float64[]
    comp = Float64[]
    for case in 0:599
        epsv = 10^(rand(rng) * ((-4) - (-15)) + (-15))
        dfrac = rand(rng) * (0.92 - max(0.08, epsv + 0.03)) + max(0.08, epsv + 0.03)
        kLv = 10^(rand(rng) * 8 - 4)
        L = 10^(rand(rng) * 2 - 1)
        k = kLv / L
        g = BoundaryGeometry(epsv, dfrac * π, k, L)
        ref = edge_residual_hp(g; dps=80)
        sc = max(abs(ref), 1e-300)
        en = abs(edge_residual_direct(g) - ref) / sc
        ec = abs(edge_residual_compensated(g) - ref) / sc
        push!(naive, en)
        push!(comp, ec)
        push!(rows, [case, epsv, dfrac, kLv, L, en, ec])
    end
    write_csv(joinpath(DATA, "high_precision_accuracy.csv"),
        ["case_index", "eps_nd", "delta_over_pi_nd", "kL_nd", "L_m",
         "naive_rel_error_nd", "compensated_rel_error_nd"], rows)
    return Dict(
        "cases" => length(rows),
        "naive_median" => median(naive),
        "naive_p95" => quantile_linear(naive, 0.95),
        "naive_max" => maximum(naive),
        "compensated_median" => median(comp),
        "compensated_p95" => quantile_linear(comp, 0.95),
        "compensated_max" => maximum(comp),
        "median_improvement" => median(naive ./ max.(comp, 1e-300)),
        "direct_wins" => sum(n .< c for (n, c) in zip(naive, comp)),
    )
end

function ensemble_decomp()
    rng = MersenneTwister(2026081402)
    rows = Vector{Vector{Any}}()
    er = Float64[]
    idmax = 0.0
    for case in 0:1499
        epsv = 10^(rand(rng) * (log10(0.2) - (-7)) + (-7))
        dfrac = rand(rng) * (0.94 - min(0.88, epsv + 0.04)) + min(0.88, epsv + 0.04)
        kLv = 10^(rand(rng) * 6 - 3)
        L = 10^(rand(rng) * 2 - 1)
        g = BoundaryGeometry(epsv, dfrac * π, kLv / L, L)
        ds, dh = coefficients_decomposed(g)
        ds0 = coefficient_four_term(g, :soft)
        dh0 = coefficient_four_term(g, :hard)
        se = abs(ds - ds0) / max(abs(ds0), 1e-14)
        he = abs(dh - dh0) / max(abs(dh0), 1e-14)
        sid = abs((dh + ds) / 2 - edge_residual_compensated(g))
        did = abs((dh - ds) / 2 - face_transition(g))
        push!(er, max(se, he))
        idmax = max(idmax, sid, did)
        push!(rows, [case, epsv, dfrac, kLv, se, he, sid, did])
    end
    write_csv(joinpath(DATA, "decomposition_ensemble.csv"),
        ["case_index", "eps_nd", "delta_over_pi_nd", "kL_nd",
         "soft_rel_error_nd", "hard_rel_error_nd",
         "sum_identity_abs_nd", "difference_identity_abs_nd"], rows)
    return Dict(
        "cases" => length(rows),
        "median_rel_error" => median(er),
        "p95_rel_error" => quantile_linear(er, 0.95),
        "max_rel_error" => maximum(er),
        "max_identity_abs" => idmax,
    )
end

function sweep_scaling()
    rows = Vector{Vector{Any}}()
    es = logspace(-10, -0.7, 180)
    for kLv in (0.01, 1.0, 100.0)
        for e in es
            g = BoundaryGeometry(e, 0.0, kLv, 1.0)
            ds, dh = coefficients_decomposed(g)
            push!(rows, [e, kLv, abs(edge_residual_compensated(g)), abs(face_transition(g)),
                         abs(ds), abs(dh), normalized_intrinsic_score(g), normalized_raw_score(g)])
        end
    end
    write_csv(joinpath(DATA, "coplanar_scaling.csv"),
        ["eps_nd", "kL_nd", "edge_abs_sqrtm", "face_abs_sqrtm",
         "soft_abs_sqrtm", "hard_abs_sqrtm", "intrinsic_score_nd", "raw_score_nd"], rows)
    ss = Dict{String,Float64}()
    for kLv in (0.01, 1.0, 100.0)
        z = [r for r in rows if r[2] == kLv && r[1] <= 1e-4]
        ss[string(kLv)] = log_slope(getindex.(z, 1), getindex.(z, 3))
    end
    return Dict("edge_slopes" => ss, "edge_slope_mid" => ss["1.0"])
end

function sweep_kL()
    rows = Vector{Vector{Any}}()
    xs = logspace(-6, 5, 220)
    for e in (0.005, 0.02, 0.1, 0.3)
        for x in xs
            g = BoundaryGeometry(e, 0.0, x, 1.0)
            push!(rows, [e, x, normalized_intrinsic_score(g), normalized_raw_score(g)])
        end
    end
    write_csv(joinpath(DATA, "electrical_length.csv"),
        ["eps_nd", "kL_nd", "intrinsic_score_nd", "raw_score_nd"], rows)
    z = [r for r in rows if r[1] == 0.02]
    small = [r for r in z if 1e-6 <= r[2] <= 1e-3]
    large = [r for r in z if 1e2 <= r[2] <= 1e5]
    return Dict(
        "small_kL_slope" => log_slope(getindex.(small, 2), getindex.(small, 3)),
        "large_kL_slope" => log_slope(getindex.(large, 2), getindex.(large, 3)),
    )
end

function sweep_asym()
    rows = Vector{Vector{Any}}()
    p95 = Dict{String,Float64}()
    for e in (1e-2, 1e-4, 1e-6)
        rels = Float64[]
        for dfrac in range(0.05, 0.92; length=180)
            dfrac <= e + 0.01 && continue
            g = BoundaryGeometry(e, dfrac * π, 20.0, 1.0)
            exact = edge_residual_compensated(g)
            first = edge_residual_first_order(g)
            re = abs(exact - first) / max(abs(exact), 1e-300)
            push!(rels, re)
            push!(rows, [e, dfrac, abs(exact) / e, abs(first) / e, re])
        end
        p95[string(e)] = quantile_linear(rels, 0.95)
    end
    write_csv(joinpath(DATA, "asymmetry_sweep.csv"),
        ["eps_nd", "delta_over_pi_nd", "scaled_exact_sqrtm", "scaled_first_order_sqrtm",
         "first_order_rel_error_nd"], rows)
    return Dict("first_order_p95_rel_error" => p95)
end

function sweep_polygon()
    Ns = unique(Int.(round.(logspace(log10(8), log10(1_000_000), 150))))
    rows = Vector{Vector{Any}}()
    kLv = 50.0
    k_fixed = 5.0
    structural = normalized_intrinsic_score(BoundaryGeometry(0.5, 0.0, kLv, 1.0))
    for N in Ns
        epsv = 2 / Float64(N)
        g = BoundaryGeometry(epsv, 0.0, kLv, 1.0)
        Lgeom = sin(π / Float64(N))
        gg = BoundaryGeometry(epsv, 0.0, k_fixed, Lgeom)
        push!(rows, [N, epsv, normalized_raw_score(g), normalized_intrinsic_score(g),
                     geometric_defect_score(g), structural, Lgeom, k_fixed * Lgeom,
                     normalized_intrinsic_score(gg), normalized_raw_score(gg)])
    end
    write_csv(joinpath(DATA, "polygon_refinement.csv"),
        ["N_count", "eps_nd", "raw_score_nd", "intrinsic_score_nd", "geometric_score_nd",
         "structural_reference_nd", "physical_L_m", "physical_kL_nd",
         "physical_intrinsic_score_nd", "physical_raw_score_nd"], rows)
    tail = [r for r in rows if r[1] >= 128]
    tail2 = [r for r in rows if r[1] >= 16384]
    Nth = first(r[1] for r in rows if r[4] < 1e-2)
    return Dict(
        "intrinsic_slope" => log_slope(getindex.(tail, 1), getindex.(tail, 4)),
        "geometric_slope" => log_slope(getindex.(tail, 1), getindex.(tail, 5)),
        "raw_slope" => log_slope(getindex.(tail, 1), getindex.(tail, 3)),
        "structural_reference" => structural,
        "physical_intrinsic_slope" => log_slope(getindex.(tail2, 1), getindex.(tail2, 9)),
        "physical_raw_slope" => log_slope(getindex.(tail2, 1), getindex.(tail2, 10)),
        "N_for_intrinsic_below_1e_2" => Int(Nth),
    )
end

function angle_factor_sweep()
    rows = Vector{Vector{Any}}()
    for n in range(1.000001, 2.0; length=360)
        epsv = n - 1.0
        # stable form: the defect is supplied, not recovered by
        # subtracting rounded angles near pi/2
        g = angle_factor(n, epsv)
        push!(rows, [n, epsv, g, 0.5 * π * epsv])
    end
    write_csv(joinpath(DATA, "angle_factor.csv"),
        ["n_nd", "eps_nd", "angle_factor_nd", "first_order_nd"], rows)
    return Dict(
        "endpoint" => rows[end][3],
        "monotone" => all(rows[i + 1][3] > rows[i][3] for i in 1:length(rows)-1),
    )
end

function condition_sweep()
    rows = Vector{Vector{Any}}()
    for e in logspace(-15, -2, 150)
        g = BoundaryGeometry(e, 0.5π, 100.0, 1.0)
        cond = condition_number(g)
        ref = edge_residual_hp(g; dps=80)
        push!(rows, [e, cond, MACH_EPS * cond,
                     abs(edge_residual_direct(g) - ref) / abs(ref),
                     abs(edge_residual_compensated(g) - ref) / abs(ref)])
    end
    write_csv(joinpath(DATA, "condition_sweep.csv"),
        ["eps_nd", "condition_number_nd", "eps_mach_kappa_nd",
         "naive_rel_error_nd", "compensated_rel_error_nd"], rows)
    z = [r for r in rows if r[1] <= 1e-6]
    naive_w = sum(r[4] < r[5] for r in rows)
    return Dict("condition_slope" => log_slope(getindex.(z, 1), getindex.(z, 2)),
                "direct_wins" => naive_w,
                "compensated_max" => maximum(r[5] for r in rows))
end

function accuracy_curve()
    rows = Vector{Vector{Any}}()
    for d in (0.2, 0.5, 0.8)
        for e in logspace(-15, -3, 120)
            g = BoundaryGeometry(e, d * π, 100.0, 1.0)
            ref = edge_residual_hp(g; dps=80)
            push!(rows, [e, d, abs(edge_residual_direct(g) - ref) / abs(ref),
                         abs(edge_residual_compensated(g) - ref) / abs(ref)])
        end
    end
    write_csv(joinpath(DATA, "accuracy_curve.csv"),
        ["eps_nd", "delta_over_pi_nd", "naive_rel_error_nd", "compensated_rel_error_nd"], rows)
    imax = argmax(r[4] for r in rows)
    return Dict(
        "compensated_max" => rows[imax][4],
        "argmax_eps" => rows[imax][1],
        "argmax_delta" => rows[imax][2],
        "argmax_direct" => rows[imax][3],
        # how much more accurate the direct evaluator is at this sample
        "argmax_ratio" => rows[imax][4] / rows[imax][3],
        "direct_wins" => sum(r[3] < r[4] for r in rows),
        "cases" => length(rows),
    )
end

function diagnostic_map()
    es = logspace(-8, -0.3, 120)
    xs = logspace(-6, 5, 140)
    rows = Vector{Vector{Any}}()
    min_int = Inf
    max_ratio = 0.0
    for e in es
        for x in xs
            g = BoundaryGeometry(e, 0.0, x, 1.0)
            ei = normalized_intrinsic_score(g)
            raw = normalized_raw_score(g)
            ratio = raw / max(ei, 1e-300)
            min_int = min(min_int, ei)
            max_ratio = max(max_ratio, ratio)
            push!(rows, [e, x, ei, ratio])
        end
    end
    write_csv(joinpath(DATA, "diagnostic_map.csv"),
        ["eps_nd", "kL_nd", "intrinsic_score_nd", "raw_to_intrinsic_nd"], rows)
    return Dict("grid_points" => length(rows), "min_intrinsic" => min_int,
                "max_raw_to_intrinsic" => max_ratio)
end

# ---------------------------------------------------------------
# Total-field and pruning studies (revision evidence)
# ---------------------------------------------------------------

"""Counterexample regression: small score, order-one reflection field."""
function counterexample_study()
    e = 1e-6
    n = 1 + e
    k = 1.0
    s = 100.0
    g = BoundaryGeometry(e, 0.0, k, s)
    split = central_split(g)
    phim = phipm = n * pi / 2
    rows = Vector{Vector{Any}}()
    stats = Dict{String,Float64}()
    for pol in (:soft, :hard)
        uf = wedge_total_field(phim, phipm, n, k, s, pol; mode=:full)
        ud = wedge_total_field(phim, phipm, n, k, s, pol; mode=:delete)
        up = wedge_total_field(phim, phipm, n, k, s, pol; mode=:preserve)
        push!(rows, [string(pol), abs(uf), abs(uf - ud), abs(uf - up)])
        stats["midpoint_delete_$(pol)"] = abs(uf - ud)
        stats["midpoint_preserve_$(pol)"] = abs(uf - up)
    end
    # Full-exterior sweep plus a dense sweep through the gap between the
    # two reflection boundaries (c in the S = pi + c*pi*eps path
    # coordinate, offset by a half step so no point lands exactly on a
    # boundary where the coefficient is defined only one-sidedly).
    for pol in (:soft, :hard)
        emax_p = 0.0
        emax_d = 0.0
        umax_f = 0.0
        for phi in range(1e-4, n * pi - 1e-4, length=4000)
            uf = wedge_total_field(phi, phipm, n, k, s, pol; mode=:full)
            up = wedge_total_field(phi, phipm, n, k, s, pol; mode=:preserve)
            ud = wedge_total_field(phi, phipm, n, k, s, pol; mode=:delete)
            emax_p = max(emax_p, abs(uf - up))
            emax_d = max(emax_d, abs(uf - ud))
            umax_f = max(umax_f, abs(uf))
        end
        for c in range(-3.0, 5.0, length=1600)
            cc = c + 0.0037
            S = pi + cc * pi * e
            phi = S - phipm
            (phi <= 0 || phi >= n * pi) && continue
            uf = wedge_total_field(phi, phipm, n, k, s, pol; mode=:full)
            up = wedge_total_field(phi, phipm, n, k, s, pol; mode=:preserve)
            ud = wedge_total_field(phi, phipm, n, k, s, pol; mode=:delete)
            emax_p = max(emax_p, abs(uf - up))
            emax_d = max(emax_d, abs(uf - ud))
            umax_f = max(umax_f, abs(uf))
        end
        stats["sweep_preserve_$(pol)"] = emax_p
        stats["sweep_delete_$(pol)"] = emax_d
        push!(rows, ["$(pol)_sweep", umax_f, emax_d, emax_p])
    end
    stats["eta"] = normalized_intrinsic_score(g)
    stats["face_over_sqrtL_abs"] = abs(split.face / sqrt(s))
    write_csv(joinpath(DATA, "counterexample.csv"),
        ["case", "u_full_abs_nd", "delete_err_nd", "preserve_err_nd"], rows)
    return stats
end

"""
Incident-shadow counterexample: the rejected-edge rule that keeps only the
reflection pair loses the incident pair where its transition is active.
Geometry: eps=1e-6, n=1+eps, phip=pi*eps/2 (grazing incidence), k=1,
L=s=100. At phi=pi+3*pi*eps/4 the symmetric midpoint score is ~1.57e-6 but
|Di|/sqrt(L) = 0.9999823: dropping the incident pair removes ~1 unit of
unit-normalized field. The sweep crosses the interior incident-shadow
boundary at Delta = phi - phip = pi (reachable only for phi - phip - pi <
pi*eps/2 of the shadow side) and reports full, safeguarded-preserve,
reflection-only-preserve, and delete errors for both polarizations.
"""
function incident_shadow_study()
    e = 1e-6
    n = 1 + e
    k = 1.0
    s = 100.0
    phip = pi * e / 2
    g = BoundaryGeometry(e, 0.0, k, s)
    stats = Dict{String,Float64}()
    stats["eta"] = normalized_intrinsic_score(g)
    phi_cex = pi + 3pi * e / 4
    rows = Vector{Vector{Any}}()
    # Field magnitudes and omitted-pair error at the counterexample point.
    for pol in (:soft, :hard)
        uf = wedge_total_field(phi_cex, phip, n, k, s, pol; mode=:full)
        up = wedge_total_field(phi_cex, phip, n, k, s, pol; mode=:preserve)
        ur = wedge_total_field(phi_cex, phip, n, k, s, pol;
                               mode=:preserve_reflection)
        push!(rows, [string(pol), abs(uf), abs(uf - ur), abs(uf - up),
                     abs(ur)])
        stats["cex_full_$(pol)"] = abs(uf)
        stats["cex_pres_refl_err_$(pol)"] = abs(uf - ur)
        stats["cex_pres_err_$(pol)"] = abs(uf - up)
    end
    # High-precision pair oracle: the actual omitted incident-pair field.
    Di_hp, _ = Reference.wedge_pairs_hp(
        BigFloat(π) + 3BigFloat(π) * BigFloat("1e-6") / 4,
        BigFloat(π) * BigFloat("1e-6") / 2,
        1 + BigFloat("1e-6"), BigFloat(1), BigFloat(100); dps=80,
    )
    stats["incident_pair_over_sqrtL"] =
        Float64(abs(Di_hp) / sqrt(BigFloat(100)))
    c_cex = wedge_coefficients(phi_cex, phip, n, k, s)
    stats["incident_pair_over_sqrtL_f64"] = abs(c_cex.De) / sqrt(s)
    # One-sided incident-boundary differences (Float64 offsets inside the
    # wedge): complete and safeguarded fields stay continuous; the
    # reflection-only preserved field jumps by ~1.
    h = 1e-7
    for pol in (:soft, :hard)
        um_f = wedge_total_field(phip + pi - h, phip, n, k, s, pol;
                                 mode=:full)
        up_f = wedge_total_field(phip + pi + h, phip, n, k, s, pol;
                                 mode=:full)
        um_r = wedge_total_field(phip + pi - h, phip, n, k, s, pol;
                                 mode=:preserve_reflection)
        up_r = wedge_total_field(phip + pi + h, phip, n, k, s, pol;
                                 mode=:preserve_reflection)
        stats["inc_boundary_full_jump_$(pol)"] = abs(up_f - um_f)
        stats["inc_boundary_pres_refl_jump_$(pol)"] = abs(up_r - um_r)
    end
    write_csv(joinpath(DATA, "incident_shadow_cex.csv"),
        ["case", "u_full_abs_nd", "pres_refl_err_nd", "preserve_err_nd",
         "u_pres_refl_abs_nd"], rows)
    # Dense sweep across the interior incident boundary. The shadow side
    # spans only delta - pi < pi*eps/2; the lit side is sampled back to
    # delta - pi = -1.0 rad so the safeguard's switch-on band
    # (k L a_min <= INCIDENT_KLA_GUARD, here |delta-pi| <~ 0.4) is visible.
    # The grid offsets avoid the exact locus delta = pi where the
    # coefficient is defined only one-sidedly.
    srows = Vector{Vector{Any}}()
    for pol in (:soft, :hard)
        for dd in range(-1.0, 0.49 * pi * e, length=1200)
            d = pi + dd
            d <= 0 && continue
            phi = phip + d
            (phi <= 0 || phi >= n * pi) && continue
            uf = wedge_total_field(phi, phip, n, k, s, pol; mode=:full)
            up = wedge_total_field(phi, phip, n, k, s, pol; mode=:preserve)
            ur = wedge_total_field(phi, phip, n, k, s, pol;
                                   mode=:preserve_reflection)
            ud = wedge_total_field(phi, phip, n, k, s, pol; mode=:delete)
            ui, ur0, urn = wedge_go_terms(phi, phip, n, k, s, pol)
            push!(srows, [dd / pi, string(pol), abs(ui + ur0 + urn),
                          abs(uf), abs(uf - up), abs(uf - ur),
                          abs(uf - ud), abs(ur)])
        end
    end
    write_csv(joinpath(DATA, "incident_shadow_sweep.csv"),
        ["delta_minus_pi_over_pi_nd", "polarization", "u_go_abs_nd",
         "u_full_abs_nd", "preserve_err_nd", "pres_refl_err_nd",
         "delete_err_nd", "u_pres_refl_abs_nd"], srows)
    # Max errors over the sweep for both pols (for macros).
    for pol in (:soft, :hard)
        sel = [r for r in srows if r[2] == string(pol)]
        stats["sweep_pres_refl_err_$(pol)"] = maximum(r[6] for r in sel)
        stats["sweep_pres_err_$(pol)"] = maximum(r[5] for r in sel)
    end
    return stats
end

"""
Total field through both reflection boundaries, parameterized by the
path coordinate c with S = pi + c*pi*eps: c = 0 and c = 2 are the two
face-reflection boundaries, c = 1 is the midpoint slice.
"""
function boundary_sweep()
    k = 1.0
    s = 100.0
    rows = Vector{Vector{Any}}()
    for e in (1e-2, 1e-3)
        n = 1 + e
        phip = n * pi / 2
        # 420 points keep the grid off the exact boundary loci c = 0, 2,
        # where the coefficient is defined only one-sidedly.
        for c in range(-2.0, 5.0, length=420)
            S = pi + c * pi * e
            phi = S - phip
            (phi <= 0 || phi >= n * pi) && continue
            for pol in (:soft, :hard)
                uf = wedge_total_field(phi, phip, n, k, s, pol; mode=:full)
                up = wedge_total_field(phi, phip, n, k, s, pol; mode=:preserve)
                ud = wedge_total_field(phi, phip, n, k, s, pol; mode=:delete)
                ui, ur0, urn = wedge_go_terms(phi, phip, n, k, s, pol)
                push!(rows, [e, c, phi / pi, string(pol), abs(uf), abs(up),
                             abs(ud), abs(uf - up), abs(uf - ud),
                             abs(ui + ur0 + urn)])
            end
        end
    end
    write_csv(joinpath(DATA, "boundary_sweep.csv"),
        ["eps_nd", "c_nd", "phi_over_pi_nd", "polarization",
         "u_full_abs_nd", "u_preserve_abs_nd", "u_delete_abs_nd",
         "preserve_err_nd", "delete_err_nd", "u_go_abs_nd"], rows)
    return Dict("rows" => length(rows))
end

"""
Coplanar path limit: |Df|/sqrt(L) vs c for S = pi + c*pi*eps approaches
the boxcar on 0 < c < 2 as eps -> 0 (independent-path limit of the
reflection pair away from the midpoint slice).
"""
function path_limit_sweep()
    k = 1.0
    s = 100.0
    rows = Vector{Vector{Any}}()
    for e in (1e-2, 1e-4, 1e-6)
        n = 1 + e
        phip = n * pi / 2
        # 450 points keep the grid off the exact boundary loci c = 0, 2.
        for c in range(-1.0, 3.5, length=450)
            S = pi + c * pi * e
            phi = S - phip
            (phi <= 0 || phi >= n * pi) && continue
            cf = wedge_coefficients(phi, phip, n, k, s)
            push!(rows, [e, c, abs(cf.Df) / sqrt(s),
                         real(cf.Df) / sqrt(s), abs(cf.De) / sqrt(s)])
        end
    end
    write_csv(joinpath(DATA, "path_limit.csv"),
        ["eps_nd", "c_nd", "Df_over_sqrtL_abs_nd",
         "Df_over_sqrtL_real_sqrtm", "De_over_sqrtL_abs_nd"], rows)
    return Dict("rows" => length(rows))
end

"""
Multi-edge object: warped 128-gon GO+UTD versus the exact PEC cylinder
eigenfunction series on an observation circle.
"""
function polygon_field_study()
    k = 2pi
    R = 1.0
    N = 128
    rho = 8.0
    beta = pi
    wp = warped_polygon(N, R)
    rows = Vector{Vector{Any}}()
    errs = Float64[]
    refs = Float64[]
    lit_errs = Float64[]
    shadow_errs = Float64[]
    refmax = 0.0
    for pol in (:soft, :hard)
        for phi in range(0, 2pi, length=361)[1:end-1]
            P = (rho * cos(phi), rho * sin(phi))
            up = polygon_total(wp, P, beta, k, pol; modes=:full)
            ug = polygon_go(wp, P, beta, k, pol)
            uc = cylinder_total(k, R, rho, phi, beta, pol)
            e = abs(up - uc)
            push!(rows, [phi / pi, string(pol), abs(up), abs(uc), e, abs(ug),
                         real(up), imag(up), real(uc), imag(uc)])
            refmax = max(refmax, abs(uc))
            push!(errs, e)
            push!(refs, abs(uc))
            abs(ug) > 0.5 ? push!(lit_errs, e) : push!(shadow_errs, e)
        end
    end
    write_csv(joinpath(DATA, "polygon_field.csv"),
        ["phi_over_pi_nd", "polarization", "u_polygon_abs_nd",
         "u_cylinder_abs_nd", "abs_err_nd", "u_go_abs_nd",
         "u_polygon_re_nd", "u_polygon_im_nd",
         "u_cylinder_re_nd", "u_cylinder_im_nd"], rows)
    return Dict(
        "max_abs_err" => maximum(errs),
        "p95_err" => quantile_linear(errs, 0.95),
        "rms_err" => sqrt(sum(e -> e^2, errs) / length(errs)),
        "lit_max_err" => maximum(lit_errs),
        "shadow_max_err" => maximum(shadow_errs),
        "ref_max" => refmax,
        "rel_l2_err" => sqrt(sum(e -> e^2, errs) /
                             sum(r -> r^2, refs)),
        "n_samples" => length(errs),
        "n_above_01" => count(>(0.1), errs),
        "n_above_05" => count(>(0.5), errs),
        "n_above_1" => count(>(1.0), errs),
    )
end

"""
Controlled refinement: regular N-gons versus the same cylinder series.
As N grows the polygon-to-cylinder geometry mismatch vanishes at the
sagitta rate O(N^-2). The experiment tests convergence of the assembled
single-diffraction model (no creeping waves or multiple diffraction); it
does not separately identify visibility and geometry errors. Same illumination, observation
circle, and error-norm definitions as the warped-polygon study.
"""
function cylinder_refinement_study()
    k = 2pi
    R = 1.0
    rho = 8.0
    beta = pi
    rows = Vector{Vector{Any}}()
    for N in (8, 16, 32, 64, 128, 256, 512)
        rp = regular_polygon(N, R)
        for pol in (:soft, :hard)
            errs = Float64[]
            refs = Float64[]
            for phi in range(0, 2pi, length=361)[1:end-1]
                P = (rho * cos(phi), rho * sin(phi))
                up = polygon_total(rp, P, beta, k, pol; modes=:full)
                uc = cylinder_total(k, R, rho, phi, beta, pol)
                push!(errs, abs(up - uc))
                push!(refs, abs(uc))
            end
            rel = sqrt(sum(e -> e^2, errs) / sum(r -> r^2, refs))
            push!(rows, [N, string(pol), maximum(errs),
                         sqrt(sum(e -> e^2, errs) / length(errs)), rel])
        end
    end
    write_csv(joinpath(DATA, "cylinder_refinement.csv"),
        ["N_count", "polarization", "max_abs_err_nd", "rms_err_nd",
         "rel_l2_err_nd"], rows)
    tail = [r for r in rows if r[1] >= 128]
    return Dict(
        "rel_l2_N512_max" => maximum(r[5] for r in rows if r[1] == 512),
        "rel_l2_N512_min" => minimum(r[5] for r in rows if r[1] == 512),
        "refine_slope" => log_slope(
            Float64[r[1] for r in tail],
            Float64[r[5] for r in tail],
        ),
    )
end

"""
Selection comparison at matched field-error tolerance. Orders the
warped-polygon vertices by each rule (intrinsic score with the local
facet-scale probe distance, angle-only defect, raw score), retains the
top m, and measures the max absolute error of the pruned field against
the unpruned polygon field. `preserve=false` repeats the intrinsic
ordering with naive deletion to expose the lost transition field. The
error probe combines the observation-circle samples with an analytic
evaluation at each rejected vertex's own gap center, so a thin
transition gap cannot hide below the angular grid.
"""
function selection_study()
    k = 2pi
    R = 1.0
    rows = Vector{Vector{Any}}()
    out = Dict{String,Any}()
    obs_phis = range(0, 2pi, length=241)[1:end-1]
    for (N, beta) in ((128, pi), (128, 2pi / 3), (256, pi))
        wp = warped_polygon(N, R)
        rho = 8.0
        obs = [(rho * cos(t), rho * sin(t)) for t in obs_phis]
        ufull = [polygon_total(wp, P, beta, k, :hard; modes=:full)
                 for P in obs]
        # per-vertex gap-center data (rejected-edge error probes)
        vtx = [vertex_wedge(wp, j) for j in 1:N]
        phipj = [Reference._local_angle(beta, w.a0) for w in vtx]
        lit = [phipj[j] < vtx[j].n * pi for j in 1:N]
        phimid = [vtx[j].n * pi - phipj[j] for j in 1:N]
        # probe distance scale for the analytic gap error: an observer at
        # the observation radius sitting on the vertex's gap center
        sj = rho
        gap_err_full = zeros(N)
        gap_err_inc = zeros(N)
        for j in 1:N
            lit[j] || continue
            c = wedge_coefficients(phimid[j], phipj[j], vtx[j].n, k, sj)
            gap_err_full[j] = abs(c.Dh) / sqrt(sj)
            gap_err_inc[j] = abs(c.De) / sqrt(sj)
        end
        Ls = [vertex_probe_L(wp, j) for j in 1:N]
        scores = Dict{Symbol,Vector{Float64}}()
        scores[:intrinsic] = edge_scores(wp, k, :intrinsic; Ls=Ls)
        scores[:angle] = edge_scores(wp, k, :angle; Ls=Ls)
        scores[:raw] = edge_scores(wp, k, :raw; Ls=Ls)
        counts = unique(Int.(round.(exp.(range(0, log(N), length=22)))))
        counts = sort(unique(vcat([0], counts, [N])))
        for (rule, sc) in scores
            # MergeSort: stable — ties keep the lower vertex index
            # (declared tie-breaking rule).
            ord = sortperm(sc; rev=true, alg=MergeSort)
            for m in counts
                keep = falses(N)
                m > 0 && (keep[ord[1:min(m, N)]] .= true)
                # observation-circle error, preserve mode
                modes = [keep[j] ? :full : :preserve for j in 1:N]
                emax = 0.0
                esq = 0.0
                for (i, P) in enumerate(obs)
                    u = polygon_total(wp, P, beta, k, :hard; modes=modes)
                    d = abs(u - ufull[i])
                    emax = max(emax, d)
                    esq += d^2
                end
                # analytic gap probes over rejected vertices: each probe is
                # the contribution the preserve rule omits at that vertex's
                # own gap center — an isolated per-edge quantity (not the
                # assembled field). Under the incident-shadow safeguard the
                # omitted incident pair is zero wherever its transition is
                # active there; otherwise it is |Di|/sqrt(s). For delete
                # mode the whole coefficient is dropped.
                gerr_p = 0.0
                gerr_d = 0.0
                gerr_i = 0.0
                for j in 1:N
                    (keep[j] || !lit[j]) && continue
                    cc = wedge_coefficients(phimid[j], phipj[j], vtx[j].n,
                                            k, sj)
                    active = incident_shadow_active(phimid[j] - phipj[j],
                                                    vtx[j].n, k, sj, cc.De)
                    gerr_p = max(gerr_p,
                                 active ? 0.0 : abs(cc.De) / sqrt(sj))
                    gerr_i = max(gerr_i, gap_err_inc[j])
                    gerr_d = max(gerr_d, gap_err_full[j])
                end
                emax = max(emax, gerr_p)
                push!(rows, [beta / pi, N, String(rule), m, emax,
                             sqrt(esq / length(obs)), gerr_p, gerr_d,
                             gerr_i])
            end
        end
        # Per-vertex release for the plotted N=128, beta=pi case:
        # defects, probe distances, scores, ranks (top-m retained), lit
        # state, and the gap-center incident-pair magnitude.
        if N == 128 && beta == pi
            vrows = Vector{Vector{Any}}()
            ords = Dict{Symbol,Vector{Int}}(
                rule => invperm(sortperm(sc; rev=true, alg=MergeSort))
                for (rule, sc) in scores
            )
            for j in 1:N
                push!(vrows, [j, vtx[j].n - 1, Ls[j], Int(lit[j]),
                              phipj[j] / pi, phimid[j] / pi, sj,
                              scores[:intrinsic][j], scores[:angle][j],
                              scores[:raw][j],
                              ords[:intrinsic][j], ords[:angle][j],
                              ords[:raw][j], gap_err_inc[j]])
            end
            write_csv(joinpath(DATA, "selection_vertices.csv"),
                ["vertex_id", "eps_nd", "probe_L_m", "lit_bool",
                 "phip_over_pi_nd", "gapcenter_phi_over_pi_nd",
                 "gap_s_m", "score_intrinsic_nd", "score_angle_nd",
                 "score_raw_nd", "rank_intrinsic", "rank_angle",
                 "rank_raw", "gap_incident_field_nd"], vrows)
        end
    end
    write_csv(joinpath(DATA, "selection_study.csv"),
        ["beta_over_pi_nd", "N_count", "rule", "retained_count",
         "preserve_max_err_nd", "preserve_rms_err_nd",
         "preserve_gap_err_nd", "delete_max_err_nd",
         "gap_incident_field_nd"], rows)
    # summarize retained counts at two tolerances: keep_* is the smallest
    # SAMPLED count meeting tol; keep_*_prev is the largest sampled count
    # below it (which necessarily failed); counts between are untested.
    for rule in ("intrinsic", "angle", "raw")
        for tol in (0.01, 0.1)
            sel = [r for r in rows if r[2] == 128 && r[1] == 1.0 &&
                                       r[3] == rule && r[5] <= tol]
            keepmin = isempty(sel) ? 128 : minimum(r[4] for r in sel)
            out["keep_$(rule)_$(tol)"] = keepmin
            below = [r[4] for r in rows if r[2] == 128 && r[1] == 1.0 &&
                                          r[3] == rule && r[4] < keepmin]
            out["keep_$(rule)_$(tol)_prev"] =
                isempty(below) ? 0 : maximum(below)
        end
    end
    sel = [r for r in rows if r[2] == 128 && r[1] == 1.0 && r[4] == 0]
    out["delete_err_m0"] = maximum(r[8] for r in sel)
    out["preserve_err_m0"] = maximum(r[5] for r in sel)
    out["N_total"] = 128
    return out
end

"""
Four-term reference with per-argument caching of the transition function:
the declared caching policy evaluates F once per distinct a and reuses it
for both cotangent factors sharing that argument (the symmetric slice has
two distinct arguments among its four terms). Near-pole terms keep the
compensated single-argument form uncached.
"""
function coefficient_four_term_cached(g::BoundaryGeometry,
                                      polarization::Symbol)
    polarization === :soft || polarization === :hard ||
        throw(ArgumentError("polarization must be :soft or :hard"))
    if iszero(g.eps)
        return polarization === :soft ? complex(-sqrt(g.L)) :
                                        complex(sqrt(g.L))
    end
    n = Reference.wedge_n(g)
    wedge_angle = Reference.alpha(g)
    betas = (g.Delta, g.Delta, wedge_angle, wedge_angle)
    signs = (1, -1, 1, -1)
    weights = polarization === :soft ? (1.0, 1.0, -1.0, -1.0) :
                                       (1.0, 1.0, 1.0, 1.0)
    cache = Dict{Float64,ComplexF64}()
    total = 0.0 + 0.0im
    for (beta, sign, weight) in zip(betas, signs, weights)
        psi = (pi + sign * beta) / (2n)
        branch = Reference.nint_kp((beta + sign * pi) / (2n * pi))
        a = 2cos((2n * pi * branch - beta) / 2)^2
        sine = sin(psi)
        if abs(sine) > 1e-7
            B = get!(() -> Reference._B(a, g.k, g.L), cache, a)
            total += weight * (cos(psi) / sine) * B
        else
            total += weight * Reference.cot_B_reference(
                psi, a, g.k, g.L)
        end
    end
    return -Reference.EIM4 * total / (2n * sqrt(2pi))
end

"""
Evaluation-count and runtime study behind the revised cost table.
Protocol: each callable is warmed up once (compilation), then timed in
`batches` batches of `reps` calls; the reported time is the median of the
per-batch mean, and min/max spread is exported for variability. Runs are
single-threaded Julia on the host recorded in the CSV header comment.
These are coefficient-level timings; they do not measure a
preservation-mode speedup in an assembled solver.
"""
function timing_study()
    reps = 200
    batches = 7
    g = BoundaryGeometry(0.01, 0.0, 50.0, 1.0)
    w = central_wedge(g)
    cals = [
        ("intrinsic_score_symmetric", 1,
         () -> normalized_intrinsic_score(g)),
        ("decomposed_symmetric_pair", 2,
         () -> central_split(g)),
        ("direct_four_term_uncached", 4,
         () -> coefficient_four_term(g, :soft)),
        ("direct_four_term_cached", 2,
         () -> coefficient_four_term_cached(g, :soft)),
        ("reflection_pair_only", 2,
         () -> wedge_pair_terms(1.62, 1.55, 1.01, g.k, g.L)[2]),
        ("general_coefficient_4term", 4,
         () -> wedge_coefficients(1.62, 1.55, 1.01, g.k, g.L)),
        ("preserve_guarded_check", 4,
         () -> begin
             c = wedge_coefficients(1.62, 1.55, 1.01, g.k, g.L)
             incident_shadow_active(1.62 - 1.55, 1.01, g.k, g.L, c.De)
         end),
    ]
    rows = Vector{Vector{Any}}()
    stats = Dict{String,Float64}()
    for (name, fcalls, f) in cals
        f()  # warmup: compile the call path before timing
        meds = Vector{Float64}(undef, batches)
        for b in 1:batches
            t = @elapsed for _ in 1:reps
                f()
            end
            meds[b] = t / reps * 1e9
        end
        sort!(meds)
        med = meds[(batches + 1) ÷ 2]
        push!(rows, [name, fcalls, med, meds[1], meds[end]])
        stats["$(name)_ns"] = med
        stats["$(name)_spread"] = meds[end] - meds[1]
        stats["$(name)_ratio"] = meds[end] / meds[1]
    end
    header = ["evaluator", "F_calls", "time_ns_median", "time_ns_min",
              "time_ns_max"]
    io = IOBuffer()
    println(io, "# timing protocol: warmup once, $(batches) batches x " *
              "$(reps) calls, median of per-batch means, single-threaded")
    println(io, "# julia $(VERSION); threads=$(Threads.nthreads()); " *
              "cpu=$(first(Sys.cpu_info()).model); os=$(Sys.MACHINE)")
    write_csv(joinpath(DATA, "evaluation_timing.csv"), header, rows;
              preamble=String(take!(io)))
    return Dict(
        "score_ns" => stats["intrinsic_score_symmetric_ns"],
        "pair_ns" => stats["decomposed_symmetric_pair_ns"],
        "direct_ns" => stats["direct_four_term_uncached_ns"],
        "cached_ns" => stats["direct_four_term_cached_ns"],
        "refl_ns" => stats["reflection_pair_only_ns"],
        "full_ns" => stats["general_coefficient_4term_ns"],
        "guard_ns" => stats["preserve_guarded_check_ns"],
        "max_spread_ratio" => maximum(stats["$(n)_ratio"] for (n, _, _) in cals),
        "score_vs_direct" => stats["direct_four_term_uncached_ns"] /
                            stats["intrinsic_score_symmetric_ns"],
        "score_vs_pair" => stats["decomposed_symmetric_pair_ns"] /
                           stats["intrinsic_score_symmetric_ns"],
    )
end

function write_macros(summary)
    hp = summary["high_precision"]
    dc = summary["decomposition"]
    sc = summary["coplanar_scaling"]
    el = summary["electrical_length"]
    po = summary["polygon_refinement"]
    aa = summary["asymmetry"]
    cond = summary["conditioning"]
    dm = summary["diagnostic_map"]
    ac = summary["accuracy_det"]
    ce = summary["counterexample"]
    ish = summary["incident_shadow"]
    pf = summary["polygon_field"]
    cr = summary["cylinder_refinement"]
    sel = summary["selection"]
    tm = summary["timing"]
    lines = String[
        "% Auto-generated by UTDKernels/reproduction/papers/162/generate.jl",
        "\\newcommand{\\HPcases}{$(hp["cases"])}",
        "\\newcommand{\\CompMedianErr}{$(tex_sci(hp["compensated_median"]))}",
        "\\newcommand{\\CompPtfErr}{$(tex_sci(hp["compensated_p95"]))}",
        "\\newcommand{\\CompMaxErr}{$(tex_sci(hp["compensated_max"]))}",
        "\\newcommand{\\NaivePtfErr}{$(tex_sci(hp["naive_p95"]))}",
        "\\newcommand{\\NaiveMaxErr}{$(tex_sci(hp["naive_max"]; digits=2))}",
        "\\newcommand{\\MedianImprovement}{$(tex_sci(hp["median_improvement"]))}",
        "\\newcommand{\\HpDirectWins}{$(hp["direct_wins"])}",
        "\\newcommand{\\DecompCases}{$(dc["cases"])}",
        "\\newcommand{\\DecompMaxErr}{$(tex_sci(dc["max_rel_error"]))}",
        "\\newcommand{\\IdentityMaxErr}{$(tex_sci(dc["max_identity_abs"]))}",
        "\\newcommand{\\LinearSlopeLow}{$(tex_fixed(sc["edge_slopes"]["0.01"]))}",
        "\\newcommand{\\LinearSlopeMid}{$(tex_fixed(sc["edge_slopes"]["1.0"]))}",
        "\\newcommand{\\LinearSlopeHigh}{$(tex_fixed(sc["edge_slopes"]["100.0"]))}",
        "\\newcommand{\\SmallKLSlope}{$(tex_fixed(el["small_kL_slope"]))}",
        "\\newcommand{\\LargeKLSlope}{$(tex_sci(el["large_kL_slope"]; digits=2))}",
        "\\newcommand{\\PolygonIntrinsicSlope}{$(tex_fixed(po["intrinsic_slope"]))}",
        "\\newcommand{\\PolygonGeometrySlope}{$(tex_fixed(po["geometric_slope"]))}",
        "\\newcommand{\\PolygonRawSlope}{$(tex_fixed(po["raw_slope"]))}",
        "\\newcommand{\\PolygonNThreshold}{$(po["N_for_intrinsic_below_1e_2"])}",
        "\\newcommand{\\PolygonPhysicalIntrinsicSlope}{$(tex_fixed(po["physical_intrinsic_slope"]))}",
        "\\newcommand{\\PolygonPhysicalRawSlope}{$(tex_fixed(po["physical_raw_slope"]))}",
        "\\newcommand{\\AsymPtfAtOneEminusSix}{$(tex_sci(aa["first_order_p95_rel_error"]["1.0e-6"]))}",
        "\\newcommand{\\AsymPtfAtOneEminusTwo}{$(tex_sci(aa["first_order_p95_rel_error"]["0.01"]))}",
        "\\newcommand{\\ConditionSlope}{$(tex_fixed(cond["condition_slope"]))}",
        "\\newcommand{\\CondDirectWins}{$(cond["direct_wins"])}",
        "\\newcommand{\\DetCompMaxErr}{$(tex_sci(ac["compensated_max"]))}",
        "\\newcommand{\\DetMaxArgEps}{$(tex_sci(ac["argmax_eps"]))}",
        "\\newcommand{\\DetMaxArgDelta}{$(tex_fixed(ac["argmax_delta"]))}",
        "\\newcommand{\\DetMaxDirect}{$(tex_sci(ac["argmax_direct"]))}",
        "\\newcommand{\\DetMaxRatio}{$(tex_fixed(ac["argmax_ratio"]))}",
        "\\newcommand{\\DetDirectWins}{$(ac["direct_wins"])}",
        "\\newcommand{\\DetCases}{$(ac["cases"])}",
        "\\newcommand{\\MapPoints}{$(dm["grid_points"])}",
        "\\newcommand{\\AngleEndpoint}{$(tex_fixed(summary["angle_factor"]["endpoint"]; digits=1))}",
        "\\newcommand{\\CexEta}{$(tex_sci(ce["eta"]))}",
        "\\newcommand{\\CexFaceAbs}{$(tex_fixed(ce["face_over_sqrtL_abs"]; digits=5))}",
        "\\newcommand{\\CexDelMid}{$(tex_fixed(ce["midpoint_delete_hard"]; digits=7))}",
        "\\newcommand{\\CexPrvMid}{$(tex_sci(ce["midpoint_preserve_hard"]))}",
        "\\newcommand{\\CexDelSweep}{$(tex_fixed(ce["sweep_delete_hard"]; digits=7))}",
        "\\newcommand{\\CexPrvSweep}{$(tex_sci(ce["sweep_preserve_hard"]))}",
        "\\newcommand{\\CexIncEta}{$(tex_sci(ish["eta"]))}",
        "\\newcommand{\\CexIncPair}{$(tex_fixed(ish["incident_pair_over_sqrtL"]; digits=7))}",
        "\\newcommand{\\CexIncFullHard}{$(tex_fixed(ish["cex_full_hard"]; digits=7))}",
        "\\newcommand{\\CexIncPresReflErrHard}{$(tex_fixed(ish["cex_pres_refl_err_hard"]; digits=7))}",
        "\\newcommand{\\CexIncPresErrHard}{$(tex_sci(ish["cex_pres_err_hard"]))}",
        "\\newcommand{\\CexIncJumpFullHard}{$(tex_sci(ish["inc_boundary_full_jump_hard"]))}",
        "\\newcommand{\\CexIncJumpReflHard}{$(tex_fixed(ish["inc_boundary_pres_refl_jump_hard"]; digits=4))}",
        "\\newcommand{\\PolyN}{$(sel["N_total"])}",
        "\\newcommand{\\PolyMaxErr}{$(tex_sci(pf["max_abs_err"]))}",
        "\\newcommand{\\PolyPtfErr}{$(tex_sci(pf["p95_err"]))}",
        "\\newcommand{\\PolyRmsErr}{$(tex_sci(pf["rms_err"]))}",
        "\\newcommand{\\PolyLitMaxErr}{$(tex_sci(pf["lit_max_err"]))}",
        "\\newcommand{\\PolyShadowMaxErr}{$(tex_fixed(pf["shadow_max_err"]))}",
        "\\newcommand{\\PolyRefMax}{$(tex_fixed(pf["ref_max"]))}",
        "\\newcommand{\\PolyRelLTwoPct}{$(tex_fixed(100 * pf["rel_l2_err"]; digits=2))}",
        "\\newcommand{\\PolyNSamples}{$(pf["n_samples"])}",
        "\\newcommand{\\PolyAboveOneTenth}{$(pf["n_above_01"])}",
        "\\newcommand{\\PolyAboveOneTenthPct}{$(tex_fixed(100 * pf["n_above_01"] / pf["n_samples"]; digits=2))}",
        "\\newcommand{\\RefineRelLTwoLargest}{$(tex_fixed(100 * cr["rel_l2_N512_max"]; digits=2))}",
        "\\newcommand{\\RefineRelLTwoSmallest}{$(tex_fixed(100 * cr["rel_l2_N512_min"]; digits=2))}",
        "\\newcommand{\\RefineSlope}{$(tex_fixed(cr["refine_slope"]))}",
        "\\newcommand{\\KeepIntrinsicLo}{$(sel["keep_intrinsic_0.01"])}",
        "\\newcommand{\\KeepIntrinsicLoPrev}{$(sel["keep_intrinsic_0.01_prev"])}",
        "\\newcommand{\\KeepIntrinsicHi}{$(sel["keep_intrinsic_0.1"])}",
        "\\newcommand{\\KeepIntrinsicHiPrev}{$(sel["keep_intrinsic_0.1_prev"])}",
        "\\newcommand{\\KeepAngleLo}{$(sel["keep_angle_0.01"])}",
        "\\newcommand{\\KeepAngleLoPrev}{$(sel["keep_angle_0.01_prev"])}",
        "\\newcommand{\\KeepAngleHi}{$(sel["keep_angle_0.1"])}",
        "\\newcommand{\\KeepAngleHiPrev}{$(sel["keep_angle_0.1_prev"])}",
        "\\newcommand{\\KeepRawLo}{$(sel["keep_raw_0.01"])}",
        "\\newcommand{\\KeepRawLoPrev}{$(sel["keep_raw_0.01_prev"])}",
        "\\newcommand{\\KeepRawHi}{$(sel["keep_raw_0.1"])}",
        "\\newcommand{\\KeepRawHiPrev}{$(sel["keep_raw_0.1_prev"])}",
        "\\newcommand{\\DelErrAllPruned}{$(tex_fixed(sel["delete_err_m0"]; digits=2))}",
        "\\newcommand{\\PrvErrAllPruned}{$(tex_sci(sel["preserve_err_m0"]))}",
        "\\newcommand{\\ScoreNs}{$(tex_fixed(tm["score_ns"]; digits=2))}",
        "\\newcommand{\\PairNs}{$(tex_fixed(tm["pair_ns"]; digits=2))}",
        "\\newcommand{\\DirectNs}{$(tex_fixed(tm["direct_ns"]; digits=2))}",
        "\\newcommand{\\CachedNs}{$(tex_fixed(tm["cached_ns"]; digits=2))}",
        "\\newcommand{\\ReflPairNs}{$(tex_fixed(tm["refl_ns"]; digits=2))}",
        "\\newcommand{\\GeneralNs}{$(tex_fixed(tm["full_ns"]; digits=2))}",
        "\\newcommand{\\GuardNs}{$(tex_fixed(tm["guard_ns"]; digits=2))}",
        "\\newcommand{\\TimingMaxSpread}{$(tex_fixed(tm["max_spread_ratio"]))}",
        "\\newcommand{\\ScoreVsDirect}{$(tex_fixed(tm["score_vs_direct"]))}",
        "\\newcommand{\\ScoreVsCached}{$(tex_fixed(tm["cached_ns"] / tm["score_ns"]))}",
        "\\newcommand{\\ScoreVsPair}{$(tex_fixed(tm["score_vs_pair"]))}",
    ]
    open(MACRO_PATH, "w") do io
        println(io, join(lines, "\n"))
    end
    cp(MACRO_PATH, joinpath(PAPER, "results_macros.tex"); force=true)
    return MACRO_PATH
end

function main()
    println("running high-precision ensemble...")
    hp = ensemble_hp()
    println("  compensated p95 = ", hp["compensated_p95"], "  naive max = ", hp["naive_max"])
    println("running decomposition ensemble...")
    dc = ensemble_decomp()
    println("running deterministic sweeps...")
    sc = sweep_scaling()
    el = sweep_kL()
    aa = sweep_asym()
    po = sweep_polygon()
    af = angle_factor_sweep()
    println("running condition and accuracy high-precision sweeps...")
    cond = condition_sweep()
    ac = accuracy_curve()
    println("  deterministic compensated max = ", ac["compensated_max"],
            "  direct wins = ", ac["direct_wins"], "/", ac["cases"])
    println("running diagnostic map...")
    dm = diagnostic_map()
    println("running counterexample study...")
    ce = counterexample_study()
    println("  eta = ", ce["eta"], "  preserve sweep err = ", ce["sweep_preserve_hard"])
    println("running incident-shadow counterexample study...")
    ish = incident_shadow_study()
    println("  |Di|/sqrt(L) = ", ish["incident_pair_over_sqrtL"],
            "  pres-refl err = ", ish["cex_pres_refl_err_hard"])
    println("running boundary sweep...")
    boundary_sweep()
    println("running coplanar path-limit sweep...")
    path_limit_sweep()
    println("running polygon field study...")
    pf = polygon_field_study()
    println("  max abs err vs cylinder = ", pf["max_abs_err"])
    println("running cylinder refinement study...")
    cr = cylinder_refinement_study()
    println("  N=512 rel l2 = ", cr["rel_l2_N512_max"],
            "  tail slope = ", cr["refine_slope"])
    println("running selection study...")
    sel = selection_study()
    println("running evaluation timing study...")
    tm = timing_study()
    println("  score = ", tm["score_ns"], " ns; pair = ", tm["pair_ns"],
            " ns; direct = ", tm["direct_ns"], " ns")
    summary = Dict(
        "high_precision" => hp,
        "decomposition" => dc,
        "coplanar_scaling" => sc,
        "electrical_length" => el,
        "asymmetry" => aa,
        "polygon_refinement" => po,
        "angle_factor" => af,
        "conditioning" => cond,
        "diagnostic_map" => dm,
        "accuracy_det" => ac,
        "counterexample" => ce,
        "incident_shadow" => ish,
        "polygon_field" => pf,
        "cylinder_refinement" => cr,
        "selection" => sel,
        "timing" => tm,
    )
    write_macros(summary)
    println("wrote CSVs to ", DATA)
    return summary
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
