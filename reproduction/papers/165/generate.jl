# Regenerates paper/data CSVs for the curvature-measure study.
# evidence_min_rows: 13
# evidence_density_justification: log-spaced facet counts (13 turning-angle
#   samples from N=12 to 512, and 13 ellipse N values) resolve the 2 delta/pi
#   first-order bias and the O(h)/O(h^2) split; aliasing map is 29 x 97;
#   pattern uses 721 observation angles; alias identity stores all 600 seeded
#   cases; the monostatic assembly uses 13 facet counts and 361 aspect angles.

include(joinpath(@__DIR__, "..", "..", "common.jl"))
using .PaperReproductionPaths: output_directory, selected_backend

using UTDKernels
const PAPER_BACKEND = string(selected_backend())
PAPER_BACKEND in ("kernels", "diffutd") || error("unknown paper backend: $PAPER_BACKEND")
if PAPER_BACKEND == "diffutd"
    using DiffUTD
    include("backend_diffutd.jl")
    const Study = DiffCurvatureStudy
else
    const Study = UTDKernels.CurvatureMeasure
end
const polygon_measure_sum = Study.polygon_measure_sum
const polygon_alias_sum = Study.polygon_alias_sum
const smooth_continuum = Study.smooth_continuum
const smooth_polygon_sum = Study.smooth_polygon_sum
const relative_error = Study.relative_error
const fit_log_slope = Study.fit_log_slope
const error_decomposition = Study.error_decomposition
const nonuniform_turning_partition = Study.nonuniform_turning_partition
const nonuniform_measure_sum = Study.nonuniform_measure_sum
const ellipse_points = Study.ellipse_points
const polygon_turning_angles = Study.polygon_turning_angles
const ellipse_field_sum = Study.ellipse_field_sum
const ellipse_continuum_reference = Study.ellipse_continuum_reference
using LinearAlgebra
using Random
using Statistics
using Printf

const PAPER = output_directory(165)
const ROOT = dirname(PAPER)
const DATA = joinpath(PAPER, "data")
mkpath(DATA)

const SEED = 20260819
const K = 40.0
const L = 0.8
const M_HARM = 3
const Z_HARM = 24.0
const PHI_HARM = 0.37

intrinsic_coefficient(delta, k, L) = Study.intrinsic_coefficient(delta, k, L)
local_bias_factor(delta) = Study.local_bias_factor(delta)
continuum_harmonic(m, z, phi) = curvature_continuum_harmonic(m, z, phi)

function write_csv(path::AbstractString, header::Vector{String}, rows)
    isempty(rows) && throw(ArgumentError("no rows for $path"))
    open(path, "w") do io
        println(io, join(header, ","))
        for row in rows
            parts = String[]
            for v in row
                if v isa Integer
                    push!(parts, string(v))
                elseif v isa AbstractFloat
                    push!(parts, @sprintf("%.16e", v))
                else
                    push!(parts, string(v))
                end
            end
            println(io, join(parts, ","))
        end
    end
    return path
end

function sci_tex(value::Real)
    value == 0 && return "0"
    exponent = floor(Int, log10(abs(value)))
    mantissa = value / 10.0^exponent
    return @sprintf("\\ensuremath{%.2f\\times10^{%d}}", mantissa, exponent)
end

# ------------------------------------------------------------------ 1. local bias
Ns_bias = [12, 16, 20, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
bias_rows = Vector{Vector{Any}}()
abs_beta_m1 = Float64[]
for N in Ns_bias
    delta = 2 * pi / N
    beta = local_bias_factor(delta)
    push!(abs_beta_m1, abs(beta - 1))
    push!(bias_rows, [N, delta, beta, abs(beta - 1)])
end
write_csv(
    joinpath(DATA, "local_bias.csv"),
    ["N_count", "delta_rad", "beta_1", "abs_beta_minus_1_1"],
    bias_rows,
)
bias_slope = fit_log_slope(Ns_bias[(end - 6):end], abs_beta_m1[(end - 6):end])

# ------------------------------------------------------------------ 2. harmonic
Ns = [8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 40, 48, 64, 80, 96]
I_cont = continuum_harmonic(M_HARM, Z_HARM, PHI_HARM)
conv_rows = Vector{Vector{Any}}()
raw_rel = Float64[]
deb_rel = Float64[]
identity_res = Float64[]
for N in Ns
    S = polygon_measure_sum(N, M_HARM, Z_HARM, PHI_HARM)
    beta = local_bias_factor(2 * pi / N)
    total = beta * S
    decomp = error_decomposition(N, M_HARM, Z_HARM, PHI_HARM, K, L)
    rraw = relative_error(total, I_cont)
    rdeb = relative_error(S, I_cont)
    push!(raw_rel, rraw)
    push!(deb_rel, rdeb)
    push!(identity_res, abs(decomp.residual))
    push!(
        conv_rows,
        [
            N,
            rraw,
            rdeb,
            abs(decomp.local_bias) / abs(I_cont),
            abs(decomp.aliasing) / abs(I_cont),
            abs(decomp.residual),
        ],
    )
end
write_csv(
    joinpath(DATA, "harmonic_convergence.csv"),
    [
        "N_count",
        "raw_rel_error_1",
        "debiased_rel_error_1",
        "local_bias_rel_1",
        "aliasing_rel_1",
        "identity_residual_1",
    ],
    conv_rows,
)
# The raw-slope fit uses only points whose alias term has decayed below one
# percent of the local-bias term; including the still-aliased coarse points
# steepens the fit away from the first-order local-bias law.
alias_decayed = [r[5] < 1e-2 * r[4] for r in conv_rows]
raw_slope = fit_log_slope(Ns[alias_decayed], raw_rel[alias_decayed])

# ------------------------------------------------------------------ 3. alias identity
# All 600 sampled tuples are stored for audit; the RNG is Julia's Xoshiro256**
# generator seeded with SEED = 20260819 (one draw per case, in order).
rng = Xoshiro(SEED)
alias_abs_errors = Float64[]
alias_rel_errors = Float64[]
alias_rows = Vector{Vector{Any}}()
for case in 0:599
    N = rand(rng, 8:99)
    mm = rand(rng, -6:6)
    zz = rand(rng) * (40.0 - 0.05) + 0.05
    pp = rand(rng) * 2 * pi - pi
    direct = polygon_measure_sum(N, mm, zz, pp)
    alias = polygon_alias_sum(N, mm, zz, pp; ell_max=16)
    ae = abs(direct - alias)
    re = relative_error(alias, direct)
    push!(alias_abs_errors, ae)
    push!(alias_rel_errors, re)
    push!(alias_rows, [case, N, mm, zz, pp, ae, re])
end
write_csv(
    joinpath(DATA, "alias_identity_samples.csv"),
    ["case_1", "N_count", "m_1", "z_1", "phi_rad", "abs_error_1", "rel_error_1"],
    alias_rows,
)

# ------------------------------------------------------------------ 4. absolute scaling
Ns_scale = Int[
    32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072,
    262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432,
    67108864, 134217728,
]
R = 1.0
fixed_rows = Vector{Vector{Any}}()
fixed_tot = Float64[]
geom_tot = Float64[]
for N in Ns_scale
    delta = 2 * pi / N
    fixed_L_total = abs(N * intrinsic_coefficient(delta, K, L))
    local_L = R * sin(pi / N)
    geom_total = abs(N * intrinsic_coefficient(delta, K, local_L))
    push!(fixed_tot, fixed_L_total)
    push!(geom_tot, geom_total)
    push!(fixed_rows, [N, fixed_L_total, geom_total, local_L])
end
write_csv(
    joinpath(DATA, "absolute_scaling.csv"),
    ["N_count", "fixed_L_total_m05", "geometry_scaled_total_m05", "L_geometry_m"],
    fixed_rows,
)
geom_slope = fit_log_slope(Ns_scale[(end - 5):end], geom_tot[(end - 5):end])
fixed_slope = fit_log_slope(Ns_scale[(end - 5):end], fixed_tot[(end - 5):end])

# ------------------------------------------------------------------ 5. smooth pattern
# Pattern error norm: unweighted discrete l2 norm over the 721-sample grid
# phi in [-pi, pi] including both periodic endpoints. The plotted/debiased
# quantity is the curvature quadrature S_N; the raw normalized seam sum is
# beta_N * S_N. Both errors are complex-field errors |S_N - I|, not magnitude
# errors.
phis = collect(range(-pi, pi; length=721))
N_patterns = [12, 24, 48]
pattern_rows = Vector{Vector{Any}}()
exact_pattern = [smooth_continuum(18.0, p) for p in phis]
approx_patterns = Dict{Int,Vector{ComplexF64}}()
pattern_errors = Dict{String,Float64}()
pattern_errors_raw = Dict{String,Float64}()
for N in N_patterns
    beta_N = local_bias_factor(2 * pi / N)
    approx = [smooth_polygon_sum(N, 18.0, p) for p in phis]
    approx_patterns[N] = approx
    pattern_errors[string(N)] = norm(approx .- exact_pattern) / norm(exact_pattern)
    pattern_errors_raw[string(N)] =
        norm(beta_N .* approx .- exact_pattern) / norm(exact_pattern)
    for (p, e, a) in zip(phis, exact_pattern, approx)
        push!(pattern_rows, [N, p, abs(e), abs(a), abs(a - e), abs(beta_N * a), abs(beta_N * a - e)])
    end
end
write_csv(
    joinpath(DATA, "smooth_pattern.csv"),
    [
        "N_count", "phi_rad", "exact_abs_1", "debiased_abs_1",
        "debiased_abs_error_1", "raw_abs_1", "raw_abs_error_1",
    ],
    pattern_rows,
)

# ------------------------------------------------------------------ 6. aliasing map CSV
N_grid = collect(8:2:64)
z_grid = collect(range(2.0, 50.0; length=97))
map_rows = Vector{Vector{Any}}()
for zz in z_grid
    ref = continuum_harmonic(2, zz, 0.2)
    for N in N_grid
        val = polygon_measure_sum(N, 2, zz, 0.2)
        push!(map_rows, [N, zz, relative_error(val, ref; floor=1e-14)])
    end
end
write_csv(
    joinpath(DATA, "aliasing_map.csv"),
    ["N_count", "z_1", "rel_error_1"],
    map_rows,
)

# ------------------------------------------------------------------ 7. nonuniform
Ns_nonuniform = [12, 16, 20, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
nonuniform_rows = Vector{Vector{Any}}()
nonuniform_ref = smooth_continuum(12.0, 0.2)
h_maxs = Float64[]
raw_nu = Float64[]
deb_nu = Float64[]
for N in Ns_nonuniform
    delta_cells, _ = nonuniform_turning_partition(N)
    raw_v = nonuniform_measure_sum(N, 12.0, 0.2; debias=false)
    deb_v = nonuniform_measure_sum(N, 12.0, 0.2; debias=true)
    hmax = maximum(delta_cells)
    rraw = relative_error(raw_v, nonuniform_ref)
    rdeb = relative_error(deb_v, nonuniform_ref)
    push!(h_maxs, hmax)
    push!(raw_nu, rraw)
    push!(deb_nu, rdeb)
    push!(nonuniform_rows, [N, hmax, rraw, rdeb])
end
write_csv(
    joinpath(DATA, "nonuniform_partition.csv"),
    ["N_count", "h_max_rad", "raw_rel_error_1", "debiased_rel_error_1"],
    nonuniform_rows,
)
nonuniform_raw_slope = fit_log_slope(h_maxs[(end - 5):end], raw_nu[(end - 5):end])
nonuniform_debiased_slope = fit_log_slope(h_maxs[(end - 5):end], deb_nu[(end - 5):end])

# ------------------------------------------------------------------ 8. resolution
z_res = collect(range(2.0, 60.0; length=59))
tolerances = [1e-3, 1e-6, 1e-9]
m_values = [0, 3, 6]
resolution_rows = Vector{Vector{Any}}()
n_for = Dict{Tuple{Int,Float64,Float64},Int}()
for mm in m_values
    for zz in z_res
        ref = continuum_harmonic(mm, zz, 0.31)
        for tol in tolerances
            nmin = 181
            for N in max(8, abs(mm) + 2):180
                val = polygon_measure_sum(N, mm, zz, 0.31)
                if relative_error(val, ref; floor=1e-14) <= tol
                    nmin = N
                    break
                end
            end
            n_for[(mm, zz, tol)] = nmin
            push!(resolution_rows, [mm, zz, tol, nmin])
        end
    end
end
write_csv(
    joinpath(DATA, "resolution_curves.csv"),
    ["m_1", "z_1", "tol_1", "N_min_count"],
    resolution_rows,
)

Ns_smooth = collect(8:2:64)
smooth_l2_rows = Vector{Vector{Any}}()
for N in Ns_smooth
    approx = [smooth_polygon_sum(N, 18.0, p) for p in phis]
    err = norm(approx .- exact_pattern) / norm(exact_pattern)
    push!(smooth_l2_rows, [N, err])
end
write_csv(
    joinpath(DATA, "resolution_pattern_l2.csv"),
    ["N_count", "relative_L2_error_1"],
    smooth_l2_rows,
)

# ------------------------------------------------------------------ 9. joint refinement
z_joint = collect(4.0:2.0:80.0)
# Python: numpy.linspace(-pi, pi, 181, endpoint=False)
phi_joint = (-pi) .+ (0:180) .* (2 * pi / 181)
joint_rows = Vector{Vector{Any}}()
rule_max = Dict{Int,Float64}(1 => 0.0, 2 => 0.0, 3 => 0.0)
for zz in z_joint
    ref = [smooth_continuum(zz, p) for p in phi_joint]
    nref = norm(ref)
    for (rule_id, N) in (
        (1, 48),
        (2, Int(ceil(zz + 16.0))),
        (3, Int(ceil(1.5 * zz + 8.0))),
    )
        approx = [smooth_polygon_sum(N, zz, p) for p in phi_joint]
        err = norm(approx .- ref) / nref
        rule_max[rule_id] = max(rule_max[rule_id], err)
        push!(joint_rows, [zz, rule_id, N, err])
    end
end
write_csv(
    joinpath(DATA, "joint_refinement.csv"),
    ["z_1", "rule_id_1", "N_count", "relative_L2_error_1"],
    joint_rows,
)

# ------------------------------------------------------------------ 10. ellipse
ellipse_a, ellipse_b = 1.4, 0.8
ellipse_k, ellipse_phi = 18.0, 0.4
ellipse_ref = ellipse_continuum_reference(ellipse_a, ellipse_b, ellipse_k, ellipse_phi)
ellipse_Ns = [12, 16, 20, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
ellipse_rows = Vector{Vector{Any}}()
ell_h = Float64[]
ell_raw = Float64[]
ell_deb = Float64[]
ell_turn = Float64[]
for N in ellipse_Ns
    raw, hmax, tsum = ellipse_field_sum(N, ellipse_a, ellipse_b, ellipse_k, ellipse_phi; debias=false)
    debiased, _, _ = ellipse_field_sum(N, ellipse_a, ellipse_b, ellipse_k, ellipse_phi; debias=true)
    rraw = relative_error(raw, ellipse_ref)
    rdeb = relative_error(debiased, ellipse_ref)
    push!(ell_h, hmax)
    push!(ell_raw, rraw)
    push!(ell_deb, rdeb)
    push!(ell_turn, tsum)
    push!(ellipse_rows, [N, hmax, tsum, rraw, rdeb])
end
write_csv(
    joinpath(DATA, "ellipse_convergence.csv"),
    ["N_count", "h_max_rad", "turning_sum_rad", "raw_rel_error_1", "debiased_rel_error_1"],
    ellipse_rows,
)
ellipse_raw_slope = fit_log_slope(ell_h[(end - 5):end], ell_raw[(end - 5):end])
ellipse_debiased_slope = fit_log_slope(ell_h[(end - 5):end], ell_deb[(end - 5):end])

# ------------------------------------------------------------------ 11. monostatic ray assembly
# Physically specified configuration: regular N-gon inscribed in a circular
# PEC cylinder of radius R = 1 m; unit-strength line source collocated with
# the observer (monostatic) on the circle of radius d = 3 m; k = 8 rad/m.
# Per visible seam: full 4-term PEC KP soft/hard coefficient -> even half-sum
# (Ds + Dh)/2; L_j = rho_j/2; u^i = exp(-i k rho)/sqrt(rho);
# cylindrical spreading 1/sqrt(rho) (KP NASA CR-2401 Eq. 47); return phase
# exp(-i k rho).
# Two references: the cell-matched arc I_N (union of the turning cells of the
# locally visible seams, isolates quadrature error) and the fixed physical
# visibility arc I_* = int_{-t*}^{t*} f with t* = acos(R/d) (contains the
# additional visibility/domain error I_N - I_*). Matched arcs use 2^17
# trapezoid subintervals; the fixed arc uses 2^20 (converged to ~1e-13).
const MA = Study.MonostaticAssembly
mono_R, mono_d, mono_k = 1.0, 3.0, 8.0
mono_Ns = [12, 16, 20, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
t_vis = MA.monostatic_visibility_half_angle(mono_R, mono_d)
I_star = MA.monostatic_arc_integral(
    mono_R, mono_d, 0.0, mono_k, -t_vis, t_vis; n_quad=1 << 20)
mono_rows = Vector{Vector{Any}}()
mono_err_raw = Float64[]
mono_err_deb = Float64[]
mono_err_raw_fixed = Float64[]
mono_err_deb_fixed = Float64[]
mono_err_domain = Float64[]
mono_hmax = Float64[]
for N in mono_Ns
    pts = MA.regular_polygon_points(N, mono_R)
    raw = MA.monostatic_even_field(pts, (mono_d, 0.0), mono_k)
    deb = MA.monostatic_even_field(pts, (mono_d, 0.0), mono_k; debias=true)
    I_arc = MA.monostatic_arc_integral(
        mono_R, mono_d, 0.0, mono_k, raw.arc_lo, raw.arc_hi; n_quad=1 << 17)
    rraw = abs(raw.field - I_arc) / abs(I_arc)
    rdeb = abs(deb.field - I_arc) / abs(I_arc)
    rdom = abs(I_arc - I_star) / abs(I_star)
    rraw_f = abs(raw.field - I_star) / abs(I_star)
    rdeb_f = abs(deb.field - I_star) / abs(I_star)
    push!(mono_err_raw, rraw)
    push!(mono_err_deb, rdeb)
    push!(mono_err_domain, rdom)
    push!(mono_err_raw_fixed, rraw_f)
    push!(mono_err_deb_fixed, rdeb_f)
    push!(mono_hmax, 2 * pi / N)
    push!(
        mono_rows,
        [
            N, raw.n_visible, raw.arc_lo, raw.arc_hi,
            real(raw.field), imag(raw.field),
            real(deb.field), imag(deb.field),
            real(I_arc), imag(I_arc),
            real(I_star), imag(I_star),
            rraw, rdeb, rdom, rraw_f, rdeb_f,
            raw.max_abs_edge, raw.max_abs_face,
        ],
    )
end
write_csv(
    joinpath(DATA, "monostatic_convergence.csv"),
    [
        "N_count", "n_visible_1", "arc_lo_rad", "arc_hi_rad",
        "raw_re_1", "raw_im_1", "debiased_re_1", "debiased_im_1",
        "matched_arc_re_1", "matched_arc_im_1",
        "fixed_arc_re_1", "fixed_arc_im_1",
        "raw_matched_rel_error_1", "debiased_matched_rel_error_1",
        "domain_rel_error_1",
        "raw_fixed_rel_error_1", "debiased_fixed_rel_error_1",
        "max_abs_edge_1", "max_abs_face_1",
    ],
    mono_rows,
)
mono_raw_slope = fit_log_slope(mono_hmax[(end - 5):end], mono_err_raw[(end - 5):end])
mono_deb_slope = fit_log_slope(mono_hmax[(end - 5):end], mono_err_deb[(end - 5):end])

# Per-seam complete soft/hard coefficients at N = 48, observer angle 0.
pts48 = MA.regular_polygon_points(48, mono_R)
seams48 = MA.monostatic_seam_geometry(pts48, (mono_d, 0.0))
seam_rows = Vector{Vector{Any}}()
for (j, s) in enumerate(seams48)
    if s.visible
        Ds, Dh = PAPER_BACKEND == "diffutd" ? DiffCurvatureStudy.seam_coefficients(s, mono_k) : pec_wedge_DsDh(
            UTDKernels.Wedge(s.alpha),
            UTDKernels.RayAngles(s.phi_kp, s.phi_kp), mono_k, s.rho / 2)
        push!(
            seam_rows,
            [
                j, s.theta, s.delta, s.rho, s.eta, s.phi_kp, 1,
                abs(Ds), abs(Dh), abs((Ds + Dh) / 2), abs((Dh - Ds) / 2),
            ],
        )
    else
        push!(seam_rows, [j, s.theta, s.delta, s.rho, s.eta, s.phi_kp, 0, NaN, NaN, NaN, NaN])
    end
end
write_csv(
    joinpath(DATA, "monostatic_seams.csv"),
    [
        "vertex_1", "theta_rad", "delta_rad", "rho_m", "eta_rad",
        "phi_kp_rad", "visible_1", "abs_Ds_1", "abs_Dh_1",
        "abs_De_even_1", "abs_Df_face_1",
    ],
    seam_rows,
)

# Monostatic pattern sweep at N = 96, plus the exact soft-cylinder
# eigenfunction-series scattered field normalized to the same incident
# cylindrical-wave convention at rho = d - R.
mono_N_pattern = 96
pts96 = MA.regular_polygon_points(mono_N_pattern, mono_R)
phi_sweep = collect(range(-pi, pi; length=361))
c_norm = MA.hankel_source_normalization(mono_k, mono_d - mono_R)
u_exact_soft = c_norm * MA.cylinder_line_source_scattered(
    mono_k, mono_R, mono_d, 0.0, mono_d, 0.0; bc=:soft)
u_exact_hard = c_norm * MA.cylinder_line_source_scattered(
    mono_k, mono_R, mono_d, 0.0, mono_d, 0.0; bc=:hard)
mono_pattern_rows = Vector{Vector{Any}}()
for p in phi_sweep
    obs = (mono_d * cos(p), mono_d * sin(p))
    raw = MA.monostatic_even_field(pts96, obs, mono_k)
    deb = MA.monostatic_even_field(pts96, obs, mono_k; debias=true)
    I_arc = MA.monostatic_arc_integral(
        mono_R, mono_d, p, mono_k, raw.arc_lo, raw.arc_hi; n_quad=1 << 17)
    push!(
        mono_pattern_rows,
        [
            p, raw.n_visible, abs(raw.field), abs(deb.field),
            abs(I_star), abs(I_arc), abs(u_exact_soft), abs(u_exact_hard),
        ],
    )
end
write_csv(
    joinpath(DATA, "monostatic_pattern.csv"),
    [
        "phi_rad", "n_visible_1", "raw_abs_1", "debiased_abs_1",
        "fixed_arc_abs_1", "matched_arc_abs_1",
        "exact_soft_abs_1", "exact_hard_abs_1",
    ],
    mono_pattern_rows,
)

# Diagnostic scale: limiting intrinsic-seam magnitude relative to the exact
# soft/hard-cylinder monostatic scattered field (same source convention).
# The fixed-arc integral I_star is aspect-independent by rotational symmetry
# of the smooth-cylinder configuration.
mono_artifact_soft = abs(I_star) / abs(u_exact_soft)
mono_artifact_hard = abs(I_star) / abs(u_exact_hard)
n48m = findfirst(==(48), mono_Ns)
mono_raw48 = mono_err_raw[n48m]
mono_deb48 = mono_err_deb[n48m]
n512m = findfirst(==(512), mono_Ns)
mono_deb_fixed512 = mono_err_deb_fixed[n512m]
mono_raw_fixed512 = mono_err_raw_fixed[n512m]
mono_domain512 = mono_err_domain[n512m]
mono_fixed_ratio = mono_deb_fixed512 / mono_err_deb[n512m]
mono_face_ratio = mono_rows[n48m][19] / mono_rows[n48m][18]
mono_vis_frac = mono_rows[end][2] / mono_Ns[end]

# geometry CSVs for king / ellipse domain panels
th_circ = collect(range(0, 2 * pi; length=600))
circ_rows = [[th, cos(th), sin(th)] for th in th_circ]
write_csv(joinpath(DATA, "king_circle.csv"), ["theta_rad", "x_m", "y_m"], circ_rows)

Ngeo = 14
tg = 2 * pi .* (0:(Ngeo - 1)) ./ Ngeo
poly_rows = Vector{Vector{Any}}()
for j in 1:Ngeo
    xg = cos(tg[j])
    yg = sin(tg[j])
    # every other vertex carries a turning-angle arrow (source experiment)
    arrow = (j - 1) % 2 == 0 ? 1.0 : 0.0
    push!(poly_rows, [j, xg, yg, 0.15 * xg * arrow, 0.15 * yg * arrow])
end
write_csv(
    joinpath(DATA, "king_polygon.csv"),
    ["vertex_1", "x_m", "y_m", "arrow_u_m", "arrow_v_m"],
    poly_rows,
)

tt_ell = collect(range(0, 2 * pi; length=800))
ell_curve_rows = [[ellipse_a * cos(t), ellipse_b * sin(t)] for t in tt_ell]
write_csv(joinpath(DATA, "ellipse_curve.csv"), ["x_m", "y_m"], ell_curve_rows)

t24, p24 = ellipse_points(24, ellipse_a, ellipse_b)
d24 = polygon_turning_angles(p24)
pick = sortperm(d24; rev=true)[1:4]
ell_poly_rows = Vector{Vector{Any}}()
for j in 1:24
    rnorm = hypot(p24[j, 1], p24[j, 2])
    is_pick = j in pick ? 1.0 : 0.0
    ux = is_pick * 0.22 * p24[j, 1] / rnorm
    uy = is_pick * 0.22 * p24[j, 2] / rnorm
    push!(ell_poly_rows, [j, p24[j, 1], p24[j, 2], ux, uy])
end
write_csv(
    joinpath(DATA, "ellipse_polygon.csv"),
    ["vertex_1", "x_m", "y_m", "arrow_u_m", "arrow_v_m"],
    ell_poly_rows,
)

# macros
n48_raw = raw_rel[findfirst(==(48), Ns)]
n48_deb = deb_rel[findfirst(==(48), Ns)]
z24 = 24.0
n_milli = n_for[(3, z24, 1e-3)]
n_micro = n_for[(3, z24, 1e-6)]
n_nano = n_for[(3, z24, 1e-9)]
n256 = findfirst(==(256), ellipse_Ns)
beta512 = local_bias_factor(2 * pi / 512)
max_turn_err = maximum(abs.(ell_turn .- 2 * pi))

macros = join(
    [
        "% Automatically generated from UTDKernels canonical curvature experiments",
        @sprintf("\\newcommand{\\BiasSlope}{%.3f}", bias_slope),
        @sprintf("\\newcommand{\\BetaFiveTwelve}{%.6f}", beta512),
        @sprintf("\\newcommand{\\RawSlope}{%.3f}", raw_slope),
        "\\newcommand{\\DecompResidual}{$(sci_tex(maximum(identity_res)))}",
        "\\newcommand{\\RawErrFortyEight}{$(sci_tex(n48_raw))}",
        "\\newcommand{\\DebiasErrFortyEight}{$(sci_tex(n48_deb))}",
        "\\newcommand{\\AliasCases}{600}",
        "\\newcommand{\\AliasMaxAbs}{$(sci_tex(maximum(alias_abs_errors)))}",
        "\\newcommand{\\AliasPNinetyFive}{$(sci_tex(quantile(alias_rel_errors, 0.95)))}",
        "\\newcommand{\\AliasMaxRel}{$(sci_tex(maximum(alias_rel_errors)))}",
        @sprintf("\\newcommand{\\FixedSlope}{%.3f}", fixed_slope),
        @sprintf("\\newcommand{\\GeomSlope}{%.3f}", geom_slope),
        "\\newcommand{\\PatternErrTwelve}{$(sci_tex(pattern_errors["12"]))}",
        "\\newcommand{\\PatternErrTwentyFour}{$(sci_tex(pattern_errors["24"]))}",
        "\\newcommand{\\PatternErrFortyEight}{$(sci_tex(pattern_errors["48"]))}",
        "\\newcommand{\\PatternRawErrTwelve}{$(sci_tex(pattern_errors_raw["12"]))}",
        "\\newcommand{\\PatternRawErrTwentyFour}{$(sci_tex(pattern_errors_raw["24"]))}",
        "\\newcommand{\\PatternRawErrFortyEight}{$(sci_tex(pattern_errors_raw["48"]))}",
        @sprintf("\\newcommand{\\NonuniformRawSlope}{%.3f}", nonuniform_raw_slope),
        @sprintf("\\newcommand{\\NonuniformDebiasSlope}{%.3f}", nonuniform_debiased_slope),
        "\\newcommand{\\NForMilli}{$(n_milli)}",
        "\\newcommand{\\NForMicro}{$(n_micro)}",
        "\\newcommand{\\NForNano}{$(n_nano)}",
        "\\newcommand{\\JointFixedMax}{$(sci_tex(rule_max[1]))}",
        "\\newcommand{\\JointModerateMax}{$(sci_tex(rule_max[2]))}",
        "\\newcommand{\\JointAdaptiveMax}{$(sci_tex(rule_max[3]))}",
        @sprintf("\\newcommand{\\EllipseRawSlope}{%.3f}", ellipse_raw_slope),
        @sprintf("\\newcommand{\\EllipseDebiasSlope}{%.3f}", ellipse_debiased_slope),
        "\\newcommand{\\EllipseRawTwoFiftySix}{$(sci_tex(ell_raw[n256]))}",
        "\\newcommand{\\EllipseDebiasTwoFiftySix}{$(sci_tex(ell_deb[n256]))}",
        "\\newcommand{\\EllipseTurnResidual}{$(sci_tex(max_turn_err))}",
        "\\newcommand{\\MonoRawErrFortyEight}{$(sci_tex(mono_raw48))}",
        "\\newcommand{\\MonoDebiasErrFortyEight}{$(sci_tex(mono_deb48))}",
        @sprintf("\\newcommand{\\MonoRawSlope}{%.3f}", mono_raw_slope),
        @sprintf("\\newcommand{\\MonoDebiasSlope}{%.3f}", mono_deb_slope),
        @sprintf("\\newcommand{\\MonoFaceRatio}{%.1f}", mono_face_ratio),
        @sprintf("\\newcommand{\\MonoVisibleFraction}{%.3f}", mono_vis_frac),
        "\\newcommand{\\MonoArtifactSoft}{$(sci_tex(mono_artifact_soft))}",
        "\\newcommand{\\MonoArtifactHard}{$(sci_tex(mono_artifact_hard))}",
        @sprintf("\\newcommand{\\MonoTStar}{%.4f}", t_vis),
        "\\newcommand{\\MonoFixedDebiasFiveTwelve}{$(sci_tex(mono_deb_fixed512))}",
        "\\newcommand{\\MonoFixedRawFiveTwelve}{$(sci_tex(mono_raw_fixed512))}",
        "\\newcommand{\\MonoDomainFiveTwelve}{$(sci_tex(mono_domain512))}",
        @sprintf("\\newcommand{\\MonoFixedRatio}{%.1f}", mono_fixed_ratio),
        "",
    ],
    "\n",
)
write(joinpath(DATA, "results_macros.tex"), macros)
write(joinpath(PAPER, "results_macros.tex"), macros)

println("wrote CSVs to ", DATA)
println("bias_slope=", bias_slope)
println("raw_slope=", raw_slope)
println("N48 raw/deb=", n48_raw, " / ", n48_deb)
println("geom_slope=", geom_slope)
println("alias max abs=", maximum(alias_abs_errors))
println("ellipse slopes=", ellipse_raw_slope, " / ", ellipse_debiased_slope)
