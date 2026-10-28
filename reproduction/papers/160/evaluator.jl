if !isdefined(@__MODULE__, :PaperReproductionPaths)
    include(joinpath(@__DIR__, "..", "..", "common.jl"))
end
using .PaperReproductionPaths: output_directory, selected_backend

# Validates the public adaptive evaluator
# UTDKernels.pec_wedge_DsDh_grazing / _info on the checks required for the
# grazing-continuation study, and writes paper/data/evaluator_validation.csv.
# Each row records the check, the measured complex value, the comparison
# reference, the achieved agreement, and the evaluator diagnostics
# (accepted order, G' call count, convergence flag). Where an independent
# erfc-based coefficient reference is also scored, it is archived in the
# reference_independent / agreement_independent columns.
#
# Checks: (1) adaptive accuracy and accepted order on representative
# finite-distance cases, against an independent Gauss-Kronrod (quadgk)
# quadrature of the same public derivative AND against the independent
# erfc-based coefficient reference; (2) h = 0 soft/hard limits and
# mirrored-face consistency with matched local offsets, plus a separate
# input-quantization case; (3) face exit, nearest-integer crossing, an
# internal transition-method switch strictly between quadrature nodes
# (acceptance test), and a genuine off-node transition-zero/cotangent-pole
# rejection test; (4) a smooth but rapidly varying interval where the
# starting order is insufficient, archiving both the starting and final
# refined values; (5) domain-guard and convergence-guard failure behavior.
#
# Run with:  julia --project=paper paper/scripts/evaluator_validation.jl

using UTDKernels
using QuadGK: quadgk
using UTDKernels: pec_wedge_DsDh_grazing_info, two_term_kernel,
    two_term_kernel_derivative, pec_wedge_prefactor, wedge_n,
    grazing_interval_report, grazing_local_angles

const HERE = @__DIR__
const DATA = joinpath(output_directory(160), "data")
mkpath(DATA)

include(joinpath(HERE, "hp_oracle.jl"))

# Independent QUADRATURE check of the same public derivative: adaptive
# Gauss-Kronrod on [-1,1] of the publicly exported two_term_kernel_derivative.
# This shares no code with the Gauss-Legendre refinement loop, so agreement
# certifies the quadrature -- but it evaluates the SAME binary64 integrand,
# so it cannot detect a bias common to every G' evaluation. Error against
# the integrand itself is measured separately, against the independent
# erfc-based coefficient reference soft_hp (a different F implementation,
# not the production scaled-erfc path).
function gk_reference_soft(wedge, phi, h, k, L; rtol=1e-13)
    n = wedge_n(wedge)
    C = pec_wedge_prefactor(k, n)
    val, err = quadgk(xi -> two_term_kernel_derivative(phi + h * xi, wedge, k, L),
        -1.0, 1.0; rtol=rtol, maxevals=10^6)
    return -C * h * val, -C * h * err
end

relerr(a, b) = abs(a - b) / max(abs(b), 1e-300)

rows = Vector{Vector{Any}}()
function emit(check, desc, measured, reference, agreement, info, outcome;
              ref_independent="", agree_independent="")
    orders = info === nothing ? "" : join(info.orders, ";")
    push!(rows, Any[
        check, desc,
        sprint(show, measured), sprint(show, reference), agreement,
        ref_independent, agree_independent,
        info === nothing ? 0 : info.accepted_order,
        info === nothing ? 0 : info.derivative_calls,
        info === nothing ? "" : string(info.converged),
        orders, outcome,
    ])
end

const W = Wedge(1.5pi)
const PHI = 1.7
const K = 100.0
const L = 1.0
const CANON_GEOM = Geom(1.5 * pi, 1.7, 100.0, 1.0)

# --- (1) adaptive accuracy and accepted order on representative cases --------
# Scored twice: quadrature agreement vs the independent GK quadrature of the
# same public integrand, and integrand-level agreement vs the independent
# erfc-based coefficient reference.
for h in (1e-2, 1e-6, 1e-10, 1e-14)
    ds, dh, info = Production.pec_wedge_DsDh_grazing_info(W, RayAngles(PHI, h), K, L)
    ref, qerr = gk_reference_soft(W, PHI, h, K, L)
    ref_hp = soft_hp(Geom(1.5 * pi, PHI, K, L), h; dps=60)
    e = relerr(ds, ref)
    e_hp = relerr(ds, ref_hp)
    emit("adaptive_accuracy", "canonical finite-L case, h=$h",
        ds, ref, e, info, e < 1e-10 ? "pass" : "FAIL";
        ref_independent=ref_hp, agree_independent=e_hp)
end

# --- (2) h = 0 limits and mirrored-face consistency --------------------------
dsz, dhz, infoz = Production.pec_wedge_DsDh_grazing_info(W, RayAngles(PHI, 0.0), K, L)
dsf, dhf = Production.pec_wedge_DsDh(W, RayAngles(PHI, 0.0), K, L)
emit("h_zero_limit", "soft limit Ds(0)", dsz, 0.0, abs(dsz), infoz,
    dsz == 0 && infoz.converged && isempty(infoz.orders) ? "pass" : "FAIL")
emit("h_zero_limit", "hard limit Dh(0) vs four-term", dhz, dhf, relerr(dhz, dhf),
    infoz, relerr(dhz, dhf) < 1e-14 ? "pass" : "FAIL")

# Mirror test with MATCHED local offset. The n-face route recovers its local
# offset as h_rec = alpha - phip in binary64, which differs from the nominal
# 1e-8 request (see mirror_face_quantization below). Feeding the same
# recovered h_rec to the o-face route makes the two evaluations see identical
# local coordinates, so they must agree to roundoff, not to the input spacing.
hrep = 1e-8
hrec = W.alpha - (W.alpha - hrep)
ang_n = RayAngles(PHI, W.alpha - hrep)
ang_m = RayAngles(W.alpha - PHI, hrec)
dsn, dhn, infon = Production.pec_wedge_DsDh_grazing_info(W, ang_n, K, L; face=:n)
dsm, dhm, infom = Production.pec_wedge_DsDh_grazing_info(W, ang_m, K, L; face=:o)
emit("mirror_face", "n-face vs mirrored o-face soft (matched local h)", dsn, dsm,
    relerr(dsn, dsm), infon, relerr(dsn, dsm) < 1e-14 ? "pass" : "FAIL")
emit("mirror_face", "n-face vs mirrored o-face hard (matched local h)", dhn, dhm,
    relerr(dhn, dhm), infon, relerr(dhn, dhm) < 1e-14 ? "pass" : "FAIL")

# Input-quantization case: the binary64 request phip = alpha - 1e-8 yields the
# local offset h_rec = alpha - (alpha - 1e-8) = 9.99999993922529e-9, which
# differs from 1e-8 by relerr(h_rec, 1e-8) ~ 6.07747e-9. Comparing the n-face
# route against an o-face evaluation at the LITERAL offset 1e-8 therefore
# recovers that input-quantization disagreement, not a quadrature tolerance.
dsq, dhq, infoq = Production.pec_wedge_DsDh_grazing_info(
    W, RayAngles(W.alpha - PHI, hrep), K, L; face=:o)
q_in = relerr(hrec, hrep)
q_out = relerr(dsn, dsq)
emit("mirror_face_quantization",
    "n-face at phip=alpha-1e-8 vs o-face at literal h=1e-8 (input quantization $(q_in))",
    dsn, dsq, q_out, infon,
    0.5 < q_out / q_in < 2.0 ? "pass" : "FAIL")

# Representable-input floor: W.alpha - 1e-16 rounds back to W.alpha, so the
# n-face offset is absorbed by the input representation; the evaluator
# correctly reports h = 0 rather than manufacturing a nonrepresentable offset.
loc_floor = grazing_local_angles(W, RayAngles(PHI, W.alpha - 1e-16); face=:n)
emit("mirror_face", "n-face offset representability floor (h rounds to 0)",
    loc_floor.h, 0.0, abs(loc_floor.h), nothing,
    loc_floor.face === :n && loc_floor.h == 0 ? "pass" : "FAIL")

# --- (3) face exit, nearest-integer crossing, internal switch, pole rejection
# Face exit: interval leaves the exterior angular region.
ang_bad = RayAngles(PHI, PHI + 0.5)
try
    Production.pec_wedge_DsDh_grazing(W, ang_bad, K, L)
    emit("face_exit", "interval exit refused", "returned", "GrazingDomainError",
        NaN, nothing, "FAIL")
catch err
    emit("face_exit", "interval exit refused", sprint(showerror, err),
        "GrazingDomainError", NaN, nothing,
        err isa GrazingDomainError ? "pass" : "FAIL")
end

# Nearest-integer crossing: for n = 1.5 the sigma=+1 KP boundary sits at
# beta = pi/2 (where (beta + pi)/(3pi) = 1/2); an interval bracketing it
# changes the nearest-integer branch mid-interval.
hX = 1e-2
ang_x = RayAngles(pi / 2, hX)   # [pi/2 - h, pi/2 + h] crosses beta = pi/2
try
    Production.pec_wedge_DsDh_grazing(W, ang_x, K, L)
    emit("integer_crossing", "N-boundary crossing refused", "returned",
        "GrazingDomainError", NaN, nothing, "FAIL")
catch err
    emit("integer_crossing", "N-boundary crossing refused",
        sprint(showerror, err), "GrazingDomainError", NaN, nothing,
        err isa GrazingDomainError ? "pass" : "FAIL")
end

# Off-node transition-zero / cotangent-pole REJECTION test. On the canonical
# n = 3/2 geometry, the sigma=-1 branch has u_{-1}(pi) = -pi and
# psi_{-1}(pi) = 0, so beta = pi is both a transition-argument zero and a
# cotangent pole. The nearest half-integer KP boundaries are pi/2 away, so a
# small interval around beta = pi crosses no nearest-integer boundary. The
# event is placed at the interval centre, which sits strictly between the two
# central Gauss-Legendre nodes, and the endpoints and sampled nodes are
# verified to pass the chosen margins: only the analytic whole-interval
# lattice test can catch it.
n_w = wedge_n(W)
h_tp = 3e-3
phi_tp = pi                  # the sigma=-1 event is the interval centre
nodes16, _ = UTDKernels._cached_gauss_legendre_rule(16)
# Verify: endpoints and GL16 nodes pass small pole/transition-argument margins
tm_tp, xm_tp = 1e-5, 1e-8
betas_node = sort(phi_tp .+ h_tp .* nodes16)
push!(betas_node, phi_tp - h_tp, phi_tp + h_tp)
min_sin_nodes = minimum(betas_node) do b
    min(abs(sin((pi + b) / (2n_w))), abs(sin((pi - b) / (2n_w))))
end
min_x_nodes = minimum(betas_node) do b
    amin = Inf
    for sigma in (+1, -1)
        N = UTDKernels.kp_Nj(b, sigma, n_w)
        a = UTDKernels.kp_aj(b, N, n_w)
        amin = min(amin, K * L * a)
    end
    return amin
end
node_dist_tp = minimum(abs.(betas_node .- pi))
(min_sin_nodes > tm_tp && min_x_nodes > xm_tp) || error(
    "off-node rejection setup failed: node/endpoint margins " *
    "min|sin psi|=$min_sin_nodes (need >$tm_tp), min x=$min_x_nodes (need >$xm_tp)",
)
rr_tp = grazing_interval_report(W, RayAngles(phi_tp, h_tp), K, L;
    face=:o, transition_margin=tm_tp, x_margin=xm_tp)
try
    Production.pec_wedge_DsDh_grazing(W, RayAngles(phi_tp, h_tp), K, L;
        face=:o, transition_margin=tm_tp, x_margin=xm_tp)
    emit("transition_pole_rejection",
        "beta=pi sigma=-1 zero-transition/cotangent pole off-node by " *
        "$(round(node_dist_tp / h_tp; digits=3)) of the interval half-width " *
        "(nodes pass margins; whole-interval guard must reject)",
        "returned", "GrazingDomainError", NaN, nothing, "FAIL")
catch err
    emit("transition_pole_rejection",
        "beta=pi sigma=-1 zero-transition/cotangent pole off-node by " *
        "$(round(node_dist_tp / h_tp; digits=3)) of the interval half-width " *
        "(nodes pass margins; report reason: $(rr_tp.reason))",
        sprint(showerror, err), "GrazingDomainError", NaN, nothing,
        err isa GrazingDomainError && !rr_tp.valid ? "pass" : "FAIL")
end

# Internal transition-method switch strictly between quadrature nodes: beta*
# solves kL a_{-1}(beta*) = 35 inside [phi - h, phi + h]; production G' calls
# F_utd_prime, which switches from the differential equation to the inverse-
# power series at x = 35. On beta in (2.0, 2.35) with n = 1.5 the sigma=-1
# branch has N = 0 and a = 1 + cos(beta) sweeps through x = 35 near
# beta ~ 2.28, while the sigma=+1 branch sits on N = 1 throughout, so the
# certified interval stays branch-constant. This is an ACCEPTANCE test of
# smooth passage through the implementation switch, scored against both the
# independent GK quadrature and the independent erfc-based reference.
xterm(b) = K * L * UTDKernels.kp_aj(b, UTDKernels.kp_Nj(b, -1, n_w), n_w)
function find_switch(lo, hi)
    (xterm(lo) - 35) * (xterm(hi) - 35) < 0 ||
        error("no internal switch bracket on [$lo, $hi]")
    for _ in 1:100
        mid = 0.5 * (lo + hi)
        if (xterm(mid) - 35) * (xterm(hi) - 35) < 0
            lo = mid
        else
            hi = mid
        end
    end
    return 0.5 * (lo + hi)
end
bstar = find_switch(2.0, 2.35)
h_sw = 3e-3
phi_sw = bstar + 0.35h_sw
ds_sw, dh_sw, info_sw = Production.pec_wedge_DsDh_grazing_info(
    W, RayAngles(phi_sw, h_sw), K, L)
ref_sw, _ = gk_reference_soft(W, phi_sw, h_sw, K, L)
ref_sw_hp = soft_hp(Geom(1.5 * pi, phi_sw, K, L), h_sw; dps=60)
nodes, _ = UTDKernels._cached_gauss_legendre_rule(info_sw.accepted_order)
node_beta = sort(phi_sw .+ h_sw .* nodes)
node_dist = minimum(abs.(node_beta .- bstar))
emit("fprime_method_switch",
    "x=35 F' implementation-method switch inside interval, off-node by " *
    "$(round(node_dist / h_sw; digits=3)) of the interval half-width",
    ds_sw, ref_sw, relerr(ds_sw, ref_sw), info_sw,
    relerr(ds_sw, ref_sw) < 1e-10 ? "pass" : "FAIL";
    ref_independent=ref_sw_hp, agree_independent=relerr(ds_sw, ref_sw_hp))

# --- (4) smooth but rapidly varying interval: starting order insufficient ----
# Search a deterministic candidate grid for a certified interval that the low
# starting rule under-resolves; the public evaluator is then called with that
# starting order and must refine to an accurate converged estimate. BOTH the
# starting GL4 value and the final refined value are archived, each scored
# against the independent GK quadrature and the erfc-based reference.
found_fast = false
nodes4, weights4 = UTDKernels._cached_gauss_legendre_rule(4)
for cand_phi in range(0.5, 4.5; length=80)
    for cand_h in (0.3, 0.2, 0.1)
        rr = grazing_interval_report(W, RayAngles(cand_phi, cand_h), K, L;
            face=:o, allow_interior=true)
        rr.valid || continue
        ds4 = -pec_wedge_prefactor(K, n_w) * cand_h * sum(
            weights4[i] * two_term_kernel_derivative(cand_phi + cand_h * nodes4[i], W, K, L)
            for i in eachindex(nodes4))
        refa, _ = gk_reference_soft(W, cand_phi, cand_h, K, L)
        refa_hp = soft_hp(Geom(1.5 * pi, cand_phi, K, L), cand_h; dps=60)
        e4 = relerr(ds4, refa)
        e4 > 1e-3 || continue
        dsa, dha, infoa = Production.pec_wedge_DsDh_grazing_info(
            W, RayAngles(cand_phi, cand_h), K, L; order=4)
        ea = relerr(dsa, refa)
        emit("underresolved_start",
            "phi=$cand_phi h=$cand_h: GL4 starting-rule value before refinement",
            ds4, refa, e4, infoa, "pass";
            ref_independent=refa_hp, agree_independent=relerr(ds4, refa_hp))
        emit("underresolved_interval",
            "phi=$cand_phi h=$cand_h: final refined value at order $(infoa.accepted_order)",
            dsa, refa, ea, infoa,
            ea < 1e-9 && infoa.accepted_order > 4 ? "pass" : "FAIL";
            ref_independent=refa_hp, agree_independent=relerr(dsa, refa_hp))
        global found_fast = true
        break
    end
    found_fast && break
end
found_fast || emit("underresolved_interval", "no under-resolved case found",
    "", "", NaN, nothing, "FAIL")

# --- (5) domain guard and convergence guard ----------------------------------
# Domain guard with on_fail=:four_term: returns the baseline and reports
# converged=false.
dsfb, dhfb, infofb = Production.pec_wedge_DsDh_grazing_info(
    W, ang_bad, K, L; on_fail=:four_term)
dsb, dhb = Production.pec_wedge_DsDh(W, ang_bad, K, L)
emit("domain_fallback", "on_fail=:four_term returns baseline",
    dsfb, dsb, relerr(dsfb, dsb), infofb,
    dsfb == dsb && dhfb == dhb && !infofb.converged ? "pass" : "FAIL")

# Convergence guard: an unreachable tolerance must raise rather than return
# an unconverged estimate.
try
    Production.pec_wedge_DsDh_grazing(W, RayAngles(PHI, 1e-8), K, L;
        quadrature_rtol=1e-17, quadrature_atol=0.0, max_order=8)
    emit("convergence_guard", "unreachable tolerance refused", "returned",
        "GrazingDomainError", NaN, nothing, "FAIL")
catch err
    emit("convergence_guard", "unreachable tolerance refused",
        sprint(showerror, err), "GrazingDomainError", NaN, nothing,
        err isa GrazingDomainError ? "pass" : "FAIL")
end
# ... and with on_fail=:four_term it falls back rather than returning a
# possibly unconverged estimate.
dsnc, dhnc, infonc = Production.pec_wedge_DsDh_grazing_info(
    W, RayAngles(PHI, 1e-8), K, L;
    quadrature_rtol=1e-17, quadrature_atol=0.0, max_order=8,
    on_fail=:four_term)
dsb2, dhb2 = Production.pec_wedge_DsDh(W, RayAngles(PHI, 1e-8), K, L)
emit("convergence_fallback", "unreachable tolerance falls back to baseline",
    dsnc, dsb2, relerr(dsnc, dsb2), infonc,
    dsnc == dsb2 && !infonc.converged ? "pass" : "FAIL")

# ---------------------------------------------------------------- write CSV
open(joinpath(DATA, "evaluator_validation.csv"), "w") do io
    println(io, "# generator=paper/scripts/evaluator_validation.jl")
    println(io, "# method=public Production.pec_wedge_DsDh_grazing(_info) adaptive evaluator")
    println(io, "# reference=independent QuadGK Gauss-Kronrod quadrature of public two_term_kernel_derivative (quadrature check; shares the integrand); four-term baseline for limits")
    println(io, "# reference_independent=independent erfc-based BigFloat coefficient reference (integrand-level check)")
    println(io, "# utdkernels_version=", pkgversion(UTDKernels))
    println(io, "# backend=", selected_backend())
    println(io, join([
        "check", "description", "measured", "reference", "agreement",
        "reference_independent", "agreement_independent",
        "accepted_order", "derivative_calls", "converged", "orders", "outcome",
    ], ","))
    for r in rows
        println(io, join(map(x -> replace(string(x), "," => ";"), r), ","))
    end
end

nfail = count(r -> r[end] == "FAIL", rows)
println("evaluator validation: ", length(rows) - nfail, "/", length(rows), " checks passed")
nfail == 0 || error("evaluator validation failed ($nfail failing checks)")
