if !isdefined(@__MODULE__, :PaperReproductionPaths)
    include(joinpath(@__DIR__, "..", "..", "common.jl"))
end
using .PaperReproductionPaths: output_directory, selected_backend

# Cost-versus-accuracy comparison for the grazing soft coefficient (R6):
# for each evaluation route -- direct subtraction, the four-term baseline,
# the O(h) linear limit, a direct/linear stable hybrid, fixed GL8
# continuation, and the public adaptive continuation -- this script records
# the relative error against the independent high-precision reference, the
# wall time, the number of kernel-derivative (G') evaluations, and the
# accepted quadrature order. Writes paper/data/cost_accuracy.csv and the
# adaptive-order statistics used in Table II.
#
# Run with:  julia --project=paper paper/scripts/cost_accuracy.jl

using UTDKernels
using Statistics
using Random

include(joinpath(@__DIR__, "hp_oracle.jl"))

const DATA = joinpath(output_directory(160), "data")
const SEED = 20260816
mkpath(DATA)

# Minimum wall time over `reps` evaluations, in nanoseconds, after one
# untimed warmup call. Canonical-sweep timings use reps=200; the accepted-
# population statistics below use reps=5 per draw. Reported table values are
# medians of these per-offset minima across the sweep.
function bench(f; reps::Int=200)
    f()  # warmup / compile
    best = Inf
    for _ in 1:reps
        t = @elapsed f()
        best = min(best, t)
    end
    return best * 1e9
end

# Stable hybrid baseline: the four-term/direction pair for h above the
# empirical crossover where the linear limit reaches 1e-6 relative accuracy,
# and the linear limit below it.
function hybrid_soft(g::Geom, h::Real; h_switch::Real=1e-3)
    return h <= h_switch ? linear_soft(g, h) : fourterm_soft(g, h)
end

const CANON = Geom(1.5 * pi, 1.7, 100.0, 1.0)
w = _wedge(CANON)

# ------------------------------------------------- canonical h-sweep table
hs = exp10.(range(-16, -1; length=31))
rows = Vector{Vector{Any}}()
for h in hs
    ref = soft_hp(CANON, h; dps=60)
    n = wedge_n(w)
    C = prefactor(CANON)
    # Method list: name => (value, gprime_calls, kernel_equiv_calls, timing fn)
    methods = (
        "direct" => let
            val = direct_soft(CANON, h)
            t = bench(() -> direct_soft(CANON, h))
            (val, 0, 2, t)
        end,
        "four_term" => let
            val = fourterm_soft(CANON, h)
            t = bench(() -> fourterm_soft(CANON, h))
            (val, 0, 4, t)
        end,
        "linear" => let
            val = linear_soft(CANON, h)
            t = bench(() -> linear_soft(CANON, h))
            (val, 1, 0, t)
        end,
        "hybrid" => let
            val = hybrid_soft(CANON, h)
            t = bench(() -> hybrid_soft(CANON, h))
            (val, h <= 1e-3 ? 1 : 0, h <= 1e-3 ? 0 : 4, t)
        end,
        "gl8" => let
            val = fixed_order_soft(CANON, h; order=8)
            t = bench(() -> fixed_order_soft(CANON, h; order=8))
            (val, 8, 0, t)
        end,
    )
    for (name, entry) in methods
        val, gcalls, kcalls, tns = entry
        push!(rows, Any[name, h, rel(val, ref), tns, gcalls, kcalls, 0, true])
    end
    # Adaptive continuation through the public info API (traceable order and
    # derivative-call count).
    ds, dh, info = Production.pec_wedge_DsDh_grazing_info(
        w, RayAngles(CANON.phi, h), CANON.k, CANON.L)
    t = bench(() -> Production.pec_wedge_DsDh_grazing(w, RayAngles(CANON.phi, h), CANON.k, CANON.L))
    push!(rows, Any["adaptive", h, rel(ds, ref), t,
        info.derivative_calls, 0, info.accepted_order, info.converged])
end

# Accuracy/cost tradeoff across requested tolerances: the adaptive evaluator
# is run with quadrature_rtol from 1e-4 down to its 1e-11 default, recording
# the accepted order, timing, and measured error. On the canonical small
# offset the integrand is trivially resolved and the knob is flat; on the
# certified but rapidly varying interval (the evaluator-validation
# under-resolved case) the same knob moves the accepted order from 16 to 32
# and the derivative-call count from 36 to 92. The hybrid has no such knob:
# below its switching offset its accuracy is the linear rule's O(h^2)
# truncation.
for (h, g_sw) in ((1e-12, CANON), (0.3, Geom(1.5 * pi, 2.7278481012658227, 100.0, 1.0)))
    ref = soft_hp(g_sw, h; dps=60)
    w_sw = _wedge(g_sw)
    for rtol in (1e-4, 1e-6, 1e-9, 1e-11)
        ds, dh, info = Production.pec_wedge_DsDh_grazing_info(
            w_sw, RayAngles(g_sw.phi, h), g_sw.k, g_sw.L;
            quadrature_rtol=rtol)
        t = bench(() -> Production.pec_wedge_DsDh_grazing(
            w_sw, RayAngles(g_sw.phi, h), g_sw.k, g_sw.L;
            quadrature_rtol=rtol))
        push!(rows, Any["adaptive_rtol_$(rtol)", h, rel(ds, ref), t,
            info.derivative_calls, 0, info.accepted_order, info.converged])
    end
end

open(joinpath(DATA, "cost_accuracy.csv"), "w") do io
    println(io, "# generator=paper/scripts/cost_accuracy.jl")
    println(io, "# method=direct/four_term/linear/hybrid/gl8/adaptive routes; adaptive_rtol_* rows vary quadrature_rtol at a small offset (flat) and on a rapidly varying interval (order moves 16->32)")
    println(io, "# reference=independent BigFloat four-term pairing (hp_oracle.jl)")
    println(io, "# geometry=alpha=1.5pi,phi=1.7,k=100,L=1 (canonical)")
    println(io, "# benchmark_protocol=one warmup call, then minimum of 200 timed evaluations per canonical offset; adaptive_rtol rows same protocol; population timings use minima of 5")
    println(io, "# spatial_work=DiffUTD backend additionally constructs the 3D ray frame, validates retained angular inputs, transports Cartesian polarization dyads, and projects coefficients; kernels backend evaluates scalar coefficients")
    println(io, "# timed_work=adaptive returns (Ds,Dh) plus the interval certificate and adaptive order selection; direct/four_term/linear/hybrid/gl8 wrappers return Ds only; fixed-order GL rules come from the cached node table (no per-call node generation)")
    println(io, "# environment=julia $(VERSION); machine=$(Sys.MACHINE); times are implementation measurements on this setup")
    println(io, "# utdkernels_version=", pkgversion(UTDKernels))
    println(io, "# backend=", selected_backend())
    println(io, join([
        "method", "h_rad", "rel_error", "time_ns", "gprime_calls",
        "kernel_calls", "accepted_order", "converged",
    ], ","))
    for r in rows
        println(io, join(r, ","))
    end
end

# --------------------------------------- adaptive-order statistics ensemble
# Accepted-order and derivative-call statistics of the adaptive evaluator on
# a fresh accepted-population draw (same acceptance criteria as the finite-L
# ensemble; separate seed so it is a documented cost sample, not the error
# ensemble itself).
rng = Xoshiro(SEED + 21)
n_stats = 300
orders = Int[]
calls = Int[]
times = Float64[]
attempts = 0
while length(orders) < n_stats
    global attempts += 1
    attempts > n_stats * 200 && error("failed to draw enough admissible cases")
    s, status, _ = sample_geometry(rng)
    status === :accepted || continue
    g, h, tm, xm = s
    ds, dh, info = Production.pec_wedge_DsDh_grazing_info(
        _wedge(g), RayAngles(g.phi, h), g.k, g.L;
        transition_margin=tm, x_margin=xm, allow_interior=true)
    info.converged || continue
    push!(orders, info.accepted_order)
    push!(calls, info.derivative_calls)
    push!(times, bench(() -> Production.pec_wedge_DsDh_grazing(
        _wedge(g), RayAngles(g.phi, h), g.k, g.L;
        transition_margin=tm, x_margin=xm, allow_interior=true); reps=5))
end

open(joinpath(DATA, "adaptive_order_stats.csv"), "w") do io
    println(io, "# generator=paper/scripts/cost_accuracy.jl")
    println(io, "# method=pec_wedge_DsDh_grazing_info on accepted-population draws")
    println(io, "# acceptance=finite-L margins transition_margin=0.02, x_margin=0.001, |G'|>=1e-4 max(1,|G|)")
    println(io, "# utdkernels_version=", pkgversion(UTDKernels))
    println(io, "# backend=", selected_backend())
    println(io, "case,accepted_order,derivative_calls,time_ns")
    for i in 1:n_stats
        println(io, join([i - 1, orders[i], calls[i], times[i]], ","))
    end
end

stats = Dict(
    "order_min" => minimum(orders), "order_med" => median(orders),
    "order_max" => maximum(orders), "order_p95" => quantile(orders, 0.95),
    "calls_med" => median(calls), "calls_max" => maximum(calls),
    "time_med_ns" => median(times), "time_p95_ns" => quantile(times, 0.95),
    "cases" => n_stats, "attempts" => attempts,
)
println("adaptive order stats: ", stats)
println("cost table rows: ", length(rows))
