if !isdefined(@__MODULE__, :PaperReproductionPaths)
    include(joinpath(@__DIR__, "..", "..", "common.jl"))
end
using .PaperReproductionPaths: output_directory, selected_backend

# Derives the LaTeX macros for the evaluator-validation, cost, and
# sensitivity evidence (R5/R6) from the CSVs produced by
# evaluator_validation.jl, cost_accuracy.jl, and sensitivity_sweep.jl.
# Writes paper/data/results_macros_extra.tex, which main.tex inputs after
# results_macros.tex.
#
# Run with:  julia --project=paper paper/scripts/evidence_macros.jl

using Statistics
using Printf

const DATA = joinpath(output_directory(160), "data")

function read_csv_rows(path)
    lines = [l for l in readlines(path) if !startswith(l, "#") && !isempty(l)]
    header = split(lines[1], ",")
    rows = [split(l, ",") for l in lines[2:end]]
    return header, rows
end

# IEEE-style mantissa-exponent string for macros.
function sci(x::Float64; sig::Int=3)
    (x == 0 || !isfinite(x)) && return string(x)
    e = floor(Int, log10(abs(x)))
    m = x / 10.0^e
    m = round(m; sigdigits=sig)
    if abs(m) >= 10
        m /= 10
        e += 1
    end
    mantissa = rstrip(rstrip(@sprintf("%.6f", m), '0'), '.')
    e == 0 && return mantissa
    return "\\mathrm{$(mantissa)}\\times10^{$e}"
end

emacros = Pair{String,String}[]

# ------------------------------------------------------ evaluator_validation.csv
ev_h, ev_rows = read_csv_rows(joinpath(DATA, "evaluator_validation.csv"))
check_idx = findfirst(==("check"), ev_h)
agree_idx = findfirst(==("agreement"), ev_h)
agree_ind_idx = findfirst(==("agreement_independent"), ev_h)
order_idx = findfirst(==("accepted_order"), ev_h)
out_idx = findfirst(==("outcome"), ev_h)
desc_idx = findfirst(==("description"), ev_h)

sub = name -> [r for r in ev_rows if r[check_idx] == name]
agree_ind = r -> (isempty(r[agree_ind_idx]) ? NaN : parse(Float64, r[agree_ind_idx]))
nchecks = length(ev_rows)
npass = count(r -> r[out_idx] == "pass", ev_rows)
npass == nchecks ||
    error("evaluator validation contains failing checks; see evaluator_validation.csv")
push!(emacros, "EvalChecks" => string(nchecks))

aa = sub("adaptive_accuracy")
push!(emacros, "EvalAdaptiveMaxErr" =>
    sci(maximum(parse(Float64, r[agree_idx]) for r in aa)))
push!(emacros, "EvalAdaptiveMaxErrIndependent" =>
    sci(maximum(agree_ind(r) for r in aa)))
push!(emacros, "EvalAdaptiveMaxOrder" =>
    string(maximum(parse(Int, r[order_idx]) for r in aa)))

hz = sub("h_zero_limit")
push!(emacros, "EvalHardLimitErr" =>
    sci(maximum(parse(Float64, r[agree_idx]) for r in hz)))

mr = sub("mirror_face")
push!(emacros, "EvalMirrorSoftErr" => sci(parse(Float64, mr[1][agree_idx])))
push!(emacros, "EvalMirrorHardErr" => sci(parse(Float64, mr[2][agree_idx])))
mq = sub("mirror_face_quantization")[1]
push!(emacros, "EvalMirrorQuantErr" => sci(parse(Float64, mq[agree_idx])))

it = sub("fprime_method_switch")[1]
push!(emacros, "EvalSwitchErr" => sci(parse(Float64, it[agree_idx])))
push!(emacros, "EvalSwitchErrIndependent" => sci(agree_ind(it)))
md = match(r"off-node by ([0-9.]+)", it[desc_idx])
md === nothing && error("could not parse off-node spacing")
push!(emacros, "EvalSwitchSpacing" => md[1])
tp = sub("transition_pole_rejection")[1]
mdp = match(r"off-node by ([0-9.]+)", tp[desc_idx])
mdp === nothing && error("could not parse pole-rejection spacing")
push!(emacros, "EvalPoleSpacing" => mdp[1])

us = sub("underresolved_start")[1]
push!(emacros, "EvalUnderStartErr" => sci(parse(Float64, us[agree_idx])))
ur = sub("underresolved_interval")[1]
push!(emacros, "EvalUnderFinalErr" => sci(parse(Float64, ur[agree_idx])))
push!(emacros, "EvalUnderFinalErrIndependent" => sci(agree_ind(ur)))
push!(emacros, "EvalUnderOrder" => ur[order_idx])

# ------------------------------------------------------------ cost_accuracy.csv
ca_h, ca_rows = read_csv_rows(joinpath(DATA, "cost_accuracy.csv"))
m_idx = findfirst(==("method"), ca_h)
h_idx = findfirst(==("h_rad"), ca_h)
e_idx = findfirst(==("rel_error"), ca_h)
t_idx = findfirst(==("time_ns"), ca_h)

methods = ("direct", "four_term", "linear", "hybrid", "gl8", "adaptive")
names = Dict(
    "direct" => "CostDirect", "four_term" => "CostFourTerm",
    "linear" => "CostLinear", "hybrid" => "CostHybrid",
    "gl8" => "CostGLEight", "adaptive" => "CostAdaptive",
)
for m in methods
    rs = [r for r in ca_rows if r[m_idx] == m]
    tmed = median(parse(Float64, r[t_idx]) for r in rs)
    r12 = rs[argmin(abs.(parse.(Float64, getindex.(rs, h_idx)) .- 1e-12))]
    parse(Float64, r12[h_idx]) == 1e-12 || error("no h=1e-12 row for $m")
    push!(emacros, names[m] * "Ns" => string(round(Int, tmed)))
    push!(emacros, names[m] * "Err" => sci(parse(Float64, r12[e_idx])))
end
hrows = [r for r in ca_rows if r[m_idx] == "hybrid"]
push!(emacros, "CostHybridMaxErr" =>
    sci(maximum(parse(Float64, r[e_idx]) for r in hrows)))
t_ad = median(parse(Float64, r[t_idx]) for r in ca_rows if r[m_idx] == "adaptive")
t_di = median(parse(Float64, r[t_idx]) for r in ca_rows if r[m_idx] == "direct")
push!(emacros, "CostAdaptiveDirectRatio" =>
    string(round(t_ad / t_di; digits=1)))

# ------------------------------------------------------ adaptive_order_stats.csv
st_h, st_rows = read_csv_rows(joinpath(DATA, "adaptive_order_stats.csv"))
o_idx = findfirst(==("accepted_order"), st_h)
c_idx = findfirst(==("derivative_calls"), st_h)
t2_idx = findfirst(==("time_ns"), st_h)
orders = Int.(parse.(Float64, getindex.(st_rows, o_idx)))
calls = Int.(parse.(Float64, getindex.(st_rows, c_idx)))
stimes = parse.(Float64, getindex.(st_rows, t2_idx))
push!(emacros, "AdaptiveStatCases" => string(length(orders)))
push!(emacros, "AdaptiveOrderMin" => string(minimum(orders)))
push!(emacros, "AdaptiveOrderMedian" => string(Int(median(orders))))
push!(emacros, "AdaptiveOrderMax" => string(maximum(orders)))
push!(emacros, "AdaptiveCallsMedian" => string(Int(median(calls))))
push!(emacros, "AdaptiveCallsMax" => string(maximum(calls)))
push!(emacros, "AdaptiveTimeUs" => string(round(median(stimes) / 1e3; digits=1)))

# --------------------------------------------------------- sensitivity_sweep.csv
se_h, se_rows = read_csv_rows(joinpath(DATA, "sensitivity_sweep.csv"))
he_idx = findfirst(==("h_rad"), se_h)
nd_idx = findfirst(==("direct_Ds_over_h_rel_error"), se_h)
ns_idx = findfirst(==("stable_Ds_over_h_rel_error"), se_h)
ad_idx = findfirst(==("ad_continuation_rel_error"), se_h)
ad4_idx = findfirst(==("ad_fourterm_rel_error"), se_h)
adl_idx = findfirst(==("ad_linear_rel_error"), se_h)
adh_idx = findfirst(==("ad_hybrid_rel_error"), se_h)
fd_idx = findfirst(==("fd4_sens_rel_error"), se_h)
hvals = parse.(Float64, getindex.(se_rows, he_idx))
i16 = findfirst(==(1e-16), hvals)
i12 = findfirst(==(1e-12), hvals)
i3 = findfirst(==(1e-3), hvals)
ip1 = findfirst(==(0.1), hvals)
push!(emacros, "SensNormDirectTiny" => sci(parse(Float64, se_rows[i16][nd_idx])))
push!(emacros, "SensNormDirectMax" =>
    sci(maximum(parse(Float64, r[nd_idx]) for r in se_rows)))
push!(emacros, "SensNormStableMax" =>
    sci(maximum(parse(Float64, r[ns_idx]) for r in se_rows)))
push!(emacros, "SensADMax" =>
    sci(maximum(parse(Float64, r[ad_idx]) for r in se_rows)))
push!(emacros, "SensADFourTermTiny" => sci(parse(Float64, se_rows[i16][ad4_idx])))
push!(emacros, "SensADFourTermMed" => sci(parse(Float64, se_rows[i12][ad4_idx])))
push!(emacros, "SensADLinearTiny" => sci(parse(Float64, se_rows[i16][adl_idx])))
push!(emacros, "SensADLinearMed" => sci(parse(Float64, se_rows[i12][adl_idx])))
push!(emacros, "SensADLinearLarge" => sci(parse(Float64, se_rows[ip1][adl_idx])))
push!(emacros, "SensADLinearSwitch" => sci(parse(Float64, se_rows[i3][adl_idx])))
push!(emacros, "SensADHybridTiny" => sci(parse(Float64, se_rows[i16][adh_idx])))
push!(emacros, "SensADHybridSwitch" => sci(parse(Float64, se_rows[i3][adh_idx])))
push!(emacros, "SensADContinuationSwitch" => sci(parse(Float64, se_rows[i3][ad_idx])))
push!(emacros, "SensFDTiny" => sci(parse(Float64, se_rows[i16][fd_idx])))
push!(emacros, "SensFDMax" =>
    sci(maximum(parse(Float64, r[fd_idx]) for r in se_rows)))

path = joinpath(DATA, "results_macros_extra.tex")
open(path, "w") do io
    for (name, val) in emacros
        println(io, "\\newcommand{\\", name, "}{", val, "}")
    end
end
cp(path, joinpath(DATA, "..", "results_macros_extra.tex"); force=true)
println("wrote ", length(emacros), " extra macros")
