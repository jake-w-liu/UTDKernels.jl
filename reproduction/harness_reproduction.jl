module HarnessReproduction
using SHA,TOML
export validate_reproduction_receipt,required_stages
const DATA_STAGES=Dict(160=>["generate.jl","evaluator.jl","cost.jl","sensitivity.jl","spatial_validation.jl","macros.jl"],
    162=>["generate.jl"],165=>["generate.jl"],168=>["generate.jl"],170=>["generate.jl"])
required_stages(paper)=vcat(DATA_STAGES[paper],["plot.jl"])
digest(path)=bytes2hex(sha256(read(path)))

"""Validate a complete, source-bound, plotted run; partial receipts cannot qualify."""
function validate_reproduction_receipt(path,paper,backend; sources,runtime,artifact_validator)
    isfile(path) || error("complete reproduction receipt is missing: $path")
    receipt=TOML.parsefile(path)
    get(receipt,"status","")=="passed" || error("reproduction did not pass: $path")
    get(receipt,"scope","")=="complete" || error("single-stage reproduction is not a complete audit")
    get(receipt,"paper",0)==paper && get(receipt,"backend","")==backend || error("wrong reproduction identity")
    get(receipt,"julia_version","")==runtime["version"] || error("reproduction Julia version differs")
    get(receipt,"runtime",nothing)==runtime || error("reproduction executable/build/options differ")
    get(receipt,"sources",nothing)==sources || error("reproduction source/dependency fingerprint is stale")
    output=realpath(receipt["output"])
    "archived" in splitpath(output) && error("archived outputs cannot satisfy an active audit")
    realpath(dirname(path))==realpath(joinpath(output,"reproduction-logs")) || error("receipt is detached from its output directory")
    stages=receipt["stages"]
    [record["script"] for record in stages]==required_stages(paper) || error("reproduction does not include all data and plotting stages")
    for record in stages
        record["exit_code"]==0 || error("a reproduction stage failed")
        log=record["log"]
        isfile(log) && realpath(dirname(log))==realpath(dirname(path)) || error("stage log is missing or detached")
        get(record,"log_sha256","")==digest(log) || error("stage log changed or lacks provenance")
    end
    get(receipt,"artifacts",nothing)==artifact_validator(paper,output;plots=true) ||
        error("complete data/figure/macro inventory differs from the completed run")
    return receipt
end
end
