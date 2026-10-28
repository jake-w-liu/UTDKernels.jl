#!/usr/bin/env julia

using Dates
using TOML
using SHA
using UTDKernels
using DiffUTD

include("common.jl")
using .PaperReproductionPaths
include("artifacts.jl")
using .PaperArtifacts
include("harness_receipts.jl")
using .HarnessReceipts: runtime_snapshot

const DATA_STAGES = Dict(
    160 => ["generate.jl", "evaluator.jl", "cost.jl", "sensitivity.jl", "spatial_validation.jl", "macros.jl"],
    162 => ["generate.jl"],
    165 => ["generate.jl"],
    168 => ["generate.jl"],
    170 => ["generate.jl"],
)

struct ReproductionStageError <: Exception
    stage::String
    exit_code::Int
    signal::Int
    log::String
end
Base.showerror(stream::IO,error::ReproductionStageError)=print(stream,
    "reproduction stage ",error.stage," failed (exit ",error.exit_code,
    ", signal ",error.signal,"); see ",error.log)

function options(arguments)
    paper = "all"
    backend = "kernels"
    output = nothing
    plots = true
    stage = nothing
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        if argument in ("--paper", "--backend", "--output", "--stage")
            index < length(arguments) || throw(ArgumentError("missing value after $argument"))
            value = arguments[index + 1]
            argument == "--paper" && (paper = value)
            argument == "--backend" && (backend = value)
            argument == "--output" && (output = abspath(expanduser(value)))
            argument == "--stage" && (stage = value)
            index += 2
        elseif argument == "--no-plots"
            plots = false
            index += 1
        else
            throw(ArgumentError("unknown argument: $argument"))
        end
    end
    backend in ("kernels", "diffutd") || throw(ArgumentError("invalid backend: $backend"))
    papers = paper == "all" ? collect(SUPPORTED_PAPERS) : [parse(Int, paper)]
    all(id -> id in SUPPORTED_PAPERS, papers) || throw(ArgumentError("unsupported paper: $paper"))
    if stage !== nothing
        length(papers)==1 || throw(ArgumentError("--stage requires one paper"))
        stage in vcat(DATA_STAGES[only(papers)],["plot.jl"]) ||
            throw(ArgumentError("unsupported stage for paper $(only(papers)): $stage"))
    end
    return (; papers, backend, output, plots, stage)
end

function write_receipt(path, receipt)
    mkpath(dirname(path))
    mktemp(dirname(path)) do temporary, stream
        TOML.print(stream, receipt)
        close(stream)
        mv(temporary, path; force=true)
    end
end

function stamp_runtime_macros(directory, backend)
    backend in ("kernels", "diffutd") || throw(ArgumentError("invalid backend: $backend"))
    paths = [joinpath(directory, "data", "results_macros.tex"),
             joinpath(directory, "results_macros.tex")]
    contents = read.(paths, String)
    contents[1] == contents[2] || error("generated manuscript macro copies differ before runtime stamping")
    metadata = (
        "ExperimentJuliaVersion" => string(VERSION),
        "ExperimentUTDKernelsVersion" => string(pkgversion(UTDKernels)),
        "ExperimentDiffUTDVersion" => string(pkgversion(DiffUTD)),
        "ExperimentBackend" => backend,
        "ExperimentThreads" => string(Threads.nthreads()),
    )
    for (name, _) in metadata
        occursin("\\"*name, contents[1]) && error("generated macros already define runtime metadata: $name")
    end
    suffix = "\n" * join(("\\newcommand{\\"*name*"}{"*value*"}" for (name,value) in metadata), "\n") * "\n"
    for path in paths
        write(path, contents[1]*suffix)
    end
    return nothing
end

function reproduce(paper, configuration)
    destination = if configuration.output === nothing
        joinpath(@__DIR__, "results", configuration.backend, string(paper), "paper")
    elseif length(configuration.papers) == 1
        configuration.output
    else
        joinpath(configuration.output, string(paper), "paper")
    end
    destination = abspath(destination)
    stages = configuration.stage === nothing ? copy(DATA_STAGES[paper]) : [configuration.stage]
    configuration.stage===nothing && configuration.plots && push!(stages, "plot.jl")
    source_directory = joinpath(@__DIR__, "papers", string(paper))
    if configuration.backend == "diffutd"
        adapter = joinpath(source_directory, "backend_diffutd.jl")
        isfile(adapter) || error("missing DiffUTD reproduction adapter: $paper/backend_diffutd.jl")
    end
    for stage in stages
        isfile(joinpath(source_directory, stage)) || error("missing reproduction stage: $paper/$stage")
    end
    before = source_snapshot(UTDKernels, DiffUTD)
    mkpath(destination)
    staging=mktempdir(dirname(destination);prefix=".paper$(paper)-$(configuration.backend)-",cleanup=false)
    # A requested single stage may consume existing data. Full runs start empty
    # so stale outputs cannot satisfy the complete-inventory gate.
    if configuration.stage!==nothing && isdir(joinpath(destination,"data"))
        cp(joinpath(destination,"data"),joinpath(staging,"data"))
    end
    logs = joinpath(destination, "reproduction-logs")
    mkpath(logs)
    receipt_name=configuration.stage===nothing ? "receipt.toml" : "receipt-$(configuration.stage).toml"
    receipt_path = joinpath(logs, receipt_name)
    receipt = Dict{String,Any}(
        "paper" => paper, "backend" => configuration.backend,
        "julia_version" => string(VERSION), "started_utc" => string(now(UTC)),
        "runtime" => runtime_snapshot(),
        "output" => destination, "status" => "running", "sources" => before,
        "scope" => configuration.stage===nothing ? "complete" : "single_stage",
        "staging" => staging,
        "stages" => Dict{String,Any}[],
    )
    write_receipt(receipt_path, receipt)
    try
        for stage in stages
            script = joinpath(source_directory, stage)
            log = joinpath(staging,"reproduction-logs",stage * ".log")
            mkpath(dirname(log))
            command = `$(Base.julia_cmd()) --startup-file=no --threads=$(Threads.nthreads()) --project=$(@__DIR__) $script`
            started = time()
            open(log, "w") do stream
                # This CLI runs stages sequentially. Inherit the normal process
                # environment without embedding its values in Cmd diagnostics.
                process=withenv("UTD_PAPER_OUTPUT"=>staging,
                                "UTD_PAPER_BACKEND"=>configuration.backend,
                                "JULIA_LOAD_PATH"=>"@:@stdlib") do
                    run(pipeline(ignorestatus(command);stdout=stream,stderr=stream))
                end
                success(process) || throw(ReproductionStageError(stage,Int(process.exitcode),Int(process.termsignal),log))
            end
            push!(receipt["stages"], Dict(
                "script" => stage, "exit_code" => 0,
                "seconds" => time() - started, "log" => log,
                "log_sha256" => bytes2hex(sha256(read(log))),
            ))
            source_snapshot(UTDKernels, DiffUTD) == before || error(
                "central source changed during reproduction; rerun on a stable checkout",
            )
            write_receipt(receipt_path, receipt)
            println("paper $paper / $(configuration.backend) / $stage: passed")
        end
        if configuration.stage===nothing
            stamp_runtime_macros(staging, configuration.backend)
            receipt["artifacts"]=validate_artifacts(paper,staging;plots=configuration.plots)
        end
        publish_artifacts(staging,destination)
        if configuration.stage===nothing
            validate_artifacts(paper,destination;plots=configuration.plots)==receipt["artifacts"] ||
                error("published artifacts differ from validated staging outputs")
        end
        source_snapshot(UTDKernels,DiffUTD)==before || error("central source changed while publishing reproduction outputs")
        for record in receipt["stages"]
            record["log"]=joinpath(logs,basename(record["log"]))
            bytes2hex(sha256(read(record["log"])))==record["log_sha256"] ||
                error("published reproduction log differs from its completed stage")
        end
        receipt["status"] = "passed"
        receipt["finished_utc"] = string(now(UTC))
        write_receipt(receipt_path, receipt)
        rm(staging;recursive=true)
    catch exception
        receipt["status"] = "failed"
        receipt["error"] = sprint(showerror, exception)
        receipt["finished_utc"] = string(now(UTC))
        write_receipt(receipt_path, receipt)
        rethrow()
    end
    return receipt
end

function main(arguments=ARGS)
    configuration = options(arguments)
    for paper in configuration.papers
        reproduce(paper, configuration)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
