#!/usr/bin/env julia
module CentralDevelopmentHarness
using Dates,Pkg,SHA,TOML
include("harness_entries.jl")
include("harness_receipts.jl")
include("harness_reproduction.jl")
include("paper_entry.jl")
include("common.jl")
include("artifacts.jl")
using .HarnessEntries,.HarnessReceipts,.HarnessReproduction

function options(arguments)
    project=nothing;codes=nothing;reuse_only=false;receipt_directory=nothing;outputs=nothing
    index=1
    while index<=length(arguments)
        argument=arguments[index]
        if argument in ("--project","--codes","--receipts","--outputs")
            index<length(arguments) || throw(ArgumentError("missing value after $argument"))
            value=abspath(expanduser(arguments[index+1]))
            argument=="--project" && (project=value)
            argument=="--codes" && (codes=value)
            argument=="--receipts" && (receipt_directory=value)
            argument=="--outputs" && (outputs=value)
            index+=2
        elseif argument=="--reuse-only"
            reuse_only=true;index+=1
        else
            throw(ArgumentError("unknown audit argument: $argument"))
        end
    end
    project===nothing && throw(ArgumentError("--project requires an active paper project directory"))
    codes===nothing && (codes=normpath(joinpath(@__DIR__,"..","..")))
    receipt_directory===nothing && (receipt_directory=joinpath(@__DIR__,"results","development-harness"))
    return (;project,codes,reuse_only,receipt_directory,outputs)
end

function validate_source_bindings(project,codes)
    environment=Pkg.Types.EnvCache(joinpath(project,"Project.toml"))
    manifest=environment.manifest_file
    isfile(manifest) || error("resolved central Manifest is missing: $manifest")
    dependencies=TOML.parsefile(manifest)["deps"]
    for (name,uuid) in HarnessReceipts.OWNER_UUIDS
        entries=get(dependencies,name,Any[])
        length(entries)==1 || error("Manifest omits or duplicates $name: $manifest")
        entry=only(entries)
        get(entry,"uuid","")==uuid || error("wrong central UUID: $name")
        path=get(entry,"path",nothing)
        path isa String || error("$name must resolve to its central source checkout")
        realpath(joinpath(dirname(manifest),path))==HarnessReceipts.canonical_root(codes,name) ||
            error("Manifest binds archived/noncentral $name")
    end
    return manifest
end

function candidate_receipts(directory)
    isdir(directory) || return String[]
    files=String[]
    for (folder,dirs,names) in walkdir(directory)
        filter!(name->!islink(joinpath(folder,name)),dirs)
        "receipt.toml" in names && push!(files,joinpath(folder,"receipt.toml"))
    end
    return sort(files;by=mtime,rev=true)
end

function find_test_receipt(directory,codes,owner,runtime,dependencies,rejections)
    for path in candidate_receipts(directory)
        try
            record=read_toml(path)
            get(record,"owner","")==owner || continue
            validate_test_receipt(record,codes,owner;runtime,dependencies)
            return path
        catch exception
            exception isa InterruptException && rethrow()
            push!(rejections,Dict("receipt"=>path,"reason"=>first(sprint(showerror,exception),800)))
        end
    end
    return nothing
end

function with_audit_lock(function_,path;timeout=21600)
    mkpath(dirname(path));started=time();acquired=false
    while !acquired
        try
            mkdir(path);acquired=true
        catch exception
            exception isa InterruptException && rethrow()
            isdir(path) || rethrow()
            time()-started<timeout || error("audit lock remains owned by another process: $path; no process was signalled")
            sleep(1)
        end
    end
    try
        write_toml(joinpath(path,"owner.toml"),Dict("pid"=>getpid(),"started_utc"=>string(now(UTC))))
        return function_()
    finally
        # Only this invocation's successfully acquired lock is removed.
        rm(path;recursive=true)
    end
end

function ensure_full_tests(configuration,runtime,dependencies)
    directory=joinpath(configuration.receipt_directory,"tests")
    records=Dict{String,Any}()
    for owner in ("UTDKernels","DiffUTD")
        rejections=Dict{String,String}[]
        path=find_test_receipt(directory,configuration.codes,owner,runtime,dependencies,rejections)
        reused=path!==nothing
        if path===nothing
            if configuration.reuse_only
                reasons=join([entry["receipt"]*": "*entry["reason"] for entry in Iterators.take(rejections,3)],"; ")
                error("no exact completed full $owner test receipt; rejected $(length(rejections)) candidates. $reasons")
            end
            # The shared lock prevents five active papers from launching the
            # same central full test concurrently. Recheck after acquiring it.
            path=with_audit_lock(joinpath(directory,owner*"-julia"*string(VERSION)*".lock")) do
                found=find_test_receipt(directory,configuration.codes,owner,runtime,dependencies,rejections)
                if found===nothing
                    run_full_package_test(owner,configuration.codes;output=directory,threads=4)
                else
                    reused=true
                    found
                end
            end
        end
        record=read_toml(path)
        validate_test_receipt(record,configuration.codes,owner;runtime,dependencies)
        records[owner]=Dict("receipt"=>path,"receipt_sha256"=>HarnessReceipts.digest(path),
            "reused"=>reused,"log"=>record["log"],"log_sha256"=>record["log_sha256"],
            "rejected_candidates"=>rejections)
        println("[PASS] $owner complete package tests (",reused ? "verified receipt reuse" : "fresh run",")")
    end
    return records
end

function main(arguments=ARGS)
    configuration=options(arguments)
    VERSION>=v"1.12" || error("central audit requires Julia1.12 or newer")
    Threads.nthreads()==4 && Int(Base.JLOptions().check_bounds)==1 ||
        error("launch the audit with --threads=4 --check-bounds=yes")
    entries=validate_paper_entries(configuration.project,configuration.codes)
    paper=entries["paper"];paper_directory=joinpath(configuration.project,"paper")
    reproduction=realpath(@__DIR__)
    realpath(dirname(Base.active_project()))==reproduction || error("use --project=$reproduction")
    try
        validate_source_bindings(paper_directory,configuration.codes)
        validate_source_bindings(reproduction,configuration.codes)
    catch exception
        exception isa InterruptException && rethrow()
        configuration.reuse_only && rethrow()
        CentralPaperEntry.bind_central_sources(paper_directory,configuration.codes)
        CentralPaperEntry.bind_central_sources(reproduction,configuration.codes)
        validate_source_bindings(paper_directory,configuration.codes)
        validate_source_bindings(reproduction,configuration.codes)
    end
    # A first-time Pkg.develop may write legitimate central [sources] entries.
    # Freeze the validated post-bootstrap contract, not its pre-setup hash.
    entries=validate_paper_entries(configuration.project,configuration.codes)
    environment=environment_snapshot();runtime=runtime_snapshot()
    # Paper timing evidence uses one thread and normal bounds policy; full
    # package tests independently require four threads and forced checks.
    experiment_runtime=runtime_snapshot(;threads=1,check_bounds=0)
    # Load only after validating the selected source bindings. Keep the module
    # objects local so no newly introduced global binding crosses world ages.
    kernels=Base.require(Base.PkgId(Base.UUID(HarnessReceipts.OWNER_UUIDS["UTDKernels"]),"UTDKernels"))
    spatial=Base.require(Base.PkgId(Base.UUID(HarnessReceipts.OWNER_UUIDS["DiffUTD"]),"DiffUTD"))
    sources=PaperReproductionPaths.source_snapshot(kernels,spatial)
    mkpath(configuration.receipt_directory)
    audit_directory=mktempdir(configuration.receipt_directory;prefix="paper$paper-",cleanup=false)
    audit_path=joinpath(audit_directory,"audit.toml")
    audit=Dict{String,Any}("status"=>"running","scope"=>"active-central-full-development-harness",
        "started_utc"=>string(now(UTC)),"paper"=>paper,"entries"=>entries,"runtime"=>runtime,
        "environment"=>environment,"sources"=>sources,"experiment_runtime"=>experiment_runtime)
    write_toml(audit_path,audit)
    try
        audit["package_tests"]=ensure_full_tests(configuration,runtime,environment["dependencies"])
        reproductions=Dict{String,Any}()
        for backend in ("kernels","diffutd")
            output=configuration.outputs===nothing ? (backend=="kernels" ? paper_directory :
                joinpath(paper_directory,"validation","diffutd")) : joinpath(configuration.outputs,backend,string(paper),"paper")
            path=joinpath(output,"reproduction-logs","receipt.toml")
            validator()=validate_reproduction_receipt(path,paper,backend;sources,runtime=experiment_runtime,
                artifact_validator=PaperArtifacts.validate_artifacts)
            reused=true;rejected=""
            try
                validator()
            catch exception
                exception isa InterruptException && rethrow()
                rejected=first(sprint(showerror,exception),1600)
                configuration.reuse_only && rethrow()
                reused=false
                command=`$(Base.julia_cmd()) --startup-file=no --check-bounds=auto --threads=1 --project=$reproduction $(joinpath(reproduction,"run.jl")) --paper $paper --backend $backend --output $output`
                log=joinpath(audit_directory,backend*"-reproduction.log")
                process=open(log,"w") do stream
                    withenv("JULIA_LOAD_PATH"=>"@:@stdlib") do
                        run(pipeline(ignorestatus(command);stdout=stream,stderr=stream))
                    end
                end
                success(process) || error("paper$paper/$backend reproduction failed (exit $(process.exitcode), signal $(process.termsignal)); see $log")
                validator()
            end
            reproductions[backend]=Dict("receipt"=>path,"receipt_sha256"=>HarnessReceipts.digest(path),
                "output"=>output,"reused"=>reused,"prior_rejection"=>rejected)
            println("[PASS] paper $paper / $backend complete numerical and figure reproduction (",reused ? "verified receipt reuse" : "fresh run",")")
        end
        audit["reproductions"]=reproductions
        validate_paper_entries(configuration.project,configuration.codes)==entries || error("active entry sources changed during audit")
        environment_snapshot()==environment || error("resolved dependencies changed during audit")
        PaperReproductionPaths.source_snapshot(kernels,spatial)==sources || error("central implementation changed during audit")
        audit["status"]="passed"
    catch exception
        audit["status"]="failed";audit["error"]=first(sprint(showerror,exception),4000)
        rethrow()
    finally
        audit["finished_utc"]=string(now(UTC));write_toml(audit_path,audit)
        println("Audit receipt: ",audit_path)
    end
    return audit_path
end
end
if abspath(PROGRAM_FILE)==abspath(@__FILE__)
    CentralDevelopmentHarness.main()
end
