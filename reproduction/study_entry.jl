module CentralStudyEntry

using Pkg
include("launcher_contract.jl")
using .CentralLauncherContract: study_launcher_sources,write_study_launchers

export central_study_main
export study_launcher_sources,write_study_launchers

function bind_central_sources(project::AbstractString,codes::AbstractString)
    roots=[joinpath(codes,"UTDKernels.jl"),joinpath(codes,"DiffUTD.jl")]
    for root in roots
        isfile(joinpath(root,"Project.toml")) || error("missing central package checkout: $root")
    end
    isfile(joinpath(project,"Project.toml")) || error("missing study environment: $project")
    cd(project) do
        Pkg.activate(".")
        Pkg.develop([Pkg.PackageSpec(path=relpath(root,project)) for root in roots])
        Pkg.instantiate()
    end
    resolved=Pkg.dependencies()
    for (name,identifier) in (("UTDKernels",Base.UUID("970fd46b-db34-494d-92d6-28101d6e9662")),
                              ("DiffUTD",Base.UUID("70df04e7-1766-4225-9ef1-e80c4be476cd")))
        haskey(resolved,identifier) || error("central dependency is missing: $name")
        realpath(resolved[identifier].source)==realpath(joinpath(codes,name*".jl")) ||
            error("$name did not resolve to its central checkout")
    end
    return nothing
end

"""Run a named study through both central packages, optionally selecting one stage."""
function central_study_main(study::AbstractString,study_directory::AbstractString,
                            codes_root::AbstractString;arguments=ARGS,stage=nothing)
    study in CentralLauncherContract.STUDIES || throw(ArgumentError("unsupported study: $study"))
    VERSION>=v"1.12" || error("study reproduction requires Julia 1.12 or newer")
    backend="both";plots=true;instantiate_only=false
    index=1
    while index<=length(arguments)
        argument=arguments[index]
        if argument=="--backend"
            index<length(arguments) || throw(ArgumentError("missing value after --backend"))
            backend=arguments[index+1];index+=2
        elseif argument=="--no-plots"
            plots=false;index+=1
        elseif argument=="--instantiate-only"
            instantiate_only=true;index+=1
        else
            throw(ArgumentError("unknown study reproduction argument: $argument"))
        end
    end
    backend in ("both","kernels","diffutd") || throw(ArgumentError("backend must be both, kernels or diffutd"))
    codes=realpath(expanduser(codes_root))
    study_directory=abspath(study_directory)
    reproduction=joinpath(codes,"UTDKernels.jl","reproduction")
    bind_central_sources(study_directory,codes)
    bind_central_sources(reproduction,codes)
    instantiate_only && return nothing
    for selected in (backend=="both" ? ("kernels","diffutd") : (backend,))
        output=selected=="kernels" ? study_directory : joinpath(study_directory,"validation","diffutd")
        flags=String["--study",string(study),"--backend",selected,"--output",output]
        plots || push!(flags,"--no-plots")
        stage===nothing || append!(flags,["--stage",stage])
        command=`$(Base.julia_cmd()) --startup-file=no --threads=$(Threads.nthreads()) --project=$reproduction $(joinpath(reproduction,"run.jl")) $flags`
        process=withenv("JULIA_LOAD_PATH"=>"@:@stdlib") do
            run(ignorestatus(command))
        end
        success(process) || error("study $study / $selected failed (exit $(process.exitcode), signal $(process.termsignal)); inspect $output/reproduction-logs")
    end
    return nothing
end

end
