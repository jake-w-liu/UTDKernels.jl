module CentralLauncherContract
export STUDIES,study_launcher_sources,write_study_launchers
const STUDIES=("face-grazing","face-edge","curvature-measure","null-uniform","shadow-sensitivity")

"""Canonical source of the thin study launchers."""
function study_launcher_sources(study::AbstractString)
    study in STUDIES || throw(ArgumentError("unsupported central study: $study"))
    central=replace(raw"""
# Active numerical implementations live in the two central checkouts.
const STUDY_SLUG = __STUDY__
const STUDY_DIRECTORY = normpath(joinpath(@__DIR__, ".."))
const CENTRAL_CODES = abspath(expanduser(get(ENV, "UTD_REPRO_CODES_ROOT",
    normpath(joinpath(@__DIR__, "..", "..", "..", "..", "..", "codes")))))
const CENTRAL_ENTRY = joinpath(CENTRAL_CODES, "UTDKernels.jl", "reproduction", "study_entry.jl")
isfile(CENTRAL_ENTRY) || error("Central reproduction entry is missing: $CENTRAL_ENTRY; set UTD_REPRO_CODES_ROOT to the directory containing both checkouts")
include(CENTRAL_ENTRY)
using .CentralStudyEntry: central_study_main
""","__STUDY__"=>repr(String(study)))
    aliases=Dict("reproduce.jl"=>"","plotting/plot_paper.jl"=>"plot.jl",
        (study=="shadow-sensitivity" ? "data_generation/generate_data.jl" : "data_generation/generate_paper_data.jl")=>"generate.jl")
    if study=="face-grazing"
        merge!(aliases,Dict("sensitivity_sweep.jl"=>"sensitivity.jl",
            "evaluator_validation.jl"=>"evaluator.jl","cost_accuracy.jl"=>"cost.jl",
            "evidence_macros.jl"=>"macros.jl"))
    end
    sources=Dict{String,Any}("central.jl"=>central)
    for (relative,stage) in aliases
        parent_count=length(split(relative,'/'))-1
        include_line=relative=="reproduce.jl" ? "include(\"central.jl\")" :
            "include(joinpath(@__DIR__, "*join(repr.([fill("..",parent_count);"central.jl"]),", ")*"))"
        options=isempty(stage) ? "" : ";stage="*repr(stage)
        call="central_study_main(STUDY_SLUG,STUDY_DIRECTORY,CENTRAL_CODES"*options*")"
        # Plot implementations are project-owned: launchers that reach the
        # plot.jl stage point run.jl at the sibling scripts under plotting/.
        if stage=="plot.jl"
            call="withenv(\"UTD_REPRO_PLOT_SCRIPT\"=>joinpath(@__DIR__,\"plot.jl\")) do\n    "*call*"\nend"
        elseif isempty(stage)
            call="withenv(\"UTD_REPRO_PLOT_SCRIPT\"=>joinpath(@__DIR__,\"plotting\",\"plot.jl\")) do\n    "*call*"\nend"
        end
        sources[relative]=include_line*"\n"*call*"\n"
    end
    # Project-owned scripts the inventory requires but whose contents are the
    # project's own implementation, not generated canonical launchers.
    sources["plotting/plot.jl"]=nothing
    study=="null-uniform" && (sources["plotting/plot_wedge.jl"]=nothing)
    return sources
end

function write_study_launchers(project::AbstractString,study::AbstractString)
    project=isdir(project) ? realpath(project) : abspath(project)
    sources=study_launcher_sources(study)
    # Preflight the entire set before creating any target. Retired numerical
    # implementations must be archived explicitly by the migration owner.
    for (relative,source) in sources
        directory=project
        for part in vcat(["paper","scripts"],split(relative,'/')[1:end-1])
            directory=joinpath(directory,part)
            islink(directory) && error("archive the existing linked launcher directory before generation: $directory")
            ispath(directory) && !isdir(directory) && error("launcher parent is not a directory: $directory")
        end
        path=joinpath(project,"paper","scripts",split(relative,'/')...)
        islink(path) && error("archive the existing linked launcher before generation: $path")
        source===nothing && continue
        ispath(path) || continue
        isfile(path) && read(path,String)==source || error(
            "archive the existing noncanonical implementation before generating launchers: $path")
    end
    paths=String[]
    for (relative,source) in sources
        source===nothing && continue
        path=joinpath(project,"paper","scripts",split(relative,'/')...)
        if !isfile(path)
            mkpath(dirname(path));write(path,source)
        end
        push!(paths,path)
    end
    return sort!(paths)
end
end
