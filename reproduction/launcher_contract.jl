module CentralLauncherContract
export PAPERS,paper_launcher_sources,write_paper_launchers
const PAPERS=(160,162,165,168,170)

"""Canonical source of the thin paper launchers and their audit fixtures."""
function paper_launcher_sources(paper::Integer)
    paper in PAPERS || throw(ArgumentError("unsupported central paper: $paper"))
    central=replace(raw"""
# Active numerical implementations live in the two central checkouts.
const PAPER_ID = __PAPER__
const PAPER_DIRECTORY = normpath(joinpath(@__DIR__, ".."))
const CENTRAL_CODES = abspath(expanduser(get(ENV, "EMPIRE_CODES_ROOT",
    normpath(joinpath(@__DIR__, "..", "..", "..", "..", "..", "codes")))))
const CENTRAL_ENTRY = joinpath(CENTRAL_CODES, "UTDKernels.jl", "reproduction", "paper_entry.jl")
isfile(CENTRAL_ENTRY) || error("Central reproduction entry is missing: $CENTRAL_ENTRY; set EMPIRE_CODES_ROOT to the directory containing both checkouts")
include(CENTRAL_ENTRY)
using .CentralPaperEntry: central_paper_main
""","__PAPER__"=>string(paper))
    aliases=Dict("reproduce.jl"=>"","plotting/plot_paper.jl"=>"plot.jl",
        (paper==170 ? "data_generation/generate_data.jl" : "data_generation/generate_paper_data.jl")=>"generate.jl")
    paper==168 && (aliases["plotting/plot_wedge.jl"]="plot.jl")
    if paper==160
        merge!(aliases,Dict("sensitivity_sweep.jl"=>"sensitivity.jl",
            "evaluator_validation.jl"=>"evaluator.jl","cost_accuracy.jl"=>"cost.jl",
            "evidence_macros.jl"=>"macros.jl"))
    end
    sources=Dict("central.jl"=>central)
    for (relative,stage) in aliases
        parent_count=length(split(relative,'/'))-1
        include_line=relative=="reproduce.jl" ? "include(\"central.jl\")" :
            "include(joinpath(@__DIR__, "*join(repr.([fill("..",parent_count);"central.jl"]),", ")*"))"
        options=isempty(stage) ? "" : ";stage="*repr(stage)
        sources[relative]=include_line*"\ncentral_paper_main(PAPER_ID,PAPER_DIRECTORY,CENTRAL_CODES"*options*")\n"
    end
    return sources
end

function write_paper_launchers(project::AbstractString,paper::Integer)
    project=isdir(project) ? realpath(project) : abspath(project)
    sources=paper_launcher_sources(paper)
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
        ispath(path) || continue
        isfile(path) && read(path,String)==source || error(
            "archive the existing noncanonical implementation before generating launchers: $path")
    end
    paths=String[]
    for (relative,source) in sources
        path=joinpath(project,"paper","scripts",split(relative,'/')...)
        if !isfile(path)
            mkpath(dirname(path));write(path,source)
        end
        push!(paths,path)
    end
    return sort!(paths)
end
end
