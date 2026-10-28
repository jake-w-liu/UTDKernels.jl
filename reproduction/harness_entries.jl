module HarnessEntries
using SHA,TOML
include("launcher_contract.jl")
using .CentralLauncherContract
export validate_paper_entries

const UUIDS=Dict("UTDKernels"=>"970fd46b-db34-494d-92d6-28101d6e9662",
                 "DiffUTD"=>"70df04e7-1766-4225-9ef1-e80c4be476cd")

_strip(node)=node
function _strip(node::Expr)
    args=[_strip(arg) for arg in node.args if !(arg isa LineNumberNode)]
    if node.head==:call && first(args)==:error
        length(args)==2 || error("unexpected diagnostic code in central launcher")
        value=args[2]
        allowed=value isa String || (value isa Expr && value.head==:string &&
            all(arg->arg isa String || arg==:CENTRAL_ENTRY,value.args))
        allowed || error("diagnostic interpolation contains executable code")
        args[2]="<diagnostic>"
    end
    return Expr(node.head,args...)
end

"""Validate both declared central dependencies and the complete thin-entry AST inventory."""
function validate_paper_entries(project,codes)
    project=realpath(project);codes=realpath(codes)
    "archived" in splitpath(project) && error("an archived project is not active evidence")
    central=joinpath(project,"paper","scripts","central.jl")
    isfile(central) || error("active central launcher is missing")
    source=read(central,String)
    matched=match(r"(?m)^\s*const\s+PAPER_ID\s*=\s*(\d+)\s*$",source)
    matched===nothing && error("central launcher has no literal paper identifier")
    paper=parse(Int,matched.captures[1]);paper in PAPERS || error("unsupported central paper: $paper")
    _strip(Meta.parseall(source))==_strip(Meta.parseall(paper_launcher_sources(paper)["central.jl"])) ||
        error("central launcher contains noncanonical execution or paths")
    paper_project=joinpath(project,"paper","Project.toml")
    specification=TOML.parsefile(paper_project)
    for (name,uuid) in UUIDS
        get(get(specification,"deps",Dict()),name,"")==uuid || error("paper environment omits $name")
        root=realpath(joinpath(codes,name*".jl"))
        "archived" in splitpath(root) && error("retired code cannot supply a central dependency")
        package=TOML.parsefile(joinpath(root,"Project.toml"))
        get(package,"uuid","")==uuid && get(package,"name","")==name || error("wrong central package identity")
        declared=get(get(specification,"sources",Dict()),name,nothing)
        if declared!==nothing
            haskey(declared,"path") || error("paper source declaration is not a central path: $name")
            realpath(joinpath(dirname(paper_project),declared["path"]))==root || error("paper source declaration resolves to retired/noncentral code")
        end
    end
    expected=paper_launcher_sources(paper);scripts=dirname(central)
    actual=String[]
    for (folder,_,names) in walkdir(scripts),name in names
        endswith(name,".jl") && push!(actual,replace(relpath(joinpath(folder,name),scripts),'\\'=>'/'))
    end
    sort(actual)==sort(collect(keys(expected))) || error("active script inventory contains missing or additional Julia implementations")
    for (relative,reference) in expected
        path=joinpath(scripts,split(relative,'/')...)
        islink(path) && error("active launchers must be regular files: $relative")
        _strip(Meta.parseall(read(path,String)))==_strip(Meta.parseall(reference)) ||
            error("active launcher contains noncanonical implementation: $relative")
    end
    files=[paper_project;[joinpath(scripts,split(relative,'/')...) for relative in actual]]
    return Dict("paper"=>paper,"project"=>project,"entry_sha256"=>Dict(relpath(path,project)=>bytes2hex(sha256(read(path))) for path in files))
end

end
