module PaperArtifacts

using CSV
using JSON
using SHA

export validate_artifacts, publish_artifacts

function allowed_nonfinite(value,column,row,rules)
    kind=isnan(value) ? "nan" : value>0 ? "posinf" : "neginf"
    return any(rules) do rule
        column in rule["columns"] && kind==rule["kind"] || return false
        return all(get(rule,"when",Dict())) do condition
            name,expected=condition
            actual=row[Symbol(name)]
            ismissing(actual) && return false
            return expected isa Number ? tryparse(Float64,actual)==expected : actual==expected
        end
    end
end

"""Check the complete declared result inventory and retain content hashes."""
function validate_artifacts(paper::Integer, directory::AbstractString; plots::Bool=true)
    specification=JSON.parsefile(joinpath(@__DIR__,"papers",string(paper),"coverage.json"))
    specification["paper"]==paper || error("coverage inventory has the wrong paper identifier")
    artifacts=Dict{String,Any}()
    for (name,expected) in specification["datasets"]
        basename(name)==name && endswith(name,".csv") || error("invalid dataset inventory name: $name")
        relative=joinpath("data",name)
        path=joinpath(directory,relative)
        isfile(path) || error("missing reproduced dataset: $relative")
        table=CSV.File(path;delim=',',comment="#",types=String,on_error=:error,normalizenames=false)
        columns=String.(propertynames(table))
        columns==expected["columns"] || error("dataset columns disagree with inventory: $relative")
        length(table)==expected["rows"] || error("dataset row count disagrees with inventory: $relative")
        rules=get(get(specification,"numeric_exceptions",Dict()),name,[])
        for (index,row) in enumerate(table),column in columns
            cell=row[Symbol(column)]
            ismissing(cell) && continue
            value=tryparse(Float64,cell)
            if value!==nothing && !isfinite(value)
                allowed_nonfinite(value,column,row,rules) ||
                    error("unexpected nonfinite value in $relative row $index column $column")
            end
        end
        artifacts[relative]=Dict("rows"=>length(table),"columns"=>columns,
            "sha256"=>bytes2hex(sha256(read(path))))
    end
    observed=sort(filter(name->endswith(name,".csv"),readdir(joinpath(directory,"data"))))
    observed==sort(collect(keys(specification["datasets"]))) ||
        error("reproduced CSV inventory differs from declared paper coverage")
    required=["results_macros.tex",joinpath("data","results_macros.tex")]
    paper==160 && append!(required,["results_macros_extra.tex",joinpath("data","results_macros_extra.tex")])
    if haskey(specification,"additional_receipt")
        push!(required,specification["additional_receipt"])
    end
    plots && append!(required,[joinpath("figs",name) for name in specification["figures"]])
    for relative in required
        path=joinpath(directory,relative)
        isfile(path) && filesize(path)>0 || error("missing or empty reproduced artifact: $relative")
        if endswith(relative,".pdf")
            open(path) do stream
                String(read(stream,5))=="%PDF-" || error("invalid PDF artifact: $relative")
            end
        end
        artifacts[relative]=Dict("bytes"=>filesize(path),"sha256"=>bytes2hex(sha256(read(path))))
    end
    for name in (paper==160 ? ("results_macros.tex","results_macros_extra.tex") : ("results_macros.tex",))
        artifacts[name]["sha256"]==artifacts[joinpath("data",name)]["sha256"] ||
            error("manuscript macro copy differs from generated data: $name")
    end
    return artifacts
end

"""Publish validated outputs without replacing unrelated manuscript files."""
function publish_artifacts(staging::AbstractString,destination::AbstractString)
    for (folder,_,names) in walkdir(staging)
        for name in names
            source=joinpath(folder,name)
            target=joinpath(destination,relpath(source,staging))
            mkpath(dirname(target))
            mktemp(dirname(target)) do temporary,stream
                close(stream)
                cp(source,temporary;force=true)
                mv(temporary,target;force=true)
            end
        end
    end
    return nothing
end

end
