module PaperReproductionPaths

using SHA
using TOML
using Pkg

export SUPPORTED_PAPERS, selected_backend, output_directory, source_snapshot

const SUPPORTED_PAPERS = (160, 162, 165, 168, 170)
const REPRODUCTION_ROOT = @__DIR__

function selected_backend()
    backend = get(ENV, "UTD_PAPER_BACKEND", "kernels")
    backend in ("kernels", "diffutd") || throw(ArgumentError(
        "UTD_PAPER_BACKEND must be kernels or diffutd; received $(repr(backend))",
    ))
    return Symbol(backend)
end

function output_directory(paper::Integer)
    paper in SUPPORTED_PAPERS || throw(ArgumentError("unsupported paper: $paper"))
    backend = selected_backend()
    path = get(ENV, "UTD_PAPER_OUTPUT", joinpath(
        REPRODUCTION_ROOT, "results", string(backend), string(paper), "paper",
    ))
    isabspath(path) || throw(ArgumentError("UTD_PAPER_OUTPUT must be an absolute path"))
    path = normpath(path)
    ispath(path) && !isdir(path) && throw(ArgumentError("output is not a directory: $path"))
    mkpath(path)
    return path
end

function implementation_digest(root::AbstractString)
    files = String[joinpath(root, "Project.toml")]
    for directory in ("src", "ext", "reproduction")
        location = joinpath(root, directory)
        isdir(location) || continue
        for (folder, directories, names) in walkdir(location)
            # Generated results and their receipts are outputs, not source.
            filter!(name -> name != "results", directories)
            for name in names
                splitext(name)[2] in (".jl", ".toml", ".json") || continue
                startswith(name,"Manifest") && endswith(name,".toml") && continue
                push!(files, joinpath(folder, name))
            end
        end
    end
    entries = [relpath(path, root) * "\0" * bytes2hex(sha256(read(path)))
               for path in sort!(unique!(files))]
    return bytes2hex(sha256(join(entries, "\n")))
end

function source_snapshot(kernels::Module, spatial::Module)
    expected = Dict(
        "UTDKernels" => normpath(joinpath(REPRODUCTION_ROOT, "..")),
        "DiffUTD" => normpath(joinpath(REPRODUCTION_ROOT, "..", "..", "DiffUTD.jl")),
    )
    snapshots = Dict{String,Any}()
    for package in (kernels, spatial)
        name = string(nameof(package))
        haskey(expected, name) || throw(ArgumentError("unexpected package module: $name"))
        source = pathof(package)
        source === nothing && throw(ArgumentError("$name has no package source path"))
        root = realpath(dirname(dirname(source)))
        root == realpath(expected[name]) || error(
            "$name must load from the central checkout $(expected[name]); loaded $root",
        )
        snapshots[name] = Dict(
            "path" => root,
            "version" => string(Base.pkgversion(package)),
            "commit" => readchomp(`git -C $root rev-parse HEAD`),
            "implementation_sha256" => implementation_digest(root),
        )
    end
    project = Base.active_project()
    project === nothing && error("reproduction requires an active Julia project")
    # Pkg selects a version-specific manifest when one is present.
    manifest = Pkg.Types.EnvCache(project).manifest_file
    isfile(manifest) || error("instantiate the reproduction environment before running studies")
    snapshots["environment"] = Dict(
        "project" => abspath(project),
        "project_sha256" => bytes2hex(sha256(read(project))),
        "manifest" => abspath(manifest),
        "manifest_sha256" => bytes2hex(sha256(read(manifest))),
        "julia_version" => string(VERSION),
    )
    return snapshots
end

end
