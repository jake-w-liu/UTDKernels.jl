module HarnessReceipts

# Pkg.test restricts the child's load path to its declared dependencies. The
# observer loads its own stdlibs by identity and leaves that sandbox unchanged.
const Dates=Base.require(Base.PkgId(Base.UUID("ade2ca70-3891-5945-98fb-dc099432e06a"),"Dates"))
const Pkg=Base.require(Base.PkgId(Base.UUID("44cfe95a-1eb2-52ea-b672-e2afdf69b78f"),"Pkg"))
const Random=Base.require(Base.PkgId(Base.UUID("9a3f8284-a2c9-5f02-9a11-845980a1fd5c"),"Random"))
const SHA=Base.require(Base.PkgId(Base.UUID("ea8e919c-243c-51af-8825-aaa63cd721ce"),"SHA"))
const TOML=Base.require(Base.PkgId(Base.UUID("fa267f1f-6049-4f14-aa54-33bafae1ed76"),"TOML"))
using .Dates, .Pkg, .Random, .SHA, .TOML

export package_inputs, package_snapshot, runtime_snapshot, environment_snapshot
export write_toml, read_toml, run_full_package_test, validate_test_receipt

const SCHEMA=1
const OWNER_UUIDS=Dict("UTDKernels"=>"970fd46b-db34-494d-92d6-28101d6e9662",
                      "DiffUTD"=>"70df04e7-1766-4225-9ef1-e80c4be476cd")
const INPUT_DIRECTORIES=("src","ext","test","validation","derivations")
const HERE=@__DIR__

digest(path)=bytes2hex(sha256(read(path)))
read_toml(path)=TOML.parsefile(path)
function write_toml(path,data)
    mkpath(dirname(path))
    mktemp(dirname(path)) do temporary,io
        TOML.print(io,data;sorted=true)
        close(io)
        mv(temporary,path;force=true)
    end
    return path
end

function canonical_root(codes,owner)
    haskey(OWNER_UUIDS,owner) || throw(ArgumentError("unknown central package: $owner"))
    root=realpath(joinpath(codes,owner*".jl"))
    "archived" in splitpath(root) && error("archived code cannot be an active package: $root")
    project=TOML.parsefile(joinpath(root,"Project.toml"))
    get(project,"name","")==owner && get(project,"uuid","")==OWNER_UUIDS[owner] ||
        error("central package identity mismatch: $root")
    return root
end

"""All production/test inputs, including non-Julia fixtures and external test helpers.

The checked test call graphs reach validation/ and (for DiffUTD) derivations/.
They do not execute documentation, examples, or the independent paper drivers.
Those drivers have separate source-bound reproduction receipts.
"""
function package_inputs(root::AbstractString)
    root=realpath(root)
    files=String[joinpath(root,"Project.toml")]
    append!(files,[joinpath(root,name) for name in readdir(root)
                   if startswith(name,"Manifest") && endswith(name,".toml")])
    for directory in INPUT_DIRECTORIES
        location=joinpath(root,directory)
        isdir(location) || continue
        for (folder,directories,names) in walkdir(location)
            filter!(name->name ∉ (".git","__pycache__"),directories)
            for name in names
                (name==".DS_Store" || endswith(name,".pyc")) && continue
                path=joinpath(folder,name)
                isfile(path) || error("test input is not a regular file: $path")
                push!(files,path)
            end
            for name in directories
                islink(joinpath(folder,name)) && error("symbolic test-input directory must be explicit: $(joinpath(folder,name))")
            end
        end
    end
    return Dict(relpath(path,root)=>digest(path) for path in sort!(unique!(files)))
end

function runtime_snapshot(;threads::Integer=Threads.nthreads(),check_bounds::Integer=Int(Base.JLOptions().check_bounds))
    executable=realpath(first(Base.julia_cmd().exec))
    return Dict("version"=>string(VERSION),"commit"=>Base.GIT_VERSION_INFO.commit,
        "machine"=>Sys.MACHINE,"kernel"=>string(Sys.KERNEL),"word_size"=>Sys.WORD_SIZE,
        "executable"=>executable,"executable_sha256"=>digest(executable),"threads"=>Int(threads),
        "check_bounds"=>Int(check_bounds),"opt_level"=>Int(Base.JLOptions().opt_level),
        "inline"=>Bool(Base.JLOptions().can_inline),"fast_math"=>Int(Base.JLOptions().fast_math))
end

function _test_body(owner,arguments,threads,probe)
    flags=["--startup-file=no","--check-bounds=yes","--threads=$threads","-L",probe]
    return "using Pkg; Pkg.test($(repr(owner)); allow_reresolve=false, test_args=$(repr(arguments)), julia_args=$(repr(flags)))"
end

function package_snapshot(codes::AbstractString,owner::AbstractString)
    root=canonical_root(codes,owner)
    return Dict("name"=>String(owner),"path"=>root,"inputs"=>package_inputs(root))
end

"""Capture the actual active environment without resolving or changing it."""
function environment_snapshot()
    project=Base.active_project()
    project===nothing && error("package-test environment has no active Project.toml")
    files=Dict(basename(path)=>digest(path) for path in
        [project;[joinpath(dirname(project),name) for name in readdir(dirname(project))
                   if startswith(name,"Manifest") && endswith(name,".toml")]])
    any(name->startswith(name,"Manifest"),keys(files)) || error("resolved test Manifest is missing")
    dependencies=Dict{String,Any}()
    for (uuid,info) in Pkg.dependencies()
        source=info.source===nothing ? "" : String(info.source)
        source=isempty(source) ? source : realpath(source)
        standard=!isempty(source) && (source==Sys.STDLIB || startswith(source,Sys.STDLIB*string(Base.Filesystem.path_separator)))
        dependencies[string(uuid)]=Dict("name"=>info.name,
            "version"=>info.version===nothing ? "" : string(info.version),
            "tree_hash"=>info.tree_hash===nothing ? "" : string(info.tree_hash),
            "source"=>source,"stdlib"=>standard,"tracking_path"=>info.is_tracking_path,
            "tracking_repo"=>info.is_tracking_repo)
    end
    return Dict("project"=>abspath(project),"files"=>files,"dependencies"=>dependencies)
end

function _probe(owner,codes,directory,nonce)
    # Pkg forwards this preload to dependency image workers as well as the
    # actual test child. A generating worker must not mutate its closed Main
    # module or the observer nonce; only ordinary processes record boundaries.
    return """
    if !Base.generating_output() && get(ENV,"CENTRAL_TEST_OBSERVER_ACTIVE","") != $(repr(nonce))
    ENV["CENTRAL_TEST_OBSERVER_ACTIVE"]=$(repr(nonce))
    include($(repr(joinpath(HERE,"harness_receipts.jl"))))
    using .HarnessReceipts
    function capture_test_boundary(name)
        environment=environment_snapshot()
        packages=Dict(name=>package_snapshot($(repr(codes)),name)
            for (name,uuid) in HarnessReceipts.OWNER_UUIDS if haskey(environment["dependencies"],uuid))
        state=Dict("nonce"=>$(repr(nonce)),"pid"=>getpid(),"phase"=>name=="test-start" ? "preload-before-test-entry" : "test-process-exit",
            "captured_utc"=>string(HarnessReceipts.now(HarnessReceipts.UTC)),
            "runtime"=>runtime_snapshot(),"environment"=>environment,
            "arguments"=>copy(ARGS),"packages"=>packages,
            "controls"=>Dict(key=>get(ENV,key,"") for key in
                ("DUTD_TEST_TIER","DUTD_RUN_EXTENDED_NUMERICAL_VALIDATIONS","MOM_CACHE_ONLY")))
        write_toml(joinpath($(repr(directory)),name*".toml"),state)
        for (file,_) in state["environment"]["files"]
            cp(joinpath(dirname(Base.active_project()),file),joinpath($(repr(directory)),name*"-"*file);force=true)
        end
    end
    capture_test_boundary("test-start")
    atexit(()->capture_test_boundary("test-end"))
    end
    """
end

function _matching_dependencies(actual,current,owner)
    for (uuid,dependency) in actual
        dependency["stdlib"] && continue
        dependency["name"]==owner && continue
        # Test-only extras are locked by the captured temporary Manifest.
        # Every dependency shared with the active reproduction environment
        # must match; it contains both central packages' production closures.
        haskey(current,uuid) || continue
        expected=current[uuid]
        for key in ("name","version","tree_hash","source","tracking_path","tracking_repo")
            dependency[key]==expected[key] || return false
        end
    end
    return true
end

"""Reject partial, failed, stale, or incompletely observed full-test receipts."""
function validate_test_receipt(receipt,codes,owner;runtime=runtime_snapshot(),dependencies=nothing)
    get(receipt,"schema",0)==SCHEMA || error("unsupported full-test receipt schema")
    get(receipt,"scope","")=="full-package-tests" || error("receipt is not a full package test")
    get(receipt,"status","")=="passed" && get(receipt,"exit_code",-1)==0 || error("package test did not pass")
    receipt["runtime"]==runtime || error("package-test runtime differs from selected runtime")
    receipt["builder_sha256"]==digest(joinpath(HERE,"harness_receipts.jl")) || error("full-test observer changed")
    get(receipt,"owner","")==owner || error("wrong package-test owner")
    observer=receipt["observer_file"]
    isfile(observer) && digest(observer)==receipt["observer_sha256"] || error("test observer is missing or changed")
    arguments=owner=="DiffUTD" ? ["full"] : String[]
    body=_test_body(owner,arguments,runtime["threads"],observer)
    command=receipt["command"]
    length(command)>=3 && command[end-1]=="-e" && command[end]==body || error("command was not the full Pkg.test gate")
    current=package_snapshot(codes,owner)
    receipt["before"]==current && receipt["after"]==current || error("package-test inputs are stale")
    isfile(receipt["log"]) && digest(receipt["log"])==receipt["log_sha256"] || error("full-test log missing or changed")
    start=read_toml(receipt["test_start"]);finish=read_toml(receipt["test_end"])
    digest(receipt["test_start"])==receipt["test_start_sha256"] || error("test-start observation changed")
    digest(receipt["test_end"])==receipt["test_end_sha256"] || error("test-end observation changed")
    start["nonce"]==finish["nonce"]==receipt["nonce"] || error("full-test observer nonce mismatch")
    start["pid"]==finish["pid"] || error("test boundary observations came from different processes")
    expected_controls=Dict("DUTD_TEST_TIER"=>"full","DUTD_RUN_EXTENDED_NUMERICAL_VALIDATIONS"=>"1","MOM_CACHE_ONLY"=>"1")
    start["controls"]==finish["controls"]==expected_controls || error("full-tier/cache test policy changed")
    start["runtime"]==finish["runtime"]==runtime || error("test-process runtime differs")
    start["environment"]==finish["environment"] || error("test environment changed during execution")
    haskey(start["packages"],owner) || error("test process did not observe its central package")
    keys(start["packages"])==keys(finish["packages"]) || error("test package dependency set changed")
    for name in keys(start["packages"])
        snapshot=package_snapshot(codes,name)
        start["packages"][name]==finish["packages"][name]==snapshot || error("test package dependency source changed: $name")
        receipt["before_packages"][name]==receipt["after_packages"][name]==snapshot || error("test dependency source was stale at a boundary: $name")
        uuid=OWNER_UUIDS[name]
        start["environment"]["dependencies"][uuid]["source"]==snapshot["path"] || error("test loaded a noncentral package: $name")
    end
    # Pkg sets test_args in its --eval body after -L preloads (verified in
    # Pkg.Operations.gen_test_code on supported runtimes). Observe that fact
    # honestly; require the final argument vector to be the full selector.
    isempty(start["arguments"]) && finish["arguments"]==arguments || error("partial test selector was used")
    for (name,hash) in start["environment"]["files"]
        for prefix in ("test-start-","test-end-")
            file=joinpath(dirname(receipt["test_start"]),prefix*name)
            isfile(file) && digest(file)==hash || error("captured test environment missing or changed")
        end
    end
    dependencies===nothing && error("current reproduction dependencies are required for full-test validation")
    for (name,uuid) in OWNER_UUIDS
        haskey(dependencies,uuid) && dependencies[uuid]["source"]==canonical_root(codes,name) ||
            error("current dependencies do not bind both central packages")
    end
    for dependency in values(start["environment"]["dependencies"])
        "archived" in splitpath(dependency["source"]) && error("test environment contains retired source")
    end
    _matching_dependencies(start["environment"]["dependencies"],dependencies,owner) ||
        error("resolved dependencies differ from the completed full test")
    occursin("Testing $owner tests passed",read(receipt["log"],String)) || error("Pkg.test completion marker is missing")
    return true
end

"""Launch one fully observed Pkg.test; never terminate or suppress failures.

Invoke this worker under the explicitly selected Julia runtime. The receipt
captures source boundaries and Pkg.test's actual temporary dependencies.
"""
function run_full_package_test(owner::AbstractString,codes::AbstractString;
                               output::AbstractString,threads::Integer=4)
    root=canonical_root(codes,owner)
    mkpath(output)
    directory=mktempdir(output;prefix=owner*"-julia"*string(VERSION)*"-",cleanup=false)
    nonce=bytes2hex(rand(UInt8,24))
    before=package_snapshot(codes,owner)
    controller=environment_snapshot()
    for (name,uuid) in OWNER_UUIDS
        haskey(controller["dependencies"],uuid) || error("run full_test_gate.jl with the central reproduction environment")
        controller["dependencies"][uuid]["source"]==canonical_root(codes,name) || error("test controller resolved a noncentral $name")
    end
    before_packages=Dict(name=>package_snapshot(codes,name) for name in keys(OWNER_UUIDS))
    probe=joinpath(directory,"observe-test.jl")
    write(probe,_probe(owner,realpath(codes),directory,nonce))
    arguments=owner=="DiffUTD" ? ["full"] : String[]
    body=_test_body(owner,arguments,threads,probe)
    command=`$(Base.julia_cmd()) --startup-file=no --threads=$threads --project=$(dirname(Base.active_project())) -e $body`
    log=joinpath(directory,"Pkg.test.log")
    receipt=Dict{String,Any}("schema"=>SCHEMA,"scope"=>"full-package-tests",
        "status"=>"running","owner"=>String(owner),"runtime"=>runtime_snapshot(;threads,check_bounds=1),
        "before"=>before,"before_packages"=>before_packages,"controller_environment"=>controller,
        "nonce"=>nonce,"builder_sha256"=>digest(joinpath(HERE,"harness_receipts.jl")),
        "started_utc"=>string(now(UTC)),"log"=>log,"command"=>copy(command.exec),
        "observer_file"=>probe,"observer_sha256"=>digest(probe))
    receipt_path=joinpath(directory,"receipt.toml")
    write_toml(receipt_path,receipt)
    try
        process=open(log,"w") do stream
            withenv("JULIA_LOAD_PATH"=>"@:@stdlib","DUTD_TEST_TIER"=>"full",
                    "DUTD_RUN_EXTENDED_NUMERICAL_VALIDATIONS"=>"1","MOM_CACHE_ONLY"=>"1") do
                run(pipeline(ignorestatus(command);stdout=stream,stderr=stream))
            end
        end
        receipt["exit_code"]=Int(process.exitcode)
        receipt["after"]=package_snapshot(codes,owner)
        receipt["after_packages"]=Dict(name=>package_snapshot(codes,name) for name in keys(OWNER_UUIDS))
        receipt["log_sha256"]=digest(log)
        receipt["finished_utc"]=string(now(UTC))
        for boundary in ("start","end")
            file=joinpath(directory,"test-"*boundary*".toml")
            receipt["test_"*boundary]=file
            isfile(file) && (receipt["test_"*boundary*"_sha256"]=digest(file))
        end
        receipt["status"]=success(process) ? "passed" : "failed"
        write_toml(receipt_path,receipt)
        success(process) || error("full $owner Pkg.test failed; see $log")
        environment_snapshot()==controller || error("test-controller environment changed during execution")
        validate_test_receipt(receipt,codes,owner;runtime=runtime_snapshot(;threads,check_bounds=1),
            dependencies=controller["dependencies"])
    catch error
        receipt["status"]="failed"
        receipt["error"]=first(sprint(showerror,error),4000)
        receipt["finished_utc"]=string(now(UTC))
        write_toml(receipt_path,receipt)
        rethrow()
    end
    return receipt_path
end

end
