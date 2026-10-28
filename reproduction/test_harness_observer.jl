using Test
isdefined(@__MODULE__,:HarnessReceipts) || include("harness_receipts.jl")
using .HarnessReceipts

@testset "isolated observer and nested-process guard (no Pkg.test)" begin
    mktempdir() do directory
        codes=normpath(joinpath(@__DIR__,"..",".."))
        probe=joinpath(directory,"observer.jl")
        write(probe,HarnessReceipts._probe("DiffUTD",codes,directory,"isolated-observer-control"))
        project=dirname(Base.active_project())
        nested=`$(Base.julia_cmd()) --startup-file=no --project=$project -L $probe -e "nothing"`
        body="append!(ARGS,[\"full\"]); run(pipeline($(repr(nested));stdout=devnull,stderr=devnull)); println(\"isolated observer control only\")"
        command=`$(Base.julia_cmd()) --startup-file=no --check-bounds=yes --threads=4 --project=$project -L $probe -e $body`
        log=joinpath(directory,"control.log")
        process=open(log,"w") do io
            withenv("DUTD_TEST_TIER"=>"full","DUTD_RUN_EXTENDED_NUMERICAL_VALIDATIONS"=>"1","MOM_CACHE_ONLY"=>"1") do
                run(pipeline(ignorestatus(command);stdout=io,stderr=io))
            end
        end
        @test success(process)
        start=read_toml(joinpath(directory,"test-start.toml"))
        finish=read_toml(joinpath(directory,"test-end.toml"))
        @test start["pid"]==finish["pid"]
        @test start["arguments"]==String[]
        @test finish["arguments"]==["full"]
        @test start["runtime"]["version"]==string(VERSION)
        @test start["runtime"]["check_bounds"]==1
        @test start["runtime"]["threads"]==4
        @test start["environment"]["files"]==finish["environment"]["files"]
        @test all(haskey(start["packages"],name) for name in ("UTDKernels","DiffUTD"))
    end
end

@testset "observer bootstrap in empty restricted project (no Pkg.test)" begin
    # Pkg's CacheFlags preload can lack the observer's named stdlib entries.
    # The observer must load by UUID without changing that sandbox's contract.
    mktempdir() do directory
        project=joinpath(directory,"empty-project");mkpath(project)
        write(joinpath(project,"Project.toml"),"[deps]\n")
        write(joinpath(project,"Manifest.toml"),
            "julia_version = \"$(VERSION)\"\nmanifest_format = \"2.0\"\n")
        before=Dict(name=>HarnessReceipts.digest(joinpath(project,name))
            for name in ("Project.toml","Manifest.toml"))
        parent_load_path=copy(LOAD_PATH)
        parent_environment=get(ENV,"JULIA_LOAD_PATH",nothing)
        legacy_log=joinpath(directory,"legacy-import.log")
        legacy=`$(Base.julia_cmd()) --startup-file=no --project=$project -e "module LegacyObserverImport; using Dates; end"`
        legacy_process=open(legacy_log,"w") do io
            withenv("JULIA_LOAD_PATH"=>"@") do
                run(pipeline(ignorestatus(legacy);stdout=io,stderr=io))
            end
        end
        @test !success(legacy_process)
        @test occursin("Package Dates not found",read(legacy_log,String))
        probe=joinpath(directory,"observer.jl")
        codes=normpath(joinpath(@__DIR__,"..",".."))
        write(probe,HarnessReceipts._probe("UTDKernels",codes,directory,
            "isolated-empty-project-observer-control"))
        state=joinpath(directory,"body-state.toml")
        nested=`$(Base.julia_cmd()) --startup-file=no --project=$project -L $probe -e "nothing"`
        body="append!(ARGS,[\"full\"]); run(pipeline($(repr(nested));stdout=devnull,stderr=devnull)); " *
             "write_toml($(repr(state)),Dict(\"load_path\"=>copy(LOAD_PATH),\"project\"=>Base.active_project()))"
        command=`$(Base.julia_cmd()) --startup-file=no --check-bounds=yes --threads=4 --project=$project -L $probe -e $body`
        log=joinpath(directory,"observer.log")
        process=open(log,"w") do io
            withenv("JULIA_LOAD_PATH"=>"@","CENTRAL_TEST_OBSERVER_ACTIVE"=>nothing,
                    "DUTD_TEST_TIER"=>"full","DUTD_RUN_EXTENDED_NUMERICAL_VALIDATIONS"=>"1",
                    "MOM_CACHE_ONLY"=>"1") do
                run(pipeline(ignorestatus(command);stdout=io,stderr=io))
            end
        end
        success(process) || error("empty-project observer failed: "*first(read(log,String),8000))
        @test success(process)
        start=read_toml(joinpath(directory,"test-start.toml"))
        finish=read_toml(joinpath(directory,"test-end.toml"))
        observed=read_toml(state)
        @test start["pid"]==finish["pid"]
        @test start["arguments"]==String[]
        @test finish["arguments"]==["full"]
        @test start["runtime"]["version"]==string(VERSION)
        @test start["runtime"]["check_bounds"]==1
        @test start["runtime"]["threads"]==4
        @test start["environment"]["files"]==before
        @test finish["environment"]["files"]==before
        @test isempty(start["environment"]["dependencies"])
        @test isempty(start["packages"])
        @test observed["load_path"]==["@"]
        @test observed["project"]==joinpath(project,"Project.toml")
        @test get(ENV,"JULIA_LOAD_PATH",nothing)==parent_environment
        @test LOAD_PATH==parent_load_path
    end
end
