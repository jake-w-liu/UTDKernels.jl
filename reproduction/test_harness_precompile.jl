using Test
isdefined(@__MODULE__,:HarnessReceipts) || include("harness_receipts.jl")
using .HarnessReceipts

@testset "real Pkg.test observer and dependency precompile" begin
    output=joinpath(@__DIR__,"results","harness-precompile-controls")
    mkpath(output)
    directory=mktempdir(output;prefix="julia$(VERSION)-",cleanup=false)
    dependency=joinpath(directory,"ObserverFixtureDependency")
    target=joinpath(directory,"ObserverFixturePackage")
    controller=joinpath(directory,"controller")
    for path in (joinpath(dependency,"src"),joinpath(target,"src"),joinpath(target,"test"),controller)
        mkpath(path)
    end
    # Fresh fixture identities force real dependency precompilation rather than
    # reusing another test invocation's package image.
    dependency_uuid=string(Base.UUID(HarnessReceipts.rand(UInt128)))
    target_uuid=string(Base.UUID(HarnessReceipts.rand(UInt128)))
    nonce="precompile-control-"*target_uuid
    write(joinpath(dependency,"Project.toml"),
        "name = \"ObserverFixtureDependency\"\nuuid = \"$dependency_uuid\"\nversion = \"0.1.0\"\n")
    write(joinpath(dependency,"src","ObserverFixtureDependency.jl"),"""
        module ObserverFixtureDependency
        const WAS_PRECOMPILED=Base.generating_output()
        const CLEAN_PRECOMPILE=WAS_PRECOMPILED && !isdefined(Main,:HarnessReceipts) &&
            get(ENV,"CENTRAL_TEST_OBSERVER_ACTIVE","") != $(repr(nonce))
        offset(x)=2x+1
        end
        """)
    project=Dict("name"=>"ObserverFixturePackage","uuid"=>target_uuid,"version"=>"0.1.0",
        "deps"=>Dict("ObserverFixtureDependency"=>dependency_uuid),
        "extras"=>Dict("Test"=>string(Base.PkgId(Test).uuid)),"targets"=>Dict("test"=>["Test"]))
    write_toml(joinpath(target,"Project.toml"),project)
    write(joinpath(target,"src","ObserverFixturePackage.jl"),"""
        module ObserverFixturePackage
        using ObserverFixtureDependency
        answer()=ObserverFixtureDependency.offset(3)
        end
        """)
    pidfile=joinpath(directory,"actual-test-pid.txt")
    write(joinpath(target,"test","runtests.jl"),"""
        using Test,ObserverFixturePackage,ObserverFixtureDependency
        write($(repr(pidfile)),string(getpid()))
        @test ObserverFixturePackage.answer()==7
        @test ObserverFixtureDependency.WAS_PRECOMPILED
        @test ObserverFixtureDependency.CLEAN_PRECOMPILE
        @test isdefined(Main,:HarnessReceipts)
        @test !Base.generating_output()
        println("OBSERVER_DEPENDENCY_PRECOMPILED=",ObserverFixtureDependency.WAS_PRECOMPILED)
        println("OBSERVER_FIXTURE_TEST_COMPLETED")
        """)
    write(joinpath(controller,"Project.toml"),"[deps]\n")
    setup="using Pkg; Pkg.offline(true); Pkg.develop([Pkg.PackageSpec(path=$(repr(dependency))),Pkg.PackageSpec(path=$(repr(target)))])"
    setup_log=joinpath(directory,"setup.log")
    setup_command=`$(Base.julia_cmd()) --startup-file=no --project=$controller -e $setup`
    setup_process=open(setup_log,"w") do io
        withenv("JULIA_LOAD_PATH"=>"@:@stdlib","JULIA_PKG_PRECOMPILE_AUTO"=>"0") do
            run(pipeline(ignorestatus(setup_command);stdout=io,stderr=io))
        end
    end
    success(setup_process) || error("local fixture setup failed: "*first(read(setup_log,String),8000))
    @test success(setup_process)
    probe=joinpath(directory,"observer.jl")
    codes=normpath(joinpath(@__DIR__,"..",".."))
    write(probe,HarnessReceipts._probe("UTDKernels",codes,directory,nonce))
    body=HarnessReceipts._test_body("ObserverFixturePackage",["full"],4,probe)
    command=`$(Base.julia_cmd()) --startup-file=no --project=$controller -e $body`
    log=joinpath(directory,"Pkg.test.log")
    process=open(log,"w") do io
        withenv("JULIA_LOAD_PATH"=>"@:@stdlib","JULIA_PKG_PRECOMPILE_AUTO"=>"1",
                "JULIA_PKG_OFFLINE"=>"true","CENTRAL_TEST_OBSERVER_ACTIVE"=>nothing,
                "DUTD_TEST_TIER"=>"full","DUTD_RUN_EXTENDED_NUMERICAL_VALIDATIONS"=>"1",
                "MOM_CACHE_ONLY"=>"1") do
            run(pipeline(ignorestatus(command);stdout=io,stderr=io))
        end
    end
    transcript=read(log,String)
    success(process) || println(stderr,"Local Pkg.test failure (full log: $log):\n",first(transcript,12000))
    succeeded=success(process)
    clean_main_log=!occursin("closed module",transcript)
    clean_precompile_log=!occursin("Failed to precompile",transcript)
    actual_precompile=occursin("OBSERVER_DEPENDENCY_PRECOMPILED=true",transcript)
    actual_tests=occursin("OBSERVER_FIXTURE_TEST_COMPLETED",transcript)
    @test succeeded
    @test clean_main_log
    @test clean_precompile_log
    @test actual_precompile
    @test actual_tests
    if success(process)
        start=read_toml(joinpath(directory,"test-start.toml"))
        finish=read_toml(joinpath(directory,"test-end.toml"))
        actual_pid=parse(Int,read(pidfile,String))
        @test start["pid"]==actual_pid
        @test finish["pid"]==actual_pid
        @test start["arguments"]==String[]
        @test finish["arguments"]==["full"]
        @test start["runtime"]["version"]==string(VERSION)
        @test start["runtime"]["threads"]==4
        @test start["runtime"]["check_bounds"]==1
        @test start["environment"]["files"]==finish["environment"]["files"]
        @test haskey(start["environment"]["dependencies"],dependency_uuid)
        @test haskey(start["environment"]["dependencies"],target_uuid)
    end
    println("Disposable Pkg.test observer evidence: ",directory)
end
