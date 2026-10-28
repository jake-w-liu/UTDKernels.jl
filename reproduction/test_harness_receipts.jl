using Test, TOML
include("harness_receipts.jl")
using .HarnessReceipts

@testset "full-test worker loads and validates its entry arguments without launching tests" begin
    worker = Module(gensym(:FullTestWorkerControl))
    Core.eval(worker, :(include(path)=Base.include(@__MODULE__, path)))
    Base.include(worker, joinpath(@__DIR__, "full_test_gate.jl"))
    entry = getfield(worker, :main)
    @test_throws ErrorException entry(String[])
    @test_throws ArgumentError entry(["unknown-package-control", @__DIR__])
end

# Synthetic receipts below are isolated validator fixtures, never published
# as evidence of a package test. No real Pkg.test or experiment is invoked.
function receipt_fixture(directory)
    codes=joinpath(directory,"codes");mkpath(codes)
    for (owner,uuid) in HarnessReceipts.OWNER_UUIDS
        root=joinpath(codes,owner*".jl");mkpath(joinpath(root,"src"));mkpath(joinpath(root,"test"));mkpath(joinpath(root,"validation"))
        write(joinpath(root,"Project.toml"),"name = \"$owner\"\nuuid = \"$uuid\"\nversion = \"0.1.0\"\n")
        write(joinpath(root,"src",owner*".jl"),"module $owner\nend\n")
        write(joinpath(root,"test","runtests.jl"),"# isolated fixture\n")
        write(joinpath(root,"validation","oracle.csv"),"x,y\n1,2\n")
    end
    observed=joinpath(directory,"observations");mkpath(observed)
    write(joinpath(observed,"test-start-Project.toml"),"[deps]\n")
    write(joinpath(observed,"test-start-Manifest.toml"),"manifest_format = \"2.0\"\n")
    for name in ("Project.toml","Manifest.toml")
        cp(joinpath(observed,"test-start-"*name),joinpath(observed,"test-end-"*name))
    end
    packages=Dict(name=>package_snapshot(codes,name) for name in keys(HarnessReceipts.OWNER_UUIDS))
    dependencies=Dict(uuid=>Dict("name"=>name,"version"=>"0.1.0","tree_hash"=>"",
        "source"=>packages[name]["path"],"stdlib"=>false,"tracking_path"=>true,"tracking_repo"=>false)
        for (name,uuid) in HarnessReceipts.OWNER_UUIDS)
    environment=Dict("project"=>joinpath(observed,"Project.toml"),
        "files"=>Dict(name=>HarnessReceipts.digest(joinpath(observed,"test-start-"*name)) for name in ("Project.toml","Manifest.toml")),
        "dependencies"=>dependencies)
    runtime=runtime_snapshot()
    observer=joinpath(observed,"observer-fixture.jl");write(observer,"# isolated observer fixture\n")
    command=[runtime["executable"],"-e",HarnessReceipts._test_body("UTDKernels",String[],runtime["threads"],observer)]
    start=Dict("nonce"=>"isolated-validator-fixture","pid"=>123,"phase"=>"preload-before-test-entry",
        "runtime"=>runtime,"environment"=>environment,"packages"=>packages,"arguments"=>String[],
        "controls"=>Dict("DUTD_TEST_TIER"=>"full","DUTD_RUN_EXTENDED_NUMERICAL_VALIDATIONS"=>"1","MOM_CACHE_ONLY"=>"1"))
    finish=deepcopy(start);finish["phase"]="test-process-exit"
    write_toml(joinpath(observed,"test-start.toml"),start)
    write_toml(joinpath(observed,"test-end.toml"),finish)
    log=joinpath(observed,"fixture.log");write(log,"Testing UTDKernels tests passed\n")
    receipt=Dict("schema"=>1,"scope"=>"full-package-tests","owner"=>"UTDKernels","status"=>"passed","exit_code"=>0,
        "runtime"=>runtime,"nonce"=>start["nonce"],"before"=>packages["UTDKernels"],"after"=>packages["UTDKernels"],
        "before_packages"=>packages,"after_packages"=>packages,
        "builder_sha256"=>HarnessReceipts.digest(joinpath(@__DIR__,"harness_receipts.jl")),
        "log"=>log,"log_sha256"=>HarnessReceipts.digest(log),
        "observer_file"=>observer,"observer_sha256"=>HarnessReceipts.digest(observer),"command"=>command,
        "test_start"=>joinpath(observed,"test-start.toml"),"test_end"=>joinpath(observed,"test-end.toml"),
        "test_start_sha256"=>HarnessReceipts.digest(joinpath(observed,"test-start.toml")),
        "test_end_sha256"=>HarnessReceipts.digest(joinpath(observed,"test-end.toml")))
    return (;codes,receipt,dependencies,runtime)
end

@testset "full-test receipt mutations fail closed" begin
    mktempdir() do directory
        fixture=receipt_fixture(directory)
        valid(receipt)=validate_test_receipt(receipt,fixture.codes,"UTDKernels";
            runtime=fixture.runtime,dependencies=fixture.dependencies)
        @test valid(fixture.receipt)
        for (key,value) in (("scope","focused"),("status","failed"),("status","running"),
                             ("exit_code",1),("schema",0),("builder_sha256","stale"),("log",joinpath(directory,"absent.log")))
            mutant=deepcopy(fixture.receipt);mutant[key]=value
            @test_throws Exception valid(mutant)
        end
        mutant=deepcopy(fixture.receipt);mutant["runtime"]["version"]="0.0.0"
        @test_throws Exception valid(mutant)
        # The owner dependency itself is qualified by its source snapshot;
        # mutate the peer dependency present in this explicit fixture instead.
        peer=HarnessReceipts.OWNER_UUIDS["DiffUTD"]
        changed=deepcopy(fixture.dependencies);changed[peer]["version"]="999.0.0"
        @test_throws Exception validate_test_receipt(fixture.receipt,fixture.codes,"UTDKernels";
            runtime=fixture.runtime,dependencies=changed)
        @test_throws Exception validate_test_receipt(fixture.receipt,fixture.codes,"UTDKernels";
            runtime=fixture.runtime)
        @test_throws Exception validate_test_receipt(fixture.receipt,fixture.codes,"UTDKernels";
            runtime=fixture.runtime,dependencies=Dict())
        path=joinpath(fixture.codes,"UTDKernels.jl","validation","oracle.csv")
        saved=read(path);write(path,"changed fixture\n")
        @test_throws Exception valid(fixture.receipt)
        write(path,saved)
        write(fixture.receipt["log"],"tampered log\n")
        @test_throws Exception valid(fixture.receipt)
        write(fixture.receipt["log"],"Testing UTDKernels tests passed\n")
        write(joinpath(dirname(fixture.receipt["test_start"]),"test-start-Manifest.toml"),"tampered env\n")
        @test_throws Exception valid(fixture.receipt)
    end
    mktempdir() do directory
        fixture=receipt_fixture(directory)
        archived=joinpath(directory,"archived");mkpath(archived)
        mv(fixture.codes,joinpath(archived,"codes"))
        @test_throws Exception package_snapshot(joinpath(archived,"codes"),"UTDKernels")
    end
end

@testset "observer source parses without executing tests" begin
    @test Meta.parseall(HarnessReceipts._probe("UTDKernels","/fixture/codes","/fixture/output","nonce")) isa Expr
end
