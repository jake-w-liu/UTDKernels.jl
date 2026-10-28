using Test,TOML,SHA
include("harness_reproduction.jl")
using .HarnessReproduction

@testset "complete reproduction receipt mutations" begin
    mktempdir() do directory
        logs=joinpath(directory,"reproduction-logs");mkpath(logs)
        stages=Dict{String,Any}[]
        for script in required_stages(168)
            log=joinpath(logs,script*".log");write(log,"isolated stage fixture\n")
            push!(stages,Dict("script"=>script,"exit_code"=>0,"log"=>log,
                "log_sha256"=>bytes2hex(sha256(read(log)))))
        end
        source=Dict("fixture"=>"source");runtime=Dict("version"=>string(VERSION),"fixture"=>"runtime")
        artifacts=Dict("data/fixture.csv"=>Dict("sha256"=>"fixture"),"figs/fixture.pdf"=>Dict("sha256"=>"fixture"))
        receipt=Dict("status"=>"passed","scope"=>"complete","paper"=>168,"backend"=>"kernels",
            "julia_version"=>string(VERSION),"runtime"=>runtime,"sources"=>source,"output"=>directory,
            "stages"=>stages,"artifacts"=>artifacts)
        path=joinpath(logs,"receipt.toml")
        function valid(record;actual=artifacts)
            open(path,"w") do stream;TOML.print(stream,record);end
            validate_reproduction_receipt(path,168,"kernels";sources=source,runtime,
                artifact_validator=(paper,output;plots)->actual)
        end
        @test valid(receipt)["status"]=="passed"
        for (key,value) in (("status","failed"),("status","running"),("scope","single_stage"),
                ("paper",170),("backend","diffutd"),("julia_version","0.0"),
                ("runtime",Dict("fixture"=>"stale")),("sources",Dict("fixture"=>"stale")))
            mutant=deepcopy(receipt);mutant[key]=value
            @test_throws Exception valid(mutant)
        end
        mutant=deepcopy(receipt);pop!(mutant["stages"])
        @test_throws Exception valid(mutant)
        mutant=deepcopy(receipt);mutant["stages"][1]["exit_code"]=1
        @test_throws Exception valid(mutant)
        mutant=deepcopy(receipt);delete!(mutant["stages"][1],"log_sha256")
        @test_throws Exception valid(mutant)
        @test_throws Exception valid(receipt;actual=Dict())
        write(stages[1]["log"],"tampered log\n")
        @test_throws Exception valid(receipt)
        rm(path)
        @test_throws Exception validate_reproduction_receipt(path,168,"kernels";sources=source,runtime,
            artifact_validator=(args...;kwargs...)->artifacts)
    end
end
