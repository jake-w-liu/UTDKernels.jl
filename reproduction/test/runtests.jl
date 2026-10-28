using Test
using CSV
using JSON
using TOML
include(joinpath(@__DIR__,"..","run.jl"))

@testset "Complete-run manuscript metadata uses the actual runtime" begin
    mktempdir() do directory
        mkpath(joinpath(directory, "data"))
        paths = (joinpath(directory, "data", "results_macros.tex"),
                 joinpath(directory, "results_macros.tex"))
        for path in paths
            write(path, "\\newcommand{\\ScientificResult}{1}\n")
        end
        stamp_runtime_macros(directory, "diffutd")
        stamped = read(first(paths), String)
        @test stamped == read(last(paths), String)
        @test occursin("\\newcommand{\\ScientificResult}{1}", stamped)
        @test occursin("\\newcommand{\\ExperimentJuliaVersion}{$VERSION}", stamped)
        @test occursin("\\newcommand{\\ExperimentUTDKernelsVersion}{$(pkgversion(UTDKernels))}", stamped)
        @test occursin("\\newcommand{\\ExperimentDiffUTDVersion}{$(pkgversion(DiffUTD))}", stamped)
        @test occursin("\\newcommand{\\ExperimentBackend}{diffutd}", stamped)
        @test occursin("\\newcommand{\\ExperimentThreads}{$(Threads.nthreads())}", stamped)
        @test_throws ErrorException stamp_runtime_macros(directory, "diffutd")
        write(last(paths), "stale macro copy\n")
        @test_throws ErrorException stamp_runtime_macros(directory, "diffutd")
        @test read(last(paths), String) == "stale macro copy\n"
        @test_throws ArgumentError stamp_runtime_macros(directory, "unknown")
    end
end

@testset "Reproduction command contract" begin
    @test options(String[]).papers==[160,162,165,168,170]
    @test options(["--paper","165","--backend","diffutd"]).backend=="diffutd"
    @test options(["--paper","160","--stage","spatial_validation.jl"]).stage=="spatial_validation.jl"
    for invalid in (["--paper"],["--backend","other"],["--paper","161"],
                    ["--stage","generate.jl"],["--paper","162","--stage","cost.jl"],["--unknown"])
        @test_throws ArgumentError options(invalid)
    end
end

@testset "Inventory rejects partial and malformed results" begin
    specification=JSON.parsefile(joinpath(@__DIR__,"..","papers","162","coverage.json"))
    mktempdir() do directory
        data=joinpath(directory,"data");mkpath(data)
        # These synthetic cells test only the structural gate. Scientific
        # correctness is enforced by the independent references in each driver.
        for (name,entry) in specification["datasets"]
            open(joinpath(data,name),"w") do stream
                println(stream,"# structural test fixture")
                println(stream,join(entry["columns"],','))
                row=join(fill("0",length(entry["columns"])),',')
                for _ in 1:entry["rows"]
                    println(stream,row)
                end
            end
        end
        for root in (directory,data)
            write(joinpath(root,"results_macros.tex"),"% structural fixture\n")
        end
        checked=validate_artifacts(162,directory;plots=false)
        @test length(checked)==length(specification["datasets"])+2
        @test_throws ErrorException validate_artifacts(162,directory;plots=true)
        write(joinpath(data,"unexpected.csv"),"a\n0\n")
        @test_throws ErrorException validate_artifacts(162,directory;plots=false)
        rm(joinpath(data,"unexpected.csv"))
        write(joinpath(directory,"results_macros.tex"),"% stale macro copy\n")
        @test_throws ErrorException validate_artifacts(162,directory;plots=false)
        cp(joinpath(data,"results_macros.tex"),joinpath(directory,"results_macros.tex");force=true)
        name=first(sort(collect(keys(specification["datasets"]))))
        path=joinpath(data,name);original=read(path,String)
        write(path,replace(original,"\n0,"=>"\nNaN,";count=1))
        @test_throws ErrorException validate_artifacts(162,directory;plots=false)
        write(path,join(split(original,'\n')[1:end-2],'\n')*"\n")
        @test_throws ErrorException validate_artifacts(162,directory;plots=false)
        write(path,original*"\"unterminated\n")
        @test_throws CSV.ParseError validate_artifacts(162,directory;plots=false)
    end
end

@testset "Receipts and publication preserve unrelated files" begin
    mktempdir() do directory
        receipt=joinpath(directory,"logs","receipt.toml")
        write_receipt(receipt,Dict("status"=>"running"))
        write_receipt(receipt,Dict("status"=>"failed","error"=>"injected failure"))
        @test TOML.parsefile(receipt)["status"]=="failed"
        staging=joinpath(directory,"staging");mkpath(joinpath(staging,"data"))
        write(joinpath(staging,"data","output.csv"),"x\n2\n")
        destination=joinpath(directory,"paper");mkpath(destination)
        write(joinpath(destination,"main.tex"),"manuscript")
        publish_artifacts(staging,destination)
        @test read(joinpath(destination,"main.tex"),String)=="manuscript"
        @test read(joinpath(destination,"data","output.csv"),String)=="x\n2\n"
    end
end

@testset "Source digest excludes runtime resolver output" begin
    mktempdir() do directory
        write(joinpath(directory,"Project.toml"),"[deps]\n")
        mkpath(joinpath(directory,"src"));write(joinpath(directory,"src","example.jl"),"x=1\n")
        mkpath(joinpath(directory,"reproduction"))
        before=PaperReproductionPaths.implementation_digest(directory)
        write(joinpath(directory,"reproduction","Manifest-v1.13.toml"),"resolver state")
        @test PaperReproductionPaths.implementation_digest(directory)==before
        write(joinpath(directory,"reproduction","coverage.json"),"{}")
        @test PaperReproductionPaths.implementation_digest(directory)!=before
        before=PaperReproductionPaths.implementation_digest(directory)
        write(joinpath(directory,"src","example.jl"),"x=2\n")
        @test PaperReproductionPaths.implementation_digest(directory)!=before
    end
end
