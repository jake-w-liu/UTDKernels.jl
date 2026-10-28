using Test,TOML
include("harness_entries.jl")
using .HarnessEntries

@testset "active central launcher mutations" begin
    mktempdir() do directory
        codes=joinpath(directory,"codes")
        for (name,uuid) in HarnessEntries.UUIDS
            root=joinpath(codes,name*".jl");mkpath(root)
            open(joinpath(root,"Project.toml"),"w") do io
                TOML.print(io,Dict("name"=>name,"uuid"=>uuid))
            end
        end
        project=joinpath(directory,"paper168");paper=joinpath(project,"paper")
        mkpath(paper)
        HarnessEntries.CentralLauncherContract.write_paper_launchers(project,168)
        specification=Dict("deps"=>HarnessEntries.UUIDS)
        open(joinpath(paper,"Project.toml"),"w") do io;TOML.print(io,specification);end
        @test validate_paper_entries(project,codes)["paper"]==168
        central=joinpath(paper,"scripts","central.jl");saved=read(central)
        write(central,String(saved)*"\nusing RetiredImplementation\n")
        @test_throws Exception validate_paper_entries(project,codes)
        write(central,saved)
        write(central,replace(String(saved),"PAPER_ID = 168"=>"PAPER_ID = 169"))
        @test_throws Exception validate_paper_entries(project,codes)
        write(central,saved)
        alias=joinpath(paper,"scripts","reproduce.jl");alias_saved=read(alias)
        write(alias,"include(\"../archived/experiments.jl\")\ncentral_paper_main(PAPER_ID,PAPER_DIRECTORY,CENTRAL_CODES)\n")
        @test_throws Exception validate_paper_entries(project,codes)
        write(alias,alias_saved)
        extra=joinpath(paper,"scripts","old_numerics.jl");write(extra,"x=1\n")
        @test_throws Exception validate_paper_entries(project,codes)
        rm(extra)
        spec=deepcopy(specification);delete!(spec["deps"],"DiffUTD")
        open(joinpath(paper,"Project.toml"),"w") do io;TOML.print(io,spec);end
        @test_throws Exception validate_paper_entries(project,codes)
        open(joinpath(paper,"Project.toml"),"w") do io;TOML.print(io,specification);end
        moved=joinpath(directory,"archived");mkpath(moved)
        mv(project,joinpath(moved,"paper168"))
        @test_throws Exception validate_paper_entries(joinpath(moved,"paper168"),codes)
    end
    mktempdir() do directory
        sources=HarnessEntries.CentralLauncherContract.paper_launcher_sources(168)
        scripts=joinpath(directory,"paper","scripts");mkpath(joinpath(scripts,"data_generation"))
        central=joinpath(scripts,"central.jl");write(central,sources["central.jl"])
        retired=joinpath(scripts,"data_generation","generate_paper_data.jl")
        write(retired,"# unarchived numerical implementation\nx=1\n")
        before=read(retired)
        @test_throws Exception HarnessEntries.CentralLauncherContract.write_paper_launchers(directory,168)
        @test read(retired)==before
        @test read(central,String)==sources["central.jl"]
        @test !ispath(joinpath(scripts,"reproduce.jl"))
        @test !ispath(joinpath(scripts,"plotting"))
    end
end
