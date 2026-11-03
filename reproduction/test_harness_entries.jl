using Test,TOML
include("harness_entries.jl")
using .HarnessEntries

@testset "active central launcher mutations" begin
    for study_slug in HarnessEntries.CentralLauncherContract.STUDIES
        mktempdir() do directory
            codes=joinpath(directory,"codes")
            for (name,uuid) in HarnessEntries.UUIDS
                root=joinpath(codes,name*".jl");mkpath(root)
                open(joinpath(root,"Project.toml"),"w") do io
                    TOML.print(io,Dict("name"=>name,"uuid"=>uuid))
                end
            end
            project=joinpath(directory,study_slug);study=joinpath(project,"paper")
            mkpath(study)
            HarnessEntries.CentralLauncherContract.write_study_launchers(project,study_slug)
            owned=["plotting/plot.jl"]
            study_slug=="null-uniform" && push!(owned,"plotting/plot_wedge.jl")
            for relative in owned
                path=joinpath(study,"scripts",split(relative,'/')...);mkpath(dirname(path))
                write(path,"# project-owned plot implementation\n")
            end
            specification=Dict("deps"=>HarnessEntries.UUIDS)
            open(joinpath(study,"Project.toml"),"w") do io;TOML.print(io,specification);end
            @test validate_study_entries(project,codes)["study"]==study_slug
            central=joinpath(study,"scripts","central.jl");saved=read(central)
            write(central,String(saved)*"\nusing RetiredImplementation\n")
            @test_throws Exception validate_study_entries(project,codes)
            write(central,saved)
            write(central,replace(String(saved),"STUDY_SLUG = "*repr(study_slug)=>"STUDY_SLUG = \"unknown-study\""))
            @test_throws Exception validate_study_entries(project,codes)
            write(central,saved)
            alias=joinpath(study,"scripts","reproduce.jl");alias_saved=read(alias)
            write(alias,"include(\"../archived/experiments.jl\")\ncentral_study_main(STUDY_SLUG,STUDY_DIRECTORY,CENTRAL_CODES)\n")
            @test_throws Exception validate_study_entries(project,codes)
            write(alias,alias_saved)
            extra=joinpath(study,"scripts","old_numerics.jl");write(extra,"x=1\n")
            @test_throws Exception validate_study_entries(project,codes)
            rm(extra)
            spec=deepcopy(specification);delete!(spec["deps"],"DiffUTD")
            open(joinpath(study,"Project.toml"),"w") do io;TOML.print(io,spec);end
            @test_throws Exception validate_study_entries(project,codes)
            open(joinpath(study,"Project.toml"),"w") do io;TOML.print(io,specification);end
            moved=joinpath(directory,"archived");mkpath(moved)
            mv(project,joinpath(moved,study_slug))
            @test_throws Exception validate_study_entries(joinpath(moved,study_slug),codes)
        end
    end
    mktempdir() do directory
        sources=HarnessEntries.CentralLauncherContract.study_launcher_sources("null-uniform")
        scripts=joinpath(directory,"paper","scripts");mkpath(joinpath(scripts,"data_generation"))
        central=joinpath(scripts,"central.jl");write(central,sources["central.jl"])
        retired=joinpath(scripts,"data_generation","generate_paper_data.jl")
        write(retired,"# unarchived numerical implementation\nx=1\n")
        before=read(retired)
        @test_throws Exception HarnessEntries.CentralLauncherContract.write_study_launchers(directory,"null-uniform")
        @test read(retired)==before
        @test read(central,String)==sources["central.jl"]
        @test !ispath(joinpath(scripts,"reproduce.jl"))
        @test !ispath(joinpath(scripts,"plotting"))
    end
end
