using Test

include(joinpath(@__DIR__, "..", "validation", "compare_wdc.jl"))

const GOOD_WDC_ROW =
    "1.500000,1.000000,89.0000000000,45.0000000000," *
    "8.789964463054159e-02,-4.460242351202439e-02," *
    "-1.862972841036014e-01,1.345128941248050e-01," *
    "-4.919881973652992e-02,4.495523530639031e-02," *
    "-1.370984643670715e-01,8.955765881841470e-02"

const EXACT_BOUNDARY_WDC_ROW =
    "1.500000,1.000000,225.0000000000,45.0000000000," *
    "-6.148426083573795e-01,9.779463156940034e-02," *
    "-3.403819498858196e-01,-1.392309545814321e-01," *
    "-4.776122791215995e-01,-2.071816150601588e-02," *
    "1.372303292357799e-01,-1.185127930754162e-01"

const BAD_REFLECTION_WDC_ROW =
    "1.500000,1.000000,89.0000000000,45.0000000000," *
    "8.789964463054159e-02,-4.460242351202439e-02," *
    "-1.862972841036014e-01,1.345128941248050e-01," *
    "-4.919881973652992e-02,4.495523530639031e-02," *
    "1.000000000000000e+01,0.000000000000000e+00"

function with_wdc_fixture(f::Function, rows::AbstractVector{<:AbstractString})
    return mktemp() do path, io
        println(io, join(WDC_COLUMNS, ','))
        foreach(row -> println(io, row), rows)
        close(io)
        f(path)
    end
end

function validation_cli_exitcode(csvfile::AbstractString)
    validation_dir = normpath(joinpath(@__DIR__, "..", "validation"))
    script = joinpath(validation_dir, "compare_wdc.jl")
    command = `$(Base.julia_cmd()) --startup-file=no --project=$validation_dir $script $csvfile`
    process = run(pipeline(ignorestatus(command); stdout=devnull, stderr=devnull))
    return process.exitcode
end

@testset "WDC validation harness fails closed" begin
    @test compare_transition_function(; verbose=false) <= MATLAB_FTF_TOL

    with_wdc_fixture([GOOD_WDC_ROW]) do path
        @test load_reference_data(path) |> length == 1
        @test run_comparison(; verbose=false, csvfile=path) ==
              (passed=1, failed=0, skipped=0)
        @test main([path]; verbose=false) == 0
        @test validation_cli_exitcode(path) == 0
    end

    with_wdc_fixture([EXACT_BOUNDARY_WDC_ROW]) do path
        @test run_comparison(; verbose=false, csvfile=path) ==
              (passed=0, failed=0, skipped=1)
    end

    # The incident component is unchanged, so a Di-only oracle would miss this
    # deliberately corrupted reflected component.
    with_wdc_fixture([BAD_REFLECTION_WDC_ROW]) do path
        @test run_comparison(; verbose=false, csvfile=path) ==
              (passed=0, failed=1, skipped=0)
        @test main([path]; verbose=false) == 1
        @test validation_cli_exitcode(path) == 1
    end

    mktemp() do path, io
        println(io, "wrong,header")
        close(io)
        @test_throws ArgumentError load_reference_data(path)
    end

    mktemp() do path, io
        println(io, join(WDC_COLUMNS, ','))
        println(io, "1.5,1.0,90.0")
        close(io)
        @test_throws ArgumentError load_reference_data(path)
    end

    mktemp() do path, io
        println(io, join(WDC_COLUMNS, ','))
        println(io, replace(GOOD_WDC_ROW, "1.500000" => "NaN"; count=1))
        close(io)
        @test_throws ArgumentError load_reference_data(path)
    end

    mktemp() do path, io
        println(io, join(WDC_COLUMNS, ','))
        close(io)
        @test_throws ArgumentError load_reference_data(path)
    end

    @test main(["first.csv", "second.csv"]; verbose=false) == 2
end
