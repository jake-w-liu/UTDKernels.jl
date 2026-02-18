#!/usr/bin/env julia
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

include(joinpath(@__DIR__, "example_13_3.jl"))
include(joinpath(@__DIR__, "example_13_4.jl"))
include(joinpath(@__DIR__, "example_13_5.jl"))
include(joinpath(@__DIR__, "example_13_6.jl"))
include(joinpath(@__DIR__, "example_13_7.jl"))

function run_all_examples(; save_png = true)
    println("Running Balanis GTD examples (13-3 to 13-7)...")
    m13_3 = run_example_13_3(; save_png = save_png)
    m13_4 = run_example_13_4(; save_png = save_png)
    m13_5 = run_example_13_5(; save_png = save_png)
    m13_6 = run_example_13_6(; save_png = save_png)
    m13_7 = run_example_13_7(; save_png = save_png)

    println("\nSummary")
    println("  13-3  max rel (soft/hard): $(m13_3.max_rel_soft), $(m13_3.max_rel_hard)")
    println("  13-4  max rel: $(m13_4.max_rel)")
    println("  13-5  max rel (σ/λ,10GHz): $(m13_5.max_rel_sigma_over_lambda), $(m13_5.max_rel_10ghz)")
    println("  13-6  max rel: $(m13_6.max_rel)")
    println("  13-7  max rel: $(m13_7.max_rel)")

    return (
        ex13_3 = m13_3,
        ex13_4 = m13_4,
        ex13_5 = m13_5,
        ex13_6 = m13_6,
        ex13_7 = m13_7,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_all_examples()
end
