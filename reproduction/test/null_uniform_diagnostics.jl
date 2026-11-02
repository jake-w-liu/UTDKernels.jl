module NullUniformDiagnosticRegressions

using Test, UTDKernels, JSON, QuadGK
using ..StudyArtifacts: allowed_nonfinite
include(joinpath(@__DIR__, "..", "studies", "null-uniform", "diagnostics.jl"))

const NUM = UTDKernels.NullUniform

@testset "Separated diagnostic domain" begin
    model(poles) = (; poles)
    finite = model([0.1 + 0.2im, -0.3 + 0.4im])
    for z in (0.0 + 0.0im, complex(-0.0, -0.0), ComplexF32(0, 0))
        singular = model([z, one(z)])
        @test _separated_slope_singular(singular, finite)
        @test _separated_slope_singular(singular, singular)
        @test_throws DomainError _separated_slope_singular(finite, singular)
    end
    for z in (1e-12 + 0im, 1e-12im, 1e-12 + 1e-12im, -1e-12 - 1e-12im)
        adjacent = model([z, one(z)])
        @test !_separated_slope_singular(adjacent, finite)
        @test !_separated_slope_singular(finite, adjacent)
        @test isfinite(NUM.canonical_slope_separated(adjacent.poles, [1.0, 1.0]))
    end
    # Classify even the nearest nonzero coordinates exactly; their outer
    # comparator can overflow, so this checks the domain guard alone.
    for z in (complex(nextfloat(0.0), 0.0), complex(0.0, nextfloat(0.0)),
              complex(nextfloat(0f0), 0f0))
        adjacent = model([z, one(z)])
        @test !_separated_slope_singular(adjacent, finite)
        @test !_separated_slope_singular(finite, adjacent)
    end
    @test !_separated_slope_singular(finite, finite)
end

@testset "Exact recorded-pole exclusions" begin
    coverage = JSON.parsefile(joinpath(@__DIR__, "..", "studies", "null-uniform", "coverage.json"))
    for name in ("wedge_diagnostics.csv", "wedge_halfplane.csv")
        rules = coverage["numeric_exceptions"][name]
        for (z1, z2, allowed) in (((0.0, 0.0), (1.0, 2.0), true),
                                 ((1.0, 2.0), (0.0, 0.0), true),
                                 ((0.0, 1.0), (1.0, 0.0), false),
                                 ((1e-300, 0.0), (0.0, 1e-300), false))
            row = (; z1_re_nd=string(z1[1]), z1_im_nd=string(z1[2]),
                     z2_re_nd=string(z2[1]), z2_im_nd=string(z2[2]))
            @test allowed_nonfinite(NaN, "slope_separated_error_nd", row, rules) == allowed
            @test !allowed_nonfinite(Inf, "slope_separated_error_nd", row, rules)
            @test !allowed_nonfinite(-Inf, "slope_separated_error_nd", row, rules)
            @test !allowed_nonfinite(NaN, "two_moment_error_nd", row, rules)
        end
    end
end

@testset "Boundary table notation" begin
    @test _separated_boundary_latex(NaN) == raw"\text{undefined}"
    @test _separated_boundary_latex(0.0) == "0"
    @test _separated_boundary_latex(-0.25) == raw"-2.50\times 10^{-1}"
    for value in (NaN, Inf, -Inf)
        @test_throws ArgumentError latex_scientific(value)
    end
    for value in (Inf, -Inf)
        @test_throws ArgumentError _separated_boundary_latex(value)
    end
end

@testset "Independent separated-slope normalization" begin
    gaussian_moment, error = quadgk(t -> t^2 * exp(-t^2), -Inf, 0.0, Inf; rtol=1e-13)
    @test error < 1e-12
    @test gaussian_moment ≈ sqrt(pi) / 2 rtol=2e-15
    for z in (0.01 + 0im, 0.01 + 0.02im, -0.01 + 0.02im)
        value = NUM.canonical_slope_separated([z], [0.0, 1.0])
        @test value * z^2 ≈ im * gaussian_moment / pi rtol=2e-15
        @test abs(NUM.canonical_slope_separated([z / 2], [0.0, 1.0])) ≈ 4abs(value) rtol=2e-15
    end
    @test NUM.moment_int_plane(0.0 + 0im, 1)[2] ≈ -im / sqrt(pi)
end

end
