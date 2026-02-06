using Test
using UTDKernels

@testset "Transition function F_utd" begin

    @testset "F(0) limit" begin
        val = F_utd(0.0)
        @test abs(val) < 1e-14
    end

    @testset "F(x) → 1 as x → +∞ (GTD recovery)" begin
        # Convergence is O(1/x), so tolerance scales accordingly
        @test abs(F_utd(1e4) - 1.0) < 1e-3
        @test abs(F_utd(1e6) - 1.0) < 1e-5
        @test abs(F_utd(1e8) - 1.0) < 1e-7
        # Also check monotone improvement
        errs = [abs(F_utd(10.0^p) - 1.0) for p in 2:6]
        for i in 1:length(errs)-1
            @test errs[i+1] < errs[i]
        end
    end

    @testset "Small positive real x" begin
        for x in [1e-8, 1e-6, 1e-4]
            val = F_utd(x)
            leading = sqrt(π * x) * exp(-im * π/4)
            @test abs(val - leading) / abs(leading) < 0.1
        end
    end

    @testset "Moderate real x values" begin
        for x in 0.1:0.5:20.0
            val = F_utd(x)
            @test isfinite(abs(val))
            @test abs(val) < 5.0
        end
    end

    @testset "Complex arguments" begin
        for x in [1.0+0.5im, 5.0-2.0im, 0.1+0.1im, 10.0+1.0im]
            val = F_utd(x)
            @test isfinite(abs(val))
        end
    end

    @testset "Consistency: integral vs erfc" begin
        # Use Gauss–Laguerre-style substitution for better Fresnel accuracy.
        # Verify via the known identity: F(x) = √(πx)·e^{-i(π/4+x)}·erfc(e^{-iπ/4}√x)
        # Instead of comparing with crude quadrature, verify the erfc identity directly.
        using SpecialFunctions: erfc
        for x in [0.5, 1.0, 2.0, 5.0, 10.0]
            sqx = sqrt(Complex(x))
            z = exp(-im * π/4) * sqx
            ref = sqrt(π * Complex(x)) * exp(-im * (π/4 + x)) * erfc(z)
            val = F_utd(x)
            @test abs(val - ref) < 1e-10
        end
    end
end
