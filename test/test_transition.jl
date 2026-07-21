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
            leading = sqrt(π * x) * exp(+im * π/4)
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
        # Verify via the known identity: F(x) = √(πx)·e^{+i(π/4+x)}·erfc(e^{+iπ/4}√x)
        using SpecialFunctions: erfc
        for x in [0.5, 1.0, 2.0, 5.0, 10.0]
            sqx = sqrt(Complex(x))
            z = exp(+im * π/4) * sqx
            ref = sqrt(π * Complex(x)) * exp(+im * (π/4 + x)) * erfc(z)
            val = F_utd(x)
            @test abs(val - ref) < 1e-10
        end
    end

    @testset "F1-2: erfcx form exact at small x (no √eps surrogate)" begin
        # Regression: F_utd previously replaced the erfcx form with the leading-order
        # surrogate √(πx)·e^{+iπ/4} for |x| < √eps. That surrogate has relative error
        # (2/√π)√x = O(√x) — ≈1.4e-4 at |x|=√eps, ≈3.6e-5 at x=1e-9 — and produced a
        # ≈1.38e-4 jump discontinuity at the floor. The erfcx form is exact at x=0
        # (√0·erfcx(0)=0) and accurate to machine precision for arbitrarily small x.
        using SpecialFunctions: erfcx
        erfcx_form(x) = (sq = sqrt(Complex(x));
                         sqrt(π) * sq * exp(+im * π/4) * erfcx(exp(+im * π/4) * sq))

        # F(0) = 0 exactly (erfcx(0)=1, √0·1=0).
        @test F_utd(0.0) == 0.0 + 0.0im

        # Continuity across the former √eps floor: prevfloat/nextfloat agree to
        # machine precision. Threshold 1e-10 ≫ new gap (~1.8e-16) yet ≪ old jump
        # (1.38e-4), so this fails on the removed surrogate.
        fl = sqrt(eps(Float64))
        Flo = F_utd(prevfloat(fl)); Fhi = F_utd(nextfloat(fl))
        @test abs(Flo - Fhi) / abs(Fhi) < 1e-10

        # Agreement with the direct erfcx form near machine precision at tiny x.
        # rtol 1e-13 lies far below the O(√x) surrogate error (3.6e-5 / 1.1e-6) so
        # it fails on the old branch and passes with the erfcx form (rel err ~1e-16).
        for x in (1e-9, 1e-12)
            ref = erfcx_form(x)
            @test abs(F_utd(x) - ref) / abs(ref) < 1e-13
        end
    end
end
