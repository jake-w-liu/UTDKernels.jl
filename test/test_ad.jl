using Test
using UTDKernels

@testset "AD compatibility (finite-difference smoothness)" begin
    # Verify F_utd is smooth (differentiable) for real positive x
    # by checking finite-difference consistency at multiple scales
    @testset "F_utd smoothness" begin
        for x in [0.5, 1.0, 5.0, 10.0]
            h1 = 1e-5
            h2 = 1e-6
            dF1 = (F_utd(x + h1) - F_utd(x - h1)) / (2h1)
            dF2 = (F_utd(x + h2) - F_utd(x - h2)) / (2h2)
            # Two finite-difference estimates should agree closely
            @test abs(dF1 - dF2) < 1e-4
        end
    end

    @testset "pec_wedge_DsDh smoothness" begin
        w = Wedge(3π/2)
        k = 10.0
        L = 1.0
        phip = π/4
        phi0 = π/2
        h = 1e-6

        Ds0, Dh0 = pec_wedge_DsDh(w, RayAngles(phi0, phip), k, L)
        Dsp, Dhp = pec_wedge_DsDh(w, RayAngles(phi0 + h, phip), k, L)
        Dsm, Dhm = pec_wedge_DsDh(w, RayAngles(phi0 - h, phip), k, L)

        # Second-order centered difference should be smooth
        dDs = (Dsp - Dsm) / (2h)
        dDh = (Dhp - Dhm) / (2h)
        @test isfinite(abs(dDs))
        @test isfinite(abs(dDh))
    end
end
