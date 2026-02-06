using Test
using UTDKernels

@testset "Symmetry and invariance" begin

    @testset "Angle wrapping invariance" begin
        w = Wedge(2π)
        k = 10.0
        L = 1.0
        phip = π/3

        phi_base = π/2
        ang1 = RayAngles(phi_base, phip)
        Ds1, Dh1 = pec_wedge_DsDh(w, ang1, k, L)

        # Adding multiples of alpha should give same result (after wrapping)
        ang2 = RayAngles(phi_base + 2π, phip)
        Ds2, Dh2 = pec_wedge_DsDh(w, ang2, k, L)

        @test abs(Ds1 - Ds2) < 1e-12
        @test abs(Dh1 - Dh2) < 1e-12
    end

    @testset "Reciprocity: D(φ,φ') related to D(φ',φ)" begin
        # For a PEC wedge: D_s(φ,φ') = D_s(φ',φ) and D_h(φ,φ') = D_h(φ',φ)
        # (reciprocity of diffraction coefficients)
        w = Wedge(3π/2)
        k = 15.0
        L = 2.0

        for (phi, phip) in [(π/4, π/3), (π/2, π/6), (1.0, 2.0)]
            ang_fwd = RayAngles(phi, phip)
            ang_rev = RayAngles(phip, phi)

            Ds_fwd, Dh_fwd = pec_wedge_DsDh(w, ang_fwd, k, L)
            Ds_rev, Dh_rev = pec_wedge_DsDh(w, ang_rev, k, L)

            @test abs(Ds_fwd - Ds_rev) < 1e-10
            @test abs(Dh_fwd - Dh_rev) < 1e-10
        end
    end

    @testset "Effective distance L" begin
        d1 = Distances(1.0, 1.0)
        @test effective_L(d1) ≈ 0.5

        d2 = Distances(2.0, Inf)
        @test effective_L(d2) ≈ 2.0

        d3 = Distances(1.0, 3.0)
        @test effective_L(d3) ≈ 0.75
    end

    @testset "Spreading factor" begin
        @test spreading_factor(1.0, Inf) ≈ 1.0
        @test spreading_factor(4.0, Inf) ≈ 0.5
        @test spreading_factor(1.0, 1.0) ≈ sqrt(1.0 / (1.0 * 2.0))
    end
end
