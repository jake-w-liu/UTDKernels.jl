using Test
using UTDKernels

@testset "Interior Wedge (alpha < pi)" begin
    
    @testset "90° interior wedge (n=0.5, alpha=pi/2)" begin
        # For n=0.5, the cotangent terms exactly cancel out regardless of F.
        # This corresponds to a 90° concave corner. 
        # Balanis WDC.m explicitly zeroes this case.
        w = Wedge(π/2)
        n = wedge_n(w)
        @test n ≈ 0.5
        
        k = 10.0
        L = 1.0
        phip = π/6
        phi = π/3
        
        Ds, Dh = pec_wedge_DsDh(w, RayAngles(phi, phip), k, L)
        
        # Should be effectively zero
        @test abs(Ds) < 1e-14
        @test abs(Dh) < 1e-14
    end

    @testset "Interior wedge n=0.75 (alpha=135°)" begin
        # Test n=0.75 case which is in the WDC reference data.
        # Exterior angle = 135 deg.
        w = Wedge(0.75 * π)
        k = 10.0
        L = 1.0
        phip = deg2rad(15.0)
        phi = deg2rad(45.0)
        
        Ds, Dh = pec_wedge_DsDh(w, RayAngles(phi, phip), k, L)
        
        @test isfinite(Ds)
        @test isfinite(Dh)
        
        # Verify reciprocity D(phi, phip) = D(phip, phi)
        Ds_rev, Dh_rev = pec_wedge_DsDh(w, RayAngles(phip, phi), k, L)
        @test Ds ≈ Ds_rev
        @test Dh ≈ Dh_rev
    end

    @testset "Flat plate n=1.0 (alpha=pi) recovery" begin
        # For n=1, cot(psi_1) + cot(psi_2) = cot((pi+beta)/2) + cot((pi-beta)/2) = 0
        # Thus D = 0.
        w = Wedge(π)
        k = 10.0
        L = 1.0
        ang = RayAngles(π/4, π/3)
        Ds, Dh = pec_wedge_DsDh(w, ang, k, L)
        @test abs(Ds) < 1e-14
        @test abs(Dh) < 1e-14
    end

end
