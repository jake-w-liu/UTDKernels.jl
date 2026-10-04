using Test, UTDKernels

@testset "Wedge prefactor preserves representable extreme-scale boundary fields" begin
    wedge = Wedge(1.5pi)
    angles = RayAngles(5pi/4, pi/4)
    material = WedgeFaceMaterial(5.31 - 0.3im)
    impedance = ImpedanceWedge(1.5pi, material)
    # Only the second KP term has a pole in this geometry. Its coefficient
    # is -sqrt(Li)/2 for both PEC channels and both Holm channels (M2=1).
    # All other terms are O(k^-1/2); here their relative contribution is
    # below 1e-250. The unscaled cot*F product need not fit in Float64.
    for (k, Li) in ((1e250, 1e300), (1e300, 1e308), (1e308, 1e308))
        expected = -sqrt(Li)/2
        for evaluate in (
            () -> pec_wedge_DsDh(wedge, angles, k, Li),
            () -> pec_wedge_DsDh(wedge, angles, k, Li, 0.7, 2.1),
            () -> impedance_wedge_DsDh(impedance, angles, k, Li),
            () -> impedance_wedge_DsDh(impedance, angles, k, Li, 0.7, 2.1),
        )
            @test all(isapprox.(evaluate(), expected; rtol=4e-15, atol=0))
        end
    end
end
