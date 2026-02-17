using Test
using UTDKernels

@testset "Generalized Li/Lro/Lrn wedge API" begin
    w = Wedge(3π / 2)
    k = 12.0
    L = 0.73

    for phi in (0.31, 0.77, 1.14, 1.92, 2.41)
        for phip in (0.25, 0.64, 1.33)
            ang = RayAngles(phi, phip)
            Ds0, Dh0 = pec_wedge_DsDh(w, ang, k, L)
            Ds1, Dh1 = pec_wedge_DsDh(w, ang, k, L, L, L)
            @test isapprox(Ds0, Ds1; rtol = 1e-12, atol = 1e-12)
            @test isapprox(Dh0, Dh1; rtol = 1e-12, atol = 1e-12)
        end
    end

    ang = RayAngles(0.9, 0.45)
    @test_throws DomainError pec_wedge_DsDh(w, ang, k, 0.0, 1.0, 1.0)
    @test_throws DomainError pec_wedge_DsDh(w, ang, k, 1.0, -1.0, 1.0)
    @test_throws DomainError pec_wedge_DsDh(w, ang, k, 1.0, 1.0, 0.0)
end

