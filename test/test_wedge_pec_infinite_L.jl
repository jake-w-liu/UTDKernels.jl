using Test
using UTDKernels

@testset "Infinite-distance L -> Inf limit" begin
    w = Wedge(3π / 2)
    k = 7.0
    Lbig = 1e12

    # Choose angles away from cotangent poles so the finite-L large-argument
    # and exact L=Inf limits should coincide closely.
    for (phi, phip) in ((0.39, 0.21), (0.87, 0.33), (1.45, 0.74), (2.11, 0.58))
        ang = RayAngles(phi, phip)

        Ds_inf, Dh_inf = pec_wedge_DsDh(w, ang, k, Inf)
        Ds_big, Dh_big = pec_wedge_DsDh(w, ang, k, Lbig)

        @test isfinite(real(Ds_inf)) && isfinite(imag(Ds_inf))
        @test isfinite(real(Dh_inf)) && isfinite(imag(Dh_inf))

        @test isapprox(Ds_inf, Ds_big; rtol = 1e-6, atol = 1e-9)
        @test isapprox(Dh_inf, Dh_big; rtol = 1e-6, atol = 1e-9)
    end

    ang = RayAngles(0.93, 0.47)
    Ds0, Dh0 = pec_wedge_DsDh(w, ang, k, Inf, Inf, Inf)
    Ds1, Dh1 = pec_wedge_DsDh(w, ang, k, Inf)
    @test isapprox(Ds0, Ds1; rtol = 1e-12, atol = 1e-12)
    @test isapprox(Dh0, Dh1; rtol = 1e-12, atol = 1e-12)
end

