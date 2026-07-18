using Test
using UTDKernels

@testset "Extreme-scale and fail-closed robustness" begin
    @testset "transition and wedge GTD limits remain finite" begin
        @test isapprox(F_utd(1e308), 1.0; atol=2eps(Float64), rtol=0)
        @test F_utd(Inf) == 1.0 + 0.0im
        @test_throws DomainError F_utd(NaN)
        @test_throws DomainError F_utd(complex(Inf, 0.0))

        w = Wedge(1.5π)
        ang = RayAngles(1.2, 0.7)
        D_finite = pec_wedge_DsDh(w, ang, 2.0, 1e308)
        D_limit = pec_wedge_DsDh(w, ang, 2.0, Inf)
        @test all(isfinite, (real(D_finite[1]), imag(D_finite[1]),
                             real(D_finite[2]), imag(D_finite[2])))
        @test isapprox(D_finite[1], D_limit[1]; atol=2e-15, rtol=2e-15)
        @test isapprox(D_finite[2], D_limit[2]; atol=2e-15, rtol=2e-15)
    end

    @testset "effective distance and spreading avoid overflow" begin
        large = floatmax(Float64) / 4
        @test effective_L(Distances(large, large)) == large / 2
        @test effective_L(Distances(Inf, 2.0)) == 2.0
        @test effective_L(Distances(2, Inf)) == 2.0

        A = spreading_factor(large, large)
        @test isfinite(A) && A > 0
        @test isapprox(A, sqrt(0.5 / large); rtol=4eps(Float64), atol=0)
        @test isapprox(spreading_factor(1e200, 1e-200), 1e-300;
                       rtol=4eps(Float64), atol=0)
        @test spreading_factor(Inf, 2.0) == 0.0

        @test_throws DomainError Distances(0.0, 1.0)
        @test_throws DomainError Distances(1.0, -1.0)
        @test_throws DomainError spreading_factor(0.0, Inf)
        @test_throws DomainError spreading_factor(1.0, 0.0)
    end

    @testset "matched media have zero Fresnel reflection" begin
        for ψ in (0.0, eps(Float64), 1e-12, 1e-8, π / 4, π / 2)
            @test fresnel_te(ψ, 1.0) == 0.0 + 0.0im
            @test fresnel_tm(ψ, 1.0) == 0.0 + 0.0im
        end
        iw = ImpedanceWedge(1.5π, WedgeFaceMaterial(1.0 + 0.0im))
        Ds, Dh = impedance_wedge_DsDh(iw, RayAngles(1.0, 0.0), 2.0, 3.0)
        @test isfinite(real(Ds)) && isfinite(imag(Ds))
        @test isfinite(real(Dh)) && isfinite(imag(Dh))
    end

    @testset "invalid public inputs fail before numerical work" begin
        @test_throws DomainError PhasorConvention(0)
        @test_throws DomainError wrap_angle(1.0, 0.0)
        @test_throws DomainError wrap_angle(Inf, 2π)
        @test_throws DomainError WedgeFaceMaterial(2.0, 1.0, 0.0)
        @test_throws DomainError WedgeFaceMaterial(2.0, -1.0, 1.0e9)
        @test_throws DomainError WedgeFaceMaterial(complex(NaN, 0.0))

        w = Wedge(1.5π)
        ang = RayAngles(1.2, 0.7)
        @test_throws DomainError wedge_transition_args(w, ang, 2.0, -3.0)
        @test_throws DomainError wedge_transition_args(w, ang, 2.0, 3.0; tol=-1.0)

        @test_throws DomainError psi_Phi(0.0, 0.0)
        @test_throws DomainError psi_Phi(Inf, 3π / 4)
        @test_throws DomainError psi_Phi(0.0, 3π / 4; rtol=0.0)
        @test_throws DomainError maliuzhinets_DsDh(
            1.5π, NaN, 2.0, 1.0, 0.7, 2π)
    end
end
