using Test
using UTDKernels

@testset "Extreme-scale and fail-closed robustness" begin
    @testset "transition and wedge GTD limits remain finite" begin
        @test isapprox(F_utd(1e308), 1.0; atol=2eps(Float64), rtol=0)
        @test F_utd(Inf) == 1.0 + 0.0im
        @test_throws DomainError F_utd(NaN)
        @test_throws DomainError F_utd(complex(Inf, 0.0))

        for x in (big"1.0", Complex{BigFloat}(1, 1))
            err = try
                F_utd(x)
                nothing
            catch caught
                caught
            end
            @test err isa ArgumentError
            @test occursin("erfcx is unavailable", sprint(showerror, err))
        end

        big_err = try
            pec_wedge_DsDh(
                Wedge(big"1.5" * big(π)),
                RayAngles(big"1.2", big"0.7"),
                big"2.0",
                big"3.0",
            )
            nothing
        catch caught
            caught
        end
        @test big_err isa ArgumentError
        @test occursin("erfcx is unavailable", sprint(showerror, big_err))

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

        # Infinitely distant observer: the diffracted amplitude vanishes, so the
        # applied dyadic must return a clean zero, not NaN from exp(-i k Inf).
        es, eh = pec_wedge_apply_sh(1.0 + 0im, 2.0 + 0im, 1.0 + 0im, 1.0 + 0im, 2π, Inf, 5.0)
        @test es == 0 && eh == 0
        @test isfinite(real(es)) && isfinite(imag(es)) && isfinite(real(eh)) && isfinite(imag(eh))

        # Applying the exact zero spreading factor must happen before products of
        # otherwise valid amplitudes can overflow. A late `* 0` produces NaN.
        huge = floatmax(Float64) / 2
        es_huge, eh_huge = pec_wedge_apply_sh(
            huge, -huge, huge, huge, 2π, Inf, 5.0,
        )
        @test es_huge == 0.0 + 0.0im
        @test eh_huge == 0.0 + 0.0im

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

    @testset "material loss ratio avoids intermediate overflow and underflow" begin
        for (sigma, freq) in ((1.0, 3.0e307), (1.0e-300, 1.0e30))
            mat = WedgeFaceMaterial(2.0, sigma, freq)
            reference_loss = setprecision(BigFloat, 256) do
                -BigFloat(sigma) /
                (2 * BigFloat(π) * BigFloat(freq) * BigFloat("8.854187817e-12"))
            end
            expected = Float64(reference_loss)
            @test expected != 0.0
            @test imag(mat.eps_r) ≈ expected rtol=2eps(Float64) atol=0
        end
    end

    @testset "non-finite coefficients fail closed" begin
        @test_throws DomainError fresnel_te(NaN, 4.0)
        @test_throws DomainError fresnel_te(0.5, complex(4.0, Inf))
        @test_throws DomainError fresnel_tm(0.5, Inf)
        @test_throws DomainError fresnel_tm(complex(0.5, NaN), 4.0)
        @test_throws DomainError fresnel_tm(π / 2, 0.0)

        w = Wedge(1.5π)
        ang = RayAngles(1.2, 0.7)
        @test_throws DomainError pec_wedge_DsDh(
            w, ang, 2.0, 1.0, 1.0, 1.0; Rs=Inf,
        )
        @test_throws DomainError pec_wedge_DsDh(
            w, ang, 2.0, 1.0, 1.0, 1.0; Rh=complex(1.0, NaN),
        )
        @test_throws DomainError pec_wedge_apply_sh(
            NaN, 1.0, 1.0, 1.0, 2.0, 1.0, Inf,
        )
        @test_throws DomainError pec_wedge_apply_sh(
            1.0, 1.0, 1.0, 1.0, 1.0 + 1.0im, 1e308, Inf,
        )
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

        for alpha in (nextfloat(0.0), 1.0e-300)
            tiny_wedge = Wedge(alpha)
            tiny_angles = RayAngles(0.0, 0.0)
            @test_throws DomainError pec_wedge_DsDh(tiny_wedge, tiny_angles, 2.0, 3.0)
            @test_throws DomainError wedge_transition_args(tiny_wedge, tiny_angles, 2.0, 3.0)
        end

        @test_throws DomainError psi_Phi(0.0, 0.0)
        @test_throws DomainError psi_Phi(Inf, 3π / 4)
        @test_throws DomainError psi_Phi(0.0, 3π / 4; rtol=0.0)
        @test_throws DomainError psi_Phi(2.0, nextfloat(0.0))
        @test_throws DomainError psi_Phi(2.0, 1.0e-12)
        @test_throws DomainError maliuzhinets_DsDh(
            1.5π, NaN, 2.0, 1.0, 0.7, 2π)
        @test_throws DomainError maliuzhinets_DsDh(
            1.5π, 0.0, 2.0, 1.0, 0.7, 2π)
        @test_throws DomainError maliuzhinets_DsDh(
            1.5π, 2.0, 0.0 + 0.0im, 1.0, 0.7, 2π)
    end
end
