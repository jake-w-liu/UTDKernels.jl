using Test
using UTDKernels
using ForwardDiff

@testset "Extreme-scale and fail-closed robustness" begin
    @testset "angular domain endpoints follow the active scalar type" begin
        material = WedgeFaceMaterial(2.0 + 0.0im)
        for T in (Float16, Float32, Float64)
            alpha_max = T(2) * T(π)
            @test Wedge(alpha_max).alpha === alpha_max
            @test ImpedanceWedge(alpha_max, material).alpha === alpha_max
            @test_throws DomainError Wedge(nextfloat(alpha_max))
            @test_throws DomainError ImpedanceWedge(nextfloat(alpha_max), material)
        end

        for bit_precision in (32, 64, 128, 256, 512)
            alpha_big = setprecision(BigFloat, bit_precision) do
                2 * BigFloat(π)
            end
            @test Wedge(alpha_big).alpha == alpha_big
            @test ImpedanceWedge(alpha_big, material).alpha == alpha_big
            @test_throws DomainError Wedge(nextfloat(alpha_big))
            @test_throws DomainError ImpedanceWedge(nextfloat(alpha_big), material)
        end

        alpha_dual = ForwardDiff.Dual(Float32(2) * Float32(π), Float32(1))
        @test Wedge(alpha_dual).alpha === alpha_dual
        @test ImpedanceWedge(alpha_dual, material).alpha === alpha_dual

        # The grazing certificate includes the half-plane endpoint but excludes
        # the nominal flat-face boundary unless interior continuation is asked
        # for explicitly.
        half_plane_32 = Wedge(Float32(2) * Float32(π))
        half_plane_report = grazing_interval_report(
            half_plane_32,
            RayAngles(Float32(1.7), Float32(1e-3)),
            Float32(100),
            Float32(1),
        )
        @test half_plane_report.valid

        flat_32 = Wedge(Float32(π))
        flat_report = grazing_interval_report(
            flat_32,
            RayAngles(Float32(1.7), Float32(1e-3)),
            Float32(100),
            Float32(1),
        )
        @test !flat_report.valid
        @test occursin("exterior wedge", flat_report.reason)

        # Maliuzhinets' exact solver has strict π and 2π bounds. A rounded
        # Float32 representation of π is still the boundary, not an interior
        # angle slightly above the Float64 irrational constant.
        @test_throws DomainError maliuzhinets_DsDh(
            Float32(π), 2.0 + 0.0im, 2.0 + 0.0im,
            Float32(1), Float32(0.5), Float32(2);
            rtol=Float32(1e-5),
        )
        @test_throws DomainError maliuzhinets_DsDh(
            Float32(2) * Float32(π), 2.0 + 0.0im, 2.0 + 0.0im,
            Float32(1), Float32(0.5), Float32(2);
            rtol=Float32(1e-5),
        )
    end

    @testset "ForwardDiff payloads do not change physical-domain guards" begin
        zero_positive_tangent = ForwardDiff.Dual(0.0, 1.0)
        zero_negative_tangent = ForwardDiff.Dual(0.0, -1.0)
        wedge = Wedge(1.5π)
        angles = RayAngles(1.7, 1e-3)

        # Strictly positive physical inputs remain invalid at a zero primal
        # value, regardless of the derivative seed direction.
        @test_throws DomainError Distances(zero_positive_tangent, 1.0)
        @test_throws DomainError wrap_angle(1.0, zero_positive_tangent)
        @test_throws DomainError F_utd_prime(zero_positive_tangent)
        @test_throws DomainError pec_wedge_DsDh(
            wedge, angles, zero_positive_tangent, 1.0,
        )
        @test_throws DomainError pec_wedge_DsDh(
            wedge, angles, 2.0, zero_positive_tangent,
        )
        @test_throws DomainError WedgeFaceMaterial(
            2.0, 0.0, zero_positive_tangent,
        )
        @test_throws DomainError psi_Phi(0.2, zero_positive_tangent)
        @test_throws DomainError psi_Phi(
            0.2, 1.0; rtol=zero_positive_tangent,
        )

        # Nonnegative controls at a zero primal value remain admissible even
        # when their tangent points outside the one-sided physical domain.
        material = WedgeFaceMaterial(2.0, zero_negative_tangent, 1.0)
        @test ForwardDiff.value(imag(material.eps_r)) == 0.0
        regimes = wedge_transition_args(
            wedge, angles, 2.0, 1.0; tol=zero_negative_tangent,
        )
        @test all(regime -> regime isa Symbol, regimes.regime)
        @test UTDKernels._valid_grazing_margin(zero_negative_tangent)
        report = grazing_interval_report(
            wedge,
            angles,
            20.0,
            1.0;
            transition_margin=zero_negative_tangent,
            x_margin=zero_negative_tangent,
            branch_margin=zero_negative_tangent,
            gprime_reltol=zero_negative_tangent,
        )
        @test !occursin("must be nonnegative", report.reason)
        routed = wedge_DsDh(
            wedge, angles, 2.0, 1.0;
            grazing_switch=zero_negative_tangent,
        )
        baseline = pec_wedge_DsDh(wedge, angles, 2.0, 1.0)
        @test routed[1] ≈ baseline[1] rtol=2e-12 atol=0
        @test routed[2] ≈ baseline[2] rtol=2e-12 atol=0

        # Fixed crossover controls are compared by primal value. Their Dual
        # payload must not turn an exact supported setting into a domain error.
        x60 = ForwardDiff.Dual(60.0, 1.0)
        threshold60 = ForwardDiff.Dual(60.0, -2.0)
        @test isfinite(real(F_utd_minus_one(x60; threshold=threshold60)))
        x35 = ForwardDiff.Dual(35.0, 1.0)
        threshold35 = ForwardDiff.Dual(35.0, -2.0)
        @test isfinite(real(F_utd_prime(x35; asymptotic_threshold=threshold35)))
    end

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

        # k*L underflows here, but sqrt(kL) and the final coefficient are both
        # representable. Compare with the independent small-x leading limit in
        # which the prefactor's sqrt(k) cancels analytically.
        tiny = 1.0e-300
        terms = UTDKernels.kp_four_terms(1.2, 0.7, wedge_n(w))
        scale = -sqrt(tiny) / (2 * wedge_n(w) * sqrt(2.0))
        leading = ntuple(4) do j
            scale * cot(terms.psi[j]) * sqrt(terms.aj[j])
        end
        Ds_ref = sum(UTDKernels.PEC_SIGMA_SOFT[j] * leading[j] for j in 1:4)
        Dh_ref = sum(leading)
        Ds_tiny, Dh_tiny = pec_wedge_DsDh(w, ang, tiny, tiny)
        @test Ds_tiny ≈ Ds_ref rtol=2e-14 atol=0
        @test Dh_tiny ≈ Dh_ref rtol=2e-14 atol=0
        @test wedge_DsDh(w, ang, tiny, tiny) == (Ds_tiny, Dh_tiny)

        # The exported branch-local helpers use the same positive inputs. Their
        # √(kLa)-scale values remain representable even though k*L underflows.
        beta = 1.2
        n = wedge_n(w)
        local_terms = UTDKernels.kp_four_terms(beta, 0.0, n)
        root_kL = sqrt(tiny) * sqrt(tiny)
        phase = cis(π / 4)
        G_leading = sqrt(π) * root_kL * phase * sum(
            cot(local_terms.psi[j]) * sqrt(local_terms.aj[j]) for j in 1:2
        )
        Gp_leading = sqrt(π) * root_kL * phase * sum(
            begin
                sigma = j == 1 ? +1 : -1
                psi = local_terms.psi[j]
                a = local_terms.aj[j]
                N = local_terms.Nj[j]
                u = 2n * π * N - beta
                dpsi = sigma / (2n)
                -dpsi * sqrt(a) / sin(psi)^2 +
                    cot(psi) * sin(u) / (2sqrt(a))
            end for j in 1:2
        )
        @test two_term_kernel(beta, w, tiny, tiny) ≈ G_leading rtol=2e-14 atol=0
        @test two_term_kernel_derivative(beta, w, tiny, tiny) ≈
              Gp_leading rtol=2e-14 atol=0

        # Ordinary inputs retain the established direct branch-local arithmetic;
        # the scaled path is an underflow repair, not a baseline rewrite.
        beta_ordinary = 0.91
        ordinary_terms = UTDKernels.kp_four_terms(beta_ordinary, 0.0, n)
        ordinary_reference = sum(
            (cos(ordinary_terms.psi[j]) / sin(ordinary_terms.psi[j])) *
            F_utd(2.0 * 3.0 * ordinary_terms.aj[j]) for j in 1:2
        )
        @test two_term_kernel(beta_ordinary, w, 2.0, 3.0) == ordinary_reference
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

        # A small finite spreading factor must be applied without a transient
        # overflow in the two input amplitudes.
        scaled = pec_wedge_apply_sh(
            1.0e200, 1.0e200, 1.0e200, 1.0e200,
            1.0e-200, 1.0e200, 1.0e-200,
        )
        scaled_ref = (1.0e200 * spreading_factor(1.0e200, 1.0e-200)) *
                     1.0e200 * cis(-1.0)
        @test scaled[1] ≈ scaled_ref rtol=8eps(Float64) atol=0
        @test scaled[2] ≈ scaled_ref rtol=8eps(Float64) atol=0

        # Even when the standalone binary64 spreading factor underflows, the
        # combined field can remain representable.
        s_extreme = 1.0e308
        sp_extreme = 1.0e-308
        @test spreading_factor(s_extreme, sp_extreme) == 0.0
        recovered = pec_wedge_apply_sh(
            1.0e200, 1.0e200, 1.0e200, 1.0e200,
            1.0e-308, s_extreme, sp_extreme,
        )
        recovered_ref = setprecision(BigFloat, 256) do
            s_big = BigFloat(s_extreme)
            sp_big = BigFloat(sp_extreme)
            A_big = sqrt(sp_big / (s_big * (s_big + sp_big)))
            ComplexF64(BigFloat(1.0e200)^2 * A_big * cis(-BigFloat(1.0)))
        end
        @test recovered[1] ≈ recovered_ref rtol=2e-14 atol=0
        @test recovered[2] ≈ recovered_ref rtol=2e-14 atol=0

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

    @testset "epsilon-near-zero Fresnel coefficients keep the normal-incidence branch" begin
        function tm_big_reference(psi, eps_r)
            return setprecision(BigFloat, 256) do
                ψ = BigFloat(psi)
                ε = BigFloat(eps_r)
                η = sqrt(complex(ε - cos(ψ)^2))
                ComplexF64((ε * sin(ψ) - η) / (ε * sin(ψ) + η))
            end
        end

        for eps_r in (1.0e-17, 1.0e-20, 1.0e-100)
            expected = tm_big_reference(π / 2, eps_r)
            @test fresnel_tm(π / 2, eps_r) ≈ expected rtol=8eps(Float64) atol=0
        end
        for delta in (1.0e-8, 1.0e-6)
            psi = π / 2 - delta
            expected = tm_big_reference(psi, 1.0e-17)
            @test fresnel_tm(psi, 1.0e-17) ≈ expected rtol=2e-15 atol=0
        end
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

    @testset "complex diagnostics retain the transition-argument phase" begin
        w = Wedge(1.5π)
        ang = RayAngles(1.2, 0.7)
        k = 1.0 + 2.0im
        L = 1.3
        terms = UTDKernels.kp_four_terms(1.2, 0.7, wedge_n(w))
        expected_X1 = round(k * L * terms.aj[1]; digits=6)
        output = mktemp() do _, io
            redirect_stdout(io) do
                inspect_kp_terms(w, ang, k, L)
            end
            flush(io)
            seekstart(io)
            read(io, String)
        end
        @test occursin("X=$(expected_X1)", output)
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
