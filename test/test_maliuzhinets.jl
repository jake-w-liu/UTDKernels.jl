using Test
using UTDKernels

@testset "Maliuzhinets function ψ_Φ" begin
    Phi = 3π / 4  # 270° wedge

    @testset "ψ_Φ(0) = 1" begin
        @test psi_Phi(0.0, Phi) ≈ 1.0 atol = 1e-12
    end

    @testset "integer arguments use floating recurrence storage" begin
        reference_real = psi_Phi(10.0, 2.0)
        reference_complex = psi_Phi(10.0 + 2.0im, 2.0)

        @test psi_Phi(10, 2) isa ComplexF64
        @test psi_Phi(10, 2) ≈ reference_real rtol = 1e-13
        @test psi_Phi(10 + 2im, 2) isa ComplexF64
        @test psi_Phi(10 + 2im, 2) ≈ reference_complex rtol = 1e-13
    end

    @testset "Even symmetry: ψ_Φ(-w) = ψ_Φ(w)" begin
        for w in [0.5, 1.0, 2.0, 0.5 + 1.0im, 1.0 + 2.0im]
            @test psi_Phi(-w, Phi) ≈ psi_Phi(w, Phi) rtol = 1e-10
        end
    end

    @testset "Functional relation: ψ(w+2Φ)/ψ(w-2Φ) = cot(w/2+π/4)" begin
        for w in [0.3, 1.0, -0.5, 0.5 + 1.0im, 1.0 + 2.0im, 0.0 + 3.0im]
            lhs = psi_Phi(w + 2Phi, Phi) / psi_Phi(w - 2Phi, Phi)
            rhs = cot(w / 2 + π / 4)
            @test lhs ≈ rhs rtol = 1e-8
        end
    end

    @testset "convergence-strip edge uses the exact recurrence" begin
        strip = π / 2 + 2Phi
        for delta in (1.0e-4, 1.0e-6, 1.0e-8, 1.0e-10, 1.0e-12)
            w = strip - delta
            direct = psi_Phi(w, Phi)
            recurrence = cot((w - 2Phi) / 2 + π / 4) * psi_Phi(w - 4Phi, Phi)
            @test direct ≈ recurrence rtol=2e-12 atol=0
        end
    end

    @testset "unattainable quadrature tolerance fails closed" begin
        err = try
            psi_Phi(5.0 + 20.0im, Phi; rtol=1.0e-18)
            nothing
        catch caught
            caught
        end
        @test err isa DomainError
        @test occursin("could not meet the requested rtol", sprint(showerror, err))
    end

    @testset "Different wedge angles" begin
        for Phi_test in [π / 2, 3π / 4, 7π / 8]
            @test psi_Phi(0.0, Phi_test) ≈ 1.0 atol = 1e-12
            w = 0.5 + 0.5im
            @test psi_Phi(-w, Phi_test) ≈ psi_Phi(w, Phi_test) rtol = 1e-10
        end
    end

    @testset "F1-3: small-|w| accuracy (cancellation-free numerator)" begin
        # Regression for the integrand numerator cosh(wη)−1. The direct form loses
        # significant digits by catastrophic cancellation once |wη|²<eps, i.e. up to
        # η ≈ √eps/|w| which lies ABOVE the √eps integration floor whenever |w|<1;
        # this capped psi_Phi at ~6e-11 (w=0.1) / ~1.4e-10 (w=0.5) relative accuracy,
        # far short of the requested quadrature rtol=1e-12. The identity
        # cosh(x)−1 = 2 sinh(x/2)² removes the cancellation and restores full
        # accuracy. References below were computed ONCE by 256-bit BigFloat
        # Gauss-Kronrod quadrature of the SAME integral using the sinh identity
        # (quadrature abs-error on log ψ ≈ 5e-46 / 8e-45; provenance script:
        # scratchpad/bigfloat_ref.jl). rtol 1e-12 fails on the pre-fix cosh(a)−1
        # code (errors 6e-11 / 1.4e-10) and passes with the sinh form (rel err ~0).
        Phi = 3π / 4
        # ψ_Φ(w) reference digits (Float64-rounded from BigFloat):
        ref01 = 0.9997685174461999364639296678064968790456642
        ref05 = 0.9942122830624337748767979977145636360391567
        for (w, r) in ((0.1, ref01), (0.5, ref05))
            val = psi_Phi(w, Phi)
            @test abs(imag(val)) < 1e-14       # real w -> real ψ_Φ
            @test real(val) ≈ r rtol = 1e-12
        end
    end

    @testset "Higher-precision quadrature follows the input precision" begin
        value_256, relation_error_256 = setprecision(BigFloat, 256) do
            Phi_big = BigFloat(3) * BigFloat(π) / 4
            w = BigFloat("0.5")
            rtol = BigFloat("1e-22")
            value = psi_Phi(w, Phi_big; rtol)
            lhs = psi_Phi(w + 2Phi_big, Phi_big; rtol) /
                  psi_Phi(w - 2Phi_big, Phi_big; rtol)
            rhs = cot(w / 2 + BigFloat(π) / 4)
            value, abs(lhs - rhs) / abs(rhs)
        end
        value_384 = setprecision(BigFloat, 384) do
            Phi_big = BigFloat(3) * BigFloat(π) / 4
            psi_Phi(BigFloat("0.5"), Phi_big; rtol=BigFloat("1e-28"))
        end

        @test value_256 isa Complex{BigFloat}
        @test relation_error_256 < BigFloat("2e-21")
        @test abs(value_256 - value_384) / abs(value_384) < BigFloat("2e-21")
    end
end

@testset "Maliuzhinets exact: argument guards" begin
    eps_r = 10.0
    k = 2π
    alpha = 1.5π

    @test_throws DomainError maliuzhinets_DsDh(π, eps_r, eps_r, 0.4alpha, 0.3alpha, k)
    @test_throws DomainError maliuzhinets_DsDh(2π, eps_r, eps_r, 0.4alpha, 0.3alpha, k)
    @test_throws DomainError maliuzhinets_DsDh(alpha, eps_r, eps_r, -0.1, 0.3alpha, k)
    @test_throws DomainError maliuzhinets_DsDh(alpha, eps_r, eps_r, 0.4alpha, alpha + 0.1, k)
    @test_throws DomainError maliuzhinets_DsDh(alpha, eps_r, eps_r, 0.4alpha, 0.3alpha, 0.0)
    @test_throws DomainError maliuzhinets_DsDh(alpha, eps_r, eps_r, 0.4alpha, 0.3alpha, -1.0)
    @test_throws DomainError maliuzhinets_DsDh(alpha, eps_r, eps_r, 0.4alpha, 0.3alpha, Inf)
end

@testset "Maliuzhinets exact: PEC validation" begin
    # Calibration constant: C(k) = -e^{-iπ/4}/√(2πk)
    # Validated against KP cotangent formula for D_h (hard/Neumann)

    @testset "D_h matches KP for PEC (χ=0)" begin
        for alpha in [1.25π, 1.5π, 1.75π]
            Phi = alpha / 2
            k = 2π
            phi0 = 0.3 * alpha
            wedge = Wedge(alpha)

            for phi_frac in [0.2, 0.4, 0.6, 0.8]
                phi = phi_frac * alpha
                theta_NO = phi - Phi
                theta0_NO = phi0 - Phi

                D_spec = UTDKernels._spectral_D(theta_NO, theta0_NO, Phi, 0.0, 0.0, k)
                _, D_h_kp = pec_wedge_DsDh(wedge, RayAngles(phi, phi0), k, Inf)

                # Spectral D at χ=0 must match KP D_h to 10⁻⁸
                @test D_spec ≈ D_h_kp rtol = 1e-7
            end
        end
    end

    @testset "Calibration constant universal across wedge angles and k" begin
        C_pred(k) = -exp(-im * π / 4) / sqrt(2π * k)

        for alpha in [1.25π, 1.5π, 1.75π]
            Phi = alpha / 2
            nu = π / (2Phi)
            phi0 = 0.2 * alpha
            phi = 0.35 * alpha
            theta_NO = phi - Phi
            theta0_NO = phi0 - Phi

            for k in [1.0, 2π, 10.0]
                # Compute spectral function manually
                Psi0_t0 = UTDKernels._Psi_product(theta0_NO, Phi, 0.0, 0.0)
                function s_full(u)
                    sigma = nu * cos(nu * theta0_NO) / (sin(nu * u) - sin(nu * theta0_NO))
                    Psi0_u = UTDKernels._Psi_product(u, Phi, 0.0, 0.0)
                    return sigma * Psi0_u / Psi0_t0
                end
                D_spectral = s_full(theta_NO + π) - s_full(theta_NO - π)

                wedge = Wedge(alpha)
                _, D_h_kp = pec_wedge_DsDh(wedge, RayAngles(phi, phi0), k, Inf)

                C_emp = D_h_kp / D_spectral
                @test C_emp ≈ C_pred(k) rtol = 1e-6
            end
        end
    end
end

@testset "Maliuzhinets exact: reciprocity" begin
    alpha = 1.5π
    k = 2π
    phi1 = 0.3 * alpha
    phi2 = 0.6 * alpha

    @testset "Real χ (D_h)" begin
        for eps_r in [4.0, 25.0, 100.0]
            _, Dh1 = maliuzhinets_DsDh(alpha, eps_r, eps_r, phi1, phi2, k)
            _, Dh2 = maliuzhinets_DsDh(alpha, eps_r, eps_r, phi2, phi1, k)
            @test Dh1 ≈ Dh2 rtol = 1e-8
        end
    end

    @testset "Complex χ (D_s)" begin
        for eps_r in [4.0, 25.0, 100.0]
            Ds1, _ = maliuzhinets_DsDh(alpha, eps_r, eps_r, phi1, phi2, k)
            Ds2, _ = maliuzhinets_DsDh(alpha, eps_r, eps_r, phi2, phi1, k)
            @test Ds1 ≈ Ds2 rtol = 1e-8
        end
    end
end

@testset "Maliuzhinets exact: concurrent calls keep independent workspaces" begin
    alpha = 1.5π
    k = 2π
    cases = [
        (4.0 + i, 7.0 + 2i, (0.18 + 0.01i) * alpha, (0.31 + 0.005i) * alpha)
        for i in 0:15
    ]
    references = map(cases) do (eps_o, eps_n, phi, phip)
        maliuzhinets_DsDh(alpha, eps_o, eps_n, phi, phip, k)
    end
    tasks = map(cases) do (eps_o, eps_n, phi, phip)
        Threads.@spawn maliuzhinets_DsDh(alpha, eps_o, eps_n, phi, phip, k)
    end
    @test fetch.(tasks) == references
end

@testset "Maliuzhinets exact: impedance limits" begin
    alpha = 1.5π
    k = 2π
    phi = 0.4 * alpha
    phi0 = 0.3 * alpha
    wedge = Wedge(alpha)

    @testset "D_h → D_h_PEC as ε_r → ∞" begin
        _, D_h_pec = pec_wedge_DsDh(wedge, RayAngles(phi, phi0), k, Inf)

        # With very large ε_r, D_h should approach PEC
        _, D_h_large = maliuzhinets_DsDh(alpha, 1e8, 1e8, phi, phi0, k)
        @test abs(D_h_large) ≈ abs(D_h_pec) rtol = 1e-3
    end

    @testset "D monotonic with ε_r" begin
        # |D_h| should increase monotonically with ε_r (approaching PEC)
        Dh_prev = 0.0
        for eps_r in [2.0, 10.0, 100.0, 1000.0]
            _, Dh = maliuzhinets_DsDh(alpha, eps_r, eps_r, phi, phi0, k)
            @test abs(Dh) > Dh_prev
            Dh_prev = abs(Dh)
        end
    end
end
