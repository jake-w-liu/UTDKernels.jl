using Test
using UTDKernels
using ForwardDiff

@testset "AD compatibility" begin
    @testset "F_utd smoothness" begin
        for x in [0.5, 1.0, 5.0, 10.0]
            h1 = 1e-5
            h2 = 1e-6
            dF1 = (F_utd(x + h1) - F_utd(x - h1)) / (2h1)
            dF2 = (F_utd(x + h2) - F_utd(x - h2)) / (2h2)
            @test abs(dF1 - dF2) < 1e-4
        end
    end

    @testset "ForwardDiff gradients vs centered finite differences" begin
        w = Wedge(2π)
        k = 10.0
        L = 1.0
        phip = π/4

        f_phi(phi) = abs(pec_wedge_DsDh(w, RayAngles(phi, phip), k, L)[1])
        for phi in [π/6, π/3, π/2, deg2rad(224.0), deg2rad(226.0), 7π/4]
            h = 1e-7
            ad = ForwardDiff.derivative(f_phi, phi)
            fd = (f_phi(phi + h) - f_phi(phi - h)) / (2h)
            @test isfinite(ad)
            @test abs(ad - fd) / max(abs(ad), 1e-30) < 1e-5
        end

        f_k(k_) = abs(pec_wedge_DsDh(w, RayAngles(π/3, phip), k_, L)[1])
        for kval in [1.0, 10.0, 100.0]
            h = kval * 1e-7
            ad = ForwardDiff.derivative(f_k, kval)
            fd = (f_k(kval + h) - f_k(kval - h)) / (2h)
            @test isfinite(ad)
            @test abs(ad - fd) / max(abs(ad), 1e-30) < 1e-5
        end

        f_L(L_) = abs(pec_wedge_DsDh(w, RayAngles(π/3, phip), k, L_)[1])
        for Lval in [0.1, 1.0, 10.0]
            h = Lval * 1e-7
            ad = ForwardDiff.derivative(f_L, Lval)
            fd = (f_L(Lval + h) - f_L(Lval - h)) / (2h)
            @test isfinite(ad)
            @test abs(ad - fd) / max(abs(ad), 1e-30) < 1e-4
        end
    end

    @testset "Near-boundary one-sided gradients remain finite" begin
        w = Wedge(2π)
        k = 10.0
        L = 1.0
        phip = π/4
        isb = π + phip
        f_phi(phi) = abs(pec_wedge_DsDh(w, RayAngles(phi, phip), k, L)[1])

        for delta in (1e-4, 1e-6, 1e-8)
            d_left = ForwardDiff.derivative(f_phi, isb - delta)
            d_right = ForwardDiff.derivative(f_phi, isb + delta)
            @test isfinite(d_left)
            @test isfinite(d_right)
        end
    end

    @testset "wrap_angle is value-identical to mod and AD-finite on boundaries" begin
        # The primal uses Base.mod for accuracy at extreme scale. The AD path
        # reconstructs that value with the local derivative of the periodic
        # translation, including at an exact seam.
        for alpha in (2π, 3π/2, 7π/4, 1.234)
            for phi in range(-3alpha, 3alpha, length=400)
                @test UTDKernels.wrap_angle(phi, alpha) == mod(phi, alpha)
            end
            for phi in (0.0, alpha, 2alpha, -alpha, -3alpha)   # exact wrap boundaries
                d = ForwardDiff.derivative(p -> UTDKernels.wrap_angle(p, alpha), phi)
                @test isfinite(d) && d ≈ 1.0
            end
        end

        for (phi, alpha) in (
            (floatmax(Float64), floatmin(Float64)),
            (1e308, 1e-300),
            (-1e308, 1e-300),
            (1e16, 0.1),
        )
            wrapped = UTDKernels.wrap_angle(phi, alpha)
            @test wrapped == mod(phi, alpha)
            @test isfinite(wrapped)
        end

        huge_phi = 1e16
        alpha = 0.1
        @test ForwardDiff.derivative(p -> UTDKernels.wrap_angle(p, alpha), huge_phi) == 1.0
        phi = 5.3
        @test ForwardDiff.derivative(a -> UTDKernels.wrap_angle(phi, a), 1.2) ==
              -fld(phi, 1.2)
        @test_throws DomainError ForwardDiff.derivative(
            a -> UTDKernels.wrap_angle(1e308, a),
            1e-300,
        )
    end

    @testset "Far-field (L=Inf) and grazing coefficient AD is finite" begin
        # Regression: the Inf-L (far-field) PEC coefficient was non-differentiable
        # at grazing (phi=0) because wrap_angle returned a NaN derivative there.
        # Exact reproduction (phi=0, phip=pi, n=1.5, L=Inf) used to give NaN.
        w0 = Wedge(3π/2)
        ad0 = ForwardDiff.derivative(
            s -> real(pec_wedge_DsDh(w0, RayAngles(0.0 * s, π + 0.0 * s), 6.283, Inf)[1]),
            0.0,
        )
        @test isfinite(ad0)

        # Dense finiteness sweep, finite and far-field L, including grazing phi'=0.
        for alpha in (2π, 3π/2, 7π/4, 5π/4)
            w = Wedge(alpha)
            for phip in (0.0, π/6, π/4, π/2), k in (1.0, 10.0), L in (1.0, Inf)
                for phi in range(0.0, alpha, length=60)
                    adp = ForwardDiff.derivative(
                        p -> real(pec_wedge_DsDh(w, RayAngles(p, phip), k, L)[1]), phi)
                    adk = ForwardDiff.derivative(
                        kk -> real(pec_wedge_DsDh(w, RayAngles(phi, phip), kk, L)[1]), k)
                    @test isfinite(adp)
                    @test isfinite(adk)
                end
            end
        end
    end

    @testset "Exact face-grazing source-angle AD matches the interior limit" begin
        # The face-grazing value mapping is necessarily one-sided on the physical
        # wedge domain.  Preserve the local source-angle derivative at each seam:
        # snapping the effective incident angle to a constant gives finite but
        # falsely zero sensitivities here.
        alpha = 3π / 2
        w = Wedge(alpha)
        phi = 1.1
        k = 8.7
        h = 1e-5

        for generalized in (false, true), phip0 in (0.0, alpha), component in (1, 2)
            coefficient(x) = generalized ?
                pec_wedge_DsDh(w, RayAngles(phi, x), k, 2.1, 3.2, 4.3)[component] :
                pec_wedge_DsDh(w, RayAngles(phi, x), k, 1.4)[component]

            for projection in (real, imag)
                f(x) = projection(coefficient(x))
                ad = ForwardDiff.derivative(f, phip0)
                fd = phip0 == 0.0 ?
                    (f(phip0 + h) - f(phip0)) / h :
                    (f(phip0) - f(phip0 - h)) / h

                @test isfinite(ad)
                # Several hard-polarized, common-L derivatives vanish by
                # symmetry; use an absolute tolerance for those cancellation
                # cases and a relative comparison otherwise.
                @test ad ≈ fd rtol=3e-5 atol=2e-5
            end
        end
    end
end
