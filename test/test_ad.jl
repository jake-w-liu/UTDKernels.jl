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
        # Regression: wrap_angle used mod(phi, alpha), whose forward-mode AD rule
        # returns a NaN derivative exactly on a wrap boundary (phi a multiple of
        # alpha). The floor form is bit-identical in value and differentiates with
        # the correct local unit slope there.
        for alpha in (2π, 3π/2, 7π/4, 1.234)
            for phi in range(-3alpha, 3alpha, length=400)
                # fld form equals mod to machine precision (defining identity)
                @test isapprox(UTDKernels.wrap_angle(phi, alpha), mod(phi, alpha); atol = 1e-12)
            end
            for phi in (0.0, alpha, 2alpha, -alpha, -3alpha)   # exact wrap boundaries
                d = ForwardDiff.derivative(p -> UTDKernels.wrap_angle(p, alpha), phi)
                @test isfinite(d) && d ≈ 1.0
            end
        end
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
end
