using Test
using UTDKernels

@testset "PEC wedge GTD limit (kL → ∞)" begin

    @testset "Half-plane GTD coefficients" begin
        w = Wedge(2π)
        n = wedge_n(w)
        phip = π/4
        phi_test = π/2

        k_large = 1e6
        L = 1.0

        ang = RayAngles(phi_test, phip)
        Ds, Dh = pec_wedge_DsDh(w, ang, k_large, L)

        # Compute GTD reference (F=1 everywhere)
        phi_w  = UTDKernels.wrap_angle(phi_test, w.alpha)
        phip_w = UTDKernels.wrap_angle(phip, w.alpha)
        C = -exp(-im * π/4) / (2 * n * sqrt(2π * k_large))
        terms = UTDKernels.kp_four_terms(phi_w, phip_w, n)

        Ds_gtd = C * sum(UTDKernels.PEC_SIGMA_SOFT[j] * cot(terms.psi[j]) for j in 1:4)
        Dh_gtd = C * sum(UTDKernels.PEC_SIGMA_HARD[j] * cot(terms.psi[j]) for j in 1:4)

        @test abs(Ds - Ds_gtd) / abs(Ds_gtd) < 1e-4
        @test abs(Dh - Dh_gtd) / abs(Dh_gtd) < 1e-4
    end

    @testset "GTD convergence with increasing kL" begin
        w = Wedge(3π/2)  # 90° wedge (avoids half-plane edge cases)
        phip = π/4
        phi = π/2
        ang = RayAngles(phi, phip)

        # Compute reference at very large kL
        k_ref = 1e8
        Ds_ref, Dh_ref = pec_wedge_DsDh(w, ang, k_ref, 1.0)

        # Normalized (remove 1/√k dependence)
        Ds_ref_norm = Ds_ref * sqrt(k_ref)
        Dh_ref_norm = Dh_ref * sqrt(k_ref)

        for k in [1e2, 1e3, 1e4, 1e5]
            Ds, Dh = pec_wedge_DsDh(w, ang, k, 1.0)
            Ds_norm = Ds * sqrt(k)
            Dh_norm = Dh * sqrt(k)
            err_s = abs(Ds_norm - Ds_ref_norm) / abs(Ds_ref_norm)
            err_h = abs(Dh_norm - Dh_ref_norm) / abs(Dh_ref_norm)
            @test err_s < 0.1
            @test err_h < 0.1
        end
    end
end
