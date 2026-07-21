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

@testset "F1-1: far-field pole gate is AND, not OR (no zeroed annulus)" begin
    # Regression for the isinf(L) branch of _cot_F_regularized (WedgePEC.jl). The
    # distance parameter a_j = 2cos²((2nπN−β)/2) is QUADRATIC in the angular
    # detuning while sin ψ_j is linear. In ψ-space a ≈ 2n²Δψ², so |a|≤√eps spans
    # |Δψ| ≲ 8.6e-5/n rad; the same window in azimuthal (φ) space has a ≈ (δφ)²/2,
    # so the OR gate `|a|≤√eps || |sinψ|≤√eps` fired over |δφ| ≲ √(2√eps) ≈ 1.7e-4
    # rad and zeroed a finite, exactly representable cot term over a whole annulus
    # around every GO boundary. The correct gate is AND (matching the finite-L
    # branch): zero only the tight |sinψ|≤√eps neighborhood of each coincident-pole
    # sample.
    alpha = 1.5π
    wedge = Wedge(alpha)
    n = wedge_n(wedge)          # 1.5
    k = 2.0
    phip = 0.3                  # not grazing -> _effective_angles is identity here

    # Independent GTD reference at F(kLa)=1: D = C·Σ σ_j cot(ψ_j), built from the
    # KP cotangent-argument formulas directly, NOT from the function under test.
    function gtd_cot_ref(phi)
        t = UTDKernels.kp_four_terms(phi, phip, n)
        C = UTDKernels.pec_wedge_prefactor(k, n)
        Ds = zero(ComplexF64); Dh = zero(ComplexF64)
        for j in 1:4
            cj = cot(t.psi[j])
            Ds += UTDKernels.PEC_SIGMA_SOFT[j] * cj
            Dh += UTDKernels.PEC_SIGMA_HARD[j] * cj
        end
        return (C * Ds, C * Dh)
    end

    # rtol: reference and corrected code both evaluate C·Σσ_j cot(ψ_j) in Float64
    # in the same term order; only floating summation rounding differs (≲ few·eps·κ,
    # κ≈0.7 here), so 1e-11 sits ~5 decades above the rounding floor yet ~9 decades
    # below the old OR-gate error (rel err ≈ 1.0 from the zeroed dominant term).
    # Sweep spans the former dead window δ ∈ [~5e-8, 1.72e-4] where OLD code
    # returned 0 for the singular term (φ = π + φ' − δ approaches the ISB at π+φ').
    for delta in (1.0e-6, 1.0e-5, 5.0e-5, 1.0e-4, 1.5e-4, 1.7e-4)
        phi = π + phip - delta
        Ds, Dh = pec_wedge_DsDh(wedge, RayAngles(phi, phip), k, Inf)
        Ds_ref, Dh_ref = gtd_cot_ref(phi)
        @test Dh ≈ Dh_ref rtol = 1e-11
        @test Ds ≈ Ds_ref rtol = 1e-11
    end

    # No jump across the former window edge (δ=1.726e-4). The cot term varies
    # smoothly here: |ΔD|/|D| ≈ Δδ/δ ≈ 1e-5/1.75e-4 ≈ 0.057. The OLD OR gate zeroed
    # the δ=1.7e-4 sample while keeping δ=1.8e-4, giving a ~1.0 relative jump; the
    # 0.15 threshold is ~10× smaller than that yet well above the smooth value.
    Ds17, Dh17 = pec_wedge_DsDh(wedge, RayAngles(π + phip - 1.7e-4, phip), k, Inf)
    Ds18, Dh18 = pec_wedge_DsDh(wedge, RayAngles(π + phip - 1.8e-4, phip), k, Inf)
    @test abs(Dh17 - Dh18) / abs(Dh18) < 0.15
    @test abs(Ds17 - Ds18) / abs(Ds18) < 0.15
end

