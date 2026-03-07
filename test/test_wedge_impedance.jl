using Test
using UTDKernels
using ForwardDiff
using SpecialFunctions: erfcx

_safe_sqrt_ref(x::Number) = sqrt(Complex(x))

function _F_ref(x::Number)
    if abs(x) < 1e-30
        return sqrt(π * Complex(x)) * exp(+im * π / 4)
    end
    z = exp(+im * π / 4) * _safe_sqrt_ref(x)
    return sqrt(π * Complex(x)) * exp(+im * π / 4) * erfcx(z)
end

function _fresnel_te_ref(psi::Number, eps_r::Number)
    s = sin(psi)
    c = cos(psi)
    eta = _safe_sqrt_ref(eps_r - c^2)
    return (s - eta) / (s + eta)
end

function _fresnel_tm_ref(psi::Number, eps_r::Number)
    s = sin(psi)
    c = cos(psi)
    eta = _safe_sqrt_ref(eps_r - c^2)
    return (eps_r * s - eta) / (eps_r * s + eta)
end

function _kp_terms_ref(phi::Real, phip::Real, n::Real)
    beta_minus = phi - phip
    beta_plus = phi + phip
    betas = (beta_minus, beta_minus, beta_plus, beta_plus)
    signs = (+1, -1, +1, -1)
    psi = ntuple(4) do j
        (π + signs[j] * betas[j]) / (2 * n)
    end
    Nj = ntuple(4) do j
        round(Int, (betas[j] + signs[j] * π) / (2 * n * π))
    end
    aj = ntuple(4) do j
        2 * cos((2 * n * π * Nj[j] - betas[j]) / 2)^2
    end
    return (psi = psi, aj = aj)
end

function _holm_reference_DsDh(
    alpha::Real,
    eps_r_o::Number,
    eps_r_n::Number,
    phi::Real,
    phip::Real,
    k::Real,
    L::Real,
)
    n = alpha / π
    phi_w = mod(phi, alpha)
    phip_w = mod(phip, alpha)
    terms = _kp_terms_ref(phi_w, phip_w, n)
    C = -exp(-im * π / 4) / (2 * n * sqrt(2 * π * k))

    c = ntuple(4) do j
        cot(terms.psi[j]) * _F_ref(k * L * terms.aj[j])
    end

    psi_o = clamp(phip_w, 0.0, π / 2)
    psi_n = clamp(alpha - phip_w, 0.0, π / 2)

    R_te_o = _fresnel_te_ref(psi_o, eps_r_o)
    R_tm_o = _fresnel_tm_ref(psi_o, eps_r_o)
    R_te_n = _fresnel_te_ref(psi_n, eps_r_n)
    R_tm_n = _fresnel_tm_ref(psi_n, eps_r_n)

    Ds = C * (c[1] + c[2] + R_te_n * c[3] + R_te_o * c[4])
    Dh = C * (c[1] + c[2] + R_tm_n * c[3] + R_tm_o * c[4])
    return (Ds, Dh)
end

@testset "Impedance wedge validation" begin
    alpha = 1.5π
    phip = π / 4
    k = 20.0
    L = 2.0
    ang = RayAngles(2.0, phip)

    @testset "Single-L API argument guards (impedance)" begin
        iw = ImpedanceWedge(alpha, WedgeFaceMaterial(complex(5.31, -0.3)))
        @test_throws DomainError impedance_wedge_DsDh(iw, ang, k, 0.0)
        @test_throws DomainError impedance_wedge_DsDh(iw, ang, k, -1.0)
        @test_throws DomainError impedance_wedge_DsDh(iw, ang, k, 0.0 + 0.0im)
        Ds_inf, Dh_inf = impedance_wedge_DsDh(iw, ang, k, Inf)
        @test isfinite(real(Ds_inf)) && isfinite(imag(Ds_inf))
        @test isfinite(real(Dh_inf)) && isfinite(imag(Dh_inf))
        Ds_c, Dh_c = impedance_wedge_DsDh(iw, ang, k, L + 0.2im)
        @test isfinite(real(Ds_c)) && isfinite(imag(Ds_c))
        @test isfinite(real(Dh_c)) && isfinite(imag(Dh_c))
    end

    @testset "PEC-limit convergence for impedance wedge" begin
        w_pec = Wedge(alpha)
        Ds_pec, Dh_pec = pec_wedge_DsDh(w_pec, ang, k, L)
        vals = [1e4, 1e8, 1e12, 1e14]
        errs_s = Float64[]
        errs_h = Float64[]
        for v in vals
            mat = WedgeFaceMaterial(complex(v, -v))
            iw = ImpedanceWedge(alpha, mat)
            Ds, Dh = impedance_wedge_DsDh(iw, ang, k, L)
            push!(errs_s, abs(Ds - Ds_pec) / abs(Ds_pec))
            push!(errs_h, abs(Dh - Dh_pec) / abs(Dh_pec))
        end
        @test all(diff(errs_s) .< 0)
        @test all(diff(errs_h) .< 0)
        @test errs_s[end] < 1e-5
        @test errs_h[end] < 1e-5
    end

    @testset "PEC-limit recovery across observation sweep" begin
        w_pec = Wedge(alpha)
        iw_metal = ImpedanceWedge(alpha, WedgeFaceMaterial(complex(1.0, -1e12)))
        phis = range(0.05, stop=alpha - 0.05, length=300)

        Ds_pec = ComplexF64[]
        Dh_pec = ComplexF64[]
        Ds_met = ComplexF64[]
        Dh_met = ComplexF64[]
        for phi in phis
            ang_loc = RayAngles(phi, phip)
            Ds_p, Dh_p = pec_wedge_DsDh(w_pec, ang_loc, k, L)
            Ds_m, Dh_m = impedance_wedge_DsDh(iw_metal, ang_loc, k, L)
            push!(Ds_pec, Ds_p); push!(Dh_pec, Dh_p)
            push!(Ds_met, Ds_m); push!(Dh_met, Dh_m)
        end

        rel_s = sqrt(sum(abs2, Ds_met .- Ds_pec)) / max(sqrt(sum(abs2, Ds_pec)), 1e-30)
        rel_h = sqrt(sum(abs2, Dh_met .- Dh_pec)) / max(sqrt(sum(abs2, Dh_pec)), 1e-30)
        @test rel_s < 1e-5
        @test rel_h < 1e-5
    end

    @testset "Generalized-L overload consistency (impedance)" begin
        eps_concrete = complex(5.31, -0.3)
        iw = ImpedanceWedge(alpha, WedgeFaceMaterial(eps_concrete))
        for phi in range(0.1, stop=alpha - 0.1, length=80)
            ang_loc = RayAngles(phi, phip)
            Ds0, Dh0 = impedance_wedge_DsDh(iw, ang_loc, k, L)
            Ds1, Dh1 = impedance_wedge_DsDh(iw, ang_loc, k, L, L, L)
            @test isapprox(Ds0, Ds1; rtol=1e-12, atol=1e-12)
            @test isapprox(Dh0, Dh1; rtol=1e-12, atol=1e-12)
        end
    end

    @testset "Finiteness across angle sweep" begin
        eps_concrete = complex(5.31, -0.3)
        iw = ImpedanceWedge(alpha, WedgeFaceMaterial(eps_concrete))
        phis = range(0.001, stop=alpha - 0.001, length=1000)
        Ds_vals = ComplexF64[]
        Dh_vals = ComplexF64[]
        for phi in phis
            Ds, Dh = impedance_wedge_DsDh(iw, RayAngles(phi, phip), k, L)
            push!(Ds_vals, Ds)
            push!(Dh_vals, Dh)
        end
        @test all(isfinite.(abs.(Ds_vals)))
        @test all(isfinite.(abs.(Dh_vals)))
    end

    @testset "Near-boundary finiteness (impedance)" begin
        eps_concrete = complex(5.31, -0.3)
        iw = ImpedanceWedge(alpha, WedgeFaceMaterial(eps_concrete))
        boundaries = (π - phip, π + phip)
        deltas = (1e-3, 1e-6, 1e-9)

        for b in boundaries
            Ds0, Dh0 = impedance_wedge_DsDh(iw, RayAngles(b, phip), k, L)
            @test isfinite(real(Ds0)) && isfinite(imag(Ds0))
            @test isfinite(real(Dh0)) && isfinite(imag(Dh0))

            for d in deltas, sgn in (-1.0, +1.0)
                phi = b + sgn * d
                if 0.0 < phi < alpha
                    Ds, Dh = impedance_wedge_DsDh(iw, RayAngles(phi, phip), k, L)
                    @test isfinite(real(Ds)) && isfinite(imag(Ds))
                    @test isfinite(real(Dh)) && isfinite(imag(Dh))
                end
            end
        end
    end

    @testset "Face-swap mirror symmetry" begin
        iw = ImpedanceWedge(
            alpha,
            WedgeFaceMaterial(complex(4.2, -0.15)),
            WedgeFaceMaterial(complex(8.1, -0.25)),
        )
        iw_sw = ImpedanceWedge(alpha, iw.face_n, iw.face_o)

        for phi in (0.43, 0.97, 1.61, 2.34, 3.52, 4.26)
            for phip_local in (0.31, 0.88, 1.46, 2.02, 2.61, 3.10)
                ang_a = RayAngles(phi, phip_local)
                ang_b = RayAngles(alpha - phi, alpha - phip_local)
                Ds_a, Dh_a = impedance_wedge_DsDh(iw, ang_a, k, L)
                Ds_b, Dh_b = impedance_wedge_DsDh(iw_sw, ang_b, k, L)
                @test isapprox(Ds_a, Ds_b; rtol=1e-11, atol=1e-12)
                @test isapprox(Dh_a, Dh_b; rtol=1e-11, atol=1e-12)
            end
        end
    end

    @testset "Homogeneous-face limit (eps_r = 1) drops reflection terms" begin
        w = Wedge(alpha)
        iw_air = ImpedanceWedge(alpha, WedgeFaceMaterial(complex(1.0, 0.0)))
        for phi in range(0.1, stop=alpha - 0.1, length=80)
            ang_loc = RayAngles(phi, phip)
            Ds_imp, Dh_imp = impedance_wedge_DsDh(iw_air, ang_loc, k, L)
            Ds_ref, Dh_ref = pec_wedge_DsDh(w, ang_loc, k, L, L, L; Rs=0, Rh=0)
            @test isapprox(Ds_imp, Ds_ref; rtol=1e-12, atol=1e-12)
            @test isapprox(Dh_imp, Dh_ref; rtol=1e-12, atol=1e-12)
        end
    end

    @testset "Holm transcription consistency" begin
        eps_concrete = complex(5.31, -0.3)
        iw = ImpedanceWedge(alpha, WedgeFaceMaterial(eps_concrete))
        for phi in range(0.05, stop=alpha - 0.05, length=50)
            Ds_lib, Dh_lib = impedance_wedge_DsDh(iw, RayAngles(phi, phip), k, L)
            Ds_ref, Dh_ref = _holm_reference_DsDh(
                alpha, eps_concrete, eps_concrete, phi, phip, k, L,
            )
            @test abs(Ds_lib - Ds_ref) / max(abs(Ds_ref), 1e-30) < 1e-11
            @test abs(Dh_lib - Dh_ref) / max(abs(Dh_ref), 1e-30) < 1e-11
        end
    end

    @testset "ForwardDiff vs finite differences (impedance paths)" begin
        eps_concrete = complex(5.31, -0.3)
        iw = ImpedanceWedge(alpha, WedgeFaceMaterial(eps_concrete))
        h = 1e-7
        for phi0 in [0.4, 1.2, 2.1, 3.0]
            f_phi(phi) = begin
                Ds, _ = impedance_wedge_DsDh(iw, RayAngles(phi, phip), k, L)
                abs2(Ds)
            end
            ad_phi = ForwardDiff.derivative(f_phi, phi0)
            fd_phi = (f_phi(phi0 + h) - f_phi(phi0 - h)) / (2h)
            @test isfinite(ad_phi)
            @test abs(ad_phi - fd_phi) / max(abs(fd_phi), 1e-30) < 1e-5

            f_eps(e) = begin
                iw_loc = ImpedanceWedge(alpha, WedgeFaceMaterial(complex(e, -0.3)))
                Ds, _ = impedance_wedge_DsDh(iw_loc, RayAngles(phi0, phip), k, L)
                abs2(Ds)
            end
            ad_eps = ForwardDiff.derivative(f_eps, 5.31)
            fd_eps = (f_eps(5.31 + h) - f_eps(5.31 - h)) / (2h)
            @test isfinite(ad_eps)
            @test abs(ad_eps - fd_eps) / max(abs(fd_eps), 1e-30) < 1e-5
        end
    end
end
