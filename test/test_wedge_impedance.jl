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
    eps_r == one(eps_r) && return zero(Complex(eps_r))
    s = sin(psi)
    c = cos(psi)
    eta = _safe_sqrt_ref(eps_r - c^2)
    return (s - eta) / (s + eta)
end

function _fresnel_tm_ref(psi::Number, eps_r::Number)
    eps_r == one(eps_r) && return zero(Complex(eps_r))
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
    return (psi = psi, aj = aj, beta = betas, Nj = Nj)
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
    Ds, Dh = _holm_reference_DsDh(alpha, eps_r_o, eps_r_n, phi, phip, k, L, L, L)
    G = phip == 0 || phip == alpha ? 0.5 : 1.0
    return (G * Ds, G * Dh)
end

function _holm_reference_DsDh(
    alpha::Real,
    eps_r_o::Number,
    eps_r_n::Number,
    phi::Real,
    phip::Real,
    k::Real,
    Li::Real,
    Lro::Real,
    Lrn::Real,
)
    n = alpha / π
    phi_w = mod(phi, alpha)
    phip_w = mod(phip, alpha)
    if phip_w == 0 && phip == alpha
        phip_w = alpha
    end
    terms = _kp_terms_ref(phi_w, phip_w, n)
    C = -exp(-im * π / 4) / (2 * n * sqrt(2 * π * k))

    L_per_term = (Li, Li, Lrn, Lro)
    c = ntuple(4) do j
        cot(terms.psi[j]) * _F_ref(k * L_per_term[j] * terms.aj[j])
    end

    # Holm's published face angles use both ray directions. This independent
    # transcription deliberately spells out the minima instead of calling a
    # production helper, so source-only-angle mutations are detectable.
    theta_o = min(phip_w, phi_w)
    theta_n = min(alpha - phip_w, alpha - phi_w)

    R_te_o = _fresnel_te_ref(theta_o, eps_r_o)
    R_tm_o = _fresnel_tm_ref(theta_o, eps_r_o)
    R_te_n = _fresnel_te_ref(theta_n, eps_r_n)
    R_tm_n = _fresnel_tm_ref(theta_n, eps_r_n)

    product_te = R_te_o * R_te_n
    product_tm = R_tm_o * R_tm_n
    # Holm fixes M1=R0*Rn and M2=1. Exchanging these weights across the source
    # half-angle is a later reciprocal modification, not the Holm coefficient.
    W_te_n, W_te_o = product_te, one(product_te)
    W_tm_n, W_tm_o = product_tm, one(product_tm)
    Ds = C * (W_te_n * c[1] + W_te_o * c[2] + R_te_n * c[3] + R_te_o * c[4])
    Dh = C * (W_tm_n * c[1] + W_tm_o * c[2] + R_tm_n * c[3] + R_tm_o * c[4])
    return (Ds, Dh)
end

@testset "passive material square-root sheet" begin
    # exp(+iωt) with exp(-ik_z z) requires the lower-bank value on the
    # negative-real cut. Preserve the principal branch everywhere off that cut,
    # including explicitly active upper-half-plane inputs.
    @test UTDKernels.radiation_sqrt(-4.0) == -2im
    @test UTDKernels.radiation_sqrt(complex(-4.0, -0.0)) == -2im
    @test UTDKernels.radiation_sqrt(4.0) == 2.0 + 0.0im
    active = -4.0 + 1.0e-12im
    @test UTDKernels.radiation_sqrt(active) == sqrt(Complex(active))

    psi = π / 4
    rte0 = fresnel_te(psi, -4.0)
    rtm0 = fresnel_tm(psi, -4.0)
    @test rte0 ≈ -0.8 + 0.6im rtol=4eps(Float64) atol=0
    @test rtm0 ≈ 0.28 - 0.96im rtol=4eps(Float64) atol=0
    for delta in (1.0e-6, 1.0e-9, 1.0e-12)
        @test fresnel_te(psi, -4.0 - delta * im) ≈ rte0 rtol=1e-5 atol=0
        @test fresnel_tm(psi, -4.0 - delta * im) ≈ rtm0 rtol=1e-5 atol=0
    end

    # Positive-real and ordinary passive lossy inputs retain their established
    # principal-root values.
    for eps_r in (4.0, 3.7 - 0.2im)
        @test fresnel_te(0.41, eps_r) ≈ _fresnel_te_ref(0.41, eps_r) rtol=4eps() atol=0
        @test fresnel_tm(0.41, eps_r) ≈ _fresnel_tm_ref(0.41, eps_r) rtol=4eps() atol=0
    end

    # The selected cut value remains differentiable along the lossless-real
    # sheet away from its branch point.
    f(eps_r) = imag(fresnel_te(psi, eps_r))
    ad = ForwardDiff.derivative(f, -4.0)
    step = 1.0e-6
    fd = (f(-4.0 + step) - f(-4.0 - step)) / (2step)
    @test ad ≈ fd rtol=1e-7 atol=1e-9

    # The Holm impedance-wedge evaluator consumes the same Fresnel roots.
    alpha = 1.5π
    ang = RayAngles(2.0, 0.7)
    exact = impedance_wedge_DsDh(
        ImpedanceWedge(alpha, WedgeFaceMaterial(-4.0)), ang, 20.0, 2.0,
    )
    limiting = impedance_wedge_DsDh(
        ImpedanceWedge(alpha, WedgeFaceMaterial(-4.0 - 1.0e-9im)), ang, 20.0, 2.0,
    )
    @test exact[1] ≈ limiting[1] rtol=1e-7 atol=1e-10
    @test exact[2] ≈ limiting[2] rtol=1e-7 atol=1e-10
end

@testset "Impedance wedge validation" begin
    alpha = 1.5π
    phip = π / 4
    k = 20.0
    L = 2.0
    ang = RayAngles(2.0, phip)

    @testset "Single-L API argument guards (impedance)" begin
        iw = ImpedanceWedge(alpha, WedgeFaceMaterial(complex(5.31, -0.3)))
        @test_throws DomainError impedance_wedge_DsDh(iw, ang, 0.0, L)
        @test_throws DomainError impedance_wedge_DsDh(iw, ang, -1.0, L)
        @test_throws DomainError impedance_wedge_DsDh(iw, ang, 0.0 + 0.0im, L)
        @test_throws DomainError impedance_wedge_DsDh(iw, ang, -1.0 + 0.0im, L)
        @test_throws DomainError impedance_wedge_DsDh(iw, ang, 0.0, L, L, L)
        @test_throws DomainError impedance_wedge_DsDh(iw, ang, -1.0, L, L, L)
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

    @testset "near-PEC grazing assembly preserves the soft coefficient" begin
        eps_pec = 1.0e100 * (1 - im)
        iw_pec = ImpedanceWedge(alpha, WedgeFaceMaterial(eps_pec))
        phi = 2.0
        for h in (1.0e-10, 1.0e-12, 1.0e-14, 1.0e-16)
            angles = RayAngles(phi, h)
            got = wedge_DsDh(iw_pec, angles, k, L)
            reference = wedge_DsDh(Wedge(alpha), angles, k, L)
            @test got[1] ≈ reference[1] rtol=5e-13 atol=0
            @test got[2] ≈ reference[2] rtol=5e-13 atol=0
        end
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

    @testset "Holm face-swap nonreciprocity is represented faithfully" begin
        iw = ImpedanceWedge(
            alpha,
            WedgeFaceMaterial(complex(4.2, -0.15)),
            WedgeFaceMaterial(complex(8.1, -0.25)),
        )
        iw_sw = ImpedanceWedge(alpha, iw.face_n, iw.face_o)

        unequal_pairs = 0
        for phi in (0.43, 0.97, 1.61, 2.34, 3.52, 4.26)
            for phip_local in (0.31, 0.88, 1.46, 2.02, 2.61, 3.10)
                ang_a = RayAngles(phi, phip_local)
                ang_b = RayAngles(alpha - phi, alpha - phip_local)
                Ds_a, Dh_a = impedance_wedge_DsDh(iw, ang_a, k, L)
                Ds_b, Dh_b = impedance_wedge_DsDh(iw_sw, ang_b, k, L)
                Ds_ref_a, Dh_ref_a = _holm_reference_DsDh(
                    alpha, iw.face_o.eps_r, iw.face_n.eps_r,
                    phi, phip_local, k, L,
                )
                Ds_ref_b, Dh_ref_b = _holm_reference_DsDh(
                    alpha, iw_sw.face_o.eps_r, iw_sw.face_n.eps_r,
                    alpha - phi, alpha - phip_local, k, L,
                )
                @test Ds_a ≈ Ds_ref_a rtol=1e-11 atol=1e-12
                @test Dh_a ≈ Dh_ref_a rtol=1e-11 atol=1e-12
                @test Ds_b ≈ Ds_ref_b rtol=1e-11 atol=1e-12
                @test Dh_b ≈ Dh_ref_b rtol=1e-11 atol=1e-12
                unequal_pairs += !isapprox(Ds_a, Ds_b; rtol=1e-8, atol=1e-10)
            end
        end
        # Holm's original heuristic is not reciprocal; this guard prevents a
        # later reciprocal weight exchange from being misattributed to Holm.
        @test unequal_pairs > 0
    end

    @testset "Homogeneous-face limit retains Holm's fixed incident term" begin
        iw_air = ImpedanceWedge(alpha, WedgeFaceMaterial(complex(1.0, 0.0)))
        for phip_local in (0.6, 3.0)
            for phi in range(0.1, stop=alpha - 0.1, length=80)
                ang_loc = RayAngles(phi, phip_local)
                Ds_imp, Dh_imp = impedance_wedge_DsDh(iw_air, ang_loc, k, L)
                terms = _kp_terms_ref(phi, phip_local, alpha / π)
                j = 2
                u = 2 * alpha * terms.Nj[j] - terms.beta[j]
                detuning = (u + π) / (2 * (alpha / π))
                c2 = UTDKernels._cot_F_regularized(
                    terms.psi[j], terms.aj[j], k, L;
                    n=alpha / π, detuning,
                )
                expected = -exp(-im * π / 4) / (2 * (alpha / π) * sqrt(2 * π * k)) * c2
                @test isapprox(Ds_imp, expected; rtol=1e-12, atol=1e-12)
                @test isapprox(Dh_imp, expected; rtol=1e-12, atol=1e-12)
            end
        end
    end

    @testset "Holm transcription consistency" begin
        eps_o = complex(4.2, -0.15)
        eps_n = complex(8.1, -0.25)
        iw = ImpedanceWedge(
            alpha, WedgeFaceMaterial(eps_o), WedgeFaceMaterial(eps_n),
        )
        Li, Lro, Lrn = 1.7, 2.3, 3.1
        for phip_local in (0.6, 3.0)
            for phi in range(0.05, stop=alpha - 0.05, length=50)
                ang_loc = RayAngles(phi, phip_local)
                Ds_lib, Dh_lib = impedance_wedge_DsDh(iw, ang_loc, k, L)
                Ds_ref, Dh_ref = _holm_reference_DsDh(
                    alpha, eps_o, eps_n, phi, phip_local, k, L,
                )
                @test abs(Ds_lib - Ds_ref) / max(abs(Ds_ref), 1e-30) < 1e-11
                @test abs(Dh_lib - Dh_ref) / max(abs(Dh_ref), 1e-30) < 1e-11

                Ds_lib_3L, Dh_lib_3L = impedance_wedge_DsDh(
                    iw, ang_loc, k, Li, Lro, Lrn,
                )
                Ds_ref_3L, Dh_ref_3L = _holm_reference_DsDh(
                    alpha, eps_o, eps_n, phi, phip_local, k, Li, Lro, Lrn,
                )
                @test abs(Ds_lib_3L - Ds_ref_3L) / max(abs(Ds_ref_3L), 1e-30) < 1e-11
                @test abs(Dh_lib_3L - Dh_ref_3L) / max(abs(Dh_ref_3L), 1e-30) < 1e-11
            end
        end

        @test UTDKernels._holm_grazing_factor(0.0, alpha) == 0.5
        @test UTDKernels._holm_grazing_factor(alpha, alpha) == 0.5
        @test UTDKernels._holm_grazing_factor(nextfloat(0.0), alpha) == 1.0
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

    @testset "Face-grazing values and one-sided derivatives" begin
        # Holm assigns an isolated G=1/2 factor at both exact face incidences.
        # The separate-distance extension deliberately omits that factor and
        # follows its one-sided continuous branch.
        iw = ImpedanceWedge(
            alpha,
            WedgeFaceMaterial(complex(5.31, -0.3)),
            WedgeFaceMaterial(complex(8.2, -0.7)),
        )
        phi = 1.1

        h = 1e-7
        for phip_grazing in (0.0, alpha), component in (1, 2)
            single_L(x) = real(impedance_wedge_DsDh(
                iw, RayAngles(phi, x), k, L,
            )[component])
            generalized_L(x) = imag(impedance_wedge_DsDh(
                iw, RayAngles(phi, x), k, 2.1, 3.2, 4.3,
            )[component])

            @test isfinite(single_L(phip_grazing))
            @test isfinite(generalized_L(phip_grazing))
            ad_single = ForwardDiff.derivative(single_L, phip_grazing)
            ad_generalized = ForwardDiff.derivative(generalized_L, phip_grazing)
            fd_single = phip_grazing == 0.0 ?
                (single_L(h) - single_L(0.0)) / h :
                (single_L(alpha) - single_L(alpha - h)) / h
            fd_generalized = phip_grazing == 0.0 ?
                (generalized_L(h) - generalized_L(0.0)) / h :
                (generalized_L(alpha) - generalized_L(alpha - h)) / h
            @test isfinite(ad_single)
            @test isfinite(ad_generalized)
            if phip_grazing == 0.0
                @test ad_single ≈ fd_single rtol=2e-5 atol=1e-8
            else
                # At the n-face, Holm's published isolated half-factor makes
                # the exact single-L value half its one-sided limit. AD at the
                # discontinuity is only a branch derivative, so do not compare
                # it with a divergent difference quotient.
                raw_equal_L = impedance_wedge_DsDh(
                    iw, RayAngles(phi, alpha), k, L, L, L,
                )[component]
                exact_single = impedance_wedge_DsDh(
                    iw, RayAngles(phi, alpha), k, L,
                )[component]
                @test exact_single ≈ raw_equal_L / 2 rtol=1e-12 atol=1e-12
            end
            @test ad_generalized ≈ fd_generalized rtol=2e-5 atol=1e-8
        end
    end
end
