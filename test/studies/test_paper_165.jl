module Paper165CanonicalTests
import UTDKernels
using Test
using UTDKernels.CurvatureMeasure
using QuadGK
using Random
using SpecialFunctions
import UTDKernels

const PI = pi

@testset "CurvatureMeasure" begin
    @testset "transition limits" begin
        @test transition_F(0.0) == 0.0 + 0.0im
        # Large-x KP limit: F(x) -> 1.
        @test abs(transition_F(1.0e6) - 1) < 1e-6
        # Independent scipy.special.erfcx oracle (Python transition_F(1), 2026-08-19).
        F1 = 0.8095254817474091 + 0.23219939005526471im
        @test abs(transition_F(1.0) - F1) < 5e-15
        F2 = 0.909203498997822 + 0.17108658129968907im
        @test abs(transition_F(2.0) - F2) < 5e-15
    end

    @testset "Bessel cylinder oracle" begin
        # Independent scipy.special.jv(3, 24) frozen value.
        @test abs(besselj(3, 24.0) - 0.16127035997227668) < 5e-15
        m, z, phi = 3, 24.0, 0.37
        I = continuum_harmonic(m, z, phi)
        I_py = -0.9076039151540007 + 0.4505717603087463im
        @test abs(I - I_py) < 2e-14
    end

    @testset "intrinsic linear limit" begin
        k, L = 25.0, 0.7
        A = linear_prefactor(k, L)
        ds = exp10.(range(-7, -3; length=20))
        err = [abs(intrinsic_coefficient(d, k, L) / (A * d) - 1) for d in ds]
        slope = fit_log_slope(ds[1:12], err[1:12])
        @test abs(slope - 1) < 0.01

        # Regression: evaluating cot(π/(2n)) directly loses the small angle
        # when n = 1 + δ/π rounds close to one.
        for d in (1e-12, 1e-16, 1e-18)
            @test local_bias_factor(d) ≈ local_bias_series(d) rtol=6e-16
            @test intrinsic_coefficient(d, k, L) / (A * d) ≈
                  local_bias_series(d) rtol=5e-16
        end
    end

    @testset "bias closed form" begin
        k, L = 40.0, 1.3
        for d in (1e-5, 1e-3, 0.05, 0.2)
            lhs = intrinsic_coefficient(d, k, L) / (linear_prefactor(k, L) * d)
            @test abs(lhs - local_bias_factor(d)) < 5e-14
        end
        # 256-bit BigFloat evaluation of the stable cofunction identity.
        @test abs(local_bias_factor(2 * pi / 512) - 0.9922453949516098) < 5e-16
    end

    @testset "bias series" begin
        for d in (1e-4, 3e-4, 1e-3)
            @test abs(local_bias_factor(d) - local_bias_series(d)) < 2e-9
        end
    end

    @testset "harmonic integral by quadrature" begin
        m, z, phi = 3, 7.3, 0.41
        integrand = t -> exp(im * m * t - im * z * cos(t - phi))
        real_part, _ = quadgk(t -> real(integrand(t)), 0, 2 * pi; atol=1e-12)
        imag_part, _ = quadgk(t -> imag(integrand(t)), 0, 2 * pi; atol=1e-12)
        I_quad = real_part + im * imag_part
        @test abs(I_quad - continuum_harmonic(m, z, phi)) < 2e-11
    end

    @testset "alias identity random" begin
        rng = Xoshiro(314159)
        for _ in 1:100
            N = rand(rng, 8:79)
            m = rand(rng, -5:5)
            z = rand(rng) * (35.0 - 0.1) + 0.1
            phi = rand(rng) * 2 * pi - pi
            direct = polygon_measure_sum(N, m, z, phi)
            alias = polygon_alias_sum(N, m, z, phi; ell_max=12)
            @test abs(direct - alias) < 2e-11
        end
    end

    @testset "error decomposition" begin
        d = error_decomposition(24, 2, 18.0, 0.3, 20.0, 1.0)
        @test abs(d.residual) < 2e-15
    end

    @testset "debiased equals measure sum" begin
        v = debiased_measure_sum(37, 2, 14.0, 0.2, 30.0, 0.8)
        r = polygon_measure_sum(37, 2, 14.0, 0.2)
        @test abs(v - r) < 2e-15
    end

    @testset "smooth amplitude integral" begin
        z, phi = 9.4, -0.37
        ref = smooth_continuum(z, phi)
        val = smooth_polygon_sum(512, z, phi)
        @test relative_error(val, ref) < 2e-12
    end

    @testset "geometry-scaled slope" begin
        k, R = 50.0, 1.0
        Ns = [2^i for i in 15:27]
        vals = Float64[]
        for N in Ns
            d = 2 * pi / N
            L = R * sin(pi / N)
            push!(vals, abs(N * intrinsic_coefficient(d, k, L)))
        end
        slope = fit_log_slope(Ns[(end - 5):end], vals[(end - 5):end])
        @test abs(slope - (-0.5)) < 0.03
    end

    @testset "fixed-distance asymptote" begin
        # Independent expectation from eq. (7): N*De(2pi/N; k, L) = 2pi*A*beta_N
        # with A = -exp(-i pi/4) F(2kL) / (2 sqrt(2 pi k)). For k = 2 and
        # L = 0.5, 2kL = 2, so F is the frozen scipy value F(2) used above.
        k, L = 2.0, 0.5
        F2 = 0.909203498997822 + 0.17108658129968907im
        limit = 2 * PI * (-cis(-PI / 4) * F2 / (2 * sqrt(2 * PI * k)))
        c2 = 1 / 12 + 3 / PI^2
        for N in (2^10, 2^14, 2^18, 2^22)
            d = 2 * PI / N
            total = N * intrinsic_coefficient(d, k, L)
            rel = abs(total - limit) / abs(limit)
            # rel = |beta_N - 1| = 2d/pi - c2 d^2 + 0.2351 d^3 + O(d^4); the
            # third-order coefficient 4/pi^3 + 1/(3pi) is below 0.25, and the
            # frozen F(2) plus coefficient roundoff stay below 1e-14.
            @test abs(rel - (2 * d / PI - c2 * d^2)) < 0.25 * d^3 + 1e-14
        end
    end

    @testset "ellipse turning measure" begin
        _, points = ellipse_points(128, 1.4, 0.8)
        delta = polygon_turning_angles(points)
        @test all(>(0), delta)
        @test abs(sum(delta) - 2 * pi) < 5e-14
    end

    @testset "nonuniform partition" begin
        delta, theta = nonuniform_turning_partition(32)
        @test abs(sum(delta) - 2 * pi) < 2e-15
        @test all(>(0), delta)
        ref = smooth_continuum(12.0, 0.2)
        Ns = [16, 24, 32, 48, 64, 96, 128, 192]
        raw = [relative_error(nonuniform_measure_sum(N, 12.0, 0.2; debias=false), ref) for N in Ns]
        deb = [relative_error(nonuniform_measure_sum(N, 12.0, 0.2; debias=true), ref) for N in Ns]
        @test abs(fit_log_slope(Ns[(end - 4):end], raw[(end - 4):end]) - (-1.0)) < 0.12
        @test abs(fit_log_slope(Ns[(end - 4):end], deb[(end - 4):end]) - (-2.0)) < 0.18
    end

    @testset "frozen harmonic N=48" begin
        # Source experiment: m=3, z=24, phi=0.37.
        I = continuum_harmonic(3, 24.0, 0.37)
        S = polygon_measure_sum(48, 3, 24.0, 0.37)
        beta = local_bias_factor(2 * pi / 48)
        raw = relative_error(beta * S, I)
        deb = relative_error(S, I)
        @test abs(raw - 0.07718530795982793) < 5e-14
        @test abs(deb - 7.3888681396476876e-9) < 5e-16
        @test deb / raw < 1e-6
    end

    @testset "input contracts" begin
        @test_throws ArgumentError polygon_alias_sum(2, 0, 1.0, 0.0)
        @test_throws ArgumentError polygon_alias_sum(4, 0, 1.0, 0.0; ell_max=-1)
        @test_throws ArgumentError normalized_seam_sum(2, 0, 1.0, 0.0, 1.0, 1.0)
        @test_throws ArgumentError physical_seam_sum(2, 0, 1.0, 0.0, 1.0, 1.0)
        @test_throws ArgumentError ellipse_continuum_reference(
            1.0, 0.8, 2.0, 0.4; n_dense=0)
        @test_throws ArgumentError local_bias_factor(NaN)
        @test_throws ArgumentError intrinsic_coefficient(0.1, Inf, 1.0)
        @test_throws ArgumentError transition_F(NaN)
        @test_throws ArgumentError transition_F(Inf)
        @test_throws ArgumentError continuum_harmonic(2, Inf, 0.0)
        @test_throws ArgumentError polygon_measure_sum(8, 2, NaN, 0.0)
        @test_throws ArgumentError polygon_alias_sum(8, 2, 1.0, Inf)
        @test_throws ArgumentError normalized_seam_sum(8, 2, 1.0, 0.0, -1.0, 1.0)
        @test_throws ArgumentError normalized_seam_sum(8, 2, 1.0, 0.0, 1.0, 0.0)
        @test_throws ArgumentError error_decomposition(8, 2, 1.0, 0.0, Inf, 1.0)
        @test_throws ArgumentError nonuniform_measure_sum(8, NaN, 0.0)
        @test_throws ArgumentError nonuniform_turning_partition(8; modulation=NaN)
        @test_throws ArgumentError nonuniform_turning_partition(8; modulation=Inf)
        @test_throws ArgumentError ellipse_field_sum(8, 1.0, 0.8, Inf, 0.0)
        @test_throws ArgumentError ellipse_curvature_density([0.0, NaN], 1.0, 0.8)
        @test_throws ArgumentError ellipse_curvature_density(Inf, 1.0, 0.8)
        @test_throws ArgumentError fit_log_slope([1.0, 1.0], [1.0, 2.0])
        @test_throws ArgumentError fit_log_slope([1.0, 2.0], [1.0])
    end

    @testset "interior versus exterior faceting" begin
        # Convex CCW polygons produce positive exterior turning; a CW square is rejected.
        square = [0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0]
        @test abs(sum(polygon_turning_angles(square)) - 2 * pi) < 1e-14
        cw = [0.0 0.0; 0.0 1.0; 1.0 1.0; 1.0 0.0]
        @test_throws ArgumentError polygon_turning_angles(cw)
        nonfinite = copy(square)
        nonfinite[2, 1] = NaN
        @test_throws ArgumentError polygon_turning_angles(nonfinite)
        nonfinite[2, 1] = Inf
        @test_throws ArgumentError polygon_turning_angles(nonfinite)
        # Near-coplanar n = 1 + delta/pi is always an exterior-like coplanar opening
        # in this model (delta > 0). Interior faceting is not a coplanar seam.
        @test local_bias_factor(1e-3) > 0
    end


    @testset "monostatic even half-sum" begin
        MA = CurvatureMeasure.MonostaticAssembly
        # Complete 4-term KP coefficients at monostatic azimuths: the even
        # half-sum equals the intrinsic seam coefficient for ANY monostatic
        # offset eta inside the wedge exterior (beta^- = 0), not only on the
        # exterior bisector.
        k, L = 8.0, 1.4
        for N in (24, 48, 96)
            delta = 2 * pi / N
            alpha = pi + delta
            kernel = intrinsic_coefficient(delta, k, L)
            for eta in (-0.6, -0.2, 0.0, 0.3, 0.9)
                phi = alpha / 2 + eta
                (0 < phi < alpha) || continue
                Ds, Dh = UTDKernels.pec_wedge_DsDh(
                    UTDKernels.Wedge(alpha), UTDKernels.RayAngles(phi, phi), k, L)
                @test abs((Ds + Dh) / 2 - kernel) / abs(kernel) < 1e-12
            end
        end
        # Independent local prefactor agrees with the production kernel.
        for (kk, ll) in ((8.0, 1.3), (18.0, 0.9), (40.0, 0.8))
            @test MA.linear_prefactor_local(kk, ll) ≈ linear_prefactor(kk, ll)
        end
    end

    @testset "monostatic geometry and visibility" begin
        MA = CurvatureMeasure.MonostaticAssembly
        R, d = 1.0, 3.0
        pts = MA.regular_polygon_points(96, R)
        obs = (d, 0.0)
        seams = MA.monostatic_seam_geometry(pts, obs)
        @test length(seams) == 96
        @test all(s -> s.delta > 0, seams)
        @test abs(sum(s.delta for s in seams) - 2 * pi) < 5e-14
        @test all(s -> abs(s.alpha - (pi + s.delta)) < 1e-15, seams)
        # Local-cone visible fraction approaches the tangent-lit fraction.
        frac = count(s -> s.visible, seams) / 96
        @test abs(frac - acos(R / d) / pi) < 2 / 96
        # The arc endpoints approach the tangency interval.
        lo, hi = MA.visible_arc(seams, obs)
        tv = acos(R / d)
        @test abs((hi - lo) - 2 * tv) < 3 * (2 * pi / 96)
        # Bisector direction for a regular polygon is the outward radius.
        s0 = seams[1]
        @test abs(s0.eta) < pi / 2
        @test abs(s0.theta) < 1e-14
        @test abs(s0.rho - (d - R)) < 1e-12
        # Spreading and incident conventions: KP cylindrical 1/sqrt(s).
        @test MA.cylindrical_spreading(2.0) ≈ 1 / sqrt(2.0)
        @test MA.cylindrical_spreading(4.0) ≈ 0.5
        @test abs(MA.incident_cylindrical_wave(8.0, 2.0)) ≈ 1 / sqrt(2.0)
        # Fixed physical visibility half-angle.
        @test MA.monostatic_visibility_half_angle(1.0, 3.0) ≈ acos(1 / 3)
        @test_throws ArgumentError MA.monostatic_visibility_half_angle(1.0, 1.0)
        @test_throws ArgumentError MA.cylindrical_spreading(0.0)
        @test_throws ArgumentError MA.cylindrical_spreading(-1.0)
    end

    @testset "monostatic assembly convergence" begin
        MA = CurvatureMeasure.MonostaticAssembly
        R, d, k = 1.0, 3.0, 8.0
        obs = (d, 0.0)
        Ns = [48, 96, 192, 384]
        errs_raw = Float64[]
        errs_deb = Float64[]
        for N in Ns
            pts = MA.regular_polygon_points(N, R)
            raw = MA.monostatic_even_field(pts, obs, k)
            deb = MA.monostatic_even_field(pts, obs, k; debias=true)
            I = MA.monostatic_arc_integral(
                R, d, 0.0, k, raw.arc_lo, raw.arc_hi; n_quad=1 << 17)
            push!(errs_raw, abs(raw.field - I) / abs(I))
            push!(errs_deb, abs(deb.field - I) / abs(I))
        end
        # Raw is first order in N^{-1}; exact local debiasing is second order.
        @test abs(fit_log_slope(Ns, errs_raw) - (-1.0)) < 0.1
        @test abs(fit_log_slope(Ns, errs_deb) - (-2.0)) < 0.15
        # The per-seam face-transition part is O(1) while the even part is
        # O(delta): it cannot be a curvature-measure atom.
        pts48 = MA.regular_polygon_points(48, R)
        res48 = MA.monostatic_even_field(pts48, obs, k)
        @test res48.max_abs_face > 50 * res48.max_abs_edge
        # Debiased assembly equals the supplied-measure curvature quadrature.
        seams = res48.seams
        deltas = Float64[]
        amps = ComplexF64[]
        for s in seams
            s.visible || continue
            push!(deltas, s.delta)
            push!(amps, MA.linear_prefactor_local(k, s.rho / 2) *
                       MA.incident_cylindrical_wave(k, s.rho) *
                       MA.cylindrical_spreading(s.rho) * cis(-k * s.rho))
        end
        direct = UTDKernels.debiased_curvature_measure_sum(
            deltas, amps; require_closed=false)
        deb48 = MA.monostatic_even_field(pts48, obs, k; debias=true)
        @test deb48.field ≈ direct
    end

    @testset "monostatic fixed-arc reference" begin
        MA = CurvatureMeasure.MonostaticAssembly
        R, d, k = 1.0, 3.0, 8.0
        obs = (d, 0.0)
        t_star = MA.monostatic_visibility_half_angle(R, d)
        I_star = MA.monostatic_arc_integral(
            R, d, 0.0, k, -t_star, t_star; n_quad=1 << 20)
        # Cylindrical-spreading limiting seam magnitude (frozen value).
        @test abs(abs(I_star) - 0.0179102460770) < 2e-11
        # Trapezoid is converged: halving the subintervals changes < 1e-11.
        I_half = MA.monostatic_arc_integral(
            R, d, 0.0, k, -t_star, t_star; n_quad=1 << 19)
        @test abs(I_star - I_half) < 1e-11
        # Fixed-arc error decomposition Q - I* = (Q - I_N) + (I_N - I*).
        for N in (96, 512)
            pts = MA.regular_polygon_points(N, R)
            raw = MA.monostatic_even_field(pts, obs, k)
            deb = MA.monostatic_even_field(pts, obs, k; debias=true)
            I_N = MA.monostatic_arc_integral(
                R, d, 0.0, k, raw.arc_lo, raw.arc_hi; n_quad=1 << 17)
            @test (deb.field - I_star) ≈ (deb.field - I_N) + (I_N - I_star)
            @test (raw.field - I_star) ≈ (raw.field - I_N) + (I_N - I_star)
            @test abs(I_N - I_star) <=
                  abs(deb.field - I_N) + abs(deb.field - I_star) + 1e-15
        end
        # At N = 512 the visibility/domain error dominates the matched-arc
        # quadrature error by more than an order of magnitude.
        pts = MA.regular_polygon_points(512, R)
        res = MA.monostatic_even_field(pts, obs, k)
        deb = MA.monostatic_even_field(pts, obs, k; debias=true)
        I_N = MA.monostatic_arc_integral(
            R, d, 0.0, k, res.arc_lo, res.arc_hi; n_quad=1 << 17)
        @test abs(deb.field - I_star) / abs(I_star) >
              10 * abs(deb.field - I_N) / abs(I_N)
        # Diagnostic scale: limiting seam magnitude is ~6.16% of the exact
        # soft-cylinder return for this normalization and geometry.
        c_norm = MA.hankel_source_normalization(k, d - R)
        u_soft = c_norm * MA.cylinder_line_source_scattered(
            k, R, d, 0.0, d, 0.0; bc=:soft)
        @test abs(abs(I_star) / abs(u_soft) - 0.0616441209308) < 1e-9
    end

    @testset "cylinder eigenfunction series" begin
        MA = CurvatureMeasure.MonostaticAssembly
        k, a, d = 8.0, 1.0, 3.0
        # Converged monostatic soft-cylinder scattered field (frozen value).
        us = MA.cylinder_line_source_scattered(k, a, d, 0.0, d, 0.0; m_max=80)
        @test abs(us - (-0.003479458731875255 + 0.020187511078738486im)) < 5e-14
        # m_max convergence: doubling the truncation changes nothing.
        us2 = MA.cylinder_line_source_scattered(k, a, d, 0.0, d, 0.0; m_max=160)
        @test abs(us - us2) < 1e-12
        # Hard polarization differs but is converged too.
        uh = MA.cylinder_line_source_scattered(k, a, d, 0.0, d, 0.0; bc=:hard)
        @test abs(uh - us) > 1e-3
        uh2 = MA.cylinder_line_source_scattered(
            k, a, d, 0.0, d, 0.0; bc=:hard, m_max=160)
        @test abs(uh - uh2) < 1e-12
        # Lossless PEC boundary data give |R_m| < 1 for every order.
        ka = k * a
        for m in 0:12
            Jm = SpecialFunctions.besselj(m, ka)
            Ym = SpecialFunctions.bessely(m, ka)
            @test abs(Jm / complex(Jm, -Ym)) < 1.0
        end
        # Source-strength normalization is finite and phase-consistent.
        cn = MA.hankel_source_normalization(k, d - a)
        @test isfinite(cn) && abs(cn) > 1.0
    end

    @testset "released adapter delegates production to UTDKernels" begin
        delta, k, L = 0.05, 25.0, 0.7
        @test intrinsic_coefficient(delta, k, L) ==
              UTDKernels.intrinsic_seam_coefficient(delta, k, L)
        @test local_bias_factor(delta) == UTDKernels._curvature_local_bias(delta)
        @test continuum_harmonic(3, 24.0, 0.37) ==
              UTDKernels.curvature_continuum_harmonic(3, 24.0, 0.37)
        turning, theta = nonuniform_turning_partition(32)
        amplitudes = smooth_amplitude.(theta) .* exp.(-im .* 12 .* cos.(theta .- 0.2))
        @test nonuniform_measure_sum(32, 12.0, 0.2; debias=true) ==
              UTDKernels.debiased_curvature_measure_sum(turning, amplitudes)
    end
end

end
