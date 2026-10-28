module Paper170CanonicalTests
import UTDKernels
using Test
using FastGaussQuadrature: gausslegendre
using QuadGK: quadgk
using UTDKernels.ShadowSensitivity
const UTDSensitivity = UTDKernels.ShadowSensitivity
import UTDKernels

relerr(a, b; floor=1e-14) = abs(a - b) / max(abs(b), floor)

@noinline _adapter_switch_allocation_probe() = canonical_switch(0.4 + 0.2im)
@noinline _adapter_kernel_allocation_probe() = canonical_kernel(0.4 + 0.2im)
@noinline _adapter_scaled_allocation_probe() = scaled_kernel(0.4, 64.0)
@noinline _adapter_multiplier_allocation_probe() = fourier_multiplier(2.0, 64.0)
@noinline _adapter_truncated_allocation_probe() =
    truncated_multiplier(2.0, 64.0, 4)
@noinline _adapter_pullback_allocation_probe() = pullback_approximations(
    1.0, 0.2, -2.0, 0.3, -12.0, 1.0, 0.6, 0.0, 0.0, 64.0,
)
@noinline _adapter_bound_allocation_probe() =
    bandlimit_remainder_bound(4.0, 64.0, 3, 1.2)
@noinline _adapter_halfplane_allocation_probe() = halfplane_d2(0.4, 64.0)

@testset "UTDSensitivity" begin
    @testset "input contracts" begin
        @test_throws ArgumentError scaled_kernel(0.0, 0.0)
        @test_throws ArgumentError scaled_kernel(Inf, 1.0)
        @test_throws ArgumentError fourier_multiplier(0.0, -1.0)
        @test_throws ArgumentError truncated_multiplier(1.0, 2.0, 0)
        @test_throws ArgumentError canonical_switch_reference(0.0; precision=64)
        @test_throws ArgumentError halfplane_derivatives_reference(
            0.0, 1.0; step=0.0)
        @test_throws ArgumentError bump_asymmetric(0.0; radius=0.0)
        @test_throws DomainError curved_inverse(-2.0, 0.6)
        @test curved_inverse(0.4, 0.0) == 0.4
        @test_throws ArgumentError composite_phase_quadrature(
            identity, 1.0, -1.0, 2.0)
        @test_throws ArgumentError composite_phase_quadrature(
            identity, -1.0, 1.0, 2.0; phase_increment=0.0)
        @test_throws ArgumentError composite_phase_quadrature(
            identity, -1.0, 1.0, 2.0; order=0)
        @test_throws ArgumentError composite_phase_quadrature(
            identity, -10.0, 10.0, 1e6; max_panels=8)
        @test_throws ArgumentError bandlimited_action(
            [1.0], [0.0, 1.0], [1.0], 2.0)
        @test_throws ArgumentError bandlimited_action(
            [1.0, 2.0, 3.0], [-1.0, 0.2, 1.0], ones(3), 2.0)
        @test_throws ArgumentError bandlimit_remainder_bound(1.0, 1.0, 0, 1.0)
        @test_throws ArgumentError pullback_approximations(
            1, 0, 0, 0, 0, 0.0, 0, 0, 0, 1.0)
        @test_throws ArgumentError translated_objective_action(0.0, 1.0, 0.0)
        @test_throws ArgumentError relative_error(1.0, 1.0; floor=0.0)
        @test_throws ArgumentError scaled_kernel(0.1, big"1e-1000")
        # q^2 = 1e310 and xi^2/4 = 2.5e309 overflow binary64, but the exact
        # kernel (modulus 1/sqrt(pi)) and multiplier (modulus one) are
        # representable: UTDKernels reduces the phase from the stored input in
        # a wider workspace. Independent 4096-bit oracle (exact square of the
        # stored double; 8192 bits give identical values). Tolerance 4*eps(1.0)
        # allows a few binary64 ulps; a phase formed in binary64 (NaN), a
        # conjugated phase, or a lost e^{i*pi/4} factor differs by O(0.1-1).
        kernel_oracle, multiplier_oracle = setprecision(BigFloat, 4096) do
            q = BigFloat(1e155)
            (ComplexF64(exp(im * BigFloat(π) / 4) * exp(-im * q * q) /
                        sqrt(BigFloat(π))),
             ComplexF64(exp(im * q * q / 4)))
        end
        @test abs(canonical_kernel(1e155) - kernel_oracle) <= 4eps(1.0)
        @test abs(fourier_multiplier(1e155, 1.0) - multiplier_oracle) <= 4eps(1.0)
        # The truncated polynomial value itself (|z|^2/2 with z = 2.5e309) is
        # not representable, so this path must still fail closed.
        @test_throws DomainError truncated_multiplier(1e155, 1.0, 3)
        large_correction = translated_first_correction(1e155, 64.0, 0.75)
        @test isfinite(real(large_correction)) && isfinite(imag(large_correction))
        @test imag(large_correction) ≈ 3.90625e307 rtol=2e-15
    end

    @testset "scalar and array paths" begin
        q = range(-2.0, 2.0; length=17)
        @test canonical_switch(q) ≈ canonical_switch.(q) atol=0 rtol=0
        @test canonical_kernel(q) ≈ canonical_kernel.(q) atol=0 rtol=0
        @test scaled_kernel(q, 8.0) ≈ scaled_kernel.(q, 8.0) atol=0 rtol=0
        @test fourier_multiplier(q, 8.0) ≈
              fourier_multiplier.(q, 8.0) atol=0 rtol=0
        @test halfplane_d2(q, 8.0) ≈ halfplane_d2.(q, 8.0) atol=0 rtol=0
        @test bump_asymmetric(q; radius=2.5) ≈
              bump_asymmetric.(q; radius=2.5) atol=0 rtol=0
        @test curved_coordinate(q, 0.2) ≈
              curved_coordinate.(q, 0.2) atol=0 rtol=0
    end

    @testset "canonical derivative and symmetry" begin
        step = 1e-6
        for q in (-2.1, -0.3, 0.0, 0.7, 2.4)
            numerical = (canonical_switch(q + step) -
                         canonical_switch(q - step)) / (2step)
            @test abs(numerical - canonical_kernel(q)) < 3e-9
            @test canonical_kernel(q) ≈ canonical_kernel(-q) atol=2e-14
        end
        # The paper phase is exp(+i*pi/4), not its complex conjugate.
        @test abs(canonical_kernel(0.0) - exp(-im * π / 4) / sqrt(π)) > 0.7
        for q in (-3.0, 0.0, 2.4)
            reference = ComplexF64(canonical_switch_reference(q; precision=192))
            @test abs(canonical_switch(q) - reference) < 3e-15
        end
    end

    @testset "exact multiplier and series" begin
        xi = range(-10.0, 10.0; length=101)
        for kappa in (7.0, 32.0, 128.0)
            values = fourier_multiplier(xi, kappa)
            @test maximum(abs.(abs.(values) .- 1)) < 2e-15
            @test maximum(abs.(values - conj.(values))) > 0.1
        end
        xi_small = range(-2.0, 2.0; length=21)
        @test maximum(abs.(fourier_multiplier(xi_small, 100.0) -
                           truncated_multiplier(xi_small, 100.0, 4))) < 1e-9
        # Reversing the multiplier phase is a visible mutation.
        xi0, kappa0 = 1.7, 16.0
        wrong = exp(-im * xi0^2 / (4kappa0))
        @test abs(fourier_multiplier(xi0, kappa0) - wrong) > 0.08
    end

    @testset "Gaussian action and negative-side panel closure" begin
        sigma = 0.8
        for kappa in (20.0, 64.0, 256.0)
            amplitude = y -> exp(-y^2 / (2sigma^2))
            numerical = composite_phase_quadrature(
                amplitude, -8.0, 8.0, kappa;
                phase_increment=1.5, order=16)
            @test abs(numerical - gaussian_action(kappa, sigma)) < 3e-12
        end
        # Regression: floating roundoff in the last negative y^2 panel once
        # left a gap next to zero for this panel count.
        kappa = 64.0
        numerical = composite_phase_quadrature(
            y -> exp(-y^2 / (2sigma^2)), -8.0, 8.0, kappa;
            phase_increment=1.5, order=16)
        @test abs(numerical - gaussian_action(kappa, sigma)) < 3e-12
    end

    @testset "half-plane derivatives and high-precision fixtures" begin
        for kappa in (4.0, 16.0, 64.0)
            @test halfplane_q(0.0, kappa) == 0.0
            @test halfplane_q_prime(0.0, kappa) ≈ sqrt(kappa / 2) atol=0 rtol=0
            @test halfplane_q_second(0.0, kappa) == 0.0
            s = 0.37
            hq = 1e-4
            q_first = (halfplane_q(s + hq, kappa) -
                       halfplane_q(s - hq, kappa)) / (2hq)
            q_second = (halfplane_q(s + hq, kappa) -
                        2halfplane_q(s, kappa) +
                        halfplane_q(s - hq, kappa)) / hq^2
            @test abs(q_first - halfplane_q_prime(s, kappa)) < 2e-8
            @test abs(q_second - halfplane_q_second(s, kappa)) < 2e-7
        end
        step1 = 2e-6
        for kappa in (4.0, 16.0, 64.0), s in (-1.0, -0.2, 0.3, 1.1)
            numerical = (halfplane_demodulated(s + step1, kappa) -
                         halfplane_demodulated(s - step1, kappa)) / (2step1)
            @test abs(numerical - halfplane_d1(s, kappa)) < 3e-8
        end
        step2 = 2e-4
        for kappa in (4.0, 16.0), s in (-0.8, 0.1, 0.9)
            numerical = (halfplane_demodulated(s + step2, kappa) -
                         2halfplane_demodulated(s, kappa) +
                         halfplane_demodulated(s - step2, kappa)) / step2^2
            @test abs(numerical - halfplane_d2(s, kappa)) < 6e-5
        end
        fixtures = (
            (4.0, -2.35,
             0.2974584498638949321 + 0.0784081456272756421im,
             0.1328007019616315931 + 0.9403590588531131627im),
            (16.0, -0.4,
             1.3889352139157096266 - 0.7189087542641516803im,
             4.620075638973616484 + 8.581164567682473722im),
            (64.0, 0.7,
             -0.3813195929747966851 - 2.9736951458217062155im,
             -122.5356523425008371 + 16.2645226137989562im),
            (256.0, 2.1,
             1.207184229203541203 - 2.937668899384709140im,
             -650.2230187502343783 - 264.2048582993608028im),
        )
        for (kappa, s, d1, d2) in fixtures
            @test relerr(halfplane_d1(s, kappa), d1) < 2e-13
            @test relerr(halfplane_d2(s, kappa), d2) < 2e-13
            reference_d1, reference_d2 = halfplane_derivatives_reference(
                s, kappa; precision=192, step=1e-5)
            @test relerr(ComplexF64(reference_d1), d1) < 2e-12
            @test relerr(ComplexF64(reference_d2), d2) < 2e-12
        end
    end

    @testset "curved coordinate inverse" begin
        for gamma in (0.2, 0.6, 1.0)
            s = range(-0.6, 0.6; length=31)
            @test maximum(abs.(curved_inverse(curved_coordinate(s, gamma), gamma) .- s)) < 3e-15
        end
        @test curved_coordinate(1e-200, 1e308) ≈ 5e-93 rtol=2e-15
        @test curved_inverse(1.0, 1e308) ≈ sqrt(2.0) * 1e-154 rtol=2e-15
    end

    @testset "finite-bandwidth remainder bound" begin
        xi_general = [-1.0, 0.0, 1.0]
        weights_general = [0.5, 1.0, 0.5]
        spectrum_general = ComplexF64[1 + 2im, 3 + 4im, 5 + 6im]
        kappa_general = 7.0
        expected_general = sum(reverse(spectrum_general) .* weights_general .*
                               fourier_multiplier(xi_general, kappa_general)) / (2π)
        @test bandlimited_action(spectrum_general, xi_general,
                                 weights_general, kappa_general) ≈
              expected_general atol=1e-15
        explicit_reflection = ComplexF64[7 + 1im, 8 + 2im, 9 + 3im]
        xi_nonsymmetric = [0.0, 0.4, 1.7]
        expected_explicit = sum(explicit_reflection .* weights_general .*
                                fourier_multiplier(xi_nonsymmetric, kappa_general)) / (2π)
        @test bandlimited_action(spectrum_general, xi_nonsymmetric,
                                 weights_general, kappa_general;
                                 reflected_spectrum=explicit_reflection) ≈
              expected_explicit atol=1e-15

        omega = 3.0
        nodes, gauss_weights = gausslegendre(256)
        xi = omega .* nodes
        weights = omega .* gauss_weights
        spectrum = max.(0.0, 1 .- (xi ./ omega).^2).^4 .*
                   (1 .+ 0.2im .* sin.(π .* xi ./ omega))
        spectral_l1 = sum(weights .* abs.(spectrum)) / (2π)
        for kappa in (8.0, 32.0, 128.0)
            exact = bandlimited_action(spectrum, xi, weights, kappa)
            for order_M in 1:3
                approximation = bandlimited_action(
                    spectrum, xi, weights, kappa; terms=order_M)
                bound = bandlimit_remainder_bound(
                    omega, kappa, order_M, spectral_l1)
                @test abs(exact - approximation) <= bound * (1 + 2e-14)
            end
        end
        @test bandlimit_remainder_bound(0.0, 2.0, 3, 1.0) == 0.0
        @test bandlimit_remainder_bound(2.0, 2.0, 3, 0.0) == 0.0
    end

    @testset "real smoother phase obstruction" begin
        xi, kappa = 1.7, 1e6
        utd = (fourier_multiplier(xi, kappa) - 1) * kappa
        gaussian = (real_gaussian_multiplier(xi, kappa) - 1) * kappa
        @test abs(imag(utd)) > 1e-3
        @test abs(imag(gaussian)) < 1e-14
        @test abs(utd - gaussian) > 1e-3
    end

    @testset "half-plane distributional orders" begin
        radius = 1.2
        lo = sqrt(2.0) * sin(-radius / 2)
        hi = sqrt(2.0) * sin(radius / 2)
        f0 = 1.0
        f1 = 0.0
        f2 = -2 / radius^2
        f3 = 0.0
        f4 = -12 / radius^4
        a = inv(sqrt(2.0))
        b = 0.0
        c = -inv(4sqrt(2.0))
        d = 0.0
        kappas = [32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0]
        errors = [Float64[] for _ in 1:3]
        for kappa in kappas
            amplitude = y -> begin
                s = 2asin(y / sqrt(2.0))
                bump_asymmetric(s; radius=radius, alpha=0.0)
            end
            exact = composite_phase_quadrature(
                amplitude, lo, hi, kappa;
                phase_increment=1.5, order=16)
            approximations = pullback_approximations(
                f0, f1, f2, f3, f4, a, b, c, d, kappa)
            for index in 1:3
                push!(errors[index], abs(approximations[index] - exact))
            end
        end
        fit = (length(kappas) - 3):length(kappas)
        @test fitted_log_slope(kappas[fit], errors[1][fit]) ≈ -1.0 atol=0.08
        @test fitted_log_slope(kappas[fit], errors[2][fit]) ≈ -2.0 atol=0.08
        @test fitted_log_slope(kappas[fit], errors[3][fit]) ≈ -3.0 atol=0.15
    end

    @testset "curvature correction" begin
        gamma, radius, alpha = 0.6, 0.8, 0.35
        ylo = curved_coordinate(-radius, gamma)
        yhi = curved_coordinate(radius, gamma)
        f0 = 1.0
        f1 = alpha / radius
        f2 = -2 / radius^2
        f3 = -6alpha / radius^3
        f4 = -12 / radius^4
        kappas = [64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0]
        width_errors = Float64[]
        full_errors = Float64[]
        for kappa in kappas
            amplitude = y -> bump_asymmetric(
                curved_inverse(y, gamma); radius=radius, alpha=alpha)
            exact = composite_phase_quadrature(
                amplitude, ylo, yhi, kappa;
                phase_increment=1.5, order=16)
            width_only = 1 - im * f2 / (4kappa)
            _, complete_first, _ = pullback_approximations(
                f0, f1, f2, f3, f4, 1.0, gamma, 0.0, 0.0, kappa)
            push!(width_errors, abs(width_only - exact))
            push!(full_errors, abs(complete_first - exact))
        end
        @test all(full_errors .< width_errors)
        @test full_errors[end] < full_errors[1] / 100
        @test fitted_log_slope(kappas[end-3:end], width_errors[end-3:end]) ≈
              -1.0 atol=0.08
        @test fitted_log_slope(kappas[end-3:end], full_errors[end-3:end]) ≈
              -2.0 atol=0.08
    end

    @testset "translated-boundary objective" begin
        kappa, sigma = 64.0, 0.75
        for nu in (0.0, 2.0, 8.0, 16.0)
            exact = translated_objective_action(nu, kappa, sigma)
            first = translated_first_correction(nu, kappa, sigma)
            gaussian = translated_gaussian_action(nu, kappa, sigma)
            @test isfinite(real(exact)) && isfinite(imag(exact))
            @test isreal(gaussian)
            if nu <= 2
                @test abs(first - exact) < abs(1 - exact)
            end
        end
        @test abs(imag(translated_objective_action(8.0, kappa, sigma))) > 0.05
        @test imag(translated_gaussian_action(8.0, kappa, sigma)) == 0.0
        @test translated_objective_action(1e155, kappa, sigma) === 0.0 + 0.0im
        @test translated_gaussian_action(1e155, kappa, sigma) === 0.0
        nearby_nu = 20.0
        alpha = inv(2sigma^2)
        direct = exp(im * π / 4) * sqrt(kappa) / sqrt(alpha + im * kappa) *
                 exp(-nearby_nu^2 / (4(alpha + im * kappa)))
        @test translated_objective_action(nearby_nu, kappa, sigma) ≈
              direct rtol=2e-15
        direct_gaussian = sqrt(kappa / (kappa + alpha)) *
                          exp(-nearby_nu^2 / (4(kappa + alpha)))
        @test translated_gaussian_action(nearby_nu, kappa, sigma) ≈
              direct_gaussian rtol=2e-15

        setprecision(BigFloat, 1024) do
            for (tiny_kappa, tiny_sigma) in (
                    (nextfloat(0.0), 2.0^-512),
                    (floatmax(Float64), 1e-200),
                )
                kb = BigFloat(tiny_kappa)
                sb = BigFloat(tiny_sigma)
                oracle = sqrt(kb / (kb + inv(2sb^2)))
                actual = translated_gaussian_action(0.0, tiny_kappa, tiny_sigma)
                @test actual > 0
                @test actual == Float64(oracle)
            end
        end
        @test relative_error(2.0, 1.0) == 1.0
    end

    @testset "released production adapters delegate to UTDKernels" begin
        for coordinate in (-1.2, 0.0, 0.7, 0.4 + 0.2im)
            @test canonical_switch(coordinate) ==
                  UTDKernels.shadow_switch(coordinate)
            @test canonical_kernel(coordinate) ==
                  UTDKernels.shadow_sensitivity_kernel(coordinate)
        end
        for (s, kappa) in ((-0.7, 4.0), (0.4, 64.0), (1.1, 256.0))
            @test scaled_kernel(s, kappa) ==
                  UTDKernels.shadow_sensitivity_kernel(s, kappa)
            @test fourier_multiplier(s, kappa) ==
                  UTDKernels.shadow_sensitivity_multiplier(s, kappa)
            @test truncated_multiplier(s, kappa, 4) ==
                  UTDKernels.shadow_sensitivity_multiplier(
                      s, kappa; terms=4,
                  )
            @test halfplane_q(s, kappa) ==
                  UTDKernels._shadow_halfplane_coordinate(s, kappa)
            @test halfplane_d1(s, kappa) ==
                  UTDKernels._shadow_halfplane_first(s, kappa)
            @test halfplane_d2(s, kappa) ==
                  UTDKernels._shadow_halfplane_second(s, kappa)
        end
        paper_pullback = pullback_approximations(
            1.0, 0.2, -2.0, 0.3, -12.0, 1.0, 0.6, 0.0, 0.0, 64.0,
        )
        central_pullback = UTDKernels.shadow_sensitivity_pullback(
            1.0, 0.2, -2.0, 0.3, -12.0, 1.0, 0.6, 0.0, 0.0, 64.0,
        )
        @test paper_pullback == Tuple(central_pullback)
        @test bandlimit_remainder_bound(4.0, 64.0, 3, 1.2) ==
              UTDKernels._shadow_sensitivity_remainder_bound(
                  4.0, 64.0, 3, 1.2,
              )

        _adapter_switch_allocation_probe()
        _adapter_kernel_allocation_probe()
        _adapter_scaled_allocation_probe()
        _adapter_multiplier_allocation_probe()
        _adapter_truncated_allocation_probe()
        _adapter_pullback_allocation_probe()
        _adapter_bound_allocation_probe()
        _adapter_halfplane_allocation_probe()
        @test @allocated(_adapter_switch_allocation_probe()) == 0
        @test @allocated(_adapter_kernel_allocation_probe()) == 0
        @test @allocated(_adapter_scaled_allocation_probe()) == 0
        @test @allocated(_adapter_multiplier_allocation_probe()) == 0
        @test @allocated(_adapter_truncated_allocation_probe()) == 0
        @test @allocated(_adapter_pullback_allocation_probe()) == 0
        @test @allocated(_adapter_bound_allocation_probe()) == 0
        @test @allocated(_adapter_halfplane_allocation_probe()) == 0
    end

    @testset "sommerfeld input contracts" begin
        @test_throws ArgumentError sommerfeld_cosine_branch(0.0, 0.0)
        @test_throws ArgumentError sommerfeld_field(Inf, 0.5, 8.0)
        @test_throws ArgumentError sommerfeld_incident(0.0, NaN, 8.0)
        @test_throws ArgumentError field_sensitivity_exact(0.0, 0.8, 3.9, 0.8)
        @test_throws ArgumentError field_sensitivity_exact(
            8.0, 0.8, 3.9, 0.8; npanels=4)
        @test_throws ArgumentError field_sensitivity_exact(
            8.0, 0.8, 3.9, -0.8)
        @test_throws ArgumentError field_transition_action(
            8.0, 0.8, 3.9, 0.8; kernel=:notakernel)
        @test_throws ArgumentError bump_jets(0.0, 0.0, 0.8, -1)
        @test_throws ArgumentError sommerfeld_branch_jets(2, 8.0, 0, 0.8, 3.9)
        @test_throws ArgumentError misfit_truncated_action(
            8.0, 0.8, 3.9, 0.8, 0)
        @test_throws ArgumentError carrier_action_exact(8.0, 0.0)
        @test_throws ArgumentError carrier_action_truncated(8.0, 0)
    end

    @testset "sommerfeld field and demodulated derivative" begin
        p, kappa = 0.8, 64.0
        # branch reconstruction: V = u_c(s-p) - u_c(s+p)
        for s in (0.4, 1.7, 3.2, 4.0)
            direct = sommerfeld_cosine_branch(s - p, kappa) -
                     sommerfeld_cosine_branch(s + p, kappa)
            @test sommerfeld_field(s, p, kappa) == direct
            @test sommerfeld_demodulated(s, p, kappa) ≈
                  direct * conj(sommerfeld_incident(s, p, kappa))
        end
        # analytic dD/dp agrees with centered finite differences of D
        step = 1e-6
        for s in (3.5, sommerfeld_boundary(p), 4.4, 2.2)
            fd = (sommerfeld_demodulated(s, p + step, kappa) -
                  sommerfeld_demodulated(s, p - step, kappa)) / (2step)
            @test abs(fd - sommerfeld_demodulated_dp(s, p, kappa)) < 5e-4
        end
        # window support and value
        @test receiver_window(0.0, 0.0, 0.8) ≈ 1.0
        @test receiver_window(0.9, 0.0, 0.8) == 0.0
        @test sommerfeld_boundary(p) ≈ p + π
        @test sommerfeld_inverse_map(0.0, p) ≈ p + π
    end

    @testset "sommerfeld observable decomposition" begin
        p, radius = 0.8, 0.8
        center = sommerfeld_boundary(p)
        # transition action + smooth remainder = exact observable
        for kappa in (32.0, 128.0)
            s1 = field_sensitivity_exact(kappa, p, center, radius)
            s1_split = field_transition_action(kappa, p, center, radius) +
                       field_sensitivity_remainder(kappa, p, center, radius)
            @test abs(s1_split - s1) < 1e-10 * max(1.0, abs(s1))
            j2 = misfit_sensitivity_exact(kappa, p, center, radius)
            j2_split = real(misfit_transition_action(
                kappa, p, center, radius)) +
                real(misfit_sensitivity_remainder(kappa, p, center, radius))
            @test abs(j2_split - j2) < 1e-10 * max(1.0, abs(j2))
        end
        # finite-difference oracle for the misfit sensitivity (adaptive
        # quadrature: fixed low-order grids alias the kappa-oscillatory
        # misfit integrand)
        kappa = 64.0
        misfit(pp) = 0.5 * quadgk(
            s -> receiver_window(s, center, radius) *
                 abs2(sommerfeld_demodulated(s, pp, kappa) - 1),
            center - radius, center + radius; rtol=1e-11)[1]
        oracle = (misfit(p + 1e-6) - misfit(p - 1e-6)) / 2e-6
        @test abs(oracle - misfit_sensitivity_exact(
            kappa, p, center, radius)) < 1e-5
        # Dirac truncation converges for the demodulated observable
        errs = [abs(field_truncated_action(
                        kappa, p, center, radius, 1;
                        remainder=field_sensitivity_remainder(
                            kappa, p, center, radius)) -
                    field_sensitivity_exact(kappa, p, center, radius))
                for kappa in (64.0, 256.0, 1024.0)]
        @test errs[2] < errs[1] / 3.5
        @test errs[3] < errs[2] / 3.5
        # second-order truncation beats the Dirac limit at large kappa
        @test abs(field_truncated_action(
                      1024.0, p, center, radius, 3;
                      remainder=field_sensitivity_remainder(
                          1024.0, p, center, radius)) -
                  field_sensitivity_exact(1024.0, p, center, radius)) <
              errs[3] / 50
    end

    @testset "bump and composed jets" begin
        # analytic bump derivatives match central differences
        center, radius = 3.9, 0.8
        for s in (3.4, 3.9, 4.3)
            jets = bump_jets(s, center, radius, 2)
            @test jets[1] ≈ receiver_window(s, center, radius)
            h = 1e-4
            d1 = (receiver_window(s + h, center, radius) -
                  receiver_window(s - h, center, radius)) / (2h)
            d2 = (receiver_window(s + h, center, radius) -
                  2receiver_window(s, center, radius) +
                  receiver_window(s - h, center, radius)) / h^2
            @test abs(jets[2] - d1) < 3e-6 * max(1.0, abs(d1))
            @test abs(jets[3] - d2) < 3e-4 * max(1.0, abs(d2))
        end
        # outside the support all derivatives vanish
        @test all(==(0.0), bump_jets(6.0, center, radius, 4))
        # composed-map identities: (w o h)'' = 2 w'', (w o h)'''' = 4 w'''' + 4 w''
        p = 0.8
        sstar = sommerfeld_boundary(p)
        wd = bump_jets(sstar, center, radius, 4)
        cj = window_composed_jets(p, center, radius)
        @test cj.f0 ≈ wd[1]
        @test cj.d2 ≈ 2 * wd[3]
        @test cj.d4 ≈ 4 * wd[5] + 4 * wd[3]
        # misfit local jets agree with a small-step complex difference of F_2
        kappa = 16.0
        F2(s) = receiver_window(s, center, radius) *
                conj(sommerfeld_demodulated(s, p, kappa) - 1)
        h = 3e-4
        fd2 = (F2(sstar + h) - 2F2(sstar) + F2(sstar - h)) / h^2
        jets = misfit_local_jets(kappa, p, center, radius)
        @test jets[1] ≈ F2(sstar)
        @test abs(jets[3] - fd2) < 1e-2 * max(1.0, abs(fd2))
    end

    @testset "carrier uniformity counterexample" begin
        for kappa in (16.0, 64.0, 256.0, 1024.0)
            exact = carrier_action_exact(kappa, 0.5)
            @test abs(exact) / sqrt(kappa) ≈ 1 / sqrt(π) rtol=1e-14
            t1 = carrier_action_truncated(kappa, 1)
            t2 = carrier_action_truncated(kappa, 2)
            t3 = carrier_action_truncated(kappa, 3)
            @test t1 ≈ exp(-im * kappa)
            @test t2 ≈ 1.5 * exp(-im * kappa)
            @test t3 ≈ 1.875 * exp(-im * kappa)
            @test abs(t1 - exact) / abs(exact) > 0.3
        end
    end

    @testset "transformed-test remainder bound" begin
        # Reviewer counterexample: g = 0.1 s, f = exp(-s^2/2), kappa = 100.
        # The physical-spectrum bound (0.0025) is violated; the bound on the
        # complete transformed test F(y) = exp(-50 y^2) (0.25) holds.
        kappa = 100.0
        exact = gaussian_action(kappa, 0.1)
        dirac_err = abs(1 - exact)
        @test exact ≈ 0.920442065259926 + 0.217286896751640im rtol=1e-12
        @test dirac_err ≈ 0.231393734746781 rtol=1e-9
        @test dirac_err > 0.0025
        @test dirac_err < 0.25
    end

    @testset "transformed-test bound: compact spectrum and nonlinear map" begin
        # Compact C-infinity spectrum exp(-1/(1-xi^2)) on |xi| < 1 under the
        # linear map g = 0.1 s: physical bound fails, transformed bound holds.
        kappa = 100.0
        nodes, wts = gausslegendre(512)
        spectrum = [abs(x) < 1 ? exp(-1 / (1 - x^2)) : 0.0 for x in nodes]
        spectrum ./= (sum(wts .* spectrum) / (2π))
        exact_spec = sum(wts .* spectrum .*
                         exp.(im .* (nodes ./ 0.1).^2 ./ (4kappa))) / (2π)
        err_spec = abs(1 - exact_spec)
        @test err_spec ≈ 0.03950303437 rtol=1e-6
        @test err_spec > 1.0 / (4kappa)          # physical bound 0.0025
        @test err_spec < 100.0 / (4kappa)        # transformed bound 0.25

        # Nonlinear scaled map g(s) = a (s + 0.3 s^2) with probe B_1.2: the
        # transformed test F = f o h is measured spectrally; the canonical
        # bound holds while the physical one is violated at a = 0.2.
        radius, gamma = 1.2, 0.6
        probe(s) = bump_asymmetric(s; radius=radius, alpha=0.0)
        a_nl = 0.2
        g_nl(s) = a_nl * curved_coordinate(s, gamma)
        h_nl(y) = curved_inverse(y / a_nl, gamma)
        F(y) = probe(h_nl(y))
        ylo, yhi = g_nl(-radius), g_nl(radius)
        exact_nl = composite_phase_quadrature(
            F, ylo, yhi, kappa; phase_increment=1.2, order=16)
        err_nl = abs(1 - exact_nl)
        @test err_nl ≈ 0.21288 rtol=2e-3
        mom_f = UTDSensitivity.spectral_moment(probe, -radius, radius)
        mom_F = UTDSensitivity.spectral_moment(F, ylo, yhi)
        @test mom_F.moment / mom_f.moment ≈ 146.85 rtol=2e-2
        @test err_nl > mom_f.moment / (4kappa)   # physical bound violated
        @test err_nl < mom_F.moment / (4kappa)   # transformed bound holds
    end

    @testset "corrected Sommerfeld field" begin
        p, radius = 0.8, 0.8
        center = sommerfeld_boundary(p)
        kappa = 64.0
        # Regression anchors for the positive-carrier field
        # u_c(x, kappa) = exp(+i kappa cos x) W(sqrt(2 kappa) cos(x/2)).
        s1 = field_sensitivity_exact(kappa, p, center, radius)
        j2 = misfit_sensitivity_exact(kappa, p, center, radius)
        @test s1 ≈ 1.0026648845 + 0.0173937061im rtol=1e-8
        @test j2 ≈ -0.494304707346 rtol=1e-9
        rem = misfit_sensitivity_remainder(kappa, p, center, radius)
        err2 = abs(misfit_truncated_action(
                       kappa, p, center, radius, 3; remainder=rem) - j2)
        @test err2 ≈ 0.0439378 rtol=1e-4
        # Helmholtz residual of the corrected branch: five-point polar
        # Laplacian of exp(+i k r cos x) W(sqrt(2 k r) cos(x/2)) at
        # r = 1, k = 4, x = 1.1 stays at the differentiation error floor,
        # while the negative-carrier branch has residual
        # -2 i k q C W'(q)/r = (-0.3614 - 0.5762 i) k^2.
        uf(r, x, k) = sommerfeld_cosine_branch(x, k * r)
        function polar_lap(r, x, k, h)
            u0 = uf(r, x, k)
            ur = (uf(r - 2h, x, k) - 8uf(r - h, x, k) +
                  8uf(r + h, x, k) - uf(r + 2h, x, k)) / (12h)
            urr = (-uf(r + 2h, x, k) + 16uf(r + h, x, k) - 30u0 +
                   16uf(r - h, x, k) - uf(r - 2h, x, k)) / (12h^2)
            uxx = (-uf(r, x + 2h, k) + 16uf(r, x + h, k) - 30u0 +
                   16uf(r, x - h, k) - uf(r, x - 2h, k)) / (12h^2)
            return urr + ur / r + uxx / r^2
        end
        res = (polar_lap(1.0, 1.1, 4.0, 1e-3) + 16uf(1.0, 1.1, 4.0)) / 16
        @test abs(res) < 1e-8
        # Dirichlet boundary values on the two screen faces
        @test abs(sommerfeld_field(0.0, p, kappa)) == 0.0
        @test abs(sommerfeld_field(2π, p, kappa)) < 1e-12
        # Lit/shadow asymptotics of the corrected branch at large kappa
        @test abs(sommerfeld_cosine_branch(1.5, 4096.0) /
                  exp(im * 4096.0 * cos(1.5)) - 1) < 0.05
        @test abs(sommerfeld_cosine_branch(4.6, 4096.0)) < 0.02
    end
end

end
