module Paper168CanonicalTests
import UTDKernels
using Test
using UTDKernels.NullUniform
const NullUniformMoments = UTDKernels.NullUniform
using FastGaussQuadrature
using SpecialFunctions: erfcx, gamma
import UTDKernels

# Independent oracles: closed-form Gaussian moments, Faddeeva identity,
# derivative identity, QuadGK real-line integral, BigFloat recurrence,
# Gauss–Hermite n=1,2 analytic rules, and Python binary64 spot checks.

relerr(a, b; floor=1e-14) = abs(a - b) / max(abs(b), floor)

const NULL_ADAPTER_COEFFICIENTS = ComplexF64[0.08, 1.0, 0.35, -0.15, 0.08]
const NULL_ADAPTER_BASE = ComplexF64[1.0, 0.2, -0.05]
@noinline null_adapter_moderate_probe() = faddeeva_moments(0.7 + 0.4im, 4)
@noinline null_adapter_large_probe() = faddeeva_moments(1e7 + 0.4im, 4)
@noinline null_adapter_hierarchy_probe() =
    hierarchy_value(250.0, 0.9 + 0.45im, NULL_ADAPTER_COEFFICIENTS)
@noinline null_adapter_shifted_probe() =
    shifted_null_hierarchy(900.0, 0.6 + 0.5im, -1.1, 3, NULL_ADAPTER_BASE)

@testset "NullUniformMoments" begin
    @testset "Gaussian moments: even gamma law and odd zeros" begin
        @test gaussian_moment(0) ≈ sqrt(π) atol = 1e-15
        @test gaussian_moment(1) == 0.0
        @test gaussian_moment(2) ≈ gamma(1.5) atol = 1e-15
        @test gaussian_moment(3) == 0.0
        @test gaussian_moment(4) ≈ 3 * sqrt(π) / 4 atol = 1e-15
        @test gaussian_moment(5) == 0.0
        # Mutation: a missing odd-zero branch would return Γ(n/2+1/2) ≠ 0.
        @test abs(gaussian_moment(1)) == 0.0
        @test abs(gaussian_moment(7)) == 0.0
        @test_throws ArgumentError gaussian_moment(-1)
    end

    @testset "Gauss–Hermite matches physicist rule and scipy nodes" begin
        # n=1: node 0, weight √π. Independent of SciPy.
        x1, w1 = hermite_rule(1)
        @test x1 ≈ [0.0] atol = 1e-15
        @test w1 ≈ [sqrt(π)] atol = 1e-15
        # n=2 analytic physicist Hermite.
        x2, w2 = hermite_rule(2)
        @test sort(x2) ≈ [-1 / sqrt(2), 1 / sqrt(2)] atol = 1e-14
        @test w2 ≈ [sqrt(π) / 2, sqrt(π) / 2] atol = 1e-14
        # FastGaussQuadrature.gausshermite is the same rule as
        # scipy.special.roots_hermite (weight e^{-t^2}).
        px, pw = gausshermite(8)
        qx, qw = hermite_rule(8)
        @test px ≈ qx atol = 0
        @test pw ≈ qw atol = 0
        scipy_x = [-2.930637420257244, -1.981656756695843,
                   -1.1571937124467802, -0.3811869902073221,
                   0.3811869902073221, 1.1571937124467802,
                   1.981656756695843, 2.930637420257244]
        scipy_w = [0.00019960407221136922, 0.017077983007413564,
                   0.20780232581489178, 0.6611470125582412,
                   0.6611470125582412, 0.20780232581489178,
                   0.017077983007413564, 0.00019960407221136922]
        @test sort(px) ≈ sort(scipy_x) atol = 1e-14
        @test sort(pw) ≈ sort(scipy_w) atol = 1e-14
        @test abs(sum(pw) - sqrt(π)) < 2e-15
        # Moments recovered by the quadrature itself.
        x8, w8 = hermite_rule(8)
        @test abs(sum(w8) - gaussian_moment(0)) < 2e-15
        @test abs(sum(w8 .* x8) - gaussian_moment(1)) < 2e-15
        @test abs(sum(w8 .* x8 .^ 2) - gaussian_moment(2)) < 2e-14

        # Public callers must not be able to corrupt the cached rule used by
        # subsequent canonical quadrature evaluations.
        mutable_nodes, mutable_weights = hermite_rule(2)
        mutable_nodes[1] = 123.0
        mutable_weights[1] = 456.0
        fresh_nodes, fresh_weights = hermite_rule(2)
        @test sort(fresh_nodes) ≈ [-1 / sqrt(2), 1 / sqrt(2)] atol = 1e-14
        @test fresh_weights ≈ [sqrt(π) / 2, sqrt(π) / 2] atol = 1e-14

        tasks = [Threads.@spawn hermite_rule(isodd(i) ? 12 : 13) for i in 1:16]
        concurrent_rules = fetch.(tasks)
        @test all(length(nodes) in (12, 13) for (nodes, _) in concurrent_rules)
        @test all(isapprox(sum(weights), sqrt(π); rtol=1e-13)
                  for (_, weights) in concurrent_rules)
    end

    @testset "W0 identity and Python Faddeeva spot check" begin
        z = 0.7 + 0.4im
        W = faddeeva_moments(z, 3)
        @test abs(W[1] - faddeeva(z)) < 1e-15
        @test abs(W[1] - erfcx(-im * z)) < 1e-15
        py = 0.4922894280872569 + 0.33153472612943524im
        @test abs(faddeeva(z) - py) < 2e-15
        hp = faddeeva_hp(z)
        hp_more = faddeeva_hp(z; precision=512)
        @test hp isa Complex{BigFloat}
        @test abs(hp - hp_more) < big"1e-100"
        @test abs(ComplexF64(hp) - py) < 2e-15

        # On the real-axis Stokes line, Re w(x)=exp(-x^2) is exponentially
        # small but must not be discarded by the high-precision branch.
        setprecision(BigFloat, 384) do
            x = BigFloat(9)
            wx = faddeeva_hp(x; precision=384)
            exact_real = exp(-x^2)
            @test abs(real(wx) / exact_real - 1) < big"1e-100"
            epsilon = big"1e-50"
            above = faddeeva_hp(complex(x, epsilon); precision=384)
            below = faddeeva_hp(complex(x, -epsilon); precision=384)
            @test abs(above - below) < big"1e-48"
            @test abs(faddeeva_hp(-x; precision=384) -
                      (2exact_real - wx)) < big"1e-100"
        end
        @test_throws ArgumentError faddeeva_hp(1.0; precision=32)
        @test_throws ArgumentError hp_moment(Inf, 0)

        @test abs(W[2] - (0.21198870920930571 - 0.13519950402224884im)) < 2e-15
        @test abs(W[3] - (0.20247189805541352 - 0.0098441691318518904im)) < 2e-15
        @test abs(W[4] - (0.14566799629153021 - 0.207996950944009im)) < 2e-15
    end

    @testset "recurrence identity and i versus -i" begin
        z = -0.8 + 0.35im
        values = moment_recurrence(z, 4)
        for m in 0:3
            rhs = z * values[m + 1] - im * gaussian_moment(m) / π
            @test abs(values[m + 2] - rhs) < 1e-15
            # Paper sign is -i. The opposite sign is a different function.
            wrong = z * values[m + 1] + im * gaussian_moment(m) / π
            if gaussian_moment(m) != 0.0
                @test abs(values[m + 2] - wrong) > 1e-4
            end
        end
        py0 = 0.45933216529679721 - 0.37769954909276415im
        @test abs(values[1] - py0) < 2e-15
    end

    @testset "W1 derivative identity" begin
        z = 0.4 + 0.3im
        w = faddeeva(z)
        wp = -2 * z * w + 2im / sqrt(π)
        W1 = faddeeva_moments(z, 1)[2]
        @test abs(W1 + 0.5 * wp) < 2e-15
        py = 0.18601386011689086 - 0.26617134578980739im
        @test abs(W1 - py) < 2e-15
    end

    @testset "polynomial hierarchy matches Gauss–Hermite" begin
        k = 250.0
        z = 0.9 + 0.45im
        coeff = [0.08, 1.0, 0.35, -0.15, 0.08]
        exact = hierarchy_value(k, z, coeff)
        quad = canonical_quadrature(k, z, polynomial_amplitude(coeff); n=800)
        @test relerr(exact, quad) < 2e-12
        py = 0.044927602623239422 + 0.023769556335399587im
        @test abs(exact - py) < 2e-14
    end

    @testset "linear null layer is exact" begin
        k = 1000.0
        z = -0.3 + 0.5im
        coeff = joint_null_coefficients(k, 1.2; tail=())
        exact = hierarchy_value(k, z, coeff)
        quad = canonical_quadrature(k, z, polynomial_amplitude(coeff); n=800)
        @test abs(exact - quad) < 5e-13
        py = 0.018969937178984025 - 0.012813278192590479im
        @test abs(exact - py) < 2e-14
        @test abs(coeff[1] - 1.2 / sqrt(k)) < 1e-15
        @test coeff[2] == 1.0 + 0im
    end

    @testset "k^{-m/2} field-order laws" begin
        z = 0.7 + 0.4im
        ks = 10 .^ range(2, 6; length=9)
        for (r, predicted) in ((1, -0.5), (2, -1.0), (3, -1.5))
            coeff = zeros(r + 1)
            coeff[r + 1] = 1.0
            vals = [abs(hierarchy_value(k, z, coeff)) for k in ks]
            slope = fitted_log_slope(ks, vals)
            @test slope ≈ predicted atol = 1e-12
        end
        # A dropped k^{-m/2} factor would give slope 0 for every r.
        coeff = [0.0, 1.0]
        v1 = abs(hierarchy_value(1e2, z, coeff))
        v2 = abs(hierarchy_value(1e6, z, coeff))
        @test v2 / v1 ≈ 1e-2 atol = 1e-10
    end

    @testset "null-uniform outer Lambda matching" begin
        z = 0.8 + 0.35im
        W = faddeeva_moments(z, 1)
        for Lambda in (20.0, 50.0, 100.0)
            ratio = abs(W[2] / (Lambda * W[1]))
            @test ratio < 0.08 / (Lambda / 20.0)
        end
    end

    @testset "remainder bound example" begin
        k = 200.0
        z = 0.6 + 0.45im
        C = 0.3
        remainder = C * k^(-1) * faddeeva_moments(z, 2)[3]
        bound = C / (π * imag(z)) * gamma(1.5) * k^(-1)
        @test abs(remainder) <= bound * (1 + 5e-15)
    end

    @testset "switch=10 recurrence versus asymptotic overlap" begin
        z_lo = 9.9 + 0.0im
        z_hi = 10.1 + 0.0im
        z_ov = 10.0 + 0.0im
        @test abs(z_lo) < DEFAULT_SWITCH
        @test abs(z_hi) > DEFAULT_SWITCH
        @test abs(z_ov) == DEFAULT_SWITCH
        Wlo = faddeeva_moments(z_lo, 2)
        @test Wlo ≈ moment_recurrence(z_lo, 2) atol = 0
        Whi = faddeeva_moments(z_hi, 2)
        for m in 0:2
            @test Whi[m + 1] == moment_asymptotic(z_hi, m)
        end
        Won = faddeeva_moments(z_ov, 2)
        @test Won ≈ moment_recurrence(z_ov, 2) atol = 0
        rec = moment_recurrence(z_ov, 3)
        asy = [moment_asymptotic(z_ov, m) for m in 0:3]
        for m in 0:3
            @test relerr(rec[m + 1], asy[m + 1]) < 5e-12
        end
    end

    @testset "large-|z| stable moments versus BigFloat recurrence" begin
        z = 1e7 + 0.4im
        val = faddeeva_moments(z, 4)
        for m in 0:4
            ref = hp_moment_asymptotic(z, m)
            @test ref isa Complex{BigFloat}
            @test relerr(val[m + 1], ref; floor=1e-300) < 2e-13
            @test relerr(hp_moment(z, m), ref; floor=1e-300) < 1e-12
        end
        # Recurrence itself is the unstable branch at this radius.
        rec = moment_recurrence(z, 4)
        @test relerr(rec[5], hp_moment_asymptotic(z, 4); floor=1e-300) > 1e6
        py0 = 2.256758334191055e-15 + 5.6418958354775823e-8im
        @test relerr(val[1], py0; floor=1e-300) < 2e-12
    end

    @testset "order-aware recurrence and Stokes crossover" begin
        cases = ((9.0 + 1e-10im, 7),
                 (6.0 + 1e-10im, 10),
                 (3.0 + 1e-10im, 20),
                 (-9.0 - 1e-10im, 7))
        for (z, m) in cases
            value = faddeeva_moments(z, m)[m + 1]
            reference = ComplexF64(hp_moment(z, m; precision=768))
            @test relerr(value, reference; floor=1e-300) < 2e-11
        end
        @test faddeeva_moments(0.0, 4) == moment_recurrence(0.0, 4)
        @test faddeeva_moments(1e-12 + 1e-12im, 4) ≈
              moment_recurrence(1e-12 + 1e-12im, 4) atol=0 rtol=0
    end

    @testset "lower-half-plane continuation term" begin
        # The algebraic inverse-power series alone is smaller than the full
        # continuation by more than sixty orders of magnitude at this point.
        z = 1.0 - 12.0im
        stable = faddeeva_moments(z, 2)
        reference = moment_recurrence(z, 2)
        for m in 0:2
            @test relerr(stable[m + 1], reference[m + 1]) < 3e-14
            @test relerr(hp_moment_asymptotic(z, m), reference[m + 1]) < 3e-14
        end
        @test abs(stable[1]) > 1e60
    end

    @testset "direct QuadGK and BigFloat integral oracles" begin
        z = 0.6 + 0.7im
        W = faddeeva_moments(z, 2)
        for m in 0:2
            ref = direct_moment(z, m)
            @test relerr(W[m + 1], ref) < 3e-12
            href = hp_moment(z, m)
            @test relerr(W[m + 1], href) < 3e-12
        end
        py0 = 0.44443364410152242 + 0.20678722339096886im
        @test abs(W[1] - py0) < 2e-15
        # High-precision real-line integral at one moderate point.
        href = hp_direct_moment(z, 0; precision=192, rtol=1e-18, atol=1e-18)
        @test href isa Complex{BigFloat}
        @test relerr(W[1], href) < 3e-12
    end

    @testset "input controls and Im z sign for quadrature" begin
        @test_throws ArgumentError faddeeva_moments(1.0, 2; switch=NaN)
        @test_throws ArgumentError faddeeva_moments(NaN + 0im, 2)
        @test_throws ArgumentError moment_asymptotic(12.0, 0; max_terms=0)
        amp = polynomial_amplitude([1.0])
        @test_throws ArgumentError canonical_quadrature(10.0, 1.0 - 0.1im, amp)
        @test_throws ArgumentError direct_moment(1.0 - 0.1im, 0)
        @test_throws ArgumentError hp_direct_moment(0.3 - 0.2im, 0)
        @test_throws ArgumentError hierarchy_value(NaN, 1.0 + im, [1.0])
        @test_throws ArgumentError hierarchy_value(Inf, 1.0 + im, [1.0])
        @test_throws ArgumentError hierarchy_value(10.0, 1.0 + im, [NaN])
        @test_throws ArgumentError joint_null_coefficients(10.0, Inf)
        @test_throws ArgumentError joint_null_coefficients(10.0, 1.0; tail=(0.2, NaN))
        @test_throws ArgumentError canonical_quadrature(NaN, 1.0 + im, amp; n=8)
        @test_throws ArgumentError canonical_quadrature(
            10.0, 1.0 + im, _ -> Inf; n=8)
        @test_throws ArgumentError shifted_null_basis(1.0 + im, Inf, 1)
        @test_throws ArgumentError shifted_null_hierarchy(
            Inf, 1.0 + im, 1.0, 1, [1.0])
        @test_throws ArgumentError scale_normalized_error(1.0, 1.0, NaN)
        @test_throws ArgumentError hierarchy_value(big"1e-1000", 1.0 + im, [1.0])
        # Pure coefficient construction preserves the promoted scalar type;
        # only production special-function evaluation rejects BigFloat.
        @test joint_null_coefficients(big"1e-1000", 1.0; tail=())[1] ≈ big"1e500" rtol=1e-70
        @test joint_null_coefficients(10.0, big"1e-1000"; tail=())[1] ≈ big"1e-1000"/sqrt(big"10") rtol=1e-70
        @test_throws DomainError gaussian_moment(400)
        @test_throws DomainError moment_recurrence(1.0 + im, 400)
        @test_throws ArgumentError hierarchy_value(10.0, 1.0 + im, [1.0]; order=-1)
        @test_throws ArgumentError shifted_null_hierarchy(
            10.0, 1.0 + im, 1.0, 1, [1.0]; order=-1)
        @test_throws ArgumentError hp_moment_asymptotic(NaN, 1)
        @test_throws ArgumentError direct_moment(0.6 + 0.7im, 0; rtol=-1.0)
        @test_throws ArgumentError hp_direct_moment(0.6 + 0.7im, 0; rtol=-1.0)
        @test_throws DomainError direct_moment(
            0.6 + 0.7im, 0; rtol=zero(Float64), atol=1e-300)
        @test_throws ArgumentError fitted_log_slope([1.0, 1.0], [1.0, 2.0])
        # Positive imaginary part is accepted.
        val = canonical_quadrature(10.0, 1.0 + 0.2im, amp; n=80)
        @test isfinite(real(val)) && isfinite(imag(val))
    end

    @testset "shifted-null binomial basis" begin
        k = 900.0
        z = 0.6 + 0.5im
        Lambda = -1.1
        for r in 1:3
            b = [1.0, 0.2, -0.05]
            base = ComplexF64[Lambda / sqrt(k), 1.0]
            poly = ComplexF64[1.0]
            for _ in 1:r
                poly = [sum(get(poly, i, 0) * get(base, j, 0)
                            for i in 1:length(poly), j in 1:length(base)
                            if i + j == n + 1) for n in 1:(length(poly) + length(base) - 1)]
            end
            coeff = [sum(get(poly, i, 0) * get(b, j, 0)
                         for i in 1:length(poly), j in 1:length(b)
                         if i + j == n + 1) for n in 1:(length(poly) + length(b) - 1)]
            direct = hierarchy_value(k, z, coeff)
            shifted = shifted_null_hierarchy(k, z, Lambda, r, b)
            @test abs(direct - shifted) < 2e-14
        end
        z2 = -0.4 + 0.7im
        for r in 0:3
            moments = faddeeva_moments(z2, r)
            @test abs(shifted_null_basis(z2, 0.0, r) - moments[r + 1]) < 2e-15
        end
    end

    @testset "domain errors" begin
        @test_throws ArgumentError hierarchy_value(0.0, 1.0 + im, [1.0])
        @test_throws ArgumentError hierarchy_value(-1.0, 1.0 + im, [1.0])
        @test_throws ArgumentError faddeeva_moments(1.0 + im, -1)
        @test_throws ArgumentError moment_asymptotic(0.0 + 0.0im, 0)
        @test_throws ArgumentError shifted_null_hierarchy(1.0, 1.0 + im, 0.0, -1, [1.0])
    end

    @testset "low-order identities W2 and W3" begin
        z = 0.55 + 0.4im
        w = faddeeva(z)
        W = faddeeva_moments(z, 3)
        @test abs(W[2] - (z * w - im / sqrt(π))) < 2e-15
        @test abs(W[3] - (z^2 * w - im * z / sqrt(π))) < 2e-15
        @test abs(W[4] - (z^3 * w - (im / sqrt(π)) * (z^2 + 0.5))) < 2e-14
    end

    @testset "released production adapters delegate to UTDKernels" begin
        z = 0.7 + 0.4im
        @test faddeeva_moments(z, 4) == UTDKernels.faddeeva_moments(z, 4)
        @test hierarchy_value(250.0, z, NULL_ADAPTER_COEFFICIENTS) ==
              UTDKernels.null_uniform_transition(
                  250.0, z, NULL_ADAPTER_COEFFICIENTS,
              )
        @test shifted_null_hierarchy(
            900.0, z, -1.1, 3, NULL_ADAPTER_BASE,
        ) == UTDKernels.shifted_null_transition(
            900.0, z, -1.1, 3, NULL_ADAPTER_BASE,
        )
        @test moment_recurrence(z, 4) == UTDKernels._null_moment_recurrence(z, 4)
        @test moment_asymptotic(12.0 + 0.2im, 3) ==
              UTDKernels._null_moment_asymptotic(12.0 + 0.2im, 3)

        null_adapter_moderate_probe()
        null_adapter_large_probe()
        null_adapter_hierarchy_probe()
        null_adapter_shifted_probe()
        @test @allocated(null_adapter_moderate_probe()) <= 512
        @test @allocated(null_adapter_large_probe()) <= 512
        @test @allocated(null_adapter_hierarchy_probe()) <= 640
        @test @allocated(null_adapter_shifted_probe()) <= 1024
    end

    @testset "exactly factored moving-null amplitude" begin
        k = 400.0
        for Lambda in (-3.0, -1.0, 0.0, 1.2, 4.0)
            coeff = factored_null_coefficients(k, Lambda)
            a = polynomial_amplitude(coeff)
            # The moving zero is exact at t = -Lambda/sqrt(k) for finite k
            # (to Horner rounding).
            @test abs(a(-Lambda / sqrt(k))) < 1e-16
            @test coeff[1] == Lambda / sqrt(k)
            @test coeff[2] == 1 + Lambda * 0.35 / sqrt(k)
            # Coefficient proxy differs from the exact root only in the
            # tail: the proxy does not vanish at the null coordinate.
            proxy = joint_null_coefficients(k, Lambda)
            pa = polynomial_amplitude(proxy)
            if Lambda != 0
                @test abs(pa(-Lambda / sqrt(k))) > 1e-4
            end
        end
        # Degenerate case b = (): amplitude is exactly t + Lambda/sqrt(k).
        coeff = factored_null_coefficients(k, 2.0; b=())
        @test coeff == ComplexF64[2.0 / sqrt(k), 1.0]
        @test_throws ArgumentError factored_null_coefficients(-1.0, 1.0)
        @test_throws ArgumentError factored_null_coefficients(10.0, Inf)
        @test_throws ArgumentError factored_null_coefficients(
            10.0, 1.0; b=(NaN,))
    end

    @testset "wedge kernel poles, zeros, residues" begin
        phi_i = deg2rad(10.0)
        N = 2.0
        poles = wedge_kernel_poles(phi_i, N)
        @test length(poles) == 12
        @test all(isapprox(p, -pi + s * phi_i + 4pi * m)
                  for p in poles for (s, m) in
                  ((s, m) for s in (-1, 1), m in -2:3) if
                  isapprox(p, -pi + s * phi_i + 4pi * m))
        # unit-residue structure: f(alpha) ~ +-1/(alpha - alpha_p)
        # (1e-9 offset sampling: pole-argument rounding gives ~5e-7 error)
        for p in (-pi + phi_i, -pi - phi_i, -pi + phi_i - 4pi)
            r_soft = wedge_kernel(p + 1e-9, phi_i, N, :soft) * 1e-9
            r_hard = wedge_kernel(p + 1e-9, phi_i, N, :hard) * 1e-9
            @test abs(abs(r_soft) - 1) < 5e-6
            @test abs(abs(r_hard) - 1) < 5e-6
        end
        # hard-kernel zeros at alpha = -pi + 2 pi m (two ~O(10) cots cancel)
        for m in -1:2
            z_alpha = -pi + m * 2pi
            @test wedge_kernel(z_alpha, phi_i, N, :hard) ≈ 0 atol = 1e-13
            # soft kernel is not zero there
            @test abs(wedge_kernel(z_alpha, phi_i, N, :soft)) > 1e-2
        end
        @test wedge_kernel_zeros(N) == [-pi + m * 2pi for m in 0:4]
        @test_throws ArgumentError wedge_kernel(1.0, phi_i, N, :bad)
        @test_throws ArgumentError wedge_kernel(1.0, phi_i, -1.0, :hard)
    end

    @testset "half-plane SDP vs eigenfunction reference" begin
        Krho = 40.0
        phi_i = deg2rad(10.0)
        N = 2.0
        worst = 0.0
        for deg in (140.5, 155.0, 169.9, 170.5, 175.0, 179.0, 180.0,
                    181.0, 185.0, 189.5, 190.5, 195.0, 205.0, 239.5)
            phi = deg2rad(deg)
            ud = wedge_diffracted(Krho, phi, phi_i, N, :hard)
            usdp = wedge_diffracted_sdp(Krho, phi, phi_i, N, :hard)
            worst = max(worst, abs(usdp - ud))
        end
        @test worst < 5e-13
        # exact diffracted-field null at phi = pi (hard kernel zero)
        @test abs(wedge_diffracted(Krho, pi, phi_i, N, :hard)) < 1e-13
        # GO boundary prescription: incident wave present/blocked
        go_lit = wedge_go(Krho, deg2rad(150.0), phi_i, N, :hard)
        go_shadow = wedge_go(Krho, deg2rad(200.0), phi_i, N, :hard)
        @test abs(go_lit) > 0.5
        @test abs(go_shadow) < 0.5 + 1e-12
    end

    @testset "canonical model vs SDP and quadrature" begin
        Krho = 40.0
        phi_i = deg2rad(10.0)
        N = 2.0
        # two unique mapped poles for the half-plane
        m = wedge_canonical_model(Krho, deg2rad(200.0), phi_i, N, :hard;
                                  saddle=-2pi)
        @test length(m.poles) == 2
        @test m.poles[1] != m.poles[2]
        @test abs(m.lambda) > 0.3      # null coordinate at phi=200 deg
        m_pi = wedge_canonical_model(Krho, pi, phi_i, N, :hard;
                                     saddle=-2pi)
        @test abs(m_pi.lambda) < 1e-12 # exact null at phi = pi
        # phase map: e^{-iK cos sigma} = e^{-iK} e^{-u^2}
        maps = wedge_phase_map(Krho; saddle=0.0)
        for sigma in (0.3, -0.7, 1.1)
            u = maps.u_of(sigma)
            @test abs(exp(-im * Krho * cos(sigma)) -
                      exp(-im * Krho) * exp(-u^2)) < 1e-14
            @test abs(maps.sigma_of(u) - sigma) < 1e-12
        end
        # canonical pair value vs independent quadrature per saddle
        worst = 0.0
        for deg in (150.0, 169.9, 171.0, 180.0, 189.9, 191.0, 230.0)
            phi = deg2rad(deg)
            for s0 in (0.0, -2pi)
                mm = wedge_canonical_model(Krho, phi, phi_i, N, :hard;
                                           saddle=s0)
                val = canonical_pole_pair_value(mm.poles,
                                                mm.coefficients, 14)
                ref = canonical_pair_quadrature(mm.poles, mm.amplitude_u)
                worst = max(worst, abs(val - ref))
            end
        end
        @test worst < 5e-11
        # assembled two-moment diffracted field vs eigenfunction - GO
        worst = 0.0
        for deg in (140.5, 155.0, 169.0, 170.5, 175.0, 179.0, 180.0,
                    181.0, 185.0, 189.0, 190.5, 195.0, 205.0, 239.5)
            phi = deg2rad(deg)
            ud = wedge_diffracted(Krho, phi, phi_i, N, :hard)
            v1 = wedge_diffracted_model(Krho, phi, phi_i, N, :hard;
                                        order=1)
            worst = max(worst, abs(v1 - ud))
        end
        @test worst < 5e-12
        # one-moment has genuine O(1) scale-normalized error near the null
        ud = wedge_diffracted(Krho, pi, phi_i, N, :hard)
        v0 = wedge_diffracted_model(Krho, pi, phi_i, N, :hard; order=0)
        m2 = wedge_canonical_model(Krho, pi, phi_i, N, :hard; saddle=-2pi)
        scale = wedge_term_scale(m2, 4)
        @test scale < 1e-6          # term scale collapses at the null
        @test abs(v0 - ud) < 5e-11  # absolute error stays small here
        # SDP model equality with canonical model
        for deg in (155.0, 200.0)
            phi = deg2rad(deg)
            usdp = wedge_diffracted_sdp(Krho, phi, phi_i, N, :hard)
            um = wedge_diffracted_model(Krho, phi, phi_i, N, :hard;
                                        order=14)
            @test abs(usdp - um) < 5e-12
        end
        @test_throws ArgumentError wedge_canonical_model(
            Krho, 1.0, phi_i, 3.0, :hard)
        @test_throws ArgumentError wedge_phase_map(Krho; saddle=pi / 2)
        @test_throws ArgumentError canonical_pole_pair_value(
            [0.1im, 0.1im], [1.0], 0)
        @test_throws DomainError canonical_pair_quadrature(
            [0.0, 1.0im], x -> 1.0)
    end

    @testset "null coordinate branch, pole limits, PV convention" begin
        Krho = 50.0
        phi_i = deg2rad(10.0)
        N = 2.0
        # Branch-consistent record: at phi=143.7735849 deg the stated
        # null coordinate is -2.198363373+2.198363373i (its negative was
        # previously recorded).  The regularized amplitude must vanish
        # at u = -Lambda on the minus-saddle model.
        m = wedge_canonical_model(Krho, deg2rad(143.7735849), phi_i, N,
                                  :hard; saddle=-2pi)
        @test real(m.lambda) ≈ -2.198363373 atol = 1e-9
        @test imag(m.lambda) ≈ 2.198363373 atol = 1e-9
        @test abs(m.amplitude_u(-m.lambda)) < 1e-12
        # Analytic removable limits match a symmetric numerical limit
        for phi_deg in (150.0, 170.0, 200.0, 235.0)
            mm = wedge_canonical_model(Krho, deg2rad(phi_deg), phi_i, N,
                                       :hard; saddle=-2pi)
            for (j, zj) in enumerate(mm.poles)
                h = 1e-7 * max(1.0, abs(zj))
                num = (mm.amplitude_u(zj + im * h) +
                       mm.amplitude_u(zj - im * h)) / 2
                @test abs(mm.pole_limits[j] - num) < 5e-6
            end
        end
        # Exact field identity: two-moment = classical limits = reference
        worst = 0.0
        for phi_deg in (150.0, 169.9, 200.0, 235.0)
            phi = deg2rad(phi_deg)
            ud = wedge_diffracted(Krho, phi, phi_i, N, :hard)
            vcl = wedge_diffracted_model(Krho, phi, phi_i, N, :hard;
                                         kind=:pole_extraction)
            worst = max(worst, abs(vcl - ud))
        end
        @test worst < 1e-11
        # Fixed-contour moments: PV convention and parity
        @test abs(moment_int_plane(0.0 + 0.0im, 0; pv_tol=1e-10)[1]) < 1e-15
        for z in (0.8 + 0.35im, 1.5 + 0.7im, -0.3 + 0.9im)
            for m in 0:3
                Cp = moment_int_plane(z, m)[m + 1]
                Cm = moment_int_plane(-z, m)[m + 1]
                @test Cm ≈ (-1)^(m + 1) * Cp atol = 1e-13
            end
        end
        # Half-residue GO + symmetric PV at the transition boundaries
        for phi in (pi - phi_i, pi, pi + phi_i)
            ue = wedge_eigenfunction(Krho, phi, phi_i, N, :hard)
            go = wedge_go(Krho, phi, phi_i, N, :hard; half_weight=true)
            ud = ue - go
            v1 = wedge_diffracted_model(Krho, phi, phi_i, N, :hard;
                                        order=1, pv_tol=1e-9)
            @test abs(v1 - ud) < 1e-12
            @test abs(go + v1 - ue) < 1e-12
        end
        # At the exact null the one-moment assembly is also zero
        v0 = wedge_diffracted_model(Krho, pi, phi_i, N, :hard;
                                    order=0, pv_tol=1e-9)
        @test abs(v0) < 1e-12
        # Affine two-saddle identity:
        # atilde_{-2pi}(u) - atilde_0(-u) = -2 c C d + 2 u with
        # C = cos(phi/2), d = cos(phi_i/2).
        phi = deg2rad(160.0)
        mm2 = wedge_canonical_model(Krho, phi, phi_i, N, :hard;
                                    saddle=-2pi)
        mm0 = wedge_canonical_model(Krho, phi, phi_i, N, :hard;
                                    saddle=0.0)
        c = mm2.maps.c
        rhs_const = -2 * c * cos(phi / 2) * cos(phi_i / 2)
        for u in (0.4 + 0.2im, -0.7 + 0.5im, 1.1 - 0.3im)
            @test abs(mm2.amplitude_u(u) - mm0.amplitude_u(-u) -
                      (rhs_const + 2u)) < 1e-12
        end
    end

    @testset "classical pole-extraction baseline" begin
        k = 400.0
        z = 0.8 + 0.35im
        coeff = factored_null_coefficients(k, 1.2)
        a = polynomial_amplitude(coeff)
        exact = hierarchy_value(k, z, coeff)
        cl = classical_pole_term(k, z, a)
        # residual = -i (a_3 + a_4 q) / (2 sqrt(pi) k^{3/2})
        q = z / sqrt(k)
        residual = -im * (coeff[4] + coeff[5] * q) /
                   (2 * sqrt(pi) * k^1.5)
        reference = setprecision(BigFloat, 256) do
            kh = BigFloat(k)
            zh = Complex{BigFloat}(z)
            ch = Complex{BigFloat}.(coeff)
            qh = zh / sqrt(kh)
            wh = [hp_moment(zh, m; precision=256) for m in 0:4]
            full = sum(ch[m + 1] * wh[m + 1] / sqrt(kh)^m for m in 0:4)
            ah(t) = sum(ch[m + 1] * t^m for m in 0:4)
            classical = ah(qh) * wh[1] -
                        im * (ah(qh) - ah(zero(qh))) /
                        (qh * sqrt(BigFloat(pi)) * sqrt(kh))
            remainder = -im * (ch[4] + ch[5] * qh) /
                        (2 * sqrt(BigFloat(pi)) * kh * sqrt(kh))
            tolerance = 64eps(BigFloat) * (abs(full) + abs(classical))
            @test abs(full - classical - remainder) <= tolerance
            (; full, classical, remainder, tolerance)
        end
        # The independent oracle uses the represented coefficient values.
        # Eight component ulps are this well-conditioned fixture's regression
        # budget for the native recurrence and pole/Horner arithmetic, consistent
        # with the moment seed's 8eps error allowance. This is not a general
        # correctly-rounded API contract.
        field_budgets = map(((exact, reference.full),
                             (cl, reference.classical))) do pair
            value, oracle = pair
            component_budgets = map((real, imag)) do part
                budget = 8eps(part(value))
                @test abs(BigFloat(part(value)) - part(oracle)) <= BigFloat(budget)
                budget
            end
            hypot(component_budgets...)
        end
        residual_budgets = map((real, imag)) do part
            budget = eps(part(residual))
            @test abs(BigFloat(part(residual)) - part(reference.remainder)) <=
                  BigFloat(budget)
            budget
        end
        # An unbudgeted 1e-17 bound on the difference of these ~0.04 fields
        # depended on compiler/library rounding. Propagate their independently
        # checked errors and allow rounding in the two final subtractions.
        subtraction_budget = eps(Float64) * (abs(exact) + abs(cl) + abs(residual))
        @test abs((exact - cl) - residual) <=
              sum(field_budgets) + hypot(residual_budgets...) + subtraction_budget
        # The same oracle contracts reject sign, factor and retained-order
        # mutations rather than merely accepting a larger binary64 tolerance.
        @test abs(reference.full - reference.classical + reference.remainder) >
              reference.tolerance
        @test abs(reference.full - reference.classical - 2reference.remainder) >
              reference.tolerance
        wrong_order = hierarchy_value(k, z, coeff; order=2)
        @test abs(Complex{BigFloat}(wrong_order) - reference.full) >
              hypot(8eps(real(wrong_order)), 8eps(imag(wrong_order)))
        # linearized pole term is algebraically identical to two-moment
        lin = linearized_pole_term(k, z, coeff[1], coeff[2])
        two = hierarchy_value(k, z, coeff; order=1)
        @test abs(lin - two) < 1e-16
        # separated-slope is nonuniform: divergence as z -> 0
        sep = slope_separated_term(k, z, coeff[1], coeff[2])
        @test abs(sep - exact) > 1e-2
        @test_throws ArgumentError slope_separated_term(k, 0.0 + 0im,
                                                        coeff[1], coeff[2])
        # "two moments are never worse" does not hold in general: for
        # a(t) = (t + 1/sqrt(k))(1 - 5t) at k=400, z=3+0.35i the
        # two-moment error exceeds the one-moment error (higher omitted
        # terms can cancel the slope term).  Documented counterexample.
        cn = factored_null_coefficients(400.0, 1.0; b=(-5.0,))
        zn = 3.0 + 0.35im
        en0 = scale_normalized_error(
            hierarchy_value(400.0, zn, cn; order=0),
            hierarchy_value(400.0, zn, cn),
            abs(cn[1] * faddeeva_moments(zn, 0)[1]) +
            abs(cn[2] / sqrt(400.0) * faddeeva_moments(zn, 1)[2]) +
            abs(cn[3] / 400.0 * faddeeva_moments(zn, 2)[3]))
        en1 = scale_normalized_error(
            hierarchy_value(400.0, zn, cn; order=1),
            hierarchy_value(400.0, zn, cn),
            abs(cn[1] * faddeeva_moments(zn, 0)[1]) +
            abs(cn[2] / sqrt(400.0) * faddeeva_moments(zn, 1)[2]) +
            abs(cn[3] / 400.0 * faddeeva_moments(zn, 2)[3]))
        @test en0 ≈ 0.0129000939 atol = 1e-6
        @test en1 ≈ 0.111322196 atol = 1e-6
        @test en1 > 8 * en0
        # canonical-level pole extraction on a two-pole pair
        zs = [0.4 + 0.3im, -0.7 + 0.5im]
        at = u -> 1.0 + 0.3u - 0.1u^2
        pair_cl = canonical_pole_extraction(zs, at)
        pair_q = canonical_pair_quadrature(zs, at)
        @test abs(pair_cl - pair_q) < 3e-3
        # pole-limit evaluation: a_fn with a removable 0*Inf pattern
        a_sing = u -> (u - zs[1]) * (u - zs[2]) *
                      (abs(u - zs[1]) < 1e-14 ? 1 / (u - zs[1] + 1e-30) :
                       exp(u))
        v = canonical_pole_extraction(zs, a_sing)
        @test isfinite(real(v)) && isfinite(imag(v))
    end

    @testset "nonpolynomial exactly factored benchmark" begin
        k = 400.0
        z = 0.8 + 0.35im
        for (Lambda, beta) in ((1.2, 0.7), (-0.5, -1.0), (0.0, 0.3),
                               (2.0, 1.5))
            a = nonpolynomial_amplitude(k, Lambda, beta)
            @test a(-Lambda / sqrt(k)) == 0 + 0im
            closed = nonpolynomial_closed_form(k, z, Lambda, beta)
            quad = canonical_quadrature(k, z, a; n=2000)
            @test relerr(closed, quad; floor=1e-14) < 5e-13
        end
        # the two closed-form expressions agree through W_1 = z w - i/sqrt(pi)
        s = 0.7 / (2 * sqrt(k))
        lam = 1.2
        zs = z - s
        W = faddeeva_moments(zs, 1)
        form1 = exp(s^2) * ((lam / sqrt(k) + 0.7 / (2k)) * W[1] +
                            W[2] / sqrt(k))
        form2 = exp(s^2) / sqrt(k) * ((z + lam) * W[1] - im / sqrt(pi))
        @test abs(form1 - form2) < 2e-16
        @test_throws ArgumentError nonpolynomial_amplitude(0.0, 1.0, 1.0)
        @test_throws ArgumentError nonpolynomial_closed_form(
            k, z, 1.0, Inf)
    end
end

end
