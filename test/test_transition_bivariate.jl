using FastGaussQuadrature: gausslegendre
using ForwardDiff

include("support/bivariate_transition_oracle.jl")

bivariate_relerr(got, reference; floor=1e-2) =
    abs(got - reference) / max(abs(reference), floor)

@noinline _bivariate_fixed_probe() =
    bivariate_fresnel_transition(0.3, -0.7, 0.4; order=64)

@noinline _bivariate_zero_correlation_probe() =
    bivariate_fresnel_transition(0.3, -0.7, 0.0; order=64)

@noinline _bivariate_adaptive_probe() =
    bivariate_fresnel_transition(0.3, -0.7, 0.4)

@noinline _bivariate_weights_probe() =
    bivariate_mechanism_weights(0.3, -0.7, 0.4; order=64)

@noinline _bivariate_hessian_probe() =
    bivariate_transition_hessian(2.3, -0.2, 0.7, 0.2, -0.4; k=5.0)

@noinline _bivariate_rule_construction_probe(order) =
    UTDKernels._cached_gauss_legendre_rule(order; maximum_order=2048)

@testset "Correlation-aware bivariate transition" begin
    @testset "shared Gauss-Legendre cache and ownership" begin
        nodes2, weights2 = UTDKernels._copy_gauss_legendre_rule(2; maximum_order=2048)
        @test sort(nodes2) ≈ [-inv(sqrt(3)), inv(sqrt(3))] atol=1e-14
        @test weights2 ≈ [1.0, 1.0] atol=1e-14
        nodes3, weights3 = UTDKernels._copy_gauss_legendre_rule(3; maximum_order=2048)
        @test sort(nodes3) ≈ [-sqrt(3 / 5), 0.0, sqrt(3 / 5)] atol=1e-14
        @test sort(weights3) ≈ [5 / 9, 5 / 9, 8 / 9] atol=1e-14

        reference_nodes, reference_weights = gausslegendre(32)
        nodes, weights = UTDKernels._copy_gauss_legendre_rule(32; maximum_order=2048)
        # Golub--Welsch and asymptotic FastGauss construction differ by a few
        # last bits at order 32; both integrate the analytic moments below.
        @test nodes ≈ reference_nodes rtol=0 atol=5e-15
        @test weights ≈ reference_weights rtol=2e-14 atol=2e-15
        nodes[1] = 99
        weights[1] = 99
        nodes_again, weights_again =
            UTDKernels._copy_gauss_legendre_rule(32; maximum_order=2048)
        @test nodes_again ≈ reference_nodes rtol=0 atol=5e-15
        @test weights_again ≈ reference_weights rtol=2e-14 atol=2e-15

        orders = collect(57:64)
        lock(UTDKernels._GL_CACHE_LOCK)
        try
            foreach(order -> delete!(UTDKernels._GL_CACHE, order), orders)
        finally
            unlock(UTDKernels._GL_CACHE_LOCK)
        end
        tasks = map(order -> Threads.@spawn(
            UTDKernels._cached_gauss_legendre_rule(order; maximum_order=2048)
        ), orders)
        rules = fetch.(tasks)
        @test all(length(rules[index][1]) == orders[index] for index in eachindex(orders))
        @test all(haskey(UTDKernels._GL_CACHE, order) for order in orders)
        @test length(UTDKernels._cached_gauss_legendre_rule(
            300; maximum_order=2048,
        )[1]) == 300

        construction_order = 511
        _bivariate_rule_construction_probe(19)
        lock(UTDKernels._GL_CACHE_LOCK)
        try
            delete!(UTDKernels._GL_CACHE, construction_order)
        finally
            unlock(UTDKernels._GL_CACHE_LOCK)
        end
        construction_bytes = @allocated _bivariate_rule_construction_probe(
            construction_order,
        )
        # Two Float64 vectors plus small cache/lock overhead; a dense
        # Golub--Welsch eigenvector matrix would exceed this linear bound.
        @test construction_bytes <= 64construction_order
        @test_throws ArgumentError UTDKernels._copy_gauss_legendre_rule(0; maximum_order=2048)
        @test_throws ArgumentError UTDKernels._copy_gauss_legendre_rule(2049; maximum_order=2048)
        @test_throws ArgumentError UTDKernels.gauss_legendre_nodes(257)
    end

    @testset "product, origin, and frozen values" begin
        for (xi, eta) in ((-2.0, 0.4), (0.0, 0.0), (1.5, -0.7))
            expected = bivariate_oracle_switch(xi) * bivariate_oracle_switch(eta)
            @test bivariate_fresnel_transition(xi, eta, 0.0) == expected
        end
        @test bivariate_fresnel_transition(-2.0, 0.4, 0.0) ≈
              -0.08688686400612034 - 0.07598816589899128im atol=1e-15 rtol=0
        @test UTDKernels._bivariate_fresnel_switch(0.4) ≈
              0.6157733576815074 + 0.10975808599271164im atol=1e-14 rtol=0

        for rho in (-0.95, -0.5, 0.0, 0.3, 0.8, 0.95)
            exact = 0.25 + asin(rho) / (2pi)
            @test bivariate_fresnel_transition(0.0, 0.0, rho; order=160) ≈
                  exact atol=8e-15 rtol=0
        end
        @test real(bivariate_fresnel_transition(0.0, 0.0, 0.8)) > 0.25
        @test real(bivariate_fresnel_transition(0.0, 0.0, -0.8)) < 0.25
    end

    @testset "exchange, complements, and mechanism weights" begin
        for (xi, eta, rho) in (
            (-1.2, 0.7, 0.6), (2.0, -0.5, -0.4), (0.3, 1.1, 0.8),
        )
            value = bivariate_fresnel_transition(xi, eta, rho)
            @test value ≈ bivariate_fresnel_transition(eta, xi, rho) atol=2e-14 rtol=0
            complement = 1 - UTDKernels._bivariate_fresnel_switch(xi) -
                         UTDKernels._bivariate_fresnel_switch(eta) + value
            @test bivariate_fresnel_transition(-xi, -eta, rho) ≈
                  complement atol=3e-13 rtol=0

            weights = bivariate_mechanism_weights(xi, eta, rho; order=120)
            @test sum(weights) ≈ 1 atol=3e-15 rtol=0
            @test weights[2] + weights[4] ≈
                  UTDKernels._bivariate_fresnel_switch(xi) atol=3e-15 rtol=0
            @test weights[3] + weights[4] ≈
                  UTDKernels._bivariate_fresnel_switch(eta) atol=3e-15 rtol=0
        end
    end

    @testset "Plackett and mixed derivative identities" begin
        @test UTDKernels._bivariate_plackett_kernel(0.0, 0.0, 0.0) ≈
              inv(2pi) atol=1e-16 rtol=0
        h_rho = 1e-5
        for (xi, eta, rho) in (
            (-1.0, 0.5, 0.4), (1.4, -0.3, -0.5), (0.2, 0.7, 0.7),
        )
            finite_difference = (
                bivariate_fresnel_transition(xi, eta, rho + h_rho) -
                bivariate_fresnel_transition(xi, eta, rho - h_rho)
            ) / (2h_rho)
            kernel = UTDKernels._bivariate_plackett_kernel(xi, eta, rho)
            @test finite_difference ≈ kernel atol=3e-8 rtol=0

            h = 1e-4
            mixed = (
                bivariate_fresnel_transition(xi + h, eta + h, rho) -
                bivariate_fresnel_transition(xi + h, eta - h, rho) -
                bivariate_fresnel_transition(xi - h, eta + h, rho) +
                bivariate_fresnel_transition(xi - h, eta - h, rho)
            ) / (4h^2)
            @test mixed ≈ im * kernel atol=3e-7 rtol=0
        end
    end

    @testset "adaptive, fixed, and independent rotated quadrature" begin
        for (xi, eta, rho) in (
            (-2.0, 0.4, 0.8), (3.0, -2.2, 0.88), (0.1, 0.2, -0.85),
        )
            adaptive = bivariate_fresnel_transition(xi, eta, rho)
            fixed = bivariate_fresnel_transition(xi, eta, rho; order=160)
            @test bivariate_relerr(fixed, adaptive) < 3e-12
        end
        # These independently selected moderate cases keep the finite rotated
        # contour's Gaussian cutoff error below the comparison tolerance.
        for (xi, eta, rho) in (
            (-1.0, 0.2, 0.4), (0.5, 1.1, -0.3), (1.4, -0.6, 0.65),
        )
            fixed = bivariate_fresnel_transition(xi, eta, rho; order=160)
            rotated = bivariate_rotated_oracle(xi, eta, rho)
            @test bivariate_relerr(fixed, rotated) < 1e-11
        end

        for (xi, eta, rho) in (
            (-2.0, 2.0, 0.98), (0.1, 0.2, 0.98), (-0.4, -1.1, 0.98),
            (-2.0, 2.0, -0.98), (0.1, 0.2, -0.98), (-0.4, -1.1, -0.98),
        )
            adaptive = bivariate_fresnel_transition(
                xi, eta, rho; rtol=5e-12, atol=5e-14,
            )
            fixed = bivariate_fresnel_transition(xi, eta, rho; order=1200)
            @test adaptive ≈ fixed atol=2e-12 rtol=0
        end
    end

    @testset "small-correlation orders and rank-one behavior" begin
        xi, eta = 0.3, -0.7
        product = UTDKernels._bivariate_fresnel_switch(xi) *
                  UTDKernels._bivariate_fresnel_switch(eta)
        derivative_product = UTDKernels._bivariate_fresnel_switch_prime(xi) *
                             UTDKernels._bivariate_fresnel_switch_prime(eta)
        first_order(rho) = product - im * rho * derivative_product
        second_order(rho) = first_order(rho) +
                            rho^2 * xi * eta * derivative_product / 2
        radii = (1e-2, 5e-3, 2.5e-3)
        product_errors = map(rho -> abs(
            bivariate_fresnel_transition(xi, eta, rho; order=160) - product,
        ), radii)
        first_errors = map(rho -> abs(
            bivariate_fresnel_transition(xi, eta, rho; order=160) - first_order(rho),
        ), radii)
        second_errors = map(rho -> abs(
            bivariate_fresnel_transition(xi, eta, rho; order=160) - second_order(rho),
        ), radii)
        @test log(product_errors[1] / product_errors[3]) / log(4) > 0.95
        @test log(first_errors[1] / first_errors[3]) / log(4) > 1.9
        @test log(second_errors[1] / second_errors[3]) / log(4) > 2.85

        for (x, y) in ((-1.0, 0.7), (0.5, 1.2), (-0.4, -1.1))
            positive = bivariate_fresnel_transition(x, y, 0.995; order=420)
            @test positive ≈ UTDKernels._bivariate_fresnel_switch(min(x, y)) atol=0.02 rtol=0
        end
    end

    @testset "normalized Hessian map and extreme scales" begin
        a, c, rho = 2.3, 0.7, 0.62
        b = -rho * sqrt(a * c)
        mapped = bivariate_transition_hessian(a, b, c, 0.2, -0.4; k=5.0)
        @test mapped.rho ≈ rho atol=1e-14 rtol=0
        @test mapped.xi ≈ sqrt(5a * (1 - rho^2)) * 0.2 atol=1e-14 rtol=0
        @test mapped.eta ≈ sqrt(5c * (1 - rho^2)) * -0.4 atol=1e-14 rtol=0

        a_extreme, c_extreme = 1e280, 1e-280
        b_extreme, k_extreme = -0.999999999999, 1e-120
        u_extreme, v_extreme = 1e-20, 1e120
        reference = setprecision(BigFloat, 256) do
            A, B, C = BigFloat(a_extreme), BigFloat(b_extreme), BigFloat(c_extreme)
            K = BigFloat(k_extreme)
            rho_big = -B / sqrt(A * C)
            scale_big = 1 - rho_big^2
            (
                Float64(sqrt(K * A * scale_big) * BigFloat(u_extreme)),
                Float64(sqrt(K * C * scale_big) * BigFloat(v_extreme)),
                Float64(rho_big),
            )
        end
        extreme = bivariate_transition_hessian(
            a_extreme, b_extreme, c_extreme, u_extreme, v_extreme; k=k_extreme,
        )
        @test extreme.xi ≈ reference[1] rtol=5e-14
        @test extreme.eta ≈ reference[2] rtol=5e-14
        @test extreme.rho == reference[3]

        mapped32 = bivariate_transition_hessian(
            2.3f0, -0.2f0, 0.7f0, 0.2f0, -0.4f0; k=5f0,
        )
        @test mapped32.xi isa Float32
        @test mapped32.eta isa Float32
        @test mapped32.rho isa Float32

        mapped_big = bivariate_transition_hessian(
            big"2.3", big"-0.2", big"0.7", big"0.2", big"-0.4"; k=big"5",
        )
        @test mapped_big.xi isa BigFloat
        @test mapped_big.eta isa BigFloat
        @test mapped_big.rho isa BigFloat

        derivative = ForwardDiff.derivative(0.2) do mixed
            bivariate_transition_hessian(2.3, mixed, 0.7, 0.2, -0.4; k=5.0).rho
        end
        h = 1e-6
        finite_difference = (
            bivariate_transition_hessian(2.3, 0.2 + h, 0.7, 0.2, -0.4; k=5.0).rho -
            bivariate_transition_hessian(2.3, 0.2 - h, 0.7, 0.2, -0.4; k=5.0).rho
        ) / (2h)
        @test derivative ≈ finite_difference rtol=2e-9
    end

    @testset "array agreement, domains, and allocation" begin
        @test bivariate_fresnel_transition(
            0.3f0, -0.7f0, 0.4f0; order=64,
        ) isa ComplexF32

        for function_ in (
            xi_value -> real(bivariate_fresnel_transition(
                xi_value, -0.7, 0.4; order=64,
            )),
            rho_value -> imag(bivariate_fresnel_transition(
                0.3, -0.7, rho_value; order=64,
            )),
        )
            point = 0.3
            step = 1e-6
            automatic = ForwardDiff.derivative(function_, point)
            centered = (function_(point + step) - function_(point - step)) / (2step)
            @test isfinite(automatic)
            @test automatic ≈ centered rtol=2e-8 atol=2e-10
        end

        xi = [-1.0 0.0; 0.5 1.0]
        eta = [0.2 0.7; -0.4 0.3]
        values = bivariate_fresnel_transition(xi, eta, 0.35; order=64)
        @test size(values) == size(xi)
        for index in eachindex(xi)
            @test values[index] ≈ bivariate_fresnel_transition(
                xi[index], eta[index], 0.35; order=64,
            ) atol=1e-14 rtol=0
        end
        @test_throws DimensionMismatch bivariate_fresnel_transition(
            zeros(2), zeros(3), 0.2,
        )
        @test_throws DomainError bivariate_fresnel_transition(0.0, 0.0, 1.0)
        @test_throws DomainError bivariate_fresnel_transition(0.0, 0.0, -1.0)
        @test_throws DomainError bivariate_fresnel_transition(0.0, 0.0, NaN)
        @test_throws DomainError bivariate_fresnel_transition(Inf, 0.0, 0.2)
        @test_throws ArgumentError bivariate_fresnel_transition(
            0.0, 0.0, 0.2; rtol=0.0, atol=0.0,
        )
        @test_throws DomainError bivariate_fresnel_transition(
            0.1, 0.2, 0.98; rtol=0.0, atol=1e-300,
        )
        @test_throws ArgumentError bivariate_fresnel_transition(
            0.0, 0.0, 0.0; order=0,
        )
        @test_throws DomainError bivariate_transition_hessian(1.0, 2.0, 1.0, 0.0, 0.0)
        @test_throws ArgumentError bivariate_fresnel_transition(
            big"0.3", big"-0.7", big"0.4"; order=32,
        )

        _bivariate_fixed_probe()
        _bivariate_zero_correlation_probe()
        _bivariate_adaptive_probe()
        _bivariate_weights_probe()
        _bivariate_hessian_probe()
        @test @allocated(_bivariate_fixed_probe()) == 0
        @test @allocated(_bivariate_zero_correlation_probe()) == 0
        @test @allocated(_bivariate_adaptive_probe()) == 0
        @test @allocated(_bivariate_weights_probe()) == 0
        @test @allocated(_bivariate_hessian_probe()) == 0
    end
end
