using ForwardDiff
using SpecialFunctions: digamma, erfc, erfcx, gamma
using UTDKernels

include("support/continuous_order_oracle.jl")

const CONTINUOUS_ALLOC_COEFFICIENTS =
    ComplexF64[1.0, -0.35 + 0.2im, 0.12]

@noinline _continuous_general_allocation_probe() =
    continuous_order_transition(1.2, 0.4 + 0.3im)
@noinline _continuous_endpoint_allocation_probe() =
    continuous_order_transition(1.7, -100.0)
@noinline _continuous_unit_allocation_probe() =
    continuous_order_transition(1.0, 0.4 + 0.3im)
@noinline _continuous_scaled_allocation_probe() =
    scaled_continuous_order_transition(1.0, 30.0)
@noinline _continuous_moment_allocation_probe() =
    continuous_order_moment(0.8, 2, 50.0, 1.3, 0.15, 0.12)
@noinline _continuous_hierarchy_allocation_probe() =
    continuous_order_moment(
        0.8, CONTINUOUS_ALLOC_COEFFICIENTS, 50.0, 1.3, 0.15,
    )

@testset "Continuous-order saddle--endpoint transition" begin
    @testset "domains and resource bounds" begin
        @test_throws DomainError continuous_order_transition(0.0, 0.0)
        @test_throws DomainError continuous_order_transition(-0.5, 0.0)
        @test_throws DomainError continuous_order_transition(Inf, 0.0)
        @test_throws DomainError continuous_order_transition(1.0, Inf)
        @test_throws DomainError continuous_order_transition(1.0, NaN + 0im)
        @test_throws DomainError continuous_order_transition(340.0, 0.1)
        @test_throws DomainError continuous_order_transition(1.0, 30.0)
        @test_throws ArgumentError continuous_order_transition(
            big"1.0", 0.0,
        )
        @test_throws ArgumentError continuous_order_transition(
            1.0, complex(big"0.1", big"0.2"),
        )
        @test_throws DomainError continuous_order_transition(
            1.0, 0.0; rtol=0.0,
        )
        @test_throws DomainError continuous_order_transition(
            1.0, 0.0; atol=-1.0,
        )
        @test_throws ArgumentError continuous_order_transition(
            1.0, 0.0; maxevals=62,
        )
        @test_throws ArgumentError continuous_order_transition(
            1.0, 0.0; maxevals=10_000_001,
        )
        @test_throws DomainError scaled_continuous_order_transition(1.0, Inf)
        @test_throws DomainError continuous_order_moment(
            1.0, -1, 10.0, 1.0, 0.0, 1.0,
        )
        @test_throws DomainError continuous_order_moment(
            1.0, 65, 10.0, 1.0, 0.0, 1.0,
        )
        @test_throws DomainError continuous_order_moment(
            1.0, 0, 0.0, 1.0, 0.0, 1.0,
        )
        @test_throws DomainError continuous_order_moment(
            1.0, 0, 1.0, 0.0, 0.0, 1.0,
        )
        @test_throws DomainError continuous_order_moment(
            1.0, 0, 1.0, 1.0, Inf, 1.0,
        )
        @test_throws DomainError continuous_order_moment(
            1.0, 0, 1.0, 1.0, 0.0, NaN,
        )
        @test_throws DomainError continuous_order_moment(
            1.0, 0, 1.0, 1.0, 0.0, 0.0; rtol=0.0,
        )
        @test_throws ArgumentError continuous_order_moment(
            1.0, ComplexF64[], 1.0, 1.0, 0.0; maxevals=10,
        )
        @test_throws DomainError continuous_order_moment(
            1.0, ComplexF64[], 1.0, 1.0, 0.0; order=-1,
        )
        @test_throws ArgumentError continuous_order_moment(
            1.0, Number[1.0], 1.0, 1.0, 0.0,
        )
        @test_throws DomainError UTDKernels._continuous_order_utd_transition(
            4097.0, 1e6,
        )
        @test_throws ArgumentError ForwardDiff.derivative(
            order -> real(continuous_order_transition(order, 0.2)), 1.2,
        )
        @test_throws DomainError continuous_order_transition(-1.0, Float64[])
        @test_throws DomainError continuous_order_transition(
            1.0, Float64[]; rtol=0,
        )
        @test_throws ArgumentError scaled_continuous_order_transition(
            1.0, Float64[]; maxevals=1,
        )
        @test_throws ArgumentError continuous_order_transition(
            1.0, Number[],
        )
        nonfinite_rtol = ForwardDiff.Dual(1e-6, Inf)
        nonfinite_atol = ForwardDiff.Dual(0.0, Inf)
        for coordinate in (
            0.2,
            ForwardDiff.Dual(0.2, 1.0),
            Float64[],
        )
            @test_throws DomainError continuous_order_transition(
                1.2, coordinate; rtol=nonfinite_rtol,
            )
            @test_throws DomainError continuous_order_transition(
                1.2, coordinate; atol=nonfinite_atol,
            )
        end
        @test_throws ArgumentError continuous_order_transition(
            1.2, 0.2; rtol=ForwardDiff.Dual(1e-6, 0.0),
        )
        for coefficients in (ComplexF64[], ComplexF64[1])
            @test_throws DomainError continuous_order_moment(
                1.2, coefficients, 1.0, 1.0, 0.2;
                rtol=nonfinite_rtol,
            )
            @test_throws DomainError continuous_order_moment(
                1.2, coefficients, 1.0, 1.0, 0.2;
                atol=nonfinite_atol,
            )
        end
        nested_nonfinite = ForwardDiff.Dual(
            ForwardDiff.Dual(1.0, Inf),
            ForwardDiff.Dual(0.0, 0.0),
        )
        @test UTDKernels._continuous_scalar_quad_norm(nested_nonfinite) == Inf
        @test !UTDKernels._continuous_scalar_allfinite(nested_nonfinite)
        @test_throws DomainError continuous_order_transition(
            1.0, nested_nonfinite,
        )
    end

    @testset "coalescence, arrays, and types" begin
        for mu in range(0.2, 3.5; length=20)
            @test continuous_order_transition(mu, 0.0) == 1.0 + 0.0im
        end
        coordinates = [-1.0, 0.0, 0.8]
        @test continuous_order_transition(1.2, coordinates) ==
              continuous_order_transition.(1.2, coordinates)
        @test scaled_continuous_order_transition(1.2, coordinates) ==
              scaled_continuous_order_transition.(1.2, coordinates)
        @test continuous_order_transition(Float32(1.2), ComplexF32(0.4, 0.3)) isa
              ComplexF32
        @test scaled_continuous_order_transition(Float32(1.2), Float32(0.4)) isa
              Float32
        @test continuous_order_transition(Float16(1), Float16(0)) isa ComplexF32
        @test continuous_order_moment(
            Float32(0.8), 1, Float32(50), Float32(1.3), Float32(0.15),
            ComplexF32(0.2, -0.1),
        ) isa ComplexF32
        @test continuous_order_moment(
            1.0, ComplexF64[], 10.0, 1.0, 0.0,
        ) == 0.0im
        coefficient_derivative = ForwardDiff.derivative(0.0) do coefficient
            imag(continuous_order_moment(
                2.0, 0, 1e-307, 1.0, 0.0, coefficient,
            ))
        end
        @test coefficient_derivative ≈ -inv(1e-307) rtol=3e-15
    end

    @testset "unit-order shared Faddeeva and UTD identities" begin
        for zeta in range(-4.0, 3.0; length=15)
            value = continuous_order_transition(1.0, zeta)
            expected = exp(zeta^2) * erfc(-zeta)
            @test _continuous_relative_error(value, expected) < 3e-14
            @test value == UTDKernels._faddeeva_w(-im * complex(zeta))
        end
        for zeta in (-0.7 + 0.2im, 0.4 + 0.3im)
            value = continuous_order_transition(1.0, zeta)
            @test value == UTDKernels._faddeeva_w(-im * zeta)
            @test _continuous_relative_error(
                value, exp(zeta^2) * erfc(-zeta),
            ) < 3e-14
        end
        for X in (0.0, 0.1, 0.7, 2.0, 8.0, 1e6)
            @test UTDKernels._continuous_order_utd_transition(1.0, X) == F_utd(X)
        end
    end

    @testset "independent quadrature and hypergeometric oracles" begin
        for zeta in (-9.0, 9.0, 8.5 + 0.75im, -8.5 + 0.75im)
            value = continuous_order_transition(2.0, zeta)
            reference = ComplexF64(
                _continuous_oracle_hypergeometric_big(
                    2.0, zeta; precision=256,
                ),
            )
            @test _continuous_relative_error(value, reference) < 8e-13
        end
        exact_order_two_dual = @inferred continuous_order_transition(
            2.0, ForwardDiff.Dual(9.0, 1.0),
        )
        exact_order_two_derivative = setprecision(BigFloat, 512) do
            zeta = BigFloat(9)
            Float64(
                sqrt(BigFloat(pi)) * erfcx(-zeta) * (1 + 2zeta^2) +
                2zeta,
            )
        end
        @test ForwardDiff.partials(real(exact_order_two_dual), 1) ==
              exact_order_two_derivative
        exact_order_two_second = ForwardDiff.derivative(9.0) do outer
            ForwardDiff.derivative(outer) do inner
                real(continuous_order_transition(2.0, inner))
            end
        end
        exact_order_two_second_reference = setprecision(BigFloat, 512) do
            zeta = BigFloat(9)
            Float64(
                sqrt(BigFloat(pi)) * erfcx(-zeta) * (6zeta + 4zeta^3) +
                4 + 4zeta^2,
            )
        end
        @test exact_order_two_second == exact_order_two_second_reference

        fixtures = (
            (0.35, -3.0 + 0im, 0.5083383409571955 + 0im),
            (0.5, -0.8 + 0im, 0.6644128887116558 + 0im),
            (1.7, 0.9 + 0im, 6.353760589544079 + 0im),
            (2.8, 2.5 + 0im, 11392.877706271765 + 0im),
            (0.7, -1.3 + 0.2im,
             0.4602405721257195 + 0.037048243605834634im),
            (1.2, 0.4 + 0.3im,
             1.5052105678028687 + 0.7981028798714632im),
            (2.3, -0.5 + 0.6im,
             0.22856408827540645 + 0.29649492948954476im),
        )
        for (mu, zeta, expected) in fixtures
            value = continuous_order_transition(mu, zeta)
            @test _continuous_relative_error(value, expected) < 3e-12
            oracle = ComplexF64(_continuous_oracle_big(mu, zeta))
            @test _continuous_relative_error(value, oracle) < 3e-12
        end
        for (mu, zeta) in (
            (0.5, -0.8), (1.0, 0.0), (1.7, 0.9),
            (0.7, -1.3 + 0.2im), (1.2, 0.4 + 0.3im),
            (2.3, -0.5 + 0.6im),
        )
            @test _continuous_relative_error(
                continuous_order_transition(mu, zeta),
                _continuous_oracle_hypergeometric(mu, zeta),
            ) < 3e-12
        end
        for (mu, zeta, expected) in (
            (2.0, -256.0, 7.629219914928242e-6),
            (4.0, -64.0, 4.464898374120168e-8),
            (1.7, -100.0, 2.0013670736373014e-4),
        )
            @test _continuous_relative_error(
                continuous_order_transition(mu, zeta), expected,
            ) < 4e-14
        end
        @test _continuous_relative_error(
            continuous_order_transition(1.0, -1e6), erfcx(1e6),
        ) < 3e-15

        # Negative endpoint scaling remains finite far beyond the range where
        # a symmetric coordinate-square guard is meaningful.
        for zeta in (-1e154, -1e200)
            mu = 0.1
            endpoint_reference = gamma((mu + 1) / 2) /
                                 (sqrt(pi) * (-zeta)^mu)
            @test _continuous_relative_error(
                continuous_order_transition(mu, zeta), endpoint_reference,
            ) < 2e-13
        end

        for (mu, zeta, tolerance) in (
            (1e-5, 3.0, 3e-12),
            (1e-6, 3.0, 3e-12),
            (1e-5, 1.0 + 2im, 3e-12),
        )
            reference = _continuous_oracle_hypergeometric(mu, zeta)
            @test _continuous_relative_error(
                continuous_order_transition(mu, zeta), reference,
            ) < tolerance
        end
        small_value32 = continuous_order_transition(1.0f-4, 3.0f0)
        small_reference32 = ComplexF32(
            _continuous_oracle_hypergeometric(1.0f-4, 3.0f0),
        )
        @test _continuous_relative_error(
            small_value32, small_reference32,
        ) < 3f-6
        for zeta in (-3.0f0, 0.2f0)
            value = continuous_order_transition(
                1.0f-20, zeta; rtol=1.0f-6,
            )
            reference = ComplexF32(
                _continuous_oracle_hypergeometric_big(1.0f-20, zeta),
            )
            @test _continuous_relative_error(value, reference) <= 1.0f-6
        end
        for zeta in (-3.0, 1.0)
            value = continuous_order_transition(
                1e-12, zeta; rtol=1.8e-15,
            )
            reference = ComplexF64(
                _continuous_oracle_hypergeometric_big(1e-12, zeta),
            )
            @test _continuous_relative_error(value, reference) <= 1.8e-15
        end
    end

    @testset "differential, order, and order-sensitivity identities" begin
        step = 2e-5
        for mu in (0.5, 1.0, 1.6, 2.4), zeta in (-2.0, -0.3, 0.8, 2.0)
            finite_difference = (
                continuous_order_transition(mu, zeta + step) -
                continuous_order_transition(mu, zeta - step)
            ) / (2step)
            @test _continuous_relative_error(
                UTDKernels._continuous_order_coordinate_derivative(mu, zeta),
                finite_difference; floor=1e-12,
            ) < 3e-8
        end
        for mu in (0.4, 0.9, 1.7, 2.6), zeta in (-2.0, 0.0, 1.4)
            @test _continuous_relative_error(
                UTDKernels._continuous_order_recurrence(mu, zeta),
                continuous_order_transition(mu + 2, zeta),
            ) < 5e-12
        end
        second_step = 1e-4
        for mu in (0.4, 1.0, 2.1), zeta in (-1.4, 0.2, 1.5)
            center = continuous_order_transition(mu, zeta)
            finite_difference = (
                continuous_order_transition(mu, zeta + second_step) - 2center +
                continuous_order_transition(mu, zeta - second_step)
            ) / second_step^2
            @test _continuous_relative_error(
                UTDKernels._continuous_order_second_coordinate_derivative(mu, zeta),
                finite_difference; floor=1e-10,
            ) < 4e-6
        end
        order_step = 2e-5
        for mu in (0.4, 0.8, 1.3, 2.2), zeta in (-1.2, 0.0, 0.9)
            finite_difference = (
                continuous_order_transition(mu + order_step, zeta) -
                continuous_order_transition(mu - order_step, zeta)
            ) / (2order_step)
            derivative = UTDKernels._continuous_order_parameter_derivative(mu, zeta)
            if iszero(zeta)
                @test derivative == 0.0im
                @test abs(finite_difference) < 1e-10
            else
                @test _continuous_relative_error(
                    derivative, finite_difference; floor=2e-9,
                ) < 3e-7
            end
        end
        for T in (Float32, Float64), mu in T.((0.3, 1.2, 5.0))
            mixed_first = ForwardDiff.derivative(zero(T)) do coordinate
                real(UTDKernels._continuous_order_parameter_derivative(
                    mu, coordinate,
                ))
            end
            reference = setprecision(BigFloat, 256) do
                order = BigFloat(mu)
                ratio = 2gamma((order + 1) / 2) / gamma(order / 2)
                T(ratio * (
                    digamma((order + 1) / 2) - digamma(order / 2)
                ) / 2)
            end
            @test mixed_first ≈ reference rtol=T(8) * eps(T)
            mixed_second = ForwardDiff.derivative(zero(T)) do outer
                ForwardDiff.derivative(outer) do coordinate
                    real(UTDKernels._continuous_order_parameter_derivative(
                        mu, coordinate,
                    ))
                end
            end
            @test mixed_second == T(2)
        end
        @test_throws ArgumentError ForwardDiff.derivative(0.0) do outer
            ForwardDiff.derivative(outer) do middle
                ForwardDiff.derivative(middle) do coordinate
                    real(UTDKernels._continuous_order_parameter_derivative(
                        0.3, coordinate,
                    ))
                end
            end
        end
        @test_throws ArgumentError ForwardDiff.derivative(0.2) do outer
            ForwardDiff.derivative(outer) do coordinate
                real(UTDKernels._continuous_order_parameter_derivative(
                    1.2, coordinate,
                ))
            end
        end
    end

    @testset "scaled and UTD-normalized branches" begin
        for mu in (0.5, 1.0, 2.0), zeta in (-3.0, 0.0, 3.0)
            expected = exp(-zeta^2) * real(
                continuous_order_transition(mu, zeta),
            )
            @test _continuous_relative_error(
                scaled_continuous_order_transition(mu, zeta), expected,
            ) < 3e-12
        end
        @test scaled_continuous_order_transition(1.0, 30.0) == erfc(-30.0)
        @test _continuous_relative_error(
            scaled_continuous_order_transition(1.7, 8.0),
            Float64(real(exp(big"-64") * _continuous_oracle_big(1.7, 8.0))),
        ) < 4e-12

        # A quadrature interval extending from the endpoint to a distant
        # positive saddle can otherwise miss the entire left half of the
        # Gaussian layer while reporting a small embedded-rule error.
        for zeta in (100.0, 1e6)
            value = scaled_continuous_order_transition(2.0, zeta)
            reference = exp(-zeta^2) +
                        sqrt(pi) * zeta * erfc(-zeta)
            @test _continuous_relative_error(value, reference) < 2e-12
        end
        @test isfinite(scaled_continuous_order_transition(2.0, 1e18))
        @test_throws DomainError scaled_continuous_order_transition(2.1, 1e18)
        @test scaled_continuous_order_transition(2.0f0, -10.0f0) >= 0.0f0
        negative_order_two = scaled_continuous_order_transition(2.0, -27.0)
        negative_order_two_reference = Float64(
            exp(-big"27"^2) + sqrt(BigFloat(pi)) * (-big"27") * erfc(big"27"),
        )
        @test negative_order_two ≈ negative_order_two_reference rtol=3e-3
        negative_derivative32 = ForwardDiff.derivative(-10.0f0) do value
            scaled_continuous_order_transition(2.0f0, value)
        end
        negative_derivative_reference32 = setprecision(BigFloat, 256) do
            Float32(sqrt(BigFloat(pi)) * erfc(BigFloat(10)))
        end
        @test negative_derivative32 == negative_derivative_reference32
        for coordinate in (-30.0, 30.0)
            tangent = floatmax(Float64)
            differentiated = @inferred scaled_continuous_order_transition(
                1.0, ForwardDiff.Dual(coordinate, tangent),
            )
            reference = setprecision(BigFloat, 512) do
                convert(
                    Float64,
                    2exp(-BigFloat(coordinate)^2) /
                    sqrt(BigFloat(pi)) * BigFloat(tangent),
                )
            end
            @test ForwardDiff.partials(differentiated, 1) == reference
        end
        differentiated32 = @inferred scaled_continuous_order_transition(
            1.0f0,
            ForwardDiff.Dual(-10.0f0, floatmax(Float32)),
        )
        reference32 = setprecision(BigFloat, 256) do
            convert(
                Float32,
                2exp(-BigFloat(10)^2) / sqrt(BigFloat(pi)) *
                BigFloat(floatmax(Float32)),
            )
        end
        @test ForwardDiff.partials(differentiated32, 1) == reference32

        @test continuous_order_transition(340.0, 0.0) == 1.0 + 0.0im
        @test scaled_continuous_order_transition(340.0, 0.0) == 1.0
        @test continuous_order_transition(1e-307, 0.0) == 1.0 + 0.0im
        @test scaled_continuous_order_transition(1e-307, 0.0) == 1.0
        @test continuous_order_transition(70.0f0, 0.0f0) == 1.0f0 + 0.0f0im
        @test scaled_continuous_order_transition(70.0f0, 0.0f0) == 1.0f0
        @test ForwardDiff.derivative(
            value -> real(continuous_order_transition(340.0, value)), 0.0,
        ) ≈ 2sqrt(170.0) rtol=2e-3
        @test isfinite(ForwardDiff.derivative(
            value -> scaled_continuous_order_transition(70.0f0, value), 0.0f0,
        ))
        for mu in (1e100, floatmax(Float64))
            derivative = ForwardDiff.derivative(0.0) do value
                real(continuous_order_transition(mu, value))
            end
            reference = sqrt(2.0) * sqrt(mu) * (1 - inv(4mu))
            @test derivative ≈ reference rtol=3e-15
        end
        for mu in (1e-5, 1e-20)
            derivative = ForwardDiff.derivative(0.0) do value
                real(continuous_order_transition(mu, value))
            end
            reference = setprecision(BigFloat, 256) do
                Float64(2gamma((BigFloat(mu) + 1) / 2) / gamma(BigFloat(mu) / 2))
            end
            @test derivative ≈ reference rtol=3e-15
        end
        for mu in (1.0f-4, 1.0f-8)
            derivative = ForwardDiff.derivative(0.0f0) do value
                scaled_continuous_order_transition(mu, value)
            end
            reference = setprecision(BigFloat, 256) do
                Float32(2gamma((BigFloat(mu) + 1) / 2) / gamma(BigFloat(mu) / 2))
            end
            @test derivative ≈ reference rtol=3f-6
        end

        mu = 4.0
        X = 16.0
        t = cispi(0.25) * sqrt(X)
        overlap = sqrt(pi) * cispi(mu / 4) / gamma((mu + 1) / 2) *
                  X^(mu / 2) * continuous_order_transition(mu, -t)
        @test _continuous_relative_error(
            UTDKernels._continuous_order_utd_transition(mu, X), overlap,
        ) < 4e-12
        @test _continuous_relative_error(
            UTDKernels._continuous_order_utd_transition(100.0, 1e6),
            0.999996684045889 + 0.0025249969824810715im,
        ) < 4e-13
        @test _continuous_relative_error(
            UTDKernels._continuous_order_utd_transition(400.0, 1e6),
            0.9991880592695962 + 0.04008892858311313im,
        ) < 4e-12
        @test _continuous_relative_error(
            UTDKernels._continuous_order_utd_transition(1000.0, 1e8),
            0.99999685621728 + 0.0025024973565497437im,
        ) < 4e-12
        @test _continuous_relative_error(
            UTDKernels._continuous_order_utd_transition(10.0, 1e100),
            1.0 + 2.75e-99im,
        ) < 4e-14
        @test_throws DomainError UTDKernels._continuous_order_utd_transition(
            4.0, 16.0; rtol=1e-15, atol=1e-15, maxevals=63,
        )
        @test_throws DomainError UTDKernels._continuous_order_utd_transition(
            1.0, -1.0,
        )
    end

    @testset "analytic-amplitude hierarchy and cubic correction" begin
        cases = (
            (0.8, ComplexF64[1.0, -0.35 + 0.2im, 0.12], 50.0, 1.3, 0.15),
            (1.4, ComplexF64[0.7 - 0.1im, 0.2, -0.08, 0.03im],
             120.0, 0.9, -0.08),
        )
        for (nu, coefficients, k, h, tau) in cases
            hierarchy = continuous_order_moment(nu, coefficients, k, h, tau)
            reference = _continuous_oracle_polynomial(
                nu, coefficients, k, h, tau,
            )
            @test _continuous_relative_error(hierarchy, reference) < 4e-12
            @test continuous_order_moment(
                nu, coefficients, k, h, tau; order=1,
            ) == continuous_order_moment(
                nu, coefficients[1:2], k, h, tau,
            )
            mutated = copy(coefficients)
            mutated[2] = 0
            @test abs(
                hierarchy - continuous_order_moment(nu, mutated, k, h, tau),
            ) > 1e-6
        end
        single = continuous_order_moment(0.8, 2, 50.0, 1.3, 0.15, 0.12)
        @test single == continuous_order_moment(
            0.8, ComplexF64[0, 0, 0.12], 50.0, 1.3, 0.15,
        )

        nu, h, cubic, scaled_offset = 0.9, 1.0, 0.15, 0.6
        wavenumbers = [40.0, 80.0, 160.0, 320.0]
        leading_errors = Float64[]
        corrected_errors = Float64[]
        for k in wavenumbers
            tau = scaled_offset * sqrt(2h / k)
            reference = _continuous_oracle_cubic(nu, k, h, tau, cubic)
            leading = continuous_order_moment(nu, 0, k, h, tau, 1.0)
            correction = continuous_order_moment(
                nu, 3, k, h, tau, -im * k * cubic / 6,
            )
            push!(leading_errors, _continuous_relative_error(leading, reference))
            push!(corrected_errors,
                  _continuous_relative_error(leading + correction, reference))
        end
        @test all(corrected_errors .< leading_errors)
        @test corrected_errors[end] < 0.01
        @test _continuous_log_slope(wavenumbers, leading_errors) ≈ -0.5 atol=0.08
        @test _continuous_log_slope(wavenumbers, corrected_errors) ≈ -1.0 atol=0.08
    end

    @testset "Float32, coordinate AD, and allocations" begin
        value32 = continuous_order_transition(Float32(1.2), ComplexF32(0.4, 0.3))
        value64 = continuous_order_transition(1.2, 0.4 + 0.3im)
        @test _continuous_relative_error(ComplexF64(value32), value64) < 3e-5

        derivative = ForwardDiff.derivative(0.4) do coordinate
            real(continuous_order_transition(1.2, coordinate + 0.3im))
        end
        expected = real(
            UTDKernels._continuous_order_coordinate_derivative(1.2, 0.4 + 0.3im),
        )
        @test derivative ≈ expected rtol=3e-8 atol=3e-10

        scaled_derivative = ForwardDiff.derivative(0.4) do coordinate
            scaled_continuous_order_transition(1.2, coordinate)
        end
        value = continuous_order_transition(1.2, 0.4)
        value_derivative =
            UTDKernels._continuous_order_coordinate_derivative(1.2, 0.4)
        scaled_expected = real(exp(-0.4^2) * (value_derivative - 0.8value))
        @test scaled_derivative ≈ scaled_expected rtol=3e-8 atol=3e-10

        # The exact-coalescence value is one, but a zero-primal Dual must keep
        # the nonzero canonical-coordinate tangent.
        for mu in (1.0, 1.2)
            expected_at_zero = 2gamma((mu + 1) / 2) / gamma(mu / 2)
            derivative_at_zero = ForwardDiff.derivative(0.0) do coordinate
                real(continuous_order_transition(mu, coordinate))
            end
            scaled_derivative_at_zero = ForwardDiff.derivative(0.0) do coordinate
                scaled_continuous_order_transition(mu, coordinate)
            end
            @test derivative_at_zero ≈ expected_at_zero rtol=3e-11
            @test scaled_derivative_at_zero ≈ expected_at_zero rtol=3e-11
        end
        for coordinate in (3.0, 5.0, 10.0)
            derivative = ForwardDiff.derivative(coordinate) do value
                scaled_continuous_order_transition(1.0, value)
            end
            expected = 2exp(-coordinate^2) / sqrt(pi)
            @test derivative ≈ expected rtol=3e-14 atol=0.0
        end
        for coordinate in (3.0f0, 5.0f0, 10.0f0)
            derivative = ForwardDiff.derivative(coordinate) do value
                scaled_continuous_order_transition(1.0f0, value)
            end
            expected = 2f0 * exp(-coordinate^2) / sqrt(Float32(pi))
            @test derivative ≈ expected rtol=3f-6 atol=0f0
        end
        for coordinate in (80.0f0, 120.0f0, 500.0f0, 1.0f6)
            derivative = ForwardDiff.derivative(coordinate) do value
                scaled_continuous_order_transition(2.0f0, value)
            end
            expected = sqrt(Float32(pi)) * erfc(-coordinate)
            @test derivative ≈ expected rtol=3f-6 atol=0f0
            tight_derivative = ForwardDiff.derivative(coordinate) do value
                scaled_continuous_order_transition(
                    2.0f0, value; rtol=3.0f-6,
                )
            end
            @test tight_derivative ≈ expected rtol=3f-6 atol=0f0
        end
        for coordinate in (1e6, 1e12)
            derivative = ForwardDiff.derivative(coordinate) do value
                scaled_continuous_order_transition(2.0, value)
            end
            @test derivative ≈ sqrt(pi) * erfc(-coordinate) rtol=2e-12 atol=0.0
        end
        for (T, coordinate) in ((Float32, 1.0f10), (Float64, 1e155))
            tangent = floatmax(T) / T(4)
            value = @inferred scaled_continuous_order_transition(
                T(2), ForwardDiff.Dual(coordinate, tangent),
            )
            expected_value, expected_tangent = setprecision(BigFloat, 4096) do
                wide_coordinate = BigFloat(coordinate)
                (
                    convert(
                        T,
                        exp(-(wide_coordinate * wide_coordinate)) +
                        sqrt(BigFloat(pi)) * wide_coordinate *
                        erfc(-wide_coordinate),
                    ),
                    convert(
                        T,
                        sqrt(BigFloat(pi)) * erfc(-wide_coordinate) *
                        BigFloat(tangent),
                    ),
                )
            end
            @test ForwardDiff.value(value) == expected_value
            @test ForwardDiff.partials(value, 1) == expected_tangent
        end
        second_derivative(f, value) = ForwardDiff.derivative(
            inner -> ForwardDiff.derivative(f, inner), value,
        )
        for coordinate in (1.0, 3.0, 5.0)
            derivative = second_derivative(
                value -> scaled_continuous_order_transition(2.0, value),
                coordinate,
            )
            @test derivative ≈ 2exp(-coordinate^2) rtol=3e-11 atol=2e-15
        end
        @test second_derivative(
            value -> scaled_continuous_order_transition(1.0, value), 0.0,
        ) == 0.0
        @test second_derivative(
            value -> scaled_continuous_order_transition(2.0, value), 0.0,
        ) ≈ 2.0 rtol=3e-15
        @test second_derivative(
            value -> real(continuous_order_transition(1.0, value)), 0.0,
        ) ≈ 2.0 rtol=3e-15
        @test_throws ArgumentError ForwardDiff.derivative(1000.0) do outer
            ForwardDiff.derivative(
                inner -> scaled_continuous_order_transition(3.0, inner), outer,
            )
        end
        for coordinate in (1e180, 1e200)
            tangent = floatmax(Float64)
            value = continuous_order_transition(
                2.0, ForwardDiff.Dual(-coordinate, tangent),
            )
            reference = setprecision(BigFloat, 4096) do
                wide_coordinate = BigFloat(coordinate)
                derivative = sqrt(BigFloat(pi)) * erfcx(wide_coordinate) *
                             (1 + 2wide_coordinate^2) - 2wide_coordinate
                convert(Float64, derivative * BigFloat(tangent))
            end
            @test ForwardDiff.value(real(value)) == 0.0
            @test ForwardDiff.partials(real(value), 1) == reference
        end
        for (T, coordinate) in (
            (Float32, Float32(10)),
            (Float64, 1e4),
            (Float64, 1e180),
        )
            tangent = floatmax(T)
            value = @inferred continuous_order_transition(
                one(T), ForwardDiff.Dual(-coordinate, tangent),
            )
            reference = setprecision(BigFloat, 4096) do
                x = BigFloat(coordinate)
                derivative = 2 / sqrt(BigFloat(pi)) - 2x * erfcx(x)
                convert(T, derivative * BigFloat(tangent))
            end
            @test ForwardDiff.partials(real(value), 1) == reference
        end
        nested_derivative(function_, value, order) = order == 0 ?
            function_(value) : ForwardDiff.derivative(
                local_value -> nested_derivative(
                    function_, local_value, order - 1,
                ),
                value,
            )
        fifth_derivative = nested_derivative(
            value -> real(continuous_order_transition(1.0, value)),
            -1e20,
            5,
        )
        @test fifth_derivative ==
              _continuous_oracle_unit_negative_derivative(1e20, 5)
        for mu in (0.5, 1.5, 2.5, 3.3)
            coordinate = 30.0
            tangent = floatmax(Float64)
            value = scaled_continuous_order_transition(
                mu, ForwardDiff.Dual(-coordinate, tangent),
            )
            reference = _continuous_oracle_scaled_negative_derivative(
                mu, coordinate, tangent,
            )
            @test ForwardDiff.value(value) == 0.0
            @test ForwardDiff.partials(value, 1) == reference
        end
        recovered_tangent = floatmax(Float64)
        recovered_value = continuous_order_transition(
            1.5, ForwardDiff.Dual(-1e180, recovered_tangent),
        )
        @test ForwardDiff.partials(real(recovered_value), 1) ==
              _continuous_oracle_negative_coordinate_derivative(
                  1.5, 1e180, recovered_tangent,
              )
        for (mu, coordinate) in ((0.5, 1e150), (2.5, 1e180))
            tangent = floatmax(Float64)
            value = UTDKernels._continuous_order_parameter_derivative(
                mu, ForwardDiff.Dual(-coordinate, tangent),
            )
            reference =
                _continuous_oracle_order_sensitivity_coordinate_derivative(
                    mu, coordinate, tangent,
                )
            @test ForwardDiff.partials(real(value), 1) == reference
        end

        gamma_value32 = UTDKernels._continuous_order_utd_transition(
            400.0f0, 1.0f6,
        )
        gamma_reference64 = UTDKernels._continuous_order_utd_transition(
            400.0, 1.0e6,
        )
        @test _continuous_relative_error(
            ComplexF64(gamma_value32), gamma_reference64,
        ) < 3f-6

        extreme_moment = continuous_order_moment(
            100.0, 64, 1e300, 1e-300, 0.0, 1.0,
        )
        extreme_reference = setprecision(BigFloat, 512) do
            order = BigFloat(164)
            ComplexF64(
                cispi(-order / 4) * gamma(order / 2) / 2 *
                (BigFloat(2) / (BigFloat(1e300) * BigFloat(1e-300)))^(order / 2),
            )
        end
        @test _continuous_relative_error(
            extreme_moment, extreme_reference,
        ) < 3e-13
        for scale in (1e-158, 1e-160, 1e-161)
            value = continuous_order_moment(
                1.0, 0, scale, scale, 0.0, 1.0,
            )
            reference = setprecision(BigFloat, 512) do
                stored_scale = BigFloat(scale)
                ComplexF64(
                    cispi(-BigFloat(1) / 4) * gamma(BigFloat(0.5)) / 2 *
                    sqrt(BigFloat(2) / (stored_scale * stored_scale)),
                )
            end
            @test _continuous_relative_error(value, reference) < 3e-13
        end
        for (k, h, coefficient) in (
            (1f30, 1f30, 1f30),
            (1e200, 1e200, 1e300),
            (1e308, 1e308, 1e308),
        )
            value = continuous_order_moment(
                typeof(k)(2), 0, k, h, zero(k), coefficient,
            )
            reference = setprecision(BigFloat, 512) do
                convert(
                    typeof(value),
                    -complex(zero(BigFloat), one(BigFloat)) *
                    BigFloat(coefficient) / (BigFloat(k) * BigFloat(h)),
                )
            end
            @test value == reference
        end
        for (k, h, tau, coefficient, tolerance) in (
            (10f0, 10f0, 0.5f0, floatmax(Float32), 4f-6),
            (10.0, 10.0, 0.5, floatmax(Float64), 4e-15),
        )
            value = continuous_order_moment(
                typeof(k)(2), 0, k, h, tau, coefficient,
            )
            reference = setprecision(BigFloat, 1024) do
                wide_k = BigFloat(k)
                wide_h = BigFloat(h)
                wide_tau = BigFloat(tau)
                coordinate = complex(cospi(BigFloat(0.25)), sinpi(BigFloat(0.25))) *
                    wide_tau * sqrt(wide_k / (2wide_h))
                canonical = _continuous_oracle_hypergeometric_big(
                    BigFloat(2), coordinate; precision=1024,
                )
                convert(
                    typeof(value),
                    -complex(zero(BigFloat), one(BigFloat)) *
                    BigFloat(coefficient) * canonical / (wide_k * wide_h),
                )
            end
            @test all(isfinite, (real(value), imag(value)))
            @test _continuous_relative_error(value, reference) < tolerance
        end
        for R in (Float32, Float64)
            tangent = floatmax(R)
            coefficient = ForwardDiff.Dual(one(R), tangent)
            value = continuous_order_moment(
                R(2), 0, R(10), R(10), R(0.5), coefficient,
            )
            unit_value = continuous_order_moment(
                R(2), 0, R(10), R(10), R(0.5), one(R),
            )
            @test ForwardDiff.value(real(value)) ≈
                  real(unit_value) rtol=8eps(R)
            @test ForwardDiff.value(imag(value)) ≈
                  imag(unit_value) rtol=8eps(R)
            @test ForwardDiff.partials(real(value), 1) ≈
                  tangent * real(unit_value) rtol=8eps(R)
            @test ForwardDiff.partials(imag(value), 1) ≈
                  tangent * imag(unit_value) rtol=8eps(R)
        end
        for R in (Float32, Float64)
            nu = R(1.3)
            k = one(R)
            h = one(R)
            tau = R(0.5)
            basis0 = continuous_order_moment(nu, 0, k, h, tau, one(R))
            basis1 = continuous_order_moment(nu, 1, k, h, tau, one(R))
            coefficient1 = R === Float32 ? R(1e30) : R(1e300)
            coefficient0 = -coefficient1 * basis1 / basis0
            hierarchy_reference = function (coefficients)
                setprecision(BigFloat, 1024) do
                    wide_nu = BigFloat(nu)
                    wide_k = BigFloat(k)
                    wide_h = BigFloat(h)
                    wide_tau = BigFloat(tau)
                    coordinate = cispi(BigFloat(0.25)) * wide_tau *
                                 sqrt(wide_k / (2wide_h))
                    total = zero(Complex{BigFloat})
                    for index in 0:1
                        order = wide_nu + index
                        canonical = _continuous_oracle_hypergeometric_big(
                            order, coordinate; precision=1024,
                        )
                        prefactor = cispi(-order / 4) * gamma(order / 2) / 2 *
                                    (2 / (wide_k * wide_h))^(order / 2)
                        total += Complex{BigFloat}(coefficients[index + 1]) *
                                 prefactor * canonical
                    end
                    convert(typeof(basis0), total)
                end
            end
            coefficients = typeof(basis0)[coefficient0, coefficient1]
            value = continuous_order_moment(nu, coefficients, k, h, tau)
            reference = hierarchy_reference(coefficients)
            tolerance = R === Float32 ? R(5e-5) : R(5e-13)
            @test _continuous_relative_error(value, reference) < tolerance

            delta = R === Float32 ? R(1e-3) : R(1e-5)
            moderate_coefficients = typeof(basis0)[
                (-basis1 / basis0) * (one(R) - delta), one(R),
            ]
            moderate_value = continuous_order_moment(
                nu, moderate_coefficients, k, h, tau,
            )
            moderate_reference = hierarchy_reference(moderate_coefficients)
            @test _continuous_relative_error(
                moderate_value, moderate_reference,
            ) < tolerance
        end
        for (R, nu, k, h, tangent0, tangent1) in (
            (Float32, 2.8347993f0, 3.3687243f0, 0.14522459f0,
             ComplexF32(-1.09268556f30, 2.0292733f30),
             ComplexF32(1.0f30, -3.0000002f29)),
            (Float64, 3.1727565186126006, 2.47231145214507,
             0.6045090663647765,
             ComplexF64(-6.673300567485151e299, 1.2393272482472423e300),
             ComplexF64(1.0e300, -3.0e299)),
        )
            coefficients = [
                complex(
                    ForwardDiff.Dual(one(R), real(tangent0)),
                    ForwardDiff.Dual(zero(R), imag(tangent0)),
                ),
                complex(
                    ForwardDiff.Dual(one(R), real(tangent1)),
                    ForwardDiff.Dual(zero(R), imag(tangent1)),
                ),
            ]
            value = continuous_order_moment(
                nu, coefficients, k, h, zero(R),
            )
            tangent = complex(
                ForwardDiff.partials(real(value), 1),
                ForwardDiff.partials(imag(value), 1),
            )
            reference = setprecision(BigFloat, 512) do
                total = zero(Complex{BigFloat})
                for (index, coefficient_tangent) in
                    enumerate((tangent0, tangent1))
                    order = BigFloat(nu) + index - 1
                    basis = cispi(-order / 4) * gamma(order / 2) / 2 *
                            (2 / (BigFloat(k) * BigFloat(h)))^(order / 2)
                    total += Complex{BigFloat}(coefficient_tangent) * basis
                end
                Complex{R}(total)
            end
            @test tangent == reference
        end

        _continuous_general_allocation_probe()
        _continuous_endpoint_allocation_probe()
        _continuous_unit_allocation_probe()
        _continuous_scaled_allocation_probe()
        _continuous_moment_allocation_probe()
        _continuous_hierarchy_allocation_probe()
        @test @allocated(_continuous_general_allocation_probe()) <= 6144
        @test @allocated(_continuous_endpoint_allocation_probe()) <= 6144
        @test @allocated(_continuous_unit_allocation_probe()) == 0
        @test @allocated(_continuous_scaled_allocation_probe()) == 0
        @test @allocated(_continuous_moment_allocation_probe()) <= 6144
        @test @allocated(_continuous_hierarchy_allocation_probe()) <= 16384
    end
end
