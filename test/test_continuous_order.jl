using ForwardDiff
using SpecialFunctions: erfc, erfcx, gamma
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
        @test_throws ArgumentError continuous_order_moment(
            1.0, Number[1.0], 1.0, 1.0, 0.0,
        )
        @test_throws DomainError UTDKernels._continuous_order_utd_transition(
            4097.0, 1e6,
        )
        @test_throws ArgumentError ForwardDiff.derivative(
            order -> real(continuous_order_transition(order, 0.2)), 1.2,
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
