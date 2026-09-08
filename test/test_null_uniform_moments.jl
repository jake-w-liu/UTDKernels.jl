using ForwardDiff

include("support/null_uniform_moments_oracle.jl")

_null_relative_error(value, reference; floor=1e-300) =
    abs(value - reference) / max(abs(reference), floor)

const NULL_ALLOC_COEFFICIENTS = ComplexF64[0.08, 1.0, 0.35, -0.15, 0.08]
const NULL_ALLOC_BASE = ComplexF64[1.0, 0.2, -0.05]
@noinline _null_moderate_allocation_probe() = faddeeva_moments(0.7 + 0.4im, 4)
@noinline _null_large_allocation_probe() = faddeeva_moments(1e7 + 0.4im, 4)
@noinline _null_hard_allocation_probe() = faddeeva_moments(6.0 + 1e-10im, 10)
@noinline _null_max_recurrence_allocation_probe() = faddeeva_moments(0.0im, 64)
@noinline _null_max_asymptotic_allocation_probe() = faddeeva_moments(1e7 + 0.4im, 64)
@noinline _null_transition_allocation_probe() =
    null_uniform_transition(250.0, 0.9 + 0.45im, NULL_ALLOC_COEFFICIENTS)
@noinline _shifted_null_allocation_probe() =
    shifted_null_transition(900.0, 0.6 + 0.5im, -1.1, 3, NULL_ALLOC_BASE)

@testset "Null-uniform Faddeeva moments" begin
    @testset "Gaussian moments and low-order identities" begin
        @test UTDKernels._null_gaussian_moment(0, 1.0) ≈ sqrt(pi) atol=1e-15
        @test UTDKernels._null_gaussian_moment(1, 1.0) == 0.0
        @test UTDKernels._null_gaussian_moment(2, 1.0) ≈ sqrt(pi) / 2 atol=1e-15
        @test UTDKernels._null_gaussian_moment(4, 1.0) ≈ 3sqrt(pi) / 4 atol=1e-15

        z = 0.55 + 0.4im
        moments = faddeeva_moments(z, 3)
        w = UTDKernels._faddeeva_w(z)
        @test moments[1] == w
        @test moments[2] ≈ z * w - im / sqrt(pi) atol=2e-15
        @test moments[3] ≈ z^2 * w - im * z / sqrt(pi) atol=2e-15
        @test moments[4] ≈ z^3 * w - im * (z^2 + 0.5) / sqrt(pi) atol=2e-14
    end

    @testset "Faddeeva, recurrence, and derivative identities" begin
        z = 0.7 + 0.4im
        moments = faddeeva_moments(z, 3)
        @test moments[1] == UTDKernels._faddeeva_w(z)
        @test moments[1] ≈ 0.4922894280872569 + 0.33153472612943524im atol=2e-15
        frozen = (
            0.21198870920930571 - 0.13519950402224884im,
            0.20247189805541352 - 0.0098441691318518904im,
            0.14566799629153021 - 0.207996950944009im,
        )
        for order in 0:2
            expected = z * moments[order + 1] -
                       im * UTDKernels._null_gaussian_moment(order, 1.0) / pi
            @test moments[order + 2] ≈ expected atol=1e-15
            @test moments[order + 2] ≈ frozen[order + 1] atol=2e-15
        end
        derivative = -2z * moments[1] + 2im / sqrt(pi)
        @test moments[2] ≈ -derivative / 2 atol=2e-15

        wrong = z * moments[1] + im / sqrt(pi)
        @test abs(moments[2] - wrong) > 1e-4
    end

    @testset "direct quadrature and polynomial hierarchy" begin
        z = 0.6 + 0.7im
        moments = faddeeva_moments(z, 3)
        for order in 0:3
            reference = _null_oracle_direct_moment(z, order)
            @test _null_relative_error(moments[order + 1], reference) < 3e-12
        end

        k = 250.0
        coefficients = ComplexF64[0.08, 1.0, 0.35, -0.15, 0.08]
        value = null_uniform_transition(k, 0.9 + 0.45im, coefficients)
        reference = _null_oracle_polynomial_integral(
            k, 0.9 + 0.45im, coefficients,
        )
        @test _null_relative_error(value, reference) < 2e-12
        @test value ≈ 0.044927602623239422 + 0.023769556335399587im atol=2e-14
    end

    @testset "exact null layer and field-order laws" begin
        k = 1000.0
        z = -0.3 + 0.5im
        coefficients = ComplexF64[1.2 / sqrt(k), 1.0]
        value = null_uniform_transition(k, z, coefficients)
        reference = _null_oracle_polynomial_integral(k, z, coefficients)
        @test value ≈ reference atol=5e-13
        @test value ≈ 0.018969937178984025 - 0.012813278192590479im atol=2e-14

        ks = 10 .^ range(2, 6; length=9)
        for (zero_order, expected_slope) in ((1, -0.5), (2, -1.0), (3, -1.5))
            coefficient = zeros(zero_order + 1)
            coefficient[end] = 1
            values = [abs(null_uniform_transition(local_k, 0.7 + 0.4im, coefficient))
                      for local_k in ks]
            x = log.(ks)
            y = log.(values)
            slope = sum((x .- sum(x) / length(x)) .* (y .- sum(y) / length(y))) /
                    sum((x .- sum(x) / length(x)).^2)
            @test slope ≈ expected_slope atol=1e-12
        end
    end

    @testset "switch, asymptotic stability, and Stokes continuation" begin
        below = faddeeva_moments(9.9 + 0im, 2)
        on = faddeeva_moments(10.0 + 0im, 2)
        above = faddeeva_moments(10.1 + 0im, 2)
        @test below == UTDKernels._null_moment_recurrence(9.9 + 0im, 2)
        @test on == UTDKernels._null_moment_recurrence(10.0 + 0im, 2)
        @test above == ComplexF64[
            UTDKernels._null_moment_asymptotic(10.1 + 0im, order)
            for order in 0:2
        ]
        recurrence = UTDKernels._null_moment_recurrence(10.0 + 0im, 3)
        for order in 0:3
            asymptotic = UTDKernels._null_moment_asymptotic(10.0 + 0im, order)
            @test _null_relative_error(recurrence[order + 1], asymptotic) < 5e-12
        end

        large_z = 1e7 + 0.4im
        stable = faddeeva_moments(large_z, 4)
        for order in 0:4
            reference = _null_oracle_asymptotic(large_z, order)
            @test _null_relative_error(stable[order + 1], reference) < 2e-13
        end
        unstable = UTDKernels._null_moment_recurrence(large_z, 4)
        @test _null_relative_error(unstable[5], _null_oracle_asymptotic(large_z, 4)) > 1e6

        lower_z = 1.0 - 12im
        lower = faddeeva_moments(lower_z, 2)
        for order in 0:2
            reference = ComplexF64(_null_oracle_moment(lower_z, order))
            @test _null_relative_error(lower[order + 1], reference) < 3e-14
            algebraic = UTDKernels._null_moment_asymptotic(
                lower_z, order; include_continuation=false,
            )
            @test abs(reference) / max(abs(algebraic), 1e-300) > 1e20
        end
    end

    @testset "order-aware cancellation recovery" begin
        cases = (
            (9.0 + 1e-10im, 7),
            (6.0 + 1e-10im, 10),
            (3.0 + 1e-10im, 20),
            (-9.0 - 1e-10im, 7),
            (9.0 + 1e-10im, 32),
        )
        for (z, order) in cases
            value = faddeeva_moments(z, order)[order + 1]
            reference = ComplexF64(_null_oracle_moment(z, order; precision=768))
            @test _null_relative_error(value, reference) < 2e-11
        end

        for (z, order) in (
            (ComplexF32(10.01, 0.1), 32),
            (ComplexF32(20.0, 0.1), 48),
            (ComplexF32(9.0, 0.1), 64),
        )
            value = faddeeva_moments(z, order)[order + 1]
            reference = ComplexF32(_null_oracle_moment(z, order; precision=768))
            @test isfinite(value)
            @test _null_relative_error(value, reference; floor=1f-30) < 3f-6
        end

        # A vanishing lower-half-plane exponential must be handled before the
        # polynomial continuation factor can overflow.
        lower_z = -1e100 - 1im
        lower_value = faddeeva_moments(lower_z, 16)[end]
        lower_reference = ComplexF64(
            _null_oracle_asymptotic(lower_z, 16; precision=768),
        )
        @test _null_relative_error(lower_value, lower_reference) < 3e-13

        cancellation_z = complex(1e5, -sqrt(1e10 - 700.0))
        cancellation_value = faddeeva_moments(cancellation_z, 64)[end]
        cancellation_reference = ComplexF64(
            _null_oracle_asymptotic(cancellation_z, 64; precision=1024),
        )
        @test isfinite(cancellation_value)
        @test _null_relative_error(
            cancellation_value, cancellation_reference,
        ) < 3e-11

        sparse_lower = faddeeva_moments(1 - 10im, 1; max_terms=1)
        for order in 0:1
            reference = ComplexF64(
                _null_oracle_moment(1 - 10im, order; precision=768),
            )
            @test isfinite(sparse_lower[order + 1])
            @test _null_relative_error(
                sparse_lower[order + 1], reference,
            ) < 3e-13
        end
    end

    @testset "shifted-null binomial basis" begin
        k = 900.0
        z = 0.6 + 0.5im
        lambda = -1.1
        base_coefficients = ComplexF64[1.0, 0.2, -0.05]
        for zero_order in 0:3
            value = shifted_null_transition(
                k, z, lambda, zero_order, base_coefficients,
            )
            moments = faddeeva_moments(z, zero_order + length(base_coefficients) - 1)
            reference = zero(ComplexF64)
            inverse_scale = inv(sqrt(k))
            for offset in 0:(length(base_coefficients) - 1)
                basis = zero(ComplexF64)
                for index in 0:zero_order
                    basis += binomial(zero_order, index) * lambda^(zero_order - index) *
                             moments[offset + index + 1]
                end
                reference += base_coefficients[offset + 1] *
                             inverse_scale^(offset + zero_order) * basis
            end
            @test value ≈ reference atol=2e-14
        end
    end

    @testset "types, AD, domains, and allocations" begin
        moments32 = faddeeva_moments(0.7f0 + 0.4f0im, 4)
        @test eltype(moments32) === ComplexF32
        @test null_uniform_transition(
            250f0, 0.9f0 + 0.45f0im, Float32[0.08, 1, 0.35],
        ) isa ComplexF32
        @test shifted_null_transition(
            900f0, 0.6f0 + 0.5f0im, -1.1f0, 2, Float32[1, 0.2],
        ) isa ComplexF32

        moments16 = faddeeva_moments(ComplexF16(0.7, 0.4), 20)
        @test eltype(moments16) === ComplexF32
        reference16 = ComplexF32(
            _null_oracle_moment(ComplexF16(0.7, 0.4), 20; precision=768),
        )
        @test _null_relative_error(
            moments16[end], reference16; floor=1f-30,
        ) < 3f-5
        moments16_large = faddeeva_moments(ComplexF16(10, 0.1), 20)
        @test eltype(moments16_large) === ComplexF32
        reference16_large = ComplexF32(
            _null_oracle_moment(ComplexF16(10, 0.1), 20; precision=768),
        )
        @test _null_relative_error(
            moments16_large[end], reference16_large; floor=1f-30,
        ) < 3f-5
        @test null_uniform_transition(
            Float16(250), ComplexF16(0.9, 0.45), Float16[0.08, 1, 0.35],
        ) isa ComplexF32
        @test shifted_null_transition(
            Float16(900), ComplexF16(0.6, 0.5), Float16(-1.1), 2,
            Float16[1, 0.2],
        ) isa ComplexF32
        inferred_coefficients = ComplexF32[1, 0.2]
        @test (@inferred null_uniform_transition(
            1f0, ComplexF32(0.7, 0.4), inferred_coefficients,
        )) isa ComplexF32
        @test (@inferred shifted_null_transition(
            1f0, ComplexF32(0.7, 0.4), 0.2f0, 1,
            inferred_coefficients,
        )) isa ComplexF32

        derivative = ForwardDiff.derivative(0.7) do real_part
            real(faddeeva_moments(complex(real_part, 0.4), 2)[3])
        end
        step = 1e-6
        centered = (
            real(faddeeva_moments(0.7 + step + 0.4im, 2)[3]) -
            real(faddeeva_moments(0.7 - step + 0.4im, 2)[3])
        ) / (2step)
        @test derivative ≈ centered rtol=2e-8 atol=2e-10

        derivative_case = ComplexF32(6.3884974, 0.023895629)
        derivative_order = 14
        differentiated_z = complex(
            ForwardDiff.Dual(real(derivative_case), one(Float32)),
            ForwardDiff.Dual(imag(derivative_case), zero(Float32)),
        )
        differentiated_moments = @inferred faddeeva_moments(
            differentiated_z, derivative_order,
        )
        differentiated_moment = differentiated_moments[end]
        recovered_derivative = complex(
            ForwardDiff.partials(real(differentiated_moment), 1),
            ForwardDiff.partials(imag(differentiated_moment), 1),
        )
        reference_derivative = ComplexF32(
            _null_oracle_moment_derivative(
                derivative_case, derivative_order; precision=1024,
            ),
        )
        @test recovered_derivative == reference_derivative
        second_component(component) = ForwardDiff.derivative(
            outer -> ForwardDiff.derivative(
                inner -> component(faddeeva_moments(
                    complex(inner, imag(derivative_case)), derivative_order,
                )[end]),
                outer,
            ),
            real(derivative_case),
        )
        recovered_second_derivative = complex(
            second_component(real), second_component(imag),
        )
        reference_second_derivative = ComplexF32(
            _null_oracle_moment_second_derivative(
                derivative_case, derivative_order; precision=1024,
            ),
        )
        @test recovered_second_derivative == reference_second_derivative

        @test_throws ArgumentError faddeeva_moments(complex(big"0.7", big"0.4"), 2)
        @test_throws DomainError faddeeva_moments(NaN + 0im, 2)
        @test_throws DomainError faddeeva_moments(1.0 + im, -1)
        @test_throws DomainError faddeeva_moments(1.0 + im, 65)
        @test length(faddeeva_moments(0.7 + 0.4im, UInt(0))) == 1
        @test length(faddeeva_moments(0.7 + 0.4im, UInt8(0))) == 1
        @test length(faddeeva_moments(0.7 + 0.4im, UInt(3))) == 4
        @test_throws DomainError faddeeva_moments(1.0 + im, 2; switch=0.0)
        @test_throws DomainError faddeeva_moments(1.0 + im, 2; rtol=0.0)
        @test_throws DomainError faddeeva_moments(1.0 + im, 2; max_terms=0)
        nonfinite_rtol = ForwardDiff.Dual(1e-6, Inf)
        nonfinite_switch = ForwardDiff.Dual(10.0, Inf)
        dual_z = complex(
            ForwardDiff.Dual(0.2, 1.0),
            ForwardDiff.Dual(0.1, 0.0),
        )
        for z in (0.2 + 0.1im, dual_z)
            @test_throws DomainError faddeeva_moments(
                z, 2; rtol=nonfinite_rtol,
            )
            @test_throws DomainError faddeeva_moments(
                z, 2; switch=nonfinite_switch,
            )
        end
        @test_throws ArgumentError faddeeva_moments(
            0.2 + 0.1im, 2; rtol=ForwardDiff.Dual(1e-6, 0.0),
        )
        for coefficients in (Float64[], Float64[1])
            @test_throws DomainError null_uniform_transition(
                1.0, dual_z, coefficients; rtol=nonfinite_rtol,
            )
            @test_throws DomainError null_uniform_transition(
                1.0, dual_z, coefficients; switch=nonfinite_switch,
            )
            @test_throws DomainError shifted_null_transition(
                1.0, dual_z, 0.1, 1, coefficients;
                rtol=nonfinite_rtol,
            )
            @test_throws DomainError shifted_null_transition(
                1.0, dual_z, 0.1, 1, coefficients;
                switch=nonfinite_switch,
            )
        end
        @test_throws ArgumentError faddeeva_moments(
            1e100 + 0.1im, 64; rtol=eps(Float64), max_terms=1,
        )
        @test_throws DomainError null_uniform_transition(0.0, 1.0 + im, [1.0])
        @test_throws DomainError null_uniform_transition(1.0, 1.0 + im, [NaN])
        @test null_uniform_transition(1.0, 1.0 + im, Float64[]) == 0.0im
        @test shifted_null_transition(
            1.0, 1.0 + im, 0.0, 2, Float64[],
        ) == 0.0im
        @test_throws DomainError null_uniform_transition(
            1.0, 1.0 + im, Float64[]; switch=0.0,
        )
        @test_throws DomainError null_uniform_transition(
            1.0, 1.0 + im, Float64[]; order=-1,
        )
        @test_throws DomainError shifted_null_transition(
            1.0, 1.0 + im, Inf, 2, Float64[],
        )
        @test_throws DomainError shifted_null_transition(
            1.0, 1.0 + im, 0.0, 2, Float64[]; max_terms=0,
        )
        @test_throws DomainError null_uniform_transition(
            1.0, 1.0 + im, [1.0, NaN]; order=0,
        )
        @test isfinite(null_uniform_transition(
            1.0, 1.0 + im, [1.0]; order=big(typemax(Int)) + 1,
        ))
        @test_throws ArgumentError null_uniform_transition(
            1.0, 1.0 + im, Number[1.0],
        )
        @test_throws DomainError shifted_null_transition(
            1.0, 1.0 + im, 0.0, -1, [1.0],
        )
        nested_nonfinite = ForwardDiff.Dual(
            ForwardDiff.Dual(1.0, Inf),
            ForwardDiff.Dual(0.0, 0.0),
        )
        @test_throws DomainError null_uniform_transition(
            1.0, nested_nonfinite, Float64[],
        )
        @test_throws DomainError null_uniform_transition(
            nested_nonfinite, 1.0 + im, Float64[],
        )
        @test_throws DomainError shifted_null_transition(
            1.0, 1.0 + im, nested_nonfinite, 1, Float64[],
        )

        sparse = null_uniform_transition(
            1e-300, 0.7 + 0.4im, [1.0, 0.0, 0.0, 0.0],
        )
        @test sparse == faddeeva_moments(0.7 + 0.4im, 0)[1]
        @test null_uniform_transition(
            1e-300, 0.7 + 0.4im, zeros(4),
        ) == 0.0im
        balanced = null_uniform_transition(
            1e-300, 0.7 + 0.4im, [0.0, 0.0, 0.0, 1e-300],
        )
        @test isfinite(balanced)
        balanced_large_k = null_uniform_transition(
            1e300, 0.7 + 0.4im, [0.0, 0.0, 0.0, 1e300],
        )
        balanced_large_reference = 1e300 *
                                   faddeeva_moments(0.7 + 0.4im, 3)[4] *
                                   big"1e-450"
        @test _null_relative_error(
            balanced_large_k, ComplexF64(balanced_large_reference),
        ) < 3e-13
        tiny_coefficient = nextfloat(0.0)
        balanced_product = null_uniform_transition(
            1e-308, 0.7 + 0.4im, [0.0, tiny_coefficient],
        )
        balanced_product_reference = ComplexF64(
            BigFloat(tiny_coefficient) *
            Complex{BigFloat}(faddeeva_moments(0.7 + 0.4im, 1)[2]) *
            inv(sqrt(big"1e-308")),
        )
        @test _null_relative_error(
            balanced_product, balanced_product_reference,
        ) < 3e-13
        for R in (Float32, Float64)
            huge_z = complex(floatmax(R), zero(R))
            for moment_order in (1, 3)
                coefficients = zeros(Complex{R}, moment_order + 1)
                coefficients[end] = complex(floatmax(R), zero(R))
                value = null_uniform_transition(one(R), huge_z, coefficients)
                reference = setprecision(BigFloat, 1536) do
                    convert(
                        typeof(value),
                        BigFloat(floatmax(R)) * _null_oracle_asymptotic(
                            huge_z, moment_order; precision=1536,
                        ),
                    )
                end
                @test value == reference
            end

            diagonal_z = complex(floatmax(R), floatmax(R))
            diagonal_coefficients = Complex{R}[
                zero(R), floatmax(R),
            ]
            diagonal_value = null_uniform_transition(
                one(R), diagonal_z, diagonal_coefficients,
            )
            diagonal_reference = setprecision(BigFloat, 1536) do
                convert(
                    typeof(diagonal_value),
                    BigFloat(floatmax(R)) * _null_oracle_asymptotic(
                        diagonal_z, 1; precision=1536,
                    ),
                )
            end
            @test diagonal_value == diagonal_reference
            @test shifted_null_transition(
                one(R), diagonal_z, zero(R), 1,
                Complex{R}[floatmax(R)],
            ) == diagonal_reference
            @test_throws ArgumentError faddeeva_moments(
                diagonal_z, 31; max_terms=1,
            )
        end
        for (R, local_k, moment_order, large_coefficient) in (
            (Float32, 1f5, 16, 1f20),
            (Float64, 1e20, 32, 1e100),
        )
            local_z = complex(R(0.7), R(0.4))
            local_coefficients = zeros(Complex{R}, moment_order + 1)
            local_coefficients[end] = large_coefficient
            stored_basis = faddeeva_moments(local_z, moment_order)[end]
            reference = setprecision(BigFloat, 1024) do
                wide_basis = complex(
                    BigFloat(real(stored_basis)),
                    BigFloat(imag(stored_basis)),
                )
                convert(
                    Complex{R},
                    BigFloat(large_coefficient) * wide_basis *
                    BigFloat(local_k)^(-BigFloat(moment_order) / 2),
                )
            end
            @test null_uniform_transition(
                local_k, local_z, local_coefficients,
            ) == reference
            @test shifted_null_transition(
                local_k, local_z, zero(R), moment_order,
                Complex{R}[large_coefficient],
            ) == reference
        end

        z = 8.001 + 0.1im
        moments = faddeeva_moments(z, 31)
        coefficients = zeros(ComplexF64, 32)
        coefficients[32] = 1
        coefficients[31] = -moments[32] / moments[31]
        value = null_uniform_transition(1.0, z, coefficients)
        reference = setprecision(BigFloat, 1536) do
            wide_total = zero(Complex{BigFloat})
            for order in 30:31
                wide_total += Complex{BigFloat}(coefficients[order + 1]) *
                    _null_oracle_moment(z, order; precision=1536)
            end
            ComplexF64(wide_total)
        end
        @test value == reference

        for R in (Float32, Float64)
            local_z = R === Float32 ? complex(R(0.7), R(0.4)) :
                                      complex(R(9), R(0.1))
            delta = R === Float32 ? R(1e-3) : R(1e-4)
            local_moments = faddeeva_moments(local_z, 1)
            coefficient1 = one(R)
            coefficient0 = Complex{R}(
                (-local_moments[2] / local_moments[1]) *
                (one(R) - delta),
            )
            local_coefficients = Complex{R}[coefficient0, coefficient1]
            local_value = null_uniform_transition(
                one(R), local_z, local_coefficients,
            )
            wide_moment0 = _null_oracle_moment(local_z, 0; precision=1024)
            wide_moment1 = _null_oracle_moment(local_z, 1; precision=1024)
            local_reference = setprecision(BigFloat, 1024) do
                convert(
                    typeof(local_value),
                    Complex{BigFloat}(coefficient0) * wide_moment0 +
                    BigFloat(coefficient1) * wide_moment1,
                )
            end
            @test local_value == local_reference

            lambda = Complex{R}(
                (-local_moments[2] / local_moments[1]) *
                (one(R) - delta),
            )
            shifted_value = shifted_null_transition(
                one(R), local_z, lambda, 1, Complex{R}[one(R)],
            )
            shifted_reference = setprecision(BigFloat, 1024) do
                convert(
                    typeof(shifted_value),
                    Complex{BigFloat}(lambda) * wide_moment0 + wide_moment1,
                )
            end
            @test shifted_value == shifted_reference
        end

        for R in (Float32, Float64)
            coefficient = ForwardDiff.Dual(one(R), floatmax(R))
            C = typeof(complex(coefficient))
            local_z = complex(R(1), R(-2))
            value = null_uniform_transition(
                R(1e4), local_z, C[complex(zero(coefficient)),
                                    complex(coefficient)],
            )
            unit = faddeeva_moments(local_z, 1)[2] / R(100)
            @test ForwardDiff.value(real(value)) ≈ real(unit) rtol=8eps(R)
            @test ForwardDiff.value(imag(value)) ≈ imag(unit) rtol=8eps(R)
            @test ForwardDiff.partials(real(value), 1) ≈
                  floatmax(R) * real(unit) rtol=8eps(R)
            @test ForwardDiff.partials(imag(value), 1) ≈
                  floatmax(R) * imag(unit) rtol=8eps(R)
            @test_throws DomainError null_uniform_transition(
                one(R), local_z, C[complex(coefficient)]; switch=zero(R),
            )
            @test_throws DomainError shifted_null_transition(
                one(R), local_z, zero(R), 1, C[complex(coefficient)];
                max_terms=0,
            )

            differentiated_z = complex(coefficient, R(-2))
            balanced_value = null_uniform_transition(
                one(R), differentiated_z, Complex{R}[R(1e-4)],
            )
            unit_value = faddeeva_moments(local_z, 0)[1] * R(1e-4)
            unit_derivative = R(1e-4) * (
                -2local_z * faddeeva_moments(local_z, 0)[1] +
                2im / sqrt(R(pi))
            )
            @test ForwardDiff.value(real(balanced_value)) ≈
                  real(unit_value) rtol=8eps(R)
            @test ForwardDiff.value(imag(balanced_value)) ≈
                  imag(unit_value) rtol=8eps(R)
            @test ForwardDiff.partials(real(balanced_value), 1) ≈
                  floatmax(R) * real(unit_derivative) rtol=12eps(R)
            @test ForwardDiff.partials(imag(balanced_value), 1) ≈
                  floatmax(R) * imag(unit_derivative) rtol=12eps(R)
        end
        @test shifted_null_transition(
            1e-200, 0.7 + 0.4im, 1.0, 2, zeros(4),
        ) == 0.0im

        _null_moderate_allocation_probe()
        _null_large_allocation_probe()
        _null_hard_allocation_probe()
        _null_max_recurrence_allocation_probe()
        _null_max_asymptotic_allocation_probe()
        _null_transition_allocation_probe()
        _shifted_null_allocation_probe()
        @test @allocated(_null_moderate_allocation_probe()) <= 512
        @test @allocated(_null_large_allocation_probe()) <= 512
        @test @allocated(_null_hard_allocation_probe()) <= 700000
        @test @allocated(_null_max_recurrence_allocation_probe()) <= 2048
        @test @allocated(_null_max_asymptotic_allocation_probe()) <= 1280
        @test @allocated(_null_transition_allocation_probe()) <= 640
        @test @allocated(_shifted_null_allocation_probe()) <= 1024
    end
end
