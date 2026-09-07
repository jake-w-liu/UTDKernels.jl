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

        derivative = ForwardDiff.derivative(0.7) do real_part
            real(faddeeva_moments(complex(real_part, 0.4), 2)[3])
        end
        step = 1e-6
        centered = (
            real(faddeeva_moments(0.7 + step + 0.4im, 2)[3]) -
            real(faddeeva_moments(0.7 - step + 0.4im, 2)[3])
        ) / (2step)
        @test derivative ≈ centered rtol=2e-8 atol=2e-10

        @test_throws ArgumentError faddeeva_moments(complex(big"0.7", big"0.4"), 2)
        @test_throws DomainError faddeeva_moments(NaN + 0im, 2)
        @test_throws DomainError faddeeva_moments(1.0 + im, -1)
        @test_throws DomainError faddeeva_moments(1.0 + im, 65)
        @test_throws DomainError faddeeva_moments(1.0 + im, 2; switch=0.0)
        @test_throws DomainError faddeeva_moments(1.0 + im, 2; rtol=0.0)
        @test_throws DomainError faddeeva_moments(1.0 + im, 2; max_terms=0)
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
