using FastGaussQuadrature: gausslegendre
using ForwardDiff
using UTDKernels

include("support/shadow_sensitivity_oracle.jl")

@noinline _shadow_switch_allocation_probe() = shadow_switch(0.4 + 0.2im)
@noinline _shadow_kernel_allocation_probe() =
    shadow_sensitivity_kernel(0.4 + 0.2im)
@noinline _shadow_scaled_allocation_probe() =
    shadow_sensitivity_kernel(0.4, 64.0)
@noinline _shadow_multiplier_allocation_probe() =
    shadow_sensitivity_multiplier(2.0, 64.0)
@noinline _shadow_truncated_allocation_probe() =
    shadow_sensitivity_multiplier(2.0, 64.0; terms=4)
@noinline _shadow_pullback_allocation_probe() = shadow_sensitivity_pullback(
    1.0, 0.2, -2.0, 0.3, -12.0, 1.0, 0.6, 0.0, 0.0, 64.0,
)
@noinline _shadow_bound_allocation_probe() =
    UTDKernels._shadow_sensitivity_remainder_bound(4.0, 64.0, 3, 1.2)
@noinline _shadow_halfplane_allocation_probe() =
    UTDKernels._shadow_halfplane_second(0.4, 64.0)

@testset "Distributional shadow-boundary sensitivity" begin
    @testset "domains, types, and array paths" begin
        @test_throws DomainError shadow_switch(Inf)
        @test_throws DomainError shadow_switch(NaN + 0im)
        @test_throws ArgumentError shadow_switch(big"0.2")
        @test isfinite(shadow_sensitivity_kernel(1e155))
        @test_throws DomainError shadow_sensitivity_kernel(0.0, 0.0)
        @test_throws DomainError shadow_sensitivity_kernel(Inf, 1.0)
        @test_throws DomainError shadow_sensitivity_multiplier(0.0, -1.0)
        @test isfinite(shadow_sensitivity_multiplier(1e155, 1.0))
        @test_throws DomainError shadow_sensitivity_multiplier(
            1.0, 2.0; terms=0,
        )
        @test_throws DomainError shadow_sensitivity_multiplier(
            1.0, 2.0; terms=257,
        )
        @test_throws DomainError UTDKernels._shadow_sensitivity_remainder_bound(
            1.0, 1.0, 0, 1.0,
        )
        @test_throws DomainError UTDKernels._shadow_sensitivity_remainder_bound(
            -1.0, 1.0, 1, 1.0,
        )
        @test_throws DomainError UTDKernels._shadow_sensitivity_remainder_bound(
            1e200, nextfloat(0.0), 256, 1e200,
        )
        @test_throws DomainError shadow_sensitivity_pullback(
            1, 0, 0, 0, 0, 0.0, 0, 0, 0, 1.0,
        )
        @test_throws DomainError shadow_sensitivity_pullback(
            1, 0, 0, 0, 0, 1.0, NaN, 0, 0, 1.0,
        )
        nested_nonfinite = ForwardDiff.Dual(
            ForwardDiff.Dual(1.0, Inf),
            ForwardDiff.Dual(0.0, 0.0),
        )
        @test !UTDKernels._number_isfinite(nested_nonfinite)
        @test_throws DomainError shadow_sensitivity_pullback(
            nested_nonfinite, 0, 0, 0, 0, 1.0, 0, 0, 0, 1.0,
        )

        coordinates = range(-2.0, 2.0; length=17)
        @test shadow_switch(coordinates) == shadow_switch.(coordinates)
        @test shadow_sensitivity_kernel(coordinates) ==
              shadow_sensitivity_kernel.(coordinates)
        @test shadow_sensitivity_kernel(coordinates, 8.0) ==
              shadow_sensitivity_kernel.(coordinates, Ref(8.0))
        @test shadow_sensitivity_multiplier(coordinates, 8.0) ==
              shadow_sensitivity_multiplier.(coordinates, Ref(8.0))
        @test shadow_switch(Float32(0.4)) isa ComplexF32
        @test shadow_switch(Float16(0.4)) isa ComplexF32
        @test shadow_sensitivity_kernel(Float32(0.4)) isa ComplexF32
        @test shadow_sensitivity_kernel(Float32(0.4), Float32(8)) isa ComplexF32
        @test shadow_sensitivity_multiplier(Float32(0.4), Float32(8)) isa
              ComplexF32
        low_precision_kernel = shadow_sensitivity_kernel(
            Float32(0.5003617), Float32(14.3032675),
        )
        @test _shadow_relative_error(
            ComplexF64(low_precision_kernel),
            shadow_sensitivity_kernel(0.5003616809844971, 14.303267478942871),
        ) < 2e-7
        @test shadow_sensitivity_pullback(
            Float32(1), Float32(0.2), Float32(-2), Float32(0.3),
            Float32(-12), Float32(1), Float32(0.6), Float32(0), Float32(0),
            Float32(64),
        ).second isa ComplexF32
        @test_throws ArgumentError shadow_sensitivity_pullback(
            big"1", 0, 0, 0, 0, 1, 0, 0, 0, 1,
        )
        for values in (Real[], BigFloat[])
            @test_throws ArgumentError shadow_switch(values)
            @test_throws ArgumentError shadow_sensitivity_kernel(values)
            @test_throws ArgumentError shadow_sensitivity_kernel(values, 8.0)
            @test_throws ArgumentError shadow_sensitivity_multiplier(values, 8.0)
            @test_throws ArgumentError UTDKernels._shadow_halfplane_switch(
                values, 8.0,
            )
            @test_throws ArgumentError UTDKernels._shadow_halfplane_first(
                values, 8.0,
            )
            @test_throws ArgumentError UTDKernels._shadow_halfplane_second(
                values, 8.0,
            )
        end
    end

    @testset "canonical switch, derivative, and symmetry" begin
        step = 1e-6
        for q in (-2.1, -0.3, 0.0, 0.7, 2.4)
            numerical = (shadow_switch(q + step) - shadow_switch(q - step)) /
                        (2step)
            @test abs(numerical - shadow_sensitivity_kernel(q)) < 3e-9
            @test shadow_sensitivity_kernel(q) ≈
                  shadow_sensitivity_kernel(-q) atol=2e-14
            @test shadow_switch(q) + shadow_switch(-q) ≈ 1 atol=3e-15
        end
        @test shadow_switch(0.0) == 0.5 + 0.0im
        @test shadow_sensitivity_kernel(0.0) == cispi(0.25) / sqrt(pi)
        wrong_phase = cispi(-0.25) / sqrt(pi)
        @test abs(shadow_sensitivity_kernel(0.0) - wrong_phase) > 0.7
        for q in (-3.0, 0.0, 2.4)
            reference = ComplexF64(_shadow_oracle_switch(q; precision=192))
            @test abs(shadow_switch(q) - reference) < 3e-15
        end
        for q in (-1.4, -0.2, 0.6, 1.7)
            derivative = ForwardDiff.derivative(q) do coordinate
                real(shadow_switch(coordinate))
            end
            @test derivative ≈ real(shadow_sensitivity_kernel(q)) rtol=2e-12
        end

        for q in (Float64(2^27 + 1), 1e12, 1e155)
            phase_reference = ComplexF64(_shadow_oracle_phase(q, -1))
            kernel_reference = cispi(0.25) * phase_reference / sqrt(pi)
            @test _shadow_relative_error(
                shadow_sensitivity_kernel(q), kernel_reference,
            ) < 3e-15
            tail_reference = ComplexF64(_shadow_oracle_switch_tail(q))
            @test _shadow_relative_error(
                shadow_switch(-q), tail_reference,
            ) < 4e-15
            if q <= 1e12
                @test abs(
                    shadow_switch(q) - (1 - tail_reference),
                ) < 3e-16
            end
        end
        for R in (Float32, Float64)
            tiny = nextfloat(zero(R))
            for real_q in (R(-1e10), R(1e10)), sign in (-1, 1)
                local_q = complex(real_q, sign * tiny)
                reference = _shadow_oracle_near_real_switch(local_q)
                @test _shadow_relative_error(
                    shadow_switch(local_q), reference; floor=floatmin(R),
                ) <= 4eps(R)
            end

            local_q = complex(R(-1e10), tiny)
            differentiated_q = complex(
                R(-1e10), ForwardDiff.Dual(tiny, one(R)),
            )
            differentiated = @inferred shadow_switch(differentiated_q)
            local_derivative = complex(
                ForwardDiff.partials(real(differentiated), 1),
                ForwardDiff.partials(imag(differentiated), 1),
            )
            derivative_reference =
                _shadow_oracle_switch_imaginary_derivative(local_q)
            @test _shadow_relative_error(
                local_derivative, derivative_reference; floor=floatmin(R),
            ) <= 8eps(R)
        end
        for sign in (-1, 1)
            local_q = complex(-1e4, sign * nextfloat(0.0))
            @test _shadow_relative_error(
                shadow_switch(local_q),
                _shadow_oracle_near_real_switch(local_q),
            ) <= 4eps(Float64)
        end
        for R in (Float32, Float64), q in R.((-10, 10, -1e10, 1e10))
            tangent = floatmax(R)
            differentiated = @inferred shadow_switch(
                ForwardDiff.Dual(q, tangent),
            )
            local_derivative = complex(
                ForwardDiff.partials(real(differentiated), 1),
                ForwardDiff.partials(imag(differentiated), 1),
            )
            derivative_reference = _shadow_oracle_switch_derivative(
                complex(q, zero(R)), complex(tangent, zero(R)),
            )
            @test _shadow_relative_error(
                local_derivative, derivative_reference; floor=floatmin(R),
            ) <= 2eps(R)
        end
        for q in (1.0 - 2.0im, 1.0 - 5.0im), component in (:real, :imag)
            tangent = floatmax(Float64)
            differentiated_q = component === :real ?
                complex(ForwardDiff.Dual(real(q), tangent), imag(q)) :
                complex(real(q), ForwardDiff.Dual(imag(q), tangent))
            differentiated = @inferred shadow_switch(differentiated_q)
            local_derivative = complex(
                ForwardDiff.partials(real(differentiated), 1),
                ForwardDiff.partials(imag(differentiated), 1),
            )
            tangent_direction = component === :real ? 1.0 + 0.0im :
                                                        0.0 + 1.0im
            derivative_reference = _shadow_oracle_switch_derivative(
                q, tangent * tangent_direction,
            )
            @test _shadow_relative_error(
                local_derivative, derivative_reference,
            ) <= 2eps(Float64)
        end
        kernel_coordinate = 0.55
        kernel_tangent = floatmax(Float64)
        differentiated_kernel = @inferred shadow_sensitivity_kernel(
            ForwardDiff.Dual(kernel_coordinate, kernel_tangent),
        )
        kernel_derivative = complex(
            ForwardDiff.partials(real(differentiated_kernel), 1),
            ForwardDiff.partials(imag(differentiated_kernel), 1),
        )
        @test _shadow_relative_error(
            kernel_derivative,
            _shadow_oracle_kernel_derivative(
                kernel_coordinate, kernel_tangent,
            ),
        ) <= 2eps(Float64)
        q = 1e12
        derivative = ForwardDiff.derivative(
            value -> real(shadow_sensitivity_kernel(value)), q,
        )
        kernel = shadow_sensitivity_kernel(q)
        @test derivative ≈ real(-2im * q * kernel) rtol=3e-15
        second_derivative = ForwardDiff.derivative(q) do outer
            ForwardDiff.derivative(
                inner -> real(shadow_sensitivity_kernel(inner)), outer,
            )
        end
        @test second_derivative ≈
              real((-2im - 4q^2) * kernel) rtol=4e-15
    end

    @testset "scaled kernel and exact multiplier" begin
        for kappa in (4.0, 16.0, 64.0), s in (-1.1, -0.2, 0.0, 0.7)
            expected = sqrt(kappa) * shadow_sensitivity_kernel(sqrt(kappa) * s)
            @test shadow_sensitivity_kernel(s, kappa) == expected
        end
        frequencies = range(-10.0, 10.0; length=101)
        for kappa in (7.0, 32.0, 128.0)
            values = shadow_sensitivity_multiplier(frequencies, kappa)
            @test maximum(abs.(abs.(values) .- 1)) < 2e-15
            @test maximum(abs.(values - conj.(values))) > 0.1
        end
        small_frequencies = range(-2.0, 2.0; length=21)
        @test maximum(abs.(
            shadow_sensitivity_multiplier(small_frequencies, 100.0) .-
            shadow_sensitivity_multiplier(
                small_frequencies, 100.0; terms=4,
            ),
        )) < 1e-9
        xi, kappa = 1.7, 16.0
        @test abs(
            shadow_sensitivity_multiplier(xi, kappa) -
            exp(-im * xi^2 / (4kappa)),
        ) > 0.08
        derivative = ForwardDiff.derivative(xi) do frequency
            imag(shadow_sensitivity_multiplier(frequency, kappa))
        end
        expected = imag(
            im * xi / (2kappa) * shadow_sensitivity_multiplier(xi, kappa),
        )
        @test derivative ≈ expected rtol=2e-12 atol=2e-14
        exact_tangent = floatmax(Float64) / 4
        exact_frequency = 1e-100
        exact_parameter = 0.01
        differentiated_exact = @inferred shadow_sensitivity_multiplier(
            ForwardDiff.Dual(exact_frequency, exact_tangent), exact_parameter,
        )
        exact_derivative = complex(
            ForwardDiff.partials(real(differentiated_exact), 1),
            ForwardDiff.partials(imag(differentiated_exact), 1),
        )
        @test exact_derivative == _shadow_oracle_multiplier_derivative(
            exact_frequency, exact_parameter, exact_tangent,
        )
        scaled_coordinate = 2.4
        scaled_parameter = 0.25
        scaled_tangent = floatmax(Float64)
        differentiated_scaled = @inferred shadow_sensitivity_kernel(
            ForwardDiff.Dual(scaled_coordinate, scaled_tangent),
            scaled_parameter,
        )
        scaled_derivative = complex(
            ForwardDiff.partials(real(differentiated_scaled), 1),
            ForwardDiff.partials(imag(differentiated_scaled), 1),
        )
        @test _shadow_relative_error(
            scaled_derivative,
            _shadow_oracle_scaled_kernel_derivative(
                scaled_coordinate, scaled_parameter, scaled_tangent,
            ),
        ) <= 2eps(Float64)
        large_scaled_coordinate = 6000.0
        large_scaled_parameter = 1e-4
        differentiated_large_scaled = @inferred shadow_sensitivity_kernel(
            ForwardDiff.Dual(large_scaled_coordinate, scaled_tangent),
            large_scaled_parameter,
        )
        large_scaled_derivative = complex(
            ForwardDiff.partials(real(differentiated_large_scaled), 1),
            ForwardDiff.partials(imag(differentiated_large_scaled), 1),
        )
        @test _shadow_relative_error(
            large_scaled_derivative,
            _shadow_oracle_scaled_kernel_derivative(
                large_scaled_coordinate, large_scaled_parameter,
                scaled_tangent,
            ),
        ) <= 2eps(Float64)
        truncated_tangent = floatmax(Float64) / 2
        differentiated_truncated = @inferred shadow_sensitivity_multiplier(
            ForwardDiff.Dual(16.0, truncated_tangent), 8.0; terms=30,
        )
        truncated_derivative = complex(
            ForwardDiff.partials(real(differentiated_truncated), 1),
            ForwardDiff.partials(imag(differentiated_truncated), 1),
        )
        @test truncated_derivative ==
              _shadow_oracle_truncated_multiplier_derivative(
                  16.0, 8.0, 30, truncated_tangent,
              )
        for T in (Float32, Float64)
            second_multiplier = ForwardDiff.derivative(zero(T)) do outer
                ForwardDiff.derivative(outer) do inner
                    imag(shadow_sensitivity_multiplier(inner, one(T)))
                end
            end
            @test second_multiplier == T(0.5)
        end

        for (frequency, parameter) in (
            (Float64(2^28 + 2), 4.0),
            (1e12, 1.0),
            (1.0f10, 3.0f0),
        )
            reference = ComplexF64(
                _shadow_oracle_multiplier(frequency, parameter),
            )
            @test _shadow_relative_error(
                ComplexF64(shadow_sensitivity_multiplier(frequency, parameter)),
                reference,
            ) < (frequency isa Float32 ? 3e-7 : 3e-15)
        end
        scaled_reference = setprecision(BigFloat, 512) do
            coordinate = BigFloat(1.0f10)
            parameter = BigFloat(3.0f0)
            ComplexF64(
                sqrt(parameter) * cis(BigFloat(pi) / 4) *
                exp(-im * parameter * coordinate^2) / sqrt(BigFloat(pi)),
            )
        end
        @test _shadow_relative_error(
            ComplexF64(shadow_sensitivity_kernel(1.0f10, 3.0f0)),
            scaled_reference,
        ) < 3e-7
        for parameter in (2.0, 4.0, 16.0)
            value = shadow_sensitivity_kernel(floatmax(Float64), parameter)
            reference = ComplexF64(_shadow_oracle_scaled_kernel(
                floatmax(Float64), parameter,
            ))
            @test isfinite(value)
            @test _shadow_relative_error(value, reference) < 3e-15
        end

        frequency = 2sqrt(50.0)
        truncated = shadow_sensitivity_multiplier(
            frequency, 1.0; terms=256,
        )
        truncated_reference = ComplexF64(
            _shadow_oracle_truncated_multiplier(frequency, 1.0, 256),
        )
        @test _shadow_relative_error(truncated, truncated_reference) < 3e-15
        truncated32 = shadow_sensitivity_multiplier(
            Float32(frequency), 1.0f0; terms=256,
        )
        truncated32_reference = ComplexF32(
            _shadow_oracle_truncated_multiplier(Float32(frequency), 1.0f0, 256),
        )
        @test _shadow_relative_error(
            truncated32, truncated32_reference,
        ) < 3f-6
        for (phase, terms) in ((4097.0, 2), (5000.0, 3), (5000.0, 10))
            large_frequency = 2sqrt(phase)
            value = shadow_sensitivity_multiplier(
                large_frequency, 1.0; terms,
            )
            reference = ComplexF64(_shadow_oracle_truncated_multiplier(
                large_frequency, 1.0, terms,
            ))
            @test isfinite(value)
            @test _shadow_relative_error(value, reference) < 3e-14
        end
        large_frequency32 = 2f0 * sqrt(4097f0)
        value32 = shadow_sensitivity_multiplier(
            large_frequency32, 1f0; terms=2,
        )
        reference32 = ComplexF32(_shadow_oracle_truncated_multiplier(
            large_frequency32, 1f0, 2,
        ))
        @test isfinite(value32)
        @test _shadow_relative_error(value32, reference32) < 3f-6
        @test_throws DomainError shadow_sensitivity_multiplier(
            2.0f20, 1.0f0; terms=2,
        )
        @test_throws DomainError shadow_sensitivity_multiplier(
            floatmax(Float16), Float16(1); terms=6,
        )
    end

    @testset "half-plane coordinate and derivatives" begin
        for kappa in (4.0, 16.0, 64.0)
            @test UTDKernels._shadow_halfplane_coordinate(0.0, kappa) == 0.0
            @test UTDKernels._shadow_halfplane_coordinate_derivative(
                0.0, kappa,
            ) == sqrt(kappa / 2)
            @test UTDKernels._shadow_halfplane_coordinate_second(0.0, kappa) == 0.0
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
        for (kappa, s, first, second) in fixtures
            @test _shadow_relative_error(
                UTDKernels._shadow_halfplane_first(s, kappa), first,
            ) < 2e-13
            @test _shadow_relative_error(
                UTDKernels._shadow_halfplane_second(s, kappa), second,
            ) < 2e-13
            reference_first, reference_second =
                _shadow_oracle_halfplane_derivatives(
                    s, kappa; precision=192, step=1e-5,
                )
            @test _shadow_relative_error(ComplexF64(reference_first), first) < 2e-12
            @test _shadow_relative_error(ComplexF64(reference_second), second) < 2e-12
        end
    end

    @testset "nonlinear pullback coefficients" begin
        gamma = 0.6
        f(s) = 1 + 0.35s - 0.8s^2 + 0.2s^3 - 0.07s^4
        inverse(y) = iszero(y) ? zero(y) : 2y / (1 + sqrt(1 + 2gamma * y))
        derivatives = ntuple(
            order -> _shadow_nested_derivative(f, 0.0, order - 1), 5,
        )
        composed(y) = f(inverse(y))
        composed_second = _shadow_nested_derivative(composed, 0.0, 2)
        composed_fourth = _shadow_nested_derivative(composed, 0.0, 4)
        kappa = 64.0
        values = shadow_sensitivity_pullback(
            derivatives..., 1.0, gamma, 0.0, 0.0, kappa,
        )
        @test values.leading == complex(derivatives[1])
        @test values.first ≈ derivatives[1] - im * composed_second / (4kappa)
        @test values.second ≈ values.first - composed_fourth / (32kappa^2)
        width_only = derivatives[1] - im * derivatives[3] / (4kappa)
        @test abs(values.first - width_only) > 1e-6

        invariant = shadow_sensitivity_pullback(
            1, 0, 0, 0, 1, 1.0, 0, 0, 0, 1.0,
        )
        large_scale = shadow_sensitivity_pullback(
            1, 0, 0, 0, 1, 1e100, 0, 0, 0, 1e-200,
        )
        @test large_scale == invariant
        reverse_scale = shadow_sensitivity_pullback(
            1, 0, 1, 0, 0, 1e-100, 0, 0, 0, 1e200,
        )
        @test reverse_scale.first == 1 - 0.25im
        @test reverse_scale.second == reverse_scale.first
        for (R, derivative, parameter) in (
            (Float32, 13.976485f0, 0.089521356f0),
            (Float64, 43.93998554077454, 0.11918598386376307),
        )
            first_cancel_f2 =
                -4im * parameter * derivative^2
            first_cancel = shadow_sensitivity_pullback(
                one(R), zero(R), first_cancel_f2, zero(R), zero(R),
                derivative, zero(R), zero(R), zero(R), parameter,
            )
            first_reference = _shadow_oracle_pullback(
                one(R), zero(R), first_cancel_f2, zero(R), zero(R),
                derivative, zero(R), zero(R), zero(R), parameter,
            )
            @test first_cancel.first == Complex{R}(first_reference.first)
        end
        for (R, derivative, parameter, second_cancel_f4) in (
            (Float32, 6.9680367f0, 4.1813226f0, 1.3189218f6),
            (Float64, 0.13611800187405707, 0.06617243353104287,
             4.810236753359424e-5),
        )
            second_cancel = shadow_sensitivity_pullback(
                one(R), zero(R), zero(R), zero(R), second_cancel_f4,
                derivative, zero(R), zero(R), zero(R), parameter,
            )
            second_reference = _shadow_oracle_pullback(
                one(R), zero(R), zero(R), zero(R), second_cancel_f4,
                derivative, zero(R), zero(R), zero(R), parameter,
            )
            @test second_cancel.second == Complex{R}(second_reference.second)
        end
        for (R, parameter) in ((Float32, 1f-15), (Float64, 1e-150))
            f0 = zero(R)
            f1 = one(R)
            f2 = R(1 / 3)
            derivative = R(3)
            second_map = one(R)
            value = shadow_sensitivity_pullback(
                f0, f1, f2, zero(R), zero(R), derivative, second_map,
                zero(R), zero(R), parameter,
            ).first
            reference = setprecision(BigFloat, 512) do
                bracket = BigFloat(f2) -
                          BigFloat(second_map) / BigFloat(derivative) * BigFloat(f1)
                convert(
                    typeof(value),
                    -im * bracket /
                    (4BigFloat(parameter) * BigFloat(derivative)^2),
                )
            end
            @test value == reference
        end
        for R in (Float32, Float64)
            probe = ForwardDiff.Dual(one(R), floatmax(R))
            value = shadow_sensitivity_pullback(
                zero(R), probe, zero(R), zero(R), zero(R), one(R), R(100),
                zero(R), zero(R), R(1e4),
            )
            @test ForwardDiff.value(real(value.first)) == zero(R)
            @test ForwardDiff.value(imag(value.first)) == R(1 / 400)
            @test ForwardDiff.partials(real(value.first), 1) == zero(R)
            @test ForwardDiff.partials(imag(value.first), 1) ≈
                  floatmax(R) / R(400) rtol=8eps(R)
            @test ForwardDiff.value(real(value.second)) == R(0.0046875)
            @test ForwardDiff.value(imag(value.second)) == R(1 / 400)
            @test ForwardDiff.partials(real(value.second), 1) ≈
                  R(0.0046875) * floatmax(R) rtol=8eps(R)
            @test ForwardDiff.partials(imag(value.second), 1) ≈
                  floatmax(R) / R(400) rtol=8eps(R)
        end
    end

    @testset "finite-bandwidth bound and distributional orders" begin
        omega = 3.0
        nodes, gauss_weights = gausslegendre(256)
        frequencies = omega .* nodes
        weights = omega .* gauss_weights
        spectrum = max.(0.0, 1 .- (frequencies ./ omega).^2).^4 .*
                   (1 .+ 0.2im .* sin.(pi .* frequencies ./ omega))
        reflected = reverse(spectrum)
        spectral_l1 = sum(weights .* abs.(spectrum)) / (2pi)
        kappas = (16.0, 32.0, 64.0, 128.0, 256.0, 512.0)
        errors = [Float64[] for _ in 1:3]
        for kappa in kappas
            exact = sum(
                reflected .* weights .*
                shadow_sensitivity_multiplier(frequencies, kappa),
            ) / (2pi)
            for retained_terms in 1:3
                approximation = sum(
                    reflected .* weights .*
                    shadow_sensitivity_multiplier(
                        frequencies, kappa; terms=retained_terms,
                    ),
                ) / (2pi)
                error = abs(exact - approximation)
                bound = UTDKernels._shadow_sensitivity_remainder_bound(
                    omega, kappa, retained_terms, spectral_l1,
                )
                @test error <= bound * (1 + 2e-14)
                push!(errors[retained_terms], error)
            end
        end
        @test _shadow_fit_slope(kappas[end-3:end], errors[1][end-3:end]) ≈
              -1 atol=0.03
        @test _shadow_fit_slope(kappas[end-3:end], errors[2][end-3:end]) ≈
              -2 atol=0.03
        @test _shadow_fit_slope(kappas[end-3:end], errors[3][end-3:end]) ≈
              -3 atol=0.04
        @test UTDKernels._shadow_sensitivity_remainder_bound(
            0.0, 2.0, 3, 1.0,
        ) == 0.0
        @test UTDKernels._shadow_sensitivity_remainder_bound(
            2.0, 2.0, 3, 0.0,
        ) == 0.0
        for T in (Float32, Float64)
            mass_derivative = ForwardDiff.derivative(zero(T)) do mass
                UTDKernels._shadow_sensitivity_remainder_bound(
                    T(2), one(T), 3, mass,
                )
            end
            @test mass_derivative == T(1 / 6)
            bandwidth_second = ForwardDiff.derivative(zero(T)) do outer
                ForwardDiff.derivative(outer) do bandwidth
                    UTDKernels._shadow_sensitivity_remainder_bound(
                        bandwidth, one(T), 1, one(T),
                    )
                end
            end
            @test bandwidth_second == T(0.5)
            tiny_bandwidth = nextfloat(zero(T))
            tangent = floatmax(T)
            differentiated = @inferred UTDKernels._shadow_sensitivity_remainder_bound(
                ForwardDiff.Dual(tiny_bandwidth, tangent), one(T), 1, one(T),
            )
            @test UTDKernels._number_isfinite(differentiated)
            @test ForwardDiff.partials(differentiated, 1) ==
                  _shadow_oracle_remainder_derivative(
                      tiny_bandwidth, one(T), 1, one(T), tangent,
                  )
        end
        @test UTDKernels._shadow_sensitivity_remainder_bound(
            Float32(2), Float32(2), 3, 1.0,
        ) isa Float64
        @test UTDKernels._shadow_sensitivity_remainder_bound(
            1e-200, 1e200, 256, 1e-200,
        ) == nextfloat(0.0)
        outward_bound = UTDKernels._shadow_sensitivity_remainder_bound(
            0.0002, 0.3, 30, 1.0,
        )
        exact_bound = setprecision(BigFloat, 2048) do
            omega = BigFloat(0.0002)
            parameter = BigFloat(0.3)
            (omega^2 / (4parameter))^30 / factorial(big(30))
        end
        @test BigFloat(outward_bound) >= exact_bound

        @test_throws DomainError shadow_sensitivity_kernel(Float64[], 0.0)
        @test_throws DomainError shadow_sensitivity_multiplier(Float64[], -1.0)
        @test_throws DomainError shadow_sensitivity_multiplier(
            Float64[], 1.0; terms=0,
        )
        @test_throws DomainError UTDKernels._shadow_halfplane_switch(
            Float64[], 0.0,
        )
        tiny_coordinate = UTDKernels._shadow_halfplane_coordinate(
            nextfloat(0.0), floatmax(Float64),
        )
        tiny_reference = setprecision(BigFloat, 512) do
            Float64(
                sqrt(2BigFloat(floatmax(Float64))) *
                sin(BigFloat(nextfloat(0.0)) / 2),
            )
        end
        @test tiny_coordinate == tiny_reference
        @test isfinite(UTDKernels._shadow_halfplane_second(
            nextfloat(0.0), floatmax(Float64),
        ))

        xi, kappa = 1.7, 1e6
        utd = (shadow_sensitivity_multiplier(xi, kappa) - 1) * kappa
        gaussian = (exp(-xi^2 / (4kappa)) - 1) * kappa
        @test abs(imag(utd)) > 1e-3
        @test iszero(imag(gaussian))
        @test abs(utd - gaussian) > 1e-3
    end

    @testset "warmed allocations" begin
        _shadow_switch_allocation_probe()
        _shadow_kernel_allocation_probe()
        _shadow_scaled_allocation_probe()
        _shadow_multiplier_allocation_probe()
        _shadow_truncated_allocation_probe()
        _shadow_pullback_allocation_probe()
        _shadow_bound_allocation_probe()
        _shadow_halfplane_allocation_probe()
        @test @allocated(_shadow_switch_allocation_probe()) == 0
        @test @allocated(_shadow_kernel_allocation_probe()) == 0
        @test @allocated(_shadow_scaled_allocation_probe()) == 0
        @test @allocated(_shadow_multiplier_allocation_probe()) == 0
        @test @allocated(_shadow_truncated_allocation_probe()) == 0
        @test @allocated(_shadow_pullback_allocation_probe()) == 0
        @test @allocated(_shadow_bound_allocation_probe()) == 0
        @test @allocated(_shadow_halfplane_allocation_probe()) == 0
    end
end
