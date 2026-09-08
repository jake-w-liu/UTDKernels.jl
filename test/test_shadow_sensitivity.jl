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
        @test_throws DomainError shadow_sensitivity_kernel(1e155)
        @test_throws DomainError shadow_sensitivity_kernel(0.0, 0.0)
        @test_throws DomainError shadow_sensitivity_kernel(Inf, 1.0)
        @test_throws DomainError shadow_sensitivity_multiplier(0.0, -1.0)
        @test_throws DomainError shadow_sensitivity_multiplier(1e155, 1.0)
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
        @test UTDKernels._shadow_sensitivity_remainder_bound(
            Float32(2), Float32(2), 3, 1.0,
        ) isa Float64
        @test UTDKernels._shadow_sensitivity_remainder_bound(
            1e-200, 1e200, 256, 1e-200,
        ) == nextfloat(0.0)

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
