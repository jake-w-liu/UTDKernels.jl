using ForwardDiff
using QuadGK
using Random
using SpecialFunctions: besselj

curvature_relative_error(value, reference; floor=1e-30) =
    abs(value - reference) / max(abs(reference), floor)

function curvature_alias_oracle(N, m, z, phi; alias_count=20)
    total = 0.0 + 0.0im
    for alias_index in -alias_count:alias_count
        order = alias_index * N - m
        total += (-1.0im)^order * besselj(order, z) * exp(-im * order * phi)
    end
    return 2pi * total
end

function curvature_uniform_data(N, m, z, phi)
    turning = fill(2pi / N, N)
    amplitudes = Vector{ComplexF64}(undef, N)
    for index in 0:(N - 1)
        theta = 2pi * index / N
        amplitudes[index + 1] = exp(im * m * theta - im * z * cos(theta - phi))
    end
    return turning, amplitudes
end

function curvature_fit_slope(x, y)
    lx = log.(Float64.(x))
    ly = log.(Float64.(y))
    mx = sum(lx) / length(lx)
    my = sum(ly) / length(ly)
    return sum((lx .- mx) .* (ly .- my)) / sum((lx .- mx).^2)
end

const CURVATURE_ALLOC_TURNING = fill(2pi / 32, 32)
const CURVATURE_ALLOC_AMPLITUDES = ComplexF64[
    exp(im * 2pi * index / 32) for index in 0:31
]

@noinline _curvature_coefficient_probe() =
    intrinsic_seam_coefficient(0.01, 25.0, 0.7)

@noinline _curvature_raw_sum_probe() =
    curvature_measure_sum(CURVATURE_ALLOC_TURNING, CURVATURE_ALLOC_AMPLITUDES)

@noinline _curvature_debiased_sum_probe() =
    debiased_curvature_measure_sum(CURVATURE_ALLOC_TURNING, CURVATURE_ALLOC_AMPLITUDES)

@testset "Curvature-measure seam limits" begin
    @testset "intrinsic coefficient and linear turning limit" begin
        k, L = 25.0, 0.7
        linear = UTDKernels._intrinsic_seam_linear_prefactor(k, L)
        turning_angles = exp10.(range(-7, -3; length=20))
        errors = map(turning_angles) do turning
            abs(intrinsic_seam_coefficient(turning, k, L) / (linear * turning) - 1)
        end
        @test abs(curvature_fit_slope(turning_angles[1:12], errors[1:12]) - 1) < 0.01

        for turning in (1e-12, 1e-16, 1e-18)
            exact = UTDKernels._curvature_local_bias(turning)
            series = UTDKernels._curvature_local_bias_series(turning)
            @test exact ≈ series rtol=6e-16
            @test intrinsic_seam_coefficient(turning, k, L) / (linear * turning) ≈
                  series rtol=8e-16
        end

        turning = 0.05
        epsilon = turning / pi
        wedge = Wedge(pi + turning)
        phase162 = pec_wedge_face_edge(wedge, 0.0, k, L, epsilon).edge
        @test intrinsic_seam_coefficient(turning, k, L) == phase162

        # The inclusive turning endpoint belongs to the angle's input type,
        # even when k or L promotes the calculation to a wider type.
        for (turning_endpoint, mixed_k) in (
            (Float16(pi), 1.0f0),
            (Float32(pi), 1.0),
        )
            mixed = intrinsic_seam_coefficient(turning_endpoint, mixed_k, mixed_k)
            reference_type = typeof(real(mixed))
            reference = intrinsic_seam_coefficient(
                reference_type(pi), reference_type(mixed_k), reference_type(mixed_k),
            )
            @test mixed ≈ reference rtol=32eps(reference_type) atol=0
        end
        @test_throws DomainError intrinsic_seam_coefficient(
            nextfloat(Float32(pi)), 1.0, 1.0,
        )
    end

    @testset "local bias closed form and series" begin
        for turning in (1e-5, 1e-3, 0.05, 0.2)
            n = 1 + turning / pi
            independent = 2tan(turning / (2n)) / (n * turning)
            @test UTDKernels._curvature_local_bias(turning) ≈ independent rtol=5e-15
        end
        @test UTDKernels._curvature_local_bias(2pi / 512) ≈
              0.9922453949516098 atol=5e-16 rtol=0
        for turning in (1e-4, 3e-4, 1e-3)
            @test abs(
                UTDKernels._curvature_local_bias(turning) -
                UTDKernels._curvature_local_bias_series(turning)
            ) < 2e-9
        end
    end

    @testset "circular harmonic and Bessel alias oracle" begin
        @test besselj(3, 24.0) ≈ 0.16127035997227668 atol=5e-15 rtol=0
        harmonic = curvature_continuum_harmonic(3, 24.0, 0.37)
        @test harmonic ≈ -0.9076039151540007 + 0.4505717603087463im atol=2e-14 rtol=0

        integrand(theta) = exp(im * 3theta - im * 7.3cos(theta - 0.41))
        real_part, _ = quadgk(theta -> real(integrand(theta)), 0, 2pi; atol=1e-12)
        imaginary_part, _ = quadgk(theta -> imag(integrand(theta)), 0, 2pi; atol=1e-12)
        @test real_part + im * imaginary_part ≈
              curvature_continuum_harmonic(3, 7.3, 0.41) atol=2e-11 rtol=0

        # Reduce the periodic phase before multiplying by the harmonic order;
        # a finite angle must not overflow merely because its unreduced
        # product with m is outside the floating-point exponent range.
        huge_phi = 1e308
        huge_phase = curvature_continuum_harmonic(2, 1.0, huge_phi)
        huge_reference = setprecision(BigFloat, 2048) do
            z_big = BigFloat(1.0)
            phi_big = BigFloat(huge_phi)
            2BigFloat(pi) * (-complex(big"0", big"1"))^2 * besselj(2, z_big) *
            exp(complex(big"0", big"1") * 2 * phi_big)
        end
        @test huge_phase ≈ ComplexF64(huge_reference) atol=8e-15 rtol=0

        harmonic16 = curvature_continuum_harmonic(2, Float16(1), Float16(0.1))
        @test harmonic16 isa ComplexF32
        @test harmonic16 == curvature_continuum_harmonic(2, 1.0f0, Float32(Float16(0.1)))
        for order in (100_000, 1_000_000)
            z32 = Float32(order)
            phi32 = Float32(0.3)
            amplitude32 = 2f0 * Float32(pi) * besselj(Cint(order), z32)
            phase_reference = setprecision(BigFloat, 128) do
                quarter = (
                    complex(BigFloat(1), BigFloat(0)),
                    complex(BigFloat(0), -BigFloat(1)),
                    complex(-BigFloat(1), BigFloat(0)),
                    complex(BigFloat(0), BigFloat(1)),
                )[mod(order, 4) + 1]
                quarter * cis(BigFloat(order) * BigFloat(phi32))
            end
            expected32 = amplitude32 * ComplexF32(phase_reference)
            @test curvature_continuum_harmonic(order, z32, phi32) ≈
                  expected32 rtol=2eps(Float32) atol=0
        end
        @test curvature_continuum_harmonic(big(3), 1.0, 0.1) isa ComplexF64
        @test_throws ArgumentError curvature_continuum_harmonic(
            typemax(Int), 1.0, 0.1,
        )
        @test_throws ArgumentError curvature_continuum_harmonic(
            typemin(Int), 1.0, 0.1,
        )
        @test_throws ArgumentError curvature_continuum_harmonic(
            typemin(Cint), 1.0f0, 0.1f0,
        )
        @test_throws ArgumentError curvature_continuum_harmonic(
            typemin(Cint), 1.0, 0.1,
        )
        for boundary_order in (-typemax(Cint), typemax(Cint))
            @test_throws ArgumentError curvature_continuum_harmonic(
                boundary_order, 1.0, 0.1,
            )
            @test_throws ArgumentError ForwardDiff.derivative(1.0) do local_z
                real(curvature_continuum_harmonic(boundary_order, local_z, 0.1))
            end
        end
        @test_throws ArgumentError curvature_continuum_harmonic(
            big(typemax(Clong)) + 1, big"1", big"0.1",
        )
        for boundary_order in (big(typemin(Clong)), big(typemax(Clong)))
            @test_throws ArgumentError curvature_continuum_harmonic(
                boundary_order, big"1", big"0.1",
            )
        end

        for (z_precision, phi_precision) in ((64, 512), (512, 64), (32, 512))
            z_mixed = setprecision(BigFloat, z_precision) do
                BigFloat("1.234567890123456789")
            end
            phi_mixed = setprecision(BigFloat, phi_precision) do
                BigFloat("0.314159265358979323")
            end
            mixed_harmonic = curvature_continuum_harmonic(2, z_mixed, phi_mixed)
            reference_harmonic, reference_tolerance = setprecision(BigFloat, 512) do
                z_reference = BigFloat(z_mixed)
                phi_reference = BigFloat(phi_mixed)
                reference = 2BigFloat(pi) *
                            (-complex(BigFloat(0), BigFloat(1)))^2 *
                            besselj(2, z_reference) *
                            exp(complex(BigFloat(0), BigFloat(1)) * 2phi_reference)
                reference, 128eps(BigFloat)
            end
            @test precision(real(mixed_harmonic)) == 512
            @test mixed_harmonic ≈ reference_harmonic rtol=reference_tolerance atol=0
        end

        # A zero Bessel value can still have a nonzero derivative.
        root_derivative = ForwardDiff.derivative(0.0) do local_z
            real(curvature_continuum_harmonic(1, local_z, pi / 2))
        end
        @test root_derivative ≈ pi atol=2e-15 rtol=0

        rng = Xoshiro(0x2026_0165)
        for _ in 1:100
            N = rand(rng, 8:79)
            m = rand(rng, -5:5)
            z = 0.1 + rand(rng) * 34.9
            phi = rand(rng) * 2pi - pi
            turning, amplitudes = curvature_uniform_data(N, m, z, phi)
            direct = debiased_curvature_measure_sum(turning, amplitudes)
            @test direct ≈ curvature_alias_oracle(N, m, z, phi; alias_count=12) atol=2e-11 rtol=0
        end
    end

    @testset "raw/debiased decomposition and convergence" begin
        N, m, z, phi = 24, 2, 18.0, 0.3
        turning, amplitudes = curvature_uniform_data(N, m, z, phi)
        raw = curvature_measure_sum(turning, amplitudes)
        debiased = debiased_curvature_measure_sum(turning, amplitudes)
        continuum = curvature_continuum_harmonic(m, z, phi)
        beta = UTDKernels._curvature_local_bias(2pi / N)
        local_error = (beta - 1) * continuum
        alias_error = beta * (debiased - continuum)
        @test raw - continuum ≈ local_error + alias_error atol=2e-15 rtol=0

        Ns = (24, 32, 48, 64, 96, 128, 192)
        raw_errors = Float64[]
        debiased_errors = Float64[]
        for count in Ns
            local_turning, local_amplitudes = curvature_uniform_data(count, 3, 24.0, 0.37)
            reference = curvature_continuum_harmonic(3, 24.0, 0.37)
            push!(raw_errors, curvature_relative_error(
                curvature_measure_sum(local_turning, local_amplitudes), reference,
            ))
            push!(debiased_errors, curvature_relative_error(
                debiased_curvature_measure_sum(local_turning, local_amplitudes), reference,
            ))
        end
        @test curvature_fit_slope(Ns[end-4:end], raw_errors[end-4:end]) < -0.9
        @test maximum(debiased_errors[end-2:end]) < 1e-10
    end

    @testset "general supplied partitions and orientation" begin
        N = 48
        raw_weights = [
            1 + 0.35cos(2pi * (index + 0.5) / N) +
            0.12sin(6pi * (index + 0.5) / N) for index in 0:(N - 1)
        ]
        turning = 2pi .* raw_weights ./ sum(raw_weights)
        edges = cumsum(vcat(0.0, turning))
        theta = edges[1:end-1] .+ turning ./ 2
        amplitudes = @. (1 + 0.35cos(2theta) - 0.2sin(3theta) + 0.12cos(5theta)) *
                         exp(-im * 12cos(theta - 0.2))
        raw = curvature_measure_sum(turning, amplitudes)
        debiased = debiased_curvature_measure_sum(turning, amplitudes)
        @test isfinite(raw)
        @test isfinite(debiased)
        @test sum(turning) ≈ 2pi atol=2e-14 rtol=0
        @test curvature_measure_sum(turning, amplitudes; require_closed=false) == raw

        @test_throws DomainError curvature_measure_sum(-turning, amplitudes)
        @test_throws DomainError curvature_measure_sum(turning[1:end-1], amplitudes[1:end-1])
        @test_throws DimensionMismatch curvature_measure_sum(turning, amplitudes[1:end-1])
        invalid_amplitudes = copy(amplitudes)
        invalid_amplitudes[1] = complex(NaN, 0.0)
        @test_throws DomainError curvature_measure_sum(turning, invalid_amplitudes)
    end

    @testset "types, AD, domains, and allocations" begin
        coefficient32 = intrinsic_seam_coefficient(0.01f0, 25f0, 0.7f0)
        @test coefficient32 isa ComplexF32
        @test (@inferred intrinsic_seam_coefficient(
            Float16(0.01), Float16(25), Float16(0.7),
        )) isa ComplexF16
        @test (@inferred intrinsic_seam_coefficient(
            Float32(0.01), Float32(25), Float32(0.7),
        )) isa ComplexF32
        turning32 = fill(2f0 * Float32(pi) / 16f0, 16)
        amplitudes32 = ComplexF32[complex(cospi(index / 8), sinpi(index / 8)) for index in 0:15]
        @test curvature_measure_sum(turning32, amplitudes32) isa ComplexF32
        @test curvature_continuum_harmonic(3, 7.3f0, 0.41f0) isa ComplexF32
        @test (@inferred curvature_continuum_harmonic(
            2, Float16(1), Float16(0.3),
        )) isa ComplexF32
        @test (@inferred curvature_continuum_harmonic(
            2, Float32(1), Float32(0.3),
        )) isa ComplexF32
        @test (@inferred curvature_continuum_harmonic(
            2, Float64(1), Float64(0.3),
        )) isa ComplexF64
        dual16 = ForwardDiff.Dual(Float16(1), Float16(1))
        dual32 = ForwardDiff.Dual(Float32(1), Float32(1))
        dual64 = ForwardDiff.Dual(Float64(1), Float64(1))
        @test (@inferred intrinsic_seam_coefficient(
            ForwardDiff.Dual(Float16(0.01), Float16(1)), Float16(25), Float16(0.7),
        )) isa Complex{ForwardDiff.Dual{Nothing,Float16,1}}
        @test (@inferred intrinsic_seam_coefficient(
            ForwardDiff.Dual(Float32(0.01), Float32(1)), Float32(25), Float32(0.7),
        )) isa Complex{ForwardDiff.Dual{Nothing,Float32,1}}
        z_ad16 = @inferred curvature_continuum_harmonic(2, dual16, Float16(0.3))
        phi_ad16 = @inferred curvature_continuum_harmonic(2, Float16(1), dual16)
        @test real(z_ad16) isa ForwardDiff.Dual{Nothing,Float32,1}
        @test real(phi_ad16) isa ForwardDiff.Dual{Nothing,Float32,1}
        @test (@inferred curvature_continuum_harmonic(
            2, dual32, Float32(0.3),
        )) isa Complex{typeof(dual32)}
        @test (@inferred curvature_continuum_harmonic(
            2, Float32(1), dual32,
        )) isa Complex{typeof(dual32)}
        @test (@inferred curvature_continuum_harmonic(
            2, dual64, Float64(0.3),
        )) isa Complex{typeof(dual64)}
        @test (@inferred curvature_continuum_harmonic(
            2, Float64(1), dual64,
        )) isa Complex{typeof(dual64)}
        @test ForwardDiff.derivative(Float16(1)) do local_z
            real(curvature_continuum_harmonic(2, local_z, Float16(0.3)))
        end isa Float32

        turning16 = fill(Float16(2) * Float16(pi) / Float16(8), 8)
        amplitudes16 = ones(Float16, 8)
        @test curvature_measure_sum(turning16, amplitudes16) isa Float32
        @test debiased_curvature_measure_sum(turning16, amplitudes16) isa Float32
        for count in (5, 13)
            quantized_turning = fill(
                Float16(2) * Float16(pi) / Float16(count), count,
            )
            @test isfinite(curvature_measure_sum(
                quantized_turning, ones(Float16, count),
            ))
            @test isfinite(debiased_curvature_measure_sum(
                quantized_turning, ones(Float16, count),
            ))
        end
        gradient16 = ForwardDiff.gradient(turning16) do local_turning
            debiased_curvature_measure_sum(
                local_turning, amplitudes16; require_closed=false,
            )
        end
        @test gradient16 ≈ ones(Float32, 8) rtol=4eps(Float32)
        for invalid_turning in (Float16[2, 2, 3], Float16[2, 2, 2])
            invalid_amplitudes = ones(Float16, length(invalid_turning))
            @test_throws DomainError curvature_measure_sum(
                invalid_turning, invalid_amplitudes,
            )
            @test_throws DomainError debiased_curvature_measure_sum(
                invalid_turning, invalid_amplitudes,
            )
        end
        @test_throws DomainError curvature_measure_sum(
            Float16[Float16(pi)], Float16[1]; require_closed=false,
        )
        @test_throws DomainError debiased_curvature_measure_sum(
            Float16[Float16(pi)], Float16[1]; require_closed=false,
        )

        for bit_precision in (32, 64, 128, 512)
            turning_big, amplitudes_big, z_big, phi_big =
                setprecision(BigFloat, bit_precision) do
                    local_turning = fill(2BigFloat(pi) / 3, 3)
                    local_amplitudes = fill(complex(BigFloat(1), BigFloat(0)), 3)
                    local_turning, local_amplitudes, BigFloat(1), BigFloat("0.3")
                end
            raw_big = curvature_measure_sum(turning_big, amplitudes_big)
            debiased_big = debiased_curvature_measure_sum(turning_big, amplitudes_big)
            harmonic_big = curvature_continuum_harmonic(2, z_big, phi_big)
            @test (@inferred curvature_continuum_harmonic(2, z_big, phi_big)) isa
                  Complex{BigFloat}
            @test precision(real(raw_big)) == bit_precision
            @test precision(real(debiased_big)) == bit_precision
            @test precision(real(harmonic_big)) == bit_precision
            expected_turning, turning_tolerance = setprecision(BigFloat, bit_precision) do
                2BigFloat(pi), 8eps(BigFloat)
            end
            @test debiased_big ≈ expected_turning rtol=turning_tolerance
            raw_big_gradient = ForwardDiff.gradient(turning_big) do local_turning
                real(curvature_measure_sum(local_turning, amplitudes_big))
            end
            debiased_big_gradient = ForwardDiff.gradient(turning_big) do local_turning
                real(debiased_curvature_measure_sum(local_turning, amplitudes_big))
            end
            @test all(isfinite, raw_big_gradient)
            @test debiased_big_gradient == ones(BigFloat, 3)
        end

        turning64 = setprecision(BigFloat, 64) do
            fill(2BigFloat(pi) / 3, 3)
        end
        amplitudes512 = setprecision(BigFloat, 512) do
            fill(BigFloat(1), 3)
        end
        phi512 = setprecision(BigFloat, 512) do
            BigFloat("0.3")
        end
        @test (@inferred curvature_continuum_harmonic(2, 1.0, phi512)) isa
              Complex{BigFloat}
        mixed_precision_sum = debiased_curvature_measure_sum(turning64, amplitudes512)
        @test precision(mixed_precision_sum) == 512
        for turning_precision in (32, 64)
            turning_mixed = setprecision(BigFloat, turning_precision) do
                BigFloat[BigFloat("0.314159265358979323")]
            end
            amplitude_mixed = setprecision(BigFloat, 512) do
                BigFloat[BigFloat("1.234567890123456789")]
            end
            raw_mixed = curvature_measure_sum(
                turning_mixed, amplitude_mixed; require_closed=false,
            )
            raw_reference, raw_tolerance = setprecision(BigFloat, 512) do
                turning_reference = BigFloat(first(turning_mixed))
                amplitude_reference = BigFloat(first(amplitude_mixed))
                n_reference = one(BigFloat) + turning_reference / BigFloat(pi)
                local_angle = turning_reference / (2n_reference)
                bias_reference = (tan(local_angle) / local_angle) / n_reference^2
                turning_reference * amplitude_reference * bias_reference,
                128eps(BigFloat)
            end
            @test precision(raw_mixed) == 512
            @test raw_mixed ≈ raw_reference rtol=raw_tolerance atol=0
        end
        for turning_mixed in (Float16(0.3), Float32(0.3), Float64(0.3))
            amplitude_mixed = setprecision(BigFloat, 512) do
                BigFloat[BigFloat("1.234567890123456789")]
            end
            raw_mixed = curvature_measure_sum(
                [turning_mixed], amplitude_mixed; require_closed=false,
            )
            raw_reference, raw_tolerance = setprecision(BigFloat, 512) do
                turning_reference = BigFloat(turning_mixed)
                amplitude_reference = BigFloat(first(amplitude_mixed))
                n_reference = one(BigFloat) + turning_reference / BigFloat(pi)
                local_angle = turning_reference / (2n_reference)
                bias_reference = (tan(local_angle) / local_angle) / n_reference^2
                turning_reference * amplitude_reference * bias_reference,
                128eps(BigFloat)
            end
            @test precision(raw_mixed) == 512
            @test raw_mixed ≈ raw_reference rtol=raw_tolerance atol=0
        end
        heterogeneous_turning = copy(turning64)
        heterogeneous_turning[2] = setprecision(BigFloat, 128) do
            BigFloat(heterogeneous_turning[2])
        end
        @test_throws ArgumentError debiased_curvature_measure_sum(
            heterogeneous_turning, amplitudes512,
        )

        derivative = ForwardDiff.derivative(0.05) do turning
            real(intrinsic_seam_coefficient(turning, 25.0, 0.7))
        end
        h = 1e-6
        centered = (
            real(intrinsic_seam_coefficient(0.05 + h, 25.0, 0.7)) -
            real(intrinsic_seam_coefficient(0.05 - h, 25.0, 0.7))
        ) / (2h)
        @test derivative ≈ centered rtol=2e-8 atol=2e-10

        turning_seed = fill(2pi / 8, 8)
        raw_gradient = ForwardDiff.gradient(turning_seed) do local_turning
            real(curvature_measure_sum(
                local_turning, ones(8); require_closed=false,
            ))
        end
        debiased_gradient = ForwardDiff.gradient(turning_seed) do local_turning
            real(debiased_curvature_measure_sum(
                local_turning, ones(8); require_closed=false,
            ))
        end
        closed_gradient = ForwardDiff.gradient(turning_seed) do local_turning
            real(debiased_curvature_measure_sum(local_turning, ones(8)))
        end
        @test all(isfinite, raw_gradient)
        @test debiased_gradient == ones(8)
        @test closed_gradient == ones(8)

        @test_throws DomainError intrinsic_seam_coefficient(0.0, 1.0, 1.0)
        @test_throws DomainError intrinsic_seam_coefficient(nextfloat(Float64(pi)), 1.0, 1.0)
        @test_throws DomainError intrinsic_seam_coefficient(0.1, Inf, 1.0)
        @test_throws ArgumentError intrinsic_seam_coefficient(big"0.1", big"1", big"1")
        @test curvature_continuum_harmonic(2, big"1", big"0") isa Complex{BigFloat}

        _curvature_coefficient_probe()
        _curvature_raw_sum_probe()
        _curvature_debiased_sum_probe()
        @test @allocated(_curvature_coefficient_probe()) == 0
        @test @allocated(_curvature_raw_sum_probe()) == 0
        @test @allocated(_curvature_debiased_sum_probe()) == 0
    end
end
