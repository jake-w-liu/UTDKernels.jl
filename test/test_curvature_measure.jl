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
        turning32 = fill(2f0 * Float32(pi) / 16f0, 16)
        amplitudes32 = ComplexF32[complex(cospi(index / 8), sinpi(index / 8)) for index in 0:15]
        @test curvature_measure_sum(turning32, amplitudes32) isa ComplexF32
        @test curvature_continuum_harmonic(3, 7.3f0, 0.41f0) isa ComplexF32

        derivative = ForwardDiff.derivative(0.05) do turning
            real(intrinsic_seam_coefficient(turning, 25.0, 0.7))
        end
        h = 1e-6
        centered = (
            real(intrinsic_seam_coefficient(0.05 + h, 25.0, 0.7)) -
            real(intrinsic_seam_coefficient(0.05 - h, 25.0, 0.7))
        ) / (2h)
        @test derivative ≈ centered rtol=2e-8 atol=2e-10

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
