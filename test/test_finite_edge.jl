using Test
using UTDKernels
using ForwardDiff
using QuadGK

const FE_GEOMETRY = FiniteEdgeGeometry(1.0, 1.4, -0.6, 0.9)
const FE_DATA = finite_edge_phase_data(FE_GEOMETRY)
const FE_K = 100.0

function fe_benchmark_amplitude(s, geometry)
    rs, rp = finite_edge_distances(s, geometry)
    return inv(sqrt(rs * rp))
end

function fe_benchmark_local_amplitude(geometry)
    data = finite_edge_phase_data(geometry)
    rho_s = geometry.rho_s
    rho_p = geometry.rho_p
    ell1 = -data.q / (2 * data.c^2) * (inv(rho_s) - inv(rho_p))
    ell2 = -(1 - data.q^2) / (2 * data.c^4) *
           (rho_s^(-2) + rho_p^(-2))
    A0 = (rho_s * rho_p)^(-0.5) / data.c
    A1 = A0 * ell1
    A2 = A0 * (ell2 + ell1^2)
    return FiniteEdgeAmplitude(complex(A0), complex(A1), complex(A2))
end

function fe_inverse_coordinate(t, geometry)
    data = finite_edge_phase_data(geometry)
    iszero(t) && return data.s0
    target = data.phi0 + t * t
    grow = max(1.0, geometry.rho_s, geometry.rho_p)
    if t > 0
        lo = data.s0
        hi = data.s0 + grow
        for _ in 1:80
            finite_edge_phase(hi, geometry) >= target && break
            hi = data.s0 + 2 * (hi - data.s0)
        end
        finite_edge_phase(hi, geometry) >= target || error("right root not bracketed")
        for _ in 1:80
            mid = (lo + hi) / 2
            if finite_edge_phase(mid, geometry) < target
                lo = mid
            else
                hi = mid
            end
        end
    else
        lo = data.s0 - grow
        hi = data.s0
        for _ in 1:80
            finite_edge_phase(lo, geometry) >= target && break
            lo = data.s0 - 2 * (data.s0 - lo)
        end
        finite_edge_phase(lo, geometry) >= target || error("left root not bracketed")
        for _ in 1:80
            mid = (lo + hi) / 2
            if finite_edge_phase(mid, geometry) > target
                lo = mid
            else
                hi = mid
            end
        end
    end
    return (lo + hi) / 2
end

function fe_reference_integral(a, b, k, geometry; amplitude=fe_benchmark_amplitude)
    b < a && return -fe_reference_integral(b, a, k, geometry; amplitude)
    data = finite_edge_phase_data(geometry)
    points = a < data.s0 < b ? (a, data.s0, b) : (a, b)
    total = 0.0 + 0.0im
    for index in 1:(length(points) - 1)
        lo = points[index]
        hi = points[index + 1]
        re, _ = quadgk(
            s -> real(amplitude(s, geometry) * exp(-im * k * finite_edge_phase(s, geometry))),
            lo,
            hi;
            atol=2e-13,
            rtol=2e-13,
        )
        imv, _ = quadgk(
            s -> imag(amplitude(s, geometry) * exp(-im * k * finite_edge_phase(s, geometry))),
            lo,
            hi;
            atol=2e-13,
            rtol=2e-13,
        )
        total += re + im * imv
    end
    return total
end

@testset "Finite-edge endpoint-uniform kernel" begin
    @testset "geometry and phase contract" begin
        @test FiniteEdgeGeometry(1.0f0, 2.0f0, -3.0f0, 4.0f0) isa
              FiniteEdgeGeometry{Float32}
        @test FiniteEdgeGeometry(1, 2.0f0, -3, 4.0) isa
              FiniteEdgeGeometry{Float64}
        for values in (
            (0.0, 1.0, 0.0, 0.0),
            (-1.0, 1.0, 0.0, 0.0),
            (Inf, 1.0, 0.0, 0.0),
            (NaN, 1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0),
            (1.0, Inf, 0.0, 0.0),
            (1.0, NaN, 0.0, 0.0),
            (1.0, 1.0, Inf, 0.0),
            (1.0, 1.0, 0.0, NaN),
        )
            @test_throws ArgumentError FiniteEdgeGeometry(values...)
        end
        @test_throws ArgumentError FiniteEdgeAmplitude(1.0, Inf, 0.0)

        @test finite_edge_phase_derivative(FE_DATA.s0, FE_GEOMETRY) ≈ 0 atol=1e-13
        grid = range(FE_DATA.s0 - 1, FE_DATA.s0 + 1; length=101)
        @test minimum(finite_edge_phase(grid, FE_GEOMETRY)) + 1e-14 >= FE_DATA.phi0
        rs, rp = finite_edge_distances(grid, FE_GEOMETRY)
        second = FE_GEOMETRY.rho_s^2 ./ rs.^3 .+
                 FE_GEOMETRY.rho_p^2 ./ rp.^3
        @test all(second .> 0)
    end

    @testset "phase data and frozen values" begin
        @test FE_DATA.q ≈ 0.625 atol=1e-15
        @test FE_DATA.c ≈ 1.1792476415070754 atol=1e-14
        @test FE_DATA.s0 ≈ 0.025000000000000022 atol=1e-15
        @test FE_DATA.phi0 ≈ 2.830194339616981 atol=1e-14
        @test FE_DATA.phi2 ≈ 1.0453654855151007 atol=1e-14
        @test FE_DATA.phi3 ≈ -0.4027090152867162 atol=1e-14
        @test FE_DATA.phi4 ≈ 0.7260397527608984 atol=1e-14

        h = 2e-4
        f(x) = finite_edge_phase(x, FE_GEOMETRY)
        s0 = FE_DATA.s0
        f2 = (-f(s0 + 2h) + 16f(s0 + h) - 30f(s0) + 16f(s0 - h) -
              f(s0 - 2h)) / (12h^2)
        f3 = (f(s0 + 2h) - 2f(s0 + h) + 2f(s0 - h) -
              f(s0 - 2h)) / (2h^3)
        @test f2 ≈ FE_DATA.phi2 atol=2e-7
        @test f3 ≈ FE_DATA.phi3 atol=2e-5
    end

    @testset "rationalized phase coordinate" begin
        transform = finite_edge_transform_data(
            FE_DATA,
            fe_benchmark_local_amplitude(FE_GEOMETRY),
        )
        for t in (-2e-3, -1e-3, 1e-3, 2e-3)
            s_series = FE_DATA.s0 + transform.alpha * t +
                       transform.beta * t^2 + transform.gamma * t^3
            @test finite_edge_phase_coordinate(s_series, FE_GEOMETRY) ≈ t atol=2e-11
        end
        for x in (1e-12, -1e-12, 1e-10, -1e-10)
            t = finite_edge_phase_coordinate(FE_DATA.s0 + x, FE_GEOMETRY)
            @test t / x ≈ inv(transform.alpha) atol=2e-6
        end
        @test finite_edge_coordinate_derivative(FE_DATA.s0, FE_GEOMETRY) ≈
              sqrt(FE_DATA.phi2 / 2) atol=2e-13
        @test_throws ArgumentError finite_edge_phase_coordinate(
            FE_DATA.s0,
            FE_GEOMETRY;
            cancellation_threshold=-eps(Float64),
        )

        geometry = FiniteEdgeGeometry(1e6, 1.0, 0.0, 0.0)
        s = 2.0
        t = finite_edge_phase_coordinate(s, geometry)
        reference = setprecision(BigFloat, 256) do
            sqrt(hypot(BigFloat(1e6), BigFloat(s)) +
                 hypot(one(BigFloat), BigFloat(s)) - (BigFloat(1e6) + 1))
        end
        @test abs(BigFloat(t) - reference) < big"2e-16"
    end

    @testset "Fresnel convention and moments" begin
        C05, S05 = finite_edge_fresnel_cs(0.5)
        @test S05 ≈ 0.06473243285999929 atol=1e-15
        @test C05 ≈ 0.4923442258714464 atol=1e-15
        C1, S1 = finite_edge_fresnel_cs(1.0)
        @test S1 ≈ 0.4382591473903547 atol=1e-15
        @test C1 ≈ 0.779893400376823 atol=1e-15
        Cm, Sm = finite_edge_fresnel_cs(-0.5)
        @test Cm == -C05
        @test Sm == -S05

        ta, tb = -0.4, 0.7
        moments = finite_edge_fresnel_moments(ta, tb, FE_K)
        for n in 0:2
            reference, _ = quadgk(
                t -> t^n * exp(-im * FE_K * t^2),
                ta,
                tb;
                atol=1e-13,
                rtol=1e-13,
            )
            @test abs(moments[n + 1] - reference) < 2e-13
        end
    end

    @testset "EPM evaluation, endpoint derivative, and limits" begin
        amplitude = fe_benchmark_local_amplitude(FE_GEOMETRY)
        a = fe_inverse_coordinate(-6 / sqrt(FE_K), FE_GEOMETRY)
        b = fe_inverse_coordinate(2 / sqrt(FE_K), FE_GEOMETRY)
        c = fe_inverse_coordinate(-0.7 / sqrt(FE_K), FE_GEOMETRY)
        value = finite_edge_epm(a, b, FE_K, FE_GEOMETRY, amplitude; order=2)
        frozen = 0.058605317128991426 - 0.16498488900161204im
        @test abs(value - frozen) < 1e-12
        reference = fe_reference_integral(a, b, FE_K, FE_GEOMETRY)
        scale = abs(finite_edge_stationary_phase(FE_K, FE_GEOMETRY, amplitude.A0))
        @test abs(value - reference) / scale < 5e-3

        split = finite_edge_epm(a, c, FE_K, FE_GEOMETRY, amplitude; order=2) +
                finite_edge_epm(c, b, FE_K, FE_GEOMETRY, amplitude; order=2)
        @test abs(value - split) < 2e-14
        @test finite_edge_epm(b, a, FE_K, FE_GEOMETRY, amplitude) == -value
        @test finite_edge_epm(a, a, FE_K, FE_GEOMETRY, amplitude) == 0

        exact = fe_benchmark_amplitude(b, FE_GEOMETRY) *
                exp(-im * FE_K * finite_edge_phase(b, FE_GEOMETRY))
        approximate = finite_edge_endpoint_derivative(
            b,
            FE_K,
            FE_GEOMETRY,
            amplitude,
        )
        @test abs(approximate - exact) / abs(exact) < 5e-3

        stationary_endpoint = finite_edge_endpoint_derivative(
            FE_DATA.s0,
            FE_K,
            FE_GEOMETRY,
            amplitude,
        )
        exact_stationary = fe_benchmark_amplitude(FE_DATA.s0, FE_GEOMETRY) *
                           exp(-im * FE_K * FE_DATA.phi0)
        @test abs(stationary_endpoint - exact_stationary) < 3e-13

        for (width, tolerance) in ((8.0, 0.08), (16.0, 0.04))
            left = fe_inverse_coordinate(-width / sqrt(FE_K), FE_GEOMETRY)
            right = fe_inverse_coordinate(width / sqrt(FE_K), FE_GEOMETRY)
            epm0 = finite_edge_epm(
                left,
                right,
                FE_K,
                FE_GEOMETRY,
                amplitude;
                order=0,
            )
            stationary = finite_edge_stationary_phase(
                FE_K,
                FE_GEOMETRY,
                amplitude.A0,
            )
            @test abs(epm0 - stationary) / abs(stationary) < tolerance
        end
    end

    @testset "continuity, translation, and AD" begin
        amplitude = fe_benchmark_local_amplitude(FE_GEOMETRY)
        a = fe_inverse_coordinate(-6 / sqrt(FE_K), FE_GEOMETRY)
        at = finite_edge_epm(a, FE_DATA.s0, FE_K, FE_GEOMETRY, amplitude)
        left = finite_edge_epm(
            a,
            FE_DATA.s0 - 1e-10,
            FE_K,
            FE_GEOMETRY,
            amplitude,
        )
        right = finite_edge_epm(
            a,
            FE_DATA.s0 + 1e-10,
            FE_K,
            FE_GEOMETRY,
            amplitude,
        )
        scale = abs(finite_edge_stationary_phase(FE_K, FE_GEOMETRY, amplitude.A0))
        @test abs(left - at) / scale < 1e-8
        @test abs(right - at) / scale < 1e-8

        base = FiniteEdgeGeometry(1.0, 1.0, 0.0, 0.0)
        translated = FiniteEdgeGeometry(1.0, 1.0, 1e9, 1e9)
        @test finite_edge_phase_coordinate(100.0, base) ==
              finite_edge_phase_coordinate(1e9 + 100.0, translated)
        base_amplitude = fe_benchmark_local_amplitude(base)
        translated_amplitude = fe_benchmark_local_amplitude(translated)
        @test finite_edge_epm(-3.0, 2.0, 20.0, base, base_amplitude) ==
              finite_edge_epm(
                  1e9 - 3.0,
                  1e9 + 2.0,
                  20.0,
                  translated,
                  translated_amplitude,
              )

        function observer_value(zp)
            geometry = FiniteEdgeGeometry(1.0, 1.4, -0.6, zp)
            local_amplitude = fe_benchmark_local_amplitude(geometry)
            return real(finite_edge_epm(a, FE_DATA.s0 + 0.15, FE_K,
                                        geometry, local_amplitude))
        end
        ad = ForwardDiff.derivative(observer_value, FE_GEOMETRY.z_p)
        h = 2e-5
        fd = (observer_value(FE_GEOMETRY.z_p - 2h) -
              8observer_value(FE_GEOMETRY.z_p - h) +
              8observer_value(FE_GEOMETRY.z_p + h) -
              observer_value(FE_GEOMETRY.z_p + 2h)) / (12h)
        @test ad ≈ fd rtol=2e-7 atol=2e-9

        coordinate_slope = ForwardDiff.derivative(
            s -> finite_edge_phase_coordinate(s, FE_GEOMETRY),
            FE_DATA.s0,
        )
        @test coordinate_slope ≈ sqrt(FE_DATA.phi2 / 2) rtol=2e-14 atol=0
        @test ForwardDiff.derivative(
            z -> first(finite_edge_fresnel_cs(z)),
            0.0,
        ) == 1.0
        @test ForwardDiff.derivative(
            z -> last(finite_edge_fresnel_cs(z)),
            0.0,
        ) == 0.0

        derivative_from_epm(endpoint) = finite_edge_epm(
            a,
            endpoint,
            FE_K,
            FE_GEOMETRY,
            amplitude,
        )
        derivative_at_stationary = complex(
            ForwardDiff.derivative(
                endpoint -> real(derivative_from_epm(endpoint)),
                FE_DATA.s0,
            ),
            ForwardDiff.derivative(
                endpoint -> imag(derivative_from_epm(endpoint)),
                FE_DATA.s0,
            ),
        )
        @test derivative_at_stationary ≈ finite_edge_endpoint_derivative(
            FE_DATA.s0,
            FE_K,
            FE_GEOMETRY,
            amplitude,
        ) rtol=2e-13 atol=0

        parameter_amplitude = FiniteEdgeAmplitude(
            0.8 + 0.1im,
            0.2 - 0.03im,
            -0.05 + 0.02im,
        )
        amplitude_derivative = FiniteEdgeAmplitude(
            0.04 - 0.01im,
            -0.02 + 0.005im,
            0.01 + 0.003im,
        )
        a0 = FE_DATA.s0 - 0.3
        b0 = FE_DATA.s0 + 0.2
        da = -0.07
        db = 0.11
        field_parameter(theta) = finite_edge_epm(
            a0 + da * theta,
            b0 + db * theta,
            FE_K,
            FE_GEOMETRY,
            FiniteEdgeAmplitude(
                parameter_amplitude.A0 + theta * amplitude_derivative.A0,
                parameter_amplitude.A1 + theta * amplitude_derivative.A1,
                parameter_amplitude.A2 + theta * amplitude_derivative.A2,
            ),
        )
        transformed_amplitude_derivative = finite_edge_transform_data(
            FE_DATA,
            amplitude_derivative,
        )
        assembled = finite_edge_parameter_derivative(
            a0,
            b0,
            FE_K,
            FE_GEOMETRY,
            parameter_amplitude,
            (
                transformed_amplitude_derivative.G0,
                transformed_amplitude_derivative.G1,
                transformed_amplitude_derivative.G2,
            ),
            0.0,
            finite_edge_coordinate_derivative(a0, FE_GEOMETRY) * da,
            finite_edge_coordinate_derivative(b0, FE_GEOMETRY) * db,
        )
        automatic = complex(
            ForwardDiff.derivative(theta -> real(field_parameter(theta)), 0.0),
            ForwardDiff.derivative(theta -> imag(field_parameter(theta)), 0.0),
        )
        @test assembled ≈ automatic rtol=3e-13 atol=0
        @test finite_edge_parameter_derivative(
            b0,
            a0,
            FE_K,
            FE_GEOMETRY,
            parameter_amplitude,
            (
                transformed_amplitude_derivative.G0,
                transformed_amplitude_derivative.G1,
                transformed_amplitude_derivative.G2,
            ),
            0.0,
            finite_edge_coordinate_derivative(b0, FE_GEOMETRY) * db,
            finite_edge_coordinate_derivative(a0, FE_GEOMETRY) * da,
        ) ≈ -assembled rtol=2e-15 atol=0
    end

    @testset "fail-closed inputs and allocation gate" begin
        amplitude = fe_benchmark_local_amplitude(FE_GEOMETRY)
        for invalid in (Inf, -Inf, NaN)
            @test_throws ArgumentError finite_edge_phase(invalid, FE_GEOMETRY)
            @test_throws ArgumentError finite_edge_phase_coordinate(invalid, FE_GEOMETRY)
            @test_throws ArgumentError finite_edge_fresnel_moments(invalid, 0.1, FE_K)
            @test_throws ArgumentError finite_edge_epm(
                invalid,
                0.1,
                FE_K,
                FE_GEOMETRY,
                amplitude,
            )
        end
        for invalid in (0.0, -1.0, Inf, NaN)
            @test_throws ArgumentError finite_edge_fresnel_moments(-0.1, 0.1, invalid)
            @test_throws ArgumentError finite_edge_epm(
                -0.1,
                0.1,
                invalid,
                FE_GEOMETRY,
                amplitude,
            )
        end
        @test_throws ArgumentError finite_edge_epm(
            -0.1,
            0.1,
            FE_K,
            FE_GEOMETRY,
            amplitude;
            order=3,
        )

        geometry32 = FiniteEdgeGeometry(1.0f0, 1.4f0, -0.6f0, 0.9f0)
        amplitude32 = FiniteEdgeAmplitude(
            ComplexF32(1.0f0),
            ComplexF32(0.1f0),
            ComplexF32(0.01f0),
        )
        @test finite_edge_phase_coordinate(0.3f0, geometry32) isa Float32
        @test finite_edge_fresnel_moments(-0.2f0, 0.3f0, 100.0f0) isa
              NTuple{3,ComplexF32}
        @test finite_edge_epm(
            -0.2f0,
            0.3f0,
            100.0f0,
            geometry32,
            amplitude32,
        ) isa ComplexF32
        @test finite_edge_endpoint_derivative(
            0.3f0,
            100.0f0,
            geometry32,
            amplitude32,
        ) isa ComplexF32

        @test finite_edge_phase_coordinate(
            big"0.3",
            FiniteEdgeGeometry(big"1.0", big"1.4", big"-0.6", big"0.9"),
        ) isa BigFloat
        @test_throws ArgumentError finite_edge_fresnel_cs(big"0.3")
        @test_throws DomainError finite_edge_phase_data(
            FiniteEdgeGeometry(floatmax(Float64), floatmax(Float64), 0.0, 1.0),
        )

        scalar_epm() = finite_edge_epm(
            -0.2,
            0.3,
            FE_K,
            FE_GEOMETRY,
            amplitude,
        )
        scalar_coordinate() = finite_edge_phase_coordinate(0.3, FE_GEOMETRY)
        scalar_endpoint() = finite_edge_endpoint_derivative(
            0.3,
            FE_K,
            FE_GEOMETRY,
            amplitude,
        )
        scalar_parameter() = finite_edge_parameter_derivative(
            -0.2,
            0.3,
            FE_K,
            FE_GEOMETRY,
            amplitude,
            (0.01 + 0.0im, -0.02 + 0.0im, 0.03 + 0.0im),
            0.04,
            -0.05,
            0.06,
        )
        scalar_epm()
        scalar_coordinate()
        scalar_endpoint()
        scalar_parameter()
        @test @allocated(scalar_epm()) == 0
        @test @allocated(scalar_coordinate()) == 0
        @test @allocated(scalar_endpoint()) == 0
        @test @allocated(scalar_parameter()) == 0
    end
end
