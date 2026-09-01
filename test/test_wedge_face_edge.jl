using Random
using ForwardDiff

include("support/face_edge_oracle.jl")

face_edge_relerr(got, reference; floor=1e-300) =
    abs(got - reference) / max(abs(reference), floor)

function face_edge_first_order(epsilon, delta, k, L)
    b = delta / 2
    x0 = 2k * L * cos(b)^2
    bracket = F_utd(x0) / cos(b)^2 - 4k * L * sin(b)^2 * F_utd_prime(x0)
    return -cis(-pi / 4) * pi * epsilon * bracket / (2sqrt(2pi * k))
end

@noinline function _face_edge_allocation_probe()
    wedge = Wedge(1.01pi)
    return pec_wedge_face_edge(wedge, 0.5pi, 20.0, 1.0, 0.01)
end

@noinline function _face_edge_score_allocation_probe()
    wedge = Wedge(1.01pi)
    return pec_wedge_intrinsic_score(wedge, 0.5pi, 20.0, 1.0, 0.01)
end

@noinline function _face_edge_endpoint_allocation_probe()
    return pec_wedge_face_edge(
        Wedge((1 + 1e-6) * pi), prevfloat(Float64(pi)), 1e-300, 1.0, 1e-6,
    )
end

@noinline function _face_edge_symmetric_allocation_probe()
    return pec_wedge_face_edge(Wedge(1.01pi), 0.0, 20.0, 1.0, 0.01)
end

@testset "PEC reflection-boundary face-edge decomposition" begin
    @testset "shared real transition derivatives through third order" begin
        for x in (0.02, 0.5, 5.0, 34.0, 35.0, 60.0, 100.0)
            value, first, second, third = UTDKernels._F_utd_derivatives3(x)
            xb = BigFloat(x)
            reference_value = face_edge_F_hp(x; digits=110)
            recurrence = im + inv(2xb)
            reference_first = recurrence * reference_value - im
            reference_second = recurrence * reference_first - reference_value / (2xb^2)
            reference_third = recurrence * reference_second - reference_first / xb^2 +
                              reference_value / xb^3
            @test face_edge_relerr(value, reference_value) < 2e-12
            @test face_edge_relerr(first, reference_first) < 1e-10
            @test face_edge_relerr(second, reference_second) < 5e-10
            @test face_edge_relerr(third, reference_third) < 5e-9
        end
    end

    @testset "public domain contract" begin
        @test_throws FaceEdgeDomainError pec_wedge_face_edge(
            Wedge(0.9pi), 0.2pi, 10.0, 1.0,
        )
        @test_throws FaceEdgeDomainError pec_wedge_face_edge(
            Wedge(1.2pi), -eps(Float64), 10.0, 1.0,
        )
        @test_throws FaceEdgeDomainError pec_wedge_face_edge(
            Wedge(1.2pi), Float64(pi), 10.0, 1.0,
        )
        @test_throws FaceEdgeDomainError pec_wedge_face_edge(
            Wedge(1.2pi), 0.1pi, 10.0, 1.0,
        )
        @test_throws DomainError pec_wedge_face_edge(Wedge(1.2pi), 0.3pi, 0.0, 1.0)
        @test_throws DomainError pec_wedge_face_edge(Wedge(1.2pi), 0.3pi, 1.0, Inf)
        @test_throws DomainError pec_wedge_face_edge(Wedge(1.2pi), 0.3pi, 1.0, NaN)
        @test_throws ArgumentError pec_wedge_face_edge(
            Wedge(1.2pi), 0.3pi, 1.0, 1.0; convention=PhasorConvention(-1),
        )
        @test_throws FaceEdgeDomainError pec_wedge_face_edge(
            Wedge(1.01pi), 0.5pi, 10.0, 1.0, 0.02,
        )

        adjacent_alpha = nextfloat(Float64(pi))
        stored_defect = adjacent_alpha - Float64(pi)
        @test_throws FaceEdgeDomainError pec_wedge_face_edge(
            Wedge(adjacent_alpha), stored_defect, 1.0, 1.0,
        )
        @test isfinite(pec_wedge_face_edge(
            Wedge(adjacent_alpha), nextfloat(stored_defect), 1.0, 1.0,
        ).edge)
    end

    @testset "decomposition, limits, and frozen fixtures" begin
        flat = pec_wedge_face_edge(Wedge(pi), 0.4pi, 3.0, 2.25)
        @test flat.edge == 0.0 + 0.0im
        @test flat.face == 1.5 + 0.0im
        @test flat.Ds == -1.5 + 0.0im
        @test flat.Dh == 1.5 + 0.0im

        fixture = pec_wedge_face_edge(Wedge((1 + 1e-12) * pi), 0.8pi, 100.0, 1.0,
                                      1e-12)
        @test face_edge_relerr(
            fixture.edge,
            -4.931717647159758e-13 + 4.2642976963138624e-13im,
        ) < 2e-14
        @test face_edge_relerr(
            fixture.face,
            0.9999999999822753 - 1.7724538508561682e-11im,
        ) < 2e-14
        @test fixture.Dh == fixture.edge + fixture.face
        @test fixture.Ds == fixture.edge - fixture.face
        @test face_edge_relerr(
            pec_wedge_intrinsic_score(
                Wedge((1 + 1e-12) * pi), 0.8pi, 100.0, 1.0, 1e-12,
            ),
            1.6342384719769234e-11,
        ) < 2e-14

        ordinary_wedge = Wedge(1.01pi)
        ordinary = pec_wedge_face_edge(ordinary_wedge, 0.5pi, 20.0, 1.0)
        carried = pec_wedge_face_edge(ordinary_wedge, 0.5pi, 20.0, 1.0, 0.01)
        @test ordinary.edge ≈ carried.edge rtol=2e-13 atol=0
        @test ordinary.face ≈ carried.face rtol=2e-15 atol=0
        @test pec_wedge_intrinsic_score(ordinary_wedge, 0.5pi, 20.0, 1.0) ≈
              sqrt(2pi * 20.0) * abs(ordinary.edge) rtol=2e-15 atol=0
    end

    @testset "canonical four-term parity" begin
        rng = Xoshiro(0x2026_0162)
        for _ in 1:250
            epsilon = 10^(rand(rng) * (log10(0.2) + 7) - 7)
            minimum_fraction = min(0.88, epsilon + 0.04)
            delta = (minimum_fraction + rand(rng) * (0.94 - minimum_fraction)) * pi
            kL = 10^(6rand(rng) - 3)
            L = 10^(2rand(rng) - 1)
            k = kL / L
            wedge = Wedge((1 + epsilon) * pi)
            split = pec_wedge_face_edge(wedge, delta, k, L, epsilon)
            Ds0, Dh0 = pec_wedge_DsDh(
                wedge,
                RayAngles((wedge.alpha + delta) / 2, (wedge.alpha - delta) / 2),
                k,
                L,
            )
            @test face_edge_relerr(split.Ds, Ds0; floor=1e-12) < 3e-8
            @test face_edge_relerr(split.Dh, Dh0; floor=1e-12) < 3e-8
        end
    end

    @testset "independent high-precision residual" begin
        for fraction in (0.2, 0.5, 0.8), epsilon in (1e-8, 1e-10, 1e-12, 1e-14)
            delta = fraction * pi
            wedge = Wedge((1 + epsilon) * pi)
            got = pec_wedge_face_edge(wedge, delta, 100.0, 1.0, epsilon).edge
            reference = face_edge_incident_hp(epsilon, delta, 100.0, 1.0; digits=100)
            @test face_edge_relerr(got, reference) < 2e-11
        end

        epsilon = 1e-12
        delta = 0.8pi
        wedge = Wedge((1 + epsilon) * pi)
        split = pec_wedge_face_edge(wedge, delta, 100.0, 1.0, epsilon)
        reference = face_edge_incident_hp(epsilon, delta, 100.0, 1.0; digits=100)
        angles = RayAngles((wedge.alpha + delta) / 2, (wedge.alpha - delta) / 2)
        Ds0, Dh0 = pec_wedge_DsDh(wedge, angles, 100.0, 1.0)
        direct = (Ds0 + Dh0) / 2
        direct_error = face_edge_relerr(direct, reference)
        compensated_error = face_edge_relerr(split.edge, reference)
        @test direct_error > 1e-5
        @test compensated_error < 2e-11
        @test direct_error / compensated_error > 1e5
    end

    @testset "endpoint-adjacent and electrical extremes" begin
        epsilon = 1e-6
        delta = prevfloat(Float64(pi))
        wedge = Wedge((1 + epsilon) * pi)
        for k in (1e-300, 1e-10, 1.0, 1e10, 1e300)
            got = pec_wedge_face_edge(wedge, delta, k, 1.0, epsilon).edge
            reference = face_edge_incident_hp(epsilon, delta, k, 1.0; digits=110)
            @test isfinite(got)
            @test face_edge_relerr(got, reference) < 2e-8
        end

        # Retain the separately represented tiny defect when it is much
        # smaller than the endpoint gap, and avoid 0*Inf in the large-x Taylor
        # difference when delta_x^2 itself overflows.
        for (small_epsilon, endpoint_gap, k) in (
            (1e-14, 1e-4, 1e-300),
            (1e-14, 1e-4, 1.0),
            (1e-14, 1e-4, 1e300),
            (1e-6, 1e-4, 1e-10),
        )
            local_delta = pi - endpoint_gap
            local_wedge = Wedge((1 + small_epsilon) * pi)
            got = pec_wedge_face_edge(
                local_wedge, local_delta, k, 1.0, small_epsilon,
            ).edge
            reference = face_edge_incident_hp(
                small_epsilon, local_delta, k, 1.0; digits=110,
            )
            @test face_edge_relerr(got, reference) < 2e-8
        end

        for kL in (floatmin(Float64), 1e-200, 1e200, floatmax(Float64) / 4)
            split = pec_wedge_face_edge(Wedge(1.01pi), 0.5pi, kL, 1.0, 0.01)
            @test isfinite(split.edge)
            @test isfinite(split.face)
        end

        extreme = pec_wedge_face_edge(Wedge(1.01pi), 0.5pi, 1e308, 1e308, 0.01)
        @test all(isfinite, values(extreme))
        @test isfinite(pec_wedge_intrinsic_score(
            Wedge(1.01pi), 0.5pi, 1e308, 1e308, 0.01,
        ))
    end

    @testset "first-order coplanar remainder" begin
        epsilons = (1e-2, 5e-3, 2.5e-3, 1.25e-3)
        errors = map(epsilons) do epsilon
            wedge = Wedge((1 + epsilon) * pi)
            exact = pec_wedge_face_edge(wedge, 0.5pi, 20.0, 1.0, epsilon).edge
            abs(exact - face_edge_first_order(epsilon, 0.5pi, 20.0, 1.0))
        end
        slopes = ntuple(i -> log(errors[i] / errors[i + 1]) / log(2), 3)
        @test minimum(slopes) > 1.85
    end

    @testset "numeric types and differentiation" begin
        epsilon32 = 0.01f0
        wedge32 = Wedge((1f0 + epsilon32) * Float32(pi))
        split32 = pec_wedge_face_edge(
            wedge32,
            0.5f0 * Float32(pi),
            20f0,
            1f0,
            epsilon32,
        )
        @test split32.edge isa ComplexF32
        @test split32.face isa ComplexF32
        @test all(isfinite, (split32.Ds, split32.Dh, split32.edge, split32.face))
        @test pec_wedge_intrinsic_score(
            wedge32, 0.5f0 * Float32(pi), 20f0, 1f0, epsilon32,
        ) isa AbstractFloat

        wedge = Wedge(1.01pi)
        edge_delta(delta) = real(pec_wedge_face_edge(wedge, delta, 20.0, 1.0, 0.01).edge)
        edge_k(k) = imag(pec_wedge_face_edge(wedge, 0.5pi, k, 1.0, 0.01).edge)
        edge_L(L) = real(pec_wedge_face_edge(wedge, 0.5pi, 20.0, L, 0.01).edge)
        for (function_, point) in ((edge_delta, 0.5pi), (edge_k, 20.0), (edge_L, 1.0))
            h = 1e-6 * max(abs(point), 1.0)
            ad = ForwardDiff.derivative(function_, point)
            finite_difference = (function_(point + h) - function_(point - h)) / (2h)
            @test isfinite(ad)
            @test ad ≈ finite_difference rtol=2e-7 atol=2e-10
        end

        @test_throws ArgumentError pec_wedge_face_edge(
            Wedge(BigFloat(pi) * big"1.01"),
            BigFloat(pi) / 2,
            big"20",
            big"1",
            big"0.01",
        )
    end

    @testset "allocation gate" begin
        _face_edge_allocation_probe()
        _face_edge_score_allocation_probe()
        _face_edge_endpoint_allocation_probe()
        _face_edge_symmetric_allocation_probe()
        @test @allocated(_face_edge_allocation_probe()) == 0
        @test @allocated(_face_edge_score_allocation_probe()) == 0
        @test @allocated(_face_edge_endpoint_allocation_probe()) == 0
        @test @allocated(_face_edge_symmetric_allocation_probe()) == 0
    end
end
