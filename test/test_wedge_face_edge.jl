using Random
using ForwardDiff
using SpecialFunctions

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

@testset "PEC face-edge decomposition on the midpoint slice" begin
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
            -4.931717647160173e-13 + 4.264297696313514e-13im,
        ) < 5e-14  # Independent exact-input integral; original gate retained.
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
        rng = Xoshiro(UInt32(539361634))
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

    @testset "compensated differences carry no series truncation" begin
        # The erfcx series once truncated at eighth order and the large-argument
        # Taylor form at third order. The first four samples then had relative
        # errors from 1e-11 to 3e-10, the others about 1e-13. The tolerance is
        # 128 eps for these samples: the erfcx derivative loses a factor 2|z|^2<=32 to
        # cancellation at the top of its range, and the final sum adds a few rounding
        # errors. Rare inputs near |z|=4 reach about 430 eps, because erfcx itself then
        # carries about 14 eps into that factor.
        tolerance = 128eps(Float64)
        for (epsilon, fraction, kL) in (
            (0.0272, 0.8282, 1.09),
            (0.162, 0.8902, 0.0337),
            (0.05, 0.60, 0.5),
            (0.00226, 0.9240, 127.0),
            (1e-3, 0.50, 200.0),
            (0.02, 0.70, 1e4),
        )
            delta = fraction * pi
            got = pec_wedge_face_edge(Wedge((1 + epsilon) * pi), delta, kL, 1.0, epsilon).edge
            reference = face_edge_incident_hp(epsilon, delta, kL, 1.0; digits=100)
            @test face_edge_relerr(got, reference) < tolerance
        end

        # The same bound over the range that the local series serve. The range
        # stops at Delta=0.95pi because the conditioning of the rearrangement
        # grows as Delta approaches pi at small kL.
        rng = MersenneTwister(20261106)
        worst = 0.0
        for _ in 1:300
            epsilon = 10^(-6 + (6 + log10(0.5)) * rand(rng))
            kL = 10^(-3 + 9 * rand(rng))
            delta = pi * (epsilon + (0.95 - epsilon) * (0.01 + 0.98 * rand(rng)))
            got = pec_wedge_face_edge(Wedge((1 + epsilon) * pi), delta, kL, 1.0, epsilon).edge
            reference = face_edge_incident_hp(epsilon, delta, kL, 1.0; digits=100)
            worst = max(worst, face_edge_relerr(got, reference))
        end
        @test worst < tolerance

        # A truncated derivative would also show here: compare AD with a
        # high-precision central difference of the independent reference.
        for (epsilon, fraction, kL) in ((0.05, 0.60, 0.5), (0.02, 0.70, 1e4))
            delta = fraction * pi
            wedge = Wedge((1 + epsilon) * pi)
            ad = ForwardDiff.derivative(
                d -> real(pec_wedge_face_edge(wedge, d, kL, 1.0, epsilon).edge),
                delta,
            )
            reference = setprecision(BigFloat, 512) do
                center = BigFloat(delta)
                h = big"1e-40"
                upper = face_edge_incident_hp(epsilon, center + h, kL, 1.0; digits=100)
                lower = face_edge_incident_hp(epsilon, center - h, kL, 1.0; digits=100)
                Float64(real(upper - lower) / (2h))
            end
            @test ad ≈ reference rtol=1e-11
        end
    end

    @testset "compensated pair at the largest and smallest arguments" begin
        # Each case once threw or returned a wrong value. References use the
        # independent oracle, with enough digits to cover the cancellation.
        tolerance = 128eps(Float64)
        for (wedge_n, delta, k, L, epsilon) in (
            (1.5, pi - 1e-3, 0.3, 1.79e308, 0.5),     # overflow of delta/sqrt(k)
            (1.5, pi - 1e-6, 0.5, 1.5e308, 0.5),
            (1 + 1e-16, 2.0, 1e155, 1e155, 1e-16),    # overflow of k*L*a
            (1 + 1e-12, 2.0, 1e155, 1e155, 1e-12),
        )
            got = pec_wedge_face_edge(Wedge(wedge_n * pi), delta, k, L, epsilon).edge
            reference = face_edge_incident_hp(epsilon, delta, k, L; digits=1400)
            @test face_edge_relerr(got, reference) < tolerance
        end

        # Near Delta=pi at small kL the two terms nearly cancel. Value tolerance as
        # above; before the wider series these reached 1e4 eps.
        for (epsilon, delta, kL) in (
            (0.0038936786775456873, 3.1415926535897865, 2.9623168736449166e-6),
            (7.93e-6, 0.99131pi, 0.000153),
            (0.000113, 0.98354pi, 0.000192),
            (0.00253, 0.99381pi, 0.00534),
            (0.0117, 0.98825pi, 0.00154),
            (0.08694985330263774, 3.1415926534686505, 1.35272496983206e-5),
        )
            got = pec_wedge_face_edge(Wedge((1 + epsilon) * pi), delta, kL, 1.0, epsilon).edge
            reference = face_edge_incident_hp(epsilon, delta, kL, 1.0; digits=200)
            @test face_edge_relerr(got, reference) < tolerance
        end
        got32 = pec_wedge_face_edge(
            Wedge((1 + 0.0053550065f0) * Float32(pi)), 3.141585f0, 4.1460644f-6, 1f0, 0.0053550065f0,
        ).edge
        reference32 = face_edge_incident_hp(0.0053550065f0, 3.141585f0, 4.1460644f-6, 1f0; digits=200)
        @test face_edge_relerr(got32, reference32) < 128eps(Float32)
    end

    @testset "derivatives at the largest and smallest arguments" begin
        central(f, x; h=big"1e-60") = setprecision(BigFloat, 2048) do
            (f(BigFloat(x) + h) - f(BigFloat(x) - h)) / (2h)
        end
        second(f, x; h=big"1e-60") = setprecision(BigFloat, 2048) do
            X = BigFloat(x)
            (f(X + h) - 2f(X) + f(X - h)) / h^2
        end
        oracle(e, d, k, L) = face_edge_incident_hp(e, d, k, L; digits=500)

        # These threw in the wide workspace.
        ad = ForwardDiff.derivative(
            t -> real(pec_wedge_face_edge(Wedge(1.01pi), t, 1e150, 1e3, 0.01).edge), 1.0,
        )
        @test ad ≈ Float64(central(t -> real(oracle(0.01, t, 1e150, 1e3)), 1.0)) rtol=1e-12
        ad = ForwardDiff.derivative(
            a -> ForwardDiff.derivative(
                t -> real(pec_wedge_face_edge(Wedge(1.01pi), t, 1e100, 1e3, 0.01).edge), a,
            ),
            1.0,
        )
        @test ad ≈ Float64(second(t -> real(oracle(0.01, t, 1e100, 1e3)), 1.0)) rtol=1e-12

        # Rounding noise in 2z*erfcx(z)-2/sqrt(pi) once gave an error 1e12 times the
        # true derivative, at |z| near 1e12. Here scale(n)*n cancels analytically, so
        # the derivative is 1e-12 of the coefficient and the bound follows the latter.
        e, d, k = 0.10981614478265225, 3.1415926535897802, 8.361994684554112e24
        edge(t) = pec_wedge_face_edge(Wedge((1 + t) * pi), d, k, 1.0, t).edge
        ad = ForwardDiff.derivative(t -> real(edge(t)), e)
        @test abs(ad - Float64(central(t -> real(oracle(t, d, k, 1.0)), e))) <=
              16eps(Float64) * abs(edge(e))

        # The rule itself, on the transition ray and off it, against an independent
        # continued fraction of erfcx in high precision. Below |z|=8 the rule keeps
        # 2z*erfcx(z)-2/sqrt(pi), which amplifies the 1e-14 or so of the Float64
        # erfcx by 2|z|^2; from |z|=8 it should reach full precision.
        for radius in (7.9, 8.1, 100.0, 1e8, 1e12), angle in (pi / 4, 0.1, 1.2)
            z = radius * cis(angle)
            dual = complex(
                ForwardDiff.Dual{:erfcx_rule}(real(z), 1.0),
                ForwardDiff.Dual{:erfcx_rule}(imag(z), 0.0),
            )
            result = SpecialFunctions.erfcx(dual)
            got = complex(ForwardDiff.partials(real(result))[1], ForwardDiff.partials(imag(result))[1])
            expected = setprecision(BigFloat, 2048) do
                w = Complex{BigFloat}(z)
                fraction = w
                for level in 400:-1:1
                    fraction = w + (level / big(2)) / fraction
                end
                root_pi = sqrt(BigFloat(pi))
                ComplexF64(2w / (root_pi * fraction) - 2 / root_pi)
            end
            @test abs(got - expected) <= (radius < 8 ? 1e-11 : 8eps(Float64)) * abs(expected)
        end

        # Off the transition ray z^2 overflows at huge |z| although the result is
        # representable: -1/(sqrt(pi)*z^2) = 1im/(2*sqrt(pi)*1e310) here. The value is
        # subnormal, so it carries about 43 bits.
        dual = complex(
            ForwardDiff.Dual{:erfcx_rule}(1e155, 1.0),
            ForwardDiff.Dual{:erfcx_rule}(1e155, 0.0),
        )
        result = SpecialFunctions.erfcx(dual)
        got = complex(ForwardDiff.partials(real(result))[1], ForwardDiff.partials(imag(result))[1])
        expected = setprecision(BigFloat, 2048) do
            Float64(1 / (2 * sqrt(BigFloat(pi)) * big"1e310"))
        end
        @test isfinite(got)
        @test abs(imag(got) - expected) <= 1e-12 * expected
        @test abs(real(got)) <= 1e-12 * expected

        # Paired 1/t0 terms near Delta=pi once cost the sign in Float32.
        e32, k32 = 0.050378844f0, 0.00018688207f0
        ad32 = ForwardDiff.derivative(
            t -> real(pec_wedge_face_edge(Wedge((1 + e32) * Float32(pi)), t, k32, 1f0, e32).edge),
            3.1415925f0,
        )
        @test ad32 ≈ Float64(central(t -> real(oracle(e32, t, k32, 1f0)), 3.1415925f0)) rtol=1e-4

        # The same pairing at second order lost every digit in Float64.
        e, d, k = 0.08694985330263774, 3.1415926534686505, 1.35272496983206e-5
        ad = ForwardDiff.derivative(
            a -> ForwardDiff.derivative(
                t -> real(pec_wedge_face_edge(Wedge((1 + e) * pi), t, k, 1.0, e).edge), a,
            ),
            d,
        )
        @test ad ≈ Float64(second(t -> real(oracle(e, t, k, 1.0)), d)) rtol=1e-6

        # The pairing also reaches large x1 while x0 stays small. The compensated pair
        # gave a second derivative wrong by 1e12 here, because the 1/t0 terms of its two
        # products cancel to noise. The references are 2048-bit difference quotients.
        e = 0.2717765281594092
        d = 3.141592653589613
        k = 34.23687158904007
        pair(t) = pec_wedge_face_edge(Wedge((1 + e) * pi), t, k, 1.0, e).edge
        ad = complex(
            ForwardDiff.derivative(t -> real(pair(t)), d),
            ForwardDiff.derivative(t -> imag(pair(t)), d),
        )
        @test ad ≈ -1.6308601977279966 - 1.6693173741302867im rtol=1e-11
        ad = complex(
            ForwardDiff.derivative(a -> ForwardDiff.derivative(t -> real(pair(t)), a), d),
            ForwardDiff.derivative(a -> ForwardDiff.derivative(t -> imag(pair(t)), a), d),
        )
        @test ad ≈ 0.11332040894093098 - 17.13623002741227im rtol=1e-8
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
