using ForwardDiff

include("support/passive_transition_oracle.jl")

passive_relerr(got, reference; floor=1e-300) =
    abs(got - reference) / max(abs(reference), floor)

@noinline _passive_bundle_probe() = (
    F_utd(3 - 2im),
    F_utd_minus_one(3 - 2im),
    F_utd_prime(3 - 2im),
    F_utd_second(3 - 2im),
)

@noinline _passive_wedge_probe() = pec_wedge_DsDh(
    Wedge(1.55pi), RayAngles(0.62pi, 0.31pi), 20 - 5im, 1.3,
)

@noinline _existing_real_transition_probe() = (
    F_utd(3.0), F_utd_minus_one(100.0), F_utd_prime(100.0),
)

@testset "Passive-sheet complex transition" begin
    @testset "sector, root, and wavenumber contracts" begin
        @test F_utd(0.0 + 0.0im) == 0.0 + 0.0im
        @test F_utd_minus_one(0.0 + 0.0im) == -1.0 + 0.0im
        for theta in range(-pi / 2, 0; length=17)
            x = 10cis(theta)
            mapped = cis(pi / 4) * sqrt(x)
            @test -2e-15 <= angle(mapped) <= pi / 4 + 2e-15
            @test is_passive_transition_argument(x)
        end
        @test is_passive_transition_argument(1.0)
        @test is_passive_transition_argument(-1e-16im)
        @test is_passive_transition_argument(complex(-1e-16, 0.0))
        @test !is_passive_transition_argument(1 + 0.1im)
        @test !is_passive_transition_argument(-1.0)
        @test !is_passive_transition_argument(complex(Inf, 0.0))
        @test !is_passive_transition_argument(complex(NaN, 0.0))
        @test_throws ArgumentError is_passive_transition_argument(1.0; axis_rtol=-1.0)
        @test_throws PassiveSheetError F_utd_prime(1 + 0.1im)
        @test_throws PassiveSheetError F_utd_second(-1.0 + 0im)
        @test_throws DomainError F_utd_prime(0.0 + 0im)
        @test_throws DomainError F_utd_second(0.0)
        @test isfinite(F_utd_prime(1e-300 + 0im))
        @test_throws DomainError F_utd_second(1e-220 + 0im)

        @test passive_wavenumber(10.0, 0.2) == 10 - 2im
        @test passive_wavenumber(10f0, 0.2f0) isa ComplexF32
        @test_throws DomainError passive_wavenumber(0.0, 0.1)
        @test_throws DomainError passive_wavenumber(1.0, -0.1)
        @test_throws DomainError passive_wavenumber(Inf, 0.1)
        @test_throws DomainError passive_wavenumber(floatmax(Float64), 2.0)
        @test_throws DomainError passive_wavenumber(floatmax(Float32), 2.0f0)
    end

    @testset "shared Faddeeva identity" begin
        for z in (0.2 - 0.1im, 3 - 5im, 20 - 10im, 5.0, 5 - 800im)
            @test passive_relerr(
                UTDKernels._faddeeva_w(im * z),
                UTDKernels._faddeeva_erfcx(z),
            ) < 2e-14
        end
    end

    @testset "independent high-precision passive-sector grid" begin
        for theta in range(-pi / 2, 0; length=7)
            for radius in exp10.(range(-12, 8; length=21))
                x = radius * cis(theta)
                reference = passive_transition_oracle(x)
                @test passive_relerr(F_utd(x), reference[1]) < 2e-12
                @test passive_relerr(F_utd_minus_one(x), reference[2]) < 2e-12
                @test passive_relerr(F_utd_prime(x), reference[3]) < 3e-11
                @test passive_relerr(F_utd_second(x), reference[4]) < 3e-10
            end
        end
    end

    @testset "limits and crossover overlap" begin
        phase = cis(pi / 4)
        for theta in (-pi / 2, -pi / 4, 0.0)
            tiny = 1e-16 * cis(theta)
            @test passive_relerr(F_utd(tiny), sqrt(pi) * phase * sqrt(tiny)) < 3e-8
            large = 1e9 * cis(theta)
            @test abs(F_utd(large) - 1) < 1e-8
            @test passive_relerr(F_utd_minus_one(large), 0.5im / large) < 4e-9
        end

        for theta in range(-pi / 2, 0; length=13), radius in (0.35, 40.0, 48.0)
            x = radius * cis(theta)
            reference = passive_transition_oracle(x)
            bundle = UTDKernels._passive_transition_all(x)
            @test passive_relerr(bundle.F, reference[1]) < 2e-12
            @test passive_relerr(bundle.Fm1, reference[2]) < 2e-12
            @test passive_relerr(bundle.Fp, reference[3]) < 3e-11
            @test passive_relerr(bundle.Fpp, reference[4]) < 3e-10
        end


        # A zero-imaginary Complex request may choose different safe
        # crossovers, but it must recover the established real-axis APIs.
        for x in (1e-6, 0.1, 1.0, 10.0, 100.0, 1e6)
            xc = complex(x)
            @test passive_relerr(F_utd(xc), F_utd(x)) < 2e-12
            @test passive_relerr(F_utd_minus_one(xc), F_utd_minus_one(x)) < 2e-12
            @test passive_relerr(F_utd_prime(xc), F_utd_prime(x)) < 3e-11
        end
    end

    @testset "cancellation-safe residual and derivatives" begin
        x = -1e6im
        reference = passive_transition_oracle(x)
        safe = F_utd_prime(x)
        naive = (im + inv(2x)) * F_utd(x) - im
        @test passive_relerr(safe, reference[3]) < 1e-12
        @test passive_relerr(naive, reference[3]) > 1e-8
        @test passive_relerr(naive, reference[3]) >
              10passive_relerr(safe, reference[3])

        x = -1e16im
        @test F_utd(x) - 1 == 0
        @test abs(F_utd_minus_one(x)) > 0
        @test passive_relerr(
            F_utd_minus_one(x), passive_transition_oracle(x)[2],
        ) < 1e-12

        radii = exp10.(range(5, 9; length=17))
        subtraction_condition = map(radii) do radius
            x = -im * radius
            (abs(F_utd(x)) + 1) / abs(F_utd_minus_one(x))
        end
        ode_condition = map(radii) do radius
            x = -im * radius
            (abs((im + inv(2x)) * F_utd(x)) + 1) / abs(F_utd_prime(x))
        end
        slope(values) = begin
            lx = log10.(radii)
            ly = log10.(values)
            mx = sum(lx) / length(lx)
            my = sum(ly) / length(ly)
            sum((lx .- mx) .* (ly .- my)) / sum((lx .- mx).^2)
        end
        @test abs(slope(subtraction_condition) - 1) < 2e-4
        @test abs(slope(ode_condition) - 2) < 5e-4
    end

    @testset "limiting absorption and Complex{Dual}" begin
        target = F_utd(3.0)
        errors = map((1e-2, 1e-3, 1e-4, 1e-5)) do attenuation
            abs(F_utd(passive_wavenumber(3.0, attenuation)) - target)
        end
        @test all(errors[index + 1] < errors[index] for index in 1:3)
        @test abs(log10(errors[3] / errors[4]) - 1) < 0.03

        for evaluator in (F_utd, F_utd_minus_one, F_utd_prime, F_utd_second)
            for projection in (real, imag)
                function_ = attenuation -> projection(evaluator(
                    passive_wavenumber(3.0, attenuation),
                ))
                point = 0.2
                h = 1e-6
                automatic = ForwardDiff.derivative(function_, point)
                centered = (function_(point + h) - function_(point - h)) / (2h)
                @test isfinite(automatic)
                @test automatic ≈ centered rtol=3e-7 atol=3e-10
            end
        end

        wedge = Wedge(1.55pi)
        angles = RayAngles(0.62pi, 0.31pi)
        coefficient(attenuation) = real(pec_wedge_DsDh(
            wedge, angles, passive_wavenumber(20.0, attenuation), 1.3,
        )[1])
        point = 0.25
        h = 1e-6
        automatic = ForwardDiff.derivative(coefficient, point)
        centered = (coefficient(point + h) - coefficient(point - h)) / (2h)
        @test isfinite(automatic)
        @test automatic ≈ centered rtol=5e-7 atol=5e-10
    end

    @testset "passive PEC wedge against independent reconstruction" begin
        alpha = 1.55pi
        phi = 0.62pi
        phip = 0.31pi
        wedge = Wedge(alpha)
        angles = RayAngles(phi, phip)
        for k in (20 - 0.2im, 20 - 5im, 80 - 40im)
            got = pec_wedge_DsDh(wedge, angles, k, 1.3)
            reference = passive_wedge_oracle(alpha, phi, phip, k, 1.3)
            @test passive_relerr(got[1], reference[1]) < 3e-12
            @test passive_relerr(got[2], reference[2]) < 3e-12
        end
    end

    @testset "general complex continuation regressions" begin
        # Positive-imaginary and left-half-plane F requests remain on the
        # pre-existing general principal-root analytic continuation.
        @test F_utd(1 + 0.5im) ≈
              0.8751297965348614 + 0.3059855179999363im rtol=2e-15 atol=0
        @test F_utd(-2 + 3im) ≈
              1.421939809058666 - 0.19242899854731038im rtol=2e-15 atol=0

        wedge = Wedge(1.5pi)
        angles = RayAngles(0.7pi, 0.3pi)
        complex_L = pec_wedge_DsDh(wedge, angles, 20 - 2im, 1.3 + 0.2im)
        @test complex_L[1] ≈
              0.5259405347188228 + 0.08411742578499362im rtol=5e-14 atol=0
        @test complex_L[2] ≈
              -0.5919652339419461 - 0.02607857406780728im rtol=5e-14 atol=0

        active = pec_wedge_DsDh(wedge, angles, 20 + 2im, 1.3)
        @test active[1] ≈
              0.5284806705823951 + 0.044773215404348365im rtol=5e-14 atol=0
        @test active[2] ≈
              -0.5884541205574884 + 0.019580972737644753im rtol=5e-14 atol=0
    end

    @testset "types, unsupported precision, and allocation gates" begin
        x32 = ComplexF32(0.5f0, -0.2f0)
        @test F_utd_minus_one(x32) isa ComplexF32
        @test F_utd_prime(x32) isa ComplexF32
        @test F_utd_second(x32) isa ComplexF32
        for x in (ComplexF32(1e-6, -1e-6), ComplexF32(100, -20))
            @test F_utd(x) isa ComplexF32
            @test all(isfinite, (
                F_utd(x), F_utd_minus_one(x), F_utd_prime(x), F_utd_second(x),
            ))
        end
        derivative32 = ForwardDiff.derivative(0.2f0) do attenuation
            real(F_utd(passive_wavenumber(3f0, attenuation)))
        end
        @test derivative32 isa Float32
        @test isfinite(derivative32)
        @test_throws ArgumentError F_utd_second(Complex{BigFloat}(1, -1))

        _passive_bundle_probe()
        _passive_wedge_probe()
        _existing_real_transition_probe()
        @test @allocated(_passive_bundle_probe()) == 0
        @test @allocated(_passive_wedge_probe()) == 0
        @test @allocated(_existing_real_transition_probe()) == 0
    end
end
