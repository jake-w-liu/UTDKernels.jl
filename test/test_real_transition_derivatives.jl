using Test, ForwardDiff, UTDKernels

# The oracle uses a convergent erfc series, independently of the production
# Gaussian quadrature. Carry the DE recurrence out at 512-bit precision.
function real_transition_reference(x)
    return setprecision(BigFloat, 512) do
        value = UTDKernels.FaceEdge.transition_F_hp(x; dps=120)
        q = BigFloat(x)
        recurrence = im + inv(2q)
        first = recurrence * value - im
        second = recurrence * first - value / (2q * q)
        third = recurrence * second - first / (q * q) + value / (q * q * q)
        value, first, second, third
    end
end

# Generate the large-argument coefficients independently before rounding any
# seeded result. At x=1e200 the omitted terms are far below the binary target.
function real_transition_large_reference(x, derivative_order)
    return setprecision(BigFloat, 2300) do
        q = BigFloat(x)
        coefficient = one(complex(q))
        total = derivative_order == 0 ? coefficient : zero(coefficient)
        for order in 1:8
            coefficient *= im * (BigFloat(order) - BigFloat(1) / 2)
            factor = prod(
                BigFloat(-order - offset) for offset in 0:(derivative_order - 1);
                init=one(q),
            )
            total += coefficient * factor / q^(order + derivative_order)
        end
        total
    end
end

@testset "real transition derivative accuracy" begin
    arguments = vcat(
        collect(range(4.0, prevfloat(64.0); length=241)),
        [nextfloat(4.0), 19.098300562505262, prevfloat(35.0), 35.0,
         nextfloat(35.0), 40.0, 48.0, prevfloat(64.0), 64.0, nextfloat(64.0)],
    )
    for x in arguments
        reference = real_transition_reference(x)
        result = UTDKernels._F_utd_derivatives3(x)
        for order in 1:3
            @test isapprox(result[order + 1], reference[order + 1]; rtol=3e-14, atol=0)
        end
        @test F_utd_prime(x) == result[2]
        @test F_utd_second(x) == result[3]
    end
    for x in (prevfloat(4.0), 4.0, nextfloat(4.0))
        reference = real_transition_reference(x)
        @test isapprox(F_utd_prime(x), reference[2]; rtol=2e-10, atol=0)
        @test isapprox(F_utd_second(x), reference[3]; rtol=2e-9, atol=0)
    end
end

@testset "face-edge transition differences" begin
    model = UTDKernels.FaceEdge
    for (epsilon, delta, k) in ((1e-12, 0.8pi, 100.0), (1e-8, 0.5pi, 20.0))
        geometry = model.BoundaryGeometry(epsilon, delta, k, 1.0)
        reference80 = model.edge_residual_hp(geometry; dps=80)
        reference120 = model.edge_residual_hp(geometry; dps=120)
        @test isapprox(reference80, reference120; rtol=big"1e-70", atol=0)
        @test isapprox(model.edge_residual_compensated(geometry), reference120; rtol=2e-14, atol=0)
    end
    for epsilon in (1e-15, 1e-12, 1e-8, 1e-4, 0.01),
        delta in (0.3pi, 0.6pi, 0.8pi, 0.95pi), x in (4.1, 12.0, 19.1, 34.0)
        k = x / (2cos(delta / 2)^2)
        geometry = model.BoundaryGeometry(epsilon, delta, k, 1.0)
        reference = model.edge_residual_hp(geometry; dps=90)
        @test isapprox(model.edge_residual_compensated(geometry), reference; rtol=2e-13, atol=0)
    end
end

@testset "real transition AD and working types" begin
    for x in (4.25, 12.0, 19.098300562505262, 34.5)
        reference = real_transition_reference(x)
        for component in (real, imag)
            first = ForwardDiff.derivative(q -> component(F_utd_prime(q)), x)
            second = ForwardDiff.derivative(
                q -> ForwardDiff.derivative(t -> component(F_utd_prime(t)), q), x,
            )
            @test isapprox(first, component(reference[3]); rtol=4e-13, atol=0)
            @test isapprox(second, component(reference[4]); rtol=8e-13, atol=0)
        end
        for (s, t) in ((1e150, 1e-150), (1e300, 1e-300), (1e-300, 1e300))
            f(u) = ForwardDiff.derivative(v -> real(F_utd_prime(x + s*u + t*v)), 0.0)
            @test isapprox(ForwardDiff.derivative(f, 0.0), real(reference[4]) * (s*t);
                           rtol=1e-12, atol=0)
        end
    end
    @test F_utd_prime(5f0) isa ComplexF64
    @test F_utd_second(5f0) isa ComplexF32
    @test_throws ArgumentError F_utd_prime(BigFloat(5))
    @test_throws ArgumentError F_utd_second(BigFloat(40))
    for x in (0.0, -1.0, Inf, NaN)
        @test_throws DomainError F_utd_prime(x)
        @test_throws DomainError F_utd_second(x)
    end
    for f in (F_utd, F_utd_prime, F_utd_second)
        @test_throws DomainError f(ForwardDiff.Dual(19.0, Inf))
    end
end

@testset "real transition calibration and value AD" begin
    for x in (4.25, 19.098300562505262, 35.0, 40.0, 48.0,
              prevfloat(64.0), 64.0, nextfloat(64.0))
        @test F_utd_prime(x; asymptotic_threshold=35) == F_utd_prime(x; asymptotic_threshold=64)
        @test F_utd_prime(x; asymptotic_threshold=ForwardDiff.Dual(35.0, 1.0)) == F_utd_prime(x)
        a = F_utd_prime(ForwardDiff.Dual(x, 1.0); asymptotic_threshold=35)
        b = F_utd_prime(ForwardDiff.Dual(x, 1.0); asymptotic_threshold=64)
        reference = real_transition_reference(x)
        for component in (real, imag)
            @test ForwardDiff.value(component(a)) == ForwardDiff.value(component(b))
            @test ForwardDiff.partials(component(a)) == ForwardDiff.partials(component(b))
            value_ad = F_utd(ForwardDiff.Dual(x, 1.0))
            @test ForwardDiff.value(component(value_ad)) == component(F_utd(x))
            first = ForwardDiff.derivative(t -> component(F_utd(t)), x)
            second = ForwardDiff.derivative(
                t -> ForwardDiff.derivative(u -> component(F_utd(u)), t), x,
            )
            @test isapprox(first, component(reference[2]); rtol=4e-13, atol=0)
            @test isapprox(second, component(reference[3]); rtol=8e-13, atol=0)
        end
    end
    for threshold in (30, 34, 36)
        @test_throws DomainError F_utd_prime(10.0; asymptotic_threshold=threshold)
    end
end

@testset "original-germ range recovery" begin
    x = 19.098300562505262
    reference = real_transition_reference(x)
    for (f, order) in ((F_utd, 1), (F_utd_prime, 2), (F_utd_second, 3)),
        seed in (1e300, 1e-300, 1e-310, 1e-320)
        result = f(ForwardDiff.Dual(x, seed))
        for component in (real, imag)
            expected = Float64(component(reference[order + 1]) * BigFloat(seed))
            actual = ForwardDiff.partials(component(result))[1]
            if abs(expected) < floatmin(Float64)
                @test actual == expected
            else
                @test isapprox(actual, expected; rtol=3e-13, atol=0)
            end
        end
    end
    for (f, order) in ((F_utd, 1), (F_utd_prime, 2), (F_utd_second, 3))
        result = f(ForwardDiff.Dual(1e200, 1e300))
        reference_large = real_transition_large_reference(1e200, order)
        for component in (real, imag)
            expected = Float64(component(reference_large) * BigFloat(1e300))
            actual = ForwardDiff.partials(component(result))[1]
            @test isapprox(actual, expected; rtol=5e-14, atol=0)
        end
    end
    for (f, order) in ((F_utd, 2), (F_utd_prime, 3)), component in (real, imag)
        s, t = 1e300, 1e-300
        fun(u) = ForwardDiff.derivative(v -> component(f(x + s*u + t*v)), 0.0)
        @test isapprox(ForwardDiff.derivative(fun, 0.0), component(reference[order + 1]) * (s*t);
                       rtol=2e-12, atol=0)
    end
end

@testset "existing BigFloat fixed-series route" begin
    for x in (35, 40, 64, 100)
        reference = setprecision(BigFloat, 256) do
            q = BigFloat(x)
            inverse_x = inv(q)
            power = inverse_x
            total = zero(complex(q))
            for (order, coefficient) in enumerate(UTDKernels.ASYM_F_COEFFS)
                total += -order * coefficient * power * inverse_x
                power *= inverse_x
            end
            total
        end
        @test isapprox(F_utd_prime(BigFloat(x)), reference; rtol=big"1e-70", atol=0)
    end
end
