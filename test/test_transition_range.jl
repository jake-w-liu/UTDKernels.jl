using Test, ForwardDiff, UTDKernels

function transition_range_reference(x)
    return setprecision(BigFloat, 2300) do
        q = BigFloat(x)
        value = UTDKernels.FaceEdge.transition_F_hp(q; dps=160)
        recurrence = im + inv(2q)
        first = recurrence * value - im
        second = recurrence * first - value / (2q * q)
        third = recurrence * second - first / (q * q) + value / (q * q * q)
        fourth = recurrence * third - 3second / (2q * q) +
                 3first / (q * q * q) - 3value / (q * q * q * q)
        value, first, second, third, fourth
    end
end

@testset "finite tiny real transition derivatives" begin
    for x in (nextfloat(0.0), 1e-320, 1e-310, 1e-309, 1e-300,
              1e-200, 1e-100, 1e-16, 0.3, 3.9)
        reference = transition_range_reference(x)
        @test isfinite(F_utd_prime(x))
        @test isapprox(F_utd_prime(x), reference[2]; rtol=3e-14, atol=0)
        for seed in (1e-300, 1e-150, 1.0)
            expected = ComplexF64(reference[3] * BigFloat(seed))
            if isfinite(expected)
                result = F_utd_prime(ForwardDiff.Dual(x, seed))
                for component in (real, imag)
                    actual = ForwardDiff.partials(component(result))[1]
                    @test isapprox(actual, component(expected); rtol=2e-12, atol=0)
                end
            else
                @test_throws DomainError F_utd_prime(ForwardDiff.Dual(x, seed))
            end
        end
    end
end

@testset "tiny original-germ first and mixed derivatives" begin
    for (x, seed) in ((1e-309, 1e-300), (1e-320, 1e-300), (1f-40, 1f-30))
        reference = transition_range_reference(x)
        second_seed = oftype(x, x isa Float32 ? 1e-25 : 1e-200)
        for (f, order) in ((F_utd, 1), (F_utd_prime, 2)), component in (real, imag)
            expected = component(reference[order + 1]) * BigFloat(seed)
            first = ForwardDiff.derivative(t -> component(f(x + seed*t)), zero(x))
            @test isapprox(first, expected; rtol=2e-12, atol=0)
            for (s, t) in ((seed, seed), (second_seed, seed))
                mixed = ForwardDiff.derivative(
                    u -> ForwardDiff.derivative(v -> component(f(x + s*u + t*v)), zero(x)),
                    zero(x),
                )
                expected_mixed = component(reference[order + 2]) * BigFloat(s) * BigFloat(t)
                @test isapprox(mixed, expected_mixed; rtol=2e-12, atol=0)
            end
        end
    end
    for (x, seed) in ((1e-200, 1e-200), (1f-20, 1f-20))
        reference = transition_range_reference(x)
        tolerance = x isa Float32 ? 3e-6 : 2e-12
        for component in (real, imag)
            first = ForwardDiff.derivative(t -> component(F_utd_second(x + seed*t)), zero(x))
            mixed = ForwardDiff.derivative(
                u -> ForwardDiff.derivative(v -> component(F_utd_second(x + seed*u + seed*v)), zero(x)),
                zero(x),
            )
            @test isapprox(first, component(reference[4]) * BigFloat(seed); rtol=tolerance, atol=0)
            @test isapprox(mixed, component(reference[5]) * BigFloat(seed)^2; rtol=tolerance, atol=0)
        end
    end
end

@testset "real principal square-root scalar parity" begin
    for T in (Float32, Float64)
        extent = T === Float32 ? 120 : 1000
        values = T[zero(T), -zero(T), one(T), -one(T), floatmin(T),
                   nextfloat(zero(T)), floatmax(T), Inf, -Inf, NaN]
        for exponent in range(-extent, extent; length=100)
            value = T(exp2(exponent))
            push!(values, value, -value)
        end
        for x in values
            result = UTDKernels.safe_sqrt(x)
            reference = sqrt(complex(x))
            @test typeof(result) === typeof(reference) && isequal(result, reference)
        end
    end
    setprecision(BigFloat, 256) do
        for x in (zero(BigFloat), -zero(BigFloat), one(BigFloat), -one(BigFloat),
                  big"1e-1000", big"1e1000", -big"1e-1000", -big"1e1000")
            result = UTDKernels.safe_sqrt(x)
            reference = sqrt(complex(x))
            @test typeof(result) === typeof(reference) && isequal(result, reference)
        end
    end
    for x in (typemin(Int), typemax(Int), -1, 0, 1, true, false, -3//7, 3//7)
        result = UTDKernels.safe_sqrt(x)
        reference = sqrt(complex(x))
        @test typeof(result) === typeof(reference) && isequal(result, reference)
    end
end

function transition_negative_reference(x)
    return setprecision(BigFloat, 768) do
        q = BigFloat(x)
        root = sqrt(complex(q))
        phase = cis(BigFloat(pi) / 4)
        z = phase * root
        value = sqrt(BigFloat(pi)) * phase * root * exp(z*z) *
                UTDKernels.FaceEdge._erfc_hp(z; force_series=true)
        value, (im + inv(2q)) * value - im
    end
end

@testset "negative real transition root tangents" begin
    for (x, seed) in ((-1e-309, 1e-300), (-1e-320, 1e-300), (-1f-40, 1f-30))
        reference = transition_negative_reference(x)
        tolerance = x isa Float32 ? 3e-6 : 2e-12
        result = F_utd(ForwardDiff.Dual(x, seed))
        for component in (real, imag)
            @test ForwardDiff.value(component(result)) == component(F_utd(x))
            actual = ForwardDiff.partials(component(result))[1]
            @test isapprox(actual, component(reference[2]) * BigFloat(seed); rtol=tolerance, atol=0)
        end
    end
end

function transition_range_complex_reference(x)
    return setprecision(BigFloat, 2300) do
        q = complex(BigFloat(real(x)), BigFloat(imag(x)))
        root = sqrt(q)
        phase = cis(BigFloat(pi) / 4)
        z = phase * root
        value = sqrt(BigFloat(pi)) * phase * root * exp(z*z) *
                UTDKernels.FaceEdge._erfc_hp(z; force_series=true)
        recurrence = im + inv(2q)
        first = recurrence * value - im
        second = recurrence * first - value / (2q * q)
        third = recurrence * second - first / (q * q) + value / (q * q * q)
        fourth = recurrence * third - 3second / (2q * q) +
                 3first / (q * q * q) - 3value / (q * q * q * q)
        value, first, second, third, fourth
    end
end

function transition_range_large_reference(x, order)
    return setprecision(BigFloat, 2300) do
        q = complex(BigFloat(real(x)), BigFloat(imag(x)))
        coefficient = one(q)
        total = order == 0 ? one(q) : zero(q)
        for m in 1:8
            coefficient *= im * (BigFloat(m) - BigFloat(1) / 2)
            factor = prod(BigFloat(-m-j) for j in 0:(order-1); init=BigFloat(1))
            total += coefficient * factor / q^(m+order)
        end
        total
    end
end

function test_transition_coefficient(actual, expected)
    R = typeof(actual)
    rounded = convert(R, expected)
    if iszero(rounded) || abs(rounded) < floatmin(R)
        @test actual == rounded
    else
        @test isapprox(actual, rounded; rtol=R === Float32 ? 3e-6 : 2e-12, atol=0)
    end
end

@testset "passive complex original-coordinate derivatives" begin
    cases = ((F_utd, 1, 1e-309, 1e-300),
             (F_utd_prime, 2, 1e-309, 1e-300),
             (F_utd_minus_one, 1, 1e-309, 1e-300),
             (F_utd, 1, 1f-40, 1f-35),
             (F_utd_prime, 2, 1f-40, 1f-35),
             (F_utd_minus_one, 1, 1f-40, 1f-35),
             (F_utd_second, 3, 1e-200, 1e-200),
             (F_utd_second, 3, 1f-20, 1f-20))
    for (f, order, x, seed) in cases
        for (yi, real_seed, imag_seed) in ((zero(x), seed, zero(seed)),
                                         (zero(x), zero(seed), -seed),
                                         (-x/2, seed, -seed))
            q = complex(x, yi)
            reference = transition_range_complex_reference(q)
            direction = complex(BigFloat(real_seed), BigFloat(imag_seed))
            for component in (real, imag)
                field(t) = component(f(complex(x + real_seed*t, yi + imag_seed*t)))
                first = ForwardDiff.derivative(field, zero(x))
                mixed = ForwardDiff.derivative(
                    u -> ForwardDiff.derivative(v -> field(u + v), zero(x)), zero(x),
                )
                test_transition_coefficient(first, component(reference[order+1] * direction))
                test_transition_coefficient(mixed, component(reference[order+2] * direction^2))
            end
        end
    end
end

@testset "small second-derivative intermediate ranges" begin
    for x in (1e-80, 1e-100, 1e-108, 1e-110, 1e-120, 1e-140, 1e-150, 1f-14, 1f-16)
        reference = transition_range_reference(x)
        for component in (real, imag)
            first = ForwardDiff.derivative(t -> component(F_utd_second(x + x*t)), zero(x))
            mixed = ForwardDiff.derivative(
                u -> ForwardDiff.derivative(v -> component(F_utd_second(x + x*(u+v))), zero(x)),
                zero(x),
            )
            test_transition_coefficient(first, component(reference[4]) * BigFloat(x))
            test_transition_coefficient(mixed, component(reference[5]) * BigFloat(x)^2)
        end
    end
end

@testset "negative real mixed transition derivatives" begin
    for (x, seed) in ((-1e-309, 1e-300), (-1e-320, 1e-300), (-1f-40, 1f-30))
        reference = transition_range_complex_reference(x)
        other = oftype(x, x isa Float32 ? 1e-25 : 1e-200)
        for second_seed in (seed, other), component in (real, imag)
            mixed = ForwardDiff.derivative(
                u -> ForwardDiff.derivative(v -> component(F_utd(x + seed*u + second_seed*v)), zero(x)),
                zero(x),
            )
            expected = component(reference[3]) * BigFloat(seed) * BigFloat(second_seed)
            test_transition_coefficient(mixed, expected)
        end
    end
end

@testset "large transition coefficients are lifted before rounding" begin
    for (x, seed) in ((1e150, 1e150), (1e155, 1e155), (1e160, 1e160),
                      (1e200, 1e200), (1e200, 1e300), (1e100, 1e150), (1f25, 1f25))
        for (f, order) in ((F_utd, 1), (F_utd_minus_one, 1), (F_utd_prime, 2)), component in (real, imag)
            actual = ForwardDiff.derivative(t -> component(f(x + seed*t)), zero(x))
            expected = component(transition_range_large_reference(x, order)) * BigFloat(seed)
            test_transition_coefficient(actual, expected)
        end
    end
    for (x, seed) in ((1e150, 1e150), (1e155, 1e155), (1f15, 1f15))
        for (yi, imag_seed) in ((zero(x), zero(seed)), (-x/2, -seed))
            q = complex(x, yi)
            direction = complex(BigFloat(seed), BigFloat(imag_seed))
            for (f, order) in ((F_utd, 1), (F_utd_minus_one, 1), (F_utd_prime, 2)), component in (real, imag)
                actual = ForwardDiff.derivative(
                    t -> component(f(complex(x + seed*t, yi + imag_seed*t))), zero(x),
                )
                expected = component(transition_range_large_reference(q, order) * direction)
                test_transition_coefficient(actual, expected)
            end
        end
    end
end

@testset "passive complex-axis polynomial tangents" begin
    for x in (1e-20, 0.1, 1.0, 10.0)
        reference = transition_range_complex_reference(x)
        for component in (real, imag)
            actual = ForwardDiff.derivative(
                t -> component(F_utd_second(complex(x, -x*t))), 0.0,
            )
            expected = component(reference[4] * (-im * BigFloat(x)))
            @test isapprox(actual, expected; rtol=3e-9, atol=0)
        end
    end
end
