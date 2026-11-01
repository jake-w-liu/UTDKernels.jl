module FiniteRangeRegressions

using Test, UTDKernels, ForwardDiff

struct Inner end
struct Outer end

function nested_distance(distance::T, seed::T) where {T<:AbstractFloat}
    inner = ForwardDiff.Dual{Inner}(distance, seed)
    return ForwardDiff.Dual{Outer}(inner, ForwardDiff.Dual{Inner}(seed, zero(T)))
end

function leaves(value)
    return (ForwardDiff.value(ForwardDiff.value(value)),
        ForwardDiff.partials(ForwardDiff.value(value))[1],
        ForwardDiff.value(ForwardDiff.partials(value)[1]),
        ForwardDiff.partials(ForwardDiff.partials(value)[1])[1])
end

function sqrt_reference(distance::T, seed::T) where {T<:AbstractFloat}
    return setprecision(BigFloat, 2048) do
        ell, s = BigFloat(distance), BigFloat(seed)
        root = sqrt(ell)
        T.((root, s/(2root), s/(2root), -(s*s)/(4ell*root)))
    end
end

@testset "flat face retains representable original-distance coefficients" begin
    for T in (Float32, Float64), tiny in (false, true), explicit in (false, true)
        distance, seed = tiny ?
            (T === Float32 ? (T(1e-40), T(1e-20)) : (T(1e-320), T(1e-160))) :
            (one(T), T(0.25))
        argument = nested_distance(distance, seed)
        expected = sqrt_reference(distance, seed)
        @test all(isfinite, expected)
        wedge, delta, k = Wedge(T(pi)), T(pi)/2, one(T)
        evaluate(L) = explicit ? pec_wedge_face_edge(wedge, delta, k, L, zero(T)) :
                                 pec_wedge_face_edge(wedge, delta, k, L)
        ordinary, result = evaluate(distance), evaluate(argument)
        tolerance = T === Float32 ? 3e-6 : 2e-12
        for name in (:edge, :face, :Ds, :Dh)
            value = getproperty(result, name)
            reference_value = getproperty(ordinary, name)
            sign = name === :edge ? 0 : name === :Ds ? -1 : 1
            real_coefficients, imag_coefficients = leaves(real(value)), leaves(imag(value))
            @test value isa Complex{typeof(argument)}
            @test all(isfinite, real_coefficients)
            @test all(isfinite, imag_coefficients)
            @test real_coefficients[1] == real(reference_value)
            @test imag_coefficients[1] == imag(reference_value)
            for index in eachindex(expected)
                reference = sign * expected[index]
                if iszero(reference)
                    @test iszero(real_coefficients[index])
                else
                    @test isapprox(real_coefficients[index], reference; rtol=tolerance, atol=0)
                end
                @test iszero(imag_coefficients[index])
            end
        end
        if !tiny
            @test isequal(result.face, complex(sqrt(argument)))
        end
    end
end

function passive_sector_reference(value, tolerance)
    return setprecision(BigFloat, 512) do
        re, im = BigFloat(real(value)), BigFloat(imag(value))
        all(isfinite, (re, im)) || return false
        radius = max(one(re), sqrt(re*re + im*im))
        re/radius >= -BigFloat(tolerance) && im/radius <= BigFloat(tolerance)
    end
end

@testset "finite passive coordinates retain their Euclidean sector" begin
    for T in (Float32, Float64)
        largest = floatmax(T)
        coordinates = ((largest, -largest), (-largest, -largest),
            (largest, largest), (-largest, largest),
            (one(T), -one(T)), (-one(T), -one(T)),
            (one(T), one(T)), (-one(T), one(T)),
            (zero(T), zero(T)), (one(T), 32eps(T)), (-32eps(T), -one(T)),
            (T(Inf), -one(T)), (one(T), T(-Inf)),
            (T(NaN), -one(T)), (one(T), T(NaN)))
        for (re, im) in coordinates, tolerance in (nothing, zero(T), T(0.25), one(T))
            value = complex(re, im)
            band = tolerance === nothing ? 64eps(T) : tolerance
            actual = tolerance === nothing ? is_passive_transition_argument(value) :
                is_passive_transition_argument(value; axis_rtol=tolerance)
            @test actual isa Bool
            @test actual == passive_sector_reference(value, band)
            if isfinite(re) && isfinite(im) && isfinite(hypot(re, im)) && band > zero(T)
                original_scale = max(one(re), hypot(re, im))
                original = re/original_scale >= -band && im/original_scale <= band
                @test actual === original
            end
        end
        for tolerance in (-one(T), T(Inf), T(NaN))
            @test_throws ArgumentError is_passive_transition_argument(
                complex(one(T), -one(T)); axis_rtol=tolerance)
        end
    end
end

@testset "zero passive axis band retains original coordinate signs" begin
    for T in (Float32, Float64)
        tiny, largest = nextfloat(zero(T)), floatmax(T)
        coordinates = ((-tiny, -largest), (largest, tiny),
            (zero(T), -largest), (-zero(T), -largest),
            (largest, zero(T)), (largest, -zero(T)))
        for (re, im) in coordinates, tolerance in (nothing, zero(T), eps(T))
            value = complex(re, im)
            band = tolerance === nothing ? 64eps(T) : tolerance
            actual = tolerance === nothing ? is_passive_transition_argument(value) :
                is_passive_transition_argument(value; axis_rtol=tolerance)
            @test isfinite(hypot(re, im))
            @test actual == passive_sector_reference(value, band)
        end
    end
end

@testset "lost coordinate signs retain original mixed-precision bands" begin
    context_precision = precision(BigFloat)
    for T in (Float32, Float64)
        tiny, largest = nextfloat(zero(T)), floatmax(T)
        small_band, larger_band = T === Float32 ? (1e-100, 1e-70) :
            setprecision(BigFloat, 256) do
                (BigFloat("1e-700"), BigFloat("1e-600"))
            end
        small_rational = big(1)//big(10)^(T === Float32 ? 100 : 700)
        larger_rational = big(1)//big(10)^(T === Float32 ? 70 : 600)
        coordinates = ((-tiny, -largest), (largest, tiny),
            (zero(T), -largest), (-zero(T), -largest),
            (largest, zero(T)), (largest, -zero(T)))
        for (re, im) in coordinates, band in
            (small_band, larger_band, 0, 1, 1//4, 1//3,
             small_rational, larger_rational, big(1))
            value = complex(re, im)
            @test is_passive_transition_argument(value; axis_rtol=band) ==
                passive_sector_reference(value, band)
        end
    end
    tolerance = setprecision(BigFloat, 1024) do
        BigFloat("1e-700")
    end
    value = complex(floatmax(Float64), nextfloat(0.0))
    @test !is_passive_transition_argument(value; axis_rtol=tolerance)
    @test precision(tolerance) == 1024
    @test precision(BigFloat) == context_precision
    excessive_precision = setprecision(BigFloat, UTDKernels._SHADOW_MAX_PHASE_WORK_BITS + 1) do
        BigFloat("1e-700")
    end
    @test_throws ArgumentError is_passive_transition_argument(value; axis_rtol=excessive_precision)
    @test precision(BigFloat) == context_precision
end

end
