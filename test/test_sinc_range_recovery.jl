module SincRangeRecovery

using Test, UTDKernels, ForwardDiff

struct Inner end
struct Outer end
struct Third end

nested(x::T,a::T,b::T,c::T) where {T<:AbstractFloat} =
    ForwardDiff.Dual{Outer}(ForwardDiff.Dual{Inner}(x,a),ForwardDiff.Dual{Inner}(b,c))
leaves(value) = (ForwardDiff.value(ForwardDiff.value(value)),
    ForwardDiff.partials(ForwardDiff.value(value))[1],
    ForwardDiff.value(ForwardDiff.partials(value)[1]),
    ForwardDiff.partials(ForwardDiff.partials(value)[1])[1])
bits(value::Float32)=reinterpret(UInt32,value)
bits(value::Float64)=reinterpret(UInt64,value)

function reference(x::T,a::T,b::T,c::T) where {T<:AbstractFloat}
    return setprecision(BigFloat,1024) do
        X,A,B,C=BigFloat.((x,a,b,c))
        f,first,second=if iszero(X)
            (one(X),zero(X),-one(X)/3)
        else
            s,q=sin(X),cos(X)
            (s/X,(X*q-s)/X^2,(-X^2*s-2X*q+2s)/X^3)
        end
        T.((f,A*first,B*first,A*B*second+C*first))
    end
end

@testset "sinc range recovery retains original factors and primal bits" begin
    for T in (Float32,Float64)
        tiny,largest=nextfloat(zero(T)),floatmax(T)
        extreme=T===Float32 ? T(2e19) : T(1e154)
        cases=((zero(T),extreme,extreme,zero(T)),
            (T(0.1),extreme,extreme,zero(T)),
            (T(0.49),extreme,extreme,zero(T)),
            (zero(T),tiny,largest,zero(T)),
            (zero(T),tiny,T(3),zero(T)),
            (zero(T),tiny,one(T),zero(T)),
            (T(0.1),tiny,largest,zero(T)),
            (zero(T),zero(T),extreme,zero(T)),
            (T(0.1),T(0.25),T(0.375),T(-0.125)))
        for (x,a,b,c) in cases
            input=nested(x,a,b,c)
            expected=reference(x,a,b,c)
            @test all(isfinite,expected)
            actual=UTDKernels._sin_over_argument(input)
            @test actual isa typeof(input)
            coefficients=leaves(actual)
            @test all(isfinite,coefficients)
            @test bits(first(coefficients))==bits(UTDKernels._sin_over_argument(x))
            tolerance=T===Float32 ? 4e-6 : 2e-12
            for (coefficient,target) in zip(coefficients,expected)
                if iszero(target)
                    @test iszero(coefficient)
                elseif abs(target)<floatmin(T)
                    @test bits(coefficient)==bits(target)
                else
                    @test isapprox(coefficient,target;rtol=tolerance,atol=0)
                end
            end
        end
    end
end

@testset "sinc range recovery rejects unrepresentable or excessive work" begin
    context_precision=precision(BigFloat)
    for T in (Float32,Float64)
        largest=floatmax(T)
        unrepresentable=nested(zero(T),largest,largest,zero(T))
        @test_throws DomainError UTDKernels._sin_over_argument(unrepresentable)
        @test precision(BigFloat)==context_precision
    end
    lower=nested(0.0,nextfloat(0.0),floatmax(Float64),0.0)
    excessive=ForwardDiff.Dual{Third}(lower,zero(lower))
    @test_throws ArgumentError UTDKernels._sin_over_argument(excessive)
    @test precision(BigFloat)==context_precision
end

end
