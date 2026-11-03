module ZeroPartialCarrierRegressions

using Test, ForwardDiff, SpecialFunctions, UTDKernels

struct FlatCarrierTag end
struct InnerCarrierTag end
struct OuterCarrierTag end

primal(x::ForwardDiff.Dual)=primal(ForwardDiff.value(x))
primal(x::Complex)=complex(primal(real(x)),primal(imag(x)))
primal(x::Real)=x
finite(x::ForwardDiff.Dual)=finite(ForwardDiff.value(x)) && all(finite,ForwardDiff.partials(x))
finite(x::Complex)=finite(real(x)) && finite(imag(x))
finite(x::Real)=isfinite(x)
wide(x::Real)=BigFloat(x)
wide(x::Complex)=complex(wide(real(x)),wide(imag(x)))
function leaves(x)
    x isa ForwardDiff.Dual || return [x]
    result=leaves(ForwardDiff.value(x))
    for value in ForwardDiff.partials(x);append!(result,leaves(value));end
    result
end
function layer(x,tags,R)
    isempty(tags) && return typeof(x)===R
    x isa ForwardDiff.Dual || return false
    tag,n=first(tags)
    typeof(x).parameters[1]===tag && length(ForwardDiff.partials(x))==n &&
        layer(ForwardDiff.value(x),tags[2:end],R) &&
        all(value->layer(value,tags[2:end],R),ForwardDiff.partials(x))
end
complex_layer(x,tags,R)=x isa Complex && layer(real(x),tags,R) && layer(imag(x),tags,R)
first_partial(x)=complex(ForwardDiff.partials(real(x))[1],ForwardDiff.partials(imag(x))[1])
function within_roundoff(actual,expected,R,operations,scale)
    setprecision(BigFloat,512) do
        u=BigFloat(eps(R))
        abs(wide(actual)-expected)<=operations*u/(1-operations*u)*max(BigFloat(1),BigFloat(scale))
    end
end

# Filled from the already independently qualified Base-only series function;
# no production F, erfc or erfcx is used to construct this reference.
function n0_reference(x::Rational{BigInt};bits=512,n=256)
    n==256 && bits in (512,768) || error("Frozen series policy required")
    0<x<n+2 || throw(DomainError(x,"Positive bounded exact input required"))
    next_magnitude=x^(n+1)/(factorial(big(n+1))*big(2n+3))
    tail=2x*next_magnitude/(1-x/(n+2))
    setprecision(BigFloat,bits) do
        xx=BigFloat(numerator(x))/BigFloat(denominator(x))
        rationalize(BigInt,xx;tol=zero(xx))==x || error("Input not exact at declared precision")
        unit=complex(zero(BigFloat),one(BigFloat))
        term=one(unit);total=term;correction=zero(unit)
        for j in 0:n-1
            term*=(-unit*xx)*(2j+1)/((j+1)*(2j+3))
            y=term-correction;updated=total+y
            correction=(updated-total)-y;total=updated
        end
        z=sqrt(xx/2)*complex(one(BigFloat),one(BigFloat))
        erf=(2/sqrt(BigFloat(pi)))*z*total
        value=sqrt(BigFloat(pi)*xx/2)*complex(one(BigFloat),one(BigFloat))*
            exp(unit*xx)*(one(unit)-erf)
        first=(unit+inv(2xx))*value-unit
        second=-value/(2xx^2)+(unit+inv(2xx))*first
        upper=setrounding(BigFloat,RoundUp) do
            BigFloat(numerator(tail),RoundUp;precision=bits)/
                BigFloat(denominator(tail),RoundDown;precision=bits)
        end
        coefficient_bound=1+inv(2x)
        (;x,bits,n,z,value,first,second,tail_exact=tail,tail_upper=upper,
            first_tail_exact=coefficient_bound*tail,
            second_tail_exact=(inv(2x^2)+coefficient_bound^2)*tail)
    end
end

@testset "Zero-partial carriers and unchanged nonzero derivatives" begin
    reference512=n0_reference(big(8)//big(1);bits=512)
    reference768=n0_reference(big(8)//big(1);bits=768)
    @test reference512.z==complex(BigFloat(2),BigFloat(2))
    @test reference512.tail_upper<=BigFloat(2)^(-700)
    for name in (:value,:first,:second)
        @test abs(getproperty(reference512,name)-getproperty(reference768,name))<=BigFloat(2)^(-400)
    end
    k0=setprecision(BigFloat,512) do
        exp(complex(BigFloat(0),BigFloat(pi)/4))/sqrt(BigFloat(pi))
    end
    erf_derivative=setprecision(BigFloat,512) do
        -BigFloat(2)/sqrt(BigFloat(pi))*complex(BigFloat(1),BigFloat(2))
    end
    functions=((:erfc,SpecialFunctions.erfc),(:erfcx,SpecialFunctions.erfcx),
        (:switch,UTDKernels.shadow_switch),(:kernel,UTDKernels.shadow_sensitivity_kernel),
        (:F,UTDKernels.F_utd),(:Fprime,UTDKernels.F_utd_prime))
    for T in (Float32,Float64),(kind,f) in functions
        point=kind in (:F,:Fprime) ? T(8) : zero(T)
        iscomplex=kind in (:erfc,:erfcx)
        scalar=f(iscomplex ? complex(point,zero(T)) : point)
        R=typeof(real(scalar))
        @test finite(scalar)
        for n in (0,1)
            @testset "$T/$kind/N$n" begin
                real_input=n==0 ? ForwardDiff.Dual{FlatCarrierTag}(point) :
                    ForwardDiff.Dual{FlatCarrierTag}(point,one(T))
                imag_input=n==0 ? ForwardDiff.Dual{FlatCarrierTag}(zero(T)) :
                    ForwardDiff.Dual{FlatCarrierTag}(zero(T),T(2))
                input=iscomplex ? complex(real_input,imag_input) : real_input
                result=f(input)
                shape=complex_layer(result,[(FlatCarrierTag,n)],R)
                @test shape
                @test finite(result)
                @test isequal(primal(result),scalar)
                if kind in (:erfc,:erfcx)
                    @test primal(result)==complex(one(R),zero(R))
                elseif kind==:switch
                    @test primal(result)==complex(R(1)/2,zero(R))
                elseif kind==:kernel
                    @test within_roundoff(primal(result),k0,R,32,abs(k0))
                end
                if n==1 && shape
                    expected,operations,scale=setprecision(BigFloat,512) do
                        if kind in (:erfc,:erfcx)
                            (erf_derivative,32,abs(erf_derivative))
                        elseif kind==:switch
                            (k0,32,abs(k0))
                        elseif kind==:kernel
                            (zero(k0),0,zero(BigFloat))
                        elseif kind==:F
                            (reference512.first,64,abs((im+BigFloat(1)/16)*reference512.value)+1)
                        else
                            (reference512.second,64,abs(reference512.value/128)+abs((im+BigFloat(1)/16)*reference512.first))
                        end
                    end
                    actual=first_partial(result)
                    @test operations==0 ? iszero(actual) : within_roundoff(actual,expected,R,operations,scale)
                end
            end
        end
    end
    for T in (Float32,Float64),(kind,f) in ((:switch,UTDKernels.shadow_switch),(:kernel,UTDKernels.shadow_sensitivity_kernel))
        @testset "$T/$kind/outer0_inner1" begin
            scalar=f(zero(T));R=typeof(real(scalar))
            input=ForwardDiff.Dual{InnerCarrierTag}(zero(T),one(T))
            inner=f(input)
            result=f(ForwardDiff.Dual{OuterCarrierTag}(input))
            shape=complex_layer(result,[(OuterCarrierTag,0),(InnerCarrierTag,1)],R)
            @test shape
            @test finite(result)
            @test isequal(primal(result),scalar)
            @test isequal(ForwardDiff.value(real(result)),real(inner))
            @test isequal(ForwardDiff.value(imag(result)),imag(inner))
        end
    end
end

@testset "Rounded work projections retain empty and nested carriers" begin
    caller_precision=precision(BigFloat)
    setprecision(BigFloat,128) do
        for T in (Float32,Float64,BigFloat)
            value=T===BigFloat ? BigFloat("0.3";precision=256) : T(0.3)
            seed=T===BigFloat ? BigFloat(1;precision=256) : one(T)
            expected_value=T===BigFloat ? BigFloat(value,RoundNearest;precision=128) : value
            expected_seed=T===BigFloat ? BigFloat(seed,RoundNearest;precision=128) : seed
            zero_part=ForwardDiff.Dual{InnerCarrierTag}(value)
            one_part=ForwardDiff.Dual{InnerCarrierTag}(value,seed)
            cases=((:N0,zero_part,[expected_value]),
                (:N1,one_part,[expected_value,expected_seed]),
                (:outer0_inner1,ForwardDiff.Dual{OuterCarrierTag}(one_part),[expected_value,expected_seed]),
                (:outer1_inner0,ForwardDiff.Dual{OuterCarrierTag}(zero_part,ForwardDiff.Dual{InnerCarrierTag}(seed)),[expected_value,expected_seed]),
                (:outer0_inner0,ForwardDiff.Dual{OuterCarrierTag}(zero_part),[expected_value]))
            for (name,input,expected) in cases
                @testset "$T/$name" begin
                    result=UTDKernels._shadow_round_work_value(input,128)
                    actual=leaves(result)
                    @test typeof(result)===typeof(input)
                    @test length(actual)==length(expected) && all(isequal.(actual,expected))
                    @test all(precision(x)==(T===BigFloat ? 128 : precision(T)) for x in actual)
                    @test precision(BigFloat)==128
                end
            end
        end
        @testset "Actual public BigFloat zero-partial signed angle" begin
            x=BigFloat("0.3");bearing=(BigFloat("0.5"),BigFloat("0.8"))
            expected=setprecision(BigFloat,512) do
                atan(BigFloat(bearing[2]),BigFloat(bearing[1]))-atan(BigFloat(x))
            end
            input=ForwardDiff.Dual{FlatCarrierTag}(x)
            result=UTDKernels.FaceEdge.signed_angle2((one(input),input),bearing)
            @test typeof(result)===typeof(input)
            @test isempty(ForwardDiff.partials(result))
            @test precision(ForwardDiff.value(result))==128
            @test abs(ForwardDiff.value(result)-expected)<=8eps(BigFloat)
        end
    end
    @test precision(BigFloat)==caller_precision
end

end
