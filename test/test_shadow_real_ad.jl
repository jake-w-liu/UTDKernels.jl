module RealShadowADTests
using Test,ForwardDiff,UTDKernels
const SS=UTDKernels.ShadowSensitivity

Base.@constprop :none @noinline _switch_probe(q)=shadow_switch(q)
Base.@constprop :none @noinline _kernel_probe(q)=shadow_sensitivity_kernel(q)
function _warmed_allocations(f,q)
    f(q);f(q)
    return @allocated f(q)
end

@testset "First-order real shadow AD" begin
    for V in (Float32,Float64),q in V.((0.,.1,2.,8.,10.,1e10)),seed in (one(V),floatmax(V),nextfloat(zero(V)))
        x=ForwardDiff.Dual(q,seed)
        actual=@inferred shadow_switch(x)
        reference=invoke(shadow_switch,Tuple{Number},x)
        @test UTDKernels._number_isfinite(actual)
        da=complex(ForwardDiff.partials(real(actual),1),ForwardDiff.partials(imag(actual),1))
        dr=complex(ForwardDiff.partials(real(reference),1),ForwardDiff.partials(imag(reference),1))
        @test abs(da-dr)<=4eps(V)*max(abs(dr),floatmin(V))
    end
    for V in (Float32,Float64),q in (zero(V),nextfloat(zero(V)),V(.1),V(2),V(10),V(1e10)),seed in (one(V),floatmax(V),nextfloat(zero(V)))
        x=ForwardDiff.Dual(q,seed)
        reference=try
            invoke(shadow_sensitivity_kernel,Tuple{Number},x)
        catch error
            error isa DomainError || rethrow()
            nothing
        end
        if reference===nothing
            @test_throws DomainError shadow_sensitivity_kernel(x)
        else
            actual=@inferred shadow_sensitivity_kernel(x)
            da=complex(ForwardDiff.partials(real(actual),1),ForwardDiff.partials(imag(actual),1))
            dr=complex(ForwardDiff.partials(real(reference),1),ForwardDiff.partials(imag(reference),1))
            @test abs(da-dr)<=4eps(V)*max(abs(dr),floatmin(V))
        end
    end
    for f in (shadow_switch,shadow_sensitivity_kernel)
        @test_throws DomainError f(ForwardDiff.Dual(1.,Inf))
        @test_throws DomainError f(ForwardDiff.Dual(NaN,1.))
        q=ForwardDiff.Dual{Nothing}(2.,ForwardDiff.Partials((.2,-.7,0.)))
        a=@inferred f(q)
        r=invoke(f,Tuple{Number},q)
        for j in 1:3
            @test complex(ForwardDiff.partials(real(a),j),ForwardDiff.partials(imag(a),j)) ≈
                  complex(ForwardDiff.partials(real(r),j),ForwardDiff.partials(imag(r),j)) rtol=1e-14
        end
    end
    q=2.
    second=ForwardDiff.derivative(x -> ForwardDiff.derivative(y -> real(shadow_switch(y)),x),q)
    @test second ≈ real(-2im*q*shadow_sensitivity_kernel(q)) rtol=2e-14
    z=.4+.2im
    jac=ForwardDiff.jacobian([real(z),imag(z)]) do xy
        value=shadow_switch(complex(xy...))
        [real(value),imag(value)]
    end
    kernel=shadow_sensitivity_kernel(z)
    @test jac ≈ [real(kernel) -imag(kernel);imag(kernel) real(kernel)] rtol=1e-13 atol=3e-15
    @test _warmed_allocations(_switch_probe,ForwardDiff.Dual(2.,1.))==0
    @test _warmed_allocations(_kernel_probe,ForwardDiff.Dual(2.,1.))==0
end

function _nested_derivative(f,x,order)
    order==0 && return f(x)
    return ForwardDiff.derivative(y -> _nested_derivative(f,y,order-1),x)
end

@testset "Analytic zero-coordinate actions" begin
    @test ForwardDiff.derivative(g -> SS.curved_inverse(.4,g),0.) ≈ -.08 rtol=2e-14
    @test SS.curved_inverse(floatmax(Float64),0.)==floatmax(Float64)
    @test SS.curved_inverse(nextfloat(0.),0.)==nextfloat(0.)
    for kappa in (2.,8.,1e4),sigma in (.2,.8)
        f(nu)=SS.translated_gaussian_action(nu,kappa,sigma)
        amplitude=sqrt(2kappa*sigma^2/(1+2kappa*sigma^2))
        b=sigma^2/(2(1+2kappa*sigma^2))
        @test ForwardDiff.derivative(f,0.)==0
        @test _nested_derivative(f,0.,2) ≈ -2b*amplitude rtol=3e-14
        @test _nested_derivative(f,0.,4) ≈ 12b^2*amplitude rtol=3e-13
        @test ForwardDiff.derivative(f,.3) ≈ -2b*.3*amplitude*exp(-b*.3^2) rtol=3e-14
    end
    @test_throws DomainError SS.translated_gaussian_action(ForwardDiff.Dual(0.,Inf),8.,.8)
end

@testset "Public receiver panel policy" begin
    @test SS.default_receiver_panels(4.)==1600
    @test SS.default_receiver_panels(100.)==4000
    @test SS.default_receiver_panels(10_000.)==40_000
    @test SS.DEFAULT_MAX_PANELS==10_000_000
    @test_throws ArgumentError SS.default_receiver_panels(0.)
    @test_throws ArgumentError SS.default_receiver_panels(Inf)
    @test_throws ArgumentError SS.default_receiver_panels(1e33)
end

_shadow_leaf_coefficients(x::Real)=[x]
_shadow_leaf_coefficients(x::ForwardDiff.Dual)=vcat(_shadow_leaf_coefficients(ForwardDiff.value(x)),(_shadow_leaf_coefficients(p) for p in ForwardDiff.partials(x))...)
_shadow_leaf_coefficients(x::Complex)=vcat(_shadow_leaf_coefficients(real(x)),_shadow_leaf_coefficients(imag(x)))
@testset "Nested real shadow AD cancellation and range" begin
for R in (Float32,Float64),q0 in (.3,2.,4.,-1.5,9.),sgn in (-1,1),
    delta0 in (0.,1e-10,1e-7,1e-4,.001,.002,.004,.1,.3,.6),
    scale0 in (R===Float32 ? (1e-15,1.,1e15) : (1e-150,1.,1e150)),
    f in (UTDKernels.shadow_switch,UTDKernels.shadow_sensitivity_kernel)
    q=R(q0);delta=R(delta0);scale=R(scale0)
    K=UTDKernels.shadow_sensitivity_kernel(q)
    first=f===UTDKernels.shadow_switch ? K : -2im*q*K
    second=f===UTDKernels.shadow_switch ? -2im*q*K : (-2im-4q*q)*K
    a=scale;b=R(sgn)*scale
    c=-real(second)/real(first)*(a*b)*(1+delta)
    x=ForwardDiff.Dual{:outer}(ForwardDiff.Dual{:inner}(q,a),ForwardDiff.Dual{:inner}(b,c))
    actual=_shadow_leaf_coefficients(f(x));reference=_shadow_leaf_coefficients(invoke(f,Tuple{Number},x))
    tolerance=R===Float32 ? 2e-5 : 3e-12
    floor=R===Float32 ? 2e-6 : 2e-13
    @test all(isapprox.(actual,reference;rtol=tolerance,atol=floor))
    # Tiny coefficients must not become false zeros, even when an absolute
    # tolerance would mask the loss.
    @test all(iszero(r) || !iszero(a) for (a,r) in zip(actual,reference))
end
end

@testset "Nested real shadow derivative invariants" begin
    for q in (0.,.2,-1.,3.),n in 1:4
        k=exp(im*pi/4-im*q*q)/sqrt(pi)
        polynomials=ComplexF64[1.,-2im*q]
        for m in 1:2
            push!(polynomials,-2im*q*polynomials[end]-2im*m*polynomials[end-1])
        end
        @test _nested_derivative(shadow_switch,q,n)≈polynomials[n]*k rtol=3e-12 atol=2e-12
    end
    for seed in (nextfloat(0.),1e-300,1e300)
        inner=ForwardDiff.Dual{:tiny_inner}(0.,seed)
        x=ForwardDiff.Dual{:tiny_outer}(inner,ForwardDiff.Dual{:tiny_inner}(seed,1.))
        for f in (shadow_switch,shadow_sensitivity_kernel)
            result=try
                _shadow_leaf_coefficients(f(x))
            catch e
                e isa DomainError || rethrow()
                nothing
            end
            reference=try
                _shadow_leaf_coefficients(invoke(f,Tuple{Number},x))
            catch e
                e isa DomainError || rethrow()
                nothing
            end
            if reference===nothing
                @test result===nothing
            else
                @test result!==nothing && all(isapprox.(result,reference;rtol=3e-12,atol=2e-13))
            end
        end
    end
    inner=ForwardDiff.Dual{:multi_inner}(.7,1.,-.3)
    x=ForwardDiff.Dual{:multi_outer}(inner,
        ForwardDiff.Dual{:multi_inner}(.4,.2,-.1),ForwardDiff.Dual{:multi_inner}(-.2,.1,.5))
    for f in (shadow_switch,shadow_sensitivity_kernel)
        @test all(isapprox.(_shadow_leaf_coefficients(f(x)),_shadow_leaf_coefficients(invoke(f,Tuple{Number},x));rtol=3e-12,atol=2e-13))
        @test typeof(@inferred f(x))==typeof(invoke(f,Tuple{Number},x))
    end
    z=.4+.2im
    direction=1+im
    @test _nested_derivative(t->real(shadow_switch(z+t*direction)),0.,2)≈
        real(-2im*z*shadow_sensitivity_kernel(z)*direction^2) rtol=3e-13 atol=3e-14
end
end
