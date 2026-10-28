module CanonicalNullBaselineTests
using Test,ForwardDiff,UTDKernels
const NU=UTDKernels.NullUniform
const RT=4e-12
@testset "classical divided difference and continuous jet" begin
 for k in (1e-300,nextfloat(0.),1.,1e16,1e308), z in (.5+.3im,2.0+.8im,-.2+.4im)
  expected=NU.faddeeva_moments(z,1)[2]/sqrt(k)
  @test isapprox(NU.classical_pole_term(k,z,t->t),expected;rtol=RT,atol=0.)
  @test isapprox(NU.linearized_pole_term(k,z,0.,1.),expected;rtol=RT,atol=0.)
 end
 for k in (1.,1e16,1e30), z in (.5+.3im,-2.0+.8im,1e-12+.2e-12im)
  c=(.2+.3im,-.7+.2im,.3-.5im)
  a=t->c[1]+c[2]*t+c[3]*t*t
  w=NU.faddeeva_moments(z,2)
  expected=c[1]*w[1]+c[2]*w[2]/sqrt(k)+c[3]*w[3]/k
  @test isapprox(NU.classical_pole_term(k,z,a),expected;rtol=RT,atol=1e-30)
 end
 @test NU.classical_pole_term(1.,0.,t->im+t)==im*(1-1/sqrt(pi))
 @test ForwardDiff.derivative(x->real(NU.classical_pole_term(1.,x,t->t+t*t)),0.)≈1.
 @test ForwardDiff.derivative(x->imag(NU.classical_pole_term(1.,x,t->t+t*t)),0.)≈-1/sqrt(pi)
 for degree in 1:4
  a=t->sum(t^j for j in 0:degree)
  # At k=1 the pole extraction is a(z)w(z)-i*sum(z^(j-1))/sqrt(pi).
  ref=x->a(x)*NU.faddeeva_moments(complex(x),0)[1]-im*sum(x^(j-1) for j in 1:degree)/sqrt(pi)
  for part in (real,imag)
   @test ForwardDiff.derivative(x->part(NU.classical_pole_term(1.,x,a)),0.)≈ForwardDiff.derivative(x->part(ref(x)),0.)
   @test ForwardDiff.derivative(x->ForwardDiff.derivative(y->part(NU.classical_pole_term(1.,y,a)),x),0.)≈ForwardDiff.derivative(x->ForwardDiff.derivative(y->part(ref(y)),x),0.)
  end
 end
 @test NU.classical_pole_term(1.,0.,t->inv(1-1000t))≈1-1000im/sqrt(pi)
 for a in (one,t->im*one(t)), k in (1.,1e-300,nextfloat(0.)), z in (0.,.5+.3im)
  @test NU.classical_pole_term(k,z,a)≈a(z)*NU.faddeeva_moments(complex(z),0)[1]
 end
 for a in (t->1.,t->1im,one), k in (1e-300,1e-200)
  @test ForwardDiff.derivative(x->real(NU.classical_pole_term(x,.5+.3im,a)),k)==0
  @test ForwardDiff.derivative(x->imag(NU.classical_pole_term(x,.5+.3im,a)),k)==0
 end
 for k in (1e-300,nextfloat(0.))
  @test ForwardDiff.derivative(x->real(NU.linearized_pole_term(x,.5+.3im,1.,0.)),k)==0
  @test ForwardDiff.derivative(x->imag(NU.linearized_pole_term(x,.5+.3im,1.,0.)),k)==0
 end
 # A zero primal coefficient with a live seed must not take the zero-slope route.
 @test ForwardDiff.derivative(b->real(NU.linearized_pole_term(1.,.5+.3im,1.,b)),0.)≈real(NU.faddeeva_moments(.5+.3im,1)[2])
 @test ForwardDiff.derivative(b->imag(NU.linearized_pole_term(1.,.5+.3im,1.,b)),0.)≈imag(NU.faddeeva_moments(.5+.3im,1)[2])
 @test_throws ArgumentError NU.classical_pole_term(1.,0.,t->ComplexF64((1+im)*(1+t)))
 @test_throws ArgumentError NU.classical_pole_term(1.,0.,t->ComplexF64(im+t))
 @test_throws ArgumentError NU.classical_pole_term(1.,0.,t->real(t))
end
@testset "separated scaled denominator and singular jet" begin
 k=nextfloat(0.);z=1e200+.3im
 v=NU.slope_separated_term(k,z,0.,1.)
 expected=setprecision(BigFloat,256) do
  ComplexF64(im/(2sqrt(BigFloat(pi))*sqrt(BigFloat(k))*Complex{BigFloat}(z)^2))
 end
 @test isapprox(v,expected;rtol=RT,atol=0.)
 @test_throws ArgumentError ForwardDiff.derivative(x->real(NU.slope_separated_term(1.,x,0.,1.)),0.)
 for k in (1e-300,nextfloat(0.))
  @test ForwardDiff.derivative(x->real(NU.slope_separated_term(x,.5+.3im,1.,0.)),k)==0
 end
 @test ForwardDiff.derivative(b->imag(NU.slope_separated_term(1.,.5+.3im,0.,b)),0.)≈imag(im/(2sqrt(pi)*(.5+.3im)^2))
end

@testset "Canonical comparator held-out boundaries" begin
    for scale in (1e200,1e308)
        @test NU.classical_pole_term(1.,0.,t->inv(one(t)-scale*t))≈1-im*scale/sqrt(pi) rtol=4e-12
        @test NU.classical_pole_term(1.,0.,t->inv(one(t)-im*scale*t))≈1+scale/sqrt(pi) rtol=4e-12
    end
    @test NU.classical_pole_term(1.,0.,t->t^3)==0
    @test NU.classical_pole_term(1.,0.,t->im*t+t^3)≈inv(sqrt(pi)) rtol=4e-12
    @test NU.classical_pole_term(1.,0.,t->t^5)==0
    @test NU.classical_pole_term(1.,0.,t->(1+im)*one(t))==1+im
    for a in (t->1.,t->1im,t->ComplexF64((1+im)*(1+.01t)))
        @test_throws ArgumentError NU.classical_pole_term(1.,0.,a)
        @test isfinite(NU.classical_pole_term(1.,.5+.3im,a))
    end
    for k in (Float32(1),floatmax(Float32))
        z=ComplexF32(.5,.3)
        @test NU.classical_pole_term(k,z,t->t) isa ComplexF32
        @test NU.linearized_pole_term(k,z,0f0,1f0) isa ComplexF32
        @test NU.slope_separated_term(k,z,0f0,1f0) isa ComplexF32
        @test NU.linearized_pole_term(k,z,0f0,1f0)≈NU.faddeeva_moments(z,1)[2]/sqrt(k) rtol=3e-5
    end
    z=.5+.3im;k=1e-300
    expected=NU.faddeeva_moments(z,1)[2]/sqrt(k)
    actual=ForwardDiff.derivative(x->NU.linearized_pole_term(x,z,1.,x-k),k)
    @test actual≈expected rtol=4e-12
    @test NU.classical_pole_term(1e308,nextfloat(0.),t->t)≈-im/(sqrt(pi)*1e154) rtol=4e-12
    for radius in (15.,40.,1e3,1e20,1e100)
        z=radius+.3im;expected=NU.faddeeva_moments(z,1)[2]
        for actual in (NU.classical_pole_term(1.,z,t->t),NU.linearized_pole_term(1.,z,0.,1.))
            @test all(isapprox(part(actual),part(expected);rtol=4e-12,atol=0) for part in (real,imag))
        end
        @test ForwardDiff.derivative(b->imag(NU.linearized_pole_term(1.,z,1.,b)),0.)≈imag(expected) rtol=4e-12 atol=0
    end
    k=nextfloat(0.);z=.5+.3im
    scaled_reference=((NU.faddeeva_moments(z,1)[2]/sqrt(k))/1e200)/1e200
    @test NU.classical_pole_term(k,z,t->(t/1e200)/1e200)≈scaled_reference rtol=4e-12 atol=0
end

@testset "Full-amplitude and completed-square range" begin
    z=1e100+.3im
    expected=NU.faddeeva_moments(z,1)[2]*1e300
    @test NU.classical_pole_term(1.,z,t->1e300*t)≈expected rtol=4e-12 atol=0
    for beta in (1e154,6e155)
        k=1e308;z=.8+.35im;lambda=.3
        shift,scale=setprecision(BigFloat,256) do
            kw=BigFloat(k);sw=BigFloat(beta)/(2sqrt(kw))
            Float64(sw),Float64(exp(sw*sw-log(kw)/2))
        end
        # Independent Hermite integration of the shifted defining integral,
        # without evaluating either Faddeeva moment in the closed form.
        expected=scale*NU.canonical_quadrature(1.,z-shift,t->t+shift+lambda;n=2000)
        @test NU.nonpolynomial_closed_form(k,z,lambda,beta)≈expected rtol=4e-12 atol=0
    end
    f(k)=real(NU.nonpolynomial_amplitude(k,0.,0.)(.5))
    @test ForwardDiff.derivative(f,1e-300)==0
    @test ForwardDiff.derivative(x->ForwardDiff.derivative(f,x),1e-300)==0
    @test_throws ArgumentError NU.nonpolynomial_amplitude(1.,0.,ForwardDiff.Dual(0.,Inf))
    @test_throws ArgumentError NU.nonpolynomial_closed_form(1.,.5+.3im,0.,ForwardDiff.Dual(0.,Inf))
    @test_throws DomainError NU.nonpolynomial_closed_form(1.,.5+.3im,0.,1e308)
    k=400.;z=.8+.35im;lambda=.3;beta=.7
    amplitude=NU.nonpolynomial_amplitude(k,lambda,beta)
    beta_reference=NU.canonical_quadrature(k,z,t->t*amplitude(t);n=2000)
    beta_second_reference=NU.canonical_quadrature(k,z,t->t*t*amplitude(t);n=2000)
    k_reference=NU.canonical_quadrature(k,z,t->-amplitude(t)*(1+beta*t)/(2k);n=2000)
    @test ForwardDiff.derivative(b->NU.nonpolynomial_closed_form(k,z,lambda,b),beta)≈beta_reference rtol=4e-11 atol=1e-15
    @test ForwardDiff.derivative(x->ForwardDiff.derivative(b->NU.nonpolynomial_closed_form(k,z,lambda,b),x),beta)≈beta_second_reference rtol=4e-10 atol=1e-15
    @test ForwardDiff.derivative(kk->NU.nonpolynomial_closed_form(kk,z,lambda,beta),k)≈k_reference rtol=4e-11 atol=1e-15
    z=1e300+.3im;A=1e300;c=1e-300
    aq=A/(c+z)
    inverse_reference=aq*NU.faddeeva_moments(z,0)[1]+im*A/(sqrt(pi)*(c*(c+z)))
    inverse_actual=NU.classical_pole_term(1.,z,t->A/(c+t))
    @test all(isapprox(part(inverse_actual),part(inverse_reference);rtol=4e-12,atol=0) for part in (real,imag))
end

@testset "Every-coefficient precision stabilization" begin
    UK=UTDKernels
    a=ForwardDiff.Dual(1.,1.);b=ForwardDiff.Dual(1.,2.)
    @test !UK._number_isequal_all(a,b)
    @test !UK._number_isequal_all(complex(a),complex(b))
    @test !UK._number_isequal_all([a],[b])
    @test !UK._number_isequal_all((x=a,),(x=b,))
    C=typeof(a);calls=Int[]
    @test_throws ArgumentError UK._shadow_stable_wide_value(C,192,"unresolved derivative probe") do
        bits=precision(BigFloat);push!(calls,bits)
        ForwardDiff.Dual(BigFloat(1),BigFloat(1)+inv(BigFloat(bits)))
    end
    @test calls==[192,384,8192]
    calls=Int[]
    stable=UK._shadow_stable_wide_value(C,192,"resolved derivative probe") do
        bits=precision(BigFloat);push!(calls,bits)
        ForwardDiff.Dual(BigFloat(1),bits<8192 ? BigFloat(2) : BigFloat(3))
    end
    @test calls==[192,384]
    @test ForwardDiff.partials(stable)[1]==2
end
end
