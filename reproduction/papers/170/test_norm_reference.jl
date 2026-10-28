using Test,ForwardDiff
include("norm_reference.jl")
using .Paper170NormReference
derivative(f,x,n)=n==0 ? f(x) : ForwardDiff.derivative(y->derivative(f,y,n-1),x)

@testset "independent derivative-norm control" begin
    r=1.2;gamma=.6
    bump(s)=exp(1-inv(1-(s/r)^2))
    for s in (-1.1,-.6,0.,.5,1.05)
        expected=Paper170NormReference.bump_derivatives(s,r)
        for n in 1:3
            @test expected[n] ≈ derivative(bump,s,n) rtol=2e-12 atol=1e-10
        end
        y=s+gamma*s*s/2
        inverse(y)=2y/(1+sqrt(1+2gamma*y))
        composed(y)=bump(inverse(y))
        second,third,_=Paper170NormReference.transformed_derivatives(s,r,gamma,1.)
        @test second ≈ derivative(composed,y,2) rtol=3e-11 atol=1e-9
        @test third ≈ derivative(composed,y,3) rtol=3e-11 atol=1e-8
    end
    base=sobolev_control(r,gamma;precision=128)
    scaled=sobolev_control(r,gamma;scale=.2,precision=128)
    @test scaled.moment_upper ≈ base.moment_upper/BigFloat(.2)^2 rtol=big"1e-20"
    physical=sobolev_control(r;precision=128)
    @test physical.moment_upper < 26
    @test_throws ArgumentError sobolev_control(r,1.)
end
