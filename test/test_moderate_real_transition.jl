module ModerateRealTransitionTests
using Test, UTDKernels, ForwardDiff
include("support/face_edge_oracle.jl")

relative(a,b) = abs(a-b)/abs(b)
differentiate(f,x) = complex(ForwardDiff.derivative(t->real(f(t)),x),
                             ForwardDiff.derivative(t->imag(f(t)),x))
function oracle(x)
    setprecision(BigFloat,512) do
        xb = BigFloat(x)
        f = face_edge_F_hp(x;digits=120)
        r = im+inv(2xb)
        fp = r*f-im
        fpp = r*fp-f/(2xb^2)
        fppp = r*fpp-fp/xb^2+f/xb^3
        return f,fp,fpp,fppp
    end
end

@testset "Moderate real transition derivative accuracy" begin
    x0 = 19.098300562505262
    # The HP oracle rebuilds the convergent erf series; it does not use the
    # production quadrature, binary64 erfcx, or binary64 differential equation.
    for x in (16.0,nextfloat(16.0),x0,20.0,nextfloat(20.0),25.5,30.0,prevfloat(35.0))
        f,fp,fpp,fppp = oracle(x)
        value,first,second,third = UTDKernels._F_utd_derivatives3(x)
        @test relative(F_utd_prime(x),fp) < 2e-14
        @test relative(first,fp) < 2e-14
        @test relative(value,f) < 2e-12
        @test relative(second,fpp) < 2e-12
        @test relative(third,fppp) < 2e-11
        @test relative(differentiate(F_utd,x),fp) < 2e-14
        @test relative(differentiate(t->differentiate(F_utd,t),x),fpp) < 2e-12
        @test relative(differentiate(F_utd_prime,x),fpp) < 2e-12
        @test relative(differentiate(t->differentiate(F_utd_prime,t),x),fppp) < 2e-11
    end
    # Neighbor arguments also cover the calibrated interval endpoints.
    for x in (prevfloat(16.0),35.0,nextfloat(35.0))
        _,fp,fpp,fppp = oracle(x)
        _,first,second,third = UTDKernels._F_utd_derivatives3(x)
        @test relative(F_utd_prime(x),fp) < 1e-10
        @test relative(first,fp) < 1e-10
        @test relative(second,fpp) < 5e-10
        @test relative(third,fppp) < 5e-9
    end
    @test F_utd_prime(20.0f0) isa ComplexF64
    @test F_utd_prime(20) isa ComplexF64
    @test F_utd_prime(20//1) isa ComplexF64
    @test_throws ArgumentError F_utd_prime(big"20.0")
    for x in (20,20//1)
        @test relative(F_utd_prime(x),oracle(x)[2]) < 5e-12
    end
end

@testset "Real derivative paths reject nonfinite AD seeds" begin
    for x in (15.0,20.0,35.0,100.0), seed in (Inf,NaN)
        ordinary = ForwardDiff.Dual(x,seed)
        nested = ForwardDiff.Dual(ForwardDiff.Dual(x,1.0),ForwardDiff.Dual(seed,0.0))
        for input in (ordinary,nested),
            evaluator in (F_utd,F_utd_prime,UTDKernels._F_utd_derivatives3)
            @test_throws DomainError evaluator(input)
        end
    end
    # F's positive-infinity value contract remains distinct from the finite
    # positive domain of its derivative and derivative bundle.
    endpoint = F_utd(ForwardDiff.Dual(Inf,1.0))
    @test ForwardDiff.value(real(endpoint)) == 1
    @test iszero(ForwardDiff.partials(real(endpoint))[1])
end

@testset "Exact-input face-edge residual fixture provenance" begin
    F = UTDKernels.FaceEdge
    cases = (
        (F.BoundaryGeometry(1e-12,0.8pi,100.0,1.0),
         0x3d719799812dea11,0x40041b2f769cf0e0,
         -4.931717647160173e-13+4.264297696313514e-13im,
         -4.931717647159758e-13+4.2642976963138624e-13im,4e-14),
        (F.BoundaryGeometry(1e-8,0.5pi,20.0,1.0),
         0x3e45798ee2308c3a,0x3ff921fb54442d18,
         -2.0682164625107836e-9+1.8735683461215226e-9im,
         -2.0682164625107207e-9+1.873568346121566e-9im,2e-14))
    for (g,epsilon_bits,delta_bits,corrected,old,old_error_floor) in cases
        @test reinterpret(UInt64,g.eps) == epsilon_bits
        @test reinterpret(UInt64,g.Delta) == delta_bits
        ref80 = face_edge_incident_hp(g.eps,g.Delta,g.k,g.L;digits=80)
        ref120 = face_edge_incident_hp(g.eps,g.Delta,g.k,g.L;digits=120)
        @test relative(ref80,ref120) < big"1e-70"
        @test relative(corrected,ref120) < 2eps(Float64)
        @test corrected == ComplexF64(ref120)
        @test relative(old,ref120) > old_error_floor
        @test relative(F.edge_residual_compensated(g),ref120) < 2e-14
    end
end
end
