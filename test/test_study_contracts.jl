module CanonicalStudyContractTests
using Test, ForwardDiff
using UTDKernels
const F = UTDKernels.FaceEdge
const C = UTDKernels.CurvatureMeasure
const N = UTDKernels.NullUniform
const S = UTDKernels.ShadowSensitivity

fd(f,x;h=2e-5) = (f(x-2h)-8f(x-h)+8f(x+h)-f(x+2h))/(12h)

@testset "Canonical study contracts" begin
    @testset "Promoted geometry and AD" begin
        @test F.BoundaryGeometry(1f-3,0f0,2f0,1f0).eps isa Float32
        for f in (e -> real(F.edge_residual_compensated(F.BoundaryGeometry(e,0.2,3.0,1.0))),
                  e -> imag(F.edge_residual_compensated(F.BoundaryGeometry(e,0.2,3.0,1.0))))
            @test ForwardDiff.derivative(f,0.01) ≈ fd(f,0.01;h=1e-5) rtol=2e-7
        end
        for f in (r -> real(C.MonostaticAssembly.monostatic_even_field(
                       C.MonostaticAssembly.regular_polygon_points(12,r),(3.0,0.1),2.0).field),
                  k -> imag(C.MonostaticAssembly.monostatic_even_field(
                       C.MonostaticAssembly.regular_polygon_points(12,1.0),(3.0,0.1),k).field))
            @test ForwardDiff.derivative(f,1.0) ≈ fd(f,1.0) rtol=2e-6 atol=1e-9
        end
        f(k) = real(N.hierarchy_value(k,0.6+0.7im,N.factored_null_coefficients(k,0.3)))
        @test ForwardDiff.derivative(f,10.0) ≈ fd(f,10.0) rtol=2e-7
        for q in (2.,50.,1e3), phi in (.4,2.7,pi,4.7,5.9), pol in (:soft,:hard), saddle in (0.,-2pi)
            model=N.wedge_canonical_model(q,phi,pi/18,2.,pol;saddle=saddle,ncoef=14)
            reference=N.wedge_cauchy_coefficients(q,phi,pi/18,2.,pol;saddle=saddle,ncoef=14)
            @test maximum(abs.(model.coefficients-reference)) < 2e-12*max(1.,maximum(abs,reference))
        end
        fmodel(q) = real(N.wedge_diffracted_model(q,2.7,pi/18,2.0,:hard;ncoef=4,order=1))
        @test ForwardDiff.derivative(fmodel,50.0) ≈ fd(fmodel,50.0;h=1e-3) rtol=2e-5 atol=1e-8
    end
    @testset "Exact discrete harmonic periodicity" begin
        for N in (8,17),m in (-3,0,3),shift in (N*2^45,BigInt(N)*2^80)
            @test C.polygon_measure_sum(N,m,2.,.3)==C.polygon_measure_sum(N,m+shift,2.,.3)
            @test C.polygon_alias_sum(N,m,2.,.3)==C.polygon_alias_sum(N,m+shift,2.,.3)
            @test C.physical_seam_sum(N,m,2.,.3,4.,1.)==C.physical_seam_sum(N,m+shift,2.,.3,4.,1.)
        end
        for N in (8,17),m in (typemin(Int),typemax(Int),-BigInt(2)^100),q in (-1,1)
            residue=mod(BigInt(m),N)
            residue=residue>N÷2 ? residue-N : residue
            shifted=BigInt(m)+q*N
            @test C.polygon_measure_sum(N,m,2.,.3)==C.polygon_measure_sum(N,shifted,2.,.3)
            @test C.polygon_alias_sum(N,m,2.,.3)==C.polygon_alias_sum(N,shifted,2.,.3)
            @test C.polygon_measure_sum(N,m,2.,.3)==C.polygon_measure_sum(N,Int(residue),2.,.3)
            # Independent ideal-grid phases keep the full integer order in
            # exact modular arithmetic; no staggered-grid identity is assumed.
            reference=(2pi/N)*sum(cispi(2*(mod(BigInt(m)*j,N)/BigFloat(N)))*cis(-2cos(2pi*j/N-.3)) for j in 0:N-1)
            @test C.polygon_measure_sum(N,m,2.,.3) ≈ reference rtol=2e-13 atol=2e-14
        end
        for N in (UInt(17),big"17")
            @test C.polygon_measure_sum(N,3,2.,.3)==C.polygon_measure_sum(17,3,2.,.3)
            @test C.polygon_alias_sum(N,3,2.,.3)==C.polygon_alias_sum(17,3,2.,.3)
        end
        result=C.error_decomposition(8,11,2.,.3,4.,1.)
        beta=C.local_bias_factor(2pi/8)
        @test result.aliasing ≈ beta*(C.polygon_measure_sum(8,3,2.,.3)-C.continuum_harmonic(11,2.,.3)) rtol=1e-14
        @test abs(result.aliasing-beta*(C.polygon_measure_sum(8,3,2.,.3)-C.continuum_harmonic(3,2.,.3)))>1e-3
        f(z)=real(C.polygon_measure_sum(17,3+17*2^45,z,.3))
        @test ForwardDiff.derivative(f,2.) ≈ fd(f,2.) rtol=2e-7
    end
    @testset "Compensated monostatic even-field assembly" begin
        M=C.MonostaticAssembly
        points=M.regular_polygon_points(32,1.)
        for obs in ((3.,.2),(2.7,-.4)), k in (2.,8.)
            result=M.monostatic_even_field(points,obs,k)
            reference=sum(UTDKernels.intrinsic_seam_coefficient(s.delta,k,s.rho/2)*cis(-2k*s.rho)/s.rho for s in result.seams if s.visible)
            @test result.field ≈ reference rtol=2e-14 atol=1e-16
        end
        @test typeof(S.curved_inverse(1.,1e308)) === Float64
        @test ForwardDiff.derivative(g -> S.curved_inverse(.4,g),0.) ≈ -.4^2/2 rtol=1e-14
    end
    @testset "Independent large-argument transition derivatives" begin
        for x in (1e4,1e8,1e12)
            result=F.transition_derivatives(x)
            setprecision(BigFloat,384) do
                xb=BigFloat(x);h=xb*big"1e-8"
                hp(j)=F.transition_F_hp(xb+j*h;dps=100)
                refs=((hp(-2)-8hp(-1)+8hp(1)-hp(2))/(12h),
                      (-hp(2)+16hp(1)-30hp(0)+16hp(-1)-hp(-2))/(12h^2),
                      (hp(-3)-8hp(-2)+13hp(-1)-13hp(1)+8hp(2)-hp(3))/(8h^3))
                for order in 1:3
                    @test abs(result[order+1]/refs[order]-1) < 2e-12
                end
            end
        end
    end
    @testset "Translated facet phase" begin
        p=F.warped_polygon(128,1.0); shift=(.37,-.26)
        moved=F.PolygonScatterer([(x+shift[1],y+shift[2]) for(x,y) in p.verts])
        beta=pi;k=2pi
        phase=cis(k*(cos(beta)*shift[1]+sin(beta)*shift[2]))
        for theta in (.7444444444444444pi,0.73,2.1), pol in (:soft,:hard)
            x=(8cos(theta),8sin(theta));xx=x.+shift
            @test F.polygon_go(moved,xx,beta,k,pol) ≈ phase*F.polygon_go(p,x,beta,k,pol) atol=2e-12
            @test F.polygon_total(moved,xx,beta,k,pol) ≈ phase*F.polygon_total(p,x,beta,k,pol) atol=5e-11
        end
        for theta in (133.85pi/180,134.13pi/180), pol in (:soft,:hard)
            value(t) = F.polygon_total(p,(8cos(t),8sin(t)),beta,k,pol)
            @test abs(value(theta+1e-6)-value(theta-1e-6)) < 0.002
        end
        @test_throws ArgumentError F.polygon_total(p,(8.,1.),beta,k,:soft;modes=[:full])
        @test_throws ArgumentError F.PolygonScatterer([(0.,0.),(1.,0.),(2.,0.)])
    end
    @testset "Scale and exact endpoint covariance" begin
        baseline=nothing
        for scale in (1.,1e-8,1e-12,1e-16)
            p=F.regular_polygon(16,scale)
            value=F.polygon_total(p,(8scale,0.),pi,2pi/scale,:soft)
            @test F.polygon_go(p,(8scale,0.),pi,2pi/scale,:soft)==0
            if baseline===nothing
                baseline=value
            else
                @test value ≈ baseline atol=5e-14
            end
        end
        for source in (pi,.63), pol in (:soft,:hard)
            p=F.regular_polygon(8,1.)
            # Every visible reflected-ray endpoint of every illuminated facet.
            d=(cos(source),sin(source))
            for facet in eachindex(p.verts)
                data=F.facet_data(p,facet)
                dot=data.n[1]*d[1]+data.n[2]*d[2]
                dot>0 || continue
                outgoing=(-(d[1]-2dot*data.n[1]),-(d[2]-2dot*data.n[2]))
                for vertex in (facet,mod1(facet+1,length(p.verts)))
                    anchor=p.verts[vertex]
                    point=anchor.+(8 .* outgoing)
                    tangent=(-outgoing[2],outgoing[1])
                    values=[F.polygon_total(p,point.+(h .* tangent),source,2pi,pol) for h in (-1e-10,-1e-14,0.,1e-14,1e-10)]
                    @test maximum(abs.(values .- values[3])) < 2e-7
                end
            end
        end
    end
    @testset "Incident tangency and distant receiver" begin
        p=F.regular_polygon(8,1.)
        for x in (8.,1000.), pol in (:soft,:hard)
            values=[F.polygon_total(p,(x,1+h),pi,2pi,pol) for h in (-1e-10,-1e-14,-eps(),0.,eps(),1e-14,1e-10)]
            @test maximum(abs.(values .- values[4])) < 2e-7
            @test !F.incident_blocked(p,(x,1.),(-1.,0.))
            @test F.incident_blocked(p,(x,prevfloat(1.)),(-1.,0.))
        end
    end
    @testset "Real Sommerfeld tails" begin
        for q in (100.,1000.,1e5,1e8,Float64(2^27+1),1000.137), sign in (-1,1)
            # Independent parent wide asymptotic reference bypasses the new
            # FMA/erfcx real hot path and reconstructs the phase in BigFloat.
            ref=UTDKernels._shadow_asymptotic_switch(complex(sign*q),Float64)
            actual=S._branch_switch(sign*q)
            @test abs(actual-ref) <= 4e-15*max(abs(ref),floatmin(Float64))
        end
    end
    @testset "Independent reference failure contracts" begin
        @test_throws ArgumentError N.wedge_eigenfunction(50.,2.7,pi/18,2.,:hard;mmax=1)
        @test_throws ArgumentError N.wedge_go(3.,.7,.2,2.,:unknown)
        @test_throws ArgumentError N.canonical_pole_extraction([.4+.3im,.4+.3im],u->1+u)
        @test_throws ArgumentError N.canonical_slope_separated([.4+.3im,.4+.3im],[1.,.2])
        @test_throws ArgumentError N.moment_int_plane(.4+.3im,1;pv_tol=-1.)
        @test N._pole_limit_value(z -> zero(z),.4+.3im)==0
        @test_throws DomainError N.canonical_pole_extraction([.4+.3im],z -> inv(z-(.4+.3im)))
        @test_throws ArgumentError F.PolygonScatterer([(0.,0.,0.),(1.,0.,0.),(0.,1.,0.)])
        @test_throws ArgumentError F.incident_blocked(F.regular_polygon(4,1.),(Inf,0.),(1.,0.))
    end
    @testset "Generic receiver quadrature" begin
        @test S.receiver_quadrature(x -> x^4,-1.,1.;npanels=3,order=4) ≈ .4 rtol=2e-14
        vector=S.receiver_quadrature(x -> [x^2,complex(x,x^2)],-1.,1.;npanels=3)
        @test vector ≈ [2/3,2im/3] rtol=2e-14 atol=1e-15
        action(k)=real(S.receiver_quadrature(x -> exp(im*k*x),-.7,.9;npanels=4))
        @test ForwardDiff.derivative(action,2.) ≈ fd(action,2.) rtol=2e-9
        @test_throws ArgumentError S.receiver_quadrature(identity,1.,-1.;npanels=2)
        @test_throws ArgumentError S.receiver_quadrature(identity,-1.,1.;npanels=0)
        @test_throws DomainError S.receiver_quadrature(x -> NaN,-1.,1.;npanels=2)
    end
    @testset "Sommerfeld branch partials and PEC polarizations" begin
        for x in (0.7,pi-0.03,4.2), kap in (2.,80.)
            jet=S.sommerfeld_cosine_branch_with_derivatives(x,kap)
            @test jet.value == S.sommerfeld_cosine_branch(x,kap)
            @test jet.dx ≈ fd(y -> S.sommerfeld_cosine_branch(y,kap),x;h=2e-6) rtol=2e-7 atol=1e-9
            @test jet.dkappa ≈ fd(y -> S.sommerfeld_cosine_branch(x,y),kap;h=2e-5) rtol=2e-7 atol=1e-9
            @test ForwardDiff.derivative(y -> real(S.sommerfeld_cosine_branch(y,kap)),x) ≈ real(jet.dx) rtol=2e-10 atol=1e-10
            @test ForwardDiff.derivative(y -> imag(S.sommerfeld_cosine_branch(x,y)),kap) ≈ imag(jet.dkappa) rtol=2e-10 atol=1e-10
        end
        for pol in (:soft,:hard)
            jet=S.sommerfeld_field_with_derivatives(3.6,.7,20.;polarization=pol)
            @test jet.value == S.sommerfeld_field(3.6,.7,20.;polarization=pol)
            @test jet.ds ≈ fd(s -> S.sommerfeld_field(s,.7,20.;polarization=pol),3.6) rtol=2e-7
            @test jet.dp ≈ fd(p -> S.sommerfeld_field(3.6,p,20.;polarization=pol),.7) rtol=2e-7
            @test jet.dkappa ≈ fd(k -> S.sommerfeld_field(3.6,.7,k;polarization=pol),20.) rtol=2e-7
        end
        @test S.sommerfeld_field(0.,0.7,20.;polarization=:soft) == 0
        @test abs(fd(s -> S.sommerfeld_field(s,.7,20.;polarization=:hard),0.)) < 1e-9
        @test_throws ArgumentError S.sommerfeld_field(0.,.7,20.;polarization=:unknown)
    end
    @testset "Window jet boundary and truncation contracts" begin
        for x in (.3,.95,.99,.999)
            jets=S._bump_exponent_jets(x,2)
            @test jets[2] ≈ -2x/(1-x*x)^2 rtol=2e-14
            @test jets[3] ≈ -2(1+3x*x)/(1-x*x)^3 rtol=2e-14
        end
        for x in (.2,.8,.95)
            f(x) = S.bump_asymmetric(x;radius=1.,alpha=0.)
            jet=S.bump_jets(x,0.,1.,4)
            @test jet[2] ≈ ForwardDiff.derivative(f,x) rtol=2e-12
            @test jet[3] ≈ ForwardDiff.derivative(y -> ForwardDiff.derivative(f,y),x) rtol=2e-12
        end
        @test_throws ArgumentError S.field_truncated_action(8.,.8,3.9,.8,4)
        @test_throws ArgumentError S.misfit_truncated_action(8.,.8,3.9,.8,4)
    end
end
end
