module CartesianStudyGeometryTests
using Test,ForwardDiff,UTDKernels
const F=UTDKernels.FaceEdge
@testset "Canonical Cartesian geometry contracts" begin
    for scale in (1e-200,1.,1e200)
        @test F.orientation2((scale,0.),(0.,scale))==1
        @test F.orientation2((0.,scale),(scale,0.))==-1
        @test F.orientation2((scale,scale),(scale,scale))==0
        @test F.signed_angle2((scale,0.),(scale,scale))≈pi/4 rtol=3eps()
        @test F.crossing_fraction2((scale,0.),(0.,scale),(0.,scale),(scale,0.))≈.5 rtol=3eps()
        hit=F.ray_segment_intersection((0.,0.),(1.,0.),(scale,-scale),(scale,scale))
        @test hit.inside && hit.forward
        @test hit.t≈scale rtol=3eps()
        @test hit.tau≈.5 rtol=3eps()
        @test all(isapprox.(F.unit_direction2((0.,0.),(scale,scale)),(inv(sqrt(2.)),inv(sqrt(2.)));rtol=3eps()))
    end
    hit=F.ray_segment_intersection((-1e308,0.),(2.,0.),(1e308,-1.),(1e308,1.))
    @test hit.inside && hit.forward && hit.t==1e308 && hit.tau==.5
    for h in (-1e-10,0.,1e-10),source_scale in (1e-200,1.,1e200),observer_scale in (1e-200,1.,1e200)
        n=1.2;phip=.4;phi=pi-phip+h
        source=source_scale.*(cos(phip),sin(phip))
        observer=observer_scale.*(-cos(phip-h),sin(phip-h))
        d=F.wedge_cartesian_detunings(phi,phip,n,(1.,0.),(cos(n*pi),sin(n*pi)),observer,source)
        @test d[4]≈-h/(2n) atol=8e-16
    end
    for seed in (-1.,1.)
        origin=(ForwardDiff.Dual(0.,seed),ForwardDiff.Dual(0.,0.))
        hit=F.ray_segment_intersection(origin,(1.,0.),(0.,-1.),(0.,1.))
        @test !hit.forward && hit.inside
        @test ForwardDiff.partials(hit.t)[1]==-seed
    end
    d=nextfloat(0.)
    endpoint=F.ray_segment_intersection((0.,0.),(d,d),(d,d),(1.,nextfloat(1.)))
    @test endpoint!==nothing && endpoint.t==1 && endpoint.tau==0 && endpoint.forward && endpoint.inside
    for seed in (-1.,1.)
        offsets=(.1,.1,.1,ForwardDiff.Dual(0.,seed))
        @test F.wedge_go_visibility(pi-.4,.4,1.2;detunings=offsets).reflection0
    end
    for scale in (1e-200,1.,1e200)
        g(y)=UTDKernels._study_distance2((0.,0.),(scale,y))
        @test ForwardDiff.derivative(y->ForwardDiff.derivative(g,y),0.)≈inv(scale) rtol=3e-14
        direction(y)=F.unit_direction2((0.,0.),(scale,y))[1]
        @test ForwardDiff.derivative(direction,scale)≈-inv(scale)/(2sqrt(2.)) rtol=3e-14
        angle(y)=F.signed_angle2((scale,0.),(scale,y))
        @test ForwardDiff.derivative(angle,0.)≈inv(scale) rtol=3e-14
    end
    fraction(x)=F.crossing_fraction2((x,0.),(0.,x),(0.,x),(x,0.))
    @test fraction(1.)==.5
    @test ForwardDiff.derivative(fraction,1.)==0
    @test ForwardDiff.derivative(x->ForwardDiff.derivative(fraction,x),1.)==0
    triangle=[1e308 0.;-1e308 1e308;-1e308 -1e308]
    @test sum(UTDKernels.CurvatureMeasure.polygon_turning_angles(triangle))≈2pi rtol=3eps()
    star=[(cos(4pi*j/5),sin(4pi*j/5)) for j in 0:4]
    @test_throws ArgumentError F.PolygonScatterer(star)
    @test_throws ArgumentError UTDKernels.CurvatureMeasure.polygon_turning_angles(hcat(first.(star),last.(star)))
    @test_throws ArgumentError F.signed_angle2((0.,0.),(1.,0.))
    @test_throws ArgumentError F.orientation2((1.,0.,0.),(0.,1.))
    @test_throws ArgumentError F.orientation2((Inf,0.),(0.,1.))
    @test_throws DomainError F.crossing_fraction2((1.,0.),(0.,1.),(1.,0.),(0.,1.))
end
end
