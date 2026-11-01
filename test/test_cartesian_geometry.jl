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

@testset "Arbitrary-precision Cartesian projection retains values and jets" begin
    for bits in (128,256)
        setprecision(BigFloat,bits) do
            point=BigFloat("0.3")
            bearing=(BigFloat("0.5"),BigFloat("0.8"))
            angle(x)=F.signed_angle2((one(x),x),bearing)
            expected,first,second=setprecision(BigFloat,2bits) do
                x=BigFloat(point)
                atan(bearing[2],bearing[1])-atan(x),
                    -inv(1+x*x),2x/(1+x*x)^2
            end
            actual=angle(point)
            derivative=ForwardDiff.derivative(angle,point)
            curvature=ForwardDiff.derivative(x->ForwardDiff.derivative(angle,x),point)
            @test precision(actual)==bits
            @test precision(derivative)==bits
            @test precision(curvature)==bits
            @test actual≈expected rtol=8eps(BigFloat)
            @test derivative≈first rtol=32eps(BigFloat)
            @test curvature≈second rtol=64eps(BigFloat)

            # Exercise the complex projection owner as well as the real angle.
            projected=UTDKernels._shadow_stable_wide_value(Complex{BigFloat},2bits,"complex projection") do
                exp(im*sqrt(BigFloat(2)))
            end
            reference=setprecision(BigFloat,2bits) do
                argument=sqrt(BigFloat(2))
                Complex(cos(argument),sin(argument))
            end
            @test precision(real(projected))==bits && precision(imag(projected))==bits
            @test abs(projected-reference)<=8eps(BigFloat)
        end
    end
end

@testset "A widened wedge retains its original angular carriers" begin
    for R in (Float32,Float64),reflection in (3,4)
        n,p=R(1.2),reflection==3 ? R(1) : R(.4)
        q=reflection==3 ? 2n*R(pi)-R(pi)-p : R(pi)-p
        q+=R(1e-10)
        face0=(one(R),zero(R));facen=(cos(n*R(pi)),sin(n*R(pi)))
        source=(cos(p),sin(p));observer=(cos(q),sin(q))
        reference=F._wedge_detuning_reference(q,p,n,face0,facen,observer,source)
        @test reference.detunings==F.wedge_cartesian_detunings(q,p,n,face0,facen,observer,source)
        @test reference.cartesian[reflection]
        @test all(value->!UTDKernels._number_contains_ad(value),reference.detunings)
        setprecision(BigFloat,512) do
            work=F._wedge_detunings_in_workspace(reference,BigFloat(q),BigFloat(p),BigFloat(n),
                BigFloat.(face0),BigFloat.(facen),BigFloat.(observer),BigFloat.(source))
            @test R.(work.detunings)==reference.detunings
            @test R.(work.parent_detunings)==reference.parent_detunings
        end
    end

    # The two centers differ by native angular rounding. Preserve the actual
    # pre-division values; multiplying an offset by 2n is not their definition.
    n,p=1.2,.4
    q=pi-p+1e-10
    face0=(1.,0.);facen=(cos(n*pi),sin(n*pi))
    source=(cos(p),sin(p));observer=(-cos(p-1e-10),sin(p-1e-10))
    reference=F._wedge_detuning_reference(q,p,n,face0,facen,observer,source)
    @test reference.cartesian[4] && reference.Nj[4]==0
    setprecision(BigFloat,256) do
        nb=BigFloat(n)
        values=(BigFloat(q),BigFloat(p),BigFloat.(face0),BigFloat.(facen),
                BigFloat.(observer),BigFloat.(source))
        work(index)=F._wedge_detunings_in_workspace(reference,values[1],values[2],index,
            values[3],values[4],values[5],values[6])
        local_angle=-BigFloat(reference.deviations[4])
        parent_numerator=BigFloat(reference.parent_numerators[4])
        for (name,numerator) in ((:detunings,local_angle),(:parent_detunings,parent_numerator))
            offset(index)=getproperty(work(index),name)[4]
            @test ForwardDiff.derivative(offset,nb)≈-numerator/(2nb^2) rtol=64eps(BigFloat)
            @test ForwardDiff.derivative(x->ForwardDiff.derivative(offset,x),nb)≈
                  numerator/nb^3 rtol=128eps(BigFloat)

            # A cached binary geometry jet would lose this mixed coefficient
            # before the completed field's scale restores it.
            seed,scale=BigFloat(1e-320),BigFloat(1e300)
            restored(t,u)=scale*offset(nb+t+seed*u)
            mixed=ForwardDiff.derivative(BigFloat(0)) do t
                ForwardDiff.derivative(u->restored(t,u),BigFloat(0))
            end
            expected=scale*seed*numerator/nb^3
            @test abs(expected)>big"1e-35"
            @test mixed≈expected rtol=256eps(BigFloat)
        end
    end
end
end
