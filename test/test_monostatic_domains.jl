module MonostaticBoundaryRegressions

using Test, UTDKernels, ForwardDiff, LinearAlgebra
const M=UTDKernels.CurvatureMeasure.MonostaticAssembly

@testset "Monostatic public geometry and reference domains" begin
    points=M.regular_polygon_points(48,1.)
    base=M.monostatic_even_field(points,(3.,0.),8.)
    for shift in ((-10.,0.),(0.,10.),(100.,-100.))
        moved=points.+reshape(collect(shift),1,2)
        result=M.monostatic_even_field(moved,(3+shift[1],shift[2]),8.)
        @test result.field≈base.field atol=4e-14 rtol=4e-13
        @test result.n_visible==base.n_visible
        @test result.arc_lo≈base.arc_lo atol=8e-14
        @test result.arc_hi≈base.arc_hi atol=8e-14
    end
    for scale in (1e-200,1e-100,1e100,1e200)
        seams=M.monostatic_seam_geometry(points.*scale,(3scale,0.))
        @test sum(s.delta for s in seams)≈2pi atol=3e-14
        @test all(s->all(isfinite,(s.theta,s.delta,s.alpha,s.rho,s.eta,s.phi_kp)),seams)
        result=M.monostatic_even_field(points.*scale,(3scale,0.),8/scale)
        @test result.field*sqrt(scale)≈base.field atol=4e-14 rtol=4e-13
        for theta in (-1.1,0.,.4,1.)
            value=M.monostatic_continuum_integrand(scale,3scale,0.,8/scale)(theta)
            reference=M.monostatic_continuum_integrand(1.,3.,0.,8.)(theta)
            @test value*sqrt(scale)≈reference atol=5e-15 rtol=3e-13
        end
    end
    huge=M.monostatic_seam_geometry(points.*1e306.+1e307,(1.3e307,1e307))
    @test [s.theta for s in huge[1:3]]≈[0.,2pi/48,4pi/48] atol=3e-14
    triangle=[1e308 0.;-1e308 1e308;-1e308 -1e308]
    @test sum(s.delta for s in M.monostatic_seam_geometry(triangle,(0.,0.)))≈2pi atol=3e-14
    star=hcat([cos(4pi*j/5) for j=0:4],[sin(4pi*j/5) for j=0:4])
    @test_throws ArgumentError M.monostatic_seam_geometry(star,(3.,0.))

    # The positive radial gap is an independent distance oracle at theta=0.
    near=nextfloat(1.)
    gap=near-1.
    expected=M.linear_prefactor_local(8.,gap/2)*cis(-8gap)^2/gap
    @test M.monostatic_continuum_integrand(1.,near,0.,8.)(0.)≈expected rtol=3e-14
    @test isfinite(M.monostatic_continuum_integrand(1e-308,3e-308,0.,1e308)(.4))
    @test isfinite(M.monostatic_continuum_integrand(1.,3.,-1e308,8.)(1e308))
    # Independent 500-digit mpmath phase reduction for exact binary64 k.
    # F(2kL)-1 is O(1e-308), below the final binary64 relative resolution.
    wide_phase_reference=-4.55157752735199741768022738698e-156-
                         8.87440034283662802136575833327e-156im
    @test M.monostatic_continuum_integrand(1.,3.,0.,1e308)(0.)≈wide_phase_reference rtol=5e-14
    @test isfinite(M.monostatic_arc_integral(1.,3.,0.,8.,-1e308,1e308;n_quad=4))
    radius=1e-308;distance=nextfloat(nextfloat(radius));width=nextfloat(0.)
    constant=M.monostatic_continuum_integrand(radius,distance,0.,1e308)(0.)
    tiny_integral=M.monostatic_arc_integral(radius,distance,0.,1e308,0.,width;n_quad=4)
    @test isfinite(tiny_integral) && !iszero(tiny_integral)
    @test tiny_integral≈constant*width rtol=3e-14
    @test M.monostatic_arc_integral(1.,3.,0.,8.,-.4,.5;n_quad=big(4))==
          M.monostatic_arc_integral(1.,3.,0.,8.,-.4,.5;n_quad=4)
    # Integer angular subtraction must not wrap before floating evaluation.
    angle_hi,angle_lo=typemax(Int),typemin(Int)
    @test M.monostatic_continuum_integrand(1,3,angle_lo,8)(angle_hi)==
          M.monostatic_continuum_integrand(1.,3.,float(angle_lo),8.)(float(angle_hi))
    @test M.monostatic_arc_integral(1,3,0,8,angle_lo,angle_hi;n_quad=4)==
          M.monostatic_arc_integral(1.,3.,0.,8.,float(angle_lo),float(angle_hi);n_quad=4)

    # Independent 100-digit mpmath integer-Bessel/Hankel sums. Exact binary64
    # input k is promoted before evaluation; truncations 40 and 80 differ by
    # less than 3.3e-42 for the two small-argument fixtures.
    fixtures=(
        (1e-8,:soft,-2.6295290221947054612706005856105+0.24912811200045964817073897176885im),
        (1e-8,:hard,0.018745752337080288934196568546925+4.0956111038467677717356302525186e-16im),
        (1e-6,:soft,-1.8999197503904896053815974215941+0.24846485151765087167296972555354im),
        (1e-6,:hard,0.018745752326412332126958163013627+2.9443185572829048794039659299093e-12im),
        (8.,:soft,-0.0034794587318752639154547106444863+0.020187511078738466162320878293245im),
        (8.,:hard,0.0057450354142628376833143314049422-0.019203105199060281131999134264152im))
    for (k,bc,reference) in fixtures
        value=M.cylinder_line_source_scattered(k,1.,3.,0.,3.,0.;bc)
        @test value≈reference rtol=4e-14 atol=3e-16
        @test M.cylinder_line_source_scattered(k,1.,3.,0.,3.,0.;bc,m_max=80)≈reference rtol=4e-14 atol=3e-16
    end
    for scale in (1e-200,1e200),bc in (:soft,:hard)
        value=M.cylinder_line_source_scattered(8/scale,scale,3scale,.2,3scale,-.1;bc)
        reference=M.cylinder_line_source_scattered(8.,1.,3.,.2,3.,-.1;bc)
        @test value≈reference rtol=6e-13 atol=2e-14
    end
    for bc in (:soft,:hard)
        a=M.cylinder_line_source_scattered(8.,1.,3.,.2,4.,-.1;bc,m_max=60)
        reciprocal=M.cylinder_line_source_scattered(8.,1.,4.,-.1,3.,.2;bc,m_max=60)
        periodic=M.cylinder_line_source_scattered(8.,1.,3.,.2+2pi,4.,-.1;bc,m_max=60)
        @test a≈reciprocal rtol=4e-14 atol=1e-15
        @test a≈periodic rtol=4e-13 atol=2e-14
    end
    # Independent 100-digit normalization at a product below binary64's
    # smallest positive value; its final complex value is representable.
    reference=6.8210033651215314088517600398823e97+1.1631555550020942082263440952057e95im
    @test M.hankel_source_normalization(1e-200,1e-200)≈reference rtol=3e-15
    @test M.hankel_source_normalization(1e200,1e200)≈2sqrt(pi*1e200)*(1+im) rtol=3e-14
    incident=M.incident_cylindrical_wave(1e200,1e200)
    @test isfinite(incident)
    @test abs(incident)≈1e-100 rtol=3e-15
    integer_scale=Int(2)^32
    @test M.incident_cylindrical_wave(integer_scale,integer_scale)==
        M.incident_cylindrical_wave(float(integer_scale),float(integer_scale))
    @test_throws ArgumentError M.cylinder_line_source_scattered(
        typemax(Int),typemax(Int),typemax(Int),0,typemax(Int),0)

    # Differentiate geometry and the native/widened special-function routes.
    value(t)=real(M.monostatic_continuum_integrand(1.,3.,.1,8.)(t))
    @test ForwardDiff.derivative(value,.4)≈(value(.400001)-value(.399999))/2e-6 rtol=3e-8 atol=1e-9
    @test isfinite(ForwardDiff.derivative(t->ForwardDiff.derivative(value,t),.4))
    for bc in (:soft,:hard)
        f(k)=real(M.cylinder_line_source_scattered(k,1.,3.,.2,3.,0.;bc))
        h=8.1e-6
        @test ForwardDiff.derivative(f,8.1)≈(f(8.1+h)-f(8.1-h))/(2h) rtol=3e-7 atol=2e-9
        @test isfinite(ForwardDiff.derivative(k->ForwardDiff.derivative(f,k),1e-8))
    end
    # Independent mpmath first and second derivatives at the same azimuths,
    # using 100 digits and 80 modes. This also detects tangent loss during
    # widening of an underflowing native Bessel factor.
    for (bc,d1,d2) in ((:soft,15860779.679609651544317403285148,-1586655621399763.9291375663383434),
                       (:hard,-3.9372024792670731674866202755258e-7,-34.316485718488133488372974691733))
        f(k)=real(M.cylinder_line_source_scattered(k,1.,3.,0.,3.,0.;bc))
        @test ForwardDiff.derivative(f,1e-8)≈d1 rtol=2e-13
        @test ForwardDiff.derivative(k->ForwardDiff.derivative(f,k),1e-8)≈d2 rtol=3e-13
    end
    normalized(p)=real(M.hankel_source_normalization(1e-200*p,1e-200))/1e98
    h=1e-4
    @test ForwardDiff.derivative(normalized,1.)≈(normalized(1+h)-normalized(1-h))/(2h) rtol=3e-7 atol=2e-11
    tiny_points=M.regular_polygon_points(12,1.).*1e-200
    translated(t)=M.monostatic_even_field(tiny_points.+1e300*t,
        (3e-200+1e300*t,1e300*t),8e200)
    for quantity in (:max_abs_edge,:max_abs_face,:arc_lo,:arc_hi)
        f(t)=getproperty(translated(t),quantity)
        @test ForwardDiff.derivative(f,0.)==0
        @test ForwardDiff.derivative(t->ForwardDiff.derivative(f,t),0.)==0
    end
    translated_field(t)=real(translated(t).field)/1e100
    @test ForwardDiff.derivative(translated_field,0.)==0
    @test ForwardDiff.derivative(t->ForwardDiff.derivative(translated_field,t),0.)==0

    @test_throws ArgumentError M.regular_polygon_points(2,1.)
    @test_throws ArgumentError M.regular_polygon_points(3,NaN)
    @test_throws ArgumentError M.monostatic_seam_geometry(zeros(0,2),(3.,0.))
    @test_throws ArgumentError M.monostatic_seam_geometry(reverse(points;dims=1),(3.,0.))
    @test_throws ArgumentError M.monostatic_seam_geometry(points,(Inf,0.))
    @test_throws ArgumentError M.monostatic_seam_geometry(points,(points[1,1],points[1,2]))
    @test_throws ArgumentError M.monostatic_even_field(points,(3.,0.),NaN)
    @test_throws ArgumentError M.visible_arc(M.MonostaticSeam{Float64}[],(3.,0.))
    @test_throws ArgumentError M.visible_arc(base.seams,(NaN,0.))
    for invalid in (-1.,0.,Inf,NaN)
        @test_throws ArgumentError M.cylindrical_spreading(invalid)
        @test_throws ArgumentError M.incident_cylindrical_wave(invalid,1.)
        @test_throws ArgumentError M.hankel_source_normalization(invalid,1.)
        @test_throws ArgumentError M.cylinder_line_source_scattered(invalid,1.,3.,0.,3.,0.)
    end
    @test_throws ArgumentError M.monostatic_visibility_half_angle(1.,1.)
    @test_throws ArgumentError M.monostatic_continuum_integrand(1.,1.,0.,8.)
    @test_throws ArgumentError M.monostatic_continuum_integrand(1.,3.,0.,8.)(NaN)
    @test_throws ArgumentError M.monostatic_arc_integral(1.,3.,0.,8.,1.,0.;n_quad=4)
    @test_throws ArgumentError M.monostatic_arc_integral(1.,3.,0.,8.,0.,1.;n_quad=0)
    @test_throws ArgumentError M.cylinder_line_source_scattered(8.,1.,3.,0.,3.,0.;bc=:unknown,m_max=0)
    @test_throws ArgumentError M.cylinder_line_source_scattered(8.,1.,3.,0.,3.,0.;m_max=-1)
    @test_throws ArgumentError M.cylinder_line_source_scattered(1e308,1e308,1e308,0.,1e308,0.)
end

end
