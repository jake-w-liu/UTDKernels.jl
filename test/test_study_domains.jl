module StudyDomainRegressionTests
using Test, ForwardDiff, QuadGK, UTDKernels
const F=UTDKernels.FaceEdge
const N=UTDKernels.NullUniform
const S=UTDKernels.ShadowSensitivity

@testset "Null empty and coefficient range validation" begin
    for empty in (Float32[],ComplexF64[])
        @test iszero(N.hierarchy_value(1.,.5+.2im,empty))
        @test iszero(N.shifted_null_hierarchy(1.,.5+.2im,.3,2,empty))
        @test_throws ArgumentError N.hierarchy_value(1.,.5+.2im,empty;order=-1)
        @test_throws ArgumentError N.hierarchy_value(1.,Inf,empty)
        @test_throws ArgumentError N.shifted_null_hierarchy(1.,Inf,.3,2,empty)
        @test_throws ArgumentError N.shifted_null_hierarchy(1.,.5+.2im,Inf,2,empty)
        @test_throws ArgumentError N.shifted_null_hierarchy(1.,.5+.2im,.3,2,empty;order=-1)
    end
    @test_throws DomainError N.joint_null_coefficients(1e-308,1e308;tail=())
    @test_throws DomainError N.factored_null_coefficients(1e-308,1e308;b=())
    @test all(isfinite,N.joint_null_coefficients(1e-308,1e100;tail=()))
end

function gaussian_reference(k,s)
    setprecision(BigFloat,256) do
        kb,sb=BigFloat(k),BigFloat(s)
        exp(im*BigFloat(pi)/4)*sqrt(kb)/sqrt(inv(2sb^2)+im*kb)
    end
end
@testset "Gaussian action scaled range and AD" begin
    for k in (1e-308,1e-100,1.,1e100,1e308),s in (1e-200,1e-100,.7,1e100,1e200)
        expected=ComplexF64(gaussian_reference(k,s))
        actual=S.gaussian_action(k,s)
        @test isfinite(actual)
        @test actual ≈ expected rtol=8eps() atol=nextfloat(0.)
    end
    for k in (1f-20,1f0,1f20),s in (1f-20,.7f0,1f20)
        @test S.gaussian_action(k,s) isa ComplexF32
        @test S.gaussian_action(k,s) ≈ ComplexF32(gaussian_reference(k,s)) rtol=8eps(Float32) atol=nextfloat(0f0)
    end
    for (k,s) in ((8.,.7),(1e308,1e-200)),component in (real,imag)
        expected=setprecision(BigFloat,256) do
            kb,sb=BigFloat(k),BigFloat(s)
            D=inv(2sb^2)+im*kb
            A=exp(im*BigFloat(pi)/4)*sqrt(kb)/sqrt(D)
            Float64(component(A/(2sb^3*D)))
        end
        @test ForwardDiff.derivative(x -> component(S.gaussian_action(k,x)),s) ≈ expected rtol=2e-14
    end
    for component in (real,imag)
        k,s=8.,.7
        expected=setprecision(BigFloat,256) do
            kb,sb=BigFloat(k),BigFloat(s)
            D=inv(2sb^2)+im*kb
            A=exp(im*BigFloat(pi)/4)*sqrt(kb)/sqrt(D)
            ratio=inv(2sb^3*D)
            Float64(component(A*(3ratio^2-3ratio/sb)))
        end
        @test ForwardDiff.derivative(x -> ForwardDiff.derivative(y -> component(S.gaussian_action(k,y)),x),s) ≈ expected rtol=2e-13
    end
end

function objective_reference(n,k,s)
    setprecision(BigFloat,512) do
        nb,kb,sb=BigFloat(n),BigFloat(k),BigFloat(s)
        D=inv(2sb^2)+im*kb
        gaussian_reference(k,s)*exp(-nb^2/(4D))
    end
end
@testset "Translated objective range and AD" begin
    for k in (1e-308,1e-100,1.,1e100,1e308),s in (1e-200,1e-100,.7,1e100,1e200)
        @test S.translated_objective_action(0.,k,s) ≈ S.gaussian_action(k,s) rtol=8eps() atol=nextfloat(0.)
    end
    for(k,s) in ((1e308,1e-200),(8.,.7),(1e-308,1e200)),n in (0.,.3,8.)
        expected=ComplexF64(objective_reference(n,k,s))
        @test S.translated_objective_action(n,k,s) ≈ expected rtol=3e-14 atol=nextfloat(0.)
    end
    for component in (real,imag)
        k,s=8.,.7
        A=gaussian_reference(k,s)
        D=inv(2BigFloat(s)^2)+im*BigFloat(k)
        f(n)=component(S.translated_objective_action(n,k,s))
        @test ForwardDiff.derivative(f,0.) == 0
        @test ForwardDiff.derivative(x -> ForwardDiff.derivative(f,x),0.) ≈ Float64(component(-A/(2D))) rtol=2e-14
    end
    @test S.translated_objective_action(.3f0,8f0,.7f0) isa ComplexF32
end

@testset "Ellipse density scale covariance" begin
    C=UTDKernels.CurvatureMeasure
    for scale in (1e-200,1.,1e200),a in (1.,2.),b in (.7,1.)
        expected=C.ellipse_curvature_density([0.,.3,pi/2],a,b)
        @test C.ellipse_curvature_density([0.,.3,pi/2],scale*a,scale*b) ≈ expected rtol=8eps()
    end
    @test C.ellipse_curvature_density(.3f0,1f0,1f0) isa Float32
    @test C.ellipse_curvature_density(1e-310,1e300,1e-300) ≈ 1e20 rtol=2e-13
    @test_throws DomainError C.ellipse_curvature_density(0.,1e300,1e-300)
    @test_throws ArgumentError C.ellipse_curvature_density(.3im,1.,1.)
    @test_throws ArgumentError C.ellipse_curvature_density([.3,Inf],1.,1.)
    for t in (.3,1.)
        @test ForwardDiff.derivative(a -> C.ellipse_curvature_density(t,a,1.),1.) ≈ cos(2t) rtol=2e-14
        @test ForwardDiff.derivative(b -> C.ellipse_curvature_density(t,1.,b),1.) ≈ -cos(2t) rtol=2e-14
    end
end

@testset "Fourier validation and finite band semantics" begin
    @test_throws ArgumentError S.fourier_samples(identity,1.,-1.,[0.])
    @test_throws ArgumentError S.fourier_samples(identity,-1.,1.,Float64[])
    @test_throws ArgumentError S.fourier_samples(identity,-1.,1.,[NaN])
    @test_throws DomainError S.fourier_samples(x -> NaN,-1.,1.,[0.,1.])
    @test_throws ArgumentError S.fourier_samples(exp,-1.,1.,[0.];max_refinements=1)
    @test_throws ArgumentError S.fourier_samples(exp,-1.,1.,[0.];atol=nextfloat(0.),max_refinements=2)
    @test_throws ArgumentError S.spectral_moment(exp,-1.,1.;xi_cut=NaN)
    @test_throws ArgumentError S.spectral_moment(exp,-1.,1.;power=-1)
    @test_throws ArgumentError S.spectral_moment(exp,-1.,1.;frequency_step=0)
    @test S.spectral_moment(exp,-1.,1.;xi_cut=0).moment == 0
    sigma=.05
    # Gaussian tails outside this support have negligible Fourier mass even
    # after the tested finite-band weights. Its transform is independent.
    transform(x,w)=sqrt(2pi)*sigma*exp(-sigma^2*(x-w)^2/2)
    for (label,frequencies,weights,cut,power) in (
        (:shifted,(3000.,),(1.,),4000.,2),
        (:separated,(-2700.,3100.),(.6,1.),4000.,2),
        (:fourth,(-70.,90.),(.6,1.),180.,4),
        (:noninteger,(17.,),(1.,),43.25,.5),
    )
        fun(x)=exp(-x^2/(2sigma^2))*sum(a*cis(w*x) for(w,a) in zip(frequencies,weights))
        fhat(x)=sum(a*transform(x,w) for(w,a) in zip(frequencies,weights))
        breaks=sort(unique(vcat(-cut,cut,[-cut/2,0.,cut/2],collect(frequencies))))
        refmoment=quadgk(x -> abs(x)^power*fhat(x)/(2pi),breaks...;atol=1e-8,rtol=2e-12)[1]
        refl1=quadgk(x -> fhat(x)/(2pi),breaks...;atol=1e-12,rtol=2e-12)[1]
        step = label === :noninteger ? pi/256 : pi/2
        value=S.spectral_moment(fun,-.5,.5;power=power,xi_cut=cut,frequency_step=step)
        fine=S.spectral_moment(fun,-.5,.5;power=power,xi_cut=cut,frequency_step=step/2)
        println((case=label,value=value,fine=fine,refmoment=refmoment,refl1=refl1))
        @test value.moment ≈ refmoment rtol=5e-5
        @test value.l1 ≈ refl1 rtol=5e-6
        @test fine.moment ≈ refmoment rtol=5e-5
        @test abs(fine.moment-value.moment) <= 5e-5*refmoment
        @test value.moment_error_estimate >= 0
        @test value.l1_error_estimate >= 0
    end
    carrier(x)=S.bump_asymmetric(x;radius=.8,alpha=0.)*cis(3000x)
    @test S.spectral_moment(carrier,-.8,.8;xi_cut=4000.).moment > 449850.0125
    # Matched cutoffs obey the exact finite-band scaling identity; comparing
    # two unmatched fixed cutoffs would mix a scaling error with missing tails.
    probe(x)=S.bump_asymmetric(x;radius=1.2,alpha=0.)
    F(y)=probe(S.curved_inverse(y,.6))
    lo,hi=S.curved_coordinate(-1.2,.6),S.curved_coordinate(1.2,.6)
    base=S.spectral_moment(F,lo,hi;xi_cut=600.)
    scaled=S.spectral_moment(y -> F(y/.2),.2lo,.2hi;xi_cut=3000.)
    @test scaled.moment ≈ 25base.moment rtol=1e-7
    @test scaled.l1 ≈ base.l1 rtol=1e-9
    println((nonlinear_base=base,nonlinear_scaled=scaled))
end
@testset "Selection validation and range" begin
    @test_throws ArgumentError F.select_modes([NaN,.2],.1)
    @test_throws ArgumentError F.select_modes(Float64[],NaN)
    @test_throws ArgumentError F.select_modes([.2],-.1)
    @test F.select_modes(Float64[],.1)==Symbol[]
    @test F.select_modes([0.,.1,.2],.1)==[:preserve,:preserve,:full]
    @test_throws ArgumentError F.incident_shadow_active(.2,1.2,2.,1.,1.;field_tol=NaN,kla_guard=NaN)
    @test_throws ArgumentError F.incident_shadow_active(.2,1.2,2.,1.,NaN)
    @test_throws ArgumentError F.incident_shadow_active(.2,1.2,2.,1.,1.;field_tol=-1.)
    g=F.BoundaryGeometry(.1,.4,1e308,1e-308)
    expected=sqrt(2pi)*sqrt(g.k)*maximum(abs,F.coefficients_decomposed(g))
    @test F.normalized_raw_score(g) ≈ expected rtol=2e-15
    @test F.normalized_raw_score(g) ≈ F.normalized_raw_score(F.BoundaryGeometry(.1,.4,1.,1.)) rtol=2e-14
end
@testset "Polygon constructor scale" begin
    for scale in (1e-200,1.,1e200),builder in (F.regular_polygon,F.warped_polygon)
        p=builder(8,scale)
        q=builder(8,1.)
        @test [F.vertex_defect(p,j) for j in 1:8] ≈ [F.vertex_defect(q,j) for j in 1:8] rtol=1e-14
    end
end
@testset "Phase-map scaled square roots" begin
    @test isfinite(N.wedge_phase_map(1e308).c)
    @test abs(N.wedge_phase_map(1e308).c) ≈ sqrt(1e308)*sqrt(2.) rtol=3e-15
    @test S.sommerfeld_cosine_branch(0.,1e308) ≈ cis(1e308) rtol=2e-15
    for k in (nextfloat(0.),1e-308,1.,1e308)
        jet=S.sommerfeld_cosine_branch_with_derivatives(.7,k)
        @test all(isfinite,jet)
        @test jet.value==S.sommerfeld_cosine_branch(.7,k)
    end
    @test_throws ArgumentError N.classical_pole_term(3.,.5+.2im,z -> NaN)
end

@testset "Quadrature and reference boundary contracts" begin
    @test_throws ArgumentError N.canonical_pole_pair_value([.3+.2im],[NaN],0)
    @test_throws ArgumentError N.canonical_slope_separated([.3+.2im],[1.,NaN])
    @test_throws ArgumentError N.canonical_pair_quadrature(ComplexF64[],x -> 1.)
    @test_throws ArgumentError N.canonical_pair_quadrature([.3+.2im],x -> 1.;T=Inf)
    @test_throws ArgumentError N.canonical_pair_quadrature([.3+.2im],x -> 1.;rtol=NaN)
    for field in (F.cylinder_scattered,F.cylinder_radial_derivative)
        @test_throws ArgumentError field(2.,1.,3.,.2,.3,:unknown;M=-1)
        @test_throws ArgumentError field(2.,1.,3.,.2,.3,:soft;M=-1)
        @test_throws ArgumentError field(2.,1.,.5,.2,.3,:soft;M=20)
    end
    @test S.receiver_quadrature(x -> 1e-308,-1e308,1e308;npanels=1,order=1) ≈ 2 rtol=3eps()
    @test S.receiver_quadrature(x -> 1e308,0.,1e-308;npanels=1,order=1) ≈ 1 rtol=3eps()
    @test S.receiver_quadrature(x -> 1e308,0.,nextfloat(0.);npanels=1,order=1) ≈ 1e308*nextfloat(0.) rtol=3eps()
    @test_throws DomainError S.receiver_quadrature(x -> 1e308,-1.,1.;npanels=1,order=1)
    @test_throws ArgumentError S.composite_phase_quadrature(x -> iszero(x) ? NaN : 1.,-1.,1.,2.;order=4)
    points=[1e308 0.;-1e308 1e308;-1e308 -1e308]
    @test sum(UTDKernels.CurvatureMeasure.polygon_turning_angles(points)) ≈ 2pi rtol=2eps()
    @test all(isapprox.(UTDKernels._study_centroid2([1e307 1e307;1.1e307 1.1e307]), (1.05e307,1.05e307)))
    @test_throws ArgumentError UTDKernels._study_unit_direction2((0.,0.),(0.,0.))
    @test N.fitted_log_slope([big"1e1000",big"1e2000"],[big"1e3000",big"1e6000"]) ≈ 3 rtol=2eps()
end
@testset "Half-plane scale-independent pole families" begin
    base=N.wedge_canonical_model(1.,2.7,pi/18,2.,:hard;ncoef=4)
    for k in (1e-308,1e308)
        model=N.wedge_canonical_model(k,2.7,pi/18,2.,:hard;ncoef=k<1 ? 1 : 4)
        @test length(model.poles)==2
        @test length(model.zeros)==2
        for j in eachindex(model.coefficients)
            expected=setprecision(BigFloat,256) do
                ComplexF64(Complex{BigFloat}(base.coefficients[j])*BigFloat(k)^((2-j)/BigFloat(2)))
            end
            @test model.coefficients[j] ≈ expected rtol=1e-13 atol=nextfloat(0.)
        end
    end
end
@testset "Signed transition windows and carrier controls" begin
    @test_throws ArgumentError S.sommerfeld_inverse_map(Inf,.8)
    @test_throws DomainError S.sommerfeld_inverse_map(2.,.8)
    @test_throws ArgumentError S.field_truncated_action(8.,.8,.8+pi,.8,1;remainder=NaN)
    @test_throws ArgumentError S.misfit_truncated_action(8.,.8,.8+pi,.8,1;remainder=NaN)
    for center in (.3,.8+pi,.8+2pi+.5),kernel in (:canonical,:gaussian)
        k,p,radius=2.,.8,.3
        density=kernel===:canonical ? S.canonical_kernel : x -> exp(-x^2)/sqrt(pi)
        expected=S.receiver_quadrature(center-radius,center+radius;npanels=64) do s
            S.receiver_window(s,center,radius)*sqrt(k)*sin((s-p)/2)/sqrt(2.)*density(sqrt(2.)*sqrt(k)*cos((s-p)/2))
        end
        @test S.field_transition_action(k,p,center,radius;kernel=kernel,npanels=64) ≈ expected rtol=3e-14 atol=1e-15
    end
    expected=setprecision(BigFloat,256) do
        ComplexF64(exp(-8im)*sum(binomial(big(2m),m)/big(4)^m for m in 0:99))
    end
    @test S.carrier_action_truncated(8.,100) ≈ expected rtol=2e-14
    for k in (nextfloat(0.),1.,1e16)
        expected=cis(pi/4)*cis(-k)*(sqrt(k)/sqrt(pi))*2
        @test S.carrier_action_exact(k,1.) ≈ expected rtol=4eps() atol=nextfloat(0.)
    end
end

@testset "Empty array shared controls" begin
    for f in (S.scaled_kernel,S.fourier_multiplier,S.real_gaussian_multiplier,
              S.halfplane_q,S.halfplane_q_prime,S.halfplane_q_second,
              S.halfplane_demodulated,S.halfplane_d1,S.halfplane_d2,S.sommerfeld_cosine_branch)
        @test isempty(f(Float64[],1.))
        @test_throws ArgumentError f(Float64[],-1.)
        @test_throws ArgumentError f(Float64[],ForwardDiff.Dual(0.,1.))
        @test_throws ArgumentError f(Float64[],ForwardDiff.Dual(1.,Inf))
    end
    @test_throws ArgumentError S.truncated_multiplier(Float64[],1.,0)
    @test_throws ArgumentError S.truncated_multiplier(Float64[],-1.,1)
    for f in (S.curved_coordinate,S.curved_inverse)
        @test isempty(f(Float64[],.6))
        @test_throws ArgumentError f(Float64[],NaN)
    end
    @test isempty(S.bump_asymmetric(Float64[];radius=1.))
    @test_throws ArgumentError S.bump_asymmetric(Float64[];radius=0.)
    @test_throws ArgumentError S.bump_asymmetric(Float64[];alpha=NaN)
    @test_throws ArgumentError N.wedge_kernel(.3,.8,Inf,:hard)
    @test_throws ArgumentError N.wedge_kernel_zeros(0.)
    @test_throws ArgumentError N.wedge_kernel_zeros(NaN)
    @test_throws ArgumentError N.wedge_kernel_poles(.8,Inf)
end
@testset "Principal promoted scalar contracts" begin
    @test F.wedge_coefficients(.2f0,.3f0,1.2f0,3f0,1f0).De isa ComplexF32
    @test F.wedge_coefficients(.2f0,.3f0,1.2f0,3.,1f0).De isa ComplexF64
    @test eltype(N.wedge_canonical_model(50f0,2.7f0,.2f0,2f0,:hard;ncoef=4).coefficients)==ComplexF32
    @test N.wedge_diffracted_model(50f0,2.7f0,.2f0,2f0,:hard;ncoef=4) isa ComplexF32
    @test N.wedge_diffracted_model(50f0,2.7,.2f0,2f0,:hard;ncoef=4) isa ComplexF64
    @test S.sommerfeld_demodulated_dp(.3f0,.7f0,8f0) isa ComplexF32
    @test S.sommerfeld_demodulated_dp(.3f0,.7,8f0) isa ComplexF64
    for phi in (.4f0,2.7f0,4.7f0),pol in (:soft,:hard),saddle in (0f0,-2Float32(pi))
        model=N.wedge_canonical_model(50f0,phi,.2f0,2f0,pol;saddle=saddle,ncoef=4)
        @test maximum(abs.(model.amplitude_u.(model.poles).-model.pole_limits))/abs(model.maps.c)<4e-6
    end
    for component in (real,imag)
        f(k)=component(F.wedge_coefficients(.2f0,.3f0,1.2f0,k,1f0).De)
        g(k)=component(F.wedge_coefficients(Float64(.2f0),Float64(.3f0),Float64(1.2f0),k,1.).De)
        @test ForwardDiff.derivative(f,3f0)≈ForwardDiff.derivative(g,3.) rtol=4e-5
    end
    @test S.sommerfeld_demodulated_dp(0.,0.,1e308)==0
    reference=setprecision(BigFloat,1536) do
        ComplexF64(exp(2im*BigFloat(1e308)))
    end
    @test UTDKernels._cis_product(1e308,2.)≈reference rtol=3eps()
end
@testset "Receiver mean and scaled phase quadrature" begin
    for order in (1,4,12,32)
        @test S.receiver_quadrature(x->1.,0.,nextfloat(0.);npanels=1,order=order)==nextfloat(0.)
        @test S.receiver_quadrature(x->1e308,0.,1e-308;npanels=1,order=order)≈1 rtol=4eps()
        @test S.receiver_quadrature(x->1e-308,-1e308,1e308;npanels=1,order=order)≈2 rtol=4eps()
    end
    vector=S.receiver_quadrature(x->[1.,2.],0.,nextfloat(0.);npanels=1,order=12)
    @test vector==[nextfloat(0.),2nextfloat(0.)]
    f(p)=S.receiver_quadrature(x->ForwardDiff.value(p)+zero(x)+p-ForwardDiff.value(p),0.,nextfloat(0.);npanels=1)
    @test ForwardDiff.derivative(f,1.)==nextfloat(0.)
    reference=S.canonical_switch(2.)-S.canonical_switch(1.)
    for scale in (1e-154,1.,1e154),side in (-1,1)
        lo,hi=side<0 ? (-2scale,-scale) : (scale,2scale)
        @test S.composite_phase_quadrature(x->1.,lo,hi,inv(scale)^2)≈reference rtol=4e-14 atol=2e-15
    end
end
@testset "Physical incident amplitude budget" begin
    @test !F.incident_shadow_active(.2,1.2,1e3,1e3,1.;incident_amplitude=0.)
    @test !F.incident_shadow_active(.2,1.2,1e3,1e3,1.;incident_amplitude=1e-300)
    @test F.incident_shadow_active(.2,1.2,1e3,1e3,1.;incident_amplitude=1e300)
    @test F.incident_shadow_active(pi,1.2,1e308,1e308,0.;incident_amplitude=0.)
    @test_throws ArgumentError F.incident_shadow_active(.2,1.2,1.,1.,1.;incident_amplitude=NaN)
    @test_throws ArgumentError F.incident_shadow_active(.2,1.2,1.,1.,1.;incident_amplitude=-1.)
end

@testset "Shared wave phase range and remainder" begin
    F=UTDKernels.FaceEdge
    S=UTDKernels.ShadowSensitivity
    for factors in ((1e308,3.,cos(.2)),(1e308,3.,cos(1.)),(-1e308,3.),(1e-308,1e-308,1e308))
        reference=setprecision(BigFloat,4096) do
            ComplexF64(exp(im*prod(BigFloat.(factors))))
        end
        @test UTDKernels._cis_product(factors...)≈reference rtol=4eps() atol=nextfloat(0.)
    end
    incident,reflection,_=F.wedge_go_terms(.6,.4,1.2,1e308,3.,:soft)
    reference(factors...)=setprecision(BigFloat,4096) do
        ComplexF64(exp(im*prod(BigFloat.(factors))))
    end
    @test incident≈reference(1e308,3.,cos(.6-.4)) rtol=4eps()
    @test reflection≈-reference(1e308,3.,cos(.6+.4)) rtol=4eps()
    @test UTDKernels._cis_product(Float32(3),Float32(.4)) isa ComplexF32
    phase(x)=UTDKernels._cis_product(x,3.)
    @test ForwardDiff.derivative(phase,.4)≈3im*exp(1.2im) rtol=3eps()
    @test ForwardDiff.derivative(x->ForwardDiff.derivative(phase,x),.4)≈-9exp(1.2im) rtol=4eps()
    # Independent direct integrand uses the canonical full branch, with the
    # zero source-bearing carrier exactly one. No overflowing 2*kappa enters.
    kap=1e308;p=0.;center=pi;radius=.1
    direct=S.receiver_quadrature(center-radius,center+radius;npanels=8) do s
        g=sqrt(2.)*cos(s/2)
        L=-sqrt(kap)*sin(s/2)/sqrt(2.)*S.canonical_kernel(sqrt(kap)*g)
        inner=-im*(kap*((2sin(s))*S.canonical_switch(sqrt(kap)*g)))+L
        -S.receiver_window(s,center,radius)*inner
    end
    @test S.field_sensitivity_remainder(kap,p,center,radius;npanels=8)≈direct rtol=5e-14
end

@testset "Zero phase factors and exact receiver identities" begin
    S=UTDKernels.ShadowSensitivity
    for scale in (1e-200,1e200)
        f(t)=imag(UTDKernels._cis_product(scale,scale,t/scale))
        @test ForwardDiff.derivative(f,0.)≈scale rtol=3eps()
        expected=setprecision(BigFloat,4096) do
            phase=BigFloat(scale)^2*BigFloat(inv(scale))
            Float64(phase*cos(phase))
        end
        @test ForwardDiff.derivative(f,1.)≈expected rtol=5eps()
    end
    @test ForwardDiff.derivative(t->imag(UTDKernels._cis_product(1e-200,1e-200,t*1e200)),0.)≈1e-200 rtol=3eps()
    for large in (1e150,1e300)
        small=inv(large)
        phase(t)=imag(UTDKernels._cis_product(one(t)+large*t,one(t)+small*t,one(t)-large*t))
        @test ForwardDiff.derivative(phase,0.)≈cos(1.)*small rtol=4eps()
    end
    @test S._osc_jets(4,1e308,0.,.8)==ComplexF64[1,0,0,0,0]
    for k in (8.,1e308),radius in (.8,1.2)
        @test S.misfit_local_jets(k,0.,pi,radius)≈-S.bump_jets(pi,pi,radius,4) rtol=3eps()
    end
    @test all(iszero,S.misfit_local_jets(1e308,.8,0.,.1))
    predicted=ForwardDiff.derivative(p->real(S.misfit_local_jets(8.,p,pi,.8)[1]),0.)
    reference=ForwardDiff.derivative(p->real(S.receiver_window(p+pi,pi,.8)*
        conj(S.sommerfeld_demodulated(p+pi,p,8.)-1)),0.)
    @test predicted≈reference rtol=2e-13 atol=2e-14
    @test_throws DomainError S.misfit_local_jets(1e308,.8,.8+pi,.8)
end
end
