module Paper170NormReference
using QuadGK
export sobolev_control,norm_control_record

# Independent derivatives of B_r(s)=exp(1-1/(1-(s/r)^2)).  Put t=1/(1-x²),
# x=s/r. These exponential-polynomial formulas do not call production jets.
function bump_derivatives(s,r)
    x=s/r
    abs(x)>=1 && return (zero(x),zero(x),zero(x))
    t=inv(1-x*x);e=exp(1-t)
    return (-2x*t^2*e/r,
            (4t^4-12t^3+6t^2)*e/r^2,
            x*(-8t^6+56t^5-84t^4+24t^3)*e/r^3)
end

function transformed_derivatives(s,r,gamma,scale)
    f1,f2,f3=bump_derivatives(s,r)
    jac=scale*(1+gamma*s);second_map=scale*gamma
    second=f2/jac^2-f1*second_map/jac^3
    third=f3/jac^3-3 * f2*second_map/jac^4+3 * f1*second_map^2/jac^5
    return second,third,jac
end

"""
Numerically evaluate the tail-free analytic bound

    (2π)^-1 ∫ ξ²|Fhat(ξ)|dξ ≤ sqrt(||F''||₂ ||F'''||₂).

For F=B_r∘g^-1 and g(s)=a(s+gamma*s²/2), derivatives are evaluated in s
and integrated with dy=g'(s)ds. The inequality follows from weighted
Cauchy--Schwarz and Parseval, optimized at ||F'''||₂/||F''||₂.
Quadrature errors below are numerical diagnostics, not interval enclosures.
"""
function sobolev_control(radius::Real,gamma::Real=0;scale::Real=1,precision::Integer=256)
    isfinite(radius) && radius>0 && isfinite(gamma) && isfinite(scale) && scale>0 ||
        throw(ArgumentError("finite radius/scale must be positive and curvature finite"))
    abs(gamma*radius)<1 || throw(ArgumentError("the map must be increasing across the complete probe support"))
    128<=precision<=1024 || throw(ArgumentError("reference precision must lie in128:1024"))
    return setprecision(BigFloat,precision) do
        r,g,a=BigFloat(radius),BigFloat(gamma),BigFloat(scale)
        tolerance=BigFloat(2)^(-2precision÷3)
        integrand2(s)=begin d2,_,jac=transformed_derivatives(s,r,g,a);d2*d2*jac end
        integrand3(s)=begin _,d3,jac=transformed_derivatives(s,r,g,a);d3*d3*jac end
        squared2,error2=quadgk(integrand2,-r,r;rtol=tolerance,atol=tolerance)
        squared3,error3=quadgk(integrand3,-r,r;rtol=tolerance,atol=tolerance)
        error2<=max(tolerance,tolerance*abs(squared2)) &&
        error3<=max(tolerance,tolerance*abs(squared3)) || error("derivative-norm reference did not converge")
        second,third=sqrt(squared2),sqrt(squared3)
        (;second_l2=second,third_l2=third,moment_upper=sqrt(second*third),
          squared_norm_errors=(error2,error3),tolerance,precision)
    end
end

function _record(low,high)
    # Round the displayed control upwards by a cent, far above the observed
    # precision/refinement changes. This is conservative numerical rounding,
    # not a claim that adaptive quadrature is interval certified.
    rounded=nextfloat(Float64(ceil(high.moment_upper*100)/100))
    Dict("second_l2"=>string(high.second_l2),"third_l2"=>string(high.third_l2),
        "moment_upper_evaluated"=>string(high.moment_upper),"moment_upper_rounded"=>rounded,
        "squared_norm_quadrature_errors"=>string.(collect(high.squared_norm_errors)),
        "precision_change"=>string(abs(high.moment_upper-low.moment_upper)),
        "precisions_bits"=>[low.precision,high.precision],"relative_tolerance"=>string(high.tolerance))
end

function norm_control_record(radius::Real,gamma::Real,kappa::Real;scales=(1.,.5,.2))
    isfinite(kappa) && kappa>0 || throw(ArgumentError("kappa must be finite and positive"))
    all(a->isfinite(a) && a>0,scales) && !isempty(scales) ||
        throw(ArgumentError("scales must be nonempty, finite and positive"))
    physical=[sobolev_control(radius,0;precision=p) for p in (128,256)]
    transformed=[sobolev_control(radius,gamma;precision=p) for p in (128,256)]
    p=_record(physical...);f=_record(transformed...)
    Dict("scope"=>"analytic whole-line Sobolev inequality with numerically evaluated derivative norms",
        "normalization"=>"M2=(2pi)^-1 integral xi^2 abs(Fhat(xi)) dxi; remainder expression M2/(4kappa)",
        "qualification"=>"QuadGK diagnostics and precision agreement are not interval certification; no Fourier-tail truncation enters this control.",
        "radius"=>Float64(radius),"curvature_gamma"=>Float64(gamma),"kappa"=>Float64(kappa),
        "physical"=>p,"transformed_base"=>f,
        "physical_remainder_control"=>p["moment_upper_rounded"]/(4kappa),
        "scales"=>collect(scales),
        "transformed_remainder_controls"=>[f["moment_upper_rounded"]/(4kappa*a^2) for a in scales],
        "scaling_identity"=>"F_a(y)=F_1(y/a), so whole-line M2(F_a)=a^-2 M2(F_1)")
end
end
