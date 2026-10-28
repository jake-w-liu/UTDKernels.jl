"""
Physically specified monostatic ray assembly for the artificial-seam study.

A convex faceted cylinder is illuminated by a unit-strength two-dimensional
line source that also serves as the observation point (monostatic
configuration). At every polygon vertex the local wedge exterior angle is
`alpha_j = pi + delta_j`, the Kouyoumjian--Pathak azimuths are equal,
`phi = phi' = alpha_j/2 + eta_j`, and the complete four-term PEC soft and
hard coefficients are evaluated for the face diagnostics. Their even part
`(Ds + Dh)/2` is evaluated by the compensated intrinsic primitive. For `phi = phi'` the difference pair carries the
whole even part, so the half-sum equals `intrinsic_seam_coefficient(delta_j,
k, L_j)` independently of the bisector offset `eta_j` inside the wedge
exterior; the full coefficient is still evaluated explicitly so the assembly
also records the nonvanishing face-transition part.

Field normalization (exp(+i omega t) convention):
- incident line-source field at the vertex: `u^i = exp(-i k rho) / sqrt(rho)`;
- KP cylindrical-wave spreading for a line source diffracted by an infinite
  straight edge at normal incidence: `A_cyl(s) = 1 / sqrt(s)` (Kouyoumjian and
  Pathak, NASA CR-2401, Eq. 47); the spherical-source factor
  `sqrt(s' / (s (s + s')))` does not apply to this illumination;
- return propagation phase `exp(-i k rho)`;

so the per-seam contribution is
`D_{e,j} exp(-2 i k rho_j) / rho_j` with `L_j = rho_j / 2`.
"""
module MonostaticAssembly

using SpecialFunctions: besselj, bessely, loggamma
import ...UTDKernels
using ...UTDKernels:
    Wedge, RayAngles, pec_wedge_DsDh, F_utd, _curvature_local_bias,
    intrinsic_seam_coefficient

export MonostaticSeam, MonostaticFieldResult
export regular_polygon_points, monostatic_seam_geometry
export cylindrical_spreading, incident_cylindrical_wave, linear_prefactor_local
export monostatic_even_field, monostatic_visibility_half_angle
export monostatic_continuum_integrand, monostatic_arc_integral
export cylinder_line_source_scattered, hankel_source_normalization

"""Per-vertex geometric record of a faceted seam under monostatic viewing."""
struct MonostaticSeam{T<:Real}
    "vertex polar angle about the polygon centroid (rad)"
    theta::T
    "exterior turning angle delta_j (rad)"
    delta::T
    "local wedge exterior angle alpha_j = pi + delta_j (rad)"
    alpha::T
    "vertex-to-observer distance rho_j (m)"
    rho::T
    "signed offset eta_j of the ray direction from the exterior bisector (rad)"
    eta::T
    "KP azimuth phi_j = phi'_j = alpha_j/2 + eta_j measured from the o-face (rad)"
    phi_kp::T
    "whether the direction lies inside the local wedge exterior (0 < phi < alpha)"
    visible::Bool
end

"""Result of [`monostatic_even_field`](@ref)."""
struct MonostaticFieldResult{C<:Number,T<:Real,A<:Real,S<:AbstractVector{<:MonostaticSeam}}
    "assembled even (intrinsic) seam field at the observer"
    field::C
    "number of locally visible seams"
    n_visible::Int
    "lower endpoint of the visible turning-angle arc (rad, unwrapped)"
    arc_lo::A
    "upper endpoint of the visible turning-angle arc (rad, unwrapped)"
    arc_hi::A
    "largest |D_e| among visible seams"
    max_abs_edge::T
    "largest |D_f| = |(Dh-Ds)/2| among visible seams"
    max_abs_face::T
    "per-vertex geometric records"
    seams::S
end

"""
    regular_polygon_points(N, R)

Counterclockwise vertices of a regular `N`-gon inscribed in a circle of
radius `R`, as an `(N, 2)` matrix.
"""
function regular_polygon_points(N::Integer, R::Real)
    N >= 3 || throw(ArgumentError("N must be at least 3"))
    UTDKernels._number_isfinite(R) && R > 0 || throw(ArgumentError("R must be positive and finite"))
    return hcat(
        [R * cos(2 * pi * j / N) for j in 0:(N - 1)],
        [R * sin(2 * pi * j / N) for j in 0:(N - 1)],
    )
end

"""
    monostatic_seam_geometry(points, observer)

Return the per-vertex [`MonostaticSeam`](@ref) records for a strictly convex
counterclockwise polygon (`points` is `(N, 2)`) viewed from
`observer = (x, y)`. The polar angle `theta` is measured about the polygon
centroid and coincides with the circumscribed-circle parameter for the
inscribed-circle experiments. The KP azimuth is
`phi = phi' = alpha_j/2 + eta_j`; the seam is visible (and, by reciprocity,
illuminated) when `0 < phi < alpha_j`, i.e. when the direction to the
observer lies inside the local wedge exterior cone.
"""
function monostatic_seam_geometry(
    points::AbstractMatrix{<:Real},
    observer::Tuple{<:Real,<:Real},
)
    size(points, 2) == 2 && size(points, 1) >= 3 ||
        throw(ArgumentError("points must have shape (N,2) with N >= 3"))
    all(UTDKernels._number_isfinite, points) ||
        throw(ArgumentError("polygon points must be finite"))
    all(UTDKernels._number_isfinite, observer) ||
        throw(ArgumentError("observer position must be finite"))
    N = size(points, 1)
    cx,cy = UTDKernels._study_centroid2(points)
    T = promote_type(typeof(float(zero(eltype(points)))), typeof(float(observer[1])), typeof(float(observer[2])))
    seams = MonostaticSeam{T}[]
    sizehint!(seams, N)
    for i in 1:N
        im1 = i == 1 ? N : i - 1
        ip1 = i == N ? 1 : i + 1
        vx, vy = float(points[i, 1]), float(points[i, 2])
        tin = UTDKernels._study_unit_direction2((points[im1,1],points[im1,2]),(vx,vy))
        tout = UTDKernels._study_unit_direction2((vx,vy),(points[ip1,1],points[ip1,2]))
        # Turning angle from the two edge directions (positive for CCW convex).
        cross_z = tin[1] * tout[2] - tin[2] * tout[1]
        dot_e = tin[1] * tout[1] + tin[2] * tout[2]
        delta = atan(cross_z, dot_e)
        delta > 0 ||
            throw(ArgumentError("polygon must be strictly convex and counterclockwise"))
        alpha = pi + delta
        # Exterior wedge bisector: the face rays from the vertex are -tin and
        # +tout; their sum points into the material, so its negative is the
        # outward bisector.
        bx, by = UTDKernels._study_unit_direction2(tout,tin)
        rho = UTDKernels._study_distance2((vx,vy),observer)
        isfinite(rho) && rho > 0 || throw(ArgumentError(
            "observer range must be positive and finite in the active numeric type"))
        ux, uy = UTDKernels._study_unit_direction2((vx,vy),observer)
        eta = atan(bx * uy - by * ux, bx * ux + by * uy)
        phi_kp = alpha / 2 + eta
        theta = UTDKernels._study_angle2((cx,cy),(vx,vy))
        all(UTDKernels._number_isfinite,(theta,delta,alpha,rho,eta,phi_kp)) ||
            throw(DomainError((theta,delta,alpha,rho,eta,phi_kp),
                "seam geometry exceeds the active numeric range"))
        push!(seams, MonostaticSeam{T}(
            theta, delta, alpha, rho, eta, phi_kp, 0 < phi_kp < alpha,
        ))
    end
    UTDKernels._study_validate_ccw_turns([s.delta for s in seams])
    return seams
end

"""
    visible_arc(seams, observer)

Unwrapped lower and upper endpoints of the turning-angle arc covered by the
locally visible seams: the union of their turning cells,
`[theta_first - delta_first/2, theta_last + delta_last/2]`. The records must
retain counterclockwise polygon order and refer to the supplied observer.
Unwrapping follows the contiguous visible block in that cyclic order, so a
translation of the polygon and observer cannot change its angular branch.
For the regular inscribed-circle experiments these are exact turning-cell
endpoints; general polygons retain the stated polar-angle diagnostic.
"""
function visible_arc(
    seams::AbstractVector{<:MonostaticSeam},
    observer::Tuple{<:Real,<:Real},
)
    all(UTDKernels._number_isfinite,observer) || throw(ArgumentError("observer position must be finite"))
    all(s->UTDKernels._number_isfinite(s.theta) && UTDKernels._number_isfinite(s.delta) &&
        0<s.delta<UTDKernels._typed_pi(s.delta),seams) ||
        throw(ArgumentError("seams require finite polar angles and turning angles in (0,pi)"))
    count_visible=count(s->s.visible,seams)
    iszero(count_visible) &&
        throw(ArgumentError("no locally visible seams for this observer"))
    N=length(seams)
    starts=[i for i in eachindex(seams) if seams[i].visible && !seams[mod1(i-1,N)].visible]
    count_visible==N || length(starts)==1 || throw(ArgumentError(
        "visible seams are not contiguous; the convex monostatic construction requires a single contiguous arc"))
    start=count_visible==N ? firstindex(seams) : only(starts)
    first_seam=seams[start]
    theta=first_seam.theta
    previous=theta
    last_seam=first_seam
    for offset in 1:count_visible-1
        last_seam=seams[mod1(start+offset,N)]
        theta+=mod(last_seam.theta-previous,UTDKernels._typed_two_pi(theta))
        previous=last_seam.theta
    end
    arc_lo = first_seam.theta-first_seam.delta/2
    arc_hi = theta+last_seam.delta/2
    return arc_lo, arc_hi
end

"""
    cylindrical_spreading(s)

Kouyoumjian--Pathak spreading factor `A_cyl(s) = 1 / sqrt(s)` for
cylindrical-wave (line-source) illumination of an infinite straight edge at
normal incidence (NASA CR-2401, Eq. 47). The spherical-source factor
`sqrt(s' / (s (s + s')))` is a different illumination model and is not used
here; the effective transition distance remains `L = s' s / (s' + s)`.
"""
function cylindrical_spreading(s::Real)
    UTDKernels._number_isfinite(s) && s > 0 ||
        throw(ArgumentError("spreading distance must be positive and finite"))
    return 1 / sqrt(s)
end

"""
    monostatic_visibility_half_angle(R, d)

Half-angle `t_* = acos(R / d)` of the fixed physical visible arc of a circular
cylinder of radius `R` viewed by a collocated source/observer at distance
`d > R` from the center: a boundary point at polar angle `t` relative to the
source bearing is illuminated when `d cos t - R > 0`, i.e. `|t| < t_*`.
"""
function monostatic_visibility_half_angle(R::Real, d::Real)
    UTDKernels._number_isfinite(R) && UTDKernels._number_isfinite(d) && R > 0 && d > R ||
        throw(ArgumentError("require finite R > 0 and d > R"))
    return acos(R / d)
end

"""
    incident_cylindrical_wave(k, rho)

Unit-strength two-dimensional line-source field `exp(-i k rho)/sqrt(rho)`
with the `exp(+i omega t)` convention.
"""
function incident_cylindrical_wave(k::Real, rho::Real)
    k,rho=promote(float(k),float(rho))
    UTDKernels._number_isfinite(k) && UTDKernels._number_isfinite(rho) && k > 0 && rho > 0 ||
        throw(ArgumentError("k and rho must be positive and finite"))
    phase=k*rho
    radius=UTDKernels._primal_value(rho)
    differentiated=UTDKernels._number_contains_ad(k) || UTDKernels._number_contains_ad(rho)
    wide_radius=differentiated && begin
        radius_squared=radius*radius
        !(isfinite(radius_squared) && radius_squared>0)
    end
    if !UTDKernels._number_isfinite(phase) ||
       wide_radius
        values=promote(float(k),float(rho),0.0)
        return _monostatic_reference_wide(Base.front(values)) do wide
            cis(-wide[1]*wide[2])/sqrt(wide[2])
        end
    end
    return cis(-phase) / sqrt(rho)
end

"""
    linear_prefactor_local(k, L)

Local first-order coefficient `A(k, L)` of the intrinsic seam coefficient,
`D_e = A(k, L) beta(delta) delta`.
"""
function linear_prefactor_local(k::Real, L::Real)
    UTDKernels._number_isfinite(k) && UTDKernels._number_isfinite(L) && k > 0 && L > 0 ||
        throw(ArgumentError("k and L must be positive and finite"))
    return UTDKernels._intrinsic_seam_linear_prefactor(k, L)
end

"""
    monostatic_even_field(points, observer, k; debias=false)

Evaluate the complete canonical PEC soft and hard wedge coefficients at every
locally visible seam of `points` for the monostatic `observer`, evaluate their
even half-sum by the compensated intrinsic primitive, and assemble the seam field
`sum_j D_{e,j} u^i_j A_cyl,j exp(-i k rho_j)` with `L_j = rho_j/2`,
`u^i_j = exp(-i k rho_j)/sqrt(rho_j)`, `A_cyl,j = 1/sqrt(rho_j)`.
When `debias=true`, each extracted coefficient is divided by the exact local
bias `beta(delta_j)`, leaving the curvature quadrature
`sum_j A(k, L_j) delta_j q_j`.
"""
function monostatic_even_field(
    points::AbstractMatrix{<:Real},
    observer::Tuple{<:Real,<:Real},
    k::Real;
    debias::Bool=false,
)
    UTDKernels._number_isfinite(k) && k > 0 ||
        throw(ArgumentError("k must be positive and finite"))
    seams = monostatic_seam_geometry(points, observer)
    arc_lo, arc_hi = visible_arc(seams, observer)
    field = complex(zero(k)+zero(first(seams).rho))
    max_face = zero(real(field))
    max_edge = zero(real(field))
    for s in seams
        s.visible || continue
        wedge = Wedge(s.alpha)
        angles = RayAngles(s.phi_kp, s.phi_kp)
        Ds, Dh = pec_wedge_DsDh(wedge, angles, k, s.rho / 2)
        # Preserve the independently supplied turning defect. Recovering this
        # small even part from two face-dominated coefficients cancels digits.
        d_even = intrinsic_seam_coefficient(s.delta, k, s.rho / 2)
        d_face = (Dh - Ds) / 2
        max_face = max(max_face, abs(d_face))
        max_edge = max(max_edge, abs(d_even))
        debias && (d_even /= _curvature_local_bias(s.delta))
        u_i = incident_cylindrical_wave(k, s.rho)
        # Source and return paths have the same radius. Reuse their common
        # cylindrical wave, including any widened phase reduction.
        field += d_even * u_i * u_i
    end
    return MonostaticFieldResult(
        field, count(s -> s.visible, seams), arc_lo, arc_hi,
        max_edge, max_face, seams,
    )
end

"""
    monostatic_continuum_integrand(R, d, phi_obs, k)

Return `theta -> A(k, L(theta)) q(theta)` for a circular cross section of
radius `R` and a monostatic source/observer on the circle of radius `d`,
where `rho(theta) = sqrt(d^2 + R^2 - 2 d R cos(theta - phi_obs))`,
`L(theta) = rho(theta)/2`, and
`q(theta) = exp(-2 i k rho(theta)) / rho(theta)`
(incident cylindrical wave times the `1/sqrt(rho)` diffracted-ray spreading).
"""
function monostatic_continuum_integrand(R::Real, d::Real, phi_obs::Real, k::Real)
    R,d,phi_obs,k=promote(float(R),float(d),float(phi_obs),float(k))
    all(UTDKernels._number_isfinite, (R, d, phi_obs, k)) ||
        throw(ArgumentError("R, d, phi_obs, and k must be finite"))
    R > 0 && d > R && k > 0 ||
        throw(ArgumentError("require R > 0, d > R, and k > 0"))
    return function (theta)
        theta=float(theta)
        UTDKernels._number_isfinite(theta) || throw(ArgumentError("boundary angle must be finite"))
        # Scale the law of cosines before forming its squares. The positive
        # radial gap is retained even when d and R are adjacent numbers.
        difference=theta-phi_obs
        sine=if UTDKernels._number_isfinite(difference)
            sin(difference/2)
        else
            values=promote(float(theta),float(phi_obs),0.0)
            _monostatic_reference_wide(Base.front(values)) do angles
                sin((angles[1]-angles[2])/2)
            end
        end
        rho = d*hypot((d-R)/d,2sqrt(R/d)*sine)
        isfinite(rho) && rho>0 || throw(DomainError(rho,
            "source-boundary range must be positive and finite"))
        L = rho / 2
        wave=incident_cylindrical_wave(k,rho)
        # Combine the prefactor with one wave before the return wave: q alone
        # can exceed the native range although A*q is finite. The shared wave
        # evaluator also retains phases whose product k*rho exceeds that range.
        return (linear_prefactor_local(k,L)*wave)*wave
    end
end

"""
    monostatic_arc_integral(R, d, phi_obs, k, theta_lo, theta_hi; n_quad)

Composite-trapezoid evaluation of
`int_{theta_lo}^{theta_hi} A(k, L(theta)) q(theta) d theta` using `n_quad`
subintervals. This is the continuum curvature integral over the union of the
turning cells of the visible seams.
"""
function monostatic_arc_integral(
    R::Real, d::Real, phi_obs::Real, k::Real,
    theta_lo::Real, theta_hi::Real;
    n_quad::Integer=262144,
)
    theta_lo,theta_hi=promote(float(theta_lo),float(theta_hi))
    4<=n_quad<=typemax(Int) || throw(ArgumentError("n_quad must be a representable integer of at least 4"))
    n_quad=Int(n_quad)
    UTDKernels._number_isfinite(theta_lo) && UTDKernels._number_isfinite(theta_hi) && theta_hi > theta_lo ||
        throw(ArgumentError("require finite theta_lo < theta_hi"))
    f = monostatic_continuum_integrand(R, d, phi_obs, k)
    width=theta_hi-theta_lo
    finite_width=isfinite(UTDKernels._primal_value(width))
    h = finite_width ? width/n_quad : theta_hi/n_quad-theta_lo/n_quad
    acc = (f(theta_lo) + f(theta_hi)) / 2
    @inbounds for i in 1:(n_quad - 1)
        theta=if finite_width
            theta_lo+h*i
        else
            fraction=i/n_quad
            (1-fraction)*theta_lo+fraction*theta_hi
        end
        acc += f(theta)
    end
    # Divide before multiplying the full interval width: an individually
    # underflowing subinterval can still have a representable integral.
    result=if finite_width
        (acc/n_quad)*width
    else
        values=promote(real(acc),imag(acc),float(theta_lo),float(theta_hi),float(n_quad))
        _monostatic_reference_wide(values) do wide
            complex(wide[1],wide[2])*(wide[4]-wide[3])/wide[5]
        end
    end
    UTDKernels._number_isfinite(result) || throw(DomainError(result,
        "arc integral is outside the active numeric range"))
    return result
end

"""
    cylinder_line_source_scattered(k, a, rho, theta, d_src, theta_src; bc, m_max)

Exact circular-cylinder eigenfunction series for the field scattered by a PEC
cylinder of radius `a` under illumination by a two-dimensional line source at
`(d_src, theta_src)`, evaluated at `(rho, theta)`. With the `exp(+i omega t)`
convention the incident field is the canonical outgoing wave
`u^i = -(i/4) H_0^(2)(k |r - r'|)`, and

    u^s = -(i/4) sum_m R_m H_m^(2)(k rho) H_m^(2)(k d_src)
                 exp(i m (theta - theta_src))

with `R_m = -J_m(ka)/H_m^(2)(ka)` for `bc = :soft` (Dirichlet) and
`R_m = -J_m'(ka)/H_m^(2)'(ka)` for `bc = :hard` (Neumann). The sum is
truncated at `|m| <= m_max`. The default `ceil(ka) + 40` was checked
for the paper geometry `ka=8`, `rho=d_src=3a`. Increase `m_max` and
check convergence for other configurations. Positive and negative modes
are paired by their exact cosine identity. Every requested mode is retained.
When native Bessel factors or their products exceed the native exponent
budget, the same finite series uses wider MPFR intermediates and converts
only its final result back to the promoted working input type.
"""
function cylinder_line_source_scattered(
    k::Real, a::Real, rho::Real, theta::Real,
    d_src::Real, theta_src::Real;
    bc::Symbol=:soft,
    m_max::Union{Nothing,Integer}=nothing,
)
    values=promote(float(k),float(a),float(rho),float(theta),float(d_src),float(theta_src),0.0)
    args=Base.front(values)
    all(UTDKernels._number_isfinite, args) ||
        throw(ArgumentError("all inputs must be finite"))
    k > 0 && a > 0 && rho >= a && d_src >= a ||
        throw(ArgumentError("require k > 0, a > 0, rho >= a, d_src >= a"))
    bc in (:soft, :hard) ||
        throw(ArgumentError("bc must be :soft or :hard"))
    order=if isnothing(m_max)
        # Validate before converting the default order to an integer. Widen
        # the product when it is outside the native exponent range.
        ka=UTDKernels._primal_value(args[1])*UTDKernels._primal_value(args[2])
        if !(isfinite(ka) && ka>0)
            ka=BigFloat(UTDKernels._primal_value(args[1]))*BigFloat(UTDKernels._primal_value(args[2]))
        end
        isfinite(ka) || throw(ArgumentError(
            "default cylinder truncation exceeds the integer range; supply a representable m_max"))
        proposed=(iszero(ka) ? BigInt(1) : ceil(BigInt,ka))+40
        proposed<=typemax(Int) || throw(ArgumentError(
            "default cylinder truncation exceeds the integer range; supply a representable m_max"))
        Int(proposed)
    else
        0<=m_max<=typemax(Int) || throw(ArgumentError("m_max must be a nonnegative representable integer"))
        Int(m_max)
    end
    if _cylinder_requires_wide(args,order)
        return _monostatic_reference_wide(args) do wide_args
            _cylinder_line_series(wide_args,bc,order)
        end
    end
    result=_cylinder_line_series(args,bc,order)
    UTDKernels._number_isfinite(result) || throw(DomainError(result,
        "cylinder reference is outside the active numeric range"))
    return result
end

function _monostatic_reference_wide(evaluate,args)
    work_precision=max(256,2maximum(x->precision(float(UTDKernels._primal_value(x))),args)+64)
    return setprecision(BigFloat,work_precision) do
        wide_args=map(UTDKernels._curvature_widen_bigfloat,args)
        value=evaluate(wide_args)
        Target=value isa Real ? typeof(first(args)) : Complex{typeof(first(args))}
        result=convert(Target,value)
        UTDKernels._number_isfinite(result) || throw(DomainError(result,
            "monostatic evaluation is outside the active numeric range"))
        result
    end
end

function _cylinder_requires_wide(args,order)
    k,a,rho,theta,d_src,theta_src=args
    primal=UTDKernels._primal_value(k)
    primal isa BigFloat && return true
    products=(k*a,k*rho,k*d_src)
    all(x->isfinite(UTDKernels._primal_value(x)) && UTDKernels._primal_value(x)>0,products) || return true
    phase=order*(theta-theta_src)
    UTDKernels._number_isfinite(phase) || return true
    # For small arguments, factorial/power growth bounds the unscaled
    # integer-order factors. Reserve exponent range for the coefficient
    # ratio and both Hankel factors, not only each factor separately.
    log_argument=log(UTDKernels._primal_value(first(products)))
    # The hard m=0 term also uses the first-order Bessel derivative.
    bound_order=max(order,1)
    bound=loggamma(float(bound_order)+1)+bound_order*max(zero(log_argument),log(2)-log_argument)
    return bound>log(floatmax(typeof(primal)))/4
end

function _cylinder_line_series(args,bc,order)
    k,a,rho,theta,d_src,theta_src=args
    ka=k*a
    acc=zero(complex(k))
    angle=theta-theta_src
    for m in 0:order
        Jm = besselj(m, ka)
        Ym = bessely(m, ka)
        Rm = if bc === :soft
            -Jm / complex(Jm, -Ym)
        else
            Jp = besselj(m - 1, ka) - m * Jm / ka
            Yp = bessely(m - 1, ka) - m * Ym / ka
            -Jp / complex(Jp, -Yp)
        end
        H_rho = complex(besselj(m, k * rho), -bessely(m, k * rho))
        H_src = complex(besselj(m, k * d_src), -bessely(m, k * d_src))
        # The -m term equals the +m term except for its angular phase.
        # Pair them exactly as 2cos(m*angle); no mode is omitted.
        weight=iszero(m) ? one(angle) : 2cos(m*angle)
        acc += weight * Rm * H_rho * H_src
    end
    return -(im / 4) * acc
end

"""
    hankel_source_normalization(k, rho)

Ratio between the assembly incident convention `exp(-i k rho)/sqrt(rho)` and
the canonical Green-function convention `-(i/4) H_0^(2)(k rho)` at distance
`rho`. Multiplying a canonical-convention field by this factor expresses it
for the same source strength as the ray assembly.
Underflowing or overflowing native `k*rho` products are formed with wider
intermediates before evaluating the same Hankel normalization.
"""
function hankel_source_normalization(k::Real, rho::Real)
    UTDKernels._number_isfinite(k) && UTDKernels._number_isfinite(rho) && k > 0 && rho > 0 ||
        throw(ArgumentError("k and rho must be positive and finite"))
    args=promote(float(k),float(rho),0.0)
    values=Base.front(args)
    argument=values[1]*values[2]
    if UTDKernels._primal_value(argument) isa BigFloat ||
       !(isfinite(UTDKernels._primal_value(argument)) && UTDKernels._primal_value(argument)>0)
        return _monostatic_reference_wide(values) do wide_values
            _hankel_source_normalization(wide_values...)
        end
    end
    result=_hankel_source_normalization(values...)
    UTDKernels._number_isfinite(result) || throw(DomainError(result,
        "source normalization is outside the active numeric range"))
    return result
end

function _hankel_source_normalization(k,rho)
    H0 = complex(besselj(0, k * rho), -bessely(0, k * rho))
    u_canonical = -(im / 4) * H0
    u_assembly = incident_cylindrical_wave(k, rho)
    return u_assembly / u_canonical
end

end # module
