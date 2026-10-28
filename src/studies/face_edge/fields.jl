# Total-field evaluators for the reflection-boundary studies.
#
# Geometry: two-dimensional PEC wedge, faces at local angles 0 and
# alpha = n*pi, exterior region 0 < phi < alpha, normal incidence,
# exp(+i omega t) phasor convention. Plane-wave illumination carries
# incident amplitude exp(i k d_src . x) at position x, where d_src is the
# unit vector pointing from the edge toward the source. The outgoing
# cylindrical factor is exp(-i k r)/sqrt(r), and the distance parameter
# equals the physical observation distance, L = s.

using SpecialFunctions: besselj, hankelh2

"""Unit-vector convenience: direction angle -> (cos, sin)."""
@inline _dir(theta::Real) = (cos(theta), sin(theta))

"""
    wedge_go_visibility(phi,phip,n;detunings=nothing)

GO visibility consistent with the parent's positive local-pole convention.
Optional Cartesian `detunings` must be the tuple used for coefficient evaluation.
Only near a pole do their signs replace rounded-azimuth comparisons.
"""
function wedge_go_visibility(phi::Real,phip::Real,n::Real;detunings=nothing)
    all(UTDKernels._number_isfinite,(phi,phip,n)) && 0<UTDKernels._primal_value(n)<=2 || throw(ArgumentError("wedge angles and index must be finite with 0<n<=2"))
    terms=UTDKernels.kp_four_terms(phi,phip,n)
    detunings===nothing || (length(detunings)==4 && all(UTDKernels._number_isfinite,detunings)) || throw(ArgumentError("detunings must contain four finite offsets"))
    incident=abs(UTDKernels._primal_value(phi-phip))<=pi
    reflection0=UTDKernels._primal_value(phi+phip)<=pi
    reflectionn=UTDKernels._primal_value(phi+phip-(2n-1)*pi)>=0
    for j in 1:4
        offset=detunings===nothing ? UTDKernels._kp_transition_detuning(j,terms,n) : detunings[j]
        threshold=sqrt(eps(typeof(float(UTDKernels._primal_value(offset)))))
        if abs(UTDKernels._primal_value(terms.aj[j]))<=threshold && abs(UTDKernels._primal_value(sin(terms.psi[j])))<=threshold
            if j<=2
                incident=UTDKernels._primal_value(offset)>=0
            elseif j==3
                reflectionn=UTDKernels._primal_value(offset)>=0
            else
                reflection0=UTDKernels._primal_value(offset)>=0
            end
        end
    end
    return (;incident,reflection0,reflectionn)
end

"""
    wedge_go_terms(phi, phip, n, k, s, pol)

Geometrical-optics terms of the canonical wedge at normal incidence with
unit incident amplitude at the edge. `pol` is `:soft` (R=-1) or `:hard`
(R=+1). Returns `(u_incident, u_reflected_0face, u_reflected_nface)`.

Visibility follows the standard two-face sectors. At an exact transition the
shared positive local-pole convention assigns the lit GO constituent, so its
jump cancels the corresponding regularized diffraction term. Supply the same
Cartesian `detunings` to this function and coefficient evaluation.
"""
function wedge_go_terms(phi::Real, phip::Real, n::Real, k::Real, s::Real,
                        pol::Symbol;detunings=nothing)
    all(isfinite,(k,s)) && k>0 && s>0 || throw(ArgumentError("k and observation distance must be positive and finite"))
    R = pol === :soft ? -1 : pol === :hard ? 1 : throw(ArgumentError("pol must be :soft or :hard"))
    visible=wedge_go_visibility(phi,phip,n;detunings=detunings)
    S=phi+phip;d=phi-phip
    absent=zero(complex(phi+phip+n+k+s))
    ui=visible.incident ? UTDKernels._cis_product(k,s,cos(d)) : absent
    ur0=visible.reflection0 ? R*UTDKernels._cis_product(k,s,cos(S)) : absent
    urn=visible.reflectionn ? R*UTDKernels._cis_product(k,s,cos(S-2n*pi)) : absent
    return ui,ur0,urn
end

"""
    wedge_pair_terms(phi, phip, n, k, L)

Evaluate the canonical coefficient's incident pair and reflection pair at
arbitrary interior angles. Returns `(incident_sum, reflection_sum)`, the
cotangent-transition products including `1/sqrt(k)`, before the remaining
common factor `-exp(-i pi/4)/(2n sqrt(2 pi))`. Nearest-pole branches and finite
transition limits share the parent kernel. Optional Cartesian `detunings`
supply stable local cotangent offsets from the actual edge geometry.
"""
function wedge_pair_terms(phi::Real, phip::Real, n::Real, k::Real, L::Real; detunings=nothing)
    all(isfinite,(phi,phip,n,k)) && 0 < n <= 2 && k > 0 || throw(ArgumentError("angles and n must be finite, 0 < n <= 2, and k positive and finite"))
    L > 0 && (isfinite(L) || isinf(L)) || throw(ArgumentError("L must be positive and finite or +Inf"))
    terms = UTDKernels.kp_four_terms(phi, phip, n)
    detunings === nothing || (length(detunings)==4 && all(isfinite,detunings)) || throw(ArgumentError("detunings must contain four finite local offsets"))
    values = ntuple(4) do j
        UTDKernels._cot_F_regularized(terms.psi[j],terms.aj[j],k,L;
            n=n, detuning=detunings === nothing ? UTDKernels._kp_transition_detuning(j,terms,n) : detunings[j],
            scale=inv(sqrt(k)))
    end
    prototype = complex(zero(float(phi)) + zero(float(phip)) + zero(float(n)) +
                        zero(float(k)) + zero(float(L)))
    if detunings !== nothing
        for offset in detunings
            prototype += zero(complex(float(offset)))
        end
    end
    C = typeof(prototype)
    incident, reflection = convert(C, values[1]+values[2]), convert(C, values[3]+values[4])
    all(UTDKernels._number_isfinite, (incident, reflection)) || throw(DomainError(
        (incident, reflection), "wedge pair exceeds the promoted numeric range"))
    return incident, reflection
end

"""
    wedge_coefficients(phi, phip, n, k, L)

Soft and hard Kouyoumjian--Pathak coefficients at arbitrary interior
angles, assembled as `(Ds, Dh, De, Df)` with `Dh = De + Df` and
`Ds = De - Df`. The half-difference `Df` is the reflection pair at every
geometry; on the midpoint slice phi+phip = n*pi it coincides with the
single collapsed face term.
"""
function wedge_coefficients(phi::Real, phip::Real, n::Real, k::Real, L::Real; detunings=nothing)
    inc, ref = wedge_pair_terms(phi, phip, n, k, L; detunings=detunings)
    # wedge_pair_terms already includes 1/sqrt(k).
    work = real(zero(inc) + zero(ref))
    scale = -conj(UTDKernels._shadow_quarter_phase(work)) /
            (2n * sqrt(2UTDKernels._typed_pi(work)))
    De = scale * inc
    Df = scale * ref
    return (; Ds=De - Df, Dh=De + Df, De=De, Df=Df)
end

# ------------------------------------------------------------ incident-shadow
# safeguard (rejected-edge preservation contract)
#
# Retaining only the reflection pair protects the GO reflection switches at
# S = pi and S = (2n-1)*pi, but it drops the incident pair, which carries the
# cancellation of the incident GO discontinuity at |phi - phip| = pi. The
# symmetric midpoint score cannot detect that loss: it probes the slice
# phi + phip = n*pi, while the incident switch lives off-slice.
#
# The safeguard therefore retains the complete incident pair (the full
# coefficient) at an actual (phi, phip) geometry whenever
#
#   * the incident transition region is active,
#     min(k*s*a_{1,2}) <= INCIDENT_KLA_GUARD, where a_{1,2} are the
#     nearest-pole distance parameters of the two delta terms and s is the
#     observation distance (L = s under plane-wave illumination); this bound
#     is geometric and does not depend on magnitude cancellation between the
#     two terms, or
#   * the actual omitted incident-pair field abs(Di)/sqrt(s) exceeds
#     INCIDENT_FIELD_TOL (unit-incident normalization). This is the declared
#     per-edge field tolerance: wherever both tests fail, the omitted
#     incident contribution is below field_tol at that observation point.
#
# A symmetric-midpoint score alone never authorizes removal.

"""Geometric incident-transition guard band in `kL*a` units (declared)."""
const INCIDENT_KLA_GUARD = 8.0

"""Absolute per-edge field tolerance for an omitted incident pair (declared)."""
const INCIDENT_FIELD_TOL = 0.05

"""
    incident_shadow_active(delta, n, k, s, De; kla_guard, field_tol, incident_amplitude=1)

Decide whether the incident pair must be retained at an actual geometry.
`delta = phi - phip`, `n` the wedge index, `s` the observation distance
(L = s for plane-wave illumination), and `De` the already evaluated
incident pair `Di(phi, phip)` at the same geometry. `incident_amplitude`
scales the physical omission without rescaling the declared field tolerance.
Returns true when the
incident-shadow transition is active or when omitting the pair would cost
more than `field_tol` of unit-incident field.
"""
function _incident_omission_exceeds(De,s,amplitude,tolerance)
    re,impart=UTDKernels._primal_value(real(De)),UTDKernels._primal_value(imag(De))
    amp=UTDKernels._primal_value(float(amplitude))
    tol=UTDKernels._primal_value(float(tolerance))
    distance=UTDKernels._primal_value(float(s))
    (iszero(amp) || (iszero(re) && iszero(impart))) && return false
    iszero(tol) && return true
    magnitude=hypot(re,impart)
    ratio=magnitude/sqrt(distance)
    omitted=amp*ratio
    R=promote_type(typeof(float(re)),typeof(float(impart)),typeof(amp),typeof(tol),typeof(distance))
    if isfinite(ratio) && ratio>0 && isfinite(omitted) && omitted>=floatmin(R) &&
       abs(omitted-tol)>16eps(R)*max(omitted,tol)
        return omitted>tol
    end
    # Squaring this positive inequality avoids both square-root rounding at
    # the budget boundary and unrepresentable intermediate amplitudes.
    bits=max(256,4maximum(x->precision(float(x)),(re,impart,amp,tol,distance))+64)
    bits<=UTDKernels._SHADOW_MAX_PHASE_WORK_BITS || throw(ArgumentError("incident amplitude comparison exceeds the bounded workspace"))
    return setprecision(BigFloat,bits) do
        rb,ib,ab,tb,sb=BigFloat.((re,impart,amp,tol,distance))
        ab^2*(rb^2+ib^2)>tb^2*sb
    end
end

function incident_shadow_active(delta::Real, n::Real, k::Real, s::Real,
                                De::Number;
                                kla_guard::Real=INCIDENT_KLA_GUARD,
                                field_tol::Real=INCIDENT_FIELD_TOL,
                                incident_amplitude::Real=1)
    all(UTDKernels._number_isfinite, (delta,n,k,s)) && 0<n<=2 && k>0 && s>0 ||
        throw(ArgumentError("finite delta, 0 < n <= 2, and positive finite k,s are required"))
    UTDKernels._number_isfinite(De) || throw(ArgumentError("incident coefficient must be finite"))
    all(UTDKernels._number_isfinite,(kla_guard,field_tol,incident_amplitude)) &&
        kla_guard>=0 && field_tol>=0 && incident_amplitude>=0 ||
        throw(ArgumentError("incident guard, field tolerance and amplitude must be finite and nonnegative"))
    _incident_omission_exceeds(De,s,incident_amplitude,field_tol) && return true
    d,nv,kv,sv=promote(UTDKernels._primal_value.((float(delta),float(n),float(k),float(s)))...)
    terms=UTDKernels.kp_four_terms(d,zero(d),nv)
    amin=minimum(2sin(nv*UTDKernels._kp_transition_detuning(j,terms,nv))^2 for j in 1:2)
    root=UTDKernels._scaled_sqrt_product(kv,sv,amin)
    return root<=sqrt(UTDKernels._primal_value(float(kla_guard)))
end

"""
    wedge_total_field(phi, phip, n, k, s, pol; mode=:full)

Total geometrical-optics-plus-UTD field at observation point (s, phi) for
plane-wave illumination from bearing phip, unit incident amplitude at the
edge, and physical distance L = s. `mode` selects the edge treatment:

* `:full`     - complete coefficient (accepted edge);
* `:preserve` - reflection pair plus the incident-shadow safeguard: the
  incident pair is retained wherever `incident_shadow_active` fires;
* `:preserve_reflection` - reflection pair only (the unguarded rule; loses
  the incident GO transition where that transition is active);
* `:delete`   - drop the entire diffracted contribution (naive pruning).
"""
function wedge_total_field(phi::Real, phip::Real, n::Real, k::Real,
                           s::Real, pol::Symbol; mode::Symbol=:full)
    ui, ur0, urn = wedge_go_terms(phi, phip, n, k, s, pol)
    ud = zero(complex(phi + phip + n + k + s))
    if mode !== :delete
        c = wedge_coefficients(phi, phip, n, k, s)
        ud = if mode === :full
            pol === :soft ? c.Ds : c.Dh
        elseif mode === :preserve_reflection
            pol === :soft ? -c.Df : c.Df
        elseif mode === :preserve
            if incident_shadow_active(phi - phip, n, k, s, c.De)
                pol === :soft ? c.Ds : c.Dh
            else
                pol === :soft ? -c.Df : c.Df
            end
        else
            throw(ArgumentError(
                "mode must be :full, :preserve, :preserve_reflection, or :delete"))
        end
    end
    return ui + ur0 + urn + ud * UTDKernels._cis_product(-k,s) / sqrt(s)
end

function _cylinder_controls(k, a, rho, phi, beta, pol, M)
    all(isfinite, (k, a, rho, phi, beta)) && k > 0 && a > 0 && rho >= a ||
        throw(ArgumentError("require finite k>0, a>0, rho>=a, and finite angles"))
    pol in (:soft, :hard) || throw(ArgumentError("pol must be :soft or :hard"))
    ka, kr = k * a, k * rho
    isfinite(ka) && isfinite(kr) && ka > 0 && kr > 0 ||
        throw(DomainError((ka, kr), "cylinder electrical distances exceed the active numeric range"))
    if M === nothing
        estimate = ka + 10cbrt(ka) + 20
        isfinite(estimate) && estimate < typemax(Int) ||
            throw(ArgumentError("cylinder series order exceeds the active integer range"))
        M = ceil(Int, estimate)
    end
    0 <= M <= typemax(Int) || throw(ArgumentError("M must be a nonnegative representable Int"))
    return ka, kr, Int(M)
end

"""
    cylinder_scattered(k, a, rho, phi, beta, pol; M)

Exact eigenfunction series for a PEC circular cylinder of radius `a`
under plane-wave illumination arriving from bearing `beta`, evaluated at
(rho, phi), for `pol = :soft` (Dirichlet) or `:hard` (Neumann).
`M` defaults to a converged truncation `ka + 10 ka^(1/3) + 20`.
"""
function cylinder_scattered(k::Real, a::Real, rho::Real, phi::Real,
                            beta::Real, pol::Symbol;
                            M::Union{Nothing,Integer}=nothing)
    ka, kr, M = _cylinder_controls(k, a, rho, phi, beta, pol, M)
    total = zero(complex(k + a + rho + phi + beta))
    for m in 0:M
        cm = if pol === :soft
            besselj(m, ka) / hankelh2(m, ka)
        elseif pol === :hard
            (besselj(m - 1, ka) - besselj(m + 1, ka)) /
            (hankelh2(m - 1, ka) - hankelh2(m + 1, ka))
        else
            throw(ArgumentError("pol must be :soft or :hard"))
        end
        total += (m == 0 ? 1 : 2) * (im)^m * cm * hankelh2(m, kr) *
                 cos(m * (phi - beta))
    end
    UTDKernels._number_isfinite(total) || throw(DomainError(total,"cylinder field exceeds the active numeric range"))
    return -total
end

"""Incident plane wave exp(i k d_src . x) plus the cylinder series."""
function cylinder_total(k::Real, a::Real, rho::Real, phi::Real,
                        beta::Real, pol::Symbol; kwargs...)
    scattered = cylinder_scattered(k, a, rho, phi, beta, pol; kwargs...)
    ui = UTDKernels._cis_product(k,rho,cos(phi-beta))
    return ui + scattered
end

"""
    cylinder_radial_derivative(k, a, rho, phi, beta, pol; M)

Radial derivative of the total cylinder field, used to verify the
Neumann boundary condition d(u_i + u_s)/d rho = 0 at rho = a analytically.
"""
function cylinder_radial_derivative(k::Real, a::Real, rho::Real,
                                    phi::Real, beta::Real, pol::Symbol;
                                    M::Union{Nothing,Integer}=nothing)
    ka, kr, M = _cylinder_controls(k, a, rho, phi, beta, pol, M)
    dui = im * k * cos(phi - beta) * UTDKernels._cis_product(k,rho,cos(phi-beta))
    dus = zero(complex(k + a + rho + phi + beta))
    for m in 0:M
        cm = if pol === :soft
            besselj(m, ka) / hankelh2(m, ka)
        elseif pol === :hard
            (besselj(m - 1, ka) - besselj(m + 1, ka)) /
            (hankelh2(m - 1, ka) - hankelh2(m + 1, ka))
        else
            throw(ArgumentError("pol must be :soft or :hard"))
        end
        hp = k * (hankelh2(m - 1, kr) - hankelh2(m + 1, kr)) / 2
        dus += (m == 0 ? 1 : 2) * (im)^m * cm * hp * cos(m * (phi - beta))
    end
    result = dui - dus
    UTDKernels._number_isfinite(result) || throw(DomainError(result,"cylinder derivative exceeds the active numeric range"))
    return result
end

# ------------------------------------------------------------ polygon object

"""
    PolygonScatterer(verts)

Convex polygonal PEC scatterer for the multi-edge study. `verts` is a
vector of `(x, y)` vertices ordered counterclockwise. Each vertex is a
PEC wedge whose faces are the rays toward its two neighbors.
"""
struct PolygonScatterer{T<:Real}
    verts::Vector{Tuple{T,T}}
    function PolygonScatterer(verts)
        length(verts) >= 3 || throw(ArgumentError("polygon needs at least three vertices"))
        all(point -> length(point)==2,verts) || throw(ArgumentError("polygon vertices must have two components"))
        values = promote((float(v) for point in verts for v in point)...)
        all(isfinite, values) || throw(ArgumentError("polygon vertices must be finite"))
        points = [(values[2j-1], values[2j]) for j in 1:length(verts)]
        turns = typeof(first(values))[]
        for j in eachindex(points)
            a, b, c = points[j], points[mod1(j+1,length(points))], points[mod1(j+2,length(points))]
            incoming = UTDKernels._study_unit_direction2(a, b)
            outgoing = UTDKernels._study_unit_direction2(b, c)
            cross = _cross2(incoming, outgoing)
            cross > 0 || throw(ArgumentError("polygon must be strictly convex and counterclockwise"))
            push!(turns, atan(cross, incoming[1]*outgoing[1]+incoming[2]*outgoing[2]))
        end
        UTDKernels._study_validate_ccw_turns(turns)
        new{typeof(first(values))}(points)
    end
end

"""Regular N-gon inscribed in a circle of radius R (vertices on the circle)."""
function regular_polygon(N::Int, R::Real)
    N >= 3 && isfinite(R) && R > 0 || throw(ArgumentError("require N >= 3 and positive finite R"))
    return PolygonScatterer([(R * cos(2pi * j / N), R * sin(2pi * j / N))
                             for j in 0:N-1])
end

"""
Warped N-gon: vertices on the circle of radius R with graded angular
density, `theta_j = pi + pi*(1 - cos(pi*j/N))`. The cluster sits at
theta = pi (toward a left-side source), so the flattest seams are
illuminated. Vertex j (1-based, theta-index m = j-1) has coplanar defect
`eps_j = sin(pi*m/N)*sin(pi/N)` for `1 <= m <= N-1`; the wrap vertex
m = 0 has `eps = 2*sin^2(pi/(2N))`.
"""
function warped_polygon(N::Int, R::Real)
    N >= 3 && isfinite(R) && R > 0 || throw(ArgumentError("require N >= 3 and positive finite R"))
    return PolygonScatterer([begin
        t = pi + pi * (1 - cos(pi * j / N))
        (R * cos(t), R * sin(t))
    end for j in 0:N-1])
end

"""Per-vertex wedge data: 0-face bearing, n-face bearing, wedge index n."""
function vertex_wedge(p::PolygonScatterer, j::Int)
    nv = length(p.verts)
    jp = mod1(j - 1, nv)
    jn = mod1(j + 1, nv)
    x, y = p.verts[j]
    xp, yp = p.verts[jp]
    xn, yn = p.verts[jn]
    a0 = UTDKernels._study_angle2((x,y), (xp,yp))
    an = UTDKernels._study_angle2((x,y), (xn,yn))
    n = mod(an - a0, 2pi) / pi
    return (; a0=a0, an=an, n=n, r=(x, y), rp=(xp, yp), rn=(xn, yn))
end

_wrap2pi(t) = mod(t, 2pi)

"""Local angle of a global bearing `b` measured from the 0-face."""
@inline _local_angle(b, a0) = _wrap2pi(b - a0)

"""
Vertex j is illuminated when the bearing to the source lies inside its
exterior wedge (convex polygon). Observation is symmetric.
"""
function vertex_lit(p::PolygonScatterer, j::Int, beta_src::Real)
    UTDKernels._number_isfinite(beta_src) || throw(ArgumentError("source bearing must be finite"))
    return _local_angle(beta_src, vertex_wedge(p,j).a0) < vertex_wedge(p,j).n*pi
end

function vertex_visible(p::PolygonScatterer, j::Int, point)
    w = vertex_wedge(p, j)
    bearing = UTDKernels._study_angle2(w.r,point)
    return _local_angle(bearing, w.a0) < w.n * pi
end

"""Facet between vertices f and f+1: midpoint, unit outward normal."""
function facet_data(p::PolygonScatterer, f::Int)
    nv = length(p.verts)
    a = p.verts[f]
    b = p.verts[mod1(f + 1, nv)]
    direction = UTDKernels._study_unit_direction2(a, b)
    # CCW polygon: the outward normal is the right-hand normal of a -> b.
    return (; a=a, b=b, n=(direction[2], -direction[1]))
end

function _ray_segment_wide(point,dirv,a,b)
    values=promote(point...,dirv...,a...,b...)
    _,bits=_cartesian_controls(values)
    flags=setprecision(BigFloat,bits) do
        p1,p2,d1,d2,a1,a2,b1,b2=BigFloat.(UTDKernels._primal_value.(values))
        e1,e2=b1-a1,b2-a2;r1,r2=a1-p1,a2-p2
        determinant=d1*e2-d2*e1
        iszero(determinant) && return nothing
        t=(r1*e2-r2*e1)/determinant
        tau=(r1*d2-r2*d1)/determinant
        (;inside=0<=tau<=1,forward=t>0)
    end
    flags===nothing && return nothing
    C=typeof(complex(sum(zero(value) for value in values)))
    result=_cartesian_wide(values,C) do x
        p1,p2,d1,d2,a1,a2,b1,b2=x
        e1,e2=b1-a1,b2-a2;r1,r2=a1-p1,a2-p2
        determinant=d1*e2-d2*e1
        complex((r1*e2-r2*e1)/determinant,(r1*d2-r2*d1)/determinant)
    end
    return (;t=real(result),tau=imag(result),flags...)
end

"""
    ray_segment_intersection(point,direction,a,b)

Adaptive Cartesian line/segment arithmetic. Parallel lines return `nothing`;
otherwise returns `(t,tau,inside,forward)`, with true ray parameter `t`, segment
fraction `tau`, closed segment sidedness, and strict forward-ray sidedness.
"""
function ray_segment_intersection(point, dirv, a, b)
    point,dirv=_cartesian_vectors2(point,dirv)
    a,b=_cartesian_vectors2(a,b)
    values=promote(point...,dirv...,a...,b...)
    point,dirv,a,b=(values[1],values[2]),(values[3],values[4]),(values[5],values[6]),(values[7],values[8])
    # Normalize lengths before determinants: t scales with the geometry,
    # while tau and all sidedness decisions are dimensionless.
    eraw=(b[1]-a[1],b[2]-a[2])
    rraw=(a[1]-point[1],a[2]-point[2])
    braw=(b[1]-point[1],b[2]-point[2])
    risk,_=_cartesian_controls(values)
    if !all(UTDKernels._number_isfinite,(eraw...,rraw...,braw...)) || risk
        return _ray_segment_wide(point,dirv,a,b)
    end
    scale=max(abs(eraw[1]),abs(eraw[2]),abs(rraw[1]),abs(rraw[2]))
    iszero(scale) && throw(ArgumentError("ray segment has zero geometric scale"))
    e=eraw./scale;r=rraw./scale
    rb=braw./scale
    if any(pair->!iszero(UTDKernels._primal_value(first(pair))) && iszero(UTDKernels._primal_value(last(pair))),zip((eraw...,rraw...,braw...),(e...,r...,rb...)))
        return _ray_segment_wide(point,dirv,a,b)
    end
    det=_cross2(dirv,e)
    iszero(UTDKernels._primal_value(det)) && return nothing
    numerator_a=_cross2(r,dirv)
    numerator_t=_cross2(r,e)
    # Classify endpoints before normalizing: independently rounded divisions
    # can reverse an exactly represented near-tangent determinant's sign.
    side_a=orientation2(rraw,dirv)
    side_b=orientation2((b[1]-point[1],b[2]-point[2]),dirv)
    determinant_positive=UTDKernels._primal_value(det)>0
    inside=determinant_positive ? side_a>=0 && side_b<=0 : side_a<=0 && side_b>=0
    numerator_primal=UTDKernels._primal_value(numerator_t)
    forward=determinant_positive ? numerator_primal>0 : numerator_primal<0
    return (;t=scale*numerator_t/det,tau=numerator_a/det,inside,forward)
end

function _incident_blocked_wide(p::PolygonScatterer,point,d_src)
    values=(point...,d_src...,(v for vertex in p.verts for v in vertex)...)
    primals=map(UTDKernels._primal_value,values)
    powers=[exponent(abs(v)) for v in primals if !iszero(v)]
    stored=maximum(UTDKernels._curvature_stored_precision(v) for v in primals)
    bits=max(256,2stored+maximum(powers)-minimum(powers)+128)
    bits<=8192 || throw(ArgumentError("geometric ray predicate exceeds the bounded 8192-bit workspace"))
    return setprecision(BigFloat,bits) do
        px,py=BigFloat.(UTDKernels._primal_value.(point))
        dx,dy=BigFloat.(UTDKernels._primal_value.(d_src))
        lower=BigFloat(0);upper=BigFloat(Inf)
        for f in eachindex(p.verts)
            a=p.verts[f];b=p.verts[mod1(f+1,length(p.verts))]
            ax,ay=BigFloat.(UTDKernels._primal_value.(a))
            bx,by=BigFloat.(UTDKernels._primal_value.(b))
            ex,ey=bx-ax,by-ay
            q=ex*(py-ay)-ey*(px-ax)
            slope=ex*dy-ey*dx
            if iszero(slope)
                q>0 || return false
            elseif slope>0
                lower=max(lower,-q/slope)
            else
                upper=min(upper,-q/slope)
            end
            lower<upper || return false
        end
        return lower<upper && upper>0
    end
end

"""True if the source ray intersects the open interior of the convex polygon.
A tangent ray touching only a vertex or facet does not block the incident GO.
"""
function incident_blocked(p::PolygonScatterer, point, d_src)
    length(point)==2 && length(d_src)==2 && all(isfinite,point) && all(isfinite,d_src) || throw(ArgumentError("point and source direction need two finite components"))
    max(abs(d_src[1]),abs(d_src[2]))>0 || throw(ArgumentError("source direction must be nonzero"))
    scale=maximum(max(abs(v[1]-point[1]),abs(v[2]-point[2])) for v in p.verts)
    iszero(scale) && throw(ArgumentError("point and polygon have zero geometric scale"))
    lower=zero(scale);upper=oftype(scale,Inf)
    for f in eachindex(p.verts)
        a=p.verts[f];b=p.verts[mod1(f+1,length(p.verts))]
        ar=((a[1]-point[1])/scale,(a[2]-point[2])/scale)
        br=((b[1]-point[1])/scale,(b[2]-point[2])/scale)
        edge=(br[1]-ar[1],br[2]-ar[2])
        q=_cross2(edge,(-ar[1],-ar[2]))
        slope=_cross2(edge,d_src)
        if iszero(UTDKernels._primal_value(slope)) || iszero(UTDKernels._primal_value(q))
            return _incident_blocked_wide(p,point,d_src)
        elseif slope>0
            lower=max(lower,-q/slope)
        else
            upper=min(upper,-q/slope)
        end
        R=typeof(float(UTDKernels._primal_value(lower)))
        if isfinite(upper) && abs(upper-lower)<=32eps(R)*max(one(lower),abs(lower),abs(upper))
            return _incident_blocked_wide(p,point,d_src)
        end
        lower<upper || return false
    end
    return lower<upper && upper>0
end

"""
    polygon_go_rays(p, point, beta_src)

Visible direct and single-reflection rays of a convex polygon. Each reflected
ray records its facet, outward normal, anchor and bearing toward the image
source. Field assemblers share this visibility decision and match phase at the
recorded facet anchor.
"""
function polygon_go_rays(p::PolygonScatterer, point, beta_src::Real)
    all(isfinite, (point[1], point[2], beta_src)) || throw(ArgumentError("point and source bearing must be finite"))
    d_src = _dir(beta_src)
    fd0 = facet_data(p, firstindex(p.verts))
    nd0 = fd0.n[1] * d_src[1] + fd0.n[2] * d_src[2]
    r0 = (d_src[1]-2nd0*fd0.n[1], d_src[2]-2nd0*fd0.n[2])
    template = (; facet=firstindex(p.verts), normal=fd0.n, anchor=fd0.a, source_bearing=r0)
    reflections = typeof(template)[]
    for f in eachindex(p.verts)
        fd = facet_data(p, f)
        ndot = fd.n[1] * d_src[1] + fd.n[2] * d_src[2]
        ndot <= 0 && continue
        r_src = (d_src[1] - 2ndot * fd.n[1], d_src[2] - 2ndot * fd.n[2])
        hit = ray_segment_intersection(point, r_src, fd.a, fd.b)
        hit === nothing && continue
        if hit.forward && hit.inside
            push!(reflections, (; facet=f, normal=fd.n, anchor=fd.a, source_bearing=r_src))
        end
    end
    return (; direct=!incident_blocked(p, point, d_src), reflections)
end

"""
    polygon_transition_detunings(p,j,point,beta_src)

Four signed local cotangent-pole offsets at vertex j. Near a GO switch they
are formed from Cartesian ray determinants, sharing the sidedness used by
`polygon_go_rays`, rather than subtracting rounded azimuths near pi.
"""
function polygon_transition_detunings(p::PolygonScatterer,j::Int,point,beta_src::Real)
    w=vertex_wedge(p,j)
    phi=_local_angle(UTDKernels._study_angle2(w.r,point),w.a0)
    phip=_local_angle(beta_src,w.a0)
    face0=(w.rp[1]-w.r[1],w.rp[2]-w.r[2])
    facen=(w.rn[1]-w.r[1],w.rn[2]-w.r[2])
    observer=(point[1]-w.r[1],point[2]-w.r[2])
    return wedge_cartesian_detunings(phi,phip,w.n,face0,facen,observer,_dir(beta_src))
end

"""
    wedge_cartesian_detunings(phi,phip,n,face0,facen,observer,source)

Signed local KP pole offsets from an oriented two-dimensional edge chart.
The vectors `face0`, `facen` point away from the edge along its two faces;
the exterior wedge is counterclockwise from `face0` to `facen`. `observer`
and `source` point from the edge toward the receiver and source. Vector
magnitudes are immaterial. The supplied azimuths are used away from a pole;
near a pole a shared Cartesian determinant retains the actual local side.
Pass the returned tuple as `detunings` to `wedge_coefficients`.
"""
function wedge_cartesian_detunings(phi::Real,phip::Real,n::Real,face0,facen,observer,source)
    all(isfinite,(phi,phip,n)) && 0<n<=2 || throw(ArgumentError("wedge angles and index must be finite with 0<n<=2"))
    all(v -> length(v)==2 && all(isfinite,v),(face0,facen,observer,source)) || throw(ArgumentError("Cartesian edge directions must contain two finite components"))
    all(v -> max(abs(v[1]),abs(v[2]))>0,(face0,facen,observer,source)) || throw(ArgumentError("Cartesian edge directions must be nonzero"))
    terms=UTDKernels.kp_four_terms(phi,phip,n)
    # Keep original Cartesian differences for the sidedness shared with ray
    # intersection; normalizing them independently can reverse an ULP offset.
    displacement=observer
    return ntuple(4) do term
        detuning=UTDKernels._kp_transition_detuning(term,terms,n)
        threshold=sqrt(eps(typeof(float(UTDKernels._primal_value(detuning)))))
        if abs(terms.aj[term])<=threshold && abs(sin(terms.psi[term]))<=threshold
            bearing = if term<=2
                source
            else
                face=term==3 ? facen : face0
                unit=unit_direction2((zero(face[1]),zero(face[2])),face)
                normal=term==3 ? (unit[2],-unit[1]) : (-unit[2],unit[1])
                ndot=normal[1]*source[1]+normal[2]*source[2]
                (source[1]-2ndot*normal[1],source[2]-2ndot*normal[2])
            end
            travel=(-bearing[1],-bearing[2])
            deviation=signed_angle2(travel,displacement)
            sign=term==1 || term==3 ? 1 : -1
            return sign*deviation/(2n)
        end
        return detuning
    end
end

"""
Geometrical-optics field of the polygon at `point` under a plane wave
arriving from bearing `beta_src` (d_src points toward the source), for
`pol = :soft` or `:hard`.
"""
function polygon_go(p::PolygonScatterer, point, beta_src::Real, k::Real,
                    pol::Symbol)
    isfinite(k) && k > 0 || throw(ArgumentError("k must be positive and finite"))
    R = pol === :soft ? -1 : pol === :hard ? 1 : throw(ArgumentError("pol must be :soft or :hard"))
    d_src = _dir(beta_src)
    rays = polygon_go_rays(p, point, beta_src)
    u = rays.direct ? UTDKernels._cis_product(k,d_src[1]*point[1]+d_src[2]*point[2]) :
        zero(complex(k + beta_src + point[1] + point[2]))
    for ray in rays.reflections
        r_src, normal, anchor = ray.source_bearing, ray.normal, ray.anchor
        ndot = normal[1]*d_src[1] + normal[2]*d_src[2]
        # Match the incident phase at the actual facet, including its offset.
        offset = 2ndot * (normal[1]*anchor[1] + normal[2]*anchor[2])
        u += R * UTDKernels._cis_product(k,r_src[1]*point[1]+r_src[2]*point[2]+offset)
    end
    return u
end

"""
    polygon_total(p, point, beta_src, k, pol; modes)

GO-plus-UTD field of the polygon. `modes` is a vector of per-vertex
treatments (`:full`, `:preserve`, `:preserve_reflection`, or `:delete`),
or a single symbol applied to every vertex. `:preserve` retains the
reflection pair and, through `incident_shadow_active`, the incident pair
wherever that transition is active at the actual vertex geometry;
`:preserve_reflection` retains the reflection pair only. A vertex that is
unlit or cannot see the observation point contributes nothing.
"""
function polygon_total(p::PolygonScatterer, point, beta_src::Real,
                       k::Real, pol::Symbol;
                       modes::Union{Symbol,AbstractVector{Symbol}}=:full)
    nv = length(p.verts)
    modes isa Symbol || length(modes) == nv || throw(ArgumentError("modes must match polygon vertices"))
    all(m -> m in (:full, :preserve, :preserve_reflection, :delete), modes isa Symbol ? (modes,) : modes) || throw(ArgumentError("unknown edge treatment mode"))
    u = polygon_go(p, point, beta_src, k, pol)
    d_src = _dir(beta_src)
    for j in 1:nv
        mode = modes isa Symbol ? modes : modes[j]
        mode === :delete && continue
        w = vertex_wedge(p, j)
        phip = _local_angle(beta_src, w.a0)
        phip >= w.n * pi && continue                # unlit vertex
        bearing = UTDKernels._study_angle2(w.r,point)
        phi = _local_angle(bearing, w.a0)
        phi >= w.n * pi && continue                 # vertex cannot see point
        s = UTDKernels._study_distance2(w.r, point)
        c = wedge_coefficients(phi, phip, w.n, k, s; detunings=polygon_transition_detunings(p,j,point,beta_src))
        ud = if mode === :full
            pol === :soft ? c.Ds : c.Dh
        elseif mode === :preserve_reflection
            pol === :soft ? -c.Df : c.Df
        elseif mode === :preserve
            if incident_shadow_active(phi - phip, w.n, k, s, c.De)
                pol === :soft ? c.Ds : c.Dh
            else
                pol === :soft ? -c.Df : c.Df
            end
        else
            throw(ArgumentError(
                "mode must be :full, :preserve, :preserve_reflection, or :delete"))
        end
        amp = UTDKernels._cis_product(k,d_src[1]*w.r[1]+d_src[2]*w.r[2])
        u += ud * amp * UTDKernels._cis_product(-k,s) / sqrt(s)
    end
    return u
end

# ------------------------------------------------------- selection machinery

"""
    vertex_defect(p, j)

Coplanar defect `eps = n - 1` of vertex j.
"""
vertex_defect(p::PolygonScatterer, j::Int) = vertex_wedge(p, j).n - 1

"""
    vertex_probe_L(p, j)

Local probe distance for score evaluation: the half mean length of the
two adjacent facets (the finite-facet analogue of L_N = R sin(pi/N)).
This is a chosen diagnostic distance, not a physical propagation length.
"""
function vertex_probe_L(p::PolygonScatterer, j::Int)
    w = vertex_wedge(p, j)
    return UTDKernels._study_distance2(w.r ./ 2, w.rp ./ 2) / 2 +
           UTDKernels._study_distance2(w.r ./ 2, w.rn ./ 2) / 2
end

"""
    edge_scores(p, k, rule; Ls=nothing)

Per-vertex selection scores at the symmetric diagnostic geometry.
`rule` is `:intrinsic`, `:angle`, or `:raw`. With `Ls=nothing`, each
diagnostic distance is `vertex_probe_L(p, j)`. Supply `Ls` to choose a
per-vertex distance vector.
"""
function edge_scores(p::PolygonScatterer, k::Real, rule::Symbol;
                     Ls::Union{Nothing,AbstractVector{<:Real}}=nothing)
    nv = length(p.verts)
    Ls===nothing || length(Ls)==nv || throw(ArgumentError("probe distances must match polygon vertices"))
    rule in (:intrinsic,:angle,:raw) || throw(ArgumentError("rule must be :intrinsic, :angle, or :raw"))
    scores = map(1:nv) do j
        epsj = vertex_defect(p, j)
        Ld = Ls === nothing ? vertex_probe_L(p, j) : Ls[j]
        g = BoundaryGeometry(epsj, 0.0, k, Ld)
        if rule === :intrinsic
            normalized_intrinsic_score(g)
        elseif rule === :angle
            epsj
        elseif rule === :raw
            normalized_raw_score(g)
        else
            throw(ArgumentError("rule must be :intrinsic, :angle, or :raw"))
        end
    end
    return scores
end

"""
    select_modes(scores, tau; preserve=true)

Return per-vertex modes: `:full` when `score > tau`, otherwise
`:preserve` or `:delete` when `preserve = false`. Preserve mode retains
the reflection pair and applies the incident-shadow safeguard when the
field is assembled.
"""
function select_modes(scores, tau::Real; preserve::Bool=true)
    isfinite(tau) && tau >= 0 || throw(ArgumentError("selection cutoff must be finite and nonnegative"))
    all(s -> s isa Real && isfinite(s) && s >= 0, scores) ||
        throw(ArgumentError("selection scores must be finite and nonnegative"))
    fallback = preserve ? :preserve : :delete
    return [UTDKernels._primal_value(s) > UTDKernels._primal_value(tau) ? :full : fallback for s in scores]
end
