module DiffCurvatureStudy
using UTDKernels, DiffUTD, StaticArrays, LinearAlgebra
const K=UTDKernels.CurvatureMeasure
const AXIS=SVector(0.,0.,1.)
const ORIGIN=SVector(0.,0.,0.)
const REFERENCE=SVector(1.,0.,0.)

for name in (:polygon_alias_sum,:smooth_continuum,:relative_error,:fit_log_slope,
             :nonuniform_turning_partition,:ellipse_points,:polygon_turning_angles,
             :ellipse_continuum_reference,:continuum_harmonic,:local_bias_factor,
             :smooth_amplitude,:linear_prefactor)
    @eval const $name=K.$name
end

function intrinsic_coefficient(delta,k,L)
    turn=delta;T=typeof(float(delta+k+L));z=zero(T);o=one(T)
    edge=StraightEdge{T}(SVector(z,z,z),SVector(z,z,o),
        SVector(cos(turn),sin(turn),z),SVector(o,z,z),T(Inf),Wedge(oftype(turn,π)+turn))
    phi=(oftype(turn,π)+turn)/2
    u=SVector(cos(phi),sin(phi),z)
    return face_edge_coefficients(edge,u,u,k,L;midpoint=true).De
end

function circle_data(N)
    N>=3 || throw(ArgumentError("N must be at least 3"))
    theta=2pi .* (0:N-1)./N
    points=[SVector(cos(t),sin(t),zero(t)) for t in theta]
    return theta,points
end

function polygon_measure_sum(N,m,z,phi)
    isfinite(z) && isfinite(phi) || throw(ArgumentError("phase parameters must be finite"))
    theta,points=circle_data(N)
    qhat=SVector(cos(phi),sin(phi),zero(phi))
    amplitudes=[cis(m*t-z*dot(qhat,p)) for (t,p) in zip(theta,points)]
    return curvature_seam_measure(points,amplitudes;axis=AXIS,debias=true)
end

function smooth_polygon_sum(N,z,phi)
    isfinite(z) && isfinite(phi) || throw(ArgumentError("phase parameters must be finite"))
    theta,points=circle_data(N)
    qhat=SVector(cos(phi),sin(phi),zero(phi))
    amplitudes=[smooth_amplitude(t)*cis(-z*dot(qhat,p)) for (t,p) in zip(theta,points)]
    return curvature_seam_measure(points,amplitudes;axis=AXIS,debias=true)
end

function error_decomposition(N,m,z,phi,k,L)
    K._positive_finite_kL(k,L)
    beta=local_bias_factor(2pi/N);s=polygon_measure_sum(N,m,z,phi)
    ref=continuum_harmonic(m,z,phi);total=beta*s-ref
    localpart=(beta-1)*ref;alias=beta*(s-ref)
    return K.ErrorDecomposition(total,localpart,alias,total-localpart-alias)
end

function nonuniform_measure_sum(N,z,phi;debias=true)
    delta,theta=nonuniform_turning_partition(N)
    # Intersections of unit support lines at the turning-cell boundaries
    # form a convex polygon with exactly those turning cells. The local
    # quadrature phase uses the normal directions, as defined by the study.
    points=[SVector(cos(t)/cos(d/2),sin(t)/cos(d/2),zero(t)) for (t,d) in zip(theta,delta)]
    qhat=SVector(cos(phi),sin(phi),zero(phi))
    amplitudes=[smooth_amplitude(t)*cis(-z*dot(qhat,SVector(cos(t),sin(t),zero(t)))) for t in theta]
    return curvature_seam_measure(points,amplitudes;axis=AXIS,debias)
end

function ellipse_field_sum(N,a,b,k,phi;debias=true)
    theta,xy=ellipse_points(N,a,b)
    points=[SVector(xy[j,1],xy[j,2],zero(xy[j,1])) for j in 1:N]
    qhat=SVector(cos(phi),sin(phi),zero(phi))
    amplitudes=[(1+.25cos(3t)-.15sin(2t))*cis(-k*dot(qhat,p)) for (t,p) in zip(theta,points)]
    value=curvature_seam_measure(points,amplitudes;axis=AXIS,debias)
    # Geometry diagnostics use the same physical 3D edge turns.
    p=ExtrudedPolygon(points;axis=AXIS,origin=ORIGIN,reference=REFERENCE)
    turns=[DiffUTD._face_edge_turn(DiffUTD._extruded_vertex_edge(p,j,0.)) for j in 1:N]
    return value,maximum(turns),sum(turns)
end

function seam_coefficients(seam,k)
    T=typeof(float(seam.delta+k));z=zero(T);o=one(T);turn=seam.delta
    edge=StraightEdge{T}(SVector(z,z,z),SVector(z,z,o),
        SVector(cos(turn),sin(turn),z),SVector(o,z,z),T(Inf),Wedge(seam.alpha))
    u=SVector(cos(seam.phi_kp),sin(seam.phi_kp),z)
    c=face_edge_coefficients(edge,u,u,k,seam.rho/2;coincident=true)
    return c.Ds,c.Dh
end

module MonostaticAssembly
using UTDKernels,DiffUTD,StaticArrays
const K=UTDKernels.CurvatureMeasure.MonostaticAssembly
for name in (:regular_polygon_points,:monostatic_seam_geometry,:monostatic_visibility_half_angle,
             :monostatic_arc_integral,:hankel_source_normalization,:cylinder_line_source_scattered)
    @eval const $name=K.$name
end
function monostatic_even_field(xy,observer,k;debias=false)
    points=[SVector(xy[j,1],xy[j,2],zero(xy[j,1])) for j in axes(xy,1)]
    return monostatic_curvature_field(points,SVector(observer[1],observer[2],zero(observer[1])),k;
        axis=SVector(0.,0.,1.),origin=SVector(0.,0.,0.),reference=SVector(1.,0.,0.),debias)
end
end
end
