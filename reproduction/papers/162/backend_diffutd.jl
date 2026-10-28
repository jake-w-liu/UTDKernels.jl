module DiffFaceEdgeStudy
using UTDKernels, DiffUTD, StaticArrays, LinearAlgebra
const K = UTDKernels.FaceEdge

# Reference evaluators, geometry generation, and ranking policy have one owner.
# Physical coefficients and fields below enter through Cartesian DiffUTD APIs.
for name in (:BoundaryGeometry, :edge_residual_hp, :edge_residual_direct,
             :edge_residual_first_order, :coefficient_four_term,
             :geometric_defect_score, :condition_number, :angle_factor,
             :wedge_go_terms, :cylinder_total, :cylinder_radial_derivative,
             :regular_polygon, :warped_polygon, :vertex_wedge, :vertex_defect,
             :vertex_probe_L, :vertex_lit, :select_modes,
             :incident_shadow_active, :INCIDENT_KLA_GUARD, :INCIDENT_FIELD_TOL)
    @eval const $name = K.$name
end

function spatial_edge(epsilon)
    turn = oftype(epsilon, π)*epsilon
    T=typeof(float(epsilon))
    return StraightEdge{T}(SVector(zero(T),zero(T),zero(T)),SVector(zero(T),zero(T),one(T)),
        SVector(cos(turn),sin(turn),zero(T)),SVector(one(T),zero(T),zero(T)),T(Inf),Wedge(oftype(turn,π)+turn))
end

direction(phi)=SVector(cos(phi),sin(phi),zero(phi))

function central_split(g::BoundaryGeometry)
    edge=spatial_edge(g.eps)
    alpha=pi*(1+g.eps)
    c=face_edge_coefficients(edge,direction((alpha-g.Delta)/2),
                            direction((alpha+g.Delta)/2),g.k,g.L;midpoint=true)
    return (;edge=c.De,face=c.Df,Ds=c.Ds,Dh=c.Dh)
end

function normalized_intrinsic_score(g::BoundaryGeometry)
    alpha=pi*(1+g.eps)
    return DiffUTD.face_edge_intrinsic_score(spatial_edge(g.eps),
        direction((alpha-g.Delta)/2),direction((alpha+g.Delta)/2),g.k,g.L)
end

function edge_scores(p,k,rule;Ls=nothing)
    n=length(p.verts)
    Ls===nothing || length(Ls)==n || throw(DimensionMismatch("one distance per vertex is required"))
    return [begin
        epsilon=vertex_defect(p,j);L=Ls===nothing ? vertex_probe_L(p,j) : Ls[j]
        g=BoundaryGeometry(epsilon,0.,k,L)
        if rule===:intrinsic
            normalized_intrinsic_score(g)
        elseif rule===:angle
            epsilon
        elseif rule===:raw
            c=central_split(g);sqrt(2pi*k)*max(abs(c.Ds),abs(c.Dh))
        else
            throw(ArgumentError("unknown edge score rule: $rule"))
        end
    end for j=1:n]
end

function wedge_coefficients(phi,phip,n,k,L)
    c=face_edge_coefficients(spatial_edge(n-1),direction(phip),direction(phi),k,L)
    return (;c.Ds,c.Dh,c.De,c.Df)
end

function wedge_pair_terms(phi,phip,n,k,L)
    c=wedge_coefficients(phi,phip,n,k,L)
    scale=-cis(-pi/4)/(2n*sqrt(2pi))
    return c.De/scale,c.Df/scale
end

function wedge_total_field(phi,phip,n,k,s,pol;mode=:full)
    axis=SVector(0.,0.,1.);d=-direction(phip)
    polarization=pol===:soft ? axis : pol===:hard ? cross(axis,d) :
        throw(ArgumentError("pol must be :soft or :hard"))
    pw=PlaneWave(d,polarization,one(k),k)
    result=face_edge_total_field(pw,s*direction(phi),spatial_edge(n-1);mode)
    return dot(axis,pol===:soft ? result.field.electric : result.field.magnetic)
end

function spatial_polygon(p)
    points=[SVector(x,y,zero(x+y)) for (x,y) in p.verts]
    return ExtrudedPolygon(points;axis=SVector(0.,0.,1.),origin=SVector(0.,0.,0.),reference=SVector(1.,0.,0.))
end

function polygon_field(p,point,beta,k,pol;modes=:full)
    axis=SVector(0.,0.,1.);d=-direction(beta)
    polarization=pol===:soft ? axis : pol===:hard ? cross(axis,d) :
        throw(ArgumentError("pol must be :soft or :hard"))
    pw=PlaneWave(d,polarization,one(k),k)
    x=SVector(point[1],point[2],zero(point[1]+point[2]))
    result=polygon_face_edge_field(pw,x,spatial_polygon(p);modes)
    return dot(axis,pol===:soft ? result.field.electric : result.field.magnetic)
end

polygon_total(p,point,beta,k,pol;kwargs...)=polygon_field(p,point,beta,k,pol;kwargs...)
polygon_go(p,point,beta,k,pol)=polygon_field(p,point,beta,k,pol;modes=:delete)
end
