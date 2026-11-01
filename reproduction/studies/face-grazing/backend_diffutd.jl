module DiffGrazingStudy
using UTDKernels,DiffUTD,StaticArrays

function spatial_evaluation(wedge,angles,k,L;method=:adaptive,soft_only=false,kwargs...)
    T=promote_type(typeof(float(wedge.alpha)),typeof(float(angles.phi)),
        typeof(float(angles.phip)),typeof(float(k)),typeof(float(L)))
    z=zero(T);o=one(T);alpha=T(wedge.alpha)
    q=SVector(z,z,z);axis=SVector(z,z,o);fo=SVector(o,z,z)
    fn=SVector(-cos(alpha),-sin(alpha),z)
    edge=StraightEdge{T}(q,axis,fn,fo,T(Inf),Wedge(alpha))
    source=SVector(cos(angles.phip),sin(angles.phip),zero(angles.phip))
    outgoing=SVector(cos(angles.phi),sin(angles.phi),zero(angles.phi))
    return grazing_ray_coefficients(edge,source,outgoing,k,L;
        method,soft_only,local_angles=angles,kwargs...)
end

function pec_wedge_DsDh(wedge,angles,k,L;kwargs...)
    result=spatial_evaluation(wedge,angles,k,L;method=:fourterm,kwargs...)
    return result.Ds,result.Dh
end

function pec_wedge_DsDh_grazing_info(wedge,angles,k,L;kwargs...)
    result=spatial_evaluation(wedge,angles,k,L;method=:adaptive,kwargs...)
    return result.Ds,result.Dh,result.info
end

function pec_wedge_DsDh_grazing(wedge,angles,k,L;kwargs...)
    result=spatial_evaluation(wedge,angles,k,L;method=:adaptive,kwargs...)
    return result.Ds,result.Dh
end

function pec_wedge_Ds_linear(wedge,angles,k,L;kwargs...)
    return spatial_evaluation(wedge,angles,k,L;method=:linear,soft_only=true,kwargs...).Ds
end

function direct_soft(wedge,angles,k,L)
    return spatial_evaluation(wedge,angles,k,L;method=:direct,soft_only=true,face=:o).Ds
end

function fixed_soft(wedge,angles,k,L;order=8,check_domain=true,transition_margin=1e-3,x_margin=1e-8)
    return spatial_evaluation(wedge,angles,k,L;method=:fixed,soft_only=true,order,
        check_domain,transition_margin,x_margin,face=:o,allow_interior=true,allow_infinite_L=true).Ds
end
end
