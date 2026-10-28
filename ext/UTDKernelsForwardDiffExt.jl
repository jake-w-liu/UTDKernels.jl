"""
Package extension: ForwardDiff support for UTDKernels.

Provides a stable large-argument derivative rule for `F_utd(::Dual)` and forward-mode
AD rules for `erfcx(::Complex{Dual})` and `erfc(::Complex{Dual})` so transition
values, passive derivatives, bivariate
switches, and wedge coefficients can be differentiated through by ForwardDiff.

The complex derivative of erfcx is:
    d/dz erfcx(z) = 2z·erfcx(z) − 2/√π
"""
module UTDKernelsForwardDiffExt

import UTDKernels
import SpecialFunctions
import ForwardDiff: Dual, Tag, value, partials, Partials

# Differentiate F itself: the erfcx product's large leading terms cancel in
# its derivative. F_utd_prime evaluates that derivative with its stable
# inverse-power series while the primal value keeps the fast erfcx evaluator.
function UTDKernels.F_utd(x::Dual{T,V,N}) where {T,V,N}
    primal = UTDKernels._primal_value(x)
    if UTDKernels._passive_supported_type(x) && isfinite(primal) &&
       primal >= UTDKernels.MIN_F_PRIME_ASYMPTOTIC_THRESHOLD
        x_value = value(x)
        f_value = UTDKernels.F_utd(x_value)
        derivative = UTDKernels.F_utd_prime(x_value + zero(Float64))
        seeds = partials(x)
        return Complex(
            Dual{T}(real(f_value), Partials(ntuple(j -> real(derivative)*seeds[j], Val(N)))),
            Dual{T}(imag(f_value), Partials(ntuple(j -> imag(derivative)*seeds[j], Val(N)))),
        )
    end
    return UTDKernels._F_utd_erfcx(x)
end

function SpecialFunctions.erfcx(z::Complex{Dual{T,V,N}}) where {T,V,N}
    # Primal evaluation
    z_val = Complex(value(real(z)), value(imag(z)))
    f_val = SpecialFunctions.erfcx(z_val)

    # d/dz erfcx(z) = 2z·erfcx(z) − 2/√π
    scalar = real(z_val)
    two = oftype(scalar, 2)
    pi_value = oftype(scalar, π)
    df_dz = two * z_val * f_val - two / sqrt(pi_value)

    # Propagate partials via complex chain rule:
    #   df = df_dz · dz,  where dz = d(Re z) + i·d(Im z)
    p_re = partials(real(z))
    p_im = partials(imag(z))
    dr = ntuple(i -> real(df_dz) * p_re[i] - imag(df_dz) * p_im[i], Val(N))
    di = ntuple(i -> imag(df_dz) * p_re[i] + real(df_dz) * p_im[i], Val(N))

    return Complex(
        Dual{T}(real(f_val), Partials(dr)),
        Dual{T}(imag(f_val), Partials(di)),
    )
end

function SpecialFunctions.erfc(z::Complex{Dual{T,V,N}}) where {T,V,N}
    z_val = Complex(value(real(z)), value(imag(z)))
    f_val = SpecialFunctions.erfc(z_val)
    scalar = real(z_val)
    two = oftype(scalar, 2)
    pi_value = oftype(scalar, π)
    df_dz = -two / sqrt(pi_value) * exp(-z_val * z_val)

    p_re = partials(real(z))
    p_im = partials(imag(z))
    dr = ntuple(i -> real(df_dz) * p_re[i] - imag(df_dz) * p_im[i], Val(N))
    di = ntuple(i -> imag(df_dz) * p_re[i] + real(df_dz) * p_im[i], Val(N))
    return Complex(
        Dual{T}(real(f_val), Partials(dr)),
        Dual{T}(imag(f_val), Partials(di)),
    )
end

# Real first-order rules avoid constructing a BigFloat nilpotent workspace
# for ordinary products. Complex Duals retain the parent's generic AD path;
# nested real Duals are handled below. Unsafe intermediates recover there.
@inline function _shadow_ordinary_product(a::V,b::V,result::V) where {V<:AbstractFloat}
    isfinite(result) || return false
    (iszero(a)||iszero(b)) && return iszero(result)
    return !iszero(result) && abs(result)>=floatmin(V)
end

function _shadow_first_order_lift(x::Dual{Tag,V,N},primal,derivative) where {Tag,V<:Union{Float32,Float64},N}
    seeds=partials(x)
    re=ntuple(i -> real(derivative)*seeds[i],Val(N))
    im=ntuple(i -> imag(derivative)*seeds[i],Val(N))
    safe=all(i -> _shadow_ordinary_product(real(derivative),seeds[i],re[i]) && _shadow_ordinary_product(imag(derivative),seeds[i],im[i]),1:N)
    safe || return nothing
    return Complex(Dual{Tag}(real(primal),Partials(re)),Dual{Tag}(imag(primal),Partials(im)))
end

function UTDKernels.shadow_switch(x::Dual{Tag,V,N}) where {Tag,V<:Union{Float32,Float64},N}
    UTDKernels._number_isfinite(x) || throw(DomainError(x,"shadow coordinate and seeds must be finite"))
    xv=value(x)
    primal=UTDKernels.shadow_switch(xv)
    derivative=UTDKernels.shadow_sensitivity_kernel(xv)
    result=_shadow_first_order_lift(x,primal,derivative)
    return result===nothing ? invoke(UTDKernels.shadow_switch,Tuple{Number},x) : result
end

function UTDKernels.shadow_sensitivity_kernel(x::Dual{Tag,V,N}) where {Tag,V<:Union{Float32,Float64},N}
    UTDKernels._number_isfinite(x) || throw(DomainError(x,"shadow coordinate and seeds must be finite"))
    xv=value(x)
    primal=UTDKernels.shadow_sensitivity_kernel(xv)
    ar=2imag(primal);ai=-2real(primal)
    dr=ar*xv;di=ai*xv
    if _shadow_ordinary_product(ar,xv,dr) && _shadow_ordinary_product(ai,xv,di)
        result=_shadow_first_order_lift(x,primal,complex(dr,di))
        result===nothing || return result
    end
    return invoke(UTDKernels.shadow_sensitivity_kernel,Tuple{Number},x)
end

# Nested real rules propagate the analytic derivatives recursively. Per-leaf
# range and cancellation guards recover through the same bounded wide owner.
_shadow_nested_binary(::Type{Float32})=true
_shadow_nested_binary(::Type{Float64})=true
_shadow_nested_binary(::Type)=false
_shadow_nested_binary(::Type{Dual{T,V,N}}) where {T,V,N}=_shadow_nested_binary(V)

Base.@noinline function _shadow_nested_sum(a::V,b::V,c::V) where {V<:Union{Float32,Float64}}
    isfinite(c) || return false
    iszero(a) && iszero(b) && return iszero(c)
    iszero(c) && return false
    abs(c)>=floatmin(V) || return false
    # Reserve roundoff margin across nested products: recover when cancellation
    # can amplify local errors beyond the fourth-root-epsilon scale. The factor
    # 16 leaves room for product/addition errors at successive derivative levels.
    limit=16sqrt(sqrt(eps(V)))
    return signbit(a)==signbit(b) || abs(c)>limit*max(abs(a),abs(b))
end
Base.@noinline function _shadow_nested_sum(a::D,b::D,c::D) where {D<:Dual}
    _shadow_nested_sum(value(a),value(b),value(c)) || return false
    return all(i->_shadow_nested_sum(partials(a)[i],partials(b)[i],partials(c)[i]),eachindex(partials(c)))
end
Base.@noinline _shadow_nested_product(a::V,b::V,c::V) where {V<:Union{Float32,Float64}}=
    _shadow_ordinary_product(a,b,c)
Base.@noinline function _shadow_nested_product(a::D,b::D,c::D) where {D<:Dual}
    _shadow_nested_product(value(a),value(b),value(c)) || return false
    return all(eachindex(partials(c))) do i
        left=partials(a)[i]*value(b)
        right=value(a)*partials(b)[i]
        _shadow_nested_product(partials(a)[i],value(b),left) &&
        _shadow_nested_product(value(a),partials(b)[i],right) &&
        _shadow_nested_sum(left,right,partials(c)[i])
    end
end
function _shadow_nested_lift(x::Dual{T,V,N},primal,derivative) where {T,V<:Dual,N}
    seeds=partials(x)
    re=ntuple(i->real(derivative)*seeds[i],Val(N))
    im=ntuple(i->imag(derivative)*seeds[i],Val(N))
    all(i->_shadow_nested_product(real(derivative),seeds[i],re[i]) &&
        _shadow_nested_product(imag(derivative),seeds[i],im[i]),1:N) || return nothing
    return Complex(Dual{T}(real(primal),Partials(re)),Dual{T}(imag(primal),Partials(im)))
end
Base.@noinline _shadow_nested_recover(::Val{:switch},x)=invoke(UTDKernels.shadow_switch,Tuple{Number},x)
Base.@noinline _shadow_nested_recover(::Val{:kernel},x)=invoke(UTDKernels.shadow_sensitivity_kernel,Tuple{Number},x)
function UTDKernels.shadow_switch(x::Dual{T,V,N}) where {T,V<:Dual,N}
    UTDKernels._number_isfinite(x) || throw(DomainError(x,"shadow coordinate and seeds must be finite"))
    _shadow_nested_binary(V) || return _shadow_nested_recover(Val(:switch),x)
    v=value(x)
    primal=UTDKernels.shadow_switch(v)
    derivative=UTDKernels.shadow_sensitivity_kernel(v)
    result=_shadow_nested_lift(x,primal,derivative)
    return result===nothing ? _shadow_nested_recover(Val(:switch),x) : result
end
function UTDKernels.shadow_sensitivity_kernel(x::Dual{T,V,N}) where {T,V<:Dual,N}
    UTDKernels._number_isfinite(x) || throw(DomainError(x,"shadow coordinate and seeds must be finite"))
    _shadow_nested_binary(V) || return _shadow_nested_recover(Val(:kernel),x)
    v=value(x)
    primal=UTDKernels.shadow_sensitivity_kernel(v)
    ar=2imag(primal);ai=-2real(primal)
    dr=ar*v;di=ai*v
    if _shadow_nested_product(ar,v,dr) && _shadow_nested_product(ai,v,di)
        result=_shadow_nested_lift(x,primal,complex(dr,di))
        result===nothing || return result
    end
    return _shadow_nested_recover(Val(:kernel),x)
end

end # module
