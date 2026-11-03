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

# An empty output tuple carries no element type; use its corresponding primal.
@inline _shadow_output_partials(primal,p::Tuple{}) = Partials{0,typeof(primal)}(p)
@inline _shadow_output_partials(primal,p::Tuple) = Partials(p)

function UTDKernels._shadow_round_work_value(x::Dual{T,V,N}, bits::Int) where {T,V,N}
    primal = UTDKernels._shadow_round_work_value(value(x), bits)
    seeds = Partials{N,typeof(primal)}(ntuple(j -> UTDKernels._shadow_round_work_value(partials(x)[j], bits), Val(N)))
    return Dual{T}(primal, seeds)
end

# Differentiate F itself with the shared real derivative owner; its ordinary
# primal value keeps the erfcx evaluator. Unsafe original germs recover before
# derivative coefficients can underflow or overflow in native arithmetic.
function UTDKernels.F_utd(x::Dual{T,V,N}) where {T,V,N}
    primal = UTDKernels._primal_value(x)
    if UTDKernels._passive_supported_type(x) && isfinite(primal) && !iszero(primal) &&
       (primal > 0 || abs(primal) < UTDKernels._REAL_TRANSITION_GAUSSIAN_MINIMUM)
        UTDKernels._number_isfinite(x) || throw(DomainError(
            x, "transition coordinate and seeds must be finite",
        ))
        # Below 4, complex sqrt sees the original scalar type before the
        # phase promotes it. Check that native range, including Float32.
        native_type = primal < UTDKernels._REAL_TRANSITION_GAUSSIAN_MINIMUM ?
            UTDKernels._passive_base_type(x) : Float64
        native_power = primal < UTDKernels._REAL_TRANSITION_GAUSSIAN_MINIMUM ? 1 : 3
        risk, _ = UTDKernels._real_transition_work_parameters((x,), native_type; native_power)
        risk && return UTDKernels._real_transition_ad_value_wide(x, UTDKernels.F_utd(primal))
        if primal >= UTDKernels._REAL_TRANSITION_GAUSSIAN_MINIMUM
            f_value = UTDKernels.F_utd(value(x))
            derivative = UTDKernels.F_utd_prime(value(x) + zero(Float64))
            seeds = partials(x)
            result = Complex(
                Dual{T}(real(f_value), _shadow_output_partials(real(f_value), ntuple(j -> real(derivative)*seeds[j], Val(N)))),
                Dual{T}(imag(f_value), _shadow_output_partials(imag(f_value), ntuple(j -> imag(derivative)*seeds[j], Val(N)))),
            )
            UTDKernels._number_isfinite(result) && return result
            return UTDKernels._real_transition_ad_value_wide(x, f_value)
        end
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
        Dual{T}(real(f_val), _shadow_output_partials(real(f_val), dr)),
        Dual{T}(imag(f_val), _shadow_output_partials(imag(f_val), di)),
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
        Dual{T}(real(f_val), _shadow_output_partials(real(f_val), dr)),
        Dual{T}(imag(f_val), _shadow_output_partials(imag(f_val), di)),
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
    return Complex(Dual{Tag}(real(primal),_shadow_output_partials(real(primal),re)),Dual{Tag}(imag(primal),_shadow_output_partials(imag(primal),im)))
end

# A first-order real derivative needs F' and F'' only once. Nested or unsafe
# germs retain the generic canonical path and its bounded wide recovery.
function UTDKernels.F_utd_prime(
    x::Dual{Tag,V,N}; asymptotic_threshold::Real=64.0,
) where {Tag,V<:Union{Float32,Float64},N}
    point = UTDKernels._primal_value(x)
    threshold = UTDKernels._primal_value(asymptotic_threshold)
    if point >= UTDKernels._REAL_TRANSITION_GAUSSIAN_MINIMUM &&
       threshold in (UTDKernels.MIN_F_PRIME_ASYMPTOTIC_THRESHOLD, UTDKernels._LEGACY_F_PRIME_THRESHOLD)
        risk, _ = UTDKernels._real_transition_work_parameters((x,); native_power=3)
        if !risk
            coordinate = UTDKernels._faddeeva_widen_float64(x)
            derivatives = UTDKernels._real_transition_derivatives(value(coordinate))
            lifted = _shadow_first_order_lift(coordinate, derivatives[1], derivatives[2])
            lifted !== nothing && return lifted
        end
    end
    return invoke(UTDKernels.F_utd_prime, Tuple{Real}, x; asymptotic_threshold)
end

function UTDKernels.shadow_switch(x::Dual{Tag,V,N}) where {Tag,V<:Union{Float32,Float64},N}
    UTDKernels._number_isfinite(x) || throw(DomainError(x,"shadow coordinate and seeds must be finite"))
    xv=value(x)
    primal=UTDKernels.shadow_switch(xv)
    derivative=UTDKernels.shadow_sensitivity_kernel(xv)
    result=_shadow_first_order_lift(x,primal,derivative)
    return result===nothing ? (_shadow_ad_recover(Val(:switch),x)::Complex{Dual{Tag,V,N}}) : result
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
    return _shadow_ad_recover(Val(:kernel),x)::Complex{Dual{Tag,V,N}}
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
    return Complex(Dual{T}(real(primal),_shadow_output_partials(real(primal),re)),Dual{T}(imag(primal),_shadow_output_partials(imag(primal),im)))
end
# Avoid inferring the bounded wide workspace at every Dual nesting type.
# Supported binary callers assert the generic evaluator's coordinate type.
Base.@noinline Base.@nospecializeinfer _shadow_ad_recover(::Val{:switch}, @nospecialize(x))=invoke(UTDKernels.shadow_switch,Tuple{Number},x)
Base.@noinline Base.@nospecializeinfer _shadow_ad_recover(::Val{:kernel}, @nospecialize(x))=invoke(UTDKernels.shadow_sensitivity_kernel,Tuple{Number},x)
function UTDKernels.shadow_switch(x::Dual{T,V,N}) where {T,V<:Dual,N}
    UTDKernels._number_isfinite(x) || throw(DomainError(x,"shadow coordinate and seeds must be finite"))
    _shadow_nested_binary(V) || return _shadow_ad_recover(Val(:switch),x)
    v=value(x)
    primal=UTDKernels.shadow_switch(v)
    derivative=UTDKernels.shadow_sensitivity_kernel(v)
    result=_shadow_nested_lift(x,primal,derivative)
    return result===nothing ? (_shadow_ad_recover(Val(:switch),x)::Complex{Dual{T,V,N}}) : result
end
function UTDKernels.shadow_sensitivity_kernel(x::Dual{T,V,N}) where {T,V<:Dual,N}
    UTDKernels._number_isfinite(x) || throw(DomainError(x,"shadow coordinate and seeds must be finite"))
    _shadow_nested_binary(V) || return _shadow_ad_recover(Val(:kernel),x)
    v=value(x)
    primal=UTDKernels.shadow_sensitivity_kernel(v)
    ar=2imag(primal);ai=-2real(primal)
    dr=ar*v;di=ai*v
    if _shadow_nested_product(ar,v,dr) && _shadow_nested_product(ai,v,di)
        result=_shadow_nested_lift(x,primal,complex(dr,di))
        result===nothing || return result
    end
    return _shadow_ad_recover(Val(:kernel),x)::Complex{Dual{T,V,N}}
end

end # module
