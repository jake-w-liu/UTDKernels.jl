"""
Numerical tolerance constants.
"""

# Transition (shadow/reflection-boundary) detector tolerance.
# The gated quantities are dimensionless: a cosine g = cos((2nπNⱼ - βⱼ)/2)
# in the KP regime classification, and grazing angles φ' measured in radians on
# a wedge face. A point is in the transition zone only when this dimensionless
# quantity is genuinely degenerate (≈ 0). The floor for a well-conditioned
# dimensionless test in double precision is √eps, the standard relative tolerance
# below which the value is numerically indistinguishable from zero.
const NUMERICS_TRANSITION_TOL = sqrt(eps(Float64))
const DEFAULT_TRANSITION_TOL = NUMERICS_TRANSITION_TOL

@inline function _transition_tolerance(values::Real...)
    types = map(value -> typeof(float(_primal_value(value))), values)
    T = promote_type(types...)
    return sqrt(eps(T))
end

# Extract the primal scalar from an AD number without depending on a particular
# differentiation package. This is used for discrete branch decisions; the
# differentiable arithmetic remains on the original number.
@inline function _primal_value(x::Real)
    hasproperty(x, :value) && return _primal_value(getproperty(x, :value))
    return x
end

# Exact seam identity must use the primal value, not a tolerance that would
# reclassify nearby periodic angles.
@inline _primal_iszero(x::Real) = iszero(_primal_value(x))

@inline function _scalar_allfinite(x::Real)
    isfinite(x) || return false
    if hasproperty(x, :partials)
        hasproperty(x, :value) &&
            !_scalar_allfinite(getproperty(x, :value)) && return false
        for partial in getproperty(x, :partials)
            _scalar_allfinite(partial) || return false
        end
    end
    return true
end

@inline _number_isfinite(x::Number) =
    _scalar_allfinite(real(x)) && _scalar_allfinite(imag(x))

@inline _scalar_contains_ad(x::Real) = hasproperty(x, :partials)
@inline _number_contains_ad(x::Number) =
    _scalar_contains_ad(real(x)) || _scalar_contains_ad(imag(x))

@inline function _validate_finite_number(x::Number, name::AbstractString)
    _number_isfinite(x) || throw(DomainError(x, "$name must be finite"))
    return x
end

@inline function _validate_wavenumber(k::Real)
    k_primal = _primal_value(k)
    isfinite(k_primal) && k_primal > zero(k_primal) ||
        throw(DomainError(k, "wavenumber k must be finite and positive"))
    return k
end

@inline function _validate_wavenumber(k::Complex)
    real_primal = _primal_value(real(k))
    _number_isfinite(k) && real_primal > zero(real_primal) ||
        throw(DomainError(k, "complex wavenumber k must be finite with positive real part"))
    return k
end

# Exact zero includes every AD coefficient; package versions can differ in
# whether their scalar iszero predicate inspects only the primal value.
function _scalar_exact_zero(x::Real)
    iszero(_primal_value(x)) || return false
    hasproperty(x,:partials) || return true
    return _scalar_exact_zero(getproperty(x,:value)) &&
        all(_scalar_exact_zero,getproperty(x,:partials))
end

# Nonzero leaf exponent bounds are shared by Cartesian and phase products.
function _scalar_leaf_range(x::Real)
    if hasproperty(x,:partials)
        low,high=_scalar_leaf_range(getproperty(x,:value))
        for partial in getproperty(x,:partials)
            l,h=_scalar_leaf_range(partial)
            low=min(low,l);high=max(high,h)
        end
        return low,high
    end
    magnitude=abs(float(x))
    return iszero(magnitude) ? (Inf,-Inf) : (log2(magnitude),log2(magnitude))
end

# A bounded unit phasor can remain representable when a product forming its
# argument overflows. Keep the ordinary product/evaluator, recovering from the
# original factors only when range or supported AD coefficients require it.
function _cis_product(first::Real, second::Real, rest::Real...)
    values=(float(first),float(second),float.(rest)...)
    all(_number_isfinite,values) || throw(DomainError(values,"wave phase factors must be finite"))
    phase=prod(values)
    lost_nonzero=_primal_iszero(phase) && all(v->!_primal_iszero(v),values)
    ad_range_risk=false
    ad_work_bits=0
    if any(_number_contains_ad,values)
        R=promote_type((typeof(float(_primal_value(v))) for v in values)...)
        ranges=_scalar_leaf_range.(values)
        low=minimum(Base.first,ranges);high=maximum(Base.last,ranges)
        degree=length(values)*(1+maximum(_shadow_ad_depth,values))
        ad_range_risk=(isfinite(low) && degree*low<log2(floatmin(R))+16) ||
            (isfinite(high) && degree*high>log2(floatmax(R))-16)
        span=isfinite(low) && isfinite(high) ? max(zero(high),high-low) : 0
        # Phase magnitude alone misses a tiny derivative between cancelling
        # large seed products. Budget precision from the original leaf span.
        stored=maximum(v->precision(float(_primal_value(v))),values)
        ad_work_bits=2stored+length(values)*span+64*(1+maximum(_shadow_ad_depth,values))
    end
    if _number_isfinite(phase) && !lost_nonzero && !ad_range_risk
        result=exp(im*phase)
        _number_isfinite(result) && return result
    end
    R=_shadow_binary_type(values...)
    C=typeof(complex(sum(zero(v) for v in values)))
    magnitudes=abs.(_primal_value.(values))
    bits=any(iszero,magnitudes) ? 192 :
        _shadow_phase_work_precision(sum(log,magnitudes),R)
    ad_work_bits<=_SHADOW_MAX_PHASE_WORK_BITS || throw(ArgumentError(
        "wave phase AD products exceed the bounded wide workspace"))
    bits=max(bits,ceil(Int,ad_work_bits))
    return _shadow_stable_wide_value(C,bits,"wave phase product") do
        exp(im*prod(_faddeeva_widen_bigfloat.(values)))
    end
end

# Some supported AD packages compare only primal values in isequal. Wide
# recovery must stabilize every requested nilpotent coefficient instead.
function _scalar_isequal_all(a::Real,b::Real)
    aa=hasproperty(a,:partials);bb=hasproperty(b,:partials)
    aa==bb || return false
    aa || return isequal(a,b)
    _scalar_isequal_all(getproperty(a,:value),getproperty(b,:value)) || return false
    pa,pb=getproperty(a,:partials),getproperty(b,:partials)
    length(pa)==length(pb) || return false
    return all(_scalar_isequal_all(x,y) for (x,y) in zip(pa,pb))
end
_number_isequal_all(a::Number,b::Number)=
    _scalar_isequal_all(real(a),real(b)) && _scalar_isequal_all(imag(a),imag(b))
_number_isequal_all(a::AbstractArray{<:Number},b::AbstractArray{<:Number})=
    axes(a)==axes(b) && all(_number_isequal_all(a[i],b[i]) for i in eachindex(a,b))
_number_isequal_all(a::NamedTuple,b::NamedTuple)=
    keys(a)==keys(b) && all(_number_isequal_all(x,y) for (x,y) in zip(values(a),values(b)))
