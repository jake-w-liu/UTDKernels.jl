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
