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

# Extract only the primal zero test from an AD scalar without depending on a
# particular differentiation package. This is used for exact seam identity;
# unlike a tolerance test, it does not reclassify nearby periodic angles.
@inline function _primal_iszero(x::Real)
    hasproperty(x, :value) && return _primal_iszero(getproperty(x, :value))
    return iszero(x)
end

@inline function _validate_wavenumber(k::Real)
    isfinite(k) && k > zero(k) ||
        throw(DomainError(k, "wavenumber k must be finite and positive"))
    return k
end

@inline function _validate_wavenumber(k::Complex)
    isfinite(real(k)) && isfinite(imag(k)) && real(k) > zero(real(k)) ||
        throw(DomainError(k, "complex wavenumber k must be finite with positive real part"))
    return k
end
