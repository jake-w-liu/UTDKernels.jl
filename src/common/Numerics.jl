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
