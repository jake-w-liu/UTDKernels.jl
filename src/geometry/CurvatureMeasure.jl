"""
Curvature-measure limit of intrinsic near-coplanar PEC seam coefficients.

The production coefficient delegates to the Phase 162 reflection-boundary
intrinsic component. This file owns only the turning-angle normalization and
general supplied-measure accumulation.
"""

using SpecialFunctions: besselj

@inline function _validate_curvature_turning_angle(turning_angle::Real)
    primal = _primal_value(turning_angle)
    T = typeof(float(primal))
    pi_value = T(pi)
    (isfinite(primal) && primal > zero(primal) && primal <= pi_value) ||
        throw(DomainError(
            turning_angle,
            "exterior turning angle must be finite and lie in (0, pi]",
        ))
    return turning_angle
end

@inline function _curvature_local_bias_series(turning_angle::Real)
    value = float(turning_angle)
    _validate_curvature_turning_angle(value)
    pi_value = zero(value) + _typed_pi(value)
    return one(value) - 2value / pi_value +
           (one(value) / 12 + 3 / pi_value^2) * value^2
end

@inline function _curvature_local_bias(turning_angle::Real)
    value = float(turning_angle)
    _validate_curvature_turning_angle(value)
    base_type = typeof(float(_primal_value(value)))
    if _primal_value(value) <= cbrt(eps(base_type))
        return _curvature_local_bias_series(value)
    end
    pi_value = zero(value) + _typed_pi(value)
    n = one(value) + value / pi_value
    local_angle = value / (2n)
    return (tan(local_angle) / local_angle) / n^2
end

@inline function _curvature_seam_parameters(turning_angle::Real, k::Real, L::Real)
    turning, wavenumber, distance = promote(
        float(turning_angle), float(k), float(L),
    )
    _validate_curvature_turning_angle(turning)
    _validate_wavenumber(wavenumber)
    distance_primal = _primal_value(distance)
    (isfinite(distance_primal) && distance_primal > zero(distance_primal)) ||
        throw(DomainError(distance, "seam distance L must be finite and positive"))
    pi_value = zero(turning) + _typed_pi(turning)
    epsilon = turning / pi_value
    wedge = Wedge(pi_value + turning)
    parameters = _face_edge_parameters(
        wedge, zero(turning), wavenumber, distance, epsilon,
    )
    return parameters
end

"""
    intrinsic_seam_coefficient(turning_angle, k, L)

Return the symmetric reflection-boundary intrinsic PEC coefficient for an
exterior turning angle in `(0, pi]`. `k` and `L` must be finite and positive.
The implementation uses the cancellation-free Phase 162 face--edge kernel;
it does not define a second transition function.
"""
function intrinsic_seam_coefficient(turning_angle::Real, k::Real, L::Real)
    parameters = _curvature_seam_parameters(turning_angle, k, L)
    n, epsilon, delta, eta, q, wavenumber, distance, pi_value = parameters
    return _face_edge_incident(
        n, epsilon, delta, eta, q, wavenumber, distance, pi_value,
    )
end

@inline function _intrinsic_seam_linear_prefactor(k::Real, L::Real)
    wavenumber, distance = promote(float(k), float(L))
    _validate_wavenumber(wavenumber)
    distance_primal = _primal_value(distance)
    (isfinite(distance_primal) && distance_primal > zero(distance_primal)) ||
        throw(DomainError(distance, "seam distance L must be finite and positive"))
    pi_value = zero(wavenumber) + _typed_pi(wavenumber)
    phase = _face_edge_phase(wavenumber + distance, pi_value)
    return -conj(phase) * _face_edge_B(
        2one(wavenumber), wavenumber, distance, pi_value,
    ) / (2sqrt(2 * (zero(wavenumber) + pi_value)))
end

@inline function _curvature_sum(
    turning_angles::AbstractVector{T},
    amplitudes::AbstractVector{A},
    apply_local_bias::Bool,
    require_closed::Bool,
) where {T<:Real,A<:Number}
    axes(turning_angles) == axes(amplitudes) || throw(DimensionMismatch(
        "turning_angles and amplitudes must have identical axes",
    ))
    isempty(turning_angles) && throw(ArgumentError(
        "turning_angles and amplitudes must be nonempty",
    ))
    (isconcretetype(T) && isconcretetype(A)) || throw(ArgumentError(
        "turning_angles and amplitudes require concrete numeric element types",
    ))

    real_type = typeof(float(zero(T)))
    value_type = promote_type(real_type, A)
    turning_total = zero(real_type)
    turning_compensation = zero(real_type)
    value_total = zero(value_type)
    value_compensation = zero(value_type)
    pi_value = zero(real_type) + _typed_pi(zero(real_type))

    @inbounds for index in eachindex(turning_angles, amplitudes)
        turning = convert(real_type, turning_angles[index])
        turning_primal = _primal_value(turning)
        (isfinite(turning_primal) && turning_primal > zero(turning_primal) &&
         turning_primal < _primal_value(pi_value)) || throw(DomainError(
            turning_angles[index],
            "every supplied turning angle must be finite and lie in (0, pi)",
        ))
        amplitude = convert(value_type, amplitudes[index])
        _number_isfinite(amplitude) || throw(DomainError(
            amplitudes[index],
            "every supplied curvature amplitude must be finite",
        ))

        turning_increment = turning - turning_compensation
        next_turning = turning_total + turning_increment
        turning_compensation = (next_turning - turning_total) - turning_increment
        turning_total = next_turning

        term = turning * amplitude
        apply_local_bias && (term *= _curvature_local_bias(turning))
        value_increment = term - value_compensation
        next_value = value_total + value_increment
        value_compensation = (next_value - value_total) - value_increment
        value_total = next_value
    end

    if require_closed
        expected = 2pi_value
        base_type = typeof(float(_primal_value(turning_total)))
        tolerance = 128eps(base_type) * max(one(base_type), abs(_primal_value(expected)))
        abs(_primal_value(turning_total - expected)) <= tolerance || throw(DomainError(
            turning_total,
            "turning angles must close to a positive 2*pi exterior measure",
        ))
    end
    _number_isfinite(value_total) || throw(DomainError(
        value_total,
        "curvature-measure sum is non-finite",
    ))
    return value_total
end

"""
    curvature_measure_sum(turning_angles, amplitudes; require_closed=true)

Accumulate the normalized raw seam measure
`sum(beta(delta_j)*delta_j*amplitude_j)` for supplied positive exterior
turning angles. By default their compensated sum must close to `2*pi`; set
`require_closed=false` for a verified open arc. Inputs are never reordered or
copied.
"""
function curvature_measure_sum(
    turning_angles::AbstractVector{T},
    amplitudes::AbstractVector{A};
    require_closed::Bool=true,
) where {T<:Real,A<:Number}
    return _curvature_sum(turning_angles, amplitudes, true, require_closed)
end

"""
    debiased_curvature_measure_sum(turning_angles, amplitudes;
                                   require_closed=true)

Accumulate `sum(delta_j*amplitude_j)` after removing the exact local
near-coplanar wedge bias. Domain and closure rules match
[`curvature_measure_sum`](@ref).
"""
function debiased_curvature_measure_sum(
    turning_angles::AbstractVector{T},
    amplitudes::AbstractVector{A};
    require_closed::Bool=true,
) where {T<:Real,A<:Number}
    return _curvature_sum(turning_angles, amplitudes, false, require_closed)
end

"""
    curvature_continuum_harmonic(m, z, phi)

Evaluate the circular continuum reference
`integral_0^(2*pi) exp(im*m*theta-im*z*cos(theta-phi)) dtheta` as
`2*pi*(-im)^m*besselj(m,z)*exp(im*m*phi)`. `z` and `phi` must be finite.
"""
function curvature_continuum_harmonic(m::Integer, z::Real, phi::Real)
    z_value, phi_value = promote(float(z), float(phi))
    isfinite(_primal_value(z_value)) || throw(DomainError(z, "z must be finite"))
    isfinite(_primal_value(phi_value)) || throw(DomainError(phi, "phi must be finite"))
    pi_value = zero(z_value) + _typed_pi(z_value)
    imaginary_unit = complex(zero(z_value), one(z_value))
    bessel = try
        besselj(m, z_value)
    catch error
        error isa MethodError || rethrow()
        throw(ArgumentError(
            "curvature_continuum_harmonic does not support $(typeof(z_value)): " *
            "besselj is unavailable",
        ))
    end
    result = 2pi_value * (-imaginary_unit)^m * bessel *
             exp(imaginary_unit * m * phi_value)
    _number_isfinite(result) || throw(DomainError(
        (m, z, phi),
        "curvature continuum harmonic is non-finite",
    ))
    return result
end
