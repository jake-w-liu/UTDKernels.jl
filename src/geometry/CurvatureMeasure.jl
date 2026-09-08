"""
Curvature-measure limit of intrinsic near-coplanar PEC seam coefficients.

The production coefficient delegates to the Phase 162 reflection-boundary
intrinsic component. This file owns only the turning-angle normalization and
general supplied-measure accumulation.
"""

using SpecialFunctions: AmosException, besselj

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

@inline function _validate_curvature_partition_turning(turning_angle::Real)
    primal = _primal_value(turning_angle)
    pi_value = _typed_pi(turning_angle)
    (isfinite(primal) && primal > zero(primal) && primal < pi_value) ||
        throw(DomainError(
            turning_angle,
            "every supplied turning angle must be finite and lie in (0, pi)",
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
    input_turning = float(turning_angle)
    _validate_curvature_turning_angle(input_turning)
    input_is_pi = _primal_value(input_turning) == _typed_pi(input_turning)
    turning, wavenumber, distance = promote(input_turning, float(k), float(L))
    if input_is_pi
        # Preserve the caller's type-local inclusive endpoint after promotion.
        # Subtracting only the promoted primal retains any ForwardDiff tangent.
        turning = (turning - _primal_value(turning)) + _typed_pi(turning)
    end
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

@inline _curvature_stored_precision(::Union{Float16,Float32,Float64}) = 0
@inline _curvature_stored_precision(value::BigFloat) = precision(value)

@inline function _curvature_stored_precision(value::Real)
    primal = _primal_value(value)
    return primal isa BigFloat ? precision(primal) : 0
end

@inline function _curvature_stored_precision(value::Complex)
    return max(
        _curvature_stored_precision(real(value)),
        _curvature_stored_precision(imag(value)),
    )
end

@inline function _curvature_stored_precision(value::Number)
    return max(
        _curvature_stored_precision(real(value)),
        _curvature_stored_precision(imag(value)),
    )
end

@inline function _curvature_backend_order(m::Integer, ::Type{BigFloat})
    typemin(Clong) < m < typemax(Clong) || throw(ArgumentError(
        "curvature_continuum_harmonic requires typemin(Clong) < m < " *
        "typemax(Clong) for BigFloat arguments",
    ))
    return Clong(m)
end

@inline function _curvature_backend_order(m::Integer, ::Type)
    -typemax(Cint) < m < typemax(Cint) || throw(ArgumentError(
        "curvature_continuum_harmonic requires -typemax(Cint) < m < " *
        "typemax(Cint) for fixed-precision arguments",
    ))
    return Cint(m)
end

@inline function _curvature_quarter_turn(order::Integer, value::Real)
    zero_value = zero(value)
    one_value = one(value)
    residue = mod(order, 4)
    residue == 0 && return complex(one_value, zero_value)
    residue == 1 && return complex(zero_value, -one_value)
    residue == 2 && return complex(-one_value, zero_value)
    return complex(zero_value, one_value)
end

@inline function _curvature_harmonic_phase(
    order::Integer,
    phi::Real,
    stored_precision::Int,
    ::Val{W},
) where {W}
    primal = _primal_value(phi)
    base_type = typeof(float(primal))
    target_precision = W ? stored_precision : precision(base_type)
    order_magnitude = abs(Int128(order))
    order_bits = iszero(order_magnitude) ? 0 : ndigits(order_magnitude; base=2)
    work_precision = target_precision + order_bits + 32
    phase_high = setprecision(BigFloat, work_precision) do
        cis(BigFloat(order) * BigFloat(primal))
    end
    phase_primal = complex(
        base_type(real(phase_high)),
        base_type(imag(phase_high)),
    )
    imaginary_unit = complex(zero(phi), one(phi))
    tangent_phase = exp(imaginary_unit * order * (phi - primal))
    return _curvature_quarter_turn(order, phi) * phase_primal * tangent_phase
end

@inline function _curvature_widen_bigfloat(value::Real)
    primal = _primal_value(value)
    return (value - primal) + BigFloat(primal)
end

@inline function _curvature_widen_bigfloat(value::Complex)
    return complex(
        _curvature_widen_bigfloat(real(value)),
        _curvature_widen_bigfloat(imag(value)),
    )
end

@generated function _curvature_harmonic_route(::Type{Z}, ::Type{P}) where {Z,P}
    function primal_type(T)
        while T isa DataType && hasfield(T, :value)
            next_type = fieldtype(T, :value)
            next_type === T && break
            T = next_type
        end
        return T
    end

    z_primal_type = primal_type(Z)
    phi_primal_type = primal_type(P)
    route = if z_primal_type === BigFloat || phi_primal_type === BigFloat
        :bigfloat
    elseif z_primal_type === Float16 && phi_primal_type === Float16
        :float16
    else
        :fixed
    end
    return :(Val{$(QuoteNode(route))}())
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
    coefficient = _face_edge_incident(
        n, epsilon, delta, eta, q, wavenumber, distance, pi_value,
    )
    return coefficient::Complex{typeof(n)}
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

@inline function _curvature_sum_impl(
    turning_angles::AbstractVector{T},
    amplitudes::AbstractVector{A},
    apply_local_bias::Bool,
    require_closed::Bool,
    ::Type{R},
    ::Type{V},
    turning_precision::Int,
    ::Val{W},
) where {T<:Real,A<:Number,R<:Real,V<:Number,W}
    turning_total = zero(R)
    turning_compensation = zero(R)
    value_total = zero(V)
    value_compensation = zero(V)
    pi_value = zero(R) + _typed_pi(zero(R))
    stored_closure_expected = _typed_two_pi(first(turning_angles))
    closure_expected = convert(R, stored_closure_expected)

    @inbounds for index in eachindex(turning_angles, amplitudes)
        _validate_curvature_partition_turning(turning_angles[index])
        turning = convert(R, turning_angles[index])
        evaluation_turning = W ?
                             _curvature_widen_bigfloat(turning_angles[index]) :
                             turning
        turning_primal = _primal_value(turning)
        (isfinite(turning_primal) && turning_primal > zero(turning_primal) &&
         turning_primal < _primal_value(pi_value)) || throw(DomainError(
            turning_angles[index],
            "every supplied turning angle must be finite and lie in (0, pi)",
        ))
        amplitude_input = W ?
                          _curvature_widen_bigfloat(amplitudes[index]) :
                          amplitudes[index]
        amplitude = convert(V, amplitude_input)
        _number_isfinite(amplitude) || throw(DomainError(
            amplitudes[index],
            "every supplied curvature amplitude must be finite",
        ))

        turning_increment = turning - turning_compensation
        next_turning = turning_total + turning_increment
        turning_compensation = (next_turning - turning_total) - turning_increment
        turning_total = next_turning

        term = evaluation_turning * amplitude
        apply_local_bias && (term *= _curvature_local_bias(evaluation_turning))
        value_increment = term - value_compensation
        next_value = value_total + value_increment
        value_compensation = (next_value - value_total) - value_increment
        value_total = next_value
    end

    if require_closed
        # Closure belongs to the stored turning-angle type. In particular,
        # Float16 partitions close against their representable 2*pi endpoint
        # even though their arithmetic is widened to Float32.
        expected = closure_expected
        base_type = typeof(float(_primal_value(turning_total)))
        tolerance = if turning_precision > 0
            setprecision(BigFloat, turning_precision) do
                128eps(BigFloat) *
                max(one(BigFloat), abs(BigFloat(_primal_value(expected))))
            end
        else
            128eps(base_type) * max(one(base_type), abs(_primal_value(expected)))
        end
        stored_endpoint = float(_primal_value(stored_closure_expected))
        tolerance = max(tolerance, eps(stored_endpoint))
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

@inline function _curvature_input_precisions(
    turning_angles::AbstractVector,
    amplitudes::AbstractVector,
    prototype_precision::Int,
)
    prototype_precision > 0 || return 0, 0
    stored_precision = 0
    turning_precision = 0
    @inbounds for index in eachindex(turning_angles, amplitudes)
        local_turning_precision =
            _curvature_stored_precision(turning_angles[index])
        if local_turning_precision > 0
            if turning_precision == 0
                turning_precision = local_turning_precision
            elseif turning_precision != local_turning_precision
                throw(ArgumentError(
                    "BigFloat turning angles must have one stored precision",
                ))
            end
        end
        stored_precision = max(
            stored_precision,
            local_turning_precision,
            _curvature_stored_precision(amplitudes[index]),
        )
    end
    stored_precision == 0 && (stored_precision = prototype_precision)
    return stored_precision, turning_precision
end

@inline function _curvature_sum_wide(
    turning_angles::AbstractVector,
    amplitudes::AbstractVector,
    apply_local_bias::Bool,
    require_closed::Bool,
    real_type::Type{R},
    value_type::Type{V},
    stored_precision::Int,
    turning_precision::Int,
) where {R<:Real,V<:Number}
    return setprecision(BigFloat, stored_precision) do
        _curvature_sum_impl(
            turning_angles, amplitudes, apply_local_bias, require_closed,
            real_type, value_type, turning_precision, Val(true),
        )
    end
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

    input_real_type = typeof(float(first(turning_angles)))
    input_primal_type = typeof(float(_primal_value(first(turning_angles))))
    real_type = input_primal_type === Float16 ?
                promote_type(input_real_type, Float32) : input_real_type
    value_type = promote_type(real_type, A)
    prototype_precision = max(
        _curvature_stored_precision(zero(real_type)),
        _curvature_stored_precision(zero(value_type)),
    )
    stored_precision, turning_precision = _curvature_input_precisions(
        turning_angles, amplitudes, prototype_precision,
    )

    if stored_precision > 0
        return _curvature_sum_wide(
            turning_angles, amplitudes, apply_local_bias, require_closed,
            real_type, value_type, stored_precision, turning_precision,
        )
    end
    return _curvature_sum_impl(
        turning_angles, amplitudes, apply_local_bias, require_closed,
        real_type, value_type, turning_precision, Val(false),
    )
end

"""
    curvature_measure_sum(turning_angles, amplitudes; require_closed=true)

Accumulate the normalized raw seam measure
`sum(beta(delta_j)*delta_j*amplitude_j)` for supplied positive exterior
turning angles. By default their compensated sum must close to `2*pi`; set
`require_closed=false` for a verified open arc. Inputs are never reordered or
copied. All-`Float16` turning data are accumulated in `Float32`. Computations
involving `BigFloat` use the highest precision stored in their input values;
all `BigFloat` turning angles in one call must share a stored precision.
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

function _curvature_continuum_harmonic_impl(
    m::Integer,
    z::Real,
    phi::Real,
    stored_precision::Int,
    ::Val{W},
) where {W}
    z_value, phi_value = promote(float(z), float(phi))
    if W
        z_value = _curvature_widen_bigfloat(z_value)
        phi_value = _curvature_widen_bigfloat(phi_value)
    end
    isfinite(_primal_value(z_value)) || throw(DomainError(z, "z must be finite"))
    isfinite(_primal_value(phi_value)) || throw(DomainError(phi, "phi must be finite"))
    base_type = typeof(float(_primal_value(z_value)))
    order = _curvature_backend_order(m, base_type)
    pi_value = zero(z_value) + _typed_pi(z_value)
    bessel = (try
        besselj(order, z_value)
    catch error
        (error isa MethodError || error isa AmosException) || rethrow()
        throw(ArgumentError(
            "curvature_continuum_harmonic does not support order $(order) with " *
            "$(typeof(z_value)): the Bessel order/argument pair is outside " *
            "the supported numerical range",
        ))
    end)::typeof(z_value)
    phase = _curvature_harmonic_phase(order, phi_value, stored_precision, Val(W))
    result = 2pi_value * bessel * phase
    _number_isfinite(result) || throw(DomainError(
        (m, z, phi),
        "curvature continuum harmonic is non-finite",
    ))
    return result
end

@inline function _curvature_continuum_harmonic_routed(
    m::Integer,
    z::Real,
    phi::Real,
    ::Val{:float16},
)
    z_value, phi_value, _ = promote(float(z), float(phi), 0.0f0)
    return _curvature_continuum_harmonic_impl(
        m, z_value, phi_value, 0, Val(false),
    )
end

@inline function _curvature_continuum_harmonic_routed(
    m::Integer,
    z::Real,
    phi::Real,
    ::Val{:fixed},
)
    return _curvature_continuum_harmonic_impl(m, z, phi, 0, Val(false))
end

@inline function _curvature_continuum_harmonic_routed(
    m::Integer,
    z::Real,
    phi::Real,
    ::Val{:bigfloat},
)
    stored_precision = max(
        _curvature_stored_precision(z),
        _curvature_stored_precision(phi),
    )
    return setprecision(BigFloat, stored_precision) do
        _curvature_continuum_harmonic_impl(
            m, z, phi, stored_precision, Val(true),
        )
    end
end

"""
    curvature_continuum_harmonic(m, z, phi)

Evaluate the circular continuum reference
`integral_0^(2*pi) exp(im*m*theta-im*z*cos(theta-phi)) dtheta` as
`2*pi*(-im)^m*besselj(m,z)*exp(im*m*phi)`. `z` and `phi` must be finite. An
all-`Float16` call is evaluated in `Float32`; a call involving `BigFloat` uses
the highest precision stored in `z` or `phi`. To keep the Bessel recurrence and
its automatic derivative inside the backend integer range, fixed-precision
arguments require `-typemax(Cint) < m < typemax(Cint)`, and `BigFloat`
arguments require `typemin(Clong) < m < typemax(Clong)`.
"""
function curvature_continuum_harmonic(m::Integer, z::Real, phi::Real)
    route = _curvature_harmonic_route(typeof(z), typeof(phi))
    return _curvature_continuum_harmonic_routed(m, z, phi, route)
end
