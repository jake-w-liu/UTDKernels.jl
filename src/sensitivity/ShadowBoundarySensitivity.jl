"""
Canonical shadow-boundary switch, derivative kernel, Fourier multiplier, and
nonlinear-coordinate pullback coefficients for the `exp(+i*omega*t)` phasor
convention.
"""

using SpecialFunctions: AmosException, erfc, loggamma

const _SHADOW_MAX_MULTIPLIER_TERMS = 256
const _SHADOW_DIRECT_PHASE_LIMIT = 64
const _SHADOW_MAX_PHASE_WORK_BITS = 8192

@inline function _shadow_array_element_type(
    values::AbstractArray,
    context::AbstractString,
)
    element_type = eltype(values)
    isconcretetype(element_type) || throw(ArgumentError(
        "$context arrays require a concrete element type",
    ))
    return element_type
end

@inline function _shadow_binary_type(values::Real...)
    primal_type = promote_type(
        (typeof(float(_primal_value(value))) for value in values)...,
    )
    primal_type === BigFloat && throw(ArgumentError(
        "shadow-sensitivity production APIs do not expose a BigFloat path; " *
        "use an independent high-precision oracle",
    ))
    primal_type === Float16 && return Float32
    primal_type in (Float32, Float64) || throw(ArgumentError(
        "shadow-sensitivity production APIs support binary16, binary32, and " *
        "binary64 inputs",
    ))
    return primal_type
end

@inline function _shadow_complex_value(value::Number, name::AbstractString)
    _validate_finite_number(value, name)
    R = _shadow_binary_type(
        _primal_value(real(value)), _primal_value(imag(value)),
    )
    scalar_type = promote_type(
        R, typeof(float(real(value))), typeof(float(imag(value))),
    )
    coordinate = complex(
        convert(scalar_type, float(real(value))),
        convert(scalar_type, float(imag(value))),
    )
    _number_isfinite(coordinate) || throw(DomainError(
        value, "$name is outside the active numeric range",
    ))
    return coordinate, R
end

@inline function _shadow_real_values(first::Real, second::Real)
    R = _shadow_binary_type(first, second)
    scalar_type = promote_type(R, typeof(float(first)), typeof(float(second)))
    first_value = convert(scalar_type, float(first))
    second_value = convert(scalar_type, float(second))
    _number_isfinite(first_value) || throw(DomainError(first, "input must be finite"))
    _number_isfinite(second_value) || throw(DomainError(second, "input must be finite"))
    return first_value, second_value, R
end

@inline function _shadow_real_values(
    first::Real,
    second::Real,
    third::Real,
)
    R = _shadow_binary_type(first, second, third)
    scalar_type = promote_type(
        R, typeof(float(first)), typeof(float(second)), typeof(float(third)),
    )
    first_value = convert(scalar_type, float(first))
    second_value = convert(scalar_type, float(second))
    third_value = convert(scalar_type, float(third))
    _number_isfinite(first_value) || throw(DomainError(first, "input must be finite"))
    _number_isfinite(second_value) || throw(DomainError(second, "input must be finite"))
    _number_isfinite(third_value) || throw(DomainError(third, "input must be finite"))
    return first_value, second_value, third_value, R
end

@inline function _shadow_positive(value::Real, original, name::AbstractString)
    primal = _primal_value(value)
    (isfinite(primal) && primal > zero(primal)) || throw(DomainError(
        original, "$name must be finite and positive",
    ))
    return value
end

@inline function _shadow_quarter_phase(value::Real)
    quarter = one(value) / 4
    return complex(cospi(quarter), sinpi(quarter))
end

@inline _shadow_work_value(value, ::Type{Float32}) =
    _faddeeva_widen_float64(value)
@inline _shadow_work_value(value, ::Type{Float64}) = value

@inline function _shadow_phase_work_precision(
    logarithmic_phase::Real,
    ::Type{R};
    cancellation_bits::Int=0,
) where {R<:AbstractFloat}
    phase_bits = isfinite(logarithmic_phase) ?
                 max(0, ceil(Int, logarithmic_phase / log(2)) + 1) :
                 _SHADOW_MAX_PHASE_WORK_BITS + 1
    required = precision(R) + phase_bits + cancellation_bits + 64
    required <= _SHADOW_MAX_PHASE_WORK_BITS || throw(DomainError(
        logarithmic_phase,
        "shadow phase requires $required bits, above the bounded " *
        "$_SHADOW_MAX_PHASE_WORK_BITS-bit workspace",
    ))
    return max(192, required)
end

@inline _shadow_ad_depth(value::Real) = hasproperty(value, :value) ?
    1 + _shadow_ad_depth(getproperty(value, :value)) : 0
@inline _shadow_ad_depth(value::Complex) = max(
    _shadow_ad_depth(real(value)), _shadow_ad_depth(imag(value)),
)

@inline function _shadow_coordinate_work_precision(
    coordinate::Number,
    ::Type{R},
) where {R<:AbstractFloat}
    magnitude = max(
        abs(_primal_value(real(coordinate))),
        abs(_primal_value(imag(coordinate))),
    )
    iszero(magnitude) && return 192
    return _shadow_phase_work_precision(2log(magnitude), R)
end

function _shadow_stable_wide_value(
    builder,
    ::Type{C},
    base_precision::Int,
    context::AbstractString,
) where {C<:Number}
    work_precisions = (
        base_precision,
        min(2base_precision, _SHADOW_MAX_PHASE_WORK_BITS),
        _SHADOW_MAX_PHASE_WORK_BITS,
    )
    previous = nothing
    last_result = nothing
    @inbounds for work_precision in unique(work_precisions)
        wide_result = setprecision(builder, BigFloat, work_precision)
        converted::C = convert(C, wide_result)
        last_result = converted
        if _number_isfinite(converted)
            previous !== nothing && isequal(converted, previous) &&
                return converted
            previous = converted
        end
    end
    last_result !== nothing && !_number_isfinite(last_result) &&
        throw(DomainError(
            last_result, "$context exceeds the active numeric range",
        ))
    throw(ArgumentError(
        "$context did not stabilize within the bounded " *
        "$_SHADOW_MAX_PHASE_WORK_BITS-bit workspace",
    ))
end

function _shadow_canonical_kernel_wide(
    coordinate::C,
    ::Type{R},
) where {C<:Number,R<:AbstractFloat}
    base_precision = _shadow_coordinate_work_precision(coordinate, R)
    return _shadow_stable_wide_value(
        C, base_precision, "shadow sensitivity kernel",
    ) do
        wide_coordinate = _faddeeva_widen_bigfloat(coordinate)
        cispi(BigFloat(0.25)) *
        exp(-im * wide_coordinate * wide_coordinate) / sqrt(BigFloat(pi))
    end
end

function _shadow_scaled_kernel_wide(
    coordinate::T,
    parameter::T,
    ::Type{C},
    ::Type{R},
) where {T<:Real,C<:Number,R<:AbstractFloat}
    coordinate_primal = abs(_primal_value(coordinate))
    parameter_primal = _primal_value(parameter)
    base_precision = if iszero(coordinate_primal)
        192
    else
        _shadow_phase_work_precision(
            log(parameter_primal) + 2log(coordinate_primal), R,
        )
    end
    return _shadow_stable_wide_value(
        C, base_precision, "scaled shadow sensitivity kernel",
    ) do
        wide_coordinate = _faddeeva_widen_bigfloat(coordinate)
        wide_parameter = _faddeeva_widen_bigfloat(parameter)
        sqrt(wide_parameter) * cispi(BigFloat(0.25)) *
        exp(-im * wide_parameter * wide_coordinate * wide_coordinate) /
        sqrt(BigFloat(pi))
    end
end

function _shadow_quadratic_exponential(
    value::Number,
    sign::Int,
    ::Type{R},
) where {R<:AbstractFloat}
    sign in (-1, 1) || throw(ArgumentError("quadratic phase sign must be +/-1"))
    primal = complex(
        _primal_value(real(value)), _primal_value(imag(value)),
    )
    magnitude = max(abs(real(primal)), abs(imag(primal)))
    if magnitude <= sqrt(R(_SHADOW_DIRECT_PHASE_LIMIT))
        direct = exp(sign * im * value * value)
        _number_isfinite(direct) || throw(DomainError(
            direct, "shadow quadratic exponential is non-finite",
        ))
        return direct
    end

    logarithmic_phase = 2log(max(magnitude, nextfloat(zero(R))))
    work_precision = _shadow_phase_work_precision(logarithmic_phase, R)
    result = setprecision(BigFloat, work_precision) do
        wide_value = _faddeeva_widen_bigfloat(value)
        convert(typeof(value), exp(sign * im * wide_value * wide_value))
    end
    _number_isfinite(result) || throw(DomainError(
        result, "shadow quadratic exponential exceeds the active range",
    ))
    return result
end

function _shadow_asymptotic_switch(
    coordinate::C,
    ::Type{R},
) where {C<:Number,R<:AbstractFloat}
    primal_real = _primal_value(real(coordinate))
    primal_imag = _primal_value(imag(coordinate))
    magnitude = max(abs(primal_real), abs(primal_imag))
    logarithmic_phase = 2log(max(magnitude, nextfloat(zero(R))))
    base_precision = _shadow_phase_work_precision(logarithmic_phase, R)
    work_precisions = (
        base_precision,
        min(2base_precision, _SHADOW_MAX_PHASE_WORK_BITS),
        _SHADOW_MAX_PHASE_WORK_BITS,
    )
    positive = primal_real > zero(R)
    previous = nothing
    last_result = nothing
    @inbounds for work_precision in unique(work_precisions)
        wide_result = setprecision(BigFloat, work_precision) do
            wide_coordinate = _faddeeva_widen_bigfloat(coordinate)
            positive_coordinate = positive ? wide_coordinate : -wide_coordinate
            inverse_coordinate = inv(positive_coordinate)
            inverse_square = inverse_coordinate * inverse_coordinate
            total = one(positive_coordinate)
            compensation = zero(total)
            term = total
            previous_magnitude = abs(complex(
                BigFloat(_primal_value(real(term))),
                BigFloat(_primal_value(imag(term))),
            ))
            for order in 1:64
                term *= im * BigFloat(2order - 1) * inverse_square / 2
                magnitude = abs(complex(
                    BigFloat(_primal_value(real(term))),
                    BigFloat(_primal_value(imag(term))),
                ))
                magnitude >= previous_magnitude && break
                total, compensation = _multipole_compensated_add(
                    total, compensation, term,
                )
                magnitude <= BigFloat(8) * eps(BigFloat) * max(
                    one(BigFloat),
                    abs(complex(
                        BigFloat(_primal_value(real(total))),
                        BigFloat(_primal_value(imag(total))),
                    )),
                ) && break
                previous_magnitude = magnitude
            end
            quarter_phase = cispi(BigFloat(0.25))
            tail = exp(-im * positive_coordinate * positive_coordinate) *
                   conj(quarter_phase) * inverse_coordinate * total /
                   (2sqrt(BigFloat(pi)))
            positive ? one(tail) - tail : tail
        end
        converted::C = convert(C, wide_result)
        last_result = converted
        if _number_isfinite(converted)
            previous !== nothing && isequal(converted, previous) &&
                return converted
            previous = converted
        end
    end
    last_result !== nothing && !_number_isfinite(last_result) &&
        throw(DomainError(
            last_result, "shadow switch exceeds the active numeric range",
        ))
    throw(ArgumentError(
        "shadow-switch asymptotic tail did not stabilize within the bounded " *
        "$_SHADOW_MAX_PHASE_WORK_BITS-bit workspace",
    ))
end

function _shadow_switch_ad_lift(
    coordinate::C,
    ::Type{R},
) where {C<:Number,R<:AbstractFloat}
    depth = _shadow_ad_depth(coordinate)
    depth > 0 || throw(ArgumentError(
        "shadow-switch AD lifting requires a differentiated coordinate",
    ))
    primal_coordinate = complex(
        convert(R, _primal_value(real(coordinate))),
        convert(R, _primal_value(imag(coordinate))),
    )
    primal_switch = shadow_switch(primal_coordinate)
    base_precision = _shadow_coordinate_work_precision(coordinate, R)
    return _shadow_stable_wide_value(
        C, base_precision, "shadow-switch AD recovery",
    ) do
        wide_coordinate = _faddeeva_widen_bigfloat(coordinate)
        wide_primal = complex(
            BigFloat(real(primal_coordinate)),
            BigFloat(imag(primal_coordinate)),
        )
        increment = wide_coordinate - wide_primal
        result = complex(
            BigFloat(real(primal_switch)), BigFloat(imag(primal_switch)),
        )
        derivative = cispi(BigFloat(0.25)) *
                     exp(-im * wide_primal * wide_primal) /
                     sqrt(BigFloat(pi))
        previous_derivative = zero(derivative)
        increment_power = one(increment)
        factorial_value = one(BigFloat)
        for order in 1:depth
            if order == 2
                next_derivative = -2im * wide_primal * derivative
                previous_derivative, derivative = derivative, next_derivative
            elseif order > 2
                next_derivative = -2im * wide_primal * derivative -
                                  2im * BigFloat(order - 2) *
                                  previous_derivative
                previous_derivative, derivative = derivative, next_derivative
            end
            increment_power *= increment
            factorial_value *= BigFloat(order)
            result += derivative * increment_power / factorial_value
        end
        result
    end
end

function _shadow_scaled_kernel_exponential(
    coordinate::Real,
    parameter::Real,
    ::Type{R},
) where {R<:AbstractFloat}
    coordinate_primal = abs(_primal_value(coordinate))
    parameter_primal = _primal_value(parameter)
    logarithmic_phase = iszero(coordinate_primal) ? -R(Inf) :
        log(parameter_primal) + 2log(coordinate_primal)
    if logarithmic_phase <= log(R(_SHADOW_DIRECT_PHASE_LIMIT))
        direct = exp(-im * parameter * coordinate * coordinate)
        _number_isfinite(direct) || throw(DomainError(
            direct, "scaled shadow quadratic exponential is non-finite",
        ))
        return direct
    end
    work_precision = _shadow_phase_work_precision(logarithmic_phase, R)
    result_type = typeof(complex(zero(coordinate)))
    result = setprecision(BigFloat, work_precision) do
        wide_coordinate = _faddeeva_widen_bigfloat(coordinate)
        wide_parameter = _faddeeva_widen_bigfloat(parameter)
        convert(
            result_type,
            exp(-im * wide_parameter * wide_coordinate * wide_coordinate),
        )
    end
    _number_isfinite(result) || throw(DomainError(
        result, "scaled shadow quadratic exponential exceeds the active range",
    ))
    return result
end

@inline function _shadow_quadratic_phase(value::Number, name::AbstractString)
    primal = complex(
        _primal_value(real(value)), _primal_value(imag(value)),
    )
    phase = primal * primal
    _number_isfinite(phase) || throw(DomainError(
        phase, "$name quadratic phase is outside the active numeric range",
    ))
    return phase
end

"""
    shadow_switch(q)

Evaluate the canonical Fresnel switch
`erfc(-exp(im*pi/4)*q)/2` for a finite binary real or complex signed
coordinate. The convention is `exp(+i*omega*t)`, and `shadow_switch(0)` is
exactly one half. Large coordinates in the near-real sector use a
phase-accurate asymptotic tail. Array inputs are evaluated elementwise.
"""
function shadow_switch(q::Number)
    coordinate, R = _shadow_complex_value(q, "shadow coordinate q")
    _number_contains_ad(coordinate) &&
        return _shadow_switch_ad_lift(coordinate, R)
    work_coordinate = _shadow_work_value(coordinate, R)
    primal_real = _primal_value(real(work_coordinate))
    primal_imag = _primal_value(imag(work_coordinate))
    real_magnitude = abs(primal_real)
    near_real_sector = abs(primal_imag) <= real_magnitude / R(8)
    if real_magnitude > R(8) && near_real_sector
        value = _shadow_asymptotic_switch(coordinate, R)
        _number_isfinite(value) || throw(DomainError(
            value, "shadow switch is non-finite",
        ))
        return value
    end

    _shadow_quadratic_phase(work_coordinate, "shadow switch")
    argument = -_shadow_quarter_phase(real(work_coordinate)) * work_coordinate
    value = try
        convert(typeof(coordinate), erfc(argument) / 2)
    catch error
        error isa AmosException || rethrow()
        throw(DomainError(q, "shadow switch exceeds the backend range"))
    end
    _number_isfinite(value) || throw(DomainError(
        value, "shadow switch is non-finite",
    ))
    return value
end

function shadow_switch(q::AbstractArray{<:Number})
    element_type = _shadow_array_element_type(q, "shadow-coordinate")
    _shadow_complex_value(zero(element_type), "shadow coordinate q")
    return map(shadow_switch, q)
end

"""
    shadow_sensitivity_kernel(q)
    shadow_sensitivity_kernel(s, kappa)

With one argument, evaluate the exact canonical derivative
`exp(im*pi/4)*exp(-im*q^2)/sqrt(pi)`. With finite real `s` and positive
`kappa`, evaluate the scaled physical-coordinate kernel
`sqrt(kappa)*K(sqrt(kappa)*s)`. Binary16 inputs widen to binary32; binary32,
binary64, and supported ForwardDiff arithmetic preserve their promoted type.
Large quadratic phases are reduced from the original stored inputs in a
bounded exponent-aware workspace.
"""
function shadow_sensitivity_kernel(q::Number)
    coordinate, R = _shadow_complex_value(q, "shadow coordinate q")
    _number_contains_ad(coordinate) &&
        return _shadow_canonical_kernel_wide(coordinate, R)
    work_coordinate = _shadow_work_value(coordinate, R)
    scalar = real(work_coordinate)
    pi_value = zero(scalar) + _typed_pi(scalar)
    value = convert(
        typeof(coordinate),
        _shadow_quarter_phase(scalar) *
        _shadow_quadratic_exponential(work_coordinate, -1, R) /
        sqrt(pi_value),
    )
    _number_isfinite(value) || throw(DomainError(
        value, "shadow sensitivity kernel is non-finite",
    ))
    return value
end


function shadow_sensitivity_kernel(q::AbstractArray{<:Number})
    element_type = _shadow_array_element_type(q, "shadow-coordinate")
    _shadow_complex_value(zero(element_type), "shadow coordinate q")
    return map(shadow_sensitivity_kernel, q)
end

function shadow_sensitivity_kernel(s::Real, kappa::Real)
    coordinate, parameter, R = _shadow_real_values(s, kappa)
    _shadow_positive(parameter, kappa, "kappa")
    work_coordinate = _shadow_work_value(coordinate, R)
    work_parameter = _shadow_work_value(parameter, R)
    root = sqrt(work_parameter)
    output_type = typeof(complex(zero(coordinate)))
    coordinate_magnitude = abs(_primal_value(work_coordinate))
    logarithmic_phase = iszero(coordinate_magnitude) ? -R(Inf) :
        log(_primal_value(work_parameter)) + 2log(coordinate_magnitude)
    output_contains_ad = _number_contains_ad(work_coordinate) ||
                         _number_contains_ad(work_parameter)
    value = if output_contains_ad
        _shadow_scaled_kernel_wide(
            work_coordinate, work_parameter, output_type, R,
        )
    elseif logarithmic_phase <= log(R(_SHADOW_DIRECT_PHASE_LIMIT))
        canonical_coordinate = root * work_coordinate
        _number_isfinite(canonical_coordinate) || throw(DomainError(
            canonical_coordinate,
            "scaled shadow coordinate is outside the active numeric range",
        ))
        convert(
            output_type, root * shadow_sensitivity_kernel(canonical_coordinate),
        )
    else
        scalar = real(work_coordinate)
        pi_value = zero(scalar) + _typed_pi(scalar)
        convert(
            output_type,
            root * _shadow_quarter_phase(scalar) *
            _shadow_scaled_kernel_exponential(
                work_coordinate, work_parameter, R,
            ) / sqrt(pi_value),
        )
    end
    _number_isfinite(value) || throw(DomainError(
        value, "scaled shadow sensitivity kernel is non-finite",
    ))
    return value
end

function shadow_sensitivity_kernel(s::AbstractArray{<:Real}, kappa::Real)
    element_type = _shadow_array_element_type(s, "scaled shadow-coordinate")
    _, parameter, _ = _shadow_real_values(zero(element_type), kappa)
    _shadow_positive(parameter, kappa, "kappa")
    return map(value -> shadow_sensitivity_kernel(value, kappa), s)
end

@inline function _shadow_multiplier_phase(xi::Real, kappa::Real)
    frequency, parameter, R = _shadow_real_values(xi, kappa)
    _shadow_positive(parameter, kappa, "kappa")
    work_frequency = _shadow_work_value(frequency, R)
    work_parameter = _shadow_work_value(parameter, R)
    scaled_frequency = (work_frequency / sqrt(work_parameter)) / 2
    _number_isfinite(scaled_frequency) || throw(DomainError(
        scaled_frequency,
        "scaled Fourier frequency is outside the active numeric range",
    ))
    phase = scaled_frequency * scaled_frequency
    _number_isfinite(phase) || throw(DomainError(
        phase, "quadratic Fourier phase is outside the active numeric range",
    ))
    return phase, typeof(complex(zero(frequency)))
end

function _shadow_exact_multiplier(xi::Real, kappa::Real)
    frequency, parameter, R = _shadow_real_values(xi, kappa)
    _shadow_positive(parameter, kappa, "kappa")
    output_type = typeof(complex(zero(frequency)))
    if _number_contains_ad(frequency) || _number_contains_ad(parameter)
        frequency_primal = abs(_primal_value(frequency))
        parameter_primal = _primal_value(parameter)
        base_precision = if iszero(frequency_primal)
            192
        else
            logarithmic_phase = 2log(frequency_primal) -
                                log(parameter_primal) - log(R(4))
            _shadow_phase_work_precision(logarithmic_phase, R)
        end
        return _shadow_stable_wide_value(
            output_type, base_precision, "shadow-sensitivity multiplier",
        ) do
            wide_frequency = _faddeeva_widen_bigfloat(frequency)
            wide_parameter = _faddeeva_widen_bigfloat(parameter)
            exp(im * wide_frequency^2 / (4wide_parameter))
        end
    end
    work_frequency = _shadow_work_value(frequency, R)
    work_parameter = _shadow_work_value(parameter, R)
    scaled_frequency = (work_frequency / sqrt(work_parameter)) / 2
    phase = scaled_frequency * scaled_frequency
    if _number_isfinite(phase) &&
       abs(_primal_value(phase)) <= R(_SHADOW_DIRECT_PHASE_LIMIT)
        value = convert(output_type, exp(im * phase))
        _number_isfinite(value) || throw(DomainError(
            value, "shadow-sensitivity multiplier is non-finite",
        ))
        return value
    end

    frequency_primal = abs(_primal_value(frequency))
    parameter_primal = _primal_value(parameter)
    logarithmic_phase = 2log(frequency_primal) - log(parameter_primal) - log(R(4))
    work_precision = _shadow_phase_work_precision(logarithmic_phase, R)
    value = setprecision(BigFloat, work_precision) do
        wide_frequency = _faddeeva_widen_bigfloat(frequency)
        wide_parameter = _faddeeva_widen_bigfloat(parameter)
        convert(
            output_type,
            exp(im * wide_frequency^2 / (4wide_parameter)),
        )
    end
    _number_isfinite(value) || throw(DomainError(
        value, "shadow-sensitivity multiplier is non-finite",
    ))
    return value
end

@inline function _shadow_truncated_direct(
    phase::Real,
    retained_terms::Int,
    ::Type{C},
)::C where {C<:Complex}
    z = im * phase
    total = one(complex(phase))
    compensation = zero(total)
    term = total
    @inbounds for order in 1:(retained_terms - 1)
        term *= z / order
        _number_isfinite(term) || throw(DomainError(
            term, "truncated shadow multiplier exceeds the active range",
        ))
        total, compensation = _multipole_compensated_add(
            total, compensation, term,
        )
    end
    value = convert(C, total)
    _number_isfinite(value) || throw(DomainError(
        value, "truncated shadow multiplier exceeds the active range",
    ))
    return value
end

function _shadow_truncated_wide(
    frequency::Real,
    parameter::Real,
    retained_terms::Int,
    ::Type{C},
    work_precision::Int,
)::C where {C<:Complex}
    return _shadow_stable_wide_value(
        C, work_precision, "truncated shadow multiplier",
    ) do
        wide_frequency = _faddeeva_widen_bigfloat(frequency)
        wide_parameter = _faddeeva_widen_bigfloat(parameter)
        wide_phase = wide_frequency^2 / (4wide_parameter)
        z = im * wide_phase
        total = one(complex(wide_phase))
        compensation = zero(total)
        term = total
        @inbounds for order in 1:(retained_terms - 1)
            term *= z / order
            total, compensation = _multipole_compensated_add(
                total, compensation, term,
            )
        end
        total
    end
end

function _shadow_truncated_multiplier(
    xi::Real,
    kappa::Real,
    retained_terms::Int,
)
    frequency, parameter, R = _shadow_real_values(xi, kappa)
    _shadow_positive(parameter, kappa, "kappa")
    output_type = typeof(complex(zero(frequency)))
    retained_terms == 1 && return one(output_type)
    work_frequency = _shadow_work_value(frequency, R)
    work_parameter = _shadow_work_value(parameter, R)
    frequency_primal = abs(_primal_value(frequency))
    parameter_primal = _primal_value(parameter)
    logarithmic_phase = iszero(frequency_primal) ? -R(Inf) :
        2log(frequency_primal) - log(parameter_primal) - log(R(4))
    phase_magnitude_primal = iszero(frequency_primal) ? zero(R) :
        (logarithmic_phase <= log(R(4096)) ?
         exp(logarithmic_phase) : R(Inf))
    dominant_polynomial_primal = phase_magnitude_primal >=
                                 R(2max(1, retained_terms - 1))
    cancellation_bits = isfinite(phase_magnitude_primal) ?
        ceil(Int, phase_magnitude_primal / log(2)) + 32 :
        _SHADOW_MAX_PHASE_WORK_BITS + 1
    if _number_contains_ad(frequency) || _number_contains_ad(parameter)
        work_precision = if iszero(frequency_primal) ||
                            dominant_polynomial_primal
            max(192, precision(R) + 64)
        else
            _shadow_phase_work_precision(
                logarithmic_phase, R; cancellation_bits,
            )
        end
        return _shadow_truncated_wide(
            frequency, parameter, retained_terms, output_type, work_precision,
        )
    end
    scaled_frequency = (work_frequency / sqrt(work_parameter)) / 2
    phase = scaled_frequency * scaled_frequency
    phase_magnitude = abs(_primal_value(phase))
    dominant_polynomial = phase_magnitude >=
                          R(2max(1, retained_terms - 1))
    if _number_isfinite(phase) && phase_magnitude <= R(8)
        return _shadow_truncated_direct(phase, retained_terms, output_type)
    end
    if _number_isfinite(phase) && dominant_polynomial
        return _shadow_truncated_wide(
            frequency, parameter, retained_terms, output_type,
            max(192, precision(R) + 64),
        )
    end

    work_precision = _shadow_phase_work_precision(
        logarithmic_phase, R; cancellation_bits,
    )
    return _shadow_truncated_wide(
        frequency, parameter, retained_terms, output_type, work_precision,
    )
end

@inline function _shadow_multiplier_terms(terms::Integer)
    value = try
        Int(terms)
    catch error
        (error isa InexactError || error isa OverflowError) || rethrow()
        throw(DomainError(terms, "multiplier term count is outside Int range"))
    end
    1 <= value <= _SHADOW_MAX_MULTIPLIER_TERMS || throw(DomainError(
        terms,
        "multiplier term count must lie in 1:$_SHADOW_MAX_MULTIPLIER_TERMS",
    ))
    return value
end

"""
    shadow_sensitivity_multiplier(xi, kappa; terms=nothing)

Evaluate the exact Fourier multiplier `exp(im*xi^2/(4*kappa))` for finite real
`xi` and positive finite `kappa`. Set `terms` to an integer in `1:256` to
evaluate that many terms of its exponential series, beginning with the Dirac
term. Cancellation-prone finite sums use a bounded wider workspace. Array
inputs are evaluated elementwise.
"""
function shadow_sensitivity_multiplier(
    xi::Real,
    kappa::Real;
    terms::Union{Nothing,Integer}=nothing,
)
    if terms === nothing
        return _shadow_exact_multiplier(xi, kappa)
    end
    retained_terms = _shadow_multiplier_terms(terms)
    return _shadow_truncated_multiplier(xi, kappa, retained_terms)
end


shadow_sensitivity_multiplier(xi::Real, kappa::Real, terms::Integer) =
    shadow_sensitivity_multiplier(xi, kappa; terms)

function shadow_sensitivity_multiplier(
    xi::AbstractArray{<:Real},
    kappa::Real;
    terms::Union{Nothing,Integer}=nothing,
)
    element_type = _shadow_array_element_type(xi, "shadow-frequency")
    _, parameter, _ = _shadow_real_values(zero(element_type), kappa)
    _shadow_positive(parameter, kappa, "kappa")
    terms === nothing || _shadow_multiplier_terms(terms)
    return map(
        value -> shadow_sensitivity_multiplier(value, kappa; terms), xi,
    )
end

function _shadow_numeric_type(values::Number...)
    primal_type = promote_type(
        (typeof(float(_primal_value(component)))
         for value in values for component in (real(value), imag(value)))...,
    )
    primal_type === BigFloat && throw(ArgumentError(
        "shadow-sensitivity pullbacks do not expose a BigFloat production path",
    ))
    primal_type === Float16 && (primal_type = Float32)
    primal_type in (Float32, Float64) || throw(ArgumentError(
        "shadow-sensitivity pullbacks support binary16, binary32, and " *
        "binary64 inputs",
    ))
    scalar_type = promote_type(
        primal_type,
        (typeof(float(component))
         for value in values for component in (real(value), imag(value)))...,
    )
    return scalar_type, primal_type
end

function _shadow_pullback_wide_components(
    values,
    derivative,
    second_map,
    third_map,
    fourth_map,
    parameter,
)
    wide_values = map(_faddeeva_widen_bigfloat, values)
    wide_derivative = _faddeeva_widen_bigfloat(derivative)
    wide_second = _faddeeva_widen_bigfloat(second_map) / wide_derivative
    wide_third = _faddeeva_widen_bigfloat(third_map) / wide_derivative
    wide_fourth = _faddeeva_widen_bigfloat(fourth_map) / wide_derivative
    wide_parameter = _faddeeva_widen_bigfloat(parameter)
    wide_inverse_scale2 = inv(
        wide_parameter * wide_derivative * wide_derivative,
    )
    wide_first_bracket = wide_values[3] - wide_second * wide_values[2]
    wide_second_bracket = wide_values[5] -
        6wide_second * wide_values[4] +
        (15wide_second^2 - 4wide_third) * wide_values[3] +
        (-15wide_second^3 + 10wide_second * wide_third - wide_fourth) *
        wide_values[2]
    wide_first_correction = wide_inverse_scale2 * wide_first_bracket / 4
    wide_second_correction = wide_inverse_scale2^2 * wide_second_bracket / 32
    wide_first = wide_values[1] - im * wide_first_correction
    wide_second_value = wide_first - wide_second_correction
    return (
        leading=wide_values[1],
        first=wide_first,
        second=wide_second_value,
    )
end

function _shadow_stable_pullback_result(
    values,
    derivative,
    second_map,
    third_map,
    fourth_map,
    parameter,
    ::Type{C},
) where {C<:Number}
    previous = nothing
    last_result = nothing
    @inbounds for work_precision in
        (256, 512, 1024, 2048, 4096, _SHADOW_MAX_PHASE_WORK_BITS)
        wide_result = setprecision(BigFloat, work_precision) do
            _shadow_pullback_wide_components(
                values, derivative, second_map, third_map, fourth_map,
                parameter,
            )
        end
        converted_leading::C = convert(C, wide_result.leading)
        converted_first::C = convert(C, wide_result.first)
        converted_second::C = convert(C, wide_result.second)
        converted = (
            leading=converted_leading,
            first=converted_first,
            second=converted_second,
        )
        last_result = converted
        if all(_number_isfinite, converted)
            previous !== nothing && isequal(converted, previous) &&
                return converted
            previous = converted
        end
    end
    last_result !== nothing && !all(_number_isfinite, last_result) &&
        throw(DomainError(
            last_result,
            "shadow-sensitivity pullback exceeds the active range",
        ))
    throw(ArgumentError(
        "shadow-sensitivity pullback did not stabilize within the bounded " *
        "$_SHADOW_MAX_PHASE_WORK_BITS-bit workspace",
    ))
end

function _shadow_pullback_result(
    values,
    derivative,
    second_map,
    third_map,
    fourth_map,
    parameter,
    ::Type{C},
) where {C<:Number}
    normalized_second = second_map / derivative
    normalized_third = third_map / derivative
    normalized_fourth = fourth_map / derivative
    scaled_derivative = derivative * sqrt(parameter)
    inverse_scaled = inv(scaled_derivative)
    inverse_scale2 = inverse_scaled * inverse_scaled
    first_term1 = values[3]
    first_term2 = normalized_second * values[2]
    first_bracket = first_term1 - first_term2
    second_term1 = values[5]
    second_term2 = -6normalized_second * values[4]
    second_term3 = (15normalized_second^2 - 4normalized_third) * values[3]
    second_term4 = (-15normalized_second^3 +
                    10normalized_second * normalized_third -
                    normalized_fourth) * values[2]
    second_bracket = second_term1 + second_term2 + second_term3 + second_term4
    first_correction = inverse_scale2 * first_bracket / 4
    second_correction = inverse_scale2^2 * second_bracket / 32
    leading = values[1]
    first = leading - im * first_correction
    second = first - second_correction
    direct_values = (
        normalized_second, normalized_third, normalized_fourth,
        scaled_derivative, inverse_scale2, first_correction,
        second_correction, first, second,
    )
    suspicious_zero = iszero(_primal_value(inverse_scale2)) &&
                      any(index -> !iszero(values[index]), 2:5)
    R = typeof(float(_primal_value(real(zero(C)))))
    first_bound = convert(R, float(_primal_value(abs(first_term1)))) +
                  convert(R, float(_primal_value(abs(first_term2))))
    second_bound = convert(R, float(_primal_value(abs(second_term1)))) +
                   convert(R, float(_primal_value(abs(second_term2)))) +
                   convert(R, float(_primal_value(abs(second_term3)))) +
                   convert(R, float(_primal_value(abs(second_term4))))
    first_magnitude = convert(R, float(_primal_value(abs(first_bracket))))
    second_magnitude = convert(R, float(_primal_value(abs(second_bracket))))
    bracket_cancellation =
        (!iszero(first_bound) &&
         (!isfinite(first_bound) ||
          first_magnitude <= sqrt(eps(R)) * first_bound)) ||
        (!iszero(second_bound) &&
         (!isfinite(second_bound) ||
          second_magnitude <= sqrt(eps(R)) * second_bound))
    outer_first_bound =
        convert(R, float(_primal_value(abs(leading)))) +
        convert(R, float(_primal_value(abs(first_correction))))
    outer_second_bound =
        convert(R, float(_primal_value(abs(first)))) +
        convert(R, float(_primal_value(abs(second_correction))))
    outer_first_magnitude = convert(
        R, float(_primal_value(abs(first))),
    )
    outer_second_magnitude = convert(
        R, float(_primal_value(abs(second))),
    )
    outer_cancellation =
        (!iszero(outer_first_bound) &&
         (!isfinite(outer_first_bound) ||
          outer_first_magnitude <= sqrt(eps(R)) * outer_first_bound)) ||
        (!iszero(outer_second_bound) &&
         (!isfinite(outer_second_bound) ||
          outer_second_magnitude <= sqrt(eps(R)) * outer_second_bound))
    suspicious_correction =
        (iszero(_primal_value(abs(first_correction))) &&
         !iszero(_primal_value(abs(first_bracket)))) ||
        (iszero(_primal_value(abs(second_correction))) &&
         !iszero(_primal_value(abs(second_bracket))))
    contains_ad = any(_number_contains_ad, values) ||
                  any(
                      _number_contains_ad,
                      (derivative, second_map, third_map, fourth_map, parameter),
                  )
    if all(_number_isfinite, direct_values) && !suspicious_zero &&
       !suspicious_correction && !bracket_cancellation &&
       !outer_cancellation && !contains_ad
        return (leading=leading, first=first, second=second)
    end

    return _shadow_stable_pullback_result(
        values, derivative, second_map, third_map, fourth_map, parameter, C,
    )
end

"""
    shadow_sensitivity_pullback(
        f0, f1, f2, f3, f4, a, b, c, d, kappa,
    )

Return `(leading, first, second)` for the nonlinear-coordinate distributional
pullback through order `kappa^-2`. `f0` through `f4` are local probe
derivatives, while `a=g'(s0)>0`, `b=g''(s0)`, `c=g'''(s0)`, and `d=g''''(s0)`
describe the real coordinate map. Every input must be finite and `kappa` must
be positive.
"""
function shadow_sensitivity_pullback(
    f0::Number,
    f1::Number,
    f2::Number,
    f3::Number,
    f4::Number,
    a::Real,
    b::Real,
    c::Real,
    d::Real,
    kappa::Real,
)
    scalar_type, _ = _shadow_numeric_type(
        f0, f1, f2, f3, f4, a, b, c, d, kappa,
    )
    C = typeof(complex(zero(scalar_type)))
    values = (
        convert(C, complex(float(f0))),
        convert(C, complex(float(f1))),
        convert(C, complex(float(f2))),
        convert(C, complex(float(f3))),
        convert(C, complex(float(f4))),
    )
    all(_number_isfinite, values) || throw(DomainError(
        values, "probe derivatives must be finite",
    ))
    derivative = convert(scalar_type, float(a))
    second_map = convert(scalar_type, float(b))
    third_map = convert(scalar_type, float(c))
    fourth_map = convert(scalar_type, float(d))
    parameter = convert(scalar_type, float(kappa))
    _shadow_positive(derivative, a, "coordinate derivative a")
    _shadow_positive(parameter, kappa, "kappa")
    all(_number_isfinite, (second_map, third_map, fourth_map)) ||
        throw(DomainError((b, c, d), "higher coordinate derivatives must be finite"))

    result = _shadow_pullback_result(
        values, derivative, second_map, third_map, fourth_map, parameter, C,
    )
    return result
end

@inline function _shadow_bound_order(order::Integer)
    value = try
        Int(order)
    catch error
        (error isa InexactError || error isa OverflowError) || rethrow()
        throw(DomainError(order, "remainder order is outside Int range"))
    end
    1 <= value <= _SHADOW_MAX_MULTIPLIER_TERMS || throw(DomainError(
        order,
        "remainder order must lie in 1:$_SHADOW_MAX_MULTIPLIER_TERMS",
    ))
    return value
end

function _shadow_sensitivity_remainder_bound(
    omega::Real,
    kappa::Real,
    order::Integer,
    spectral_l1::Real,
)
    bandwidth, parameter, mass, R =
        _shadow_real_values(omega, kappa, spectral_l1)
    bandwidth_primal = _primal_value(bandwidth)
    mass_primal = _primal_value(mass)
    bandwidth_primal >= zero(bandwidth_primal) || throw(DomainError(
        omega, "bandwidth omega must be finite and nonnegative",
    ))
    _shadow_positive(parameter, kappa, "kappa")
    mass_primal >= zero(mass_primal) || throw(DomainError(
        spectral_l1, "spectral_l1 must be finite and nonnegative",
    ))
    retained_order = _shadow_bound_order(order)
    primal_bandwidth = convert(R, bandwidth_primal)
    primal_parameter = convert(R, _primal_value(parameter))
    primal_mass = convert(R, mass_primal)
    zero_primal = iszero(primal_bandwidth) || iszero(primal_mass)
    upper = zero_primal ? zero(R) : primal_mass
    overflowed = false
    if !zero_primal
        @inbounds for index in 1:retained_order
            upper = nextfloat(upper * primal_bandwidth)
            upper = nextfloat(upper * primal_bandwidth)
            upper = nextfloat(upper / R(4))
            upper = nextfloat(upper / primal_parameter)
            upper = nextfloat(upper / R(index))
            if !isfinite(upper)
                overflowed = true
                break
            end
        end
    end
    overflowed && (upper = _shadow_remainder_wide(
        primal_bandwidth, primal_parameter, primal_mass, retained_order, R,
    ))
    if _number_contains_ad(bandwidth) || _number_contains_ad(parameter) ||
       _number_contains_ad(mass)
        differentiated::typeof(bandwidth) = _shadow_remainder_ad_wide(
            bandwidth, parameter, mass, retained_order, upper,
        )
        return differentiated
    end
    return upper
end

function _shadow_remainder_ad_wide(
    bandwidth::T,
    parameter::T,
    mass::T,
    retained_order::Int,
    upper::R,
)::T where {T<:Real,R<:AbstractFloat}
    value = _shadow_stable_wide_value(
        T, 256, "shadow-sensitivity remainder derivative",
    ) do
        wide_bandwidth = _faddeeva_widen_bigfloat(bandwidth)
        wide_parameter = _faddeeva_widen_bigfloat(parameter)
        result = _faddeeva_widen_bigfloat(mass)
        for index in 1:retained_order
            result *= wide_bandwidth
            result *= wide_bandwidth
            result /= 4wide_parameter
            result /= BigFloat(index)
        end
        result
    end
    result = (value - _primal_value(value)) + upper
    _number_isfinite(result) || throw(DomainError(
        result,
        "shadow-sensitivity remainder derivative exceeds the active range",
    ))
    return result
end

function _shadow_remainder_zero_primal(
    bandwidth::Real,
    parameter::Real,
    mass::Real,
    retained_order::Int,
)
    value = mass
    @inbounds for index in 1:retained_order
        value *= bandwidth
        value *= bandwidth
        value /= 4one(parameter)
        value /= parameter
        value /= index
    end
    _number_isfinite(value) || throw(DomainError(
        value, "shadow-sensitivity remainder derivative exceeds the active range",
    ))
    return value
end

function _shadow_remainder_wide(
    bandwidth::R,
    parameter::R,
    mass::R,
    retained_order::Int,
    ::Type{R},
)::R where {R<:AbstractFloat}
    return setprecision(BigFloat, 512) do
        wide = setrounding(BigFloat, RoundUp) do
            value = BigFloat(mass)
            wide_bandwidth = BigFloat(bandwidth)
            wide_parameter = BigFloat(parameter)
            for index in 1:retained_order
                value *= wide_bandwidth
                value *= wide_bandwidth
                value /= BigFloat(4)
                value /= wide_parameter
                value /= BigFloat(index)
            end
            value
        end
        converted = convert(R, wide)
        isfinite(converted) || throw(DomainError(
            wide, "shadow-sensitivity remainder bound overflows",
        ))
        BigFloat(converted) < wide ? nextfloat(converted) : converted
    end
end

@inline function _shadow_halfplane_values(s::Real, kappa::Real)
    coordinate, parameter, _ = _shadow_real_values(s, kappa)
    _shadow_positive(parameter, kappa, "kappa")
    return coordinate, parameter
end

function _shadow_halfplane_coordinate(s::Real, kappa::Real)
    coordinate, parameter = _shadow_halfplane_values(s, kappa)
    if abs(_primal_value(coordinate)) <= one(_primal_value(coordinate))
        pi_value = zero(coordinate) + _typed_pi(coordinate)
        halved = parameter / 2
        scale = iszero(_primal_value(halved)) ?
                sqrt(parameter) / sqrt(2one(parameter)) : sqrt(halved)
        return scale * coordinate * sinc(coordinate / (2pi_value))
    end
    doubled = 2parameter
    scale = _number_isfinite(doubled) ? sqrt(doubled) :
            sqrt(2one(parameter)) * sqrt(parameter)
    return scale * sin(coordinate / 2)
end

function _shadow_halfplane_coordinate_derivative(s::Real, kappa::Real)
    coordinate, parameter = _shadow_halfplane_values(s, kappa)
    halved = parameter / 2
    scale = iszero(_primal_value(halved)) ?
            sqrt(parameter) / sqrt(2one(parameter)) : sqrt(halved)
    return scale * cos(coordinate / 2)
end

function _shadow_halfplane_coordinate_second(s::Real, kappa::Real)
    return -_shadow_halfplane_coordinate(s, kappa) / 4
end

function _shadow_halfplane_switch(s::Real, kappa::Real)
    return shadow_switch(_shadow_halfplane_coordinate(s, kappa))
end

function _shadow_halfplane_first(s::Real, kappa::Real)
    coordinate = _shadow_halfplane_coordinate(s, kappa)
    derivative = _shadow_halfplane_coordinate_derivative(s, kappa)
    value = shadow_sensitivity_kernel(coordinate) * derivative
    _number_isfinite(value) || throw(DomainError(
        value, "half-plane first shadow derivative is non-finite",
    ))
    return value
end

function _shadow_halfplane_second(s::Real, kappa::Real)
    coordinate = _shadow_halfplane_coordinate(s, kappa)
    first_coordinate = _shadow_halfplane_coordinate_derivative(s, kappa)
    second_coordinate = _shadow_halfplane_coordinate_second(s, kappa)
    kernel = shadow_sensitivity_kernel(coordinate)
    value = kernel * (
        second_coordinate - 2im * coordinate * first_coordinate^2
    )
    _number_isfinite(value) || throw(DomainError(
        value, "half-plane second shadow derivative is non-finite",
    ))
    return value
end

for function_name in (
    :_shadow_halfplane_coordinate,
    :_shadow_halfplane_coordinate_derivative,
    :_shadow_halfplane_coordinate_second,
    :_shadow_halfplane_switch,
    :_shadow_halfplane_first,
    :_shadow_halfplane_second,
)
    @eval function $function_name(
        values::AbstractArray{<:Real}, kappa::Real,
    )
        element_type = _shadow_array_element_type(
            values, "half-plane shadow-coordinate",
        )
        _shadow_halfplane_values(zero(element_type), kappa)
        return map(value -> $function_name(value, kappa), values)
    end
end
