"""
Internal scaled-complementary-error-function/Faddeeva identities shared by
transition families. Public APIs keep problem-specific names and domains.
"""

using SpecialFunctions: erfcx

const _FADDEEVA_FORWARD_RADIUS = 1.5
const _FADDEEVA_TAYLOR_RADIUS = 12.0
const _FADDEEVA_STABLE_MAX_ORDER = 256

@inline function _faddeeva_erfcx(z::Number)
    try
        return erfcx(z)
    catch error
        error isa MethodError || rethrow()
        throw(ArgumentError(
            "scaled complementary error function is unavailable for $(typeof(z))",
        ))
    end
end

"""Internal Faddeeva function `w(z)=erfcx(-im*z)`."""
@inline function _faddeeva_w(
    z::T,
)::typeof(complex(float(zero(T)))) where {T<:Number}
    radius = float(_primal_value(abs(z)))
    if radius > oftype(radius, _FADDEEVA_TAYLOR_RADIUS)
        return _faddeeva_scaled_derivative_asymptotic(z, 0)
    end
    return _faddeeva_erfcx(-im * z)
end

@inline function _faddeeva_scaled_derivative_seed(z::Number)
    value = _faddeeva_w(z)
    scalar = real(zero(value))
    pi_value = zero(scalar) + _typed_pi(scalar)
    derivative = -2z * value + complex(zero(scalar), 2one(scalar) / sqrt(pi_value))
    return value, derivative
end

@inline function _faddeeva_scaled_derivative_next(
    z::Number,
    previous::Number,
    current::Number,
    order::Integer,
)
    return (-2z * current - 2previous) / (order + 1)
end

@inline function _faddeeva_forward_recurrence_safe(z::Number)
    radius = float(_primal_value(abs(z)))
    return radius <= oftype(radius, _FADDEEVA_FORWARD_RADIUS)
end

@inline function _faddeeva_taylor_route(z::Number)
    radius = float(_primal_value(abs(z)))
    return radius <= oftype(radius, _FADDEEVA_TAYLOR_RADIUS)
end

@inline function _faddeeva_widen_bigfloat(value::Real)
    primal = _primal_value(value)
    return (value - primal) + BigFloat(primal)
end

@inline function _faddeeva_widen_bigfloat(value::Complex)
    return complex(
        _faddeeva_widen_bigfloat(real(value)),
        _faddeeva_widen_bigfloat(imag(value)),
    )
end

@inline function _faddeeva_widen_float64(value::Real)
    primal = _primal_value(value)
    return (value - primal) + Float64(primal)
end

@inline function _faddeeva_widen_float64(value::Complex)
    return complex(
        _faddeeva_widen_float64(real(value)),
        _faddeeva_widen_float64(imag(value)),
    )
end

@inline function _faddeeva_output_type(z::Number)
    return typeof(complex(float(zero(typeof(z)))))
end

function _faddeeva_taylor_work_precision(z::Number, maximum_order::Integer)
    maximum_order <= _FADDEEVA_STABLE_MAX_ORDER || throw(ArgumentError(
        "stable Faddeeva derivatives support orders through " *
        "$_FADDEEVA_STABLE_MAX_ORDER",
    ))
    primal_radius = Float64(_primal_value(abs(z)))
    base_type = typeof(float(_primal_value(real(z))))
    target_bits = precision(base_type)
    series_bits = ceil(Int, primal_radius^2 / log(2))
    recurrence_bits = ceil(
        Int,
        2maximum_order * log2(max(2.0, primal_radius)),
    )
    return target_bits + series_bits + recurrence_bits + 96
end

function _faddeeva_taylor_seed(z::Number, primal_radius::Float64)
    even = one(z)
    even_term = one(z)
    pi_value = BigFloat(pi)
    odd = 2im * z / sqrt(pi_value)
    odd_term = odd
    small_run = 0
    # Higher derivative orders deliberately raise the working precision so the
    # recurrence can retain type-local accuracy.  The Taylor seed must be able
    # to reach that same precision; a radius-only cap can stop too early.
    maximum_terms = max(
        256,
        ceil(Int, 8(primal_radius^2 + 1)),
        2 * precision(BigFloat),
    )
    tolerance = 8eps(BigFloat)
    @inbounds for index in 1:maximum_terms
        even_term *= -(z * z) / index
        odd_term *= -(z * z) / (BigFloat(index) + BigFloat(1) / 2)
        even += even_term
        odd += odd_term
        term_magnitude = _primal_value(abs(even_term) + abs(odd_term))
        total_magnitude = _primal_value(abs(even) + abs(odd))
        small_run = term_magnitude <=
                    tolerance * max(one(BigFloat), total_magnitude) ?
                    small_run + 1 : 0
        small_run >= 4 && return even + odd
    end
    throw(ArgumentError("high-precision Faddeeva Taylor series did not converge"))
end

function _faddeeva_scaled_derivatives_taylor(
    z::T,
    maximum_order::Integer,
)::Vector{typeof(complex(float(zero(T))))} where {T<:Number}
    work_precision = _faddeeva_taylor_work_precision(z, maximum_order)
    primal_radius = Float64(_primal_value(abs(z)))
    output_type = _faddeeva_output_type(z)
    return setprecision(BigFloat, work_precision) do
        high_z = _faddeeva_widen_bigfloat(z)
        value = _faddeeva_taylor_seed(high_z, primal_radius)
        values = Vector{output_type}(undef, maximum_order + 1)
        values[1] = convert(output_type, value)
        maximum_order == 0 && return values
        scalar = real(zero(value))
        derivative = -2high_z * value +
                     complex(zero(scalar), 2one(scalar) / sqrt(BigFloat(pi)))
        values[2] = convert(output_type, derivative)
        previous, current = value, derivative
        @inbounds for order in 1:(maximum_order - 1)
            previous, current = current, _faddeeva_scaled_derivative_next(
                high_z, previous, current, order,
            )
            values[order + 2] = convert(output_type, current)
        end
        values
    end
end

@inline function _faddeeva_scaled_gaussian(
    z::T,
    order::Integer,
)::typeof(complex(float(zero(T)))) where {T<:Number}
    primal_z = complex(_primal_value(real(z)), _primal_value(imag(z)))
    primal_squared = primal_z * primal_z
    if !_number_isfinite(primal_squared)
        return setprecision(BigFloat, 256) do
            high_z = _faddeeva_widen_bigfloat(z)
            previous = exp(-(high_z * high_z))
            if order == 0
                return convert(_faddeeva_output_type(z), previous)
            end
            current = -2high_z * previous
            @inbounds for index in 1:(order - 1)
                previous, current = current,
                    (-2high_z * current - 2previous) / (index + 1)
            end
            convert(_faddeeva_output_type(z), current)
        end
    end
    previous = exp(-(z * z))
    order == 0 && return previous
    current = -2z * previous
    @inbounds for index in 1:(order - 1)
        previous, current = current,
            (-2z * current - 2previous) / (index + 1)
    end
    return current
end

function _faddeeva_scaled_derivative_asymptotic_upper(
    z::T,
    order::Integer,
)::typeof(complex(float(zero(T)))) where {T<:Number}
    primal_z = complex(_primal_value(real(z)), _primal_value(imag(z)))
    imag(primal_z) >= zero(imag(primal_z)) || throw(ArgumentError(
        "upper-half-plane Faddeeva asymptotics require nonnegative imaginary part",
    ))
    if real(primal_z) < zero(real(primal_z))
        parity = isodd(order) ? -1 : 1
        return parity * conj(
            _faddeeva_scaled_derivative_asymptotic_upper(-conj(z), order),
        )
    end

    work_z = _faddeeva_widen_float64(z)
    scalar_type = typeof(float(_primal_value(real(work_z))))
    inverse_z = inv(work_z)
    inverse_square = inverse_z * inverse_z
    term = inverse_z^(order + 1)
    total = term
    previous_magnitude = float(_primal_value(abs(term)))
    tolerance = 8eps(scalar_type)
    converged = iszero(previous_magnitude)
    @inbounds for index in 0:512
        numerator = scalar_type(2index + order + 2) *
                    scalar_type(2index + order + 1)
        denominator = scalar_type(4(index + 1))
        next_term = term * (numerator / denominator) * inverse_square
        next_magnitude = float(_primal_value(abs(next_term)))
        next_magnitude < previous_magnitude || break
        total += next_term
        total_magnitude = float(_primal_value(abs(total)))
        if next_magnitude <= tolerance * max(floatmin(scalar_type), total_magnitude)
            converged = true
            break
        end
        term = next_term
        previous_magnitude = next_magnitude
    end
    converged || throw(ArgumentError(
        "Faddeeva asymptotic derivative did not reach type-local tolerance",
    ))
    parity = isodd(order) ? -one(work_z) : one(work_z)
    scalar = real(zero(work_z))
    pi_value = zero(scalar) + _typed_pi(scalar)
    value = parity * (im * one(work_z)) * total / sqrt(pi_value)
    if iszero(imag(primal_z))
        value += _faddeeva_scaled_gaussian(work_z, order)
    end
    return convert(_faddeeva_output_type(z), value)
end

function _faddeeva_scaled_derivative_asymptotic(
    z::T,
    order::Integer,
)::typeof(complex(float(zero(T)))) where {T<:Number}
    primal_z = complex(_primal_value(real(z)), _primal_value(imag(z)))
    if imag(primal_z) < zero(imag(primal_z)) ||
       (iszero(imag(primal_z)) && real(primal_z) < zero(real(primal_z)))
        parity = isodd(order) ? -1 : 1
        value = 2 * _faddeeva_scaled_gaussian(z, order) -
                parity * _faddeeva_scaled_derivative_asymptotic_upper(-z, order)
        return convert(_faddeeva_output_type(z), value)
    end
    return _faddeeva_scaled_derivative_asymptotic_upper(z, order)
end

function _faddeeva_scaled_derivative_forward(
    z::T,
    order::Integer,
)::typeof(complex(float(zero(T)))) where {T<:Number}
    order >= 0 || throw(DomainError(order, "Faddeeva derivative order must be nonnegative"))
    _validate_finite_number(z, "Faddeeva argument")
    value, derivative = _faddeeva_scaled_derivative_seed(z)
    order == 0 && return value
    previous, current = value, derivative
    @inbounds for index in 1:(order - 1)
        previous, current = current,
            _faddeeva_scaled_derivative_next(z, previous, current, index)
    end
    _number_isfinite(current) || throw(DomainError(
        (z, order),
        "scaled Faddeeva derivative is non-finite",
    ))
    return current
end

function _faddeeva_scaled_derivative(
    z::T,
    order::Integer,
)::typeof(complex(float(zero(T)))) where {T<:Number}
    order >= 0 || throw(DomainError(order, "Faddeeva derivative order must be nonnegative"))
    _validate_finite_number(z, "Faddeeva argument")
    if _faddeeva_forward_recurrence_safe(z)
        return _faddeeva_scaled_derivative_forward(z, order)
    elseif _faddeeva_taylor_route(z)
        return _faddeeva_scaled_derivatives_taylor(z, order)[end]
    end
    return _faddeeva_scaled_derivative_asymptotic(z, order)
end

"""
Internal sequence `w^(n)(z)/n!` for `n=0:nmax`. The implementation uses the
direct recurrence only in its verified stable regime, a widened entire series
at intermediate arguments, and differentiated asymptotics at large arguments.
"""
function _faddeeva_scaled_derivatives_forward(
    z::T,
    nmax::Integer,
)::Vector{typeof(complex(float(zero(T))))} where {T<:Number}
    nmax >= 0 || throw(DomainError(nmax, "Faddeeva derivative order must be nonnegative"))
    _validate_finite_number(z, "Faddeeva argument")
    value, derivative = _faddeeva_scaled_derivative_seed(z)
    values = Vector{typeof(value)}(undef, nmax + 1)
    values[1] = value
    nmax == 0 && return values
    values[2] = derivative
    @inbounds for index in 1:(nmax - 1)
        values[index + 2] = _faddeeva_scaled_derivative_next(
            z, values[index], values[index + 1], index,
        )
        _number_isfinite(values[index + 2]) || throw(DomainError(
            (z, nmax),
            "scaled Faddeeva derivative sequence became non-finite",
        ))
    end
    return values
end

function _faddeeva_scaled_derivatives(
    z::T,
    nmax::Integer,
)::Vector{typeof(complex(float(zero(T))))} where {T<:Number}
    nmax >= 0 || throw(DomainError(nmax, "Faddeeva derivative order must be nonnegative"))
    _validate_finite_number(z, "Faddeeva argument")
    if _faddeeva_forward_recurrence_safe(z)
        return _faddeeva_scaled_derivatives_forward(z, nmax)
    elseif _faddeeva_taylor_route(z)
        return _faddeeva_scaled_derivatives_taylor(z, nmax)
    end
    output_type = _faddeeva_output_type(z)
    values = Vector{output_type}(undef, nmax + 1)
    @inbounds for order in 0:nmax
        values[order + 1] = _faddeeva_scaled_derivative_asymptotic(z, order)
    end
    return values
end
