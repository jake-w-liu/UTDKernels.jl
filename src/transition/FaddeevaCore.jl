"""
Internal scaled-complementary-error-function/Faddeeva identities shared by
transition families. Public APIs keep problem-specific names and domains.
"""

using SpecialFunctions: erfcx

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
@inline _faddeeva_w(z::Number) = _faddeeva_erfcx(-im * z)

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

function _faddeeva_scaled_derivative(z::Number, order::Integer)
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

"""
Internal sequence `w^(n)(z)/n!` for `n=0:nmax`.

The recurrence avoids factorial formation and preserves the input arithmetic
supported by the central Faddeeva evaluator.
"""
function _faddeeva_scaled_derivatives(z::Number, nmax::Integer)
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
