"""
Canonical Faddeeva moments and null-uniform amplitude hierarchies.

The binary evaluator shares `_faddeeva_w` with other transition families,
routes between a certified forward recurrence and independent least-term
moments, and uses bounded precision recovery only when neither binary
representation meets its error target.
"""

const _NULL_MOMENT_MAX_ORDER = 64
const _NULL_MOMENT_DEFAULT_SWITCH = 10.0
const _NULL_MOMENT_DEFAULT_MAX_TERMS = 80
const _NULL_MOMENT_MAX_TERMS = 256
const _NULL_MOMENT_CERTIFICATE_SAFETY = 2
const _NULL_MOMENT_MAX_RECOVERY_BITS = 1024

@inline function _null_gaussian_moment(order::Integer, scalar::Real)
    order >= 0 || throw(DomainError(order, "Gaussian moment order must be nonnegative"))
    isodd(order) && return zero(scalar)
    pi_value = zero(scalar) + _typed_pi(scalar)
    value = sqrt(pi_value)
    @inbounds for index in 1:(order ÷ 2)
        value *= (zero(scalar) + (2index - 1)) / (2one(scalar))
    end
    _number_isfinite(value) || throw(DomainError(
        order, "Gaussian moment exceeds the active numeric range",
    ))
    return value
end

function _null_complex_value(z::Number)
    value = complex(float(z))
    _number_isfinite(value) || throw(DomainError(z, "z must be finite"))
    primal = _primal_value(real(value))
    primal isa BigFloat && throw(ArgumentError(
        "faddeeva_moments does not expose a BigFloat production path; " *
        "use an independent high-precision oracle",
    ))
    return value, typeof(float(primal))
end

@inline function _null_real_magnitude(value, ::Type{R}) where {R<:Real}
    return convert(R, float(_primal_value(abs(value))))
end

function _null_moment_recurrence_with_error(z::Number, max_order::Integer)
    max_order >= 0 || throw(DomainError(max_order, "moment order must be nonnegative"))
    max_order <= _NULL_MOMENT_MAX_ORDER || throw(DomainError(
        max_order, "moment order exceeds $_NULL_MOMENT_MAX_ORDER",
    ))
    value_z, R = _null_complex_value(z)
    C = typeof(value_z)
    values = Vector{C}(undef, max_order + 1)
    errors = Vector{R}(undef, max_order + 1)
    values[1] = _faddeeva_w(value_z)
    _number_isfinite(values[1]) || throw(DomainError(
        values[1], "Faddeeva value exceeds the active numeric range",
    ))
    unit_roundoff = eps(R)
    smallest = nextfloat(zero(R))
    errors[1] = 8unit_roundoff * max(
        _null_real_magnitude(values[1], R), smallest,
    )
    scalar = real(zero(C))
    pi_value = zero(scalar) + _typed_pi(scalar)
    @inbounds for order in 0:(max_order - 1)
        product = value_z * values[order + 1]
        subtraction = complex(
            zero(scalar), _null_gaussian_moment(order, scalar) / pi_value,
        )
        value = product - subtraction
        _number_isfinite(value) || throw(DomainError(
            value,
            "moment recurrence exceeds the active numeric range at order $(order + 1)",
        ))
        values[order + 2] = value
        errors[order + 2] = _null_real_magnitude(value_z, R) * errors[order + 1] +
            8unit_roundoff * (
                _null_real_magnitude(product, R) +
                _null_real_magnitude(subtraction, R) +
                _null_real_magnitude(value, R)
            )
    end
    return values, errors
end

function _null_moment_recurrence(z::Number, max_order::Integer)
    values, _ = _null_moment_recurrence_with_error(z, max_order)
    return values
end

function _null_moment_asymptotic_with_error(
    z::Number,
    order::Integer;
    max_terms::Integer=_NULL_MOMENT_DEFAULT_MAX_TERMS,
    include_continuation::Bool=true,
)
    order >= 0 || throw(DomainError(order, "moment order must be nonnegative"))
    order <= _NULL_MOMENT_MAX_ORDER || throw(DomainError(
        order, "moment order exceeds $_NULL_MOMENT_MAX_ORDER",
    ))
    1 <= max_terms <= _NULL_MOMENT_MAX_TERMS || throw(DomainError(
        max_terms,
        "max_terms must lie in 1:$_NULL_MOMENT_MAX_TERMS",
    ))
    value_z, R = _null_complex_value(z)
    iszero(value_z) && throw(DomainError(z, "large-z moments are undefined at z=0"))
    C = typeof(value_z)
    scalar = real(zero(C))
    pi_value = zero(scalar) + _typed_pi(scalar)
    inverse_z = inv(value_z)
    power = inverse_z
    total = zero(C)
    best = zero(C)
    best_magnitude = R(Inf)
    nonzero_terms = 0
    @inbounds for offset in 0:(max_terms - 1)
        gaussian = _null_gaussian_moment(order + offset, scalar)
        if !iszero(_primal_value(gaussian))
            term = complex(zero(scalar), gaussian / pi_value) * power
            total += term
            magnitude = _null_real_magnitude(term, R)
            nonzero_terms += 1
            if magnitude < best_magnitude
                best_magnitude = magnitude
                best = total
            elseif nonzero_terms >= 5
                break
            end
        end
        power *= inverse_z
    end
    isfinite(best_magnitude) || throw(ArgumentError(
        "moment asymptotic produced no nonzero term",
    ))

    continuation = zero(C)
    if include_continuation && _primal_value(imag(value_z)) < 0
        continuation = 2value_z^order * exp(-value_z * value_z)
        best += continuation
    end
    _number_isfinite(best) || throw(DomainError(
        best, "moment asymptotic exceeds the active numeric range",
    ))
    roundoff_scale = max(
        _null_real_magnitude(best, R),
        _null_real_magnitude(continuation, R),
        best_magnitude,
        nextfloat(zero(R)),
    )
    error_estimate = 32best_magnitude + 16eps(R) * roundoff_scale
    return best, error_estimate
end

function _null_moment_asymptotic(z::Number, order::Integer; kwargs...)
    value, _ = _null_moment_asymptotic_with_error(z, order; kwargs...)
    return value
end

function _null_big_erf_series(z::Complex{BigFloat})
    squared = z * z
    term = one(z)
    total = one(z)
    for index in 1:(8precision(BigFloat))
        term *= -squared / index
        addition = term / (2index + 1)
        total += addition
        abs(addition) <= eps(BigFloat) * (abs(total) + eps(BigFloat)) &&
            return 2z * total / sqrt(BigFloat(pi))
    end
    throw(ArgumentError("internal Faddeeva recovery series did not converge"))
end

function _null_big_asymptotic(z::Complex{BigFloat}; max_terms::Int=256)
    inverse_z = inv(z)
    power = inverse_z
    total = zero(z)
    best = total
    best_magnitude = BigFloat(Inf)
    nonzero_terms = 0
    for offset in 0:(max_terms - 1)
        gaussian = _null_gaussian_moment(offset, real(z))
        if !iszero(gaussian)
            term = im * gaussian * power / BigFloat(pi)
            total += term
            magnitude = abs(term)
            nonzero_terms += 1
            if magnitude < best_magnitude
                best_magnitude = magnitude
                best = total
            elseif nonzero_terms >= 5
                break
            end
        end
        power *= inverse_z
    end
    imag(z) < 0 && (best += 2exp(-z * z))
    return best
end

function _null_big_faddeeva(z::Complex{BigFloat}, target_precision::Int)
    threshold = BigFloat(target_precision + 32) * log(BigFloat(2))
    if abs2(z) >= threshold
        return _null_big_asymptotic(z)
    end
    guard_bits = ceil(Int, abs2(z) / log(BigFloat(2))) + 64
    work_precision = target_precision + guard_bits
    value = setprecision(BigFloat, work_precision) do
        local_z = complex(
            BigFloat(real(z); precision=work_precision),
            BigFloat(imag(z); precision=work_precision),
        )
        exp(-local_z * local_z) * (1 - _null_big_erf_series(-im * local_z))
    end
    return complex(
        BigFloat(real(value); precision=target_precision),
        BigFloat(imag(value); precision=target_precision),
    )
end

function _null_precision_recovery!(values::Vector{C}, z::C) where {T<:AbstractFloat,C<:Complex{T}}
    radius = max(one(T), abs(z))
    lost_bits = ceil(Int, (length(values) - 1) * log2(radius))
    required_precision = precision(T) + max(lost_bits, 0) + 32
    target_precision = max(128, 32 * cld(required_precision, 32))
    target_precision <= _NULL_MOMENT_MAX_RECOVERY_BITS || throw(ArgumentError(
        "moment precision recovery requires $target_precision bits, above the " *
        "bounded $_NULL_MOMENT_MAX_RECOVERY_BITS-bit workspace",
    ))
    return setprecision(BigFloat, target_precision) do
        value_z = complex(BigFloat(real(z)), BigFloat(imag(z)))
        value = _null_big_faddeeva(value_z, target_precision)
        values[1] = convert(C, value)
        _number_isfinite(values[1]) || throw(DomainError(
            values[1],
            "precision-recovered Faddeeva value exceeds the active numeric range",
        ))
        @inbounds for order in 0:(length(values) - 2)
            value = value_z * value -
                    im * _null_gaussian_moment(order, real(value_z)) / BigFloat(pi)
            converted = convert(C, value)
            _number_isfinite(converted) || throw(DomainError(
                converted,
                "precision-recovered moment exceeds the active numeric range",
            ))
            values[order + 2] = converted
        end
        values
    end
end

function _null_recover_or_throw!(values, value_z)
    scalar = real(zero(eltype(values)))
    scalar isa AbstractFloat || throw(ArgumentError(
        "moment precision recovery is unavailable for $(typeof(value_z)); " *
        "choose an order/coordinate certified by a binary branch",
    ))
    return _null_precision_recovery!(values, value_z)
end

@inline function _null_default_rtol(::Type{R}) where {R<:AbstractFloat}
    return max(R(1e-11), 64eps(R))
end

"""
    faddeeva_moments(z, max_order; switch=10, rtol=nothing, max_terms=80)

Return canonical moments `W_0:W_max_order` satisfying
`W_0=w(z)` and `W_(m+1)=z*W_m-im*mu_m/pi`. A propagated-error recurrence is
preferred for `abs(z)<=switch`; independently evaluated least-term moments are
preferred beyond it. The lower-half-plane asymptotic includes its Stokes
continuation. If neither binary representation meets `rtol`, a bounded
internal precision recovery supplies the requested binary result.

Orders `0:64` are supported. Float32 and Float64 inputs retain their complex
element type. Complex BigFloat is intentionally reserved for independent
oracles.
"""
function faddeeva_moments(
    z::Number,
    max_order::Integer;
    switch::Real=_NULL_MOMENT_DEFAULT_SWITCH,
    rtol::Union{Nothing,Real}=nothing,
    max_terms::Integer=_NULL_MOMENT_DEFAULT_MAX_TERMS,
)
    0 <= max_order <= _NULL_MOMENT_MAX_ORDER || throw(DomainError(
        max_order,
        "max_order must lie in 0:$_NULL_MOMENT_MAX_ORDER",
    ))
    switch_primal = _primal_value(switch)
    (isfinite(switch_primal) && switch_primal > zero(switch_primal)) ||
        throw(DomainError(switch, "switch must be finite and positive"))
    1 <= max_terms <= _NULL_MOMENT_MAX_TERMS || throw(DomainError(
        max_terms,
        "max_terms must lie in 1:$_NULL_MOMENT_MAX_TERMS",
    ))

    value_z, R = _null_complex_value(z)
    relative_tolerance = rtol === nothing ? _null_default_rtol(R) :
                         convert(R, _primal_value(rtol))
    (isfinite(relative_tolerance) && relative_tolerance > zero(R)) ||
        throw(DomainError(rtol, "rtol must be finite and positive"))
    switch_value = convert(R, switch_primal)
    (isfinite(switch_value) && switch_value > zero(R)) ||
        throw(DomainError(switch, "switch is outside the active numeric range"))

    magnitude = _null_real_magnitude(value_z, R)
    if magnitude > switch_value
        values = Vector{typeof(value_z)}(undef, max_order + 1)
        @inbounds for order in 0:max_order
            asymptotic, asymptotic_error = _null_moment_asymptotic_with_error(
                value_z, order; max_terms=max_terms,
            )
            asymptotic_scale = max(
                _null_real_magnitude(asymptotic, R), nextfloat(zero(R)),
            )
            if _NULL_MOMENT_CERTIFICATE_SAFETY * asymptotic_error >
               relative_tolerance * asymptotic_scale
                return _null_recover_or_throw!(values, value_z)
            end
            values[order + 1] = asymptotic
        end
        return values
    end

    values, recurrence_errors = _null_moment_recurrence_with_error(value_z, max_order)
    needs_recovery = false
    @inbounds for order in 0:max_order
        recurrence_scale = max(
            _null_real_magnitude(values[order + 1], R), nextfloat(zero(R)),
        )
        recurrence_ok = _NULL_MOMENT_CERTIFICATE_SAFETY *
                        recurrence_errors[order + 1] <=
                        relative_tolerance * recurrence_scale
        recurrence_ok && continue

        asymptotic_ok = false
        asymptotic = zero(eltype(values))
        if !iszero(value_z)
            asymptotic, asymptotic_error = _null_moment_asymptotic_with_error(
                value_z, order; max_terms=max_terms,
            )
            asymptotic_scale = max(
                _null_real_magnitude(asymptotic, R), nextfloat(zero(R)),
            )
            asymptotic_ok = _NULL_MOMENT_CERTIFICATE_SAFETY *
                            asymptotic_error <=
                            relative_tolerance * asymptotic_scale
        end
        if asymptotic_ok
            values[order + 1] = asymptotic
        else
            needs_recovery = true
            break
        end
    end

    if needs_recovery
        _null_recover_or_throw!(values, value_z)
    end
    return values
end

@inline function _null_transition_type(k::Real, z::Number, ::Type{A}) where {A<:Number}
    isconcretetype(A) || throw(ArgumentError(
        "amplitude coefficients require a concrete numeric element type",
    ))
    return promote_type(
        typeof(complex(float(k))),
        typeof(complex(float(z))),
        typeof(complex(float(zero(A)))),
    )
end

@inline function _null_validate_positive_k(k::Real)
    primal = _primal_value(k)
    (isfinite(primal) && primal > zero(primal)) ||
        throw(DomainError(k, "k must be finite and positive"))
    return k
end

@inline function _null_validate_coefficient(value::Number, name::AbstractString)
    _number_isfinite(value) || throw(DomainError(value, "$name must be finite"))
    return value
end

"""
    null_uniform_transition(k, z, coefficients; order=nothing, kwargs...)

Evaluate `sum(a_m*k^(-m/2)*W_m(z))` for caller-supplied ascending amplitude
coefficients. `k` must be finite and positive. `order` may truncate the
available coefficients; no coefficient array is copied.
"""
function null_uniform_transition(
    k::Real,
    z::Number,
    coefficients::AbstractVector{A};
    order::Union{Nothing,Integer}=nothing,
    switch::Real=_NULL_MOMENT_DEFAULT_SWITCH,
    rtol::Union{Nothing,Real}=nothing,
    max_terms::Integer=_NULL_MOMENT_DEFAULT_MAX_TERMS,
) where {A<:Number}
    _null_validate_positive_k(k)
    C = _null_transition_type(k, z, A)
    coefficient_count = length(coefficients)
    coefficient_count == 0 && return zero(C)
    for index in eachindex(coefficients)
        _null_validate_coefficient(coefficients[index], "amplitude coefficient")
    end
    requested_order = order === nothing ? coefficient_count - 1 : order
    requested_order >= 0 || throw(DomainError(order, "order must be nonnegative"))
    maximum_order = Int(min(requested_order, coefficient_count - 1))
    maximum_order <= _NULL_MOMENT_MAX_ORDER || throw(DomainError(
        maximum_order,
        "retained amplitude order exceeds $_NULL_MOMENT_MAX_ORDER",
    ))

    value_z = convert(C, complex(float(z)))
    moments = faddeeva_moments(
        value_z, maximum_order; switch=switch, rtol=rtol, max_terms=max_terms,
    )
    k_value = real(zero(C)) + float(k)
    inverse_scale = inv(sqrt(k_value))
    scale = one(k_value)
    total = zero(C)
    compensation = zero(C)
    first_coefficient_index = firstindex(coefficients)
    @inbounds for moment_order in 0:maximum_order
        coefficient = convert(C, coefficients[first_coefficient_index + moment_order])
        term = coefficient * scale * moments[moment_order + 1]
        total, compensation = _multipole_compensated_add(total, compensation, term)
        scale *= inverse_scale
    end
    _number_isfinite(total) || throw(DomainError(
        total, "null-uniform transition exceeds the active numeric range",
    ))
    return total
end

@inline function _null_shifted_basis(moments, lambda, zero_order::Int, offset::Int)
    basis = moments[offset + 1]
    @inbounds for index in 1:zero_order
        basis = basis * lambda +
                binomial(zero_order, index) * moments[offset + index + 1]
    end
    return basis
end

"""
    shifted_null_transition(k, z, Lambda, zero_order, coefficients;
                            order=nothing, kwargs...)

Evaluate the hierarchy for
`a(t)=(t+Lambda/sqrt(k))^zero_order*sum(coefficients[q+1]*t^q)`.
All required moments are computed once. Each shifted binomial basis is
evaluated by Horner's rule without temporary powers or moment vectors.
"""
function shifted_null_transition(
    k::Real,
    z::Number,
    Lambda::Number,
    zero_order::Integer,
    coefficients::AbstractVector{A};
    order::Union{Nothing,Integer}=nothing,
    switch::Real=_NULL_MOMENT_DEFAULT_SWITCH,
    rtol::Union{Nothing,Real}=nothing,
    max_terms::Integer=_NULL_MOMENT_DEFAULT_MAX_TERMS,
) where {A<:Number}
    _null_validate_positive_k(k)
    0 <= zero_order <= _NULL_MOMENT_MAX_ORDER || throw(DomainError(
        zero_order,
        "zero_order must lie in 0:$_NULL_MOMENT_MAX_ORDER",
    ))
    C = promote_type(
        _null_transition_type(k, z, A), typeof(complex(float(Lambda))),
    )
    coefficient_count = length(coefficients)
    coefficient_count == 0 && return zero(C)
    for index in eachindex(coefficients)
        _null_validate_coefficient(coefficients[index], "amplitude coefficient")
    end
    requested_order = order === nothing ? coefficient_count - 1 : order
    requested_order >= 0 || throw(DomainError(order, "order must be nonnegative"))
    maximum_coefficient_order = Int(min(requested_order, coefficient_count - 1))
    maximum_moment_order = maximum_coefficient_order + Int(zero_order)
    maximum_moment_order <= _NULL_MOMENT_MAX_ORDER || throw(DomainError(
        maximum_moment_order,
        "shifted hierarchy requires moments above $_NULL_MOMENT_MAX_ORDER",
    ))

    value_z = convert(C, complex(float(z)))
    lambda = convert(C, complex(float(Lambda)))
    _null_validate_coefficient(lambda, "Lambda")
    moments = faddeeva_moments(
        value_z, maximum_moment_order;
        switch=switch, rtol=rtol, max_terms=max_terms,
    )
    k_value = real(zero(C)) + float(k)
    inverse_scale = inv(sqrt(k_value))
    scale = inverse_scale^Int(zero_order)
    total = zero(C)
    compensation = zero(C)
    first_coefficient_index = firstindex(coefficients)
    @inbounds for coefficient_order in 0:maximum_coefficient_order
        basis = _null_shifted_basis(
            moments, lambda, Int(zero_order), coefficient_order,
        )
        coefficient = convert(
            C, coefficients[first_coefficient_index + coefficient_order],
        )
        term = coefficient * scale * basis
        total, compensation = _multipole_compensated_add(total, compensation, term)
        scale *= inverse_scale
    end
    _number_isfinite(total) || throw(DomainError(
        total, "shifted null transition exceeds the active numeric range",
    ))
    return total
end
