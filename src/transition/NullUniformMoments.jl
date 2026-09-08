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

@inline function _null_widen_float32(value::Real)
    primal = _primal_value(value)
    return (value - primal) + Float32(primal)
end

@inline function _null_widen_float32(value::Complex)
    return complex(
        _null_widen_float32(real(value)),
        _null_widen_float32(imag(value)),
    )
end

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
    primal_type = typeof(float(primal))
    primal_type === Float16 && return _null_widen_float32(value), Float32
    primal_type in (Float32, Float64) || throw(ArgumentError(
        "faddeeva_moments supports binary16, binary32, and binary64 inputs",
    ))
    return value, primal_type
end

@inline function _null_real_magnitude(value, ::Type{R}) where {R<:Real}
    return convert(R, float(_primal_value(abs(value))))
end

function _null_moment_recurrence_with_error(z::Number, max_order::Integer)
    max_order >= 0 || throw(DomainError(max_order, "moment order must be nonnegative"))
    max_order <= _NULL_MOMENT_MAX_ORDER || throw(DomainError(
        max_order, "moment order exceeds $_NULL_MOMENT_MAX_ORDER",
    ))
    maximum_order = Int(max_order)
    value_z, R = _null_complex_value(z)
    C = typeof(value_z)
    if R === Float32
        work_values, work_errors = _null_moment_recurrence_with_error(
            _faddeeva_widen_float64(value_z), maximum_order,
        )
        values = Vector{C}(undef, maximum_order + 1)
        errors = Vector{Float32}(undef, maximum_order + 1)
        @inbounds for index in eachindex(values)
            converted = convert(C, work_values[index])
            _number_isfinite(converted) || throw(DomainError(
                converted,
                "moment recurrence exceeds the active numeric range at " *
                "order $(index - 1)",
            ))
            values[index] = converted
            conversion_error = _null_real_magnitude(
                work_values[index] - _faddeeva_widen_float64(converted),
                Float64,
            )
            wide_error = work_errors[index] + conversion_error
            rounded_error = Float32(wide_error)
            if isfinite(rounded_error) && Float64(rounded_error) < wide_error
                rounded_error = nextfloat(rounded_error)
            end
            errors[index] = rounded_error
        end
        return values, errors
    end
    values = Vector{C}(undef, maximum_order + 1)
    errors = Vector{R}(undef, maximum_order + 1)
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
    @inbounds for order in 0:(maximum_order - 1)
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
    moment_order = Int(order)
    maximum_terms = Int(max_terms)
    value_z, R = _null_complex_value(z)
    iszero(value_z) && throw(DomainError(z, "large-z moments are undefined at z=0"))
    # Float32 Gaussian moments overflow before their inverse-power factor can
    # make the asymptotic term representable.  Build the complete term in a
    # Float64 workspace, preserving any ForwardDiff tangent, and round only
    # the finished moment back to the public element type.
    work_z = R === Float32 ? _faddeeva_widen_float64(value_z) : value_z
    C = typeof(work_z)
    scalar = real(zero(C))
    pi_value = zero(scalar) + _typed_pi(scalar)
    inverse_z = inv(work_z)
    power = inverse_z
    total = zero(C)
    best = zero(C)
    best_magnitude = R(Inf)
    nonzero_terms = 0
    @inbounds for offset in 0:(maximum_terms - 1)
        gaussian = _null_gaussian_moment(moment_order + offset, scalar)
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
    if include_continuation && _primal_value(imag(work_z)) < 0
        gaussian = exp(-work_z * work_z)
        power = work_z^moment_order
        direct_continuation = 2power * gaussian
        continuation = if _number_isfinite(direct_continuation) &&
                          (!iszero(direct_continuation) ||
                           iszero(gaussian) && _number_isfinite(power))
            direct_continuation
        else
            setprecision(BigFloat, 256) do
                wide_z = _faddeeva_widen_bigfloat(work_z)
                wide_continuation = 2wide_z^moment_order * exp(-wide_z * wide_z)
                convert(C, wide_continuation)
            end
        end
        _number_isfinite(continuation) || throw(DomainError(
            continuation,
            "moment continuation exceeds the active numeric range at order " *
            "$moment_order",
        ))
        best += continuation
    end
    output_type = typeof(value_z)
    output = convert(output_type, best)
    output_continuation = convert(output_type, continuation)
    _number_isfinite(output) || throw(DomainError(
        output, "moment asymptotic exceeds the active numeric range",
    ))
    roundoff_scale = max(
        _null_real_magnitude(output, R),
        _null_real_magnitude(output_continuation, R),
        convert(R, best_magnitude),
        nextfloat(zero(R)),
    )
    error_estimate = 32convert(R, best_magnitude) +
                     16eps(R) * roundoff_scale
    return output, error_estimate
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
    radius = max(
        one(BigFloat),
        hypot(BigFloat(real(z)), BigFloat(imag(z))),
    )
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

@inline function _null_control_primal(
    value::Real,
    name::AbstractString,
)
    _number_isfinite(value) || throw(DomainError(
        value, "$name must be finite",
    ))
    _number_contains_ad(value) && throw(ArgumentError(
        "$name does not accept automatic-differentiation inputs",
    ))
    return _primal_value(value)
end

"""
    faddeeva_moments(z, max_order; switch=10, rtol=nothing, max_terms=80)

Return canonical moments `W_0:W_max_order` satisfying
`W_0=w(z)` and `W_(m+1)=z*W_m-im*mu_m/pi`. A propagated-error recurrence is
preferred for `abs(z)<=switch`; independently evaluated least-term moments are
preferred beyond it. The lower-half-plane asymptotic includes its Stokes
continuation. If neither binary representation meets `rtol`, a bounded
internal precision recovery supplies the requested binary result.

ForwardDiff coordinates recompute and stabilize the complete requested moment
vector in bounded wide workspaces before conversion.

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
    maximum_order = Int(max_order)
    switch_primal = _null_control_primal(switch, "switch")
    (isfinite(switch_primal) && switch_primal > zero(switch_primal)) ||
        throw(DomainError(switch, "switch must be finite and positive"))
    1 <= max_terms <= _NULL_MOMENT_MAX_TERMS || throw(DomainError(
        max_terms,
        "max_terms must lie in 1:$_NULL_MOMENT_MAX_TERMS",
    ))
    maximum_terms = Int(max_terms)

    value_z, R = _null_complex_value(z)
    relative_tolerance = rtol === nothing ? _null_default_rtol(R) :
                         convert(R, _null_control_primal(rtol, "rtol"))
    (isfinite(relative_tolerance) && relative_tolerance > zero(R)) ||
        throw(DomainError(rtol, "rtol must be finite and positive"))
    switch_value = convert(R, switch_primal)
    (isfinite(switch_value) && switch_value > zero(R)) ||
        throw(DomainError(switch, "switch is outside the active numeric range"))

    _number_contains_ad(value_z) &&
        return _null_stable_wide_moments(value_z, maximum_order)

    magnitude = _null_real_magnitude(value_z, R)
    if magnitude > switch_value
        values = Vector{typeof(value_z)}(undef, maximum_order + 1)
        @inbounds for order in 0:maximum_order
            asymptotic, asymptotic_error = try
                _null_moment_asymptotic_with_error(
                    value_z, order; max_terms=maximum_terms,
                )
            catch error
                error isa ArgumentError || rethrow()
                return _null_recover_or_throw!(values, value_z)
            end
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

    values, recurrence_errors = try
        _null_moment_recurrence_with_error(value_z, maximum_order)
    catch error
        error isa DomainError || rethrow()
        recovered = Vector{typeof(value_z)}(undef, maximum_order + 1)
        return _null_recover_or_throw!(recovered, value_z)
    end
    needs_recovery = false
    @inbounds for order in 0:maximum_order
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
            try
                asymptotic, asymptotic_error =
                    _null_moment_asymptotic_with_error(
                        value_z, order; max_terms=maximum_terms,
                    )
                asymptotic_scale = max(
                    _null_real_magnitude(asymptotic, R), nextfloat(zero(R)),
                )
                asymptotic_ok = _NULL_MOMENT_CERTIFICATE_SAFETY *
                                asymptotic_error <=
                                relative_tolerance * asymptotic_scale
            catch error
                error isa ArgumentError || rethrow()
            end
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
    candidate = promote_type(
        typeof(complex(float(k))),
        typeof(complex(float(z))),
        typeof(complex(float(zero(A)))),
    )
    primal_type = typeof(float(_primal_value(real(zero(candidate)))))
    if primal_type === Float16
        scalar = _null_widen_float32(real(zero(candidate)))
        return typeof(complex(scalar, scalar))
    end
    return candidate
end

@inline function _null_validate_positive_k(k::Real)
    primal = _primal_value(k)
    (_number_isfinite(k) && primal > zero(primal)) ||
        throw(DomainError(k, "k must be finite and positive"))
    return k
end

@inline function _null_validate_coefficient(value::Number, name::AbstractString)
    _number_isfinite(value) || throw(DomainError(value, "$name must be finite"))
    return value
end

function _null_validate_evaluation_controls(
    value_z::Number,
    maximum_order::Int,
    switch::Real,
    rtol::Union{Nothing,Real},
    max_terms::Integer,
)
    0 <= maximum_order <= _NULL_MOMENT_MAX_ORDER || throw(DomainError(
        maximum_order,
        "max_order must lie in 0:$_NULL_MOMENT_MAX_ORDER",
    ))
    switch_primal = _null_control_primal(switch, "switch")
    (isfinite(switch_primal) && switch_primal > zero(switch_primal)) ||
        throw(DomainError(switch, "switch must be finite and positive"))
    1 <= max_terms <= _NULL_MOMENT_MAX_TERMS || throw(DomainError(
        max_terms,
        "max_terms must lie in 1:$_NULL_MOMENT_MAX_TERMS",
    ))
    _, R = _null_complex_value(value_z)
    relative_tolerance = rtol === nothing ? _null_default_rtol(R) :
                         convert(R, _null_control_primal(rtol, "rtol"))
    (isfinite(relative_tolerance) && relative_tolerance > zero(R)) ||
        throw(DomainError(rtol, "rtol must be finite and positive"))
    switch_value = convert(R, switch_primal)
    (isfinite(switch_value) && switch_value > zero(R)) ||
        throw(DomainError(switch, "switch is outside the active numeric range"))
    return R, relative_tolerance
end

@inline function _null_basis_requires_wide(value::Number)
    _number_isfinite(value) || return true
    real_primal = _primal_value(real(value))
    imag_primal = _primal_value(imag(value))
    return (iszero(real_primal) && iszero(imag_primal)) ||
           (!iszero(real_primal) && issubnormal(real_primal)) ||
           (!iszero(imag_primal) && issubnormal(imag_primal))
end

function _null_wide_erf_series(z::Number)
    squared = z * z
    term = one(z)
    total = one(z)
    for index in 1:(8precision(BigFloat))
        term *= -squared / index
        addition = term / (2index + 1)
        total += addition
        addition_magnitude = _primal_value(abs(addition))
        total_magnitude = _primal_value(abs(total))
        addition_magnitude <= eps(BigFloat) *
            (total_magnitude + eps(BigFloat)) &&
            return 2z * total / sqrt(BigFloat(pi))
    end
    throw(ArgumentError("wide Faddeeva recovery series did not converge"))
end

function _null_wide_asymptotic_moment(z::Number, order::Int)
    inverse_z = inv(z)
    power = inverse_z
    total = zero(z)
    best = total
    best_magnitude = BigFloat(Inf)
    nonzero_terms = 0
    @inbounds for offset in 0:(_NULL_MOMENT_MAX_TERMS - 1)
        gaussian = _null_gaussian_moment(order + offset, BigFloat(0))
        if !iszero(gaussian)
            term = complex(zero(gaussian), gaussian / BigFloat(pi)) * power
            total += term
            magnitude = BigFloat(_primal_value(abs(term)))
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
        "wide moment asymptotic produced no nonzero term",
    ))
    if _primal_value(imag(z)) < 0
        best += 2z^order * exp(-z * z)
    end
    return best
end

function _null_wide_faddeeva(value_z::Number)
    target_precision = precision(BigFloat)
    wide_z = _faddeeva_widen_bigfloat(value_z)
    radius_squared = BigFloat(_primal_value(abs2(wide_z)))
    threshold = BigFloat(target_precision + 32) * log(BigFloat(2))
    radius_squared >= threshold &&
        return _null_wide_asymptotic_moment(wide_z, 0)
    guard_bits = ceil(Int, radius_squared / log(BigFloat(2))) + 64
    return setprecision(BigFloat, target_precision + guard_bits) do
        local_z = _faddeeva_widen_bigfloat(value_z)
        exp(-local_z * local_z) *
            (one(local_z) - _null_wide_erf_series(-im * local_z))
    end
end

function _null_wide_moments(value_z::Number, maximum_order::Int)
    input_precision = precision(typeof(float(_primal_value(real(value_z)))))
    radius = hypot(
        BigFloat(_primal_value(real(value_z))),
        BigFloat(_primal_value(imag(value_z))),
    )
    lost_bits = iszero(maximum_order) || radius <= one(BigFloat) ? 0 :
                ceil(Int, 2(maximum_order + 1) * log2(radius))
    recurrence_precision = input_precision + max(0, lost_bits) + 96
    if recurrence_precision > _NULL_MOMENT_MAX_RECOVERY_BITS
        wide_z = _faddeeva_widen_bigfloat(value_z)
        first = _null_wide_asymptotic_moment(wide_z, 0)
        values = Vector{typeof(first)}(undef, maximum_order + 1)
        values[1] = first
        @inbounds for order in 1:maximum_order
            values[order + 1] = _null_wide_asymptotic_moment(wide_z, order)
        end
        return values
    end

    work_precision = max(precision(BigFloat), recurrence_precision)
    return setprecision(BigFloat, work_precision) do
        wide_z = _faddeeva_widen_bigfloat(value_z)
        value = _null_wide_faddeeva(value_z)
        values = Vector{typeof(value)}(undef, maximum_order + 1)
        values[1] = value
        @inbounds for order in 0:(maximum_order - 1)
            value = wide_z * value -
                complex(
                    zero(BigFloat),
                    _null_gaussian_moment(order, BigFloat(0)) / BigFloat(pi),
                )
            values[order + 2] = value
        end
        values
    end
end

function _null_stable_wide_moments(
    value_z::C,
    maximum_order::Int,
)::Vector{C} where {C<:Number}
    previous = nothing
    @inbounds for work_precision in (256, 512, _NULL_MOMENT_MAX_RECOVERY_BITS)
        wide_values = setprecision(BigFloat, work_precision) do
            _null_wide_moments(value_z, maximum_order)
        end
        converted = Vector{C}(undef, maximum_order + 1)
        for index in eachindex(converted)
            value::C = convert(C, wide_values[index])
            _number_isfinite(value) || throw(DomainError(
                value,
                "differentiated Faddeeva moment exceeds the active numeric " *
                "range at order $(index - 1)",
            ))
            converted[index] = value
        end
        previous !== nothing && isequal(converted, previous) && return converted
        previous = converted
    end
    throw(ArgumentError(
        "differentiated Faddeeva moments did not stabilize within the " *
        "bounded $_NULL_MOMENT_MAX_RECOVERY_BITS-bit workspace",
    ))
end

function _null_stable_wide_result(
    ::Type{C},
    evaluate::F,
    context::AbstractString,
) where {C<:Number,F}
    previous = nothing
    @inbounds for work_precision in (256, 512, _NULL_MOMENT_MAX_RECOVERY_BITS)
        wide_result = setprecision(BigFloat, work_precision) do
            evaluate()
        end
        converted::C = convert(C, wide_result)
        _number_isfinite(converted) || throw(DomainError(
            converted, "$context exceeds the active numeric range",
        ))
        previous !== nothing && isequal(converted, previous) && return converted
        previous = converted
    end
    throw(ArgumentError(
        "$context did not stabilize within the bounded " *
        "$_NULL_MOMENT_MAX_RECOVERY_BITS-bit workspace",
    ))
end

function _null_uniform_transition_wide(
    ::Type{C},
    value_z::C,
    k_value,
    coefficients,
    maximum_order::Int,
) where {C<:Number}
    first_coefficient_index = firstindex(coefficients)
    return _null_stable_wide_result(
        C,
        () -> begin
            moments = _null_wide_moments(value_z, maximum_order)
            wide_k = _faddeeva_widen_bigfloat(k_value)
            inverse_scale = inv(sqrt(wide_k))
            scale = one(wide_k)
            total = zero(eltype(moments))
            @inbounds for order in 0:maximum_order
                coefficient = convert(
                    C, coefficients[first_coefficient_index + order],
                )
                total += _faddeeva_widen_bigfloat(coefficient) *
                         moments[order + 1] * scale
                scale *= inverse_scale
            end
            total
        end,
        "null-uniform transition",
    )::C
end

function _shifted_null_transition_wide(
    ::Type{C},
    value_z::C,
    k_value,
    lambda::C,
    zero_order::Int,
    coefficients,
    maximum_coefficient_order::Int,
) where {C<:Number}
    first_coefficient_index = firstindex(coefficients)
    maximum_moment_order = maximum_coefficient_order + zero_order
    return _null_stable_wide_result(
        C,
        () -> begin
            moments = _null_wide_moments(value_z, maximum_moment_order)
            wide_k = _faddeeva_widen_bigfloat(k_value)
            wide_lambda = _faddeeva_widen_bigfloat(lambda)
            inverse_scale = inv(sqrt(wide_k))
            scale = inverse_scale^zero_order
            total = zero(eltype(moments))
            @inbounds for coefficient_order in 0:maximum_coefficient_order
                basis = moments[coefficient_order + 1]
                for index in 1:zero_order
                    basis = basis * wide_lambda +
                        binomial(zero_order, index) *
                        moments[coefficient_order + index + 1]
                end
                coefficient = convert(
                    C, coefficients[first_coefficient_index + coefficient_order],
                )
                total += _faddeeva_widen_bigfloat(coefficient) * basis * scale
                scale *= inverse_scale
            end
            total
        end,
        "shifted null transition",
    )::C
end

function _null_scaled_term(
    coefficient::C,
    basis::C,
    scale,
    k_value,
    order::Int,
) where {C<:Number}
    (iszero(coefficient) || iszero(basis)) && return zero(C)
    direct = coefficient * basis * scale
    if !_null_basis_requires_wide(scale) &&
       !_null_basis_requires_wide(direct)
        return direct
    end

    # A type-local power of a very small or large k may overflow or underflow
    # before a coefficient balances it.  Recover only that term in a bounded
    # wide workspace and convert the finished value back to the public type.
    return setprecision(BigFloat, 256) do
        wide_coefficient = _faddeeva_widen_bigfloat(coefficient)
        wide_basis = _faddeeva_widen_bigfloat(basis)
        wide_k = _faddeeva_widen_bigfloat(k_value)
        logarithmic_scale = -(BigFloat(order) / 2) * log(wide_k)
        recovered = wide_coefficient * wide_basis * exp(logarithmic_scale)
        converted = convert(C, recovered)
        _number_isfinite(converted) || throw(DomainError(
            converted,
            "null-uniform term exceeds the active numeric range at order $order",
        ))
        converted
    end
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
    order !== nothing && order < 0 &&
        throw(DomainError(order, "order must be nonnegative"))
    value_z = convert(C, complex(float(z)))
    k_value = real(zero(C)) + float(k)
    if coefficient_count == 0
        _null_validate_evaluation_controls(
            value_z, 0, switch, rtol, max_terms,
        )
        return zero(C)
    end
    for index in eachindex(coefficients)
        _null_validate_coefficient(coefficients[index], "amplitude coefficient")
    end
    requested_order = order === nothing ? coefficient_count - 1 : order
    maximum_order = Int(min(requested_order, coefficient_count - 1))
    maximum_order <= _NULL_MOMENT_MAX_ORDER || throw(DomainError(
        maximum_order,
        "retained amplitude order exceeds $_NULL_MOMENT_MAX_ORDER",
    ))

    R, relative_tolerance = _null_validate_evaluation_controls(
        value_z, maximum_order, switch, rtol, max_terms,
    )
    if _number_contains_ad(value_z) || _number_contains_ad(k_value) ||
       any(index -> _number_contains_ad(convert(C, coefficients[index])),
           eachindex(coefficients))
        return _null_uniform_transition_wide(
            C, value_z, k_value, coefficients, maximum_order,
        )
    end
    moments = faddeeva_moments(
        value_z, maximum_order; switch=switch, rtol=rtol, max_terms=max_terms,
    )
    condition_threshold = min(
        R(0.5), max(R(0.125), 16eps(R) / relative_tolerance),
    )
    first_coefficient_index = firstindex(coefficients)
    @inbounds for moment_order in 0:maximum_order
        coefficient = convert(
            C, coefficients[first_coefficient_index + moment_order],
        )
        if !iszero(coefficient) &&
           _null_basis_requires_wide(moments[moment_order + 1])
            return _null_uniform_transition_wide(
                C, value_z, k_value, coefficients, maximum_order,
            )
        end
    end
    inverse_scale = inv(sqrt(k_value))
    scale = one(k_value)
    total = zero(C)
    compensation = zero(C)
    absolute_sum = zero(R)
    @inbounds for moment_order in 0:maximum_order
        coefficient = convert(C, coefficients[first_coefficient_index + moment_order])
        if !iszero(coefficient)
            term = try
                _null_scaled_term(
                    coefficient, moments[moment_order + 1], scale, k_value,
                    moment_order,
                )
            catch error
                error isa DomainError || rethrow()
                return _null_uniform_transition_wide(
                    C, value_z, k_value, coefficients, maximum_order,
                )
            end
            total, compensation = _multipole_compensated_add(
                total, compensation, term,
            )
            absolute_sum += convert(R, float(_primal_value(abs(term))))
        end
        scale *= inverse_scale
    end
    total_magnitude = convert(R, float(_primal_value(abs(total))))
    if !_number_isfinite(total) ||
       (!iszero(absolute_sum) &&
        (!isfinite(absolute_sum) ||
         total_magnitude <= condition_threshold * absolute_sum))
        return _null_uniform_transition_wide(
            C, value_z, k_value, coefficients, maximum_order,
        )
    end
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

@inline function _null_shifted_basis_with_bound(
    moments,
    lambda,
    zero_order::Int,
    offset::Int,
    ::Type{R},
) where {R<:AbstractFloat}
    basis = moments[offset + 1]
    bound = convert(R, float(_primal_value(abs(basis))))
    lambda_magnitude = convert(R, float(_primal_value(abs(lambda))))
    @inbounds for index in 1:zero_order
        addition = binomial(zero_order, index) * moments[offset + index + 1]
        basis = basis * lambda + addition
        bound = bound * lambda_magnitude +
                convert(R, float(_primal_value(abs(addition))))
    end
    return basis, bound
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
    order !== nothing && order < 0 &&
        throw(DomainError(order, "order must be nonnegative"))
    value_z = convert(C, complex(float(z)))
    lambda = convert(C, complex(float(Lambda)))
    _null_validate_coefficient(lambda, "Lambda")
    k_value = real(zero(C)) + float(k)
    if coefficient_count == 0
        _null_validate_evaluation_controls(
            value_z, 0, switch, rtol, max_terms,
        )
        return zero(C)
    end
    for index in eachindex(coefficients)
        _null_validate_coefficient(coefficients[index], "amplitude coefficient")
    end
    requested_order = order === nothing ? coefficient_count - 1 : order
    maximum_coefficient_order = Int(min(requested_order, coefficient_count - 1))
    maximum_moment_order = maximum_coefficient_order + Int(zero_order)
    maximum_moment_order <= _NULL_MOMENT_MAX_ORDER || throw(DomainError(
        maximum_moment_order,
        "shifted hierarchy requires moments above $_NULL_MOMENT_MAX_ORDER",
    ))

    R, relative_tolerance = _null_validate_evaluation_controls(
        value_z, maximum_moment_order, switch, rtol, max_terms,
    )
    if _number_contains_ad(value_z) || _number_contains_ad(lambda) ||
       _number_contains_ad(k_value) ||
       any(index -> _number_contains_ad(convert(C, coefficients[index])),
           eachindex(coefficients))
        return _shifted_null_transition_wide(
            C, value_z, k_value, lambda, Int(zero_order), coefficients,
            maximum_coefficient_order,
        )
    end
    moments = faddeeva_moments(
        value_z, maximum_moment_order;
        switch=switch, rtol=rtol, max_terms=max_terms,
    )
    condition_threshold = min(
        R(0.5), max(R(0.125), 16eps(R) / relative_tolerance),
    )
    inverse_scale = inv(sqrt(k_value))
    scale = inverse_scale^Int(zero_order)
    total = zero(C)
    compensation = zero(C)
    absolute_sum = zero(R)
    first_coefficient_index = firstindex(coefficients)
    @inbounds for coefficient_order in 0:maximum_coefficient_order
        coefficient = convert(
            C, coefficients[first_coefficient_index + coefficient_order],
        )
        if !iszero(coefficient)
            basis, basis_bound = _null_shifted_basis_with_bound(
                moments, lambda, Int(zero_order), coefficient_order, R,
            )
            basis_magnitude = convert(R, float(_primal_value(abs(basis))))
            if _null_basis_requires_wide(basis) ||
               !isfinite(basis_bound) ||
               (!iszero(basis_bound) &&
                basis_magnitude <= condition_threshold * basis_bound)
                return _shifted_null_transition_wide(
                    C, value_z, k_value, lambda, Int(zero_order), coefficients,
                    maximum_coefficient_order,
                )
            end
            term = try
                _null_scaled_term(
                    coefficient, basis, scale, k_value,
                    Int(zero_order) + coefficient_order,
                )
            catch error
                error isa DomainError || rethrow()
                return _shifted_null_transition_wide(
                    C, value_z, k_value, lambda, Int(zero_order), coefficients,
                    maximum_coefficient_order,
                )
            end
            total, compensation = _multipole_compensated_add(
                total, compensation, term,
            )
            absolute_sum += convert(R, float(_primal_value(abs(term))))
        end
        scale *= inverse_scale
    end
    total_magnitude = convert(R, float(_primal_value(abs(total))))
    if !_number_isfinite(total) ||
       (!iszero(absolute_sum) &&
        (!isfinite(absolute_sum) ||
         total_magnitude <= condition_threshold * absolute_sum))
        return _shifted_null_transition_wide(
            C, value_z, k_value, lambda, Int(zero_order), coefficients,
            maximum_coefficient_order,
        )
    end
    return total
end
