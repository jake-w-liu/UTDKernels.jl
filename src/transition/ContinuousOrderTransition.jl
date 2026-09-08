"""
Continuous-order saddle--endpoint canonical functions.

Production evaluation uses normalized log-domain half-line quadrature. The
hypergeometric, arbitrary-precision, separated-asymptotic, and perturbed-phase
representations remain independent paper/test oracles.
"""

using QuadGK: quadgk
using SpecialFunctions: digamma, erfc, loggamma

const _CONTINUOUS_DEFAULT_RTOL = 2e-13
const _CONTINUOUS_DEFAULT_ATOL = 0.0
const _CONTINUOUS_DEFAULT_MAXEVALS = 1_000_000
const _CONTINUOUS_MAXEVALS = 10_000_000
const _CONTINUOUS_MOMENT_MAX_ORDER = 64
const _CONTINUOUS_UTD_MAX_ORDER = 4096
const _CONTINUOUS_RANGE_MARGIN = 16
const _CONTINUOUS_MAX_HIERARCHY_RECOVERY_BITS = 4096

@inline function _continuous_plain_real_type(values::Real...)
    primal_type = promote_type(
        (typeof(float(_primal_value(value))) for value in values)...,
    )
    input_type = promote_type((typeof(float(value)) for value in values)...)
    input_type === primal_type || throw(ArgumentError(
        "automatic differentiation of continuous order or physical scales is " *
        "not supported; differentiate the canonical coordinate or use the " *
        "order-sensitivity identity",
    ))
    primal_type === BigFloat && throw(ArgumentError(
        "continuous-order production APIs do not expose a BigFloat path; " *
        "use an independent high-precision oracle",
    ))
    primal_type === Float16 && return Float32
    primal_type in (Float32, Float64) || throw(ArgumentError(
        "continuous-order production APIs support binary16, binary32, and " *
        "binary64 inputs",
    ))
    return primal_type
end

@inline function _continuous_transition_values(mu::Real, zeta::Number)
    _validate_finite_number(zeta, "continuous-order coordinate zeta")
    R = _continuous_plain_real_type(
        mu, _primal_value(real(zeta)), _primal_value(imag(zeta)),
    )
    scalar_type = promote_type(
        R, typeof(float(real(zeta))), typeof(float(imag(zeta))),
    )
    isconcretetype(scalar_type) || throw(ArgumentError(
        "continuous-order coordinates require a concrete numeric type",
    ))
    order = convert(R, float(mu))
    order > zero(R) && isfinite(order) || throw(DomainError(
        mu, "continuous order mu must be finite and positive",
    ))
    coordinate = complex(
        convert(scalar_type, float(real(zeta))),
        convert(scalar_type, float(imag(zeta))),
    )
    _number_isfinite(coordinate) || throw(DomainError(
        zeta, "continuous-order coordinate zeta is outside the active range",
    ))
    return order, coordinate, R
end

@inline function _continuous_log_normalizer(order::R, context::AbstractString) where {R}
    value = loggamma(order / 2)
    limit = log(floatmax(R)) - R(_CONTINUOUS_RANGE_MARGIN)
    (isfinite(value) && value <= limit) || throw(DomainError(
        order,
        "$context exceeds the type-local direct-normalization range",
    ))
    return value
end

@inline function _continuous_tolerances(
    rtol::Union{Nothing,Real},
    atol::Union{Nothing,Real},
    ::Type{R},
) where {R<:AbstractFloat}
    if rtol !== nothing
        _number_isfinite(rtol) || throw(DomainError(
            rtol, "continuous-order rtol must be finite and positive",
        ))
        _number_contains_ad(rtol) && throw(ArgumentError(
            "continuous-order rtol does not accept automatic-differentiation " *
            "inputs",
        ))
    end
    if atol !== nothing
        _number_isfinite(atol) || throw(DomainError(
            atol, "continuous-order atol must be finite and nonnegative",
        ))
        _number_contains_ad(atol) && throw(ArgumentError(
            "continuous-order atol does not accept automatic-differentiation " *
            "inputs",
        ))
    end
    relative = rtol === nothing ?
               max(R(_CONTINUOUS_DEFAULT_RTOL), 128eps(R)) :
               convert(R, _primal_value(rtol))
    absolute = atol === nothing ? R(_CONTINUOUS_DEFAULT_ATOL) :
               convert(R, _primal_value(atol))
    (isfinite(relative) && relative > zero(R)) || throw(DomainError(
        rtol, "continuous-order rtol must be finite and positive",
    ))
    (isfinite(absolute) && absolute >= zero(R)) || throw(DomainError(
        atol, "continuous-order atol must be finite and nonnegative",
    ))
    return relative, absolute
end

@inline function _continuous_maxevals(maxevals::Integer)
    value = try
        Int(maxevals)
    catch error
        (error isa InexactError || error isa OverflowError) || rethrow()
        throw(ArgumentError("continuous-order maxevals is not representable as Int"))
    end
    63 <= value <= _CONTINUOUS_MAXEVALS || throw(ArgumentError(
        "continuous-order maxevals must lie in 63:$_CONTINUOUS_MAXEVALS",
    ))
    return value
end

@inline function _continuous_scalar_quad_norm(value::Real)
    result = float(abs(_primal_value(value)))
    if hasproperty(value, :partials)
        if hasproperty(value, :value)
            result = max(
                result,
                _continuous_scalar_quad_norm(getproperty(value, :value)),
            )
        end
        for partial in getproperty(value, :partials)
            result = max(result, _continuous_scalar_quad_norm(partial))
        end
    end
    return result
end

@inline _continuous_quad_norm(value::Real) =
    _continuous_scalar_quad_norm(value)
@inline _continuous_quad_norm(value::Complex) = max(
    _continuous_scalar_quad_norm(real(value)),
    _continuous_scalar_quad_norm(imag(value)),
)

@inline _continuous_ad_depth(value::Real) = hasproperty(value, :value) ?
    1 + _continuous_ad_depth(getproperty(value, :value)) : 0

@inline _continuous_ad_depth(value::Complex) = max(
    _continuous_ad_depth(real(value)), _continuous_ad_depth(imag(value)),
)

@inline _continuous_nested_ad(value::Real) = _continuous_ad_depth(value) >= 2

@inline _continuous_nested_ad(value::Complex) =
    _continuous_ad_depth(value) >= 2

@inline function _continuous_scalar_allfinite(value::Real)
    isfinite(_primal_value(value)) || return false
    if hasproperty(value, :partials)
        hasproperty(value, :value) &&
            !_continuous_scalar_allfinite(getproperty(value, :value)) &&
            return false
        for partial in getproperty(value, :partials)
            _continuous_scalar_allfinite(partial) || return false
        end
    end
    return true
end

@inline _continuous_allfinite(value::Real) =
    _continuous_scalar_allfinite(value)
@inline _continuous_allfinite(value::Complex) =
    _continuous_scalar_allfinite(real(value)) &&
    _continuous_scalar_allfinite(imag(value))

@inline function _continuous_tail_mode(order::R, real_coordinate::R) where {R}
    if order >= one(R)
        radial = hypot(real_coordinate, sqrt(2max(zero(R), order - one(R))))
        if real_coordinate >= zero(R)
            return (real_coordinate + radial) / 2
        end
        denominator = radial - real_coordinate
        return iszero(denominator) ? zero(R) :
               (order - one(R)) / denominator
    end
    real_coordinate > zero(R) || return zero(R)
    discriminant = muladd(real_coordinate, real_coordinate, 2(order - one(R)))
    discriminant > zero(R) || return zero(R)
    return (real_coordinate + sqrt(discriminant)) / 2
end

@inline function _continuous_underflow(log_magnitude, ::Type{R}) where {R}
    primal = _primal_value(log_magnitude)
    return primal < log(nextfloat(zero(R))) - R(4)
end

function _continuous_exp_phase_product(
    log_magnitude::Number,
    phase::Number,
    weight::Number,
    ::Type{V},
    ::Type{R},
)::V where {V<:Number,R<:AbstractFloat}
    if _continuous_underflow(log_magnitude, R)
        if !_number_contains_ad(log_magnitude) &&
           !_number_contains_ad(phase) && !_number_contains_ad(weight)
            return zero(V)
        end
        return setprecision(BigFloat, 256) do
            converted::V = convert(
                V,
                exp(_faddeeva_widen_bigfloat(log_magnitude)) *
                cis(_faddeeva_widen_bigfloat(phase)) *
                _faddeeva_widen_bigfloat(weight),
            )
            _number_isfinite(converted) || throw(DomainError(
                converted,
                "continuous-order integrand exceeds the active range",
            ))
            converted
        end
    end
    value = exp(log_magnitude) * cis(phase) * weight
    _number_isfinite(value) || throw(DomainError(
        value, "continuous-order integrand exceeds the active range",
    ))
    return convert(V, value)
end

function _continuous_exp_expm1_product(
    log_magnitude::Number,
    exponent_delta::Number,
    ::Type{V},
    ::Type{R},
)::V where {V<:Number,R<:AbstractFloat}
    if _continuous_underflow(log_magnitude, R)
        if !_number_contains_ad(log_magnitude) &&
           !_number_contains_ad(exponent_delta)
            return zero(V)
        end
        return setprecision(BigFloat, 256) do
            wide_delta = _faddeeva_widen_bigfloat(exponent_delta)
            converted::V = convert(
                V,
                exp(_faddeeva_widen_bigfloat(log_magnitude)) *
                (exp(wide_delta) - one(wide_delta)),
            )
            _number_isfinite(converted) || throw(DomainError(
                converted,
                "continuous-order correction integrand exceeds the active range",
            ))
            converted
        end
    end
    value = exp(log_magnitude) * _continuous_expm1(exponent_delta, R)
    _number_isfinite(value) || throw(DomainError(
        value,
        "continuous-order correction integrand exceeds the active range",
    ))
    return convert(V, value)
end

function _continuous_expm1(value::Number, ::Type{R}) where {R<:AbstractFloat}
    magnitude = convert(R, float(_primal_value(abs(value))))
    magnitude > R(0.5) && return exp(value) - one(value)
    term = value
    total = value
    @inbounds for order in 2:64
        term *= value / order
        total += term
        term_magnitude = convert(R, float(_primal_value(abs(term))))
        total_magnitude = convert(R, float(_primal_value(abs(total))))
        term_magnitude <= 4eps(R) * max(one(R), total_magnitude) && return total
    end
    return total
end

@inline function _continuous_expected_magnitude(
    order::R,
    real_coordinate::R,
    real_shift::R,
    logarithmic_weight::Bool,
    centered_real::Val{C}=Val(false),
) where {R<:AbstractFloat,C}
    log_estimate = if real_coordinate <= -one(R)
        loggamma((order + one(R)) / 2) - log(sqrt(_typed_pi(zero(R)))) -
        order * log(-real_coordinate) + real_shift
    elseif real_coordinate >= one(R)
        log(R(2) * sqrt(_typed_pi(zero(R)))) - loggamma(order / 2) +
        (order - one(R)) * log(real_coordinate) +
        (C ? zero(R) : real_coordinate^2 + real_shift)
    else
        real_shift
    end
    if logarithmic_weight
        log_estimate += log(max(
            one(R), abs(digamma(order / 2)) + log1p(abs(real_coordinate)),
        ))
    end
    log_estimate >= log(floatmax(R)) && return floatmax(R)
    log_estimate <= log(nextfloat(zero(R))) && return nextfloat(zero(R))
    return exp(log_estimate)
end

@inline function _continuous_real_exponent(
    u,
    coordinate,
    ::Val{false},
)
    return -u^2 + 2 * (real(coordinate) * u)
end

@inline function _continuous_real_exponent(
    u,
    coordinate,
    ::Val{true},
)
    return -(u - real(coordinate))^2
end

function _continuous_rescale_value(
    value::V,
    log_scale::R,
    context::AbstractString,
)::V where {V<:Number,R<:AbstractFloat}
    iszero(log_scale) && return value
    lower_direct_limit = log(floatmin(R)) + R(8)
    upper_direct_limit = log(floatmax(R)) - R(8)
    if lower_direct_limit <= log_scale <= upper_direct_limit
        result = value * exp(log_scale)
        _number_isfinite(result) || throw(DomainError(
            result, "$context exceeds the active numeric range",
        ))
        return result
    end
    iszero(value) && return zero(value)
    work_precision = max(192, precision(R) + 64)
    result = setprecision(BigFloat, work_precision) do
        wide_value = _faddeeva_widen_bigfloat(value)
        convert(V, wide_value * exp(BigFloat(log_scale)))
    end
    _number_isfinite(result) || throw(DomainError(
        result, "$context exceeds the active numeric range",
    ))
    return result
end

@inline function _continuous_rescale_error(error::R, log_scale::R) where {R}
    iszero(error) && return zero(R)
    log_error = log(error) + log_scale
    log_error > log(floatmax(R)) && return R(Inf)
    log_error < log(nextfloat(zero(R))) && return zero(R)
    return exp(log_error)
end

@inline function _continuous_unscale_tolerance(
    tolerance::R,
    log_scale::R,
) where {R<:AbstractFloat}
    iszero(log_scale) && return tolerance
    log_tolerance = log(tolerance) - log_scale
    log_tolerance <= log(nextfloat(zero(R))) && return nextfloat(zero(R))
    return exp(log_tolerance)
end

function _continuous_certify(
    value::V,
    estimated_error::R,
    rtol::R,
    atol::R,
    context::AbstractString,
)::V where {V<:Number,R<:AbstractFloat}
    _number_isfinite(value) || throw(DomainError(
        value, "$context produced a non-finite result",
    ))
    value_scale = convert(R, _continuous_quad_norm(value))
    tolerance = max(atol, rtol * value_scale)
    (isfinite(estimated_error) && estimated_error <= tolerance) ||
        throw(DomainError(
            estimated_error,
            "$context quadrature did not meet the requested tolerance",
        ))
    return value
end

function _continuous_zero_linearization(
    order::R,
    coordinate,
) where {R<:AbstractFloat}
    scalar = real(coordinate)
    if hasproperty(scalar, :value) &&
       hasproperty(getproperty(scalar, :value), :value)
        throw(ArgumentError(
            "nested coordinate differentiation outside the direct-normalization " *
            "range is not supported",
        ))
    end
    order_bits = max(0, exponent(order))
    work_precision = max(192, precision(R) + order_bits + 64)
    slope = setprecision(BigFloat, work_precision) do
        wide_order = BigFloat(order)
        convert(
            R,
            exp(log(BigFloat(2)) + loggamma((wide_order + 1) / 2) -
                loggamma(wide_order / 2)),
        )
    end
    result = one(coordinate) + slope * coordinate
    _number_isfinite(result) || throw(DomainError(
        result, "continuous-order coalescence derivative exceeds the active range",
    ))
    return result
end

function _continuous_small_order_baseline(
    order::R,
    scale::R,
    exponent_at_zero::Real,
    near_shift::R,
    real_shift::Real,
    ::Val{C},
    ::Type{V},
)::V where {R<:AbstractFloat,C,V<:Number}
    work_precision = max(192, 4precision(R))
    value = setprecision(BigFloat, work_precision) do
        wide_order = BigFloat(order)
        wide_exponent = _faddeeva_widen_bigfloat(exponent_at_zero)
        wide_shift = C ? zero(wide_exponent) :
                     _faddeeva_widen_bigfloat(real_shift)
        logarithm = log(BigFloat(2)) - loggamma(wide_order / 2) -
                    wide_order * log(BigFloat(scale)) + wide_shift +
                    wide_exponent - BigFloat(near_shift) - log(wide_order)
        convert(V, exp(logarithm))
    end
    _number_isfinite(value) || throw(DomainError(
        value, "continuous-order endpoint baseline exceeds the active range",
    ))
    return value
end

function _continuous_integral(
    order::R,
    coordinate::Complex{T};
    rtol::R,
    atol::R,
    maxevals::Int,
    real_shift::T=zero(T),
    logarithmic_weight::Val{L}=Val(false),
    centered_real::Val{C}=Val(false),
)::Complex{T} where {R<:AbstractFloat,T<:Real,L,C}
    log_normalizer = _continuous_log_normalizer(order, "continuous order mu")
    real_coordinate = convert(R, _primal_value(real(coordinate)))
    coordinate_limit = sqrt(floatmax(R)) / R(4)
    real_coordinate <= coordinate_limit || throw(DomainError(
        coordinate, "continuous-order coordinate is outside the quadrature range",
    ))
    scale = if real_coordinate < -one(R)
        magnitude = -real_coordinate
        magnitude <= floatmax(R) / 2 ? 2magnitude : magnitude
    else
        one(R)
    end
    inverse_scale = inv(scale)
    base = log(R(2)) - log_normalizer - order * log(scale) +
           (C ? zero(real_shift) : real_shift)
    inverse_order = inv(order)
    log_order = log(order)
    digamma_term = L ? digamma(order / 2) / 2 : zero(R)

    near_upper = inverse_scale
    near_mode = clamp(real_coordinate, zero(R), near_upper)
    near_peak = convert(
        R,
        _primal_value(
            base - log_order + _continuous_real_exponent(
                near_mode, coordinate, centered_real,
            ),
        ),
    )
    near_shift = max(zero(R), near_peak)

    tail_mode = _continuous_tail_mode(order, real_coordinate)
    tail_mode_scaled = scale * tail_mode
    tail_at_one = convert(
        R,
        _primal_value(
            base + _continuous_real_exponent(
                inverse_scale, coordinate, centered_real,
            ),
        ),
    )
    tail_peak = tail_at_one
    if tail_mode_scaled > one(R) && isfinite(tail_mode_scaled)
        tail_peak = max(
            tail_peak,
            convert(
                R,
                _primal_value(
                    base + (order - one(R)) * log(tail_mode_scaled) +
                    _continuous_real_exponent(
                        tail_mode, coordinate, centered_real,
                    ),
                ),
            ),
        )
    end
    tail_shift = max(zero(R), tail_peak)

    near = function (x::R)
        iszero(x) && return zero(coordinate)
        scaled_u = exp(log(x) * inverse_order)
        u = scaled_u * inverse_scale
        log_magnitude = base - log_order +
                        _continuous_real_exponent(
                            u, coordinate, centered_real,
                        ) - near_shift
        phase = 2 * (imag(coordinate) * u)
        weight = L ?
                 log(x) * inverse_order - log(scale) - digamma_term :
                 one(log_magnitude)
        return _continuous_exp_phase_product(
            log_magnitude, phase, weight, typeof(coordinate), R,
        )
    end
    coordinate_offset = real(coordinate) - real_coordinate
    tail = function (scaled_u::R)
        isfinite(scaled_u) || return zero(coordinate)
        u = scaled_u * inverse_scale
        log_magnitude = base + (order - one(R)) * log(scaled_u) +
                        _continuous_real_exponent(
                            u, coordinate, centered_real,
                        ) - tail_shift
        phase = 2 * (imag(coordinate) * u)
        weight = L ? log(scaled_u) - log(scale) - digamma_term :
                 one(log_magnitude)
        return _continuous_exp_phase_product(
            log_magnitude, phase, weight, typeof(coordinate), R,
        )
    end
    centered_tail = function (offset::R)
        scaled_u = real_coordinate + offset
        scaled_u > zero(R) || return zero(coordinate)
        centered_difference = offset - coordinate_offset
        log_magnitude = base + (order - one(R)) * log(scaled_u) -
                        centered_difference^2 - tail_shift
        phase = 2 * (imag(coordinate) * scaled_u)
        weight = L ? log(scaled_u) - log(scale) - digamma_term :
                 one(log_magnitude)
        return _continuous_exp_phase_product(
            log_magnitude, phase, weight, typeof(coordinate), R,
        )
    end
    centered_pair = function (offset::R)
        if typeof(real(coordinate)) === R
            return centered_tail(offset) + centered_tail(-offset)
        end

        # Opposite sides of a distant saddle have large, cancelling AD
        # tangents.  Form the pair before narrowing so the small derivative is
        # not lost to the spacing of the public binary type.
        coordinate_bits = max(0, exponent(abs(real_coordinate)))
        work_precision = precision(R) + coordinate_bits + 64
        return setprecision(BigFloat, work_precision) do
            wide_coordinate = _faddeeva_widen_bigfloat(coordinate)
            wide_order = BigFloat(order)
            wide_center = BigFloat(real_coordinate)
            wide_offset = BigFloat(offset)
            wide_base = _faddeeva_widen_bigfloat(base)
            wide_shift = _faddeeva_widen_bigfloat(tail_shift)
            wide_scale = _faddeeva_widen_bigfloat(scale)
            wide_digamma = _faddeeva_widen_bigfloat(digamma_term)
            coordinate_delta = real(wide_coordinate) - wide_center
            evaluate_side = function (signed_offset)
                wide_u = wide_center + signed_offset
                log_magnitude = wide_base + (wide_order - 1) * log(wide_u) -
                                (signed_offset - coordinate_delta)^2 - wide_shift
                integrand_value = exp(log_magnitude) *
                                  cis(2 * (imag(wide_coordinate) * wide_u))
                L && (integrand_value *=
                    log(wide_u) - log(wide_scale) - wide_digamma)
                return integrand_value
            end
            convert(
                typeof(coordinate),
                evaluate_side(wide_offset) + evaluate_side(-wide_offset),
            )
        end
    end

    expected_magnitude = _continuous_expected_magnitude(
        order, real_coordinate, convert(R, _primal_value(real_shift)), L,
        centered_real,
    )
    local_rtol = max(rtol / (L ? R(32) : R(16)), 8eps(R))
    final_piece_atol = max(
        atol / R(4),
        rtol * expected_magnitude / (L ? R(64) : R(32)),
        R(8) * nextfloat(zero(R)),
    )
    near_atol = _continuous_unscale_tolerance(final_piece_atol, near_shift)
    tail_atol = _continuous_unscale_tolerance(final_piece_atol, tail_shift)
    quadrature_order = L ? 31 : (real_coordinate <= R(-8) ? 7 : 15)
    if !L && order < R(0.01)
        zero_u = zero(R)
        exponent_at_zero = _continuous_real_exponent(
            zero_u, coordinate, centered_real,
        )
        baseline = _continuous_small_order_baseline(
            order, scale, exponent_at_zero, near_shift, real_shift,
            centered_real, typeof(coordinate),
        )
        correction = function (scaled_u::R)
            iszero(scaled_u) && return zero(coordinate)
            u = scaled_u * inverse_scale
            real_exponent = _continuous_real_exponent(
                u, coordinate, centered_real,
            )
            exponent_delta = real_exponent - exponent_at_zero +
                             2im * (imag(coordinate) * u)
            log_magnitude = base + (order - one(R)) * log(scaled_u) +
                            exponent_at_zero - near_shift
            return _continuous_exp_expm1_product(
                log_magnitude, exponent_delta, typeof(coordinate), R,
            )
        end
        correction_value, correction_error = quadgk(
            correction, zero(R), one(R);
            rtol=local_rtol, atol=near_atol, maxevals=maxevals,
            norm=_continuous_quad_norm, order=quadrature_order,
        )
        near_value = baseline + correction_value
        near_error = correction_error + 8eps(R) *
                     convert(R, float(_primal_value(abs(baseline))))
    else
        near_value, near_error = quadgk(
            near, zero(R), one(R);
            rtol=local_rtol, atol=near_atol, maxevals=maxevals,
            norm=_continuous_quad_norm, order=quadrature_order,
        )
    end
    if tail_mode_scaled > one(R) && isfinite(tail_mode_scaled)
        if C && real_coordinate > zero(R)
            focus_radius = scale * sqrt(max(R(64), -log(rtol) + R(16)))
            focus_left = real_coordinate - focus_radius
            focus_right = real_coordinate + focus_radius
            (isfinite(focus_right) && focus_right > tail_mode_scaled) ||
                throw(DomainError(
                coordinate,
                "scaled continuous-order saddle cannot be resolved in the " *
                "active numeric precision",
            ))
            if focus_left > one(R)
                focus_left < tail_mode_scaled || throw(DomainError(
                    coordinate,
                    "scaled continuous-order saddle cannot be resolved in " *
                    "the active numeric precision",
                ))
                far_left, far_left_error = quadgk(
                    tail, one(R), focus_left;
                    rtol=local_rtol, atol=tail_atol, maxevals=maxevals,
                    norm=_continuous_quad_norm, order=quadrature_order,
                )
                paired, paired_error = quadgk(
                    centered_pair, zero(R), focus_radius;
                    rtol=local_rtol, atol=tail_atol, maxevals=maxevals,
                    norm=_continuous_quad_norm, order=quadrature_order,
                )
                left = paired
                left_error = paired_error
                right = zero(coordinate)
                right_error = zero(R)
            else
                far_left = zero(coordinate)
                far_left_error = zero(R)
                if real_coordinate > one(R)
                    left, left_error = quadgk(
                        centered_tail, one(R) - real_coordinate, zero(R);
                        rtol=local_rtol, atol=tail_atol, maxevals=maxevals,
                        norm=_continuous_quad_norm, order=quadrature_order,
                    )
                else
                    left = zero(coordinate)
                    left_error = zero(R)
                end
            end
            if focus_left <= one(R)
                right_lower = real_coordinate > one(R) ?
                              zero(R) : one(R) - real_coordinate
                right, right_error = quadgk(
                    centered_tail, right_lower, focus_right - real_coordinate;
                    rtol=local_rtol, atol=tail_atol, maxevals=maxevals,
                    norm=_continuous_quad_norm, order=quadrature_order,
                )
            end
            far_right, far_right_error = quadgk(
                tail, focus_right, R(Inf);
                rtol=local_rtol, atol=tail_atol, maxevals=maxevals,
                norm=_continuous_quad_norm, order=quadrature_order,
            )
            tail_value = far_left + left + right + far_right
            tail_error = far_left_error + left_error + right_error +
                         far_right_error
        else
            tail_value, tail_error = quadgk(
                tail, one(R), tail_mode_scaled, R(Inf);
                rtol=local_rtol, atol=tail_atol, maxevals=maxevals,
                norm=_continuous_quad_norm, order=quadrature_order,
            )
        end
    else
        tail_value, tail_error = quadgk(
            tail, one(R), R(Inf);
            rtol=local_rtol, atol=tail_atol, maxevals=maxevals,
            norm=_continuous_quad_norm, order=quadrature_order,
        )
    end
    near_result = _continuous_rescale_value(
        near_value, near_shift, "continuous-order near-endpoint integral",
    )
    tail_result = _continuous_rescale_value(
        tail_value, tail_shift, "continuous-order tail integral",
    )
    combined_value = near_result + tail_result
    estimated_error = _continuous_rescale_error(near_error, near_shift) +
                      _continuous_rescale_error(tail_error, tail_shift)
    context = L ?
              "continuous-order parameter derivative" :
              "continuous-order transition"
    return _continuous_certify(
        combined_value, estimated_error, rtol, atol, context,
    )
end

function _continuous_integral_wide_ad(
    order::R,
    coordinate::C,
    maxevals::Int;
    scaled::Bool=false,
    logarithmic_weight::Bool=false,
) where {R<:AbstractFloat,C<:Number}
    real_primal = _primal_value(real(coordinate))
    imag_primal = _primal_value(imag(coordinate))
    coordinate_bits = max(
        iszero(real_primal) ? 0 : max(0, exponent(abs(real_primal))),
        iszero(imag_primal) ? 0 : max(0, exponent(abs(imag_primal))),
    )
    base_precision = max(
        256,
        precision(R) + 2coordinate_bits + max(0, exponent(order)) + 192,
    )
    base_precision <= _CONTINUOUS_MAX_HIERARCHY_RECOVERY_BITS ||
        throw(ArgumentError(
            "continuous-order AD recovery requires $base_precision bits, " *
            "above the bounded $_CONTINUOUS_MAX_HIERARCHY_RECOVERY_BITS-bit " *
            "workspace",
        ))
    work_precisions = (
        base_precision,
        min(2base_precision, _CONTINUOUS_MAX_HIERARCHY_RECOVERY_BITS),
        _CONTINUOUS_MAX_HIERARCHY_RECOVERY_BITS,
    )
    previous = nothing
    @inbounds for work_precision in unique(work_precisions)
        wide_result = setprecision(BigFloat, work_precision) do
            wide_order = BigFloat(order)
            wide_coordinate = _faddeeva_widen_bigfloat(coordinate)
            quadrature_bits = min(140, 2precision(R) + 34)
            quadrature_tolerance = BigFloat(2)^(-quadrature_bits)
            if scaled
                _continuous_integral(
                    wide_order, wide_coordinate;
                    rtol=quadrature_tolerance,
                    atol=zero(BigFloat),
                    maxevals=maxevals,
                    real_shift=-(real(wide_coordinate)^2),
                    centered_real=Val(true),
                )
            elseif logarithmic_weight
                _continuous_integral(
                    wide_order, wide_coordinate;
                    rtol=quadrature_tolerance,
                    atol=zero(BigFloat),
                    maxevals=maxevals,
                    logarithmic_weight=Val(true),
                )
            else
                _continuous_integral(
                    wide_order, wide_coordinate;
                    rtol=quadrature_tolerance,
                    atol=zero(BigFloat),
                    maxevals=maxevals,
                )
            end
        end
        converted::C = convert(C, wide_result)
        _number_isfinite(converted) || throw(DomainError(
            converted,
            "continuous-order AD recovery exceeds the active range",
        ))
        previous !== nothing && isequal(converted, previous) &&
            return converted
        previous = converted
    end
    throw(ArgumentError(
        "continuous-order AD recovery did not stabilize within the bounded " *
        "$_CONTINUOUS_MAX_HIERARCHY_RECOVERY_BITS-bit workspace",
    ))
end

function _continuous_exact_unit_real_wide(
    coordinate::T,
    ::Type{R};
    scaled::Bool,
)::T where {T<:Real,R<:AbstractFloat}
    coordinate_primal = _primal_value(coordinate)
    coordinate_bits = iszero(coordinate_primal) ? 0 :
                      max(0, exponent(abs(coordinate_primal)))
    differentiation_depth = max(1, _continuous_ad_depth(coordinate))
    base_precision = max(
        192,
        precision(R) + 2differentiation_depth * coordinate_bits + 128,
    )
    base_precision <= _CONTINUOUS_MAX_HIERARCHY_RECOVERY_BITS ||
        throw(ArgumentError(
            "exact unit-order AD recovery requires $base_precision bits, " *
            "above the bounded $_CONTINUOUS_MAX_HIERARCHY_RECOVERY_BITS-bit " *
            "workspace",
        ))
    work_precisions = (
        base_precision,
        min(2base_precision, _CONTINUOUS_MAX_HIERARCHY_RECOVERY_BITS),
        _CONTINUOUS_MAX_HIERARCHY_RECOVERY_BITS,
    )
    previous = nothing
    last_result = nothing
    @inbounds for work_precision in unique(work_precisions)
        wide_result = setprecision(BigFloat, work_precision) do
            wide_coordinate = _faddeeva_widen_bigfloat(coordinate)
            scaled ? erfc(-wide_coordinate) :
                     _faddeeva_erfcx(-wide_coordinate)
        end
        converted::T = convert(T, wide_result)
        last_result = converted
        if _number_isfinite(converted)
            previous !== nothing && isequal(converted, previous) &&
                return converted
            previous = converted
        end
    end
    context = scaled ? "scaled continuous-order transition" :
                       "unscaled continuous-order transition"
    last_result !== nothing && !_number_isfinite(last_result) &&
        throw(DomainError(
            last_result, "$context exceeds the active numeric range",
        ))
    throw(ArgumentError(
        "$context AD recovery did not stabilize within the bounded " *
        "$_CONTINUOUS_MAX_HIERARCHY_RECOVERY_BITS-bit workspace",
    ))
end

function _continuous_exact_order_two_scaled_real_wide(
    coordinate::T,
    ::Type{R},
)::T where {T<:Real,R<:AbstractFloat}
    coordinate_primal = _primal_value(coordinate)
    coordinate_bits = iszero(coordinate_primal) ? 0 :
                      max(0, exponent(abs(coordinate_primal)))
    differentiation_depth = max(1, _continuous_ad_depth(coordinate))
    base_precision = max(
        192,
        precision(R) + 2differentiation_depth * coordinate_bits + 128,
    )
    base_precision <= _CONTINUOUS_MAX_HIERARCHY_RECOVERY_BITS ||
        throw(ArgumentError(
            "exact order-two scaled AD recovery requires $base_precision " *
            "bits, above the bounded " *
            "$_CONTINUOUS_MAX_HIERARCHY_RECOVERY_BITS-bit workspace",
        ))
    work_precisions = (
        base_precision,
        min(2base_precision, _CONTINUOUS_MAX_HIERARCHY_RECOVERY_BITS),
        _CONTINUOUS_MAX_HIERARCHY_RECOVERY_BITS,
    )
    previous = nothing
    last_result = nothing
    @inbounds for work_precision in unique(work_precisions)
        wide_result = setprecision(BigFloat, work_precision) do
            wide_coordinate = _faddeeva_widen_bigfloat(coordinate)
            exp(-(wide_coordinate * wide_coordinate)) +
            sqrt(BigFloat(pi)) * wide_coordinate * erfc(-wide_coordinate)
        end
        converted::T = convert(T, wide_result)
        last_result = converted
        if _number_isfinite(converted)
            previous !== nothing && isequal(converted, previous) &&
                return converted
            previous = converted
        end
    end
    last_result !== nothing && !_number_isfinite(last_result) &&
        throw(DomainError(
            last_result,
            "scaled continuous-order transition exceeds the active " *
            "numeric range",
        ))
    throw(ArgumentError(
        "scaled continuous-order transition AD recovery did not stabilize " *
        "within the bounded " *
        "$_CONTINUOUS_MAX_HIERARCHY_RECOVERY_BITS-bit workspace",
    ))
end

function _continuous_order_two_transition(
    coordinate::C,
    ::Type{R},
) where {C<:Number,R<:AbstractFloat}
    real_primal = _primal_value(real(coordinate))
    imag_primal = _primal_value(imag(coordinate))
    coordinate_bits = max(
        iszero(real_primal) ? 0 : max(0, exponent(abs(real_primal))),
        iszero(imag_primal) ? 0 : max(0, exponent(abs(imag_primal))),
    )
    radius_large = hypot(BigFloat(real_primal), BigFloat(imag_primal)) > 8
    value = if radius_large || _number_contains_ad(coordinate)
        work_precision = precision(R) + 2coordinate_bits + 192
        setprecision(BigFloat, work_precision) do
            wide_coordinate = _faddeeva_widen_bigfloat(coordinate)
            wide_value = one(wide_coordinate) +
                sqrt(BigFloat(pi)) * wide_coordinate *
                _null_wide_faddeeva(-im * wide_coordinate)
            convert(C, wide_value)
        end
    else
        scalar = zero(real(coordinate)) + _typed_pi(real(coordinate))
        one(coordinate) + sqrt(scalar) * coordinate *
        _faddeeva_w(-im * coordinate)
    end
    _number_isfinite(value) || throw(DomainError(
        value,
        "unscaled continuous-order transition exceeds the active range; " *
        "use scaled_continuous_order_transition on the real saddle branch",
    ))
    return value
end

"""
    continuous_order_transition(mu, zeta;
                                rtol=nothing, atol=nothing,
                                maxevals=1_000_000)

Evaluate
`2/Gamma(mu/2) * integral(u^(mu-1)*exp(-u^2+2*zeta*u), u=0..Inf)`
for finite positive `mu` and a finite binary real or complex canonical
coordinate. The first interval uses a power substitution, the tail is split at
its analytic mode, and the combined adaptive-quadrature error estimate must
meet the requested tolerance. Exact unit order shares the package Faddeeva
backend; exact coalescence returns one.

Binary16 inputs are widened to binary32. Binary32 and binary64 inputs preserve
their promoted complex type. First coordinate derivatives are supported on the
general quadrature path; nested coordinate differentiation is limited to exact
closed members. Differentiation with respect to `mu` is available through the
paper's explicit order-sensitivity identity. BigFloat is reserved for
independent oracles.
"""
function continuous_order_transition(
    mu::Real,
    zeta::Number;
    rtol::Union{Nothing,Real}=nothing,
    atol::Union{Nothing,Real}=nothing,
    maxevals::Integer=_CONTINUOUS_DEFAULT_MAXEVALS,
)
    order, coordinate, R = _continuous_transition_values(mu, zeta)
    relative_tolerance, absolute_tolerance =
        _continuous_tolerances(rtol, atol, R)
    evaluation_limit = _continuous_maxevals(maxevals)
    C = typeof(coordinate)
    if iszero(_primal_value(abs(coordinate))) &&
       typeof(real(coordinate)) === R
        return one(C)
    end
    if order == one(R)
        if _number_contains_ad(coordinate) && iszero(imag(coordinate))
            exact_real = _continuous_exact_unit_real_wide(
                real(coordinate), R; scaled=false,
            )
            return convert(C, exact_real)
        end
        value = convert(C, _faddeeva_w(-im * coordinate))
        _number_isfinite(value) || throw(DomainError(
            zeta,
            "unscaled continuous-order transition exceeds the active range; " *
            "use scaled_continuous_order_transition on the real saddle branch",
        ))
        return value
    end
    order == R(2) && return _continuous_order_two_transition(coordinate, R)
    if iszero(_primal_value(abs(coordinate)))
        return _continuous_zero_linearization(order, coordinate)
    end
    _continuous_log_normalizer(order, "continuous order mu")
    _continuous_nested_ad(coordinate) && throw(ArgumentError(
        "nested coordinate differentiation is supported only by exact " *
        "continuous-order members",
    ))
    if _number_contains_ad(coordinate) &&
       _continuous_quad_norm(coordinate) > sqrt(floatmax(R))
        return _continuous_integral_wide_ad(
            order, coordinate, evaluation_limit,
        )
    end
    return _continuous_integral(
        order, coordinate;
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        maxevals=evaluation_limit,
    )
end

function continuous_order_transition(
    mu::Real,
    zeta::AbstractArray{<:Number};
    rtol::Union{Nothing,Real}=nothing,
    atol::Union{Nothing,Real}=nothing,
    maxevals::Integer=_CONTINUOUS_DEFAULT_MAXEVALS,
)
    coordinate_type = eltype(zeta)
    isconcretetype(coordinate_type) || throw(ArgumentError(
        "continuous-order coordinate arrays require a concrete element type",
    ))
    _, _, R = _continuous_transition_values(mu, zero(coordinate_type))
    _continuous_tolerances(rtol, atol, R)
    _continuous_maxevals(maxevals)
    return map(
        value -> continuous_order_transition(
            mu, value; rtol, atol, maxevals,
        ),
        zeta,
    )
end

"""
    scaled_continuous_order_transition(mu, zeta; kwargs...)

Evaluate `exp(-zeta^2) * continuous_order_transition(mu, zeta)` directly for a
finite real saddle coordinate. The exponential shift stays inside the
log-domain integrand, and a positive saddle is integrated in separately
resolved Gaussian neighborhoods, so large positive coordinates do not form
the overflowing unscaled value or a falsely vanishing unresolved interval.
Exact unit order reduces to `erfc(-zeta)`.
"""
function scaled_continuous_order_transition(
    mu::Real,
    zeta::Real;
    rtol::Union{Nothing,Real}=nothing,
    atol::Union{Nothing,Real}=nothing,
    maxevals::Integer=_CONTINUOUS_DEFAULT_MAXEVALS,
)
    order, coordinate, R = _continuous_transition_values(mu, zeta)
    relative_tolerance, absolute_tolerance =
        _continuous_tolerances(rtol, atol, R)
    evaluation_limit = _continuous_maxevals(maxevals)
    coordinate_primal = convert(R, _primal_value(real(coordinate)))
    if iszero(coordinate_primal) && typeof(real(coordinate)) === R
        return one(real(coordinate))::typeof(real(coordinate))
    end
    if order == one(R)
        unit_coordinate = real(coordinate)
        value = _number_contains_ad(unit_coordinate) ?
                _continuous_exact_unit_real_wide(
                    unit_coordinate, R; scaled=true,
                ) : erfc(-unit_coordinate)
        _number_isfinite(value) || throw(DomainError(
            value, "scaled continuous-order transition is non-finite",
        ))
        return value::typeof(real(coordinate))
    end
    if order == R(2)
        order_two_coordinate = real(coordinate)
        value = if _number_contains_ad(order_two_coordinate)
            _continuous_exact_order_two_scaled_real_wide(
                order_two_coordinate, R,
            )
        elseif coordinate_primal >= zero(R)
            scalar = zero(order_two_coordinate) +
                     _typed_pi(order_two_coordinate)
            exp(-(order_two_coordinate * order_two_coordinate)) +
            sqrt(scalar) * order_two_coordinate * erfc(-order_two_coordinate)
        else
            coordinate_bits = max(0, exponent(-coordinate_primal))
            work_precision = max(192, precision(R) + 2coordinate_bits + 64)
            setprecision(BigFloat, work_precision) do
                wide_coordinate = _faddeeva_widen_bigfloat(
                    order_two_coordinate,
                )
                wide_value = exp(-(wide_coordinate * wide_coordinate)) +
                    sqrt(BigFloat(pi)) * wide_coordinate * erfc(-wide_coordinate)
                convert(typeof(order_two_coordinate), wide_value)
            end
        end
        _number_isfinite(value) || throw(DomainError(
            value, "scaled continuous-order transition is non-finite",
        ))
        return value::typeof(real(coordinate))
    end
    if iszero(coordinate_primal)
        return real(_continuous_zero_linearization(
            order, coordinate,
        ))::typeof(real(coordinate))
    end
    _continuous_log_normalizer(order, "continuous order mu")
    _continuous_nested_ad(coordinate) && throw(ArgumentError(
        "nested coordinate differentiation is supported only by exact " *
        "continuous-order members",
    ))
    if _number_contains_ad(coordinate) &&
       _continuous_quad_norm(coordinate) > sqrt(floatmax(R))
        return real(_continuous_integral_wide_ad(
            order, coordinate, evaluation_limit; scaled=true,
        ))::typeof(real(coordinate))
    end
    value = _continuous_integral(
        order, coordinate;
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        maxevals=evaluation_limit,
        real_shift=-(real(coordinate) * real(coordinate)),
        centered_real=Val(true),
    )
    imaginary_scale = convert(R, abs(_primal_value(imag(value))))
    real_scale = max(one(R), convert(R, abs(_primal_value(real(value)))))
    imaginary_scale <= relative_tolerance * real_scale || throw(DomainError(
        value, "scaled real-axis continuous-order transition acquired an " *
               "uncertified imaginary component",
    ))
    return real(value)::typeof(real(coordinate))
end

function scaled_continuous_order_transition(
    mu::Real,
    zeta::AbstractArray{<:Real};
    rtol::Union{Nothing,Real}=nothing,
    atol::Union{Nothing,Real}=nothing,
    maxevals::Integer=_CONTINUOUS_DEFAULT_MAXEVALS,
)
    coordinate_type = eltype(zeta)
    isconcretetype(coordinate_type) || throw(ArgumentError(
        "scaled continuous-order coordinate arrays require a concrete " *
        "element type",
    ))
    _, _, R = _continuous_transition_values(mu, zero(coordinate_type))
    _continuous_tolerances(rtol, atol, R)
    _continuous_maxevals(maxevals)
    return map(
        value -> scaled_continuous_order_transition(
            mu, value; rtol, atol, maxevals,
        ),
        zeta,
    )
end

function _continuous_order_coordinate_derivative(mu::Real, zeta::Number)
    order, coordinate, R = _continuous_transition_values(mu, zeta)
    logarithmic_ratio = log(R(2)) + loggamma((order + one(R)) / 2) -
                        loggamma(order / 2)
    adjacent = continuous_order_transition(order + one(R), coordinate)
    return _continuous_rescale_value(
        adjacent, logarithmic_ratio,
        "continuous-order coordinate derivative",
    )
end

function _continuous_order_second_coordinate_derivative(mu::Real, zeta::Number)
    order, coordinate, _ = _continuous_transition_values(mu, zeta)
    value = continuous_order_transition(order, coordinate)
    derivative = _continuous_order_coordinate_derivative(order, coordinate)
    result = 2coordinate * derivative + 2order * value
    _number_isfinite(result) || throw(DomainError(
        result, "continuous-order second coordinate derivative is non-finite",
    ))
    return result
end

function _continuous_order_recurrence(mu::Real, zeta::Number)
    order, coordinate, R = _continuous_transition_values(mu, zeta)
    value = continuous_order_transition(order, coordinate)
    adjacent = continuous_order_transition(order + one(R), coordinate)
    logarithmic_ratio = loggamma((order + one(R)) / 2) - loggamma(order / 2)
    coefficient = 2coordinate * exp(logarithmic_ratio) / order
    result = value + coefficient * adjacent
    _number_isfinite(result) || throw(DomainError(
        result, "continuous-order recurrence result is non-finite",
    ))
    return result
end

function _continuous_order_parameter_derivative_endpoint(
    order::R,
    coordinate::Complex{T};
    rtol::R,
    atol::R,
    maxevals::Int,
)::Complex{T} where {R<:AbstractFloat,T<:Real}
    real_coordinate = convert(R, _primal_value(real(coordinate)))
    scale = max(one(R), -2real_coordinate)
    inverse_scale = inv(scale)
    inverse_order = inv(order)
    log_scale = log(scale)
    near = function (x::R)
        iszero(x) && return zero(coordinate)
        u = exp(log(x) * inverse_order) * inverse_scale
        logarithm = log(x) * inverse_order - log_scale
        return logarithm * exp(-u^2 + 2coordinate * u) / order
    end
    tail = function (scaled_u::R)
        isfinite(scaled_u) || return zero(coordinate)
        u = scaled_u * inverse_scale
        return (log(scaled_u) - log_scale) *
               exp((order - one(R)) * log(scaled_u) - u^2 + 2coordinate * u)
    end
    local_rtol = max(rtol / R(8), 8eps(R))
    local_atol = max(R(2e-15), 16eps(R))
    first, first_error = quadgk(
        near, zero(R), one(R);
        rtol=local_rtol, atol=local_atol, maxevals=maxevals,
        norm=_continuous_quad_norm, order=15,
    )
    tail_mode = scale * _continuous_tail_mode(order, real_coordinate)
    if tail_mode > one(R) && isfinite(tail_mode)
        rest, rest_error = quadgk(
            tail, one(R), tail_mode, R(Inf);
            rtol=local_rtol, atol=local_atol, maxevals=maxevals,
            norm=_continuous_quad_norm, order=15,
        )
    else
        rest, rest_error = quadgk(
            tail, one(R), R(Inf);
            rtol=local_rtol, atol=local_atol, maxevals=maxevals,
            norm=_continuous_quad_norm, order=15,
        )
    end
    log_normalization = log(R(2)) - loggamma(order / 2) - order * log_scale
    logarithmic_moment = _continuous_rescale_value(
        first + rest, log_normalization,
        "continuous-order logarithmic endpoint moment",
    )
    value = continuous_order_transition(
        order, coordinate; rtol=rtol, atol=atol, maxevals=maxevals,
    )
    digamma_factor = digamma(order / 2) / 2
    result = logarithmic_moment - digamma_factor * value
    quadrature_error = _continuous_rescale_error(
        first_error + rest_error, log_normalization,
    )
    value_error = abs(digamma_factor) * max(
        atol,
        rtol * convert(R, float(_primal_value(abs(value)))),
    )
    return _continuous_certify(
        result, quadrature_error + value_error, rtol, atol,
        "continuous-order parameter derivative",
    )
end

function _continuous_order_parameter_derivative(
    mu::Real,
    zeta::Number;
    rtol::Union{Nothing,Real}=nothing,
    atol::Union{Nothing,Real}=nothing,
    maxevals::Integer=_CONTINUOUS_DEFAULT_MAXEVALS,
)
    order, coordinate, R = _continuous_transition_values(mu, zeta)
    _continuous_log_normalizer(order, "continuous order mu")
    derivative_atol = atol === nothing ?
                      max(R(2e-14), 64eps(R)) : atol
    relative_tolerance, absolute_tolerance =
        _continuous_tolerances(rtol, derivative_atol, R)
    evaluation_limit = _continuous_maxevals(maxevals)
    evaluation_limit >= 127 || throw(ArgumentError(
        "continuous-order parameter derivatives require maxevals >= 127",
    ))
    if iszero(_primal_value(abs(coordinate)))
        _continuous_ad_depth(coordinate) <= 2 || throw(ArgumentError(
            "zero-coordinate order sensitivity supports at most second-order " *
            "coordinate differentiation",
        ))
        order_bits = max(0, exponent(order))
        work_precision = max(192, precision(R) + order_bits + 64)
        linear_coefficient = setprecision(BigFloat, work_precision) do
            wide_order = BigFloat(order)
            adjacent = (wide_order + 1) / 2
            endpoint = wide_order / 2
            ratio = exp(
                log(BigFloat(2)) + loggamma(adjacent) - loggamma(endpoint),
            )
            convert(
                R,
                ratio * (digamma(adjacent) - digamma(endpoint)) / 2,
            )
        end
        result = linear_coefficient * coordinate + coordinate * coordinate
        _number_isfinite(result) || throw(DomainError(
            result,
            "continuous-order parameter derivative exceeds the active range",
        ))
        return result
    end
    _continuous_nested_ad(coordinate) && throw(ArgumentError(
        "nested coordinate differentiation of order sensitivity is supported " *
        "only at exact coalescence",
    ))
    real_coordinate = convert(R, _primal_value(real(coordinate)))
    if real_coordinate <= R(-8)
        if _number_contains_ad(coordinate) &&
           _continuous_quad_norm(coordinate) > sqrt(floatmax(R))
            return _continuous_integral_wide_ad(
                order, coordinate, evaluation_limit;
                logarithmic_weight=true,
            )
        end
        return _continuous_order_parameter_derivative_endpoint(
            order, coordinate;
            rtol=relative_tolerance,
            atol=absolute_tolerance,
            maxevals=evaluation_limit,
        )
    end
    return _continuous_integral(
        order, coordinate;
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        maxevals=evaluation_limit,
        logarithmic_weight=Val(true),
    )
end

function _continuous_gamma_expectation(
    order::R,
    X::R;
    rtol::R,
    atol::R,
    maxevals::Int,
)::Complex{R} where {R<:AbstractFloat}
    if R === Float32
        wide_rtol = max(Float64(rtol) / 16, 2e-13)
        wide_atol = Float64(atol) / 16
        wide_value = _continuous_gamma_expectation(
            Float64(order), Float64(X);
            rtol=wide_rtol,
            atol=wide_atol,
            maxevals=maxevals,
        )
        value = ComplexF32(wide_value)
        conversion_error = abs(ComplexF64(value) - wide_value)
        tolerance = max(
            Float64(atol), Float64(rtol) * abs(ComplexF64(value)),
        )
        conversion_error <= tolerance || throw(DomainError(
            conversion_error,
            "binary32 UTD-normalized transition cannot meet the requested " *
            "tolerance after rounding",
        ))
        return value
    end
    log_normalizer = loggamma(order)
    isfinite(log_normalizer) || throw(DomainError(
        order, "UTD-normalized continuous order is outside the active range",
    ))
    local_rtol = max(rtol / R(2), 8eps(R))
    local_atol = max(
        atol / R(2), rtol / R(4), R(8) * nextfloat(zero(R)),
    )
    phase_denominator = 4X

    if order >= R(32)
        sigma = sqrt(order)
        high_mode = order - one(R)
        lower = -high_mode / sigma
        integrand = function (t::R)
            y = muladd(sigma, t, high_mode)
            y <= zero(R) && return zero(Complex{R})
            log_magnitude = (order - one(R)) * log(y) - y -
                            log_normalizer + log(sigma)
            _continuous_underflow(log_magnitude, R) && return zero(Complex{R})
            return exp(log_magnitude) * cis(y^2 / phase_denominator)
        end
        if lower < R(-12)
            value, estimated_error = quadgk(
                integrand,
                lower, R(-12), R(-8), R(-6), R(-4), R(-2), zero(R),
                R(2), R(4), R(6), R(8), R(12), R(Inf);
                rtol=local_rtol, atol=local_atol, maxevals=maxevals,
            )
        elseif lower < R(-8)
            value, estimated_error = quadgk(
                integrand,
                lower, R(-8), R(-6), R(-4), R(-2), zero(R),
                R(2), R(4), R(6), R(8), R(12), R(Inf);
                rtol=local_rtol, atol=local_atol, maxevals=maxevals,
            )
        elseif lower < R(-6)
            value, estimated_error = quadgk(
                integrand,
                lower, R(-6), R(-4), R(-2), zero(R),
                R(2), R(4), R(6), R(8), R(12), R(Inf);
                rtol=local_rtol, atol=local_atol, maxevals=maxevals,
            )
        else
            value, estimated_error = quadgk(
                integrand,
                lower, R(-4), R(-2), zero(R),
                R(2), R(4), R(6), R(8), R(12), R(Inf);
                rtol=local_rtol, atol=local_atol, maxevals=maxevals,
            )
        end
        return _continuous_certify(
            value, estimated_error, rtol, atol,
            "UTD-normalized continuous-order transition",
        )
    end

    near = function (x::R)
        iszero(x) && return zero(Complex{R})
        y = exp(log(x) / order)
        log_magnitude = -y - log_normalizer - log(order)
        _continuous_underflow(log_magnitude, R) && return zero(Complex{R})
        return exp(log_magnitude) * cis(y^2 / phase_denominator)
    end
    tail = function (y::R)
        isfinite(y) || return zero(Complex{R})
        log_magnitude = (order - one(R)) * log(y) - y - log_normalizer
        _continuous_underflow(log_magnitude, R) && return zero(Complex{R})
        return exp(log_magnitude) * cis(y^2 / phase_denominator)
    end
    near_value, near_error = quadgk(
        near, zero(R), one(R);
        rtol=local_rtol, atol=local_atol, maxevals=maxevals,
    )
    tail_mode = order - one(R)
    if tail_mode > one(R)
        tail_value, tail_error = quadgk(
            tail, one(R), tail_mode, R(Inf);
            rtol=local_rtol, atol=local_atol, maxevals=maxevals,
        )
    else
        tail_value, tail_error = quadgk(
            tail, one(R), R(Inf);
            rtol=local_rtol, atol=local_atol, maxevals=maxevals,
        )
    end
    value = near_value + tail_value
    return _continuous_certify(
        value, near_error + tail_error, rtol, atol,
        "UTD-normalized continuous-order transition",
    )
end

function _continuous_order_utd_transition(
    mu::Real,
    X::Real;
    rtol::Union{Nothing,Real}=nothing,
    atol::Union{Nothing,Real}=nothing,
    maxevals::Integer=_CONTINUOUS_DEFAULT_MAXEVALS,
)
    R = _continuous_plain_real_type(mu, X)
    order = convert(R, float(mu))
    argument = convert(R, float(X))
    (isfinite(order) && order > zero(R)) || throw(DomainError(
        mu, "continuous order mu must be finite and positive",
    ))
    order <= R(_CONTINUOUS_UTD_MAX_ORDER) || throw(DomainError(
        mu,
        "UTD-normalized continuous order exceeds $_CONTINUOUS_UTD_MAX_ORDER",
    ))
    (isfinite(argument) && argument >= zero(R)) || throw(DomainError(
        X, "UTD-normalized argument X must be finite and nonnegative",
    ))
    relative_tolerance, absolute_tolerance =
        _continuous_tolerances(rtol, atol, R)
    evaluation_limit = _continuous_maxevals(maxevals)
    iszero(argument) && return zero(Complex{R})
    order == one(R) && return F_utd(argument)
    if argument >= max(R(16), order)
        return _continuous_gamma_expectation(
            order, argument;
            rtol=relative_tolerance,
            atol=absolute_tolerance,
            maxevals=evaluation_limit,
        )
    end
    coordinate = -complex(cospi(R(0.25)), sinpi(R(0.25))) * sqrt(argument)
    value = continuous_order_transition(
        order, coordinate;
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        maxevals=evaluation_limit,
    )
    log_prefactor = log(sqrt(_typed_pi(zero(R)))) -
                    loggamma((order + one(R)) / 2) +
                    (order / 2) * log(argument)
    phase = complex(cospi(order / 4), sinpi(order / 4))
    return _continuous_rescale_value(
        phase * value, log_prefactor,
        "UTD-normalized continuous-order transition",
    )
end

@inline function _continuous_moment_index(index::Integer)
    value = try
        Int(index)
    catch error
        (error isa InexactError || error isa OverflowError) || rethrow()
        throw(DomainError(index, "continuous-order moment index is outside Int range"))
    end
    0 <= value <= _CONTINUOUS_MOMENT_MAX_ORDER || throw(DomainError(
        index,
        "continuous-order moment index must lie in 0:$_CONTINUOUS_MOMENT_MAX_ORDER",
    ))
    return value
end

@inline function _continuous_physical_values(
    nu::Real,
    k::Real,
    h::Real,
    tau::Real,
)
    R = _continuous_plain_real_type(nu, k, h, tau)
    order = convert(R, float(nu))
    wavenumber = convert(R, float(k))
    curvature = convert(R, float(h))
    displacement = convert(R, float(tau))
    (isfinite(order) && order > zero(R)) || throw(DomainError(
        nu, "endpoint order nu must be finite and positive",
    ))
    (isfinite(wavenumber) && wavenumber > zero(R)) || throw(DomainError(
        k, "wavenumber k must be finite and positive",
    ))
    (isfinite(curvature) && curvature > zero(R)) || throw(DomainError(
        h, "quadratic curvature h must be finite and positive",
    ))
    isfinite(displacement) || throw(DomainError(
        tau, "signed saddle--endpoint displacement tau must be finite",
    ))
    return order, wavenumber, curvature, displacement, R
end

function _continuous_coordinate(
    wavenumber::R,
    curvature::R,
    displacement::R,
) where {R<:AbstractFloat}
    iszero(displacement) && return zero(Complex{R})
    direct_magnitude = displacement * sqrt(wavenumber) /
                       (sqrt(R(2)) * sqrt(curvature))
    if isfinite(direct_magnitude) && !iszero(direct_magnitude)
        return complex(cospi(R(0.25)), sinpi(R(0.25))) * direct_magnitude
    end
    log_magnitude = log(abs(displacement)) +
                    (log(wavenumber) - log(R(2)) - log(curvature)) / 2
    log_magnitude <= log(floatmax(R)) - R(8) || throw(DomainError(
        (wavenumber, curvature, displacement),
        "continuous-order transition coordinate exceeds the active range",
    ))
    log_magnitude < log(nextfloat(zero(R))) && return zero(Complex{R})
    magnitude = copysign(exp(log_magnitude), displacement)
    return complex(cospi(R(0.25)), sinpi(R(0.25))) * magnitude
end

@inline function _continuous_coefficient_type(::Type{R}, ::Type{A}) where {R,A}
    isconcretetype(A) || throw(ArgumentError(
        "continuous-order coefficients require a concrete numeric element type",
    ))
    A <: Number || throw(ArgumentError(
        "continuous-order coefficients must be numeric",
    ))
    primal_type = promote_type(
        R,
        typeof(float(_primal_value(real(zero(A))))),
        typeof(float(_primal_value(imag(zero(A))))),
    )
    primal_type === BigFloat && throw(ArgumentError(
        "continuous-order production moments do not expose BigFloat coefficients",
    ))
    return promote_type(Complex{R}, typeof(complex(float(zero(A)))))
end

function _continuous_order_moment_prepared(
    base_order::R,
    index::Int,
    wavenumber::R,
    curvature::R,
    coordinate::Complex{R},
    coefficient::C;
    rtol::Union{Nothing,Real},
    atol::Union{Nothing,Real},
    maxevals::Integer,
) where {R<:AbstractFloat,C<:Number}
    order = base_order + R(index)
    _continuous_log_normalizer(order, "endpoint order plus moment index")
    iszero(coefficient) && return zero(promote_type(Complex{R}, C))
    value = continuous_order_transition(
        order, coordinate; rtol=rtol, atol=atol, maxevals=maxevals,
    )
    product = wavenumber * curvature
    logarithmic_product = if isfinite(product) && !iszero(product) &&
                             !issubnormal(product)
        log(product)
    else
        wavenumber_fraction, wavenumber_exponent = frexp(wavenumber)
        curvature_fraction, curvature_exponent = frexp(curvature)
        log(wavenumber_fraction * curvature_fraction) +
        (wavenumber_exponent + curvature_exponent) * log(R(2))
    end
    log_prefactor = loggamma(order / 2) - log(R(2)) +
                    (order / 2) * (log(R(2)) - logarithmic_product)
    phase = complex(cospi(-order / 4), sinpi(-order / 4))
    unscaled = coefficient * phase * value
    primal_real = _primal_value(real(unscaled))
    primal_imag = _primal_value(imag(unscaled))
    native_product_is_usable = typeof(real(coefficient)) === R &&
        _continuous_allfinite(unscaled) &&
        !(iszero(primal_real) && iszero(primal_imag)) &&
        (iszero(primal_real) || !issubnormal(primal_real)) &&
        (iszero(primal_imag) || !issubnormal(primal_imag))
    lower_direct_limit = log(floatmin(R)) + R(8)
    upper_direct_limit = log(floatmax(R)) - R(8)
    if native_product_is_usable &&
       lower_direct_limit <= log_prefactor <= upper_direct_limit
        return _continuous_rescale_value(
            unscaled, log_prefactor, "continuous-order physical moment",
        )
    end

    work_precision = max(192, precision(R) + max(0, exponent(order)) + 64)
    result_type = promote_type(C, Complex{R})
    result = setprecision(BigFloat, work_precision) do
        wide_order = BigFloat(order)
        wide_product = BigFloat(wavenumber) * BigFloat(curvature)
        wide_log_prefactor = loggamma(wide_order / 2) - log(BigFloat(2)) +
            (wide_order / 2) * (log(BigFloat(2)) - log(wide_product))
        convert(
            result_type,
            _faddeeva_widen_bigfloat(coefficient) *
            _faddeeva_widen_bigfloat(phase) *
            _faddeeva_widen_bigfloat(value) * exp(wide_log_prefactor),
        )
    end
    _continuous_allfinite(result) || throw(DomainError(
        result, "continuous-order physical moment exceeds the active range",
    ))
    return result
end

function _continuous_order_hierarchy_wide(
    ::Type{C},
    base_order::R,
    wavenumber::R,
    curvature::R,
    displacement::R,
    coefficients,
    maximum_order::Int,
    maxevals::Int,
) where {C<:Number,R<:AbstractFloat}
    first_coefficient = firstindex(coefficients)
    previous = nothing
    @inbounds for work_precision in
        (256, 512, _CONTINUOUS_MAX_HIERARCHY_RECOVERY_BITS)
        wide_result = setprecision(BigFloat, work_precision) do
            wide_base_order = BigFloat(base_order)
            wide_wavenumber = BigFloat(wavenumber)
            wide_curvature = BigFloat(curvature)
            wide_displacement = BigFloat(displacement)
            coordinate = complex(
                cospi(BigFloat(0.25)), sinpi(BigFloat(0.25)),
            ) * wide_displacement *
                sqrt(wide_wavenumber / (2wide_curvature))
            quadrature_rtol = BigFloat(2)^(-min(work_precision ÷ 2, 256))
            total = zero(Complex{BigFloat})
            for index in 0:maximum_order
                order = wide_base_order + index
                canonical = iszero(coordinate) ? one(coordinate) :
                    _continuous_integral(
                        order, coordinate;
                        rtol=quadrature_rtol,
                        atol=zero(BigFloat),
                        maxevals=maxevals,
                    )
                log_prefactor = loggamma(order / 2) - log(BigFloat(2)) +
                    (order / 2) *
                    (log(BigFloat(2)) -
                     log(wide_wavenumber * wide_curvature))
                phase = complex(cospi(-order / 4), sinpi(-order / 4))
                coefficient = convert(
                    C, complex(float(coefficients[first_coefficient + index])),
                )
                total += _faddeeva_widen_bigfloat(coefficient) * phase *
                         canonical * exp(log_prefactor)
            end
            total
        end
        converted = convert(C, wide_result)
        _continuous_allfinite(converted) || throw(DomainError(
            converted,
            "continuous-order moment hierarchy exceeds the active range",
        ))
        previous !== nothing && isequal(converted, previous) && return converted
        previous = converted
    end
    throw(ArgumentError(
        "continuous-order moment hierarchy did not stabilize within the " *
        "bounded $_CONTINUOUS_MAX_HIERARCHY_RECOVERY_BITS-bit workspace",
    ))
end

"""
    continuous_order_moment(nu, index, k, h, tau, coefficient; kwargs...)

Evaluate one analytic-amplitude term
`coefficient * exp(-im*pi*(nu+index)/4) * Gamma((nu+index)/2)/2 *
 (2/(k*h))^((nu+index)/2) * B_(nu+index)(zeta)`, where
`zeta=exp(im*pi/4)*tau*sqrt(k/(2h))`. The moment index is limited to `0:64`.
Prefactors are combined in the log domain before scaling the finite canonical
value.
"""
function continuous_order_moment(
    nu::Real,
    index::Integer,
    k::Real,
    h::Real,
    tau::Real,
    coefficient::Number;
    rtol::Union{Nothing,Real}=nothing,
    atol::Union{Nothing,Real}=nothing,
    maxevals::Integer=_CONTINUOUS_DEFAULT_MAXEVALS,
)
    moment_index = _continuous_moment_index(index)
    base_order, wavenumber, curvature, displacement, R =
        _continuous_physical_values(nu, k, h, tau)
    relative_tolerance, absolute_tolerance =
        _continuous_tolerances(rtol, atol, R)
    evaluation_limit = _continuous_maxevals(maxevals)
    C = _continuous_coefficient_type(R, typeof(coefficient))
    converted_coefficient = convert(C, complex(float(coefficient)))
    _continuous_allfinite(converted_coefficient) || throw(DomainError(
        coefficient, "continuous-order amplitude coefficient must be finite",
    ))
    coordinate = _continuous_coordinate(wavenumber, curvature, displacement)
    return _continuous_order_moment_prepared(
        base_order, moment_index, wavenumber, curvature, coordinate,
        converted_coefficient;
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        maxevals=evaluation_limit,
    )
end

"""
    continuous_order_moment(nu, coefficients, k, h, tau;
                            order=nothing, kwargs...)

Evaluate the bounded analytic-amplitude hierarchy for ascending coefficients.
All coefficients are validated in place; no coefficient or term array is
constructed. `order` may truncate the available vector.
"""
function continuous_order_moment(
    nu::Real,
    coefficients::AbstractVector{A},
    k::Real,
    h::Real,
    tau::Real;
    order::Union{Nothing,Integer}=nothing,
    rtol::Union{Nothing,Real}=nothing,
    atol::Union{Nothing,Real}=nothing,
    maxevals::Integer=_CONTINUOUS_DEFAULT_MAXEVALS,
) where {A<:Number}
    base_order, wavenumber, curvature, displacement, R =
        _continuous_physical_values(nu, k, h, tau)
    relative_tolerance, absolute_tolerance =
        _continuous_tolerances(rtol, atol, R)
    evaluation_limit = _continuous_maxevals(maxevals)
    C = _continuous_coefficient_type(R, A)
    coefficient_count = length(coefficients)
    order !== nothing && order < 0 && throw(DomainError(
        order, "continuous-order hierarchy order must be nonnegative",
    ))
    coefficient_count == 0 && return zero(C)
    for coefficient in coefficients
        _number_isfinite(coefficient) || throw(DomainError(
            coefficient, "continuous-order amplitude coefficient must be finite",
        ))
    end
    requested_order = order === nothing ? coefficient_count - 1 : order
    maximum_order = Int(min(requested_order, coefficient_count - 1))
    maximum_order <= _CONTINUOUS_MOMENT_MAX_ORDER || throw(DomainError(
        maximum_order,
        "continuous-order hierarchy exceeds moment index " *
        "$_CONTINUOUS_MOMENT_MAX_ORDER",
    ))
    first_coefficient = firstindex(coefficients)
    @inbounds for index in 0:maximum_order
        coefficient = convert(
            C, complex(float(coefficients[first_coefficient + index])),
        )
        if _number_contains_ad(coefficient)
            return _continuous_order_hierarchy_wide(
                C, base_order, wavenumber, curvature, displacement,
                coefficients, maximum_order, evaluation_limit,
            )
        end
    end
    coordinate = _continuous_coordinate(wavenumber, curvature, displacement)
    total = zero(C)
    compensation = zero(C)
    absolute_sum = zero(R)
    @inbounds for index in 0:maximum_order
        coefficient = convert(
            C, complex(float(coefficients[first_coefficient + index])),
        )
        term = try
            _continuous_order_moment_prepared(
                base_order, index, wavenumber, curvature, coordinate, coefficient;
                rtol=relative_tolerance,
                atol=absolute_tolerance,
                maxevals=evaluation_limit,
            )
        catch error
            error isa DomainError || rethrow()
            return _continuous_order_hierarchy_wide(
                C, base_order, wavenumber, curvature, displacement,
                coefficients, maximum_order, evaluation_limit,
            )
        end
        total, compensation = _multipole_compensated_add(
            total, compensation, term,
        )
        absolute_sum += convert(R, float(_primal_value(abs(term))))
    end
    total_magnitude = convert(R, float(_primal_value(abs(total))))
    condition_threshold = min(
        R(0.5), max(R(0.125), 16eps(R) / relative_tolerance),
    )
    if !_number_isfinite(total) ||
       (!iszero(absolute_sum) &&
        (!isfinite(absolute_sum) ||
         total_magnitude <= condition_threshold * absolute_sum))
        return _continuous_order_hierarchy_wide(
            C, base_order, wavenumber, curvature, displacement, coefficients,
            maximum_order, evaluation_limit,
        )
    end
    return total
end
