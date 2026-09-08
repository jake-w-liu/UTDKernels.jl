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

@inline _continuous_quad_norm(value) = float(_primal_value(abs(value)))

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

@inline function _continuous_expected_magnitude(
    order::R,
    real_coordinate::R,
    real_shift::R,
    logarithmic_weight::Bool,
) where {R<:AbstractFloat}
    log_estimate = if real_coordinate <= -one(R)
        loggamma((order + one(R)) / 2) - log(sqrt(_typed_pi(zero(R)))) -
        order * log(-real_coordinate) + real_shift
    elseif real_coordinate >= one(R)
        log(R(2) * sqrt(_typed_pi(zero(R)))) - loggamma(order / 2) +
        real_coordinate^2 + (order - one(R)) * log(real_coordinate) +
        real_shift
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

function _continuous_rescale_value(
    value::V,
    log_scale::R,
    context::AbstractString,
)::V where {V<:Number,R<:AbstractFloat}
    iszero(log_scale) && return value
    direct_limit = log(floatmax(R)) - R(8)
    if log_scale <= direct_limit
        result = value * exp(log_scale)
        _number_isfinite(result) || throw(DomainError(
            result, "$context exceeds the active numeric range",
        ))
        return result
    end
    primal_magnitude = convert(R, float(_primal_value(abs(value))))
    iszero(primal_magnitude) && return zero(value)
    isfinite(primal_magnitude) || throw(DomainError(
        value, "$context produced a non-finite scaled integral",
    ))
    result_log_magnitude = log(primal_magnitude) + log_scale
    result_log_magnitude <= direct_limit || throw(DomainError(
        result_log_magnitude, "$context exceeds the active numeric range",
    ))
    magnitude = exp(result_log_magnitude)
    return (value / primal_magnitude) * magnitude
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
    value_scale = convert(R, float(_primal_value(abs(value))))
    tolerance = max(atol, rtol * value_scale)
    (isfinite(estimated_error) && estimated_error <= tolerance) ||
        throw(DomainError(
            estimated_error,
            "$context quadrature did not meet the requested tolerance",
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
)::Complex{T} where {R<:AbstractFloat,T<:Real,L}
    log_normalizer = _continuous_log_normalizer(order, "continuous order mu")
    real_coordinate = convert(R, _primal_value(real(coordinate)))
    coordinate_limit = sqrt(floatmax(R)) / R(4)
    abs(real_coordinate) <= coordinate_limit || throw(DomainError(
        coordinate, "continuous-order coordinate is outside the quadrature range",
    ))
    scale = max(one(R), -2real_coordinate)
    inverse_scale = inv(scale)
    base = log(R(2)) - log_normalizer - order * log(scale) + real_shift
    inverse_order = inv(order)
    log_order = log(order)
    digamma_term = L ? digamma(order / 2) / 2 : zero(R)

    near_upper = inverse_scale
    near_mode = clamp(real_coordinate, zero(R), near_upper)
    near_peak = convert(
        R,
        _primal_value(
            base - log_order - near_mode^2 + 2real_coordinate * near_mode,
        ),
    )
    near_shift = max(zero(R), near_peak)

    tail_mode = _continuous_tail_mode(order, real_coordinate)
    tail_mode_scaled = scale * tail_mode
    tail_at_one = convert(
        R,
        _primal_value(
            base - inverse_scale^2 + 2real_coordinate * inverse_scale,
        ),
    )
    tail_peak = tail_at_one
    if tail_mode_scaled > one(R) && isfinite(tail_mode_scaled)
        tail_peak = max(
            tail_peak,
            convert(
                R,
                _primal_value(
                    base + (order - one(R)) * log(tail_mode_scaled) -
                    tail_mode^2 + 2real_coordinate * tail_mode,
                ),
            ),
        )
    end
    tail_shift = max(zero(R), tail_peak)

    near = function (x::R)
        iszero(x) && return zero(coordinate)
        scaled_u = exp(log(x) * inverse_order)
        u = scaled_u * inverse_scale
        log_magnitude = base - log_order - u^2 +
                        2real(coordinate) * u - near_shift
        _continuous_underflow(log_magnitude, R) && return zero(coordinate)
        integrand_value = exp(log_magnitude) * cis(2imag(coordinate) * u)
        if L
            integrand_value *=
                log(x) * inverse_order - log(scale) - digamma_term
        end
        return integrand_value
    end
    tail = function (scaled_u::R)
        isfinite(scaled_u) || return zero(coordinate)
        u = scaled_u * inverse_scale
        log_magnitude = base + (order - one(R)) * log(scaled_u) - u^2 +
                        2real(coordinate) * u - tail_shift
        _continuous_underflow(log_magnitude, R) && return zero(coordinate)
        integrand_value = exp(log_magnitude) * cis(2imag(coordinate) * u)
        L &&
            (integrand_value *= log(scaled_u) - log(scale) - digamma_term)
        return integrand_value
    end

    expected_magnitude = _continuous_expected_magnitude(
        order, real_coordinate, convert(R, _primal_value(real_shift)), L,
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
    near_value, near_error = quadgk(
        near, zero(R), one(R);
        rtol=local_rtol, atol=near_atol, maxevals=maxevals,
        norm=_continuous_quad_norm, order=quadrature_order,
    )
    if tail_mode_scaled > one(R) && isfinite(tail_mode_scaled)
        tail_value, tail_error = quadgk(
            tail, one(R), tail_mode_scaled, R(Inf);
            rtol=local_rtol, atol=tail_atol, maxevals=maxevals,
            norm=_continuous_quad_norm, order=quadrature_order,
        )
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
their promoted complex type. Coordinate ForwardDiff arithmetic is supported;
differentiation with respect to `mu` is available through the paper's explicit
order-sensitivity identity. BigFloat is reserved for independent oracles.
"""
function continuous_order_transition(
    mu::Real,
    zeta::Number;
    rtol::Union{Nothing,Real}=nothing,
    atol::Union{Nothing,Real}=nothing,
    maxevals::Integer=_CONTINUOUS_DEFAULT_MAXEVALS,
)
    order, coordinate, R = _continuous_transition_values(mu, zeta)
    _continuous_log_normalizer(order, "continuous order mu")
    relative_tolerance, absolute_tolerance =
        _continuous_tolerances(rtol, atol, R)
    evaluation_limit = _continuous_maxevals(maxevals)
    C = typeof(coordinate)
    if iszero(_primal_value(abs(coordinate)))
        return one(C)
    end
    if order == one(R)
        value = convert(C, _faddeeva_w(-im * coordinate))
        _number_isfinite(value) || throw(DomainError(
            zeta,
            "unscaled continuous-order transition exceeds the active range; " *
            "use scaled_continuous_order_transition on the real saddle branch",
        ))
        return value
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
    kwargs...,
)
    return map(value -> continuous_order_transition(mu, value; kwargs...), zeta)
end

"""
    scaled_continuous_order_transition(mu, zeta; kwargs...)

Evaluate `exp(-zeta^2) * continuous_order_transition(mu, zeta)` directly for a
finite real saddle coordinate. The exponential shift stays inside the
log-domain integrand, so large positive coordinates do not form the
overflowing unscaled value. Exact unit order reduces to `erfc(-zeta)`.
"""
function scaled_continuous_order_transition(
    mu::Real,
    zeta::Real;
    rtol::Union{Nothing,Real}=nothing,
    atol::Union{Nothing,Real}=nothing,
    maxevals::Integer=_CONTINUOUS_DEFAULT_MAXEVALS,
)
    order, coordinate, R = _continuous_transition_values(mu, zeta)
    _continuous_log_normalizer(order, "continuous order mu")
    relative_tolerance, absolute_tolerance =
        _continuous_tolerances(rtol, atol, R)
    evaluation_limit = _continuous_maxevals(maxevals)
    coordinate_primal = convert(R, _primal_value(real(coordinate)))
    if iszero(coordinate_primal)
        return one(real(coordinate))
    end
    if order == one(R) && typeof(real(coordinate)) === R
        value = erfc(-real(coordinate))
        isfinite(value) || throw(DomainError(
            value, "scaled continuous-order transition is non-finite",
        ))
        return value
    end
    value = _continuous_integral(
        order, coordinate;
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        maxevals=evaluation_limit,
        real_shift=-(real(coordinate) * real(coordinate)),
    )
    imaginary_scale = convert(R, abs(_primal_value(imag(value))))
    real_scale = max(one(R), convert(R, abs(_primal_value(real(value)))))
    imaginary_scale <= relative_tolerance * real_scale || throw(DomainError(
        value, "scaled real-axis continuous-order transition acquired an " *
               "uncertified imaginary component",
    ))
    return real(value)
end

function scaled_continuous_order_transition(
    mu::Real,
    zeta::AbstractArray{<:Real};
    kwargs...,
)
    return map(
        value -> scaled_continuous_order_transition(mu, value; kwargs...),
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
    iszero(_primal_value(abs(coordinate))) && return zero(coordinate)
    real_coordinate = convert(R, _primal_value(real(coordinate)))
    if real_coordinate <= R(-8)
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
    log_prefactor = loggamma(order / 2) - log(R(2)) +
                    (order / 2) * (
                        log(R(2)) - log(wavenumber) - log(curvature)
                    )
    phase = complex(cospi(-order / 4), sinpi(-order / 4))
    return _continuous_rescale_value(
        coefficient * phase * value,
        log_prefactor,
        "continuous-order physical moment",
    )
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
    _number_isfinite(converted_coefficient) || throw(DomainError(
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
    coefficient_count == 0 && return zero(C)
    for coefficient in coefficients
        _number_isfinite(coefficient) || throw(DomainError(
            coefficient, "continuous-order amplitude coefficient must be finite",
        ))
    end
    requested_order = order === nothing ? coefficient_count - 1 : order
    requested_order >= 0 || throw(DomainError(
        order, "continuous-order hierarchy order must be nonnegative",
    ))
    maximum_order = Int(min(requested_order, coefficient_count - 1))
    maximum_order <= _CONTINUOUS_MOMENT_MAX_ORDER || throw(DomainError(
        maximum_order,
        "continuous-order hierarchy exceeds moment index " *
        "$_CONTINUOUS_MOMENT_MAX_ORDER",
    ))
    coordinate = _continuous_coordinate(wavenumber, curvature, displacement)
    first_coefficient = firstindex(coefficients)
    total = zero(C)
    compensation = zero(C)
    @inbounds for index in 0:maximum_order
        coefficient = convert(
            C, complex(float(coefficients[first_coefficient + index])),
        )
        term = _continuous_order_moment_prepared(
            base_order, index, wavenumber, curvature, coordinate, coefficient;
            rtol=relative_tolerance,
            atol=absolute_tolerance,
            maxevals=evaluation_limit,
        )
        total, compensation = _multipole_compensated_add(
            total, compensation, term,
        )
    end
    _number_isfinite(total) || throw(DomainError(
        total, "continuous-order moment hierarchy is non-finite",
    ))
    return total
end
