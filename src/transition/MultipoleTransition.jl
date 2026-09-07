"""
Faddeeva divided differences for finite clusters of saddle-adjacent poles.

The direct barycentric representation is efficient for separated nodes. A
centered complete-homogeneous expansion evaluates the same analytic function
at partial or complete coalescence without subtracting divergent residues.
"""

const _MULTIPOLE_MAX_ORDER = 7
const _MULTIPOLE_DEFAULT_RADIUS_THRESHOLD = 0.075
const _MULTIPOLE_DEFAULT_CANCELLATION_THRESHOLD = 2.0e4
const _MULTIPOLE_DEFAULT_MAX_TERMS = 48
const _MULTIPOLE_MIN_TERMS = 2
const _MULTIPOLE_MAX_TERMS = 128

"""
    MultipoleEvaluationInfo

Diagnostics from [`multipole_transition`](@ref). `method` is one of
`:single`, `:direct`, or `:cluster`; `direct_condition_estimate` is the
barycentric cancellation estimate and is `Inf` when no finite direct
representation exists.
"""
struct MultipoleEvaluationInfo{C<:Number,R<:Real}
    method::Symbol
    order::Int
    center::C
    cluster_radius::R
    direct_condition_estimate::R
    terms_used::Int
end

@inline function _multipole_node_type(::Type{T}) where {T<:Number}
    isconcretetype(T) || throw(ArgumentError(
        "multipole nodes require a concrete numeric element type",
    ))
    C = typeof(complex(float(zero(T))))
    isconcretetype(C) || throw(ArgumentError(
        "multipole nodes could not be promoted to a concrete complex type",
    ))
    return C
end

@inline function _multipole_real_type(::Type{C}) where {C<:Number}
    return typeof(float(_primal_value(abs(zero(C)))))
end

function _validate_multipole_nodes(nodes::AbstractVector{T}) where {T<:Number}
    isempty(nodes) && throw(ArgumentError("multipole nodes must be nonempty"))
    order = length(nodes)
    order <= _MULTIPOLE_MAX_ORDER || throw(ArgumentError(
        "multipole order $order exceeds the validated maximum $_MULTIPOLE_MAX_ORDER",
    ))
    C = _multipole_node_type(T)
    @inbounds for index in eachindex(nodes)
        node = convert(C, nodes[index])
        _number_isfinite(node) || throw(DomainError(
            nodes[index],
            "every multipole node must be finite",
        ))
    end
    return C, _multipole_real_type(C), order
end

@inline function _multipole_compensated_add(total, compensation, term)
    increment = term - compensation
    next_total = total + increment
    next_compensation = (next_total - total) - increment
    return next_total, next_compensation
end

function _multipole_center(nodes::AbstractVector, ::Type{C}) where {C<:Number}
    total = zero(C)
    compensation = zero(C)
    @inbounds for index in eachindex(nodes)
        total, compensation = _multipole_compensated_add(
            total, compensation, convert(C, nodes[index]),
        )
    end
    return total / length(nodes)
end

function _multipole_all_equal(nodes::AbstractVector, ::Type{C}) where {C<:Number}
    first_value = convert(C, nodes[firstindex(nodes)])
    @inbounds for index in eachindex(nodes)
        convert(C, nodes[index]) == first_value || return false
    end
    return true
end

function _multipole_nodes_distinct(nodes::AbstractVector, ::Type{C}) where {C<:Number}
    @inbounds for index in eachindex(nodes)
        node = convert(C, nodes[index])
        for other in eachindex(nodes)
            other == index && break
            node == convert(C, nodes[other]) && return false
        end
    end
    return true
end

function _multipole_radius(
    nodes::AbstractVector,
    center::C,
    ::Type{R},
) where {C<:Number,R<:Real}
    radius = zero(R)
    @inbounds for index in eachindex(nodes)
        distance = convert(R, float(_primal_value(abs(convert(C, nodes[index]) - center))))
        radius = max(radius, distance)
    end
    return radius
end

function _multipole_direct_evaluate(
    nodes::AbstractVector,
    ::Type{C},
    ::Type{R},
) where {C<:Number,R<:Real}
    length(nodes) == 1 && return (
        _faddeeva_w(convert(C, nodes[firstindex(nodes)])), one(R), true,
    )

    total = zero(C)
    compensation = zero(C)
    magnitude_total = zero(R)
    magnitude_compensation = zero(R)
    @inbounds for index in eachindex(nodes)
        node = convert(C, nodes[index])
        denominator = one(C)
        for other in eachindex(nodes)
            other == index && continue
            difference = node - convert(C, nodes[other])
            iszero(difference) && return zero(C), R(Inf), false
            denominator *= difference
        end
        (iszero(denominator) || !_number_isfinite(denominator)) &&
            return zero(C), R(Inf), false
        term = _faddeeva_w(node) / denominator
        _number_isfinite(term) || return zero(C), R(Inf), false
        total, compensation = _multipole_compensated_add(total, compensation, term)
        term_magnitude = convert(R, float(_primal_value(abs(term))))
        magnitude_total, magnitude_compensation = _multipole_compensated_add(
            magnitude_total, magnitude_compensation, term_magnitude,
        )
    end

    _number_isfinite(total) || return zero(C), R(Inf), false
    total_magnitude = convert(R, float(_primal_value(abs(total))))
    condition = if iszero(total_magnitude) || !isfinite(magnitude_total)
        R(Inf)
    else
        max(one(R), magnitude_total / total_magnitude)
    end
    return total, condition, true
end

"""
    faddeeva_divided_difference(nodes; return_condition=false)

Evaluate the direct barycentric divided difference of the Faddeeva function
at distinct finite nodes. Set `return_condition=true` to return
`(value, cancellation_estimate)`. This separated-node representation exposes
its cancellation diagnostic but does not regularize coalescence; use
[`multipole_transition`](@ref) for an automatic stable representation.

Orders `1:7` are accepted, matching the validated multipole evidence range.
"""
function faddeeva_divided_difference(
    nodes::AbstractVector{T};
    return_condition::Bool=false,
) where {T<:Number}
    C, R, _ = _validate_multipole_nodes(nodes)
    _multipole_nodes_distinct(nodes, C) || throw(ArgumentError(
        "direct Faddeeva divided differences require distinct nodes",
    ))
    value, condition, succeeded = _multipole_direct_evaluate(nodes, C, R)
    succeeded || throw(DomainError(
        nodes,
        "direct Faddeeva divided difference is not representable; use multipole_transition",
    ))
    return return_condition ? (value, condition) : value
end

function _multipole_complete_homogeneous!(
    coefficients::AbstractVector{C},
    nodes::AbstractVector,
    center::C,
) where {C<:Number}
    fill!(coefficients, zero(C))
    coefficients[1] = one(C)
    @inbounds for index in eachindex(nodes)
        delta = convert(C, nodes[index]) - center
        for degree in 1:(length(coefficients) - 1)
            coefficients[degree + 1] += delta * coefficients[degree]
        end
    end
    return coefficients
end

function _multipole_cluster_expansion(
    nodes::AbstractVector,
    center::C,
    order::Int,
    relative_tolerance::R,
    max_terms::Int,
) where {C<:Number,R<:Real}
    if _multipole_all_equal(nodes, C)
        value = _faddeeva_scaled_derivative(center, order - 1)
        return value, 1
    end

    homogeneous = Vector{C}(undef, max_terms)
    _multipole_complete_homogeneous!(homogeneous, nodes, center)

    value0, value1 = _faddeeva_scaled_derivative_seed(center)
    target_order = order - 1
    previous, current = value0, value1
    @inbounds for derivative_order in 1:(target_order - 1)
        previous, current = current, _faddeeva_scaled_derivative_next(
            center, previous, current, derivative_order,
        )
    end

    total = zero(C)
    compensation = zero(C)
    small_run = 0
    required_small_run = max(order, 2)
    @inbounds for expansion_order in 0:(max_terms - 1)
        term = current * homogeneous[expansion_order + 1]
        total, compensation = _multipole_compensated_add(total, compensation, term)
        term_magnitude = convert(R, float(_primal_value(abs(term))))
        total_magnitude = convert(R, float(_primal_value(abs(total))))
        threshold = relative_tolerance * max(one(R), total_magnitude)
        small_run = term_magnitude <= threshold ? small_run + 1 : 0
        if expansion_order >= 6 && small_run >= required_small_run
            _number_isfinite(total) || throw(DomainError(
                nodes,
                "multipole cluster expansion produced a non-finite value",
            ))
            return total, expansion_order + 1
        end
        if expansion_order + 1 < max_terms
            derivative_order = target_order + expansion_order
            previous, current = current, _faddeeva_scaled_derivative_next(
                center, previous, current, derivative_order,
            )
            _number_isfinite(current) || throw(DomainError(
                nodes,
                "scaled Faddeeva derivative recurrence became non-finite",
            ))
        end
    end
    throw(ArgumentError(
        "multipole cluster expansion did not converge in max_terms=$max_terms",
    ))
end

@inline function _multipole_validate_positive_finite(value::Real, name::AbstractString)
    primal = _primal_value(value)
    (isfinite(primal) && primal > zero(primal)) || throw(DomainError(
        value,
        "$name must be finite and positive",
    ))
    return value
end

"""
    multipole_transition(nodes; return_info=false, kwargs...)

Evaluate the Faddeeva divided difference for a finite pole cluster. The direct
representation is selected only when the nodes are distinct, their centered
radius exceeds the configured boundary, and their cancellation estimate is
below the configured limit. Otherwise a centered confluent expansion is used.

Keyword defaults are `cluster_radius_threshold=0.075`,
`cancellation_threshold=2e4` for binary64 (reduced for lower precision),
`max_terms=48`, and a type-local relative tolerance. Equality at either
selection boundary uses the cluster representation. Exhausting the bounded
series raises `ArgumentError`. Set `return_info=true` to return
`(value, MultipoleEvaluationInfo)`.

This canonical scalar does not determine geometry-specific residues, pole
sheets, boundary conditions, or complete diffraction-mechanism matching.
Orders `1:7` are supported by the validated contract.
"""
function multipole_transition(
    nodes::AbstractVector{T};
    cluster_radius_threshold::Real=_MULTIPOLE_DEFAULT_RADIUS_THRESHOLD,
    cancellation_threshold::Union{Nothing,Real}=nothing,
    max_terms::Integer=_MULTIPOLE_DEFAULT_MAX_TERMS,
    relative_tolerance::Union{Nothing,Real}=nothing,
    return_info::Bool=false,
) where {T<:Number}
    C, R, order = _validate_multipole_nodes(nodes)
    _multipole_validate_positive_finite(
        cluster_radius_threshold, "cluster_radius_threshold",
    )
    _MULTIPOLE_MIN_TERMS <= max_terms <= _MULTIPOLE_MAX_TERMS || throw(DomainError(
        max_terms,
        "max_terms must lie in $_MULTIPOLE_MIN_TERMS:$_MULTIPOLE_MAX_TERMS",
    ))

    radius_threshold = convert(R, _primal_value(cluster_radius_threshold))
    _multipole_validate_positive_finite(radius_threshold, "cluster_radius_threshold")
    default_condition = min(
        R(_MULTIPOLE_DEFAULT_CANCELLATION_THRESHOLD), inv(sqrt(eps(R))),
    )
    condition_limit = cancellation_threshold === nothing ?
        default_condition : convert(R, _primal_value(cancellation_threshold))
    _multipole_validate_positive_finite(condition_limit, "cancellation_threshold")
    tolerance = relative_tolerance === nothing ?
        4eps(R) : convert(R, _primal_value(relative_tolerance))
    _multipole_validate_positive_finite(tolerance, "relative_tolerance")

    all_equal = _multipole_all_equal(nodes, C)
    center = all_equal ? convert(C, nodes[firstindex(nodes)]) :
             _multipole_center(nodes, C)
    if order == 1
        value = _faddeeva_w(center)
        _number_isfinite(value) || throw(DomainError(
            center, "single-node Faddeeva value is non-finite",
        ))
        info = MultipoleEvaluationInfo(
            :single, 1, center, zero(R), one(R), 1,
        )
        return return_info ? (value, info) : value
    end

    radius = all_equal ? zero(R) : _multipole_radius(nodes, center, R)
    distinct = !all_equal && _multipole_nodes_distinct(nodes, C)
    direct_value = zero(C)
    direct_condition = R(Inf)
    direct_succeeded = false
    scaled_radius_boundary = radius_threshold * max(
        one(R), convert(R, float(_primal_value(abs(center)))),
    )
    # A small radius already selects the confluent representation. Avoid p
    # extra special-function calls unless the caller requested its diagnostic.
    if distinct && (radius > scaled_radius_boundary || return_info)
        direct_value, direct_condition, direct_succeeded =
            _multipole_direct_evaluate(nodes, C, R)
    end
    use_cluster = !distinct || !direct_succeeded ||
                  radius <= scaled_radius_boundary ||
                  direct_condition >= condition_limit
    if use_cluster
        value, terms_used = _multipole_cluster_expansion(
            nodes, center, order, tolerance, Int(max_terms),
        )
        method = :cluster
    else
        value, terms_used = direct_value, 1
        method = :direct
    end

    info = MultipoleEvaluationInfo(
        method, order, center, radius, direct_condition, terms_used,
    )
    return return_info ? (value, info) : value
end
