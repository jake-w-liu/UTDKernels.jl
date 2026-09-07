"""Shared bounded, thread-safe Gauss--Legendre rules."""

const _MAX_SHARED_GAUSS_LEGENDRE_ORDER = 2048
const _GL_CACHE = Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}()
const _GL_CACHE_LOCK = ReentrantLock()

@inline function _validate_shared_gauss_legendre_order(
    order::Integer,
    maximum_order::Integer,
)
    1 <= maximum_order <= _MAX_SHARED_GAUSS_LEGENDRE_ORDER || throw(ArgumentError(
        "maximum Gauss--Legendre order must lie in " *
        "1:$(_MAX_SHARED_GAUSS_LEGENDRE_ORDER)",
    ))
    value = try
        Int(order)
    catch error
        (error isa InexactError || error isa OverflowError) || rethrow()
        throw(ArgumentError("Gauss--Legendre order is not representable as Int"))
    end
    1 <= value <= maximum_order || throw(ArgumentError(
        "Gauss--Legendre order must lie in 1:$maximum_order",
    ))
    return value
end

function _construct_gauss_legendre_rule(order::Int)
    if order == 1
        return [0.0], [2.0]
    end

    # Symmetric Newton construction uses O(order) retained and temporary
    # memory, unlike a full Golub--Welsch eigenvector matrix. The asymptotic
    # root guesses converge quadratically; evaluating the Legendre recurrence
    # at the final root also gives the weights without another dependency.
    nodes = Vector{Float64}(undef, order)
    weights = Vector{Float64}(undef, order)
    half_count = (order + 1) ÷ 2
    tolerance = 4eps(Float64)
    for index in 1:half_count
        root = cos(pi * (index - 0.25) / (order + 0.5))
        derivative = 0.0
        converged = false
        for _ in 1:20
            polynomial = 1.0
            previous = 0.0
            for degree in 1:order
                older = previous
                previous = polynomial
                polynomial = (
                    (2degree - 1) * root * previous - (degree - 1) * older
                ) / degree
            end
            derivative = order * (root * polynomial - previous) / (root^2 - 1)
            updated = root - polynomial / derivative
            if abs(updated - root) <= tolerance * max(1.0, abs(updated))
                root = updated
                converged = true
                break
            end
            root = updated
        end
        converged || error("Gauss--Legendre root iteration did not converge")

        # Re-evaluate P_n and P_{n-1} at the accepted root before forming P'_n.
        polynomial = 1.0
        previous = 0.0
        for degree in 1:order
            older = previous
            previous = polynomial
            polynomial = (
                (2degree - 1) * root * previous - (degree - 1) * older
            ) / degree
        end
        derivative = order * (root * polynomial - previous) / (root^2 - 1)
        weight = 2 / ((1 - root^2) * derivative^2)
        mirror = order + 1 - index
        if index == mirror
            nodes[index] = 0.0
        else
            nodes[index] = -root
            nodes[mirror] = root
        end
        weights[index] = weight
        weights[mirror] = weight
    end
    return nodes, weights
end

function _cached_gauss_legendre_rule(
    order::Integer;
    maximum_order::Integer=_MAX_SHARED_GAUSS_LEGENDRE_ORDER,
)
    value = _validate_shared_gauss_legendre_order(order, maximum_order)
    lock(_GL_CACHE_LOCK)
    try
        cached = get(_GL_CACHE, value, nothing)
        cached !== nothing && return cached
        rule = _construct_gauss_legendre_rule(value)
        _GL_CACHE[value] = rule
        return rule
    finally
        unlock(_GL_CACHE_LOCK)
    end
end

function _copy_gauss_legendre_rule(
    order::Integer;
    maximum_order::Integer=_MAX_SHARED_GAUSS_LEGENDRE_ORDER,
)
    nodes, weights = _cached_gauss_legendre_rule(order; maximum_order)
    return copy(nodes), copy(weights)
end
