using FastGaussQuadrature: gausshermite

const HERMITE_CACHE = Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}()
const HERMITE_CACHE_LOCK = ReentrantLock()

function cached_hermite_rule(order::Integer)
    order >= 1 || throw(ArgumentError("Hermite order must be positive"))
    return lock(HERMITE_CACHE_LOCK) do
        get!(HERMITE_CACHE, Int(order)) do
            gausshermite(Int(order))
        end
    end
end

"""
    hermite_rule(order) -> (nodes, weights)

Paper-oracle physicists' Gauss--Hermite rule for
`integral exp(-t^2) f(t) dt`. Returned arrays are defensive copies of a
locked cache.
"""
function hermite_rule(order::Integer)
    nodes, weights = cached_hermite_rule(order)
    return copy(nodes), copy(weights)
end

"""
    canonical_quadrature(k, z, amplitude; n=200)

Paper-oracle Gauss--Hermite evaluation of the scaled canonical integral for
`imag(z)>0`. `amplitude` receives the unscaled coordinate `t=u/sqrt(k)`.
"""
function canonical_quadrature(
    k::Real,
    z::Number,
    amplitude;
    n::Integer=200,
)
    k_value = _positive_real(k, "k")
    z_value = _as_complex(z)
    imag(z_value) > 0 || throw(ArgumentError(
        "quadrature reference requires Im(z)>0",
    ))
    nodes, weights = cached_hermite_rule(n)
    scale = sqrt(k_value)
    total = zero(z_value + k_value)
    @inbounds for index in eachindex(nodes)
        value = amplitude(nodes[index] / scale)
        value isa Number || throw(ArgumentError("amplitude must return a number"))
        total += weights[index] *
                 _as_complex(value, "amplitude value") /
                 (z_value - nodes[index])
    end
    return im / pi * total
end
