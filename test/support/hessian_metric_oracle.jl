using FastGaussQuadrature: gausshermite
using LinearAlgebra
using QuadGK

function _hessian_oracle_random_spd(rng, dimension; condition_max=100.0)
    orthogonal, _ = qr(randn(rng, dimension, dimension))
    values = exp.(rand(rng, dimension) .* log(condition_max))
    matrix = Matrix(orthogonal)
    return matrix * Diagonal(values) * matrix'
end

function _hessian_oracle_transform(H, g, A)
    return A' * H * A, A' * g
end

function _hessian_oracle_faddeeva_quadgk(zeta)
    imag(zeta) > 0 || error("oracle requires an upper-half-plane pole")
    integrand(value) = exp(-value * value) / (zeta - value)
    integral, estimate = quadgk(
        integrand, -Inf, Inf; rtol=1e-13, atol=pi * 1e-13,
    )
    result = im / pi * integral
    estimate / pi <= max(1e-13, 1e-13 * abs(result)) ||
        error("Faddeeva contour oracle did not meet its tolerance")
    return result
end

function _hessian_oracle_gh(H, g, zeta; order=42)
    dimension = size(H, 1)
    dimension in (2, 3) || error("oracle supports dimensions two and three")
    factor = cholesky(Hermitian(Matrix{Float64}(H)))
    transformed = factor.L \ Vector{Float64}(g)
    dual_norm = norm(transformed)
    nodes, weights = gausshermite(order)
    total = zero(ComplexF64)
    if dimension == 2
        @inbounds for first in eachindex(nodes), second in eachindex(nodes)
            denominator = zeta * dual_norm -
                transformed[1] * nodes[first] - transformed[2] * nodes[second]
            total += weights[first] * weights[second] / denominator
        end
    else
        @inbounds for first in eachindex(nodes), second in eachindex(nodes),
                      third in eachindex(nodes)
            denominator = zeta * dual_norm -
                transformed[1] * nodes[first] - transformed[2] * nodes[second] -
                transformed[3] * nodes[third]
            total += weights[first] * weights[second] * weights[third] / denominator
        end
    end
    return im * dual_norm * total / pi^((dimension + 1) / 2)
end
