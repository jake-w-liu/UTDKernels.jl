using FastGaussQuadrature: gausshermite
using QuadGK

function _null_oracle_gaussian_moment(order::Integer, ::Type{T}) where {T<:AbstractFloat}
    isodd(order) && return zero(T)
    value = sqrt(T(pi))
    for index in 1:(order ÷ 2)
        value *= T(2index - 1) / 2
    end
    return value
end

function _null_oracle_erf_series(z::Complex{BigFloat})
    squared = z * z
    term = one(z)
    total = one(z)
    for index in 1:(8precision(BigFloat))
        term *= -squared / index
        addition = term / (2index + 1)
        total += addition
        abs(addition) <= eps(BigFloat) * (abs(total) + eps(BigFloat)) && break
    end
    return 2z * total / sqrt(BigFloat(pi))
end

function _null_oracle_faddeeva(z::Complex{BigFloat}; target_precision=512)
    return setprecision(BigFloat, target_precision) do
        guard = ceil(Int, abs2(z) / log(BigFloat(2))) + 96
        value = setprecision(BigFloat, target_precision + guard) do
            local_z = Complex{BigFloat}(z)
            exp(-local_z * local_z) * (1 - _null_oracle_erf_series(-im * local_z))
        end
        Complex{BigFloat}(value)
    end
end

function _null_oracle_moment(z, order; precision=512)
    return setprecision(BigFloat, precision) do
        value_z = complex(BigFloat(real(z)), BigFloat(imag(z)))
        value = _null_oracle_faddeeva(value_z; target_precision=precision)
        for index in 0:(order - 1)
            value = value_z * value -
                    im * _null_oracle_gaussian_moment(index, BigFloat) / BigFloat(pi)
        end
        value
    end
end

function _null_oracle_asymptotic(z, order; precision=512, max_terms=500)
    return setprecision(BigFloat, precision) do
        value_z = complex(BigFloat(real(z)), BigFloat(imag(z)))
        inverse = inv(value_z)
        power = inverse
        total = zero(value_z)
        best = total
        best_magnitude = BigFloat(Inf)
        nonzero_terms = 0
        for offset in 0:(max_terms - 1)
            moment = _null_oracle_gaussian_moment(order + offset, BigFloat)
            if !iszero(moment)
                term = im * moment * power / BigFloat(pi)
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
            power *= inverse
        end
        imag(value_z) < 0 && (best += 2value_z^order * exp(-value_z^2))
        best
    end
end

function _null_oracle_direct_moment(z, order)
    imag(z) > 0 || error("direct moment oracle requires Im(z)>0")
    integrand(value) = value^order * exp(-value * value) / (z - value)
    integral, estimate = quadgk(
        integrand, -Inf, 0.0, Inf; rtol=1e-13, atol=pi * 1e-13,
    )
    result = im / pi * integral
    estimate / pi <= max(1e-13, 1e-13 * abs(result)) ||
        error("direct moment oracle did not meet its tolerance")
    return result
end

function _null_oracle_polynomial_integral(k, z, coefficients; order=800)
    nodes, weights = gausshermite(order)
    inverse_scale = inv(sqrt(k))
    total = zero(ComplexF64)
    @inbounds for index in eachindex(nodes)
        coordinate = nodes[index] * inverse_scale
        amplitude = zero(ComplexF64)
        for coefficient in reverse(coefficients)
            amplitude = amplitude * coordinate + coefficient
        end
        total += weights[index] * amplitude / (z - nodes[index])
    end
    return im / pi * total
end
