using QuadGK

function _multipole_oracle_w_taylor(
    z::Complex{BigFloat};
    tolerance::BigFloat=eps(BigFloat)^2,
    max_terms::Int=4000,
)
    minus_z_squared = -z * z
    even = one(z)
    even_term = one(z)
    odd = 2im * z / sqrt(BigFloat(pi))
    odd_term = odd
    for index in 1:max_terms
        even_term *= minus_z_squared / index
        odd_term *= minus_z_squared / (index + BigFloat(1) / 2)
        even += even_term
        odd += odd_term
        if abs(even_term) + abs(odd_term) <=
           tolerance * max(one(BigFloat), abs(even) + abs(odd))
            return even + odd
        end
    end
    error("independent Faddeeva Taylor oracle did not converge")
end

function _multipole_oracle_divided_difference(nodes; precision::Int=768)
    return setprecision(BigFloat, precision) do
        values = Complex{BigFloat}[
            complex(BigFloat(real(node)), BigFloat(imag(node))) for node in nodes
        ]
        total = zero(Complex{BigFloat})
        @inbounds for index in eachindex(values)
            denominator = one(Complex{BigFloat})
            for other in eachindex(values)
                index == other && continue
                difference = values[index] - values[other]
                iszero(difference) && error("oracle requires distinct nodes")
                denominator *= difference
            end
            total += _multipole_oracle_w_taylor(values[index]) / denominator
        end
        total
    end
end

function _multipole_oracle_contour(nodes; numerator=(one(ComplexF64),))
    all(node -> imag(node) > 0, nodes) ||
        error("contour oracle requires upper-half-plane nodes")
    integrand(t) = begin
        polynomial = zero(ComplexF64)
        power = one(ComplexF64)
        for coefficient in numerator
            polynomial += coefficient * power
            power *= t
        end
        denominator = one(ComplexF64)
        for node in nodes
            denominator *= t - node
        end
        exp(-t * t) * polynomial / denominator
    end
    integral, estimate = quadgk(integrand, -Inf, Inf; rtol=2e-13, atol=2e-14)
    value = -im / pi * integral
    estimate / pi <= max(2e-14, 2e-13 * abs(value)) ||
        error("independent contour oracle did not meet its tolerance")
    return value
end

function _multipole_oracle_newton_integral(nodes, coefficients)
    polynomial(node) = begin
        total = zero(ComplexF64)
        for coefficient in reverse(coefficients)
            total = total * node + coefficient
        end
        total
    end
    table = ComplexF64[polynomial(node) for node in nodes]
    newton = similar(table)
    newton[1] = table[1]
    @inbounds for level in 1:(length(nodes) - 1)
        for index in length(nodes):-1:(level + 1)
            table[index] = (table[index] - table[index - 1]) /
                           (nodes[index] - nodes[index - level])
        end
        newton[level + 1] = table[level + 1]
    end
    total = zero(ComplexF64)
    @inbounds for index in eachindex(newton)
        total += newton[index] * multipole_transition(view(nodes, index:lastindex(nodes)))
    end
    return total
end
