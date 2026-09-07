using FastGaussQuadrature: gausslegendre
using SpecialFunctions: erfc

bivariate_oracle_switch(x::Number) =
    erfc(-cis(pi / 4) * complex(x) / sqrt(2)) / 2

function bivariate_rotated_oracle(
    xi::Real,
    eta::Real,
    rho::Real;
    order::Integer=180,
    cutoff::Real=14.5,
)
    nodes, weights = gausslegendre(order)
    sine_square = 1 - rho^2
    a11 = inv(sine_square)
    a12 = -rho / sine_square
    phase_boundary = a11 * xi^2 + 2a12 * xi * eta + a11 * eta^2
    coordinates = cutoff .* (nodes .+ 1) ./ 2
    scaled_weights = cutoff .* weights ./ 2
    rotation = im * cis(-pi / 4)
    total = 0.0 + 0.0im
    for first_index in eachindex(coordinates)
        x = coordinates[first_index]
        wx = scaled_weights[first_index]
        for second_index in eachindex(coordinates)
            y = coordinates[second_index]
            quadratic = a11 * x^2 + 2a12 * x * y + a11 * y^2
            linear = (a11 * x + a12 * y) * xi +
                     (a12 * x + a11 * y) * eta
            total += wx * scaled_weights[second_index] *
                     exp(-quadratic / 2 + rotation * linear)
        end
    end
    return exp(-im * phase_boundary / 2) * total / (2pi * sqrt(sine_square))
end
