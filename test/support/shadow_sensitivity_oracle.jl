using QuadGK: quadgk
using SpecialFunctions: gamma

function _shadow_oracle_switch(q::Real; precision::Int=256)
    return setprecision(BigFloat, precision) do
        coordinate = BigFloat(q)
        iszero(coordinate) && return complex(BigFloat(0.5), zero(BigFloat))
        tolerance = BigFloat(2)^(-precision ÷ 2)
        integral, = quadgk(
            t -> exp(-im * t * t), zero(coordinate), coordinate;
            rtol=tolerance, atol=tolerance,
        )
        complex(BigFloat(0.5), zero(BigFloat)) +
        cis(BigFloat(pi) / 4) * integral / sqrt(BigFloat(pi))
    end
end

function _shadow_oracle_phase(q::Real, sign::Int; precision::Int=512)
    return setprecision(BigFloat, precision) do
        coordinate = BigFloat(q)
        exp(sign * im * coordinate * coordinate)
    end
end

function _shadow_oracle_multiplier(
    xi::Real,
    kappa::Real;
    precision::Int=512,
)
    return setprecision(BigFloat, precision) do
        frequency = BigFloat(xi)
        parameter = BigFloat(kappa)
        exp(im * frequency^2 / (4parameter))
    end
end

function _shadow_oracle_scaled_kernel(
    coordinate::Real,
    kappa::Real;
    precision::Int=4096,
)
    return setprecision(BigFloat, precision) do
        value = BigFloat(coordinate)
        parameter = BigFloat(kappa)
        sqrt(parameter) * cis(BigFloat(pi) / 4) *
        exp(-im * parameter * value^2) / sqrt(BigFloat(pi))
    end
end

function _shadow_oracle_switch_tail(q::Real; precision::Int=512)
    return setprecision(BigFloat, precision) do
        coordinate = abs(BigFloat(q))
        inverse_square = inv(coordinate * coordinate)
        total = one(Complex{BigFloat})
        term = total
        previous = abs(term)
        for order in 1:256
            term *= im * BigFloat(2order - 1) * inverse_square / 2
            magnitude = abs(term)
            magnitude >= previous && break
            total += term
            magnitude <= eps(BigFloat) * abs(total) && break
            previous = magnitude
        end
        quarter = cis(BigFloat(pi) / 4)
        exp(-im * coordinate^2) * conj(quarter) * total /
        (2sqrt(BigFloat(pi)) * coordinate)
    end
end

function _shadow_oracle_near_real_switch(
    q::Complex{R};
    precision::Int=512,
) where {R<:AbstractFloat}
    return setprecision(BigFloat, precision) do
        wide_q = complex(BigFloat(real(q)), BigFloat(imag(q)))
        positive = real(wide_q) > 0
        negative_q = positive ? -wide_q : wide_q
        argument = -cispi(BigFloat(0.25)) * negative_q
        inverse_square = inv(argument * argument)
        term = one(argument)
        total = term
        for order in 1:256
            term *= -(2order - 1) * inverse_square / 2
            total += term
            abs(term) <= eps(BigFloat) * abs(total) && break
        end
        tail = exp(-(argument * argument)) * total /
               (2sqrt(BigFloat(pi)) * argument)
        convert(Complex{R}, positive ? one(tail) - tail : tail)
    end
end

function _shadow_oracle_switch_imaginary_derivative(
    q::Complex{R};
    precision::Int=512,
) where {R<:AbstractFloat}
    return setprecision(BigFloat, precision) do
        wide_q = complex(BigFloat(real(q)), BigFloat(imag(q)))
        kernel = cispi(BigFloat(0.25)) * exp(-im * wide_q^2) /
                 sqrt(BigFloat(pi))
        convert(Complex{R}, im * kernel)
    end
end

function _shadow_oracle_switch_derivative(
    q::Complex{R},
    tangent::Complex{R};
    precision::Int=512,
) where {R<:AbstractFloat}
    return setprecision(BigFloat, precision) do
        wide_q = complex(BigFloat(real(q)), BigFloat(imag(q)))
        wide_tangent = complex(
            BigFloat(real(tangent)), BigFloat(imag(tangent)),
        )
        kernel = cispi(BigFloat(0.25)) * exp(-im * wide_q^2) /
                 sqrt(BigFloat(pi))
        convert(Complex{R}, kernel * wide_tangent)
    end
end

function _shadow_oracle_kernel_derivative(
    q::R,
    tangent::R;
    precision::Int=512,
) where {R<:AbstractFloat}
    return setprecision(BigFloat, precision) do
        wide_q = BigFloat(q)
        kernel = cispi(BigFloat(0.25)) * exp(-im * wide_q^2) /
                 sqrt(BigFloat(pi))
        convert(Complex{R}, -2im * wide_q * kernel * BigFloat(tangent))
    end
end

function _shadow_oracle_scaled_kernel_derivative(
    coordinate::R,
    kappa::R,
    tangent::R;
    precision::Int=512,
) where {R<:AbstractFloat}
    return setprecision(BigFloat, precision) do
        wide_coordinate = BigFloat(coordinate)
        wide_kappa = BigFloat(kappa)
        value = sqrt(wide_kappa) * cispi(BigFloat(0.25)) *
                exp(-im * wide_kappa * wide_coordinate^2) /
                sqrt(BigFloat(pi))
        convert(
            Complex{R},
            -2im * wide_kappa * wide_coordinate * value *
            BigFloat(tangent),
        )
    end
end

function _shadow_oracle_truncated_multiplier_derivative(
    xi::R,
    kappa::R,
    terms::Int,
    tangent::R;
    precision::Int=512,
) where {R<:AbstractFloat}
    return setprecision(BigFloat, precision) do
        frequency = BigFloat(xi)
        parameter = BigFloat(kappa)
        phase = frequency^2 / (4parameter)
        term = one(Complex{BigFloat})
        derivative_sum = term
        for order in 1:(terms - 2)
            term *= im * phase / order
            derivative_sum += term
        end
        convert(
            Complex{R},
            im * frequency / (2parameter) * derivative_sum *
            BigFloat(tangent),
        )
    end
end

function _shadow_oracle_multiplier_derivative(
    xi::R,
    kappa::R,
    tangent::R;
    precision::Int=512,
) where {R<:AbstractFloat}
    return setprecision(BigFloat, precision) do
        frequency = BigFloat(xi)
        parameter = BigFloat(kappa)
        multiplier = exp(im * frequency^2 / (4parameter))
        convert(
            Complex{R},
            im * frequency / (2parameter) * multiplier * BigFloat(tangent),
        )
    end
end

function _shadow_oracle_remainder_derivative(
    omega::R,
    kappa::R,
    order::Int,
    spectral_l1::R,
    tangent::R;
    precision::Int=512,
) where {R<:AbstractFloat}
    return setprecision(BigFloat, precision) do
        bandwidth = BigFloat(omega)
        parameter = BigFloat(kappa)
        mass = BigFloat(spectral_l1)
        derivative = 2order * mass * bandwidth^(2order - 1) /
                     ((4parameter)^order * gamma(BigFloat(order + 1)))
        convert(R, derivative * BigFloat(tangent))
    end
end

function _shadow_oracle_truncated_multiplier(
    xi::Real,
    kappa::Real,
    terms::Int;
    precision::Int=512,
)
    return setprecision(BigFloat, precision) do
        phase = BigFloat(xi)^2 / (4BigFloat(kappa))
        term = one(Complex{BigFloat})
        total = term
        for order in 1:(terms - 1)
            term *= im * phase / order
            total += term
        end
        total
    end
end

function _shadow_oracle_pullback(
    f0::Number,
    f1::Number,
    f2::Number,
    f3::Number,
    f4::Number,
    a::Real,
    b::Real,
    c::Real,
    d::Real,
    kappa::Real;
    precision::Int=512,
)
    return setprecision(BigFloat, precision) do
        values = map(
            value -> complex(
                BigFloat(real(value)), BigFloat(imag(value)),
            ),
            (f0, f1, f2, f3, f4),
        )
        derivative = BigFloat(a)
        normalized_second = BigFloat(b) / derivative
        normalized_third = BigFloat(c) / derivative
        normalized_fourth = BigFloat(d) / derivative
        inverse_scale2 = inv(BigFloat(kappa) * derivative^2)
        first_bracket = values[3] - normalized_second * values[2]
        second_bracket = values[5] -
            6normalized_second * values[4] +
            (15normalized_second^2 - 4normalized_third) * values[3] +
            (-15normalized_second^3 +
             10normalized_second * normalized_third - normalized_fourth) *
            values[2]
        first = values[1] - im * inverse_scale2 * first_bracket / 4
        second = first - inverse_scale2^2 * second_bracket / 32
        return (leading=values[1], first=first, second=second)
    end
end

function _shadow_oracle_halfplane_derivatives(
    s::Real,
    kappa::Real;
    precision::Int=256,
    step::Real=1e-5,
)
    return setprecision(BigFloat, precision) do
        coordinate = BigFloat(s)
        parameter = BigFloat(kappa)
        h = BigFloat(step)
        q0 = sqrt(2parameter) * sin(coordinate / 2)
        tolerance = BigFloat(2)^(-precision ÷ 2)
        prefactor = cis(BigFloat(pi) / 4) / sqrt(BigFloat(pi))
        local_value(offset) = begin
            shifted = coordinate + offset * h
            q = sqrt(2parameter) * sin(shifted / 2)
            iszero(offset) && return zero(Complex{BigFloat})
            integral, = quadgk(
                t -> exp(-im * t * t), q0, q;
                rtol=tolerance, atol=tolerance,
            )
            prefactor * integral
        end
        fm2 = local_value(-2)
        fm1 = local_value(-1)
        fp1 = local_value(1)
        fp2 = local_value(2)
        first = (fm2 - 8fm1 + 8fp1 - fp2) / (12h)
        second = (-fp2 + 16fp1 + 16fm1 - fm2) / (12h^2)
        return first, second
    end
end

_shadow_relative_error(value, reference; floor=1e-300) =
    abs(value - reference) / max(abs(reference), floor)

function _shadow_fit_slope(x, y)
    lx = log.(Float64.(x))
    ly = log.(Float64.(y))
    mx = sum(lx) / length(lx)
    my = sum(ly) / length(ly)
    return sum((lx .- mx) .* (ly .- my)) / sum(abs2, lx .- mx)
end

function _shadow_nested_derivative(f, x, order::Int)
    order == 0 && return f(x)
    return ForwardDiff.derivative(
        value -> _shadow_nested_derivative(f, value, order - 1), x,
    )
end
