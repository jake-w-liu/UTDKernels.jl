using QuadGK: quadgk

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
