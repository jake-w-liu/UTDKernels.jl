using HypergeometricFunctions: _₁F₁
using QuadGK: quadgk
using SpecialFunctions: gamma

function _continuous_oracle_big(mu::Real, zeta::Number; precision::Int=256)
    return setprecision(BigFloat, precision) do
        m = BigFloat(mu)
        z = complex(BigFloat(real(zeta)), BigFloat(imag(zeta)))
        scale = max(one(m), -2real(z))
        tolerance = BigFloat(2)^(-precision ÷ 2)
        near = x -> begin
            u = x^(inv(m)) / scale
            exp(-u * u + 2z * u) / m
        end
        tail = v -> begin
            isinf(v) && return zero(z)
            u = v / scale
            v^(m - 1) * exp(-u * u + 2z * u)
        end
        first, = quadgk(
            near, zero(m), one(m); rtol=tolerance, atol=tolerance,
        )
        rest, = quadgk(
            tail, one(m), BigFloat(Inf); rtol=tolerance, atol=tolerance,
        )
        2scale^(-m) * (first + rest) / gamma(m / 2)
    end
end

function _continuous_oracle_hypergeometric(mu::Real, zeta::Number)
    m = Float64(mu)
    z = ComplexF64(zeta)
    return _₁F₁(m / 2, 0.5, z * z) +
           2z * gamma((m + 1) / 2) / gamma(m / 2) *
           _₁F₁((m + 1) / 2, 1.5, z * z)
end

function _continuous_oracle_horner(coefficients, x)
    value = zero(ComplexF64)
    for coefficient in Iterators.reverse(coefficients)
        value = muladd(value, x, ComplexF64(coefficient))
    end
    return value
end

function _continuous_oracle_scaled_integral(
    nu::Float64,
    zeta::ComplexF64,
    amplitude;
    cubic::ComplexF64=zero(ComplexF64),
    rtol::Float64=2e-13,
    atol::Float64=2e-15,
)
    scale = max(1.0, -2real(zeta))
    near = x -> begin
        u = x^(1 / nu) / scale
        amplitude(u) * exp(-u * u + 2zeta * u + cubic * u^3) / nu
    end
    tail = v -> begin
        isfinite(v) || return zero(ComplexF64)
        u = v / scale
        v^(nu - 1) * amplitude(u) *
            exp(-u * u + 2zeta * u + cubic * u^3)
    end
    first, = quadgk(near, 0.0, 1.0; rtol=rtol, atol=atol)
    rest, = quadgk(tail, 1.0, Inf; rtol=rtol, atol=atol)
    return scale^(-nu) * (first + rest)
end

function _continuous_oracle_polynomial(
    nu::Real,
    coefficients,
    k::Real,
    h::Real,
    tau::Real,
)
    n = Float64(nu)
    kf = Float64(k)
    hf = Float64(h)
    zeta = cispi(0.25) * Float64(tau) * sqrt(kf / (2hf))
    xscale = cispi(-0.25) * sqrt(2 / (kf * hf))
    amplitude = u -> _continuous_oracle_horner(coefficients, xscale * u)
    integral = _continuous_oracle_scaled_integral(n, zeta, amplitude)
    return cispi(-n / 4) * (2 / (kf * hf))^(n / 2) * integral
end

function _continuous_oracle_cubic(
    nu::Real,
    k::Real,
    h::Real,
    tau::Real,
    cubic_coefficient::Real,
)
    n = Float64(nu)
    kf = Float64(k)
    hf = Float64(h)
    zeta = cispi(0.25) * Float64(tau) * sqrt(kf / (2hf))
    epsilon = Float64(cubic_coefficient) * 2.0^(3 / 2) * cispi(0.75) /
              (6hf^(3 / 2) * sqrt(kf))
    integral = _continuous_oracle_scaled_integral(
        n, zeta, _ -> one(ComplexF64); cubic=epsilon,
    )
    return cispi(-n / 4) * (2 / (kf * hf))^(n / 2) * integral
end

_continuous_relative_error(value, reference; floor=1e-300) =
    abs(value - reference) / max(abs(reference), floor)

function _continuous_log_slope(x, y)
    lx = log.(Float64.(x))
    ly = log.(Float64.(y))
    mx = sum(lx) / length(lx)
    my = sum(ly) / length(ly)
    return sum((lx .- mx) .* (ly .- my)) / sum(abs2, lx .- mx)
end
