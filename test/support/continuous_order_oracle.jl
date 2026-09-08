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

function _continuous_oracle_hypergeometric_big(
    mu::Real,
    zeta::Number;
    precision::Int=768,
)
    return setprecision(BigFloat, precision) do
        m = BigFloat(mu)
        z = complex(BigFloat(real(zeta)), BigFloat(imag(zeta)))
        _₁F₁(m / 2, BigFloat(0.5), z * z) +
        2z * gamma((m + 1) / 2) / gamma(m / 2) *
        _₁F₁((m + 1) / 2, BigFloat(1.5), z * z)
    end
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

function _continuous_oracle_scaled_negative_derivative(
    mu::Real,
    x::Real,
    tangent::Real;
    precision::Int=512,
)
    return setprecision(BigFloat, precision) do
        order = BigFloat(mu)
        endpoint = BigFloat(x)
        integrand = function (t)
            isone(t) && return zero(BigFloat)
            y = t / (1 - t)
            2(endpoint + y / endpoint) * y^(order - 1) *
                exp(-2y - (y / endpoint)^2) / (1 - t)^2
        end
        integral, = quadgk(
            integrand, zero(BigFloat), one(BigFloat);
            rtol=BigFloat(2)^(-128), order=21,
        )
        derivative = 2 / gamma(order / 2) * endpoint^(-order) *
                     exp(-endpoint^2) * integral
        convert(typeof(float(tangent)), derivative * BigFloat(tangent))
    end
end

function _continuous_oracle_order_sensitivity_coordinate_derivative(
    mu::Float64,
    x::Float64,
    tangent::Float64;
    precision::Int=768,
)
    return setprecision(BigFloat, precision) do
        order = BigFloat(mu)
        endpoint = BigFloat(x)
        digamma_factor = digamma(order / 2) / 2
        integrand = function (t)
            isone(t) && return zero(BigFloat)
            y = t / (1 - t)
            y^order * (log(y) - log(2endpoint) - digamma_factor) *
                exp(-y - y^2 / (4endpoint^2)) / (1 - t)^2
        end
        integral, = quadgk(
            integrand, zero(BigFloat), one(BigFloat);
            rtol=big"1e-60", order=21,
        )
        derivative = 4integral /
                     (gamma(order / 2) * (2endpoint)^(order + 1))
        convert(Float64, derivative * BigFloat(tangent))
    end
end

function _continuous_oracle_negative_coordinate_derivative(
    mu::Float64,
    x::Float64,
    tangent::Float64;
    precision::Int=768,
)
    return setprecision(BigFloat, precision) do
        order = BigFloat(mu)
        endpoint = BigFloat(x)
        integrand = function (t)
            isone(t) && return zero(BigFloat)
            y = t / (1 - t)
            y^order * exp(-y - y^2 / (4endpoint^2)) / (1 - t)^2
        end
        integral, = quadgk(
            integrand, zero(BigFloat), one(BigFloat);
            rtol=big"1e-60", order=21,
        )
        derivative = 4integral /
                     (gamma(order / 2) * (2endpoint)^(order + 1))
        convert(Float64, derivative * BigFloat(tangent))
    end
end

function _continuous_oracle_unit_negative_derivative(
    x::Float64,
    order::Int;
    precision::Int=2048,
)
    order >= 0 || throw(DomainError(order, "derivative order must be nonnegative"))
    return setprecision(BigFloat, precision) do
        coordinate = BigFloat(x)
        previous = erfcx(coordinate)
        order == 0 && return Float64(previous)
        current = 2coordinate * previous - 2 / sqrt(BigFloat(pi))
        for index in 1:(order - 1)
            previous, current = current,
                2coordinate * current + 2index * previous
        end
        Float64(isodd(order) ? -current : current)
    end
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
