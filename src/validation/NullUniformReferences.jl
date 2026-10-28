function _erf_series(z::Complex{BigFloat})
    z2 = z * z
    term = one(z)
    acc = one(z)
    n = 0
    maxterms = 8 * precision(BigFloat)
    while n < maxterms
        n += 1
        term *= -z2 / n
        add = term / (2n + 1)
        acc += add
        abs(add) <= eps(BigFloat) * (abs(acc) + eps(BigFloat)) && break
    end
    return (2 / sqrt(BigFloat(π))) * z * acc
end

function _moment_asymptotic_big(
    z::Complex{BigFloat},
    m::Integer;
    max_terms::Integer=max(120, 2 * precision(BigFloat)),
)
    iszero(z) && throw(ArgumentError("large-z asymptotic is undefined at z=0"))
    max_terms >= 1 || throw(ArgumentError("max_terms must be positive"))
    initial_m = m
    total = zero(z)
    best = zero(z)
    best_mag = BigFloat(Inf)
    nonzero_terms = 0
    inv_z = 1 / z
    power = inv_z
    mm = Int(m)
    for _ in 1:max_terms
        mu = gaussian_moment(mm, BigFloat)
        if !iszero(mu)
            term = im * mu * power / BigFloat(π)
            total += term
            mag = abs(term)
            nonzero_terms += 1
            if mag < best_mag
                best_mag = mag
                best = total
            elseif nonzero_terms >= 5 && mag > best_mag
                break
            end
        end
        mm += 1
        power *= inv_z
    end
    if imag(z) < 0
        best += 2 * z^initial_m * exp(-z * z)
    end
    return best
end

function _faddeeva_big(zz::Complex{BigFloat}, target_precision::Int)
    # Least-term asymptotics omit an exponentially small Stokes contribution.
    # Use them only when exp(-|z|^2) is below the requested bit precision;
    # otherwise evaluate the entire defining identity with enough guard bits to
    # recover cancellation in 1-erf(-im*z).
    asymptotic_threshold = BigFloat(target_precision + 32) * log(BigFloat(2))
    if abs2(zz) >= asymptotic_threshold
        return _moment_asymptotic_big(
            zz, 0; max_terms=max(120, 2 * target_precision),
        )
    end

    guard_bits = ceil(Int, abs2(zz) / log(BigFloat(2))) + 64
    work_precision = target_precision + guard_bits
    value = setprecision(BigFloat, work_precision) do
        zw = Complex{BigFloat}(
            BigFloat(real(zz); precision=work_precision),
            BigFloat(imag(zz); precision=work_precision),
        )
        exp(-zw * zw) * (1 - _erf_series(-im * zw))
    end
    return Complex{BigFloat}(
        BigFloat(real(value); precision=target_precision),
        BigFloat(imag(value); precision=target_precision),
    )
end

"""
    faddeeva_hp(z; precision=384)

High-precision Faddeeva function. SpecialFunctions.erfcx has no
`Complex{BigFloat}` method, so this evaluator combines a guarded, convergent
error-function series with a least-term expansion used only after its
exponentially small remainder falls below the requested precision.
"""
function faddeeva_hp(z::Number; precision::Integer=384)
    precision >= 64 || throw(ArgumentError("precision must be at least 64 bits"))
    isfinite(real(z)) && isfinite(imag(z)) || throw(ArgumentError("z must be finite"))
    return setprecision(BigFloat, precision) do
        zz = Complex{BigFloat}(real(z), imag(z))
        _faddeeva_big(zz, Int(precision))
    end
end

"""
    hp_moment(z, m; precision=384) -> W_m

High-precision recurrence oracle, analog of the mpmath recurrence.
"""
function hp_moment(z::Number, m::Integer; precision::Integer=384)
    m < 0 && throw(ArgumentError("m must be nonnegative"))
    precision >= 64 || throw(ArgumentError("precision must be at least 64 bits"))
    isfinite(real(z)) && isfinite(imag(z)) || throw(ArgumentError("z must be finite"))
    return setprecision(BigFloat, precision) do
        zz = Complex{BigFloat}(real(z), imag(z))
        value = _faddeeva_big(zz, Int(precision))
        @inbounds for j in 0:(m - 1)
            mu = gaussian_moment(j, BigFloat)
            value = zz * value - im * mu / BigFloat(π)
        end
        return value
    end
end

"""
    hp_moment_asymptotic(z, m; precision=384) -> W_m

Independent BigFloat least-term inverse-power evaluation of `W_m`.
"""
function hp_moment_asymptotic(z::Number, m::Integer; precision::Integer=384)
    m < 0 && throw(ArgumentError("m must be nonnegative"))
    precision >= 64 || throw(ArgumentError("precision must be at least 64 bits"))
    isfinite(real(z)) && isfinite(imag(z)) || throw(ArgumentError("z must be finite"))
    return setprecision(BigFloat, precision) do
        zz = Complex{BigFloat}(real(z), imag(z))
        return _moment_asymptotic_big(zz, m)
    end
end

"""
    direct_moment(z, m; rtol=1e-14, atol=1e-14) -> W_m

Independent real-line QuadGK of the defining moment integral. Requires
`Im(z) > 0`.
"""
function _quadrature_tolerances(rtol::Real, atol::Real)
    rf = Float64(rtol)
    af = Float64(atol)
    isfinite(rf) && rf >= 0 ||
        throw(ArgumentError("rtol must be finite and nonnegative"))
    isfinite(af) && af >= 0 ||
        throw(ArgumentError("atol must be finite and nonnegative"))
    rf > 0 || af > 0 || throw(ArgumentError("rtol and atol cannot both be zero"))
    return rf, af
end

function direct_moment(z::Number, m::Integer; rtol=1e-14, atol=1e-14)
    m < 0 && throw(ArgumentError("m must be nonnegative"))
    zz = _as_complex(z)
    imag(zz) <= 0 && throw(ArgumentError("Im(z) must be positive"))
    rf, af = _quadrature_tolerances(rtol, atol)
    f = t -> (t^m) * exp(-t * t) / (zz - t)
    val, err = quadgk(f, -Inf, 0.0, Inf; rtol=rf, atol=π * af)
    result = (im / π) * val
    estimated_error = Float64(err) / π
    target = max(af, rf * abs(result))
    estimated_error <= target || throw(DomainError(
        estimated_error, "direct moment quadrature did not meet its requested tolerance"))
    return result
end

"""
    hp_direct_moment(z, m; precision=256, rtol=1e-20) -> W_m

BigFloat real-line quadrature of the defining integral (mpmath analog).
"""
function hp_direct_moment(z::Number, m::Integer; precision::Integer=256,
                         rtol=1e-20, atol=1e-20)
    m < 0 && throw(ArgumentError("m must be nonnegative"))
    imag(complex(z)) <= 0 && throw(ArgumentError("Im(z) must be positive"))
    precision >= 128 || throw(ArgumentError("precision must be at least 128 bits"))
    isfinite(rtol) && rtol >= 0 ||
        throw(ArgumentError("rtol must be finite and nonnegative"))
    isfinite(atol) && atol >= 0 ||
        throw(ArgumentError("atol must be finite and nonnegative"))
    rtol > 0 || atol > 0 || throw(ArgumentError("rtol and atol cannot both be zero"))
    return setprecision(BigFloat, precision) do
        zz = Complex{BigFloat}(real(z), imag(z))
        f = t -> (t^m) * exp(-t * t) / (zz - t)
        rb = BigFloat(rtol)
        ab = BigFloat(atol)
        val, err = quadgk(f, BigFloat(-Inf), BigFloat(0), BigFloat(Inf);
                          rtol=rb, atol=BigFloat(π) * ab)
        result = (im / BigFloat(π)) * val
        estimated_error = err / BigFloat(π)
        target = max(ab, rb * abs(result))
        estimated_error <= target || throw(DomainError(
            estimated_error,
            "high-precision direct moment quadrature did not meet its requested tolerance"))
        return result
    end
end
