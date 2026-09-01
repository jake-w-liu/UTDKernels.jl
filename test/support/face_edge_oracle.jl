# Independent arbitrary-precision oracle for the paper-162 reflection-boundary
# incident pair. This deliberately evaluates the unrearranged cotangent terms
# and reconstructs erfc rather than calling any UTDKernels transition helper.

function _face_edge_erfc_hp(z::Complex{BigFloat}; force_series::Bool=false)
    if force_series || abs(z) < 10
        total = z
        term = z
        z2 = z * z
        tolerance = 8eps(BigFloat)
        for order in 1:20_000
            term *= -z2 / order
            increment = term / (2order + 1)
            total += increment
            abs(increment) <= tolerance * max(abs(total), eps(BigFloat)) && order > 8 &&
                return 1 - (2 / sqrt(BigFloat(pi))) * total
        end
        error("high-precision erfc series did not converge")
    end
    real(z) < 0 && return 2 - _face_edge_erfc_hp(-z)

    z2 = z * z
    inverse = inv(2z2)
    term = one(z)
    total = one(z)
    odd = one(BigFloat)
    previous = BigFloat(Inf)
    for order in 1:400
        odd *= 2order - 1
        term *= -inverse
        increment = odd * term
        magnitude = abs(increment)
        magnitude >= previous && break
        total += increment
        magnitude <= 8eps(BigFloat) * abs(total) && break
        previous = magnitude
    end
    return exp(-z2) * total / (z * sqrt(BigFloat(pi)))
end

function _face_edge_F_asymptotic_hp(x::BigFloat)
    coefficient = one(Complex{BigFloat})
    total = one(Complex{BigFloat})
    inverse = inv(x)
    power = one(BigFloat)
    previous = BigFloat(Inf)
    for order in 1:max(128, 2precision(BigFloat))
        coefficient *= im * (order - BigFloat(1) / 2)
        power *= inverse
        increment = coefficient * power
        magnitude = abs(increment)
        magnitude >= previous && break
        total += increment
        magnitude <= 8eps(BigFloat) * abs(total) && order > 4 && break
        previous = magnitude
    end
    return total
end

function face_edge_F_hp(x::Real; digits::Int=100)
    digits > 0 || throw(ArgumentError("digits must be positive"))
    bits = max(256, ceil(Int, digits * log2(10)) + 32)
    return setprecision(BigFloat, bits) do
        xb = BigFloat(x)
        xb >= 0 || throw(DomainError(x))
        iszero(xb) && return zero(Complex{BigFloat})
        isinf(xb) && return one(Complex{BigFloat})
        threshold = BigFloat(digits + 12) * log(BigFloat(10))
        xb >= threshold && return _face_edge_F_asymptotic_hp(xb)

        guard_bits = ceil(Int, xb / log(BigFloat(2))) + 64
        setprecision(BigFloat, bits + guard_bits) do
            xw = BigFloat(xb)
            phase = cis(BigFloat(pi) / 4)
            z = phase * sqrt(xw)
            sqrt(BigFloat(pi) * xw) * phase * exp(z * z) *
                _face_edge_erfc_hp(z; force_series=true)
        end
    end
end

function face_edge_incident_hp(epsilon::Real, delta::Real, k::Real, L::Real;
                               digits::Int=100)
    bits = max(256, ceil(Int, digits * log2(10)) + 32)
    return setprecision(BigFloat, bits) do
        e = BigFloat(epsilon)
        n = 1 + e
        d = BigFloat(delta)
        kk = BigFloat(k)
        ll = BigFloat(L)
        phase = cis(-BigFloat(pi) / 4)
        if iszero(d)
            return -phase * cot(BigFloat(pi) / (2n)) *
                   face_edge_F_hp(2kk * ll; digits) /
                   (n * sqrt(2BigFloat(pi) * kk))
        end
        psi1 = (BigFloat(pi) + d) / (2n)
        psi2 = (BigFloat(pi) - d) / (2n)
        a0 = 2cos(d / 2)^2
        a1 = 2cos(BigFloat(pi) * e - d / 2)^2
        incident = cot(psi1) * face_edge_F_hp(kk * ll * a1; digits) / sqrt(kk)
        incident += cot(psi2) * face_edge_F_hp(kk * ll * a0; digits) / sqrt(kk)
        return -phase * incident / (2n * sqrt(2BigFloat(pi)))
    end
end

function face_edge_direct_hp(epsilon::Real, delta::Real, k::Real, L::Real;
                             digits::Int=100)
    ComplexF64(face_edge_incident_hp(epsilon, delta, k, L; digits))
end
