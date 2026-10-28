# Independent high-precision oracles. F is rebuilt from a BigFloat complementary
# error function; the edge residual is the unrearranged incident pair, not the
# compensated Float64 rearrangement.

function _erfc_hp(z::Complex{BigFloat}; force_series::Bool=false)
    az = abs(z)
    if force_series || az < 10
        s = z
        term = z
        z2 = z * z
        n = 0
        tol = eps(BigFloat) * 8
        while n < 20000
            n += 1
            term *= -z2 / n
            add = term / (2n + 1)
            s += add
            abs(add) <= tol * max(abs(s), eps(BigFloat)) && n > 8 && break
        end
        return 1 - (2 / sqrt(BigFloat(π))) * s
    end
    real(z) < 0 && return 2 - _erfc_hp(-z)
    z2 = z * z
    inv = 1 / (2 * z2)
    term = one(z)
    series = one(z)
    m = 0
    odd = one(BigFloat)
    while m < 400
        m += 1
        odd *= (2m - 1)
        term *= -inv
        add = odd * term
        series += add
        abs(add) <= eps(BigFloat) * 8 * abs(series) && break
    end
    return exp(-z2) / (z * sqrt(BigFloat(π))) * series
end

function _F_asymptotic_hp(x::BigFloat)
    c = one(Complex{BigFloat})
    series = one(Complex{BigFloat})
    inv = 1 / x
    power = one(BigFloat)
    previous = BigFloat(Inf)
    max_terms = max(128, 2 * precision(BigFloat))
    for m in 1:max_terms
        c *= im * (m - BigFloat(1) / 2)
        power *= inv
        add = c * power
        magnitude = abs(add)
        magnitude >= previous && break
        series += add
        magnitude <= eps(BigFloat) * 8 * abs(series) && m > 4 && break
        previous = magnitude
    end
    return series
end

@inline function _precision_bits(dps::Int)
    dps >= 1 || throw(ArgumentError("dps must be positive"))
    return max(256, ceil(Int, dps * log2(10)) + 32)
end

function _big(x::Real, prec::Int)
    return BigFloat(x; precision=prec)
end

function transition_F_hp(x::Real; dps::Int=80)
    target_precision = _precision_bits(dps)
    return setprecision(BigFloat, target_precision) do
        xb = _big(x, target_precision)
        (xb >= 0 && (isfinite(xb) || isinf(xb))) ||
            throw(DomainError(x, "transition_F_hp requires x >= 0 and finite or +Inf"))
        iszero(xb) && return zero(Complex{BigFloat})
        isinf(xb) && return one(Complex{BigFloat})

        # Optimal truncation of the large-x expansion has an exponentially
        # small remainder O(exp(-x)). Use it only once that remainder lies below
        # the requested decimal precision. At smaller x, evaluate the convergent
        # erf series with enough guard bits for its exp(x)-scale cancellation.
        asymptotic_threshold = BigFloat(dps + 12) * log(BigFloat(10))
        xb >= asymptotic_threshold && return _F_asymptotic_hp(xb)

        guard_bits = ceil(Int, xb / log(BigFloat(2))) + 64
        work_precision = target_precision + guard_bits
        return setprecision(BigFloat, work_precision) do
            xw = BigFloat(xb; precision=work_precision)
            phase = cis(BigFloat(π) / 4)
            z = phase * sqrt(xw)
            sqrt(BigFloat(π) * xw) * phase * exp(z * z) *
                _erfc_hp(z; force_series=true)
        end
    end
end

"""
High-precision unrearranged incident-pair residual (the coefficient oracle).

This is independent of `edge_residual_compensated`: it evaluates the two
cotangent–transition products directly at the working precision.
"""
function edge_residual_hp(g::BoundaryGeometry; dps::Int=80)
    prec = _precision_bits(dps)
    return setprecision(BigFloat, prec) do
        e = _big(g.eps, prec)
        n = 1 + e
        d = _big(g.Delta, prec)
        kk = _big(g.k, prec)
        ll = _big(g.L, prec)
        EIM4b = cis(-BigFloat(π) / 4)
        if iszero(d)
            cotv = cot(BigFloat(π) / (2n))
            return -EIM4b * cotv * transition_F_hp(2 * kk * ll; dps=dps) /
                   (n * sqrt(2 * BigFloat(π) * kk))
        end
        p1 = (BigFloat(π) + d) / (2n)
        p2 = (BigFloat(π) - d) / (2n)
        a0 = 2 * cos(d / 2)^2
        a1 = 2 * cos(BigFloat(π) * e - d / 2)^2
        s = cot(p1) * transition_F_hp(kk * ll * a1; dps=dps) / sqrt(kk)
        s += cot(p2) * transition_F_hp(kk * ll * a0; dps=dps) / sqrt(kk)
        return -EIM4b * s / (2n * sqrt(2 * BigFloat(π)))
    end
end

_nint_hp(x::BigFloat) = x >= 0 ? floor(x + BigFloat(1) / 2) :
                                ceil(x - BigFloat(1) / 2)

"""
High-precision general-angle incident/reflection pair oracle.

Evaluates the unrearranged Kouyoumjian--Pathak terms at arbitrary interior
angles `(phi, phip)` with `psi_{3,4} = (pi +/- S)/(2n)`, `S = phi + phip`
(the off-midpoint sum-angle form; Eq. (8) of the paper is the midpoint
specialization only). Nearest-pole integers use half-away-from-zero ties.
Returns `(Di, Dr)` — the incident pair and the reflection pair — as
`Complex{BigFloat}`. Independent of the compensated identities: every
cotangent-transition product is formed directly.
"""
function wedge_pairs_hp(phi::Real, phip::Real, n::Real, k::Real, L::Real;
                        dps::Int=80)
    prec = _precision_bits(dps)
    return setprecision(BigFloat, prec) do
        pib = BigFloat(π; precision=prec)
        phib = _big(phi, prec)
        phipb = _big(phip, prec)
        nb = _big(n, prec)
        kb = _big(k, prec)
        lb = _big(L, prec)
        delta = phib - phipb
        sigma = phib + phipb
        inc = zero(Complex{BigFloat})
        ref = zero(Complex{BigFloat})
        for (beta, sgn, is_incident) in
            ((delta, 1, true), (delta, -1, true),
             (sigma, 1, false), (sigma, -1, false))
            psi = (pib + sgn * beta) / (2nb)
            branch = _nint_hp((beta + sgn * pib) / (2pib * nb))
            a = 2 * cos((2pib * nb * branch - beta) / 2)^2
            term = cot(psi) * transition_F_hp(kb * lb * a; dps=dps)
            if is_incident
                inc += term
            else
                ref += term
            end
        end
        scale = -cis(-pib / 4) / (2nb * sqrt(2pib * kb))
        return scale * inc, scale * ref
    end
end
