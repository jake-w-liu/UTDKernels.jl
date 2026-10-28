# Shared definitions for the paper evidence scripts: the production-method
# wrappers (Geom, direct/four-term/linear/fixed-order evaluations) and the
# independent high-precision oracle used to score them. Included by
# generate_paper_data.jl, evaluator_validation.jl, cost_accuracy.jl, and
# sensitivity_sweep.jl so every experiment cites the same code path.
#
# Coefficient references use an independent BigFloat implementation of the
# transition function: a convergent erfc series below x = 100 and an optimally
# truncated asymptotic expansion above it (no fixed-order truncation). Working
# precision is _prec(dps) = max(320, ceil(dps*log2(10))+32) bits. The minimum
# supplies about 96 decimal digits. Near x = 100 the series loses about 44
# digits; the observed maximum coefficient condition number loses another
# 18. This budget leaves more than 30 digits before asymptotic truncation.
# Coefficient accuracy is checked independently, not inferred from this
# budget alone. The asymptotic branch carries a rigorous bound
# |R_M| <= (1/2)_{M+1}/x^{M+1} (< 1e-40 throughout x >= 100), which bounds
# series truncation only, not total floating-point error. A coefficient-level
# convergence check raises requested decimal accuracy from 60 to 120 digits
# (320 to 431 working bits) in run_precision_convergence().
# It does not call UTDKernels.F_utd. A separate nonoscillatory
# Laplace-contour quadrature provides a second independent F/F' oracle.
# Every reference is evaluated at the exact binary64 inputs passed to
# UTDKernels; BigFloat(x) is exact, whereas parsing string(x) would
# substitute the shortest decimal representation and perturb each input by up
# to half an ulp, which is amplified near cotangent poles.

using UTDKernels
using UTDKernels: two_term_kernel, two_term_kernel_derivative, pec_wedge_prefactor,
    pec_wedge_DsDh, pec_wedge_DsDh_grazing, pec_wedge_DsDh_grazing_info,
    pec_wedge_Ds_linear, wedge_n, grazing_interval_report
using QuadGK: quadgk
using Random

if !isdefined(@__MODULE__, :PaperReproductionPaths)
    include(joinpath(@__DIR__, "..", "..", "common.jl"))
end
using .PaperReproductionPaths: selected_backend
if selected_backend() === :diffutd
    include(joinpath(@__DIR__, "backend_diffutd.jl"))
    const Production = DiffGrazingStudy
else
    const Production = UTDKernels
end

# ------------------------------------------------------------------ geometry
struct Geom
    alpha::Float64
    phi::Float64
    k::Float64
    L::Float64
end
_wedge(g::Geom) = Wedge(g.alpha)
electrical_distance(g::Geom) = g.k * g.L   # +Inf in the infinite-distance limit

# ------------------------------------------------------------ method (UTDKernels)
prefactor(g::Geom) = pec_wedge_prefactor(g.k, wedge_n(_wedge(g)))
kern(g::Geom, beta::Real) = two_term_kernel(beta, _wedge(g), g.k, g.L)
kernderiv(g::Geom, beta::Real) = two_term_kernel_derivative(beta, _wedge(g), g.k, g.L)

# Textbook direct subtraction baseline: Ds = C[G(phi-h) - G(phi+h)].
direct_soft(g::Geom, h::Real) = selected_backend() === :diffutd ?
    Production.direct_soft(_wedge(g), RayAngles(g.phi, h), g.k, g.L) :
    prefactor(g) * (kern(g, g.phi - h) - kern(g, g.phi + h))
# Matched branch-safe software baseline: the four-term KP evaluator.
fourterm_soft(g::Geom, h::Real) = Production.pec_wedge_DsDh(_wedge(g), RayAngles(g.phi, h), g.k, g.L)[1]
# Leading linear limit.
linear_soft(g::Geom, h::Real) = Production.pec_wedge_Ds_linear(_wedge(g), RayAngles(g.phi, h), g.k, g.L; face=:o)

function report(g::Geom, h::Real; transition_margin::Real=1e-3, x_margin::Real=1e-8)
    return grazing_interval_report(
        _wedge(g), RayAngles(g.phi, h), g.k, g.L;
        face=:o, transition_margin, x_margin,
        allow_interior=true, allow_infinite_L=true,
    )
end

# Fixed-order form of the continuation, used to measure the order-dependent
# error. The public package entry point treats `order` as an adaptive starting
# order, so the experiment assembles the stated m-point rule from public G'
# values. The rule comes from the package's cached Gauss-Legendre table so
# node/weight generation is not part of any timed evaluation path.
function fixed_order_soft(
    g::Geom,
    h::Real;
    order::Int=8,
    transition_margin::Real=1e-3,
    x_margin::Real=1e-8,
    check_domain::Bool=true,
)
    if selected_backend() === :diffutd
        return Production.fixed_soft(_wedge(g), RayAngles(g.phi,h),g.k,g.L;
            order,check_domain,transition_margin,x_margin)
    end
    # Fixed-rule certification does not need the report's optional G/G'
    # diagnostics. Omitting that unused evaluation makes the stated m-call
    # integration work exact and matches the adaptive production certificate.
    r = UTDKernels._interval_report_local(_wedge(g),g.phi,h,:o,g.k,g.L;
        transition_margin,x_margin,include_gprime=false,
        allow_interior=true,allow_infinite_L=true)
    if check_domain && !r.valid
        throw(GrazingDomainError(r.reason))
    end
    if isinf(g.L)
        ds, _ = Production.pec_wedge_DsDh_grazing(
            _wedge(g),RayAngles(g.phi,h),g.k,g.L;
            face=:o,on_fail=:error,check_domain,transition_margin,x_margin,
            allow_interior=true,allow_infinite_L=true)
        return ds
    end
    return UTDKernels._soft_continuation(_wedge(g),g.phi,h,g.k,g.L,order)
end

# The references below are unrounded BigFloat values, so the binary64 result is
# compared with the exact reference rather than with its binary64 rounding.
rel(value::Number, ref::Number; floor::Real=1e-300) =
    Float64(abs(value - ref) / max(abs(ref), floor))

# --------------------------------------------- independent high-precision oracle
const _F_ASYM_XMIN = BigFloat(100)          # |z| >= 10
const _F_ASYM_BOUND = BigFloat("1e-40")     # certified |Delta F| truncation ceiling

# Convergent power series for erfc(z). The reference path calls it for
# |z| < 10 (x < _F_ASYM_XMIN), where e^{|z|^2} <= e^100 amplifies the
# internal cancellation by at most ~44 digits. Ordinary coefficient
# references use at least 320 bits (~96 decimal digits), leaving about 52
# digits before coefficient subtraction at the crossover. The separate
# overlap check uses 384 bits (~115 digits): about 72 digits remain at
# x = 100 and about 52 at x = 144. It checks branch agreement separately
# from coefficient-level precision convergence, which includes actual
# ensemble maxima and arguments straddling the current crossover.
function _erfc_series_hp(z::Complex{BigFloat})
    abs(z) <= 12 || error("erfc series path requires |z| <= 12, got |z| = $(abs(z))")
    s = z; term = z; z2 = z * z; n = 0
    tol = eps(BigFloat) * 8
    while n < 20000
        n += 1
        term *= -z2 / n
        add = term / (2n + 1)
        s += add
        abs(add) <= tol * max(abs(s), eps(BigFloat)) && n > 8 && break
    end
    return 1 - (2 / sqrt(BigFloat(pi))) * s
end

# F(x) ~ sum_{m>=0} c_m x^{-m}, c_m = i (m - 1/2) c_{m-1}, truncated at its
# least term. Because F = z sqrt(pi) erfcx(z) on the pi/4 ray, the F-series
# terms coincide with the erfcx terms, and Taylor's integral remainder for
# F(x) = sqrt(x) int_0^Inf e^{-s} (x - is)^{-1/2} ds gives the rigorous
# truncation bound |R_M| <= (1/2)_{M+1} / x^{M+1} -- about 5.29e-44 at
# x = 100, M = 100 and decreasing in x. The enforced certificate is the
# more conservative 64 times the least term (~3.37e-42 at the crossover,
# measured against the Laplace-contour oracle to overestimate the actual
# remainder ~3.7e-44 by a factor <100). Both are SERIES-TRUNCATION bounds:
# they do not bound the total floating-point error of the evaluation.
# The bound is enforced before the value is returned, so the crossover is
# controlled by the required coefficient-level error.
function _F_asymptotic_hp(x::BigFloat)
    x >= _F_ASYM_XMIN ||
        error("asymptotic F branch requires x >= $(_F_ASYM_XMIN), got x = $x")
    c = one(Complex{BigFloat}); series = one(Complex{BigFloat})
    inv = 1 / x; power = one(BigFloat)
    least = BigFloat(Inf)
    m = 0
    while m < 10000
        m += 1
        c *= im * (m - BigFloat(1) / 2)
        power *= inv
        add = c * power
        aadd = abs(add)
        # Stop once the terms grow again (optimal truncation) or once they fall
        # below the working precision.
        (aadd >= least && m > 4) && break
        series += add
        least = aadd
        aadd <= eps(BigFloat) * 8 * abs(series) && break
    end
    bound = 64 * least
    bound <= _F_ASYM_BOUND || error(
        "asymptotic F cannot certify bound $(_F_ASYM_BOUND) at x = $x; bound = $bound",
    )
    return series
end

function _F_hp_big(x::BigFloat)
    iszero(x) && return zero(Complex{BigFloat})
    isinf(x) && return one(Complex{BigFloat})
    x >= _F_ASYM_XMIN && return _F_asymptotic_hp(x)
    z = cis(BigFloat(pi) / 4) * sqrt(x)
    return sqrt(BigFloat(pi) * x) * cis(BigFloat(pi) / 4) * exp(z * z) *
           _erfc_series_hp(z)
end

# --------------------------------------------------------------- sampling helper
# Admissible o-face sample spanning interior and exterior wedges. `far_field`
# draws the L = Inf limit; otherwise it draws a finite common distance.
# Returns `((g, h, tmargin, xmargin), :accepted, "")` on success, or
# `(nothing, :domain | :small_derivative, reason)` for a rejected draw so the
# caller can report the proposed/accepted/rejected population by reason.
# Acceptance margins: finite-L uses transition_margin=0.02 and x_margin=0.001;
# the far-field L=Inf ensemble uses transition_margin=0.01 and x_margin=1e-8;
# every accepted case satisfies |G'(phi)| >= 1e-4 max(1, |G(phi)|).
function sample_geometry(rng; far_field::Bool=false)
    n = 0.25 + (1.92 - 0.25) * rand(rng)
    alpha = n * pi
    phi = (0.12 + (0.88 - 0.12) * rand(rng)) * alpha
    tmargin = far_field ? 1e-2 : 2e-2
    if far_field
        k = exp10(0 + 6 * rand(rng))   # k in [1, 1e6]
        g = Geom(alpha, phi, k, Inf)
        h = exp10(-16 + 13 * rand(rng))
        xmargin = 1e-8
    else
        kL = exp10(-2 + 8 * rand(rng))
        g = Geom(alpha, phi, kL, 1.0)  # k*L = kL
        h = exp10(-16 + 13 * rand(rng))
        xmargin = 1e-3
    end
    r = report(g, h; transition_margin=tmargin, x_margin=xmargin)
    r.valid || return nothing, :domain, r.reason
    gg = kern(g, g.phi); gp = kernderiv(g, g.phi)
    abs(gp) < 1e-4 * max(1.0, abs(gg)) &&
        return nothing, :small_derivative, "|G'| < 1e-4 max(1, |G|)"
    return (g, h, tmargin, xmargin), :accepted, ""
end

_nint_hp(x::BigFloat) = x >= 0 ? Int(floor(x + BigFloat(1) / 2)) : Int(ceil(x - BigFloat(1) / 2))

function _prec(dps::Int)
    # This minimum protects the reference through both series cancellation
    # and the observed coefficient subtraction; accuracy is checked below
    # rather than claimed uniformly for arbitrary untested conditions.
    return max(320, Int(ceil(dps * log2(10))) + 32)
end

function kernel_hp(beta::BigFloat, g::Geom, prec::Int)
    alpha = BigFloat(g.alpha; precision=prec)
    kk = BigFloat(g.k; precision=prec)
    LL = isinf(g.L) ? BigFloat(Inf; precision=prec) : BigFloat(g.L; precision=prec)
    kL = kk * LL
    n = alpha / BigFloat(pi)
    acc = zero(Complex{BigFloat})
    for sigma in (+1, -1)
        psi = (BigFloat(pi) + sigma * beta) / (2n)
        N = _nint_hp((beta + sigma * BigFloat(pi)) / (2n * BigFloat(pi)))
        u = 2n * BigFloat(pi) * N - beta
        a = 1 + cos(u)
        acc += cot(psi) * _F_hp_big(kL * a)
    end
    return acc
end

function soft_hp_terms(g::Geom, h::Real; dps::Int=70, phi_at=nothing)
    prec = _prec(dps)
    return setprecision(BigFloat, prec) do
        n = BigFloat(g.alpha; precision=prec) / BigFloat(pi)
        k = BigFloat(g.k; precision=prec)
        phi = BigFloat(something(phi_at, g.phi); precision=prec)
        hh = BigFloat(h; precision=prec)
        C = -cis(-BigFloat(pi) / 4) / (2n * sqrt(2 * BigFloat(pi) * k))
        gm = kernel_hp(phi - hh, g, prec)
        gp = kernel_hp(phi + hh, g, prec)
        return C * (gm - gp), gm, gp
    end
end

soft_hp(g::Geom, h::Real; dps::Int=70, phi_at=nothing) =
    soft_hp_terms(g, h; dps, phi_at)[1]

# Mathematical condition number kappa_s(h) = (|G-| + |G+|)/|G- - G+| evaluated
# from the high-precision kernel terms: both numerator and denominator are
# formed at the reference precision and only the final ratio is rounded to
# binary64. Evaluating the denominator from the production binary64
# subtraction would reproduce the very loss under study (it returns Inf or a
# shifted value once the difference underflows).
function cond_from_terms(gm::Complex{BigFloat}, gp::Complex{BigFloat})
    den = abs(gm - gp)
    den == 0 && return Inf
    return Float64((abs(gm) + abs(gp)) / den)
end

# cond_number recomputes the mathematical condition number independently
# (used where the kernel terms are not already in hand). Where soft_hp_terms
# has already run, cond_from_terms reuses its kernel values.
function cond_number(g::Geom, h::Real; dps::Int=50)
    h <= 0 && return Inf
    _, gm, gp = soft_hp_terms(g, h; dps)
    return cond_from_terms(gm, gp)
end

function kernderiv_hp(g::Geom; dps::Int=70)
    prec = _prec(dps)
    return setprecision(BigFloat, prec) do
        phi = BigFloat(g.phi; precision=prec)
        delta = BigFloat(10)^(-max(8, min(20, dps ÷ 6)))
        gp = kernel_hp(phi + delta, g, prec)
        gm = kernel_hp(phi - delta, g, prec)
        gpp = kernel_hp(phi + 2delta, g, prec)
        gmm = kernel_hp(phi - 2delta, g, prec)
        return (-gpp + 8gp - 8gm + gmm) / (12delta)
    end
end

# Per-piece summands of G' at high precision (same algebra as
# UTDKernels.two_term_kernel_derivative but evaluated from the independent F
# oracle; F' is formed through the identity F' = i(F - 1) + F/(2x)). Each
# sigma contributes two pieces, -dpsi F / sin(psi)^2 and
# cot(psi) F' (kL) a', which can be large and opposing near a transition
# boundary even when the sum stays finite.
function _kernderiv_terms_hp(beta::BigFloat, g::Geom, prec::Int)
    alpha = BigFloat(g.alpha; precision=prec)
    kk = BigFloat(g.k; precision=prec)
    LL = isinf(g.L) ? BigFloat(Inf; precision=prec) : BigFloat(g.L; precision=prec)
    kL = kk * LL
    n = alpha / BigFloat(pi)
    terms = Complex{BigFloat}[]
    for sigma in (+1, -1)
        psi = (BigFloat(pi) + sigma * beta) / (2n)
        N = _nint_hp((beta + sigma * BigFloat(pi)) / (2n * BigFloat(pi)))
        u = 2n * BigFloat(pi) * N - beta
        a = 1 + cos(u)
        da = sin(u)
        spsi = sin(psi)
        x = kL * a
        F = _F_hp_big(x)
        dpsi = sigma / (2n)
        Fp = im * (F - one(F)) + F / (2x)   # F' = i(F - 1) + F/(2x)
        push!(terms, -dpsi * F / (spsi * spsi))
        push!(terms, (cos(psi) / spsi) * Fp * kL * da)
    end
    return terms
end

# Rotating the defining Fresnel tail onto its steepest-descent contour gives
# F(x) = sqrt(x) integral_0^Inf exp(-s) / sqrt(x-is) ds. Differentiation under
# this absolutely convergent integral supplies an independent F' oracle.
function transition_laplace_hp(x::Real; precision::Int=256)
    x > 0 && isfinite(x) || throw(DomainError(x, "x must be finite and positive"))
    return setprecision(BigFloat, precision) do
        xb = BigFloat(x; precision)
        rtol = BigFloat("1e-40")
        F, F_error = quadgk(
            s -> exp(-s) * sqrt(xb) / sqrt(complex(xb, -s)),
            zero(xb), BigFloat(Inf); rtol,
        )
        Fprime, Fprime_error = quadgk(
            s -> exp(-s) * (-im * s) /
                 (2sqrt(xb) * complex(xb, -s)^(BigFloat(3) / 2)),
            zero(xb), BigFloat(Inf); rtol,
        )
        return F, Fprime, F_error, Fprime_error
    end
end
