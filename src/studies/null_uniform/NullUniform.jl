"""
Canonical Faddeeva-moment hierarchy for saddle–pole diffraction transitions.

Phasor convention: `exp(+i ω t)` is used only at the paper level. The
canonical integral implemented here is the paper's explicit definition

    W_m(z) = (i/π) ∫_{-∞}^{∞} t^m exp(-t^2) / (z - t) dt,    Im(z) > 0.

The recurrence is `W_{m+1} = z W_m - i μ_m / π` with Gaussian moments
`μ_n = ∫ t^n exp(-t^2) dt`.
"""
module NullUniform

using QuadGK
using SpecialFunctions
import ..UTDKernels

export gaussian_moment, faddeeva, moment_recurrence, moment_asymptotic
export faddeeva_moments, hierarchy_value, joint_null_coefficients
export hermite_rule, canonical_quadrature, polynomial_amplitude
export scale_normalized_error, faddeeva_hp, hp_moment, hp_moment_asymptotic
export direct_moment, hp_direct_moment
export fitted_log_slope, shifted_null_basis, shifted_null_hierarchy
export DEFAULT_SWITCH, DEFAULT_TAIL, REFERENCE_HERMITE_ORDER
export wedge_cauchy_coefficients
export wedge_kernel, wedge_kernel_zeros, wedge_kernel_poles
export wedge_eigenfunction, wedge_go, wedge_diffracted
export wedge_sdp_quadrature, wedge_diffracted_sdp, wedge_diffracted_model
export wedge_phase_map, wedge_canonical_model, canonical_pole_pair_value
export canonical_pole_extraction, canonical_slope_separated, moment_int_plane
export canonical_pair_quadrature, wedge_saddle_field, wedge_term_scale
export classical_pole_term, linearized_pole_term, slope_separated_term
export nonpolynomial_amplitude, nonpolynomial_closed_form
export factored_null_coefficients

const DEFAULT_SWITCH = 10.0
const DEFAULT_TAIL = (0.35, -0.15, 0.08)

# Gauss--Hermite order of the paper's quadrature reference. For a pole at
# distance Im(z) from the real axis the rule error decays roughly like
# exp(-2 Im(z) sqrt(2n)). The randomized ensemble samples Im(z) >= 0.3, so
# n = 2000 (exp(-0.6 sqrt(4000)) ~ 3e-17) leaves only binary64 summation
# error; n = 700 leaves rule errors up to 2e-10 at Im(z) = 0.3.
const REFERENCE_HERMITE_ORDER = 2000

# Independent Gauss--Hermite reference methods are defined in the
# validation layer; production moments use the parent kernel.
function hermite_rule end
function canonical_quadrature end

function _as_complex(z::Number, name::AbstractString="z")
    UTDKernels._number_isfinite(z) || throw(ArgumentError("$name must be finite"))
    return complex(float(z))
end

function _positive_real(value::Real, name::AbstractString)
    UTDKernels._number_isfinite(value) && UTDKernels._primal_value(value) > 0 || throw(ArgumentError("$name must be positive and finite"))
    return float(value)
end

function _finite_coefficients(values, name::AbstractString)
    all(v -> v isa Number, values) || throw(ArgumentError("$name must contain numbers"))
    return [_as_complex(value, name) for value in values]
end

"""
    gaussian_moment(n) -> μ_n

Gaussian moment `μ_n = ∫_{-∞}^{∞} t^n exp(-t^2) dt`. Odd moments vanish;
even moments are `Γ(n/2 + 1/2)`.
"""
function gaussian_moment(n::Integer)
    n < 0 && throw(ArgumentError("n must be nonnegative"))
    return UTDKernels._null_gaussian_moment(n, 1.0)
end

function gaussian_moment(n::Integer, ::Type{T}) where {T<:AbstractFloat}
    n < 0 && throw(ArgumentError("n must be nonnegative"))
    return UTDKernels._null_gaussian_moment(n, one(T))
end

"""
    faddeeva(z) -> w(z)

Faddeeva function `w(z) = exp(-z^2) erfc(-i z) = erfcx(-i z)`.
"""
function faddeeva(z::Number)
    zz = _as_complex(z)
    value = UTDKernels._faddeeva_w(zz)
    isfinite(real(value)) && isfinite(imag(value)) ||
        throw(DomainError(value, "Faddeeva value is outside the finite binary64 range"))
    return value
end
faddeeva(z::AbstractArray) = faddeeva.(z)

"""
    moment_recurrence(z, max_m) -> W_0,…,W_{max_m}

Forward recurrence `W_{m+1} = z W_m - i μ_m / π` with `W_0 = w(z)`.
This branch is accurate for moderate `|z|` and loses digits when
`z W_m` nearly cancels `i μ_m / π`.
"""
function moment_recurrence(z::Number, max_m::Integer)
    max_m < 0 && throw(ArgumentError("max_m must be nonnegative"))
    return UTDKernels._null_moment_recurrence(_as_complex(z), max_m)
end

"""
    moment_asymptotic(z, m; max_terms=80) -> W_m

Least-term inverse-power evaluation of `W_m` for large `|z|`:

    W_m(z) ∼ (i/π) Σ_{q=0}^{∞} μ_{m+q} / z^{q+1}.
"""
function moment_asymptotic(z::Number, m::Integer; max_terms::Integer=80)
    m < 0 && throw(ArgumentError("m must be nonnegative"))
    max_terms >= 1 || throw(ArgumentError("max_terms must be positive"))
    zz = _as_complex(z)
    iszero(zz) && throw(ArgumentError("large-z asymptotic is undefined at z=0"))
    return UTDKernels._null_moment_asymptotic(
        zz, m; max_terms=max_terms,
    )
end

"""
    faddeeva_moments(z, max_m; switch=10) -> W_0,…,W_{max_m}

Stable evaluator: recurrence for `|z| ≤ switch`, independent least-term
asymptotics for `|z| > switch`.
"""
function faddeeva_moments(z::Number, max_m::Integer; switch::Real=DEFAULT_SWITCH)
    max_m < 0 && throw(ArgumentError("max_m must be nonnegative"))
    isfinite(switch) && switch > 0 ||
        throw(ArgumentError("switch must be positive and finite"))
    zz = _as_complex(z)
    return UTDKernels.faddeeva_moments(zz, max_m; switch=switch)
end

"""
    hierarchy_value(k, z, coefficients; order=nothing) -> I

Evaluate `Σ_{m=0}^{M} a_m k^{-m/2} W_m(z)`.
"""
function hierarchy_value(k::Real, z::Number, coefficients::AbstractVector;
                         order::Union{Nothing,Integer}=nothing)
    kf = _positive_real(k, "k")
    for coefficient in coefficients
        coefficient isa Number || throw(ArgumentError("coefficients must contain numbers"))
        _as_complex(coefficient, "coefficients")
    end
    order === nothing || order >= 0 ||
        throw(ArgumentError("order must be nonnegative"))
    return UTDKernels.null_uniform_transition(
        kf, _as_complex(z), coefficients; order=order,
    )
end

"""
    joint_null_coefficients(k, Λ; tail=(0.35, -0.15, 0.08))

Taylor coefficients of `a(t) = Λ/√k + t + tail[1] t^2 + ⋯`.

This is the coefficient-based null *proxy*: the polynomial's actual root
is near, but not exactly at, `t = -Λ/√k`.  For an amplitude whose moving
zero is exact at that coordinate use [`factored_null_coefficients`](@ref).
"""
function joint_null_coefficients(k::Real, Lambda::Number;
                                 tail::Tuple=DEFAULT_TAIL)
    kf = _positive_real(k, "k")
    lambda = _as_complex(Lambda, "Lambda")
    kf += zero(real(lambda))
    tail_values = _finite_coefficients(tail, "tail")
    coefficients = [lambda / sqrt(kf), one(lambda), tail_values...]
    all(UTDKernels._number_isfinite, coefficients) || throw(DomainError(
        coefficients, "null coefficients exceed the active numeric range",
    ))
    return coefficients
end

"""
    factored_null_coefficients(k, Λ; b=(b1, b2, ...)) -> a_m

Taylor coefficients of the exactly factored moving-null amplitude

    a(t) = (t + Λ/√k) (1 + b_1 t + b_2 t^2 + ⋯ + b_p t^p),

whose zero is exactly at `t = -Λ/√k` for every finite `k`.  With the
default `b = (0.35, -0.15, 0.08)` the amplitude is a quartic and the
five-moment hierarchy is a finite identity.
"""
function factored_null_coefficients(k::Real, Lambda::Number;
                                    b::Tuple=(0.35, -0.15, 0.08))
    kf = _positive_real(k, "k")
    lambda = _as_complex(Lambda, "Lambda")
    kf += zero(real(lambda))
    bvals = _finite_coefficients(b, "b")
    tau = lambda / sqrt(kf)
    # (t + tau) * (1 + b_1 t + ... + b_p t^p)
    inner = [one(tau), bvals...]
    coeff = zeros(typeof(tau + first(inner)), length(inner) + 1)
    @inbounds for j in eachindex(inner)
        coeff[j + 1] += inner[j]      # t * inner
        coeff[j] += tau * inner[j]    # tau * inner
    end
    all(UTDKernels._number_isfinite, coeff) || throw(DomainError(
        coeff, "factored null coefficients exceed the active numeric range",
    ))
    return coeff
end

"""
    polynomial_amplitude(coefficients) -> t -> a(t)

Horner evaluation of `a(t) = Σ a_m t^m`.
"""
function polynomial_amplitude(coefficients::AbstractVector)
    coeff = _finite_coefficients(coefficients, "coefficients")
    return function (t::Number)
        out = zero(complex(t) + (isempty(coeff) ? zero(t) : first(coeff)))
        @inbounds for i in lastindex(coeff):-1:firstindex(coeff)
            out = out * t + coeff[i]
        end
        return out
    end
end

"""
    scale_normalized_error(approx, exact, scale)

Relative residual against a scale that remains finite at a field null.
"""
function scale_normalized_error(approx::Number, exact::Number, scale::Number)
    floor = 10 * eps(Float64)
    approx_value = _as_complex(approx, "approx")
    exact_value = _as_complex(exact, "exact")
    scale_value = _as_complex(scale, "scale")
    return abs(approx_value - exact_value) / max(abs(scale_value), floor)
end

include("../../validation/NullUniformReferences.jl")
include("../../validation/NullUniformHermiteReference.jl")

"""
    fitted_log_slope(x, y)

Ordinary least-squares slope of `log10(y)` versus `log10(x)` on positive
finite samples.
"""
function fitted_log_slope(x, y)
    length(x) == length(y) || throw(ArgumentError("x and y lengths must match"))
    xx = Float64[]
    yy = Float64[]
    for (a, b) in zip(x, y)
        if isfinite(a) && isfinite(b) && a > 0 && b > 0
            push!(xx, Float64(log10(a)))
            push!(yy, Float64(log10(b)))
        end
    end
    length(xx) < 2 &&
        throw(ArgumentError("at least two positive finite samples are required"))
    mx = sum(xx) / length(xx)
    my = sum(yy) / length(yy)
    num = 0.0
    den = 0.0
    @inbounds for i in eachindex(xx)
        dx = xx[i] - mx
        num += dx * (yy[i] - my)
        den += dx * dx
    end
    den > 0 || throw(ArgumentError("x samples must not all be equal"))
    return num / den
end

"""
    shifted_null_basis(z, Λ, r; moment_offset=q) -> U_{r,q}

Binomial combination `Σ_j C(r,j) Λ^{r-j} W_{q+j}(z)`.
"""
function shifted_null_basis(z::Number, Lambda::Number, zero_order::Integer;
                            moment_offset::Integer=0)
    (zero_order < 0 || moment_offset < 0) &&
        throw(ArgumentError("orders must be nonnegative"))
    moments = faddeeva_moments(z, moment_offset + zero_order)
    Λ = _as_complex(Lambda, "Lambda")
    return UTDKernels._null_shifted_basis(
        moments, Λ, Int(zero_order), Int(moment_offset),
    )
end

"""
    shifted_null_hierarchy(k, z, Λ, r, b_coefficients; order=nothing)

Evaluate the hierarchy for `a(t) = (t + Λ/√k)^r b(t)`.
"""
function shifted_null_hierarchy(k::Real, z::Number, Lambda::Number,
                                zero_order::Integer,
                                b_coefficients::AbstractVector;
                                order::Union{Nothing,Integer}=nothing)
    kf = _positive_real(k, "k")
    zero_order < 0 && throw(ArgumentError("zero_order must be nonnegative"))
    for coefficient in b_coefficients
        coefficient isa Number || throw(ArgumentError("b_coefficients must contain numbers"))
        _as_complex(coefficient, "b_coefficients")
    end
    order === nothing || order >= 0 ||
        throw(ArgumentError("order must be nonnegative"))
    return UTDKernels.shifted_null_transition(
        kf, _as_complex(z), _as_complex(Lambda, "Lambda"), zero_order,
        b_coefficients; order=order,
    )
end

include("wedge.jl")
include("../../validation/NullUniformCauchyReference.jl")
include("baselines.jl")

end # module
