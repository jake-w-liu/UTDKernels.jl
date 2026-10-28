"""
Canonical shadow-boundary sensitivity models and receiver-weighted fields.
The parent kernels own switches, multipliers, bounds and pullbacks; this module
owns complete Sommerfeld fields, objective actions, jets and validation methods.

Paper-level time convention: `exp(+i omega t)`. The real signed-coordinate
switch and its derivative are

    W(q) = erfc(-exp(i*pi/4) q) / 2,
    K(q) = exp(i*pi/4) exp(-i q^2) / sqrt(pi).
"""
module ShadowSensitivity

using FastGaussQuadrature: gausslegendre
using QuadGK: quadgk
using SpecialFunctions: erfcx
import ..UTDKernels

export canonical_switch, canonical_kernel, scaled_kernel
export canonical_switch_reference, halfplane_derivatives_reference
export fourier_multiplier, truncated_multiplier, real_gaussian_multiplier
export halfplane_q, halfplane_q_prime, halfplane_q_second
export halfplane_demodulated, halfplane_d1, halfplane_d2
export bump_asymmetric, curved_coordinate, curved_inverse
export composite_phase_quadrature, gaussian_action
export bandlimited_action, bandlimit_remainder_bound
export pullback_approximations, translated_objective_action
export translated_first_correction, translated_gaussian_action
export relative_error, fitted_log_slope
export sommerfeld_field_with_derivatives, sommerfeld_cosine_branch_with_derivatives, sommerfeld_cosine_branch, sommerfeld_field, sommerfeld_incident
export sommerfeld_demodulated, sommerfeld_demodulated_dp
export DEFAULT_MAX_PANELS, default_receiver_panels
export receiver_quadrature, receiver_window, field_sensitivity_exact, misfit_sensitivity_exact
export field_sensitivity_remainder, misfit_sensitivity_remainder
export sommerfeld_inverse_map, sommerfeld_boundary, field_transition_action
export misfit_transition_action, sommerfeld_branch_jets, bump_jets
export misfit_local_jets, window_composed_jets
export field_truncated_action, misfit_truncated_action
export carrier_action_exact, carrier_action_truncated
export fourier_samples, spectral_moment

const DEFAULT_MAX_PANELS = 10_000_000

function _validate_finite(x::Number, name::AbstractString)
    UTDKernels._number_isfinite(x) || throw(ArgumentError("$name must be finite"))
    return complex(float(x))
end

function _validate_positive(x::Real, name::AbstractString)
    UTDKernels._number_isfinite(x) && UTDKernels._primal_value(x) > 0 || throw(ArgumentError("$name must be positive and finite"))
    return float(x)
end

function _finite_result(value::Number, context::AbstractString)
    UTDKernels._number_isfinite(value) ||
        throw(DomainError(value, "$context is outside the active numeric range"))
    return value
end

function _log1pexp(x::Real)
    return x > 0 ? x + log1p(exp(-x)) : log1p(exp(x))
end

"""Canonical Fresnel switch `W(q)`."""
function canonical_switch(q::Number)
    _validate_finite(q, "q")
    return UTDKernels.shadow_switch(q)
end

canonical_switch(q::AbstractArray) = canonical_switch.(q)

"""
    canonical_switch_reference(q; precision=256)

Independent BigFloat real-path Fresnel integral for real `q`:

    W(q) = 1/2 + exp(i*pi/4)/sqrt(pi) * integral_0^q exp(-i*t^2) dt.

The result retains the requested precision and does not call `erfc`.
"""
function canonical_switch_reference(q::Real; precision::Integer=256)
    isfinite(q) || throw(ArgumentError("q must be finite"))
    precision >= 128 || throw(ArgumentError("precision must be at least 128 bits"))
    return setprecision(BigFloat, precision) do
        qb = BigFloat(q)
        iszero(qb) && return Complex{BigFloat}(BigFloat(0.5), BigFloat(0))
        tolerance = BigFloat(2)^(-precision ÷ 2)
        integrand = t -> exp(-im * t * t)
        integral, = quadgk(integrand, zero(qb), qb;
                            rtol=tolerance, atol=tolerance)
        Complex{BigFloat}(BigFloat(0.5), BigFloat(0)) +
            exp(im * BigFloat(π) / 4) / sqrt(BigFloat(π)) * integral
    end
end

"""Canonical sensitivity kernel `K(q) = W'(q)`."""
function canonical_kernel(q::Number)
    _validate_finite(q, "q")
    return UTDKernels.shadow_sensitivity_kernel(q)
end

canonical_kernel(q::AbstractArray) = canonical_kernel.(q)

"""Scaled kernel `sqrt(kappa) K(sqrt(kappa) s)` for `kappa > 0`."""
function scaled_kernel(s::Real, kappa::Real)
    _validate_finite(s, "s")
    _validate_positive(kappa, "kappa")
    return UTDKernels.shadow_sensitivity_kernel(s, kappa)
end

function scaled_kernel(s::AbstractArray, kappa::Real)
    _validate_positive(kappa, "kappa")
    return scaled_kernel.(s, Ref(kappa))
end

"""Exact Fourier multiplier `exp(i xi^2 / (4 kappa))`."""
function fourier_multiplier(xi::Real, kappa::Real)
    _validate_finite(xi, "xi")
    _validate_positive(kappa, "kappa")
    return UTDKernels.shadow_sensitivity_multiplier(xi, kappa)
end

function fourier_multiplier(xi::AbstractArray, kappa::Real)
    _validate_positive(kappa, "kappa")
    return fourier_multiplier.(xi, Ref(kappa))
end

"""First `terms` terms of the exact multiplier's exponential series."""
function truncated_multiplier(xi::Real, kappa::Real, terms::Integer)
    terms >= 1 || throw(ArgumentError("terms must be at least one"))
    _validate_finite(xi, "xi")
    _validate_positive(kappa, "kappa")
    return UTDKernels.shadow_sensitivity_multiplier(xi, kappa; terms)
end

function truncated_multiplier(xi::AbstractArray, kappa::Real, terms::Integer)
    _validate_positive(kappa, "kappa")
    terms >= 1 || throw(ArgumentError("terms must be at least one"))
    return truncated_multiplier.(xi, Ref(kappa), Ref(terms))
end

"""Fourier multiplier of a real Gaussian smoother."""
function real_gaussian_multiplier(xi::Real, kappa::Real)
    x = real(_validate_finite(xi, "xi"))
    kap = _validate_positive(kappa, "kappa")
    scaled = x / (2sqrt(kap))
    isfinite(scaled) || return zero(complex(kap))
    phase = scaled * scaled
    isfinite(phase) || return zero(complex(kap))
    return complex(exp(-phase))
end

function real_gaussian_multiplier(xi::AbstractArray, kappa::Real)
    _validate_positive(kappa, "kappa")
    return real_gaussian_multiplier.(xi, Ref(kappa))
end

function halfplane_q(s::Real, kappa::Real)
    _validate_finite(s, "s")
    _validate_positive(kappa, "kappa")
    return UTDKernels._shadow_halfplane_coordinate(s, kappa)
end


function halfplane_q_prime(s::Real, kappa::Real)
    _validate_finite(s, "s")
    _validate_positive(kappa, "kappa")
    return UTDKernels._shadow_halfplane_coordinate_derivative(s, kappa)
end

function halfplane_q_second(s::Real, kappa::Real)
    _validate_finite(s, "s")
    _validate_positive(kappa, "kappa")
    return UTDKernels._shadow_halfplane_coordinate_second(s, kappa)
end

for function_name in (:halfplane_q, :halfplane_q_prime, :halfplane_q_second)
    @eval function $function_name(s::AbstractArray, kappa::Real)
        _validate_positive(kappa, "kappa")
        return $function_name.(s, Ref(kappa))
    end
end

halfplane_demodulated(s::Real, kappa::Real) = begin
    _validate_finite(s, "s")
    _validate_positive(kappa, "kappa")
    UTDKernels._shadow_halfplane_switch(s, kappa)
end
function halfplane_demodulated(s::AbstractArray, kappa::Real)
    _validate_positive(kappa, "kappa")
    return halfplane_demodulated.(s, Ref(kappa))
end

function halfplane_d1(s::Real, kappa::Real)
    _validate_finite(s, "s")
    _validate_positive(kappa, "kappa")
    return UTDKernels._shadow_halfplane_first(s, kappa)
end

function halfplane_d1(s::AbstractArray, kappa::Real)
    _validate_positive(kappa, "kappa")
    return halfplane_d1.(s, Ref(kappa))
end

function halfplane_d2(s::Real, kappa::Real)
    _validate_finite(s, "s")
    _validate_positive(kappa, "kappa")
    return UTDKernels._shadow_halfplane_second(s, kappa)
end


function halfplane_d2(s::AbstractArray, kappa::Real)
    _validate_positive(kappa, "kappa")
    return halfplane_d2.(s, Ref(kappa))
end

"""
    halfplane_derivatives_reference(s, kappa; precision=256, step=1e-5)

Independent five-point BigFloat derivatives of the half-plane switch. The
stencil uses local Fresnel integrals between the center and each offset, so
the constant switch value cancels before differencing.
"""
function halfplane_derivatives_reference(s::Real, kappa::Real;
                                         precision::Integer=256,
                                         step::Real=1e-5)
    isfinite(s) || throw(ArgumentError("s must be finite"))
    _validate_positive(kappa, "kappa")
    _validate_positive(step, "step")
    precision >= 128 || throw(ArgumentError("precision must be at least 128 bits"))
    return setprecision(BigFloat, precision) do
        sb = BigFloat(s)
        kap = BigFloat(kappa)
        h = BigFloat(step)
        q0 = sqrt(2kap) * sin(sb / 2)
        tolerance = BigFloat(2)^(-precision ÷ 2)
        prefactor = exp(im * BigFloat(π) / 4) / sqrt(BigFloat(π))
        value(offset) = begin
            coordinate = sb + offset * h
            q = sqrt(2kap) * sin(coordinate / 2)
            iszero(offset) && return zero(Complex{BigFloat})
            integral, = quadgk(t -> exp(-im * t * t), q0, q;
                                rtol=tolerance, atol=tolerance)
            prefactor * integral
        end
        fm2 = value(-2)
        fm1 = value(-1)
        f0 = value(0)
        fp1 = value(1)
        fp2 = value(2)
        first = (fm2 - 8fm1 + 8fp1 - fp2) / (12h)
        second = (-fp2 + 16fp1 - 30 * f0 + 16fm1 - fm2) / (12h^2)
        return first, second
    end
end

"""Smooth compact probe with optional odd component."""
function bump_asymmetric(s::Real; radius::Real=0.8, alpha::Real=0.35)
    _validate_positive(radius, "radius")
    isfinite(s) || throw(ArgumentError("s must be finite"))
    isfinite(alpha) || throw(ArgumentError("alpha must be finite"))
    x = float(s) / float(radius)
    abs(x) < 1 || return 0.0
    return (1 + float(alpha) * x) * exp(1 - inv(1 - x^2))
end

function bump_asymmetric(s::AbstractArray{<:Real}; radius::Real=0.8, alpha::Real=0.35)
    _validate_positive(radius, "radius")
    isfinite(alpha) || throw(ArgumentError("alpha must be finite"))
    return bump_asymmetric.(s; radius=radius, alpha=alpha)
end

"""Quadratic coordinate `g(s) = s + gamma*s^2/2`."""
function curved_coordinate(s::Real, gamma::Real=0.6)
    isfinite(s) || throw(ArgumentError("s must be finite"))
    isfinite(gamma) || throw(ArgumentError("gamma must be finite"))
    sf = float(s)
    gf = float(gamma)
    value = sf + (0.5gf * sf) * sf
    return real(_finite_result(value, "curved coordinate"))
end

function curved_coordinate(s::AbstractArray{<:Real}, gamma::Real=0.6)
    isfinite(gamma) || throw(ArgumentError("gamma must be finite"))
    return curved_coordinate.(s, Ref(gamma))
end

"""Near-origin inverse of [`curved_coordinate`](@ref)."""
function curved_inverse(y::Real, gamma::Real=0.6)
    isfinite(y) || throw(ArgumentError("y must be finite"))
    isfinite(gamma) || throw(ArgumentError("gamma must be finite"))
    g = float(gamma)
    yf = float(y)
    discriminant = 1 + 2g * yf
    if !isfinite(discriminant)
        value = setprecision(BigFloat, 256) do
            gb = UTDKernels._faddeeva_widen_bigfloat(g)
            yb = UTDKernels._faddeeva_widen_bigfloat(yf)
            db = 1 + 2gb * yb
            db >= 0 || throw(DomainError(y,
                "1 + 2*gamma*y must be nonnegative"))
            2yb / (1 + sqrt(db))
        end
        result = convert(typeof(g + yf), value)
        UTDKernels._number_isfinite(result) &&
            (!UTDKernels._primal_iszero(value) ? !UTDKernels._primal_iszero(result) : true) ||
            throw(DomainError(value,
                "curved inverse is outside the active numeric range"))
        return result
    end
    discriminant >= 0 ||
        throw(DomainError(y, "1 + 2*gamma*y must be nonnegative"))
    return real(_finite_result(yf / ((1 + sqrt(discriminant)) / 2),"curved inverse"))
end

function curved_inverse(y::AbstractArray{<:Real}, gamma::Real=0.6)
    isfinite(gamma) || throw(ArgumentError("gamma must be finite"))
    return curved_inverse.(y, Ref(gamma))
end

"""
    composite_phase_quadrature(amplitude, lo, hi, kappa; ...)

Integrate `scaled_kernel(y, kappa) * amplitude(y)` on `[lo, hi]`. Panels
are uniform in `y^2` on each side of zero, which bounds the quadratic phase
increment. Fixed Gauss--Legendre quadrature is used within every panel.
"""
function composite_phase_quadrature end
include("quadrature.jl")

# The centered action depends only on v = sqrt(kappa)*sigma. Both ordinary
# and bounded-wide/AD evaluations use this same normalized expression.
function _gaussian_action_scaled(v::Real)
    return if UTDKernels._primal_value(v) > 1
        inv(sqrt(complex(one(v), -inv(v)^2 / 2)))
    else
        complex(v, v) / sqrt(complex(one(v), 2v^2))
    end
end

function _gaussian_action_ad(kappa::Real, sigma::Real)
    UTDKernels._shadow_binary_type(kappa, sigma)
    C = typeof(complex(zero(float(kappa)) + zero(float(sigma))))
    return UTDKernels._shadow_stable_wide_value(C, 192, "Gaussian action AD") do
        kw = UTDKernels._faddeeva_widen_bigfloat(kappa)
        sw = UTDKernels._faddeeva_widen_bigfloat(sigma)
        _gaussian_action_scaled(sqrt(kw) * sw)
    end
end

"""Closed-form scaled-kernel action on a centered Gaussian probe."""
function gaussian_action(kappa::Real, sigma::Real)
    kap, sig = promote(
        _validate_positive(kappa, "kappa"), _validate_positive(sigma, "sigma"),
    )
    any(UTDKernels._number_contains_ad, (kap, sig)) &&
        return _gaussian_action_ad(kap, sig)
    # Scale the positive product before squaring. Forming sigma^2 first can
    # underflow even while sqrt(kappa)*sigma is representable.
    v = UTDKernels._scaled_sqrt_product(kap, sig, sig)
    return _finite_result(_gaussian_action_scaled(v), "Gaussian action")
end

function _validate_spectral_vectors(spectrum, xi, weights)
    length(spectrum) == length(xi) == length(weights) ||
        throw(ArgumentError("spectrum, xi, and weights lengths must match"))
    isempty(spectrum) && throw(ArgumentError("spectral vectors must not be empty"))
    all(v -> v isa Real && isfinite(v), xi) ||
        throw(ArgumentError("xi values must be finite and real"))
    all(v -> v isa Real && isfinite(v), weights) ||
        throw(ArgumentError("weights must be finite and real"))
    all(v -> isfinite(real(v)) && isfinite(imag(v)), spectrum) ||
        throw(ArgumentError("spectrum values must be finite"))
    return nothing
end

"""
    bandlimited_action(spectrum, xi, weights, kappa;
                       terms=nothing, reflected_spectrum=nothing)

Apply the exact multiplier, or a specified number of Taylor terms, using the
Fourier pairing with samples of `hat(f)(-xi)` and the `1/(2*pi)` factor.
When `reflected_spectrum` is omitted, `xi` must be an ordered symmetric grid
and the reflected samples are obtained by reversing `spectrum`. Supply
`reflected_spectrum` explicitly for a nonsymmetric grid.
"""
function bandlimited_action(spectrum, xi, weights, kappa::Real;
                            terms::Union{Nothing,Integer}=nothing,
                            reflected_spectrum=nothing)
    _validate_spectral_vectors(spectrum, xi, weights)
    _validate_positive(kappa, "kappa")
    terms === nothing || terms >= 1 ||
        throw(ArgumentError("terms must be at least one"))
    use_reversed_grid = reflected_spectrum === nothing
    if use_reversed_grid
        n = length(xi)
        for index in 1:n
            isapprox(float(xi[index]), -float(xi[n + 1 - index]);
                     rtol=1e-12, atol=1e-14) ||
                throw(ArgumentError(
                    "xi must be ordered symmetrically when reflected_spectrum is omitted"))
        end
    else
        _validate_spectral_vectors(reflected_spectrum, xi, weights)
    end
    total = zero(complex(kappa + first(xi) + first(weights)) + first(spectrum))
    if use_reversed_grid
        for (index, reflected_index) in zip(
            eachindex(spectrum, xi, weights), Iterators.reverse(eachindex(spectrum)))
            multiplier = terms === nothing ?
                fourier_multiplier(xi[index], kappa) :
                truncated_multiplier(xi[index], kappa, terms)
            total += complex(spectrum[reflected_index]) *
                     float(weights[index]) * multiplier
        end
    else
        for (index, reflected_index) in zip(
            eachindex(spectrum, xi, weights), eachindex(reflected_spectrum))
            multiplier = terms === nothing ?
                fourier_multiplier(xi[index], kappa) :
                truncated_multiplier(xi[index], kappa, terms)
            total += complex(reflected_spectrum[reflected_index]) *
                     float(weights[index]) * multiplier
        end
    end
    return total / (2π)
end

"""
    bandlimit_remainder_bound(omega, kappa, order_M, spectral_l1)

Finite-bandwidth Taylor remainder bound. `spectral_l1` is
`norm(hat(f), 1)/(2*pi)`, including the Fourier inversion factor.
"""
function bandlimit_remainder_bound(omega::Real, kappa::Real, order_M::Integer,
                                   spectral_l1::Real)
    isfinite(omega) && omega >= 0 ||
        throw(ArgumentError("omega must be finite and nonnegative"))
    _validate_positive(kappa, "kappa")
    order_M >= 1 || throw(ArgumentError("order_M must be at least one"))
    isfinite(spectral_l1) && spectral_l1 >= 0 ||
        throw(ArgumentError("spectral_l1 must be finite and nonnegative"))
    return UTDKernels._shadow_sensitivity_remainder_bound(
        omega, kappa, order_M, spectral_l1,
    )
end

"""
    pullback_approximations(f0, f1, f2, f3, f4, a, b, c, d, kappa)

Leading, complete first-order, and complete second-order actions of the
nonlinear-coordinate pullback. `a = g'(s0)` must be positive.
"""
function pullback_approximations(f0::Number, f1::Number, f2::Number,
                                 f3::Number, f4::Number,
                                 a::Real, b::Real, c::Real, d::Real,
                                 kappa::Real)
    _validate_positive(a, "a")
    _validate_positive(kappa, "kappa")
    all(isfinite, (b, c, d)) || throw(ArgumentError("b, c, and d must be finite"))
    all(v -> isfinite(real(v)) && isfinite(imag(v)), (f0, f1, f2, f3, f4)) ||
        throw(ArgumentError("probe derivatives must be finite"))
    values = UTDKernels.shadow_sensitivity_pullback(
        f0, f1, f2, f3, f4, a, b, c, d, kappa,
    )
    return values.leading, values.first, values.second
end

function _translated_objective_action_wide(nu::Real, kappa::Real, sigma::Real)
    R = UTDKernels._shadow_binary_type(nu, kappa, sigma)
    C = typeof(complex(zero(float(nu)) + zero(float(kappa)) + zero(float(sigma))))
    n = abs(UTDKernels._primal_value(nu))
    # |Im(nu^2/(4D))| <= nu^2/(4*kappa). This bounds the phase work
    # precision before exponentiation; ordinary binary intermediates need
    # not represent the phase or sigma^2.
    bits = iszero(n) ? 192 : UTDKernels._shadow_phase_work_precision(
        2log(n) - log(UTDKernels._primal_value(kappa)) - log(4), R,
    )
    return UTDKernels._shadow_stable_wide_value(C, bits, "translated objective action") do
        nw = UTDKernels._faddeeva_widen_bigfloat(nu)
        kw = UTDKernels._faddeeva_widen_bigfloat(kappa)
        sw = UTDKernels._faddeeva_widen_bigfloat(sigma)
        denominator = inv(2sw^2) + im * kw
        _gaussian_action_scaled(sqrt(kw) * sw) * exp(-nw^2 / (4denominator))
    end
end

"""Exact translated-boundary action for a Gaussian-windowed carrier."""
function translated_objective_action(nu::Real, kappa::Real, sigma::Real)
    isfinite(nu) || throw(ArgumentError("nu must be finite"))
    nuf, kap, sig = promote(float(nu),
        _validate_positive(kappa, "kappa"), _validate_positive(sigma, "sigma"))
    any(UTDKernels._number_contains_ad, (nuf, kap, sig)) &&
        return _translated_objective_action_wide(nuf, kap, sig)
    alpha = inv(2sig^2)
    denominator = alpha + im * kap
    denominator_magnitude = abs(denominator)
    if !(isfinite(alpha) && alpha > 0 && isfinite(denominator_magnitude))
        return _translated_objective_action_wide(nuf, kap, sig)
    end
    decay_ratio = abs(nuf) / (2denominator_magnitude / sqrt(alpha))
    log_magnitude = -(decay_ratio * decay_ratio)
    if !isfinite(log_magnitude) || log_magnitude < log(nextfloat(zero(kap)))
        return zero(complex(kap))
    end
    phase_ratio = nuf / (2denominator_magnitude)
    phase_square = phase_ratio * phase_ratio
    phase = phase_square * kap
    if !isfinite(phase) || (!iszero(nuf) && iszero(phase_square))
        return _translated_objective_action_wide(nuf, kap, sig)
    end
    return _finite_result(gaussian_action(kap, sig) * exp(log_magnitude + im * phase),
                          "translated objective action")
end

"""First distributional correction for the translated Gaussian carrier."""
function translated_first_correction(nu::Real, kappa::Real, sigma::Real)
    isfinite(nu) || throw(ArgumentError("nu must be finite"))
    _validate_positive(kappa, "kappa")
    _validate_positive(sigma, "sigma")
    nuf = float(nu)
    kap = float(kappa)
    sig = float(sigma)
    frequency = nuf / (2sqrt(kap))
    width = inv(2sqrt(kap) * sig)
    correction = frequency * frequency + width * width
    isfinite(correction) || throw(DomainError(
        correction, "first translated correction is outside binary64 range"))
    return 1 + im * correction
end

# The Gaussian is analytic at nu=0. A logarithm of abs(nu) followed by a
# value-only zero shortcut loses nested derivatives (and differs between
# ForwardDiff versions). Evaluate AD in the shared bounded workspace using
# algebraic forms that avoid large-u cancellation and preserve nilpotents.
function _translated_gaussian_action_ad(nu::Real,kappa::Real,sigma::Real)
    R=UTDKernels._shadow_binary_type(nu,kappa,sigma)
    C=typeof(zero(float(nu))+zero(float(kappa))+zero(float(sigma)))
    return UTDKernels._shadow_stable_wide_value(C,192,"translated Gaussian AD action") do
        nw=UTDKernels._faddeeva_widen_bigfloat(nu)
        kw=UTDKernels._faddeeva_widen_bigfloat(kappa)
        sw=UTDKernels._faddeeva_widen_bigfloat(sigma)
        u2=2kw*sw^2
        prefactor=UTDKernels._primal_value(u2)>=1 ?
            inv(sqrt(1+inv(u2))) : sqrt(u2/(1+u2))
        inverse_variance=inv(sw)^2+2kw
        prefactor*exp(-nw^2/(2inverse_variance))
    end
end

"""Action of the real Gaussian smoother on the translated carrier."""
function translated_gaussian_action(nu::Real, kappa::Real, sigma::Real)
    isfinite(nu) || throw(ArgumentError("nu must be finite"))
    _validate_positive(kappa, "kappa")
    _validate_positive(sigma, "sigma")
    any(UTDKernels._number_contains_ad,(nu,kappa,sigma)) &&
        return _translated_gaussian_action_ad(nu,kappa,sigma)
    kap = float(kappa)
    sig = float(sigma)
    # With u^2 = 2*kappa*sigma^2, the exact action is
    #
    #   u / sqrt(1 + u^2) * exp(-nu^2*sigma^2 / (2*(1 + u^2))).
    #
    # Use logarithms for u^2 and the decay so neither alpha = 1/(2*sigma^2)
    # nor a ratio that can underflow before taking its square root is formed.
    # The direct u value retains full relative accuracy for a representable
    # subnormal prefactor; hypot keeps its normalization scale-safe.
    log_u = log(sig) + 0.5log(kap) + 0.5log(2.0)
    log_one_plus_u2 = _log1pexp(2log_u)
    u = sqrt(2.0) * sqrt(kap) * sig
    prefactor = if isinf(u)
        1.0
    elseif iszero(u)
        # `exp(log_u)` decides whether the exact prefactor still rounds into
        # the subnormal range when the direct product rounded to zero.
        exp(log_u)
    else
        u / hypot(1.0, u)
    end
    iszero(prefactor) && return 0.0

    nuf = float(nu)
    attenuation = 1.0
    if !iszero(nuf)
        log_decay = 2(log(abs(nuf)) + log(sig)) - log(2.0) - log_one_plus_u2
        if log_decay > log(floatmax(Float64))
            return 0.0
        end
        attenuation = exp(-exp(log_decay))
    end
    return _finite_result(prefactor * attenuation, "translated Gaussian action")
end

function relative_error(value::Number, reference::Number; floor::Real=1e-30)
    _validate_positive(floor, "floor")
    return abs(value - reference) / max(abs(reference), float(floor))
end

# One quadrature owner supplies public samples and the spectral integrator's
# observed Fourier refinement changes. A callback may contain an unknown
# carrier, so unresolved refinement is an error rather than a successful value.
function _fourier_samples_with_error(fun, lo::Real, hi::Real, xi_values;
                                     atol::Real=1e-12,
                                     max_refinements::Integer=5)
    isfinite(lo) && isfinite(hi) && lo < hi || throw(ArgumentError(
        "Fourier support requires finite lo < hi",
    ))
    isfinite(atol) && atol > 0 || throw(ArgumentError("atol must be positive and finite"))
    max_refinements >= 2 || throw(ArgumentError("max_refinements must be at least two"))
    xis = collect(xi_values)
    ndims(xis) == 1 && !isempty(xis) || throw(ArgumentError(
        "Fourier frequencies must be a nonempty vector",
    ))
    all(x -> x isa Real && isfinite(x), xis) || throw(ArgumentError(
        "Fourier frequencies must be finite real values",
    ))
    width = float(hi) - float(lo)
    isfinite(width) || throw(ArgumentError("Fourier support width overflowed"))
    ximax = max(1.0, maximum(x -> abs(float(x)), xis))
    estimate = 8 * width * (ximax / (2π)) + 96
    isfinite(estimate) && estimate < typemax(Int) || throw(ArgumentError(
        "Fourier quadrature order exceeds the active integer range",
    ))
    npoints = ceil(Int, estimate)
    mid = float(lo) / 2 + float(hi) / 2
    hw = width / 2
    previous = nothing
    change = nothing
    for refinement in 1:max_refinements
        nodes, wts = gausslegendre(npoints)
        svals = mid .+ hw .* nodes
        values = fun.(svals)
        all(v -> v isa Number && UTDKernels._number_isfinite(v), values) ||
            throw(DomainError(values, "Fourier integrand must return finite numbers"))
        fvals = values .* wts .* hw
        next = exp.(-im .* xis .* svals') * fvals
        all(UTDKernels._number_isfinite, next) ||
            throw(DomainError(next, "Fourier samples are non-finite"))
        if previous !== nothing
            change = abs.(next .- previous)
            maximum(change) <= atol && return (values=next, change=change)
        end
        previous = next
        refinement == max_refinements && break
        npoints <= (typemax(Int) - 1) ÷ 2 || throw(ArgumentError(
            "Fourier quadrature refinement exceeds the active integer range",
        ))
        npoints = ceil(Int, 1.5npoints)
    end
    throw(ArgumentError(
        "Fourier quadrature did not converge within max_refinements=$max_refinements " *
        "(largest last change=$(maximum(change)), atol=$atol)",
    ))
end

"""
    fourier_samples(fun, lo, hi, xi_values; atol=1e-12, max_refinements=5)

Numerical finite-support Fourier samples `∫_lo^hi fun(s) exp(-im*xi*s) ds`.
Gauss--Legendre order is refined until the largest sample change is at most
`atol`; unresolved refinement throws. This convergence test is numerical and
cannot certify an arbitrary callback's unsampled oscillations or singularities.
"""
fourier_samples(fun, lo::Real, hi::Real, xi_values; kwargs...) =
    _fourier_samples_with_error(fun, lo, hi, xi_values; kwargs...).values

"""
    spectral_moment(fun, lo, hi; power=2, xi_cut=6000, frequency_step=nothing)

Numerically integrate `abs(xi)^power * abs(fhat(xi)) / (2π)` and
`abs(fhat(xi)) / (2π)` over the complete finite band `[-xi_cut, xi_cut]`.
Both frequency signs are sampled even for complex callbacks. The default fine
frequency step is at most `π/(2*(hi-lo))`; the final interval ends at `xi_cut`.

The returned `moment_error_estimate` and `l1_error_estimate` add the observed
fine/coarse trapezoidal change and integrated Fourier-refinement changes.
They are numerical diagnostics, not rigorous error bounds. No quiet-gap stopping,
noise clipping, or infinite-band tail certificate is applied. Narrow spectral
features require a smaller `frequency_step` and a convergence check by the caller.
"""
function spectral_moment(fun, lo::Real, hi::Real;
                         power::Real=2, xi_cut::Real=6000.0,
                         frequency_step::Union{Nothing,Real}=nothing)
    isfinite(lo) && isfinite(hi) && lo < hi || throw(ArgumentError(
        "spectral support requires finite lo < hi",
    ))
    isfinite(power) && power >= 0 || throw(ArgumentError(
        "spectral power must be finite and nonnegative",
    ))
    isfinite(xi_cut) && xi_cut >= 0 || throw(ArgumentError(
        "spectral cutoff must be finite and nonnegative",
    ))
    width = float(hi) - float(lo)
    isfinite(width) || throw(ArgumentError("spectral support width overflowed"))
    step = frequency_step === nothing ? π / (2width) :
        _validate_positive(frequency_step, "frequency_step")
    isfinite(step) && step > 0 || throw(ArgumentError("frequency step is outside the active numeric range"))
    iszero(xi_cut) && return (
        moment=0.0, l1=0.0, moment_error_estimate=0.0, l1_error_estimate=0.0,
    )
    # Use an even fine-grid interval count so every other node is a complete
    # coarse grid on precisely the same finite band, including its endpoint.
    estimate = float(xi_cut) / (2step)
    isfinite(estimate) && estimate < typemax(Int) ÷ 2 || throw(ArgumentError(
        "spectral frequency grid exceeds the active integer range",
    ))
    intervals = 2max(1, ceil(Int, estimate))
    h = float(xi_cut) / intervals
    moment = l1 = coarse_moment = coarse_l1 = 0.0
    fourier_moment_change = fourier_l1_change = 0.0
    # Blocks bound the Fourier matrix size; all requested frequencies are
    # covered regardless of how small an earlier block happens to be.
    for first_index in 0:256:intervals
        indices = first_index:min(first_index + 255, intervals)
        xis = [j == intervals ? float(xi_cut) : j * h for j in indices]
        samples = _fourier_samples_with_error(fun, lo, hi, vcat(xis, -xis))
        half = length(xis)
        for (i, j) in enumerate(indices)
            magnitude = abs(samples.values[i]) + abs(samples.values[i + half])
            change = samples.change[i] + samples.change[i + half]
            weight = (j == 0 || j == intervals) ? 0.5 : 1.0
            weighted_power = xis[i]^power
            moment += weight * h * weighted_power * magnitude
            l1 += weight * h * magnitude
            fourier_moment_change += weight * h * weighted_power * change
            fourier_l1_change += weight * h * change
            if iseven(j)
                coarse_moment += weight * 2h * weighted_power * magnitude
                coarse_l1 += weight * 2h * magnitude
            end
        end
    end
    result = (
        moment=moment / (2π), l1=l1 / (2π),
        moment_error_estimate=(abs(moment - coarse_moment) + fourier_moment_change) / (2π),
        l1_error_estimate=(abs(l1 - coarse_l1) + fourier_l1_change) / (2π),
    )
    all(isfinite, result) || throw(DomainError(result, "spectral integral exceeds the active numeric range"))
    return result
end

function fitted_log_slope(x, y)
    length(x) == length(y) || throw(ArgumentError("x and y lengths must match"))
    xx = Float64[]
    yy = Float64[]
    for (a, b) in zip(x, y)
        if isfinite(a) && isfinite(b) && a > 0 && b > 0
            push!(xx, log10(float(a)))
            push!(yy, log10(float(b)))
        end
    end
    length(xx) >= 2 || throw(ArgumentError("at least two positive finite samples are required"))
    mean_x = sum(xx) / length(xx)
    mean_y = sum(yy) / length(yy)
    denominator = sum((value - mean_x)^2 for value in xx)
    denominator > 0 || throw(ArgumentError("x samples must not all be equal"))
    return sum((xx[i] - mean_x) * (yy[i] - mean_y)
               for i in eachindex(xx)) / denominator
end

include("sommerfeld.jl")

end # module
