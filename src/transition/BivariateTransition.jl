"""
Correlation-aware bivariate Fresnel transition from a finite Plackett
correlation integral.

The scalar canonical variables may be real or complex; the correlation is real
and satisfies `abs(rho)<1`. The default evaluator uses certified adaptive
QuadGK. A caller may request the shared fixed Gauss--Legendre rule explicitly.
"""

using SpecialFunctions: erfc

const BIVARIATE_DEFAULT_RTOL = 2e-13
const BIVARIATE_DEFAULT_ATOL = 2e-13
const BIVARIATE_MAX_FIXED_ORDER = 2048

@inline function _bivariate_values(xi::Number, eta::Number, rho::Real)
    xi_real, xi_imaginary, eta_real, eta_imaginary, correlation = promote(
        float(real(xi)),
        float(imag(xi)),
        float(real(eta)),
        float(imag(eta)),
        float(rho),
    )
    return complex(xi_real, xi_imaginary),
           complex(eta_real, eta_imaginary), correlation
end

@inline function _bivariate_constants(value::Number)
    scalar = real(value)
    pi_value = zero(scalar) + _typed_pi(scalar)
    two = one(scalar) + one(scalar)
    imaginary_unit = complex(zero(scalar), one(scalar))
    return pi_value, two, imaginary_unit, cis(pi_value / 4)
end

@inline function _validate_bivariate_coordinate(value::Number, name::AbstractString)
    _number_isfinite(value) || throw(DomainError(value, "$name must be finite"))
    return value
end

@inline function _validate_bivariate_correlation(rho::Real)
    primal = _primal_value(rho)
    (isfinite(primal) && abs(primal) < one(primal)) || throw(DomainError(
        rho,
        "bivariate correlation rho must be finite with abs(rho) < 1",
    ))
    return rho
end

function _bivariate_tolerances(rtol::Real, atol::Real)
    relative, absolute = promote(float(rtol), float(atol))
    relative_primal = _primal_value(relative)
    absolute_primal = _primal_value(absolute)
    (isfinite(relative_primal) && relative_primal >= zero(relative_primal)) ||
        throw(ArgumentError("rtol must be finite and nonnegative"))
    (isfinite(absolute_primal) && absolute_primal >= zero(absolute_primal)) ||
        throw(ArgumentError("atol must be finite and nonnegative"))
    (!iszero(relative_primal) || !iszero(absolute_primal)) ||
        throw(ArgumentError("rtol and atol cannot both be zero"))
    return relative, absolute
end

"""Specifically named one-boundary Fresnel switch for the bivariate family."""
function _bivariate_fresnel_switch(x::Number)
    value = complex(float(real(x)), float(imag(x)))
    _validate_bivariate_coordinate(value, "transition coordinate")
    _, two, _, phase = _bivariate_constants(value)
    argument = -phase * value / sqrt(two)
    result = try
        erfc(argument) / two
    catch error
        error isa MethodError || rethrow()
        throw(ArgumentError(
            "bivariate Fresnel switch does not support $(typeof(value)): " *
            "erfc is unavailable for $(typeof(argument))",
        ))
    end
    _number_isfinite(result) || throw(DomainError(
        x,
        "bivariate one-boundary transition is non-finite",
    ))
    return result
end

function _bivariate_fresnel_switch_prime(x::Number)
    value = complex(float(real(x)), float(imag(x)))
    _validate_bivariate_coordinate(value, "transition coordinate")
    pi_value, two, imaginary_unit, phase = _bivariate_constants(value)
    result = phase / sqrt(two * pi_value) *
             exp(-imaginary_unit * value^2 / two)
    _number_isfinite(result) || throw(DomainError(
        x,
        "bivariate one-boundary derivative is non-finite",
    ))
    return result
end

function _bivariate_plackett_kernel(xi::Number, eta::Number, rho::Real)
    x, y, correlation = _bivariate_values(xi, eta, rho)
    _validate_bivariate_coordinate(x, "xi")
    _validate_bivariate_coordinate(y, "eta")
    _validate_bivariate_correlation(correlation)
    pi_value, two, imaginary_unit, _ = _bivariate_constants(x + y + correlation)
    determinant_scale = muladd(-correlation, correlation, one(correlation))
    phase = (
        x^2 - two * correlation * x * y + y^2
    ) / (two * determinant_scale)
    result = exp(-imaginary_unit * phase) /
             (two * pi_value * sqrt(determinant_scale))
    _number_isfinite(result) || throw(DomainError(
        (xi, eta, rho),
        "bivariate Plackett kernel is non-finite",
    ))
    return result
end

@inline function _bivariate_theta_integrand(x, y, theta)
    pi_value, two, imaginary_unit, _ = _bivariate_constants(x + y + theta)
    correlation = sin(theta)
    cosine = cos(theta)
    phase = (
        x^2 - two * correlation * x * y + y^2
    ) / (two * cosine^2)
    return exp(-imaginary_unit * phase) / (two * pi_value)
end

function _bivariate_fixed_rule(x, y, correlation, nodes, weights)
    base = _bivariate_fresnel_switch(x) * _bivariate_fresnel_switch(y)
    iszero(_primal_value(correlation)) && return base
    theta_end = asin(correlation)
    two = one(correlation) + one(correlation)
    total = zero(base)
    @inbounds for index in eachindex(nodes)
        node = oftype(correlation, nodes[index])
        weight = oftype(correlation, weights[index])
        theta = theta_end * (node + one(node)) / two
        total += theta_end * weight / two * _bivariate_theta_integrand(x, y, theta)
    end
    value = base + total
    _number_isfinite(value) || throw(DomainError(
        (x, y, correlation),
        "fixed bivariate transition is non-finite",
    ))
    return value
end

function _bivariate_adaptive(x, y, correlation, rtol, atol)
    base = _bivariate_fresnel_switch(x) * _bivariate_fresnel_switch(y)
    iszero(_primal_value(correlation)) && return base
    theta_end = asin(correlation)
    zero_theta = zero(theta_end)
    lower, upper, orientation = if _primal_value(theta_end) > zero(_primal_value(theta_end))
        zero_theta, theta_end, one(theta_end)
    else
        theta_end, zero_theta, -one(theta_end)
    end
    integral, estimate = QuadGK.quadgk(
        theta -> _bivariate_theta_integrand(x, y, theta),
        lower,
        upper;
        rtol,
        atol,
    )
    estimate_primal = _primal_value(estimate)
    target = max(atol, rtol * abs(integral))
    target_primal = _primal_value(target)
    (isfinite(estimate_primal) && estimate_primal <= target_primal) ||
        throw(DomainError(
            (x, y, correlation, rtol, atol),
            "adaptive bivariate quadrature did not meet its requested tolerance",
        ))
    value = base + orientation * integral
    _number_isfinite(value) || throw(DomainError(
        (x, y, correlation),
        "adaptive bivariate transition is non-finite",
    ))
    return value
end

"""
    bivariate_fresnel_transition(xi, eta, rho;
                                 order=nothing, rtol=2e-13, atol=2e-13)

Evaluate the correlation-aware bivariate Fresnel transition. The default
`order=nothing` uses adaptive QuadGK and accepts its value only when the
reported error meets `max(atol, rtol*abs(integral))`. A positive integer
`order` explicitly selects a fixed shared Gauss--Legendre rule, bounded to
`1:2048`; tolerance keywords do not control that rule.

`xi` and `eta` must be finite and `rho` must be finite and real with
`abs(rho)<1`. At `rho==0`, the result is exactly the product of the two
one-boundary Fresnel switches.
"""
function bivariate_fresnel_transition(
    xi::Number,
    eta::Number,
    rho::Real;
    order::Union{Nothing,Integer}=nothing,
    rtol::Real=BIVARIATE_DEFAULT_RTOL,
    atol::Real=BIVARIATE_DEFAULT_ATOL,
)
    x, y, correlation = _bivariate_values(xi, eta, rho)
    _validate_bivariate_coordinate(x, "xi")
    _validate_bivariate_coordinate(y, "eta")
    _validate_bivariate_correlation(correlation)
    if order === nothing
        relative, absolute = _bivariate_tolerances(rtol, atol)
        return _bivariate_adaptive(x, y, correlation, relative, absolute)
    end
    fixed_order = _validate_shared_gauss_legendre_order(
        order,
        BIVARIATE_MAX_FIXED_ORDER,
    )
    if iszero(_primal_value(correlation))
        return _bivariate_fresnel_switch(x) * _bivariate_fresnel_switch(y)
    end
    nodes, weights = _cached_gauss_legendre_rule(
        fixed_order;
        maximum_order=BIVARIATE_MAX_FIXED_ORDER,
    )
    return _bivariate_fixed_rule(x, y, correlation, nodes, weights)
end

function bivariate_fresnel_transition(
    xi::AbstractArray,
    eta::AbstractArray,
    rho::Real;
    order::Union{Nothing,Integer}=nothing,
    rtol::Real=BIVARIATE_DEFAULT_RTOL,
    atol::Real=BIVARIATE_DEFAULT_ATOL,
)
    size(xi) == size(eta) || throw(DimensionMismatch(
        "xi and eta arrays must have the same shape",
    ))
    _validate_bivariate_correlation(rho)
    if order === nothing
        relative, absolute = _bivariate_tolerances(rtol, atol)
        return map(xi, eta) do x, y
            bivariate_fresnel_transition(
                x, y, rho; rtol=relative, atol=absolute,
            )
        end
    end
    fixed_order = _validate_shared_gauss_legendre_order(
        order,
        BIVARIATE_MAX_FIXED_ORDER,
    )
    if iszero(_primal_value(rho))
        return map(xi, eta) do xi_value, eta_value
            _bivariate_fresnel_switch(xi_value) *
                _bivariate_fresnel_switch(eta_value)
        end
    end
    nodes, weights = _cached_gauss_legendre_rule(
        fixed_order;
        maximum_order=BIVARIATE_MAX_FIXED_ORDER,
    )
    return map(xi, eta) do xi_value, eta_value
        x, y, correlation = _bivariate_values(xi_value, eta_value, rho)
        _validate_bivariate_coordinate(x, "xi")
        _validate_bivariate_coordinate(y, "eta")
        _bivariate_fixed_rule(x, y, correlation, nodes, weights)
    end
end

"""
    bivariate_mechanism_weights(xi, eta, rho; kwargs...)

Return correlated four-region weights `(W00, W10, W01, W11)`. Their sum is
one, while `W10+W11` and `W01+W11` recover the two one-boundary marginals.
Quadrature keywords are forwarded to [`bivariate_fresnel_transition`](@ref).
"""
function bivariate_mechanism_weights(
    xi::Number,
    eta::Number,
    rho::Real;
    order::Union{Nothing,Integer}=nothing,
    rtol::Real=BIVARIATE_DEFAULT_RTOL,
    atol::Real=BIVARIATE_DEFAULT_ATOL,
)
    first = _bivariate_fresnel_switch(xi)
    second = _bivariate_fresnel_switch(eta)
    joint = bivariate_fresnel_transition(xi, eta, rho; order, rtol, atol)
    return (
        one(joint) - first - second + joint,
        first - joint,
        second - joint,
        joint,
    )
end

# Error-free scaled products for the normalized determinant. The products are
# formed near unit magnitude, so fma tails retain the bits that a direct
# `a*c-b*b` or `1-rho*rho` subtraction would discard near rank one.
function _bivariate_hessian_primal(a::T, b::T, c::T) where {T<:Union{Float32,Float64}}
    mantissa_a, exponent_a = frexp(a)
    mantissa_c, exponent_c = frexp(c)
    if iszero(b)
        return -b, one(T)
    end
    mantissa_b, exponent_b = frexp(abs(b))
    product_exponent = exponent_a + exponent_c
    square_exponent = 2exponent_b
    common_exponent = max(product_exponent, square_exponent)

    scaled_a = ldexp(mantissa_a, product_exponent - common_exponent)
    product = scaled_a * mantissa_c
    product_tail = fma(scaled_a, mantissa_c, -product)
    scaled_b = ldexp(mantissa_b, square_exponent - common_exponent)
    square = scaled_b * mantissa_b
    square_tail = fma(scaled_b, mantissa_b, -square)
    determinant = (product - square) + (product_tail - square_tail)
    denominator = product + product_tail
    scale = determinant / denominator

    root_product = denominator
    even_exponent = common_exponent
    if isodd(even_exponent)
        root_product *= 2
        even_exponent -= 1
    end
    scaled_ratio = ldexp(mantissa_b, exponent_b - even_exponent ÷ 2)
    rho = -copysign(scaled_ratio / sqrt(root_product), b)
    return rho, scale
end

function _bivariate_hessian_primal(a::T, b::T, c::T) where {T<:Real}
    diagonal = sqrt(a) * sqrt(c)
    rho = -b / diagonal
    return rho, muladd(-rho, rho, one(rho))
end

@inline function _bivariate_correct_primal(value, robust_primal)
    return value + (robust_primal - _primal_value(value))
end

"""
    bivariate_transition_hessian(a, b, c, u_boundary, v_boundary; k=1)

Map the positive-definite quadratic phase
`k*(a*u^2 + 2b*u*v + c*v^2)/2` to canonical `(xi, eta, rho)`. All inputs must
be finite, `a>0`, `c>0`, `k>0`, and `a*c-b^2>0`. The normalized determinant is
formed from scaled error-free products, avoiding a BigFloat fallback and
retaining near-rank-one product bits. The return value is a named tuple.
"""
function bivariate_transition_hessian(
    a::Real,
    b::Real,
    c::Real,
    u_boundary::Real,
    v_boundary::Real;
    k::Real=1,
)
    a_value, b_value, c_value, u_value, v_value, k_value = promote(
        float(a), float(b), float(c), float(u_boundary), float(v_boundary), float(k),
    )
    isfinite(_primal_value(a_value)) || throw(DomainError(a_value, "a must be finite"))
    isfinite(_primal_value(b_value)) || throw(DomainError(b_value, "b must be finite"))
    isfinite(_primal_value(c_value)) || throw(DomainError(c_value, "c must be finite"))
    isfinite(_primal_value(u_value)) ||
        throw(DomainError(u_value, "u_boundary must be finite"))
    isfinite(_primal_value(v_value)) ||
        throw(DomainError(v_value, "v_boundary must be finite"))
    isfinite(_primal_value(k_value)) || throw(DomainError(k_value, "k must be finite"))
    a_primal = _primal_value(a_value)
    b_primal = _primal_value(b_value)
    c_primal = _primal_value(c_value)
    k_primal = _primal_value(k_value)
    (a_primal > zero(a_primal) && c_primal > zero(c_primal) &&
     k_primal > zero(k_primal)) || throw(DomainError(
        (a, c, k),
        "a, c, and k must be positive",
    ))

    robust_rho, robust_scale = _bivariate_hessian_primal(a_primal, b_primal, c_primal)
    (isfinite(robust_rho) && isfinite(robust_scale) && robust_scale > zero(robust_scale)) ||
        throw(DomainError(
            (a, b, c),
            "the Hessian must be positive definite",
        ))

    rho_raw = -b_value / (sqrt(a_value) * sqrt(c_value))
    rho = _bivariate_correct_primal(rho_raw, robust_rho)
    scale_raw = muladd(-rho, rho, one(rho))
    determinant_scale = _bivariate_correct_primal(scale_raw, robust_scale)
    xi = _scaled_sqrt_product(k_value, a_value, determinant_scale) * u_value
    eta = _scaled_sqrt_product(k_value, c_value, determinant_scale) * v_value
    (_number_isfinite(xi) && _number_isfinite(eta) && _number_isfinite(rho)) ||
        throw(DomainError(
            (a, b, c, u_boundary, v_boundary, k),
            "canonical Hessian coordinates are non-finite",
        ))
    return (; xi, eta, rho)
end
