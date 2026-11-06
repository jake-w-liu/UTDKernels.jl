# Positive-real derivative backend. The Gaussian rational representation is
# F(x) = π^(-1/2) ∫ exp(-t²) x/(x-it²) dt over the real line.
# Differentiating the rational factor avoids the cancellation in the DE.
# The 64-node rule is calibrated for binary inputs on 4 ≤ x < 64. Its
# exp(-81) tail is below 1e-37 for all three derivatives; the quadrature and
# floating-point errors are checked separately against independent references.
const _REAL_TRANSITION_GAUSSIAN_MINIMUM = 4.0
const _REAL_TRANSITION_GAUSSIAN_RULE = let
    nodes, weights = _cached_gauss_legendre_rule(64)
    squared_nodes = ((nodes .+ 1) .* (9 / 2)).^2
    squared_nodes, (9 / sqrt(pi)) .* weights .* exp.(-squared_nodes) .* squared_nodes
end

function _real_transition_gaussian_derivatives(x::Real)
    argument = x + zero(Float64)
    square = argument * argument
    first = zero(complex(argument))
    second = first
    third = first
    nodes, weights = _REAL_TRANSITION_GAUSSIAN_RULE
    @inbounds for index in eachindex(nodes)
        u = nodes[index]
        inverse_denominator = inv(square + u * u)
        kernel = complex(argument * inverse_denominator, u * inverse_denominator)
        weight = (weights[index] * inverse_denominator) * inverse_denominator
        term = complex((2argument * u) * weight, (u * u - square) * weight)
        first += term
        second -= 2term * kernel
        third += 6term * kernel * kernel
    end
    return first, second, third
end

function _real_transition_asymptotic_derivatives(x::Real)
    inverse_x = inv(x + zero(Float64))
    power = inverse_x
    first = zero(complex(inverse_x))
    second = first
    third = first
    for (order, coefficient) in enumerate(ASYM_F_COEFFS)
        next_power = power * inverse_x
        next_next_power = next_power * inverse_x
        first -= order * coefficient * next_power
        second += order * (order + 1) * coefficient * next_next_power
        third -= order * (order + 1) * (order + 2) * coefficient *
                 next_next_power * inverse_x
        power = next_power
    end
    return first, second, third
end

function _real_transition_work_parameters(
    values::Tuple, native_type::Type{<:AbstractFloat}=Float64; native_power::Int=1,
)
    any(_number_contains_ad, values) || return false, 256
    all(_number_isfinite, values) || throw(DomainError(
        values, "real transition coordinates and seeds must be finite",
    ))
    ranges = _scalar_leaf_range.(values)
    low = minimum(first, ranges)
    high = maximum(last, ranges)
    degree = 1 + maximum(_shadow_ad_depth, values)
    arithmetic_degree = native_power * degree
    risk = (isfinite(low) && arithmetic_degree * low < log2(floatmin(native_type)) + 16) ||
           (isfinite(high) && arithmetic_degree * high > log2(floatmax(native_type)) - 16)
    span = isfinite(low) && isfinite(high) ? max(0, high - low) : 0
    work = max(256, ceil(Int, 106 + degree * span + 64degree))
    return risk, work
end

# Recovery uses the existing precision-retaining Faddeeva owner, not widened
# binary quadrature data. This is internal range recovery for binary outputs;
# it does not add a public arbitrary-precision transition interface.
function _transition_faddeeva_value(x::Number)
    root = x isa Real && _primal_value(x) > 0 ? sqrt(x) : safe_sqrt(x)
    phase = cis(BigFloat(pi) / 4)
    return sqrt(BigFloat(pi)) * root * phase *
           _null_wide_faddeeva(im * phase * root)
end

function _transition_faddeeva_components(x::Number)
    value = _transition_faddeeva_value(x)
    inverse_x = inv(x)
    recurrence = im + inverse_x / 2
    first = recurrence * value - im
    # Explicit polynomial products retain live imaginary AD coefficients at
    # a zero imaginary value across the supported ForwardDiff versions.
    inverse_squared = inverse_x * inverse_x
    inverse_cubed = inverse_squared * inverse_x
    second = recurrence * first - value * inverse_squared / 2
    third = recurrence * second - first * inverse_squared + value * inverse_cubed
    return (; F=value, Fm1=value-one(value), Fp=first, Fpp=second, Fppp=third)
end

# Unconverted component in the caller's widened workspace. Keeping this
# representation owner shared also lets scaled physical outputs postpone their
# final conversion until after their original factors have been combined.
function _transition_work_component(x::Number, component::Symbol, policy::Symbol)
    radius = _primal_value(abs(x))
    threshold = component === :Fm1 ? MIN_F_MINUS_ONE_ASYMPTOTIC_THRESHOLD :
                MIN_F_PRIME_ASYMPTOTIC_THRESHOLD
    result = if policy === :real && _primal_value(real(x)) >= threshold
        if component in (:F, :Fm1)
            inverse_x = inv(x)
            power = one(inverse_x)
            residual = zero(complex(inverse_x))
            for coefficient in ASYM_F_COEFFS
                power *= inverse_x
                residual += coefficient * power
            end
            component === :F ? one(residual) + residual : residual
        else
            derivatives = _real_transition_asymptotic_derivatives(x)
            index = component === :Fp ? 1 : component === :Fpp ? 2 : 3
            derivatives[index]
        end
    elseif policy === :passive &&
           (radius >= PASSIVE_LARGE_RADIUS ||
            (component === :Fm1 && radius >= PASSIVE_FM1_RADIUS))
        getproperty(_passive_asymptotic_bundle(x), component)
    else
        getproperty(_transition_faddeeva_components(x), component)
    end
    return result
end

# The requested component alone is converted to the output type. An unused
# higher derivative may exceed that range even when this result and all its
# AD coefficients are representable.
function _transition_component_wide(
    argument::Number, component::Symbol, policy::Symbol, ::Type{C}; native=nothing,
) where {C<:Number}
    policy in (:real, :passive) || throw(ArgumentError("unknown transition recovery policy"))
    component in (:F, :Fm1, :Fp, :Fpp, :Fppp) ||
        throw(ArgumentError("unknown transition component"))
    _, work = _real_transition_work_parameters((real(argument), imag(argument)))
    work <= _SHADOW_MAX_PHASE_WORK_BITS || throw(ArgumentError(
        "transition coordinates and seeds exceed the bounded wide workspace",
    ))
    return _shadow_stable_wide_value(C, work, "transition component $component") do
        x = _faddeeva_widen_bigfloat(argument)
        result = _transition_work_component(x, component, policy)
        if native !== nothing
            base = complex(_primal_value(real(result)), _primal_value(imag(result)))
            native_base = complex(_primal_value(real(native)), _primal_value(imag(native)))
            return (result - base) + _faddeeva_widen_bigfloat(native_base)
        end
        return result
    end
end

function _real_transition_derivative_wide(x::Real, index::Int)
    component = (:Fp, :Fpp, :Fppp)[index]
    return _transition_component_wide(x, component, :real, typeof(complex(x + zero(Float64))))
end

function _real_transition_derivatives_wide(x::Real)
    return ntuple(index -> _real_transition_derivative_wide(x, index), 3)
end

function _real_transition_derivative(x::Real, index::Int)
    risk, _ = _real_transition_work_parameters((x,); native_power=3)
    risk && return _real_transition_derivative_wide(x, index)
    result = _primal_value(x) < MIN_F_PRIME_ASYMPTOTIC_THRESHOLD ?
        _real_transition_gaussian_derivatives(x)[index] :
        _real_transition_asymptotic_derivatives(x)[index]
    _number_isfinite(result) && return result
    return _real_transition_derivative_wide(x, index)
end

function _real_transition_derivatives(x::Real)
    risk, _ = _real_transition_work_parameters((x,); native_power=3)
    risk && return _real_transition_derivatives_wide(x)
    result = _primal_value(x) < MIN_F_PRIME_ASYMPTOTIC_THRESHOLD ?
        _real_transition_gaussian_derivatives(x) :
        _real_transition_asymptotic_derivatives(x)
    all(_number_isfinite, result) && return result
    return _real_transition_derivatives_wide(x)
end

# The exact rational divided difference avoids both endpoint subtraction and
# finite Taylor truncation. Multiply by the interval only after summation so
# tiny intervals retain their representable result.
function _real_transition_gaussian_divided_difference(x::Real, delta::Real)
    left = x + zero(Float64)
    right = left + delta
    total = zero(complex(left))
    nodes, weights = _REAL_TRANSITION_GAUSSIAN_RULE
    @inbounds for index in eachindex(nodes)
        u = nodes[index]
        total -= im * weights[index] * inv(complex(left, -u)) *
                 inv(complex(right, -u))
    end
    return total
end

# The same exact form for the inverse-power series. With l=1/x0 and r=1/x1,
# r^m-l^m = -(x1-x0)*l*r*h_m, where h_1=1 and h_m=r*h_(m-1)+l^(m-1). The sum of
# c_m*h_m stays of order one over every range, so the callers multiply by the
# interval, l and r last.
function _real_transition_inverse_power_sum(inverse_left::Number, inverse_right::Number)
    h = one(inverse_left)
    left_power = one(inverse_left)
    total = zero(complex(inverse_left))
    for coefficient in ASYM_F_COEFFS
        total += coefficient * h
        left_power *= inverse_left
        h = inverse_right * h + left_power
    end
    return total
end

function _real_transition_asymptotic_scaled_difference(x::Real, delta::Real, k::Real)
    left = x + zero(Float64)
    inverse_left = inv(left)
    inverse_right = inv(left + delta)
    total = _real_transition_inverse_power_sum(inverse_left, inverse_right)
    # delta*inverse_left stays bounded; delta/sqrt(k) can overflow near floatmax.
    return -total * (((delta * inverse_left) / sqrt(k)) * inverse_right)
end

@inline function _real_transition_uses_inverse_powers(x::Real, delta::Real)
    x_primal = _primal_value(x)
    return min(x_primal, x_primal + _primal_value(delta)) >= MIN_F_PRIME_ASYMPTOTIC_THRESHOLD
end

# Below 64 the Gaussian rule needs one endpoint in its calibrated range, and it
# only improves toward the upper limit 128. From 64 the series is valid.
@inline function _real_transition_difference_supported(x0::Real, x1::Real)
    (isfinite(x0) && isfinite(x1)) || return false
    lower, upper = minmax(x0, x1)
    lower < _REAL_TRANSITION_GAUSSIAN_MINIMUM && return false
    return lower >= MIN_F_PRIME_ASYMPTOTIC_THRESHOLD ||
           upper < 2MIN_F_PRIME_ASYMPTOTIC_THRESHOLD
end

function _real_transition_scaled_difference(x::Real, delta::Real, k::Real)
    risk, work = _real_transition_work_parameters((x, delta, k))
    if !risk
        result = _real_transition_uses_inverse_powers(x, delta) ?
            _real_transition_asymptotic_scaled_difference(x, delta, k) :
            _real_transition_gaussian_divided_difference(x, delta) * (delta / sqrt(k))
        _number_isfinite(result) && return result
    end
    work <= _SHADOW_MAX_PHASE_WORK_BITS || throw(ArgumentError(
        "transition difference seeds exceed the bounded wide workspace",
    ))
    C = typeof(complex(zero(x) + zero(delta) + zero(k) + zero(Float64)))
    return _shadow_stable_wide_value(C, work, "real transition difference") do
        xw, dw, kw = _faddeeva_widen_bigfloat.((x, delta, k))
        # Subtracting two values near 1 costs 2log2(x) bits, which no bounded
        # workspace covers at the largest x; the series form loses nothing.
        _real_transition_uses_inverse_powers(xw, dw) ?
            _real_transition_asymptotic_scaled_difference(xw, dw, kw) :
            (_transition_faddeeva_value(xw + dw) -
             _transition_faddeeva_value(xw)) / sqrt(kw)
    end
end

function _real_transition_ad_value_wide(x::Real, native)
    return _transition_component_wide(
        x, :F, :real, typeof(complex(x + zero(Float64))); native,
    )
end
