"""
Coordinate-invariant dual Hessian metric for astigmatic saddle--pole
transitions.

Small fixed dimensions use scalar Cholesky solves. Higher dimensions use one
typed in-place Cholesky workspace. No path forms an explicit inverse.
"""

@inline function _hessian_numeric_type(::Type{TH}, ::Type{TG}) where {TH<:Real,TG<:Real}
    (isconcretetype(TH) && isconcretetype(TG)) || throw(ArgumentError(
        "Hessian and covector require concrete real element types",
    ))
    return promote_type(typeof(float(zero(TH))), typeof(float(zero(TG))))
end

@inline function _hessian_symmetric_entry(H, row::Int, column::Int, ::Type{T}) where {T<:Real}
    first_value = convert(T, H[row, column])
    row == column && return first_value
    second_value = convert(T, H[column, row])
    return first_value + (second_value - first_value) / 2
end

function _validate_hessian_metric_inputs(
    H::AbstractMatrix{TH},
    g::AbstractVector{TG},
) where {TH<:Real,TG<:Real}
    Base.require_one_based_indexing(H, g)
    dimension = size(H, 1)
    dimension == size(H, 2) || throw(DimensionMismatch("H must be square"))
    dimension > 0 || throw(ArgumentError("H must be nonempty"))
    length(g) == dimension || throw(DimensionMismatch(
        "g must have the same dimension as H",
    ))
    T = _hessian_numeric_type(TH, TG)

    sample_primal = float(_primal_value(convert(T, H[1, 1])))
    R = typeof(sample_primal)
    matrix_scale = zero(R)
    @inbounds for column in 1:dimension, row in 1:dimension
        value = convert(T, H[row, column])
        _number_isfinite(value) || throw(DomainError(
            H[row, column], "H must contain only finite values",
        ))
        matrix_scale = max(matrix_scale, convert(R, abs(_primal_value(value))))
    end

    symmetry_tolerance = 64 * dimension * eps(one(sample_primal)) * matrix_scale
    @inbounds for column in 1:dimension, row in 1:(column - 1)
        difference = _primal_value(
            convert(T, H[row, column]) - convert(T, H[column, row]),
        )
        abs(difference) <= symmetry_tolerance || throw(DomainError(
            H,
            "H must be symmetric to its input precision",
        ))
    end

    covector_scale = zero(R)
    @inbounds for index in 1:dimension
        value = convert(T, g[index])
        _number_isfinite(value) || throw(DomainError(
            g[index], "g must contain only finite values",
        ))
        covector_scale = max(
            covector_scale, convert(R, abs(_primal_value(value))),
        )
    end
    covector_scale > zero(R) || throw(DomainError(g, "g must be nonzero"))
    return T, R, dimension, covector_scale
end

@inline function _hessian_positive_root(value::Real, label::AbstractString)
    primal = _primal_value(value)
    (isfinite(primal) && primal > zero(primal)) || throw(DomainError(
        value, "H must be positive definite; nonpositive $label",
    ))
    root = sqrt(value)
    _number_isfinite(root) || throw(DomainError(
        value, "H Cholesky $label is non-finite",
    ))
    return root
end

@inline function _hessian_scaled_covector(g, index::Int, ::Type{T}, scale) where {T<:Real}
    return convert(T, g[index]) / (zero(T) + scale)
end

function _hessian_dual_norm_1(H, g, ::Type{T}, scale) where {T<:Real}
    l11 = _hessian_positive_root(
        _hessian_symmetric_entry(H, 1, 1, T), "first pivot",
    )
    return abs(_hessian_scaled_covector(g, 1, T, scale) / l11)
end

function _hessian_dual_norm_2(H, g, ::Type{T}, scale) where {T<:Real}
    h11 = _hessian_symmetric_entry(H, 1, 1, T)
    h21 = _hessian_symmetric_entry(H, 2, 1, T)
    h22 = _hessian_symmetric_entry(H, 2, 2, T)
    l11 = _hessian_positive_root(h11, "first pivot")
    l21 = h21 / l11
    l22 = _hessian_positive_root(
        muladd(-l21, l21, h22), "second pivot",
    )
    y1 = _hessian_scaled_covector(g, 1, T, scale) / l11
    y2 = (_hessian_scaled_covector(g, 2, T, scale) - l21 * y1) / l22
    return hypot(y1, y2)
end

function _hessian_dual_norm_3(H, g, ::Type{T}, scale) where {T<:Real}
    h11 = _hessian_symmetric_entry(H, 1, 1, T)
    h21 = _hessian_symmetric_entry(H, 2, 1, T)
    h31 = _hessian_symmetric_entry(H, 3, 1, T)
    h22 = _hessian_symmetric_entry(H, 2, 2, T)
    h32 = _hessian_symmetric_entry(H, 3, 2, T)
    h33 = _hessian_symmetric_entry(H, 3, 3, T)

    l11 = _hessian_positive_root(h11, "first pivot")
    l21 = h21 / l11
    l31 = h31 / l11
    l22 = _hessian_positive_root(
        muladd(-l21, l21, h22), "second pivot",
    )
    l32 = muladd(-l31, l21, h32) / l22
    third_pivot = muladd(-l32, l32, muladd(-l31, l31, h33))
    l33 = _hessian_positive_root(third_pivot, "third pivot")

    y1 = _hessian_scaled_covector(g, 1, T, scale) / l11
    y2 = (_hessian_scaled_covector(g, 2, T, scale) - l21 * y1) / l22
    y3 = (_hessian_scaled_covector(g, 3, T, scale) - l31 * y1 - l32 * y2) / l33
    return hypot(hypot(y1, y2), y3)
end

function _hessian_dual_norm_generic(H, g, ::Type{T}, dimension::Int, scale) where {T<:Real}
    workspace = Matrix{T}(undef, dimension, dimension)
    @inbounds for column in 1:dimension, row in 1:dimension
        workspace[row, column] = _hessian_symmetric_entry(H, row, column, T)
    end
    factor = cholesky!(Hermitian(workspace); check=false)
    issuccess(factor) || throw(DomainError(H, "H must be positive definite"))

    transformed = Vector{T}(undef, dimension)
    @inbounds for index in 1:dimension
        transformed[index] = _hessian_scaled_covector(g, index, T, scale)
    end
    ldiv!(factor.L, transformed)
    dual_norm = norm(transformed)
    primal = _primal_value(dual_norm)
    (isfinite(primal) && primal > zero(primal)) || throw(DomainError(
        (H, g), "Hessian dual norm must be finite and positive",
    ))
    return dual_norm
end

function _hessian_scaled_dual_norm(
    H::AbstractMatrix{TH},
    g::AbstractVector{TG},
) where {TH<:Real,TG<:Real}
    T, _, dimension, scale = _validate_hessian_metric_inputs(H, g)
    dual_norm = if dimension == 1
        _hessian_dual_norm_1(H, g, T, scale)
    elseif dimension == 2
        _hessian_dual_norm_2(H, g, T, scale)
    elseif dimension == 3
        _hessian_dual_norm_3(H, g, T, scale)
    else
        _hessian_dual_norm_generic(H, g, T, dimension, scale)
    end
    primal = _primal_value(dual_norm)
    (isfinite(primal) && primal > zero(primal)) || throw(DomainError(
        (H, g), "Hessian dual norm must be finite and positive",
    ))
    return dual_norm, scale
end

@inline function _hessian_scaled_ratio(
    numerator::T,
    first_denominator::T,
    second_denominator::T,
) where {T<:AbstractFloat}
    iszero(numerator) && return numerator
    numerator_mantissa, numerator_exponent = frexp(numerator)
    first_mantissa, first_exponent = frexp(first_denominator)
    second_mantissa, second_exponent = frexp(second_denominator)
    mantissa = (numerator_mantissa / first_mantissa) / second_mantissa
    return ldexp(
        mantissa,
        numerator_exponent - first_exponent - second_exponent,
    )
end

@inline function _hessian_scaled_ratio(numerator, first_denominator, second_denominator)
    denominator = first_denominator * second_denominator
    if isfinite(_primal_value(denominator)) && !iszero(_primal_value(denominator))
        return numerator / denominator
    end
    return abs(_primal_value(first_denominator)) >=
           abs(_primal_value(second_denominator)) ?
           (numerator / first_denominator) / second_denominator :
           (numerator / second_denominator) / first_denominator
end

@inline function _hessian_coordinate_from_scaled_norm(delta, dual_norm, scale)
    numerator, first_denominator, second_denominator = promote(
        float(delta), dual_norm, zero(dual_norm) + scale,
    )
    return _hessian_scaled_ratio(
        numerator, first_denominator, second_denominator,
    )
end

"""
    hessian_metric_q2(H, g)

Return the coordinate-invariant dual metric `g' * (H \\ g)` for a finite,
real, symmetric positive-definite Hessian `H` and a finite nonzero covector
`g`. The implementation uses a Cholesky solve and never forms `inv(H)`.
"""
function hessian_metric_q2(
    H::AbstractMatrix{TH},
    g::AbstractVector{TG},
) where {TH<:Real,TG<:Real}
    dual_norm, scale = _hessian_scaled_dual_norm(H, g)
    value = dual_norm * (zero(dual_norm) + scale)
    return value * value
end

"""
    hessian_effective_L(H, g)

Return the Hessian effective distance `1 / (g' * (H \\ g))`. The inverse
dual-norm length is formed before squaring so representable extreme distances
are not lost through an overflowing or underflowing intermediate `q2`.
"""
function hessian_effective_L(
    H::AbstractMatrix{TH},
    g::AbstractVector{TG},
) where {TH<:Real,TG<:Real}
    dual_norm, scale = _hessian_scaled_dual_norm(H, g)
    one_value = one(dual_norm)
    inverse_norm = _hessian_scaled_ratio(
        one_value, dual_norm, zero(dual_norm) + scale,
    )
    return inverse_norm * inverse_norm
end

"""
    hessian_transition_coordinate(delta, H, g)

Return the signed canonical coordinate
`delta / sqrt(g' * (H \\ g))`. `delta` must be finite. Its sign is retained;
changing only the sign of `g` leaves the coordinate unchanged.
"""
function hessian_transition_coordinate(
    delta::Real,
    H::AbstractMatrix{TH},
    g::AbstractVector{TG},
) where {TH<:Real,TG<:Real}
    isfinite(_primal_value(delta)) ||
        throw(DomainError(delta, "delta must be finite"))
    dual_norm, scale = _hessian_scaled_dual_norm(H, g)
    return _hessian_coordinate_from_scaled_norm(delta, dual_norm, scale)
end

"""
    hessian_transition_argument(k, delta, H, g)

Return the nonnegative UTD transition argument
`k * delta^2 / (2 * g' * (H \\ g))` for finite positive `k`. Use
[`F_utd`](@ref) to evaluate the corresponding canonical transition.
"""
function hessian_transition_argument(
    k::Real,
    delta::Real,
    H::AbstractMatrix{TH},
    g::AbstractVector{TG},
) where {TH<:Real,TG<:Real}
    k_primal = _primal_value(k)
    (isfinite(k_primal) && k_primal > zero(k_primal)) || throw(DomainError(
        k, "k must be finite and positive",
    ))
    coordinate = hessian_transition_coordinate(delta, H, g)
    k_value, coordinate_value = promote(float(k), coordinate)
    scaled_coordinate = coordinate_value * sqrt(k_value) / sqrt(2one(k_value))
    return scaled_coordinate * scaled_coordinate
end

"""
    directional_effective_L(R1, R2, beta)

Return the directional harmonic projection
`(cos(beta)^2/R1 + sin(beta)^2/R2)^(-1)` of two positive finite principal
effective distances. The ratio-ordered form avoids intermediate radius
products and preserves the promoted input type.
"""
function directional_effective_L(R1::Real, R2::Real, beta::Real)
    first_radius, second_radius, angle = promote(
        float(R1), float(R2), float(beta),
    )
    first_primal = _primal_value(first_radius)
    second_primal = _primal_value(second_radius)
    angle_primal = _primal_value(angle)
    (isfinite(first_primal) && first_primal > zero(first_primal)) ||
        throw(DomainError(R1, "R1 must be finite and positive"))
    (isfinite(second_primal) && second_primal > zero(second_primal)) ||
        throw(DomainError(R2, "R2 must be finite and positive"))
    isfinite(angle_primal) || throw(DomainError(beta, "beta must be finite"))

    first_radius == second_radius && return first_radius
    cosine_squared = cos(angle)^2
    sine_squared = sin(angle)^2
    if first_primal <= second_primal
        return first_radius /
               (cosine_squared + sine_squared * (first_radius / second_radius))
    end
    return second_radius /
           (sine_squared + cosine_squared * (second_radius / first_radius))
end
