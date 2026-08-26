"""
Core types for UTDKernels.jl
"""

# Angular domain constants must be rounded in the active scalar type. Comparing
# Float32(2π) or a high-precision BigFloat(2π) with the Float64 value produced
# by the literal `2π` rejects a valid half-plane endpoint. Form 2π from the
# type-local representation of π so both inclusive and strict bounds are exact
# in the caller's numerical type and, for BigFloat, its stored precision.
@inline function _angular_primal(x::Real)
    hasproperty(x, :value) && return _angular_primal(getproperty(x, :value))
    return x
end

@inline _typed_pi_primal(x::Real) = oftype(float(x), π)
@inline _typed_pi_primal(x::Irrational{:π}) = x
@inline function _typed_pi_primal(x::BigFloat)
    return setprecision(BigFloat, precision(x)) do
        BigFloat(π)
    end
end

@inline _typed_two_pi_primal(x::Real) = 2 * _typed_pi_primal(x)
@inline function _typed_two_pi_primal(x::BigFloat)
    return setprecision(BigFloat, precision(x)) do
        2 * BigFloat(π)
    end
end

@inline _typed_pi(x::Real) = _typed_pi_primal(_angular_primal(x))
@inline _typed_two_pi(x::Real) = _typed_two_pi_primal(_angular_primal(x))

"""
    PhasorConvention(sgn)

Time-harmonic phasor convention. `sgn = +1` for exp(+iωt).
"""
struct PhasorConvention
    sgn::Int
    function PhasorConvention(sgn::Int)
        sgn in (-1, +1) || throw(DomainError(sgn, "phasor sign must be -1 or +1"))
        new(sgn)
    end
end

"""
    EXP_IWT

Default phasor convention constant for ``exp(+i \\omega t)``.
"""
const EXP_IWT = PhasorConvention(+1)

"""
    Wedge(alpha)

Wedge with exterior (free-space) angle `alpha` ∈ (0, 2π].
Faces at φ=0 and φ=alpha. The half-plane corresponds to alpha=2π.
The inclusive endpoint is interpreted in `alpha`'s active floating-point type.
"""
struct Wedge{T<:Real}
    alpha::T
    function Wedge(alpha::T) where T<:Real
        alpha_primal = _angular_primal(alpha)
        zero(alpha_primal) < alpha_primal <= _typed_two_pi(alpha) ||
            throw(DomainError(alpha, "Wedge angle must be in (0, 2π]"))
        new{T}(alpha)
    end
end

"""Wedge parameter n = α/π (standard KP parameter)."""
wedge_n(w::Wedge) = w.alpha / π

"""Wedge parameter ν = π/α."""
wedge_nu(w::Wedge) = π / w.alpha

"""
    RayAngles(phi, phip)

Observation azimuth `phi` and incident azimuth `phip`.
Inputs may be any real angles; kernel routines map them to the wedge interval.
"""
struct RayAngles{T<:Real}
    phi::T
    phip::T
end
RayAngles(phi::T1, phip::T2) where {T1<:Real, T2<:Real} =
    RayAngles(promote(phi, phip)...)

@inline function _validate_distance(x::Real, name::AbstractString)
    x_primal = _angular_primal(x)
    (x_primal > zero(x_primal) && (isfinite(x_primal) || isinf(x_primal))) ||
        throw(DomainError(x, "$name must be positive and finite or +Inf"))
    return x
end

"""
    Distances(s, sp)

`s`: edge-to-observer distance. `sp`: source-to-edge distance (Inf for plane wave).
"""
struct Distances{T<:Real}
    s::T
    sp::T
    function Distances(s::T, sp::T) where {T<:Real}
        _validate_distance(s, "observer distance s")
        _validate_distance(sp, "source distance sp")
        new{T}(s, sp)
    end
end
Distances(s::T1, sp::T2) where {T1<:Real,T2<:Real} = Distances(promote(s, sp)...)

"""Effective distance parameter L = s·s'/(s+s')."""
function effective_L(d::Distances)
    isinf(d.s) && return d.sp
    if isinf(d.sp)
        return d.s
    end
    # Scale by the larger distance. This is algebraically s*sp/(s+sp) but
    # avoids overflowing the product when both finite distances are large.
    return d.s <= d.sp ? d.s / (1 + d.s / d.sp) : d.sp / (1 + d.sp / d.s)
end
