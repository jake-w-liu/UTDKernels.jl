"""
Core types for UTDKernels.jl
"""

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
"""
struct Wedge{T<:Real}
    alpha::T
    function Wedge(alpha::T) where T<:Real
        0 < alpha <= 2π || throw(DomainError(alpha, "Wedge angle must be in (0, 2π]"))
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
    (x > zero(x) && (isfinite(x) || isinf(x))) ||
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
