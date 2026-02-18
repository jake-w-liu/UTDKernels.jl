"""
Core types for UTDKernels.jl
"""

"""
    PhasorConvention(sgn)

Time-harmonic phasor convention. `sgn = +1` for exp(+iωt).
"""
struct PhasorConvention
    sgn::Int
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

"""
    Distances(s, sp)

`s`: edge-to-observer distance. `sp`: source-to-edge distance (Inf for plane wave).
"""
struct Distances{T<:Real}
    s::T
    sp::T
end

"""Effective distance parameter L = s·s'/(s+s')."""
function effective_L(d::Distances)
    if isinf(d.sp)
        return d.s
    end
    return d.s * d.sp / (d.s + d.sp)
end
