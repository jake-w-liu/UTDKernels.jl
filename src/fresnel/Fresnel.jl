"""
Fresnel reflection coefficients for planar impedance surfaces.

Convention: exp(+iωt), using `im` (= i) throughout.

All angles are **grazing angles** ψ measured from the surface (NOT from the
surface normal). The relation is ψ = π/2 − θ_i where θ_i is the conventional
angle of incidence from the normal.

The effective complex relative permittivity folds conductivity into the
imaginary part:
    ε_r_eff = ε_r − i σ / (ω ε₀)

For our exp(+iωt) convention, a lossy material has Im(ε_r_eff) < 0.
"""

"""
    WedgeFaceMaterial(eps_r)
    WedgeFaceMaterial(eps_r_real, sigma, freq)

Material description for one face of an impedance wedge.

Stores the effective complex relative permittivity.  When constructed from
real permittivity + conductivity + frequency, the effective ε_r is:
    ε_r_eff = ε_r − i σ / (ω ε₀)
with the sign chosen for exp(+iωt) convention (lossy → negative imaginary).
"""
struct WedgeFaceMaterial{T<:Number}
    eps_r::T  # complex relative permittivity (includes loss)
    function WedgeFaceMaterial(eps_r::T) where {T<:Number}
        _validate_finite_number(eps_r, "relative permittivity")
        new{T}(eps_r)
    end
end

function WedgeFaceMaterial(eps_r_real::Real, sigma::Real, freq::Real)
    isfinite(eps_r_real) || throw(DomainError(eps_r_real, "relative permittivity must be finite"))
    isfinite(sigma) && sigma >= zero(sigma) ||
        throw(DomainError(sigma, "conductivity must be finite and nonnegative"))
    isfinite(freq) && freq > zero(freq) ||
        throw(DomainError(freq, "frequency must be finite and positive"))
    # ε₀ = 8.854187817e-12 F/m
    eps0 = 8.854187817e-12
    # exp(+iωt) convention: ε_eff = ε_r - i σ/(ω ε₀)
    # i = +im, so -i·σ/(ωε₀) → negative imaginary part for lossy materials.
    # Divide by `freq` before forming the fixed 2πε₀ factor. Computing
    # ω = 2πf first can overflow for a valid finite frequency and erase a
    # loss term that remains representable in the documented ratio.
    loss_ratio = (sigma / freq) / (2π * eps0)
    eps_r_eff = complex(eps_r_real, -loss_ratio)
    WedgeFaceMaterial(eps_r_eff)
end

"""
    ImpedanceWedge(alpha, face_o, face_n)

Wedge with exterior angle `alpha` ∈ (0, 2π] and material properties on each
face.  Face o is at φ=0, face n is at φ=α.
"""
struct ImpedanceWedge{T<:Real, M<:Number}
    alpha::T
    face_o::WedgeFaceMaterial{M}  # o-face (φ = 0)
    face_n::WedgeFaceMaterial{M}  # n-face (φ = α)
    function ImpedanceWedge(alpha::T, face_o::WedgeFaceMaterial{M},
                            face_n::WedgeFaceMaterial{M}) where {T<:Real, M<:Number}
        0 < alpha <= 2π || throw(DomainError(alpha, "Wedge angle must be in (0, 2π]"))
        new{T, M}(alpha, face_o, face_n)
    end
end

function ImpedanceWedge(alpha::T, face_o::WedgeFaceMaterial{M1},
                        face_n::WedgeFaceMaterial{M2}) where {T<:Real,M1<:Number,M2<:Number}
    eps_o, eps_n = promote(face_o.eps_r, face_n.eps_r)
    return ImpedanceWedge(alpha, WedgeFaceMaterial(eps_o), WedgeFaceMaterial(eps_n))
end

# Convenience: same material on both faces
function ImpedanceWedge(alpha::Real, mat::WedgeFaceMaterial)
    ImpedanceWedge(alpha, mat, mat)
end

# Convert to base Wedge for geometry computations
_to_wedge(iw::ImpedanceWedge) = Wedge(iw.alpha)

"""
    fresnel_te(psi, eps_r)

TE (perpendicular / soft) Fresnel reflection coefficient at grazing angle ψ.

    R_TE = [sin(ψ) − √(ε_r − cos²(ψ))] / [sin(ψ) + √(ε_r − cos²(ψ))]

PEC limit (|ε_r| → ∞): R_TE → −1.

Both inputs must be finite. A finite request whose formula has a singular
denominator raises `DomainError` instead of returning `NaN` or `Inf`.
"""
function fresnel_te(psi::Number, eps_r::Number)
    _validate_finite_number(psi, "grazing angle psi")
    _validate_finite_number(eps_r, "relative permittivity")
    sin_psi = sin(psi)
    eps_r == one(eps_r) && return zero(safe_sqrt(eps_r))
    # Use safe_sqrt for branch safety (principal branch)
    # eps_r - cos(psi)^2 = (eps_r - 1) + sin(psi)^2; the latter avoids
    # catastrophic cancellation for matched media near grazing.
    eta = safe_sqrt((eps_r - one(eps_r)) + sin_psi^2)
    result = (sin_psi - eta) / (sin_psi + eta)
    _number_isfinite(result) || throw(DomainError(
        (psi, eps_r),
        "TE Fresnel coefficient is non-finite at the requested parameters",
    ))
    return result
end

"""
    fresnel_tm(psi, eps_r)

TM (parallel / hard) Fresnel reflection coefficient at grazing angle ψ.

    R_TM = [ε_r sin(ψ) − √(ε_r − cos²(ψ))] / [ε_r sin(ψ) + √(ε_r − cos²(ψ))]

PEC limit (|ε_r| → ∞): R_TM → +1.

Both inputs must be finite. A finite request whose formula has a singular
denominator raises `DomainError` instead of returning `NaN` or `Inf`.
"""
function fresnel_tm(psi::Number, eps_r::Number)
    _validate_finite_number(psi, "grazing angle psi")
    _validate_finite_number(eps_r, "relative permittivity")
    sin_psi = sin(psi)
    eps_r == one(eps_r) && return zero(safe_sqrt(eps_r))
    eta = safe_sqrt((eps_r - one(eps_r)) + sin_psi^2)
    result = (eps_r * sin_psi - eta) / (eps_r * sin_psi + eta)
    _number_isfinite(result) || throw(DomainError(
        (psi, eps_r),
        "TM Fresnel coefficient is non-finite at the requested parameters",
    ))
    return result
end
