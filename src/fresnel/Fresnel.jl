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
end

function WedgeFaceMaterial(eps_r_real::Real, sigma::Real, freq::Real)
    # ε₀ = 8.854187817e-12 F/m
    eps0 = 8.854187817e-12
    omega = 2π * freq
    # exp(+iωt) convention: ε_eff = ε_r - i σ/(ω ε₀)
    # i = +im, so -i·σ/(ωε₀) → negative imaginary part for lossy materials.
    eps_r_eff = complex(eps_r_real, -sigma / (omega * eps0))
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
"""
function fresnel_te(psi::Number, eps_r::Number)
    sin_psi = sin(psi)
    cos_psi = cos(psi)
    # Use safe_sqrt for branch safety (principal branch)
    eta = safe_sqrt(eps_r - cos_psi^2)
    return (sin_psi - eta) / (sin_psi + eta)
end

"""
    fresnel_tm(psi, eps_r)

TM (parallel / hard) Fresnel reflection coefficient at grazing angle ψ.

    R_TM = [ε_r sin(ψ) − √(ε_r − cos²(ψ))] / [ε_r sin(ψ) + √(ε_r − cos²(ψ))]

PEC limit (|ε_r| → ∞): R_TM → +1.
"""
function fresnel_tm(psi::Number, eps_r::Number)
    sin_psi = sin(psi)
    cos_psi = cos(psi)
    eta = safe_sqrt(eps_r - cos_psi^2)
    return (eps_r * sin_psi - eta) / (eps_r * sin_psi + eta)
end
