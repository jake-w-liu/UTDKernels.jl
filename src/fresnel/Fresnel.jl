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

@inline function _scaled_loss_ratio(σ::T, f::T, scale::T) where {T<:AbstractFloat}
    iszero(σ) && return zero(T)
    mσ, eσ = frexp(σ)
    mf, ef = frexp(f)
    ms, es = frexp(scale)
    return ldexp((mσ / mf) / ms, eσ - ef - es)
end

@inline _scaled_loss_ratio(σ::Real, f::Real, scale::Real) = (σ / f) / scale

@inline function _material_loss_ratio(sigma::Real, freq::Real, eps0::Real)
    scale = 2π * eps0
    σ, f, c = promote(float(sigma), float(freq), scale)
    return _scaled_loss_ratio(σ, f, c)
end

function WedgeFaceMaterial(eps_r_real::Real, sigma::Real, freq::Real)
    isfinite(eps_r_real) || throw(DomainError(eps_r_real, "relative permittivity must be finite"))
    sigma_primal = _primal_value(sigma)
    freq_primal = _primal_value(freq)
    isfinite(sigma_primal) && sigma_primal >= zero(sigma_primal) ||
        throw(DomainError(sigma, "conductivity must be finite and nonnegative"))
    isfinite(freq_primal) && freq_primal > zero(freq_primal) ||
        throw(DomainError(freq, "frequency must be finite and positive"))
    # ε₀ = 8.854187817e-12 F/m
    eps0 = 8.854187817e-12
    # exp(+iωt) convention: ε_eff = ε_r - i σ/(ω ε₀)
    # i = +im, so -i·σ/(ωε₀) → negative imaginary part for lossy materials.
    # Exponent-scaled division avoids losing a representable ratio through either
    # overflow in 2πf or underflow in σ/f.
    loss_ratio = _material_loss_ratio(sigma, freq, eps0)
    eps_r_eff = complex(eps_r_real, -loss_ratio)
    WedgeFaceMaterial(eps_r_eff)
end

"""
    ImpedanceWedge(alpha, face_o, face_n)

Wedge with exterior angle `alpha` ∈ (0, 2π] and material properties on each
face.  Face o is at φ=0, face n is at φ=α.
The inclusive endpoint is interpreted in `alpha`'s active floating-point type.
"""
struct ImpedanceWedge{T<:Real, M<:Number}
    alpha::T
    face_o::WedgeFaceMaterial{M}  # o-face (φ = 0)
    face_n::WedgeFaceMaterial{M}  # n-face (φ = α)
    function ImpedanceWedge(alpha::T, face_o::WedgeFaceMaterial{M},
                            face_n::WedgeFaceMaterial{M}) where {T<:Real, M<:Number}
        alpha_primal = _angular_primal(alpha)
        zero(alpha_primal) < alpha_primal <= _typed_two_pi(alpha) ||
            throw(DomainError(alpha, "Wedge angle must be in (0, 2π]"))
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

@inline _fresnel_cos(psi::Number) = cos(psi)

@inline function _fresnel_cos(psi::Real)
    s = sin(psi)
    c = cos(psi)
    abs(_primal_value(c)) >= abs(_primal_value(s)) && return c

    # Near an odd multiple of π/2, form cosine as a sine of the small local
    # offset. This makes cos(±π/2) exactly zero in the input precision and
    # avoids a library-constant residual dominating epsilon-near-zero media.
    psi0 = _primal_value(psi)
    T = typeof(float(psi0))
    πT = T(π)
    q = try
        round(Int, (psi0 - πT / 2) / πT)
    catch err
        (err isa InexactError || err isa OverflowError) || rethrow()
        return c
    end
    centre0 = (T(q) + T(0.5)) * πT
    delta = psi - (zero(psi) + centre0)
    return iseven(q) ? -sin(delta) : sin(delta)
end

@inline function _fresnel_eta(psi::Number, eps_r::Number)
    sin_psi = sin(psi)
    cos_psi = _fresnel_cos(psi)
    # Choose the algebraic form that avoids subtracting two nearly equal O(1)
    # quantities in the active angular regime.
    radicand = if abs(cos_psi) <= abs(sin_psi)
        eps_r - cos_psi^2
    else
        (eps_r - one(eps_r)) + sin_psi^2
    end
    return sin_psi, radiation_sqrt(radicand)
end

"""
    fresnel_te(psi, eps_r)

TE (perpendicular / soft) Fresnel reflection coefficient at grazing angle ψ.

    R_TE = [sin(ψ) − √(ε_r − cos²(ψ))] / [sin(ψ) + √(ε_r − cos²(ψ))]

PEC limit (|ε_r| → ∞): R_TE → −1.

Both inputs must be finite. A finite request whose formula has a singular
denominator raises `DomainError` instead of returning `NaN` or `Inf`.
For a lossless negative-real radicand, the square root uses the passive
`Im(ε_r) → 0⁻` limiting value required by the package's `exp(+iωt)` convention.
"""
function fresnel_te(psi::Number, eps_r::Number)
    _validate_finite_number(psi, "grazing angle psi")
    _validate_finite_number(eps_r, "relative permittivity")
    eps_r == one(eps_r) && return zero(safe_sqrt(eps_r))
    sin_psi, eta = _fresnel_eta(psi, eps_r)
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
For a lossless negative-real radicand, the square root uses the passive
`Im(ε_r) → 0⁻` limiting value required by the package's `exp(+iωt)` convention.
"""
function fresnel_tm(psi::Number, eps_r::Number)
    _validate_finite_number(psi, "grazing angle psi")
    _validate_finite_number(eps_r, "relative permittivity")
    eps_r == one(eps_r) && return zero(safe_sqrt(eps_r))
    sin_psi, eta = _fresnel_eta(psi, eps_r)
    result = (eps_r * sin_psi - eta) / (eps_r * sin_psi + eta)
    _number_isfinite(result) || throw(DomainError(
        (psi, eps_r),
        "TM Fresnel coefficient is non-finite at the requested parameters",
    ))
    return result
end
