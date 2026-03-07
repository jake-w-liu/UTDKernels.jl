"""
Exact impedance-wedge diffraction coefficient via the Maliuzhinets spectral
function method (Kotelnikov et al., Phys. Rev. A 87, 023828, 2013).

The diffraction coefficient in the far field is obtained from the saddle-point
evaluation of the Sommerfeld integral:

    D = C(k) × [s(θ+π) − s(θ-π)]

where C(k) = −e^{−iπ/4}/√(2πk) is the universal saddle-point prefactor, and
the spectral function is (Kotelnikov Eq. 24):

    s(u) = σ(u) · Ψ₀(u) / Ψ₀(θ₀)

with σ(u) = ν cos(νθ₀) / (sin(νu) − sin(νθ₀))  (Kotelnikov Eq. 26),
ν = π/(2Φ), and Ψ₀ is a product of four Maliuzhinets functions encoding
the impedance boundary conditions on both faces (Kotelnikov Eqs. 19–20).

Angle convention mapping:
  Norris–Osipov / Kotelnikov: θ ∈ (−Φ, Φ), faces at ±Φ, 2Φ = exterior angle.
  UTDKernels:                 φ ∈ (0, α),   faces at 0 (o) and α (n).
  Shift:                      θ_NO = φ_UTD − α/2.
  Face mapping:               + face (at +Φ) = n-face,  − face (at −Φ) = o-face.

Impedance angle convention (Kotelnikov):
  BC on each face: ∂B/∂n + ik₀ sin(χ) B = 0.
  χ = 0 → Neumann (hard), χ → π/2+i∞ → Dirichlet (soft).
  For TE (H_z): sin(χ_TE) = η_s = Z_s/Z₀   → PEC: χ=0 (Neumann for H_z)
  For TM (E_z): sin(χ_TM) = 1/η_s = Z₀/Z_s → PEC: χ→∞ (Dirichlet for E_z)
"""

# ──────────────────────────────────────────────────────────────────────
# Psi product: Ψ₀(u) = Ψ(u,χ₊)·Ψ(u−2Φ,χ₋)  (Kotelnikov Eqs. 19–20)
# ──────────────────────────────────────────────────────────────────────

"""
    _Psi_product(z, Phi, chi_p, chi_m; rtol)

Product of four Maliuzhinets functions (Kotelnikov Eqs. 19–20):

    Ψ₀(z) = ψ_Φ(z+Φ+π/2−χ₊)·ψ_Φ(z+Φ−π/2+χ₊)·ψ_Φ(z−Φ+π/2−χ₋)·ψ_Φ(z−Φ−π/2+χ₋)

χ₊ = impedance angle on the + face (n-face), χ₋ = impedance angle on − face (o-face).
"""
function _Psi_product(z::Number, Phi::Real, chi_p::Number, chi_m::Number;
                      rtol::Real = 1e-12)
    psi_Phi(z + Phi + π/2 - chi_p, Phi; rtol = rtol) *
    psi_Phi(z + Phi - π/2 + chi_p, Phi; rtol = rtol) *
    psi_Phi(z - Phi - π/2 + chi_m, Phi; rtol = rtol) *
    psi_Phi(z - Phi + π/2 - chi_m, Phi; rtol = rtol)
end

# ──────────────────────────────────────────────────────────────────────
# Spectral function and diffraction coefficient
# ──────────────────────────────────────────────────────────────────────

"""
    _spectral_D(theta_NO, theta0_NO, Phi, chi_p, chi_m, k; rtol)

Compute the far-field diffraction coefficient for a single polarization via
the Maliuzhinets spectral function.

Returns D = C(k) × [s(θ+π) − s(θ−π)] where s(u) = σ(u)·Ψ₀(u)/Ψ₀(θ₀).
"""
function _spectral_D(theta_NO::Number, theta0_NO::Number, Phi::Real,
                     chi_p::Number, chi_m::Number, k::Real;
                     rtol::Real = 1e-12)
    nu = π / (2Phi)
    Psi0_t0 = _Psi_product(theta0_NO, Phi, chi_p, chi_m; rtol = rtol)

    # Spectral function s(u) = σ(u) · Ψ₀(u) / Ψ₀(θ₀)
    function _s(u)
        sigma = nu * cos(nu * theta0_NO) / (sin(nu * u) - sin(nu * theta0_NO))
        Psi0_u = _Psi_product(u, Phi, chi_p, chi_m; rtol = rtol)
        return sigma * Psi0_u / Psi0_t0
    end

    # Saddle-point prefactor (Kotelnikov Eqs. 40–41)
    C = -exp(-im * π / 4) / sqrt(2π * k)

    return C * (_s(theta_NO + π) - _s(theta_NO - π))
end

# ──────────────────────────────────────────────────────────────────────
# Impedance angle from complex permittivity
# ──────────────────────────────────────────────────────────────────────

"""
    _impedance_angle_TE(eps_r)

Impedance angle for TE (hard) polarization: sin(χ) = η_s = 1/√ε_r.
For PEC (ε_r → ∞): χ → 0 (Neumann for H_z).
"""
_impedance_angle_TE(eps_r::Number) = asin(1 / sqrt(Complex(eps_r)))

"""
    _impedance_angle_TM(eps_r)

Impedance angle for TM (soft) polarization: sin(χ) = 1/η_s = √ε_r.
For PEC (ε_r → ∞): χ → π/2 + i∞ (Dirichlet for E_z).
"""
_impedance_angle_TM(eps_r::Number) = asin(sqrt(Complex(eps_r)))

# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

"""
    maliuzhinets_DsDh(alpha, eps_r_o, eps_r_n, phi, phip, k; rtol=1e-12)

Exact impedance-wedge diffraction coefficients via the Maliuzhinets spectral
function method, in the Kouyoumjian–Pathak convention.

Returns `(Ds, Dh)` — soft and hard coefficients.

# Arguments
- `alpha`: exterior wedge angle ∈ (π, 2π)
- `eps_r_o`: complex relative permittivity of the o-face (φ = 0)
- `eps_r_n`: complex relative permittivity of the n-face (φ = α)
- `phi`: observation azimuth (radians, in [0, α])
- `phip`: source azimuth (radians, in [0, α])
- `k`: wavenumber (real, positive)

# Notes
The soft coefficient D_s uses TM impedance angles (sin χ = √ε_r) and is
well-validated only for finite impedance. For PEC soft, use `pec_wedge_DsDh`.
The hard coefficient D_h uses TE impedance angles (sin χ = 1/√ε_r) and
reduces exactly to the PEC D_h at ε_r → ∞.
"""
function maliuzhinets_DsDh(
    alpha::Real,
    eps_r_o::Number,
    eps_r_n::Number,
    phi::Real,
    phip::Real,
    k::Real;
    rtol::Real = 1e-12,
)
    π < alpha < 2π || throw(DomainError(alpha, "alpha must satisfy π < alpha < 2π"))
    k > 0 || throw(DomainError(k, "k must be positive"))
    0 <= phi <= alpha || throw(DomainError(phi, "phi must satisfy 0 <= phi <= alpha"))
    0 <= phip <= alpha || throw(DomainError(phip, "phip must satisfy 0 <= phip <= alpha"))

    Phi = alpha / 2

    # Convert to Norris–Osipov centered angles
    theta_NO  = phi  - Phi
    theta0_NO = phip - Phi

    # TE → D_h (hard / Neumann-like): sin(χ) = 1/√ε_r
    chi_TE_n = _impedance_angle_TE(eps_r_n)
    chi_TE_o = _impedance_angle_TE(eps_r_o)
    Dh = _spectral_D(theta_NO, theta0_NO, Phi, chi_TE_n, chi_TE_o, k; rtol = rtol)

    # TM → D_s (soft / Dirichlet-like): sin(χ) = √ε_r
    chi_TM_n = _impedance_angle_TM(eps_r_n)
    chi_TM_o = _impedance_angle_TM(eps_r_o)
    Ds = _spectral_D(theta_NO, theta0_NO, Phi, chi_TM_n, chi_TM_o, k; rtol = rtol)

    return (Ds, Dh)
end
