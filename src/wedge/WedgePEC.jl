"""
PEC wedge diffraction coefficients (Kouyoumjian–Pathak form).

Convention: exp(+iωt), outgoing ~ exp(−ikr).
"""

# PEC sign factors: σ_j for j=1..4
# Soft (Dirichlet): +1, +1, -1, -1
# Hard (Neumann):   +1, +1, +1, +1
const PEC_SIGMA_SOFT = (+1, +1, -1, -1)
const PEC_SIGMA_HARD = (+1, +1, +1, +1)

@inline function _complex_finite(z::Complex)
    return isfinite(real(z)) && isfinite(imag(z))
end

@inline function _transition_tolerances(::Type{T}) where {T<:Real}
    # Use machine-precision-derived windows; avoid hand-tuned absolute values.
    # a_j is quadratic in angular detuning, so sqrt(eps) is a practical
    # threshold for "numerically at transition" in Float64 arithmetic.
    τ = sqrt(eps(T))
    return (τa = τ, τs = τ)
end

@inline function _validate_effective_L(L::Number)
    if L isa Real
        L > 0 || throw(DomainError(L, "L must be positive (or +Inf for far-field limit)"))
        return
    end

    Lc = Complex(L)
    isfinite(real(Lc)) && isfinite(imag(Lc)) && abs(Lc) > 0 ||
        throw(DomainError(L, "Complex L must be finite and nonzero"))
    return
end

function _wrap_angle_centered(phi::Real, alpha::Real)
    w = mod(phi + alpha / 2, alpha) - alpha / 2
    return w == -alpha / 2 ? alpha / 2 : w
end

@inline function _kp_term_sign_pm(j::Int)
    # KP four-term ordering:
    # j=1,3 -> (π + β)/(2n)
    # j=2,4 -> (π - β)/(2n)
    return (j == 1 || j == 3) ? +1 : -1
end

@inline function _kp_transition_detuning(j::Int, terms, n::Real)
    s = _kp_term_sign_pm(j)
    β = terms.beta[j]
    Nj = terms.Nj[j]
    u = 2 * n * π * Nj - β
    target = s == +1 ? π : -π
    return u - target
end

function _effective_angles_for_kp(wedge::Wedge, ang::RayAngles)
    alpha = wedge.alpha
    phi  = wrap_angle(ang.phi, alpha)
    phip = wrap_angle(ang.phip, alpha)

    # At grazing incidence (φ' = 0 or φ' = α), collapse β⁻ = β⁺ = φ by
    # setting φ' exactly to zero.  Keep φ in the standard [0, α) range so
    # that the sign of sin(ψ₂) flips as φ crosses the ISB at π.  Centered
    # wrapping [-α/2, α/2) would place its branch cut at ±α/2, which for a
    # half-plane (α = 2π) coincides with the ISB and destroys the
    # compensating discontinuity that makes the total field continuous.
    if min(abs(phip), abs(alpha - phip)) <= DEFAULT_TRANSITION_TOL
        return (phi, zero(typeof(phi)))
    end
    return (phi, phip)
end

"""
    pec_wedge_prefactor(k, n)

Common amplitude prefactor C(k,n) = -exp(-iπ/4) / (2n√(2πk)).
"""
function pec_wedge_prefactor(k::Number, n::Real)
    -exp(-im * π/4) / (2 * n * sqrt(2π * k))
end

"""
    _cot_F_regularized(psi, a, k, L)

Compute cot(ψ)·F(kLa), regularized at shadow boundaries where
cot(ψ) → ±∞ and a → 0 simultaneously.

At shadow/reflection boundaries the cotangent pole in ψ_j coincides
with the zero of the distance parameter a_j.  The product cot(ψ)·F(X)
is finite in the limit, but the naïve `cot(ψ) * F(X)` overflows in
IEEE 754 arithmetic.  This helper reformulates the computation by
factoring out the cancellation:

    cot(ψ)·F(X) = cos(ψ) · [√(πX)/sin(ψ)] · e^{+iπ/4} · erfcx(z)

where the bracketed ratio √X/sin(ψ) → n·√(2kL) at the boundary.
"""
function _cot_F_regularized(
    psi::Real,
    a::Real,
    k::Number,
    L::Number;
    n::Real = 1.0,
    detuning::Real = 0.0,
)
    T = promote_type(Float64, typeof(float(real(psi))), typeof(float(real(a))),
                     typeof(float(real(k))), typeof(float(real(L))))
    CT = Complex{T}
    τ = _transition_tolerances(T)
    sin_psi = sin(psi)

    if isinf(L)
        # Exact infinite-distance (far-field) limit: F(kLa) -> 1.
        # At transition boundaries (a -> 0), individual cot terms are singular.
        # Return symmetric midpoint value to keep coefficients finite.
        if abs(a) <= τ.τa || abs(sin_psi) <= τ.τs
            return zero(CT)
        end
        return CT(cot(psi))
    end

    a_eval = a
    psi_eval = psi
    sin_eval = sin_psi

    if abs(a) <= τ.τa && abs(sin_psi) <= τ.τs
        # Exact transition sample: enforce a matched one-sided surrogate
        # (a ~ 2 n^2 (Δψ)^2) instead of midpoint-zero convention.
        # This keeps cot(ψ)·F(kLa) on a physically consistent finite branch.
        sgn = detuning == 0 ? one(T) : sign(detuning)
        dψ = max(τ.τs, 10 * eps(T))
        psi_eval = psi + sgn * dψ
        sin_eval = sin(psi_eval)
        a_eval = 2 * n^2 * dψ^2
    end

    X = k * L * a_eval

    # Numerically stable form:
    # cot(ψ)F(X) = cos(ψ) * [√(πX)/sin(ψ)] * e^{+iπ/4} * erfcx(e^{+iπ/4}√X)
    # This avoids overflow/underflow from multiplying cot(ψ) and F(X) separately.
    sqrtX = safe_sqrt(X)
    z = exp(+im * π/4) * sqrtX
    ratio = sqrt(π * Complex(X)) / sin_eval
    v = cos(psi_eval) * exp(+im * π/4) * erfcx(z) * ratio
    if _complex_finite(v)
        return v
    end

    # Fallback: direct form away from exact transition.
    vd = cot(psi_eval) * F_utd(X)
    if _complex_finite(vd)
        return vd
    end

    return zero(CT)
end

"""
    pec_wedge_DsDh(wedge, ang, k, L; convention=EXP_IWT) -> (Ds, Dh)

Compute the soft and hard scalar UTD diffraction coefficients for a
PEC wedge using the Kouyoumjian–Pathak four-term form.

# Arguments
- `wedge::Wedge`: wedge geometry
- `ang::RayAngles`: observation and incident azimuths
- `k::Number`: wavenumber (real positive for lossless media)
- `L::Real`: effective distance parameter s·s'/(s+s')

# Returns
- `(Ds, Dh)`: tuple of complex diffraction coefficients
"""
function pec_wedge_DsDh(
    wedge::Wedge,
    ang::RayAngles,
    k::Number,
    L::Number;
    convention::PhasorConvention = EXP_IWT,
)
    convention.sgn == +1 || error("Only exp(+iωt) convention is supported")
    _validate_effective_L(L)

    n = wedge_n(wedge)
    phi, phip = _effective_angles_for_kp(wedge, ang)

    terms = kp_four_terms(phi, phip, n)
    C = pec_wedge_prefactor(k, n)
    CT = typeof(C)

    Ds = zero(CT)
    Dh = zero(CT)

    for j in 1:4
        psi_j = terms.psi[j]
        a_j   = terms.aj[j]
        detuning_j = _kp_transition_detuning(j, terms, n)

        contrib = _cot_F_regularized(psi_j, a_j, k, L; n = n, detuning = detuning_j)
        Ds += PEC_SIGMA_SOFT[j] * contrib
        Dh += PEC_SIGMA_HARD[j] * contrib
    end

    return (C * Ds, C * Dh)
end

"""
    pec_wedge_DsDh(wedge, ang, k, Li, Lro, Lrn; Rs=-1, Rh=+1, convention=EXP_IWT)

Generalized PEC wedge coefficient with separate transition distances:
- `Li`  for incident-shadow terms (`D1`,`D2`)
- `Lrn` for reflection from face `n` (`D3`)
- `Lro` for reflection from face `o` (`D4`)

For PEC, use `Rs=-1`, `Rh=+1` (soft/hard reflection signs).
"""
function pec_wedge_DsDh(
    wedge::Wedge,
    ang::RayAngles,
    k::Number,
    Li::Real,
    Lro::Real,
    Lrn::Real;
    Rs::Number = -1,
    Rh::Number = +1,
    convention::PhasorConvention = EXP_IWT,
)
    convention.sgn == +1 || error("Only exp(+iωt) convention is supported")
    Li > 0 || throw(DomainError(Li, "Li must be positive"))
    Lro > 0 || throw(DomainError(Lro, "Lro must be positive"))
    Lrn > 0 || throw(DomainError(Lrn, "Lrn must be positive"))

    n = wedge_n(wedge)
    phi, phip = _effective_angles_for_kp(wedge, ang)

    terms = kp_four_terms(phi, phip, n)
    C = pec_wedge_prefactor(k, n)

    c1 = _cot_F_regularized(
        terms.psi[1],
        terms.aj[1],
        k,
        Li;
        n = n,
        detuning = _kp_transition_detuning(1, terms, n),
    )
    c2 = _cot_F_regularized(
        terms.psi[2],
        terms.aj[2],
        k,
        Li;
        n = n,
        detuning = _kp_transition_detuning(2, terms, n),
    )
    c3 = _cot_F_regularized(
        terms.psi[3],
        terms.aj[3],
        k,
        Lrn;
        n = n,
        detuning = _kp_transition_detuning(3, terms, n),
    )
    c4 = _cot_F_regularized(
        terms.psi[4],
        terms.aj[4],
        k,
        Lro;
        n = n,
        detuning = _kp_transition_detuning(4, terms, n),
    )

    common = c1 + c2
    refl = c3 + c4
    Ds = common + Rs * refl
    Dh = common + Rh * refl

    return (C * Ds, C * Dh)
end
