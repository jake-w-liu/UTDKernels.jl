using Test
using UTDKernels
using SpecialFunctions: erfc

# ═══════════════════════════════════════════════════════════════════════
# 1. F_utd against independent numerical quadrature
#
#   KP integral definition (eq. 12): F_KP(x) = 2j√x·e^{jx}·∫_{√x}^∞ e^{-jτ²}dτ
#
#   With j = +i (standard), F_utd(x) = F_KP(x).
#   We compute F_KP via contour rotation and verify F_utd = F_KP.
#
#   Contour rotation τ = √x + e^{-iπ/4}·σ gives:
#     F_KP(x) = 2i·e^{-iπ/4}·√x · ∫_0^∞ exp(-σ² + c·σ) dσ
#   where c = -√(2x)·(1 + i).  This is INDEPENDENT of erfc/erfcx.
# ═══════════════════════════════════════════════════════════════════════

function _F_kp_quadrature(x::Real)
    sqx = sqrt(x)
    c = -sqrt(2x) * (1 + im)   # from contour rotation with e^{-iπ/4}

    T = 12.0
    N = 200_000
    h = T / N

    g(sigma) = exp(-sigma^2 + c * sigma)

    s = 0.5 * g(0.0) + 0.5 * g(T)
    for k in 1:N-1
        s += g(k * h)
    end
    integral = s * h

    return 2im * exp(-im * π/4) * sqx * integral
end

@testset "F_utd vs independent numerical quadrature" begin
    # exp(+iωt) convention: F_utd(x) = F_KP(x)
    for x in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
        val = F_utd(x)
        ref = _F_kp_quadrature(x)
        err = abs(val - ref) / max(abs(ref), 1e-15)
        @test err < 1e-5
    end
end

# ═══════════════════════════════════════════════════════════════════════
# 2. Half-plane D vs exact GTD (sec formula, kL → ∞)
#
#   For n=2, summing KP cot pairs with F→1 gives:
#     cot(ψ₁) + cot(ψ₂) = 2·sec(β/2)
#   Prefactor C = -exp(-iπ/4)/(2√(2πk)), so:
#     D_s = -e^{-iπ/4}/(2√(2πk)) · [sec((φ-φ')/2) - sec((φ+φ')/2)]
#     D_h = -e^{-iπ/4}/(2√(2πk)) · [sec((φ-φ')/2) + sec((φ+φ')/2)]
# ═══════════════════════════════════════════════════════════════════════

function _halfplane_gtd_DsDh(phi, phip, k)
    C = -exp(-im * π/4) / (2 * sqrt(2π * k))
    sec_minus = 1.0 / cos((phi - phip) / 2)
    sec_plus  = 1.0 / cos((phi + phip) / 2)
    Ds = C * (sec_minus - sec_plus)
    Dh = C * (sec_minus + sec_plus)
    return (Ds, Dh)
end

@testset "Half-plane D vs GTD exact (kL → ∞)" begin
    w = Wedge(2π)
    k = 1e7
    L = 1.0

    for (phi, phip) in [(π/2, π/4), (π/3, π/6), (2π/3, π/5),
                         (π/2, π/3), (5π/6, π/4)]
        ang = RayAngles(phi, phip)
        Ds_utd, Dh_utd = pec_wedge_DsDh(w, ang, k, L)
        Ds_gtd, Dh_gtd = _halfplane_gtd_DsDh(phi, phip, k)

        rel_s = abs(Ds_utd - Ds_gtd) / abs(Ds_gtd)
        rel_h = abs(Dh_utd - Dh_gtd) / abs(Dh_gtd)
        @test rel_s < 1e-5
        @test rel_h < 1e-5
    end
end

# ═══════════════════════════════════════════════════════════════════════
# 3. UTD total field vs exact Sommerfeld (half-plane)
#
#   Exact: u_s = V(φ-φ') - V(φ+φ'),  u_h = V(φ-φ') + V(φ+φ')
#   where V(ψ) = exp(+ikρ cosψ)·erfc(-ξ·e^{+iπ/4})/2    [exp(+iωt) convention]
#         ξ = √(2kρ)·cos(ψ/2)
# ═══════════════════════════════════════════════════════════════════════

function _sommerfeld_V(k, rho, psi)
    xi = sqrt(2 * k * rho) * cos(psi / 2)
    z  = -xi * exp(+im * π/4)
    return exp(+im * k * rho * cos(psi)) * erfc(z) / 2
end

function _sommerfeld_exact(phi, phip, k, rho)
    V_minus = _sommerfeld_V(k, rho, phi - phip)
    V_plus  = _sommerfeld_V(k, rho, phi + phip)
    return (V_minus - V_plus, V_minus + V_plus)   # (soft, hard)
end

function _utd_total_field(phi, phip, k, rho)
    w = Wedge(2π)
    L = rho  # plane wave: sp=∞ → L=ρ

    isb = π + phip
    rsb = π - phip

    GO_s = zero(ComplexF64)
    GO_h = zero(ComplexF64)

    # Incident wave (present for φ < ISB)
    if phi < isb
        inc = exp(+im * k * rho * cos(phi - phip))
        GO_s += inc
        GO_h += inc
    end

    # Reflected wave from PEC face at φ=0 (present for φ < RSB, φ' < π)
    if phip < π && phi < rsb
        refl = exp(+im * k * rho * cos(phi + phip))
        GO_s += -refl   # soft: R = -1
        GO_h += +refl   # hard: R = +1
    end

    # Diffracted field (exp(+iωt): outgoing phase is exp(-ikρ))
    ang = RayAngles(phi, phip)
    Ds, Dh = pec_wedge_DsDh(w, ang, k, L)
    A = spreading_factor(rho, Inf)
    phase = exp(-im * k * rho)

    total_s = GO_s + Ds * A * phase
    total_h = GO_h + Dh * A * phase
    return (total_s, total_h)
end

@testset "UTD total field vs exact Sommerfeld (half-plane)" begin
    k = 10.0
    rho = 5.0   # kρ = 50

    for phip in [π/6, π/4, π/3]
        isb = π + phip
        rsb = π - phip

        # Test in each region + near boundaries
        test_phis = [
            rsb / 2,              # Region I (inc + refl)
            (rsb + isb) / 2,      # Region II (inc only)
            (isb + 2π) / 2,       # Region III (shadow)
            isb - 0.01,           # just before ISB
            isb + 0.01,           # just after ISB
            rsb - 0.01,           # just before RSB
            rsb + 0.01,           # just after RSB
        ]

        for phi in test_phis
            phi < 0.01 && continue
            phi > 2π - 0.01 && continue

            Es_utd, Eh_utd = _utd_total_field(phi, phip, k, rho)
            Es_ex, Eh_ex   = _sommerfeld_exact(phi, phip, k, rho)

            # For kρ=50, UTD should agree very well with exact solution
            if abs(Es_ex) > 1e-10
                @test abs(Es_utd - Es_ex) / abs(Es_ex) < 0.05
            else
                @test abs(Es_utd - Es_ex) < 1e-3
            end
            if abs(Eh_ex) > 1e-10
                @test abs(Eh_utd - Eh_ex) / abs(Eh_ex) < 0.05
            else
                @test abs(Eh_utd - Eh_ex) < 1e-3
            end
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════
# 4. GO + D total field continuity across ISB and RSB
# ═══════════════════════════════════════════════════════════════════════

@testset "GO+D total field continuity across ISB" begin
    k = 10.0
    rho = 1.0  # kρ = 10

    for phip in [π/6, π/4, π/3]
        isb = π + phip
        delta = 1e-3

        Es_m, Eh_m = _utd_total_field(isb - delta, phip, k, rho)
        Es_p, Eh_p = _utd_total_field(isb + delta, phip, k, rho)

        # Without UTD compensation, the GO jump is ~1 (incident wave drops).
        # With UTD, jump ~ O(delta · kρ) ≈ 0.01.
        @test abs(Es_p - Es_m) < 0.1
        @test abs(Eh_p - Eh_m) < 0.1
    end
end

@testset "GO+D total field continuity across RSB" begin
    k = 10.0
    rho = 1.0

    for phip in [π/6, π/4, π/3]
        rsb = π - phip
        delta = 1e-3

        Es_m, Eh_m = _utd_total_field(rsb - delta, phip, k, rho)
        Es_p, Eh_p = _utd_total_field(rsb + delta, phip, k, rho)

        @test abs(Es_p - Es_m) < 0.1
        @test abs(Eh_p - Eh_m) < 0.1
    end
end
