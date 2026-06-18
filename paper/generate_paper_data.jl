#!/usr/bin/env julia
"""
Generate CSV data files for UTDKernels.jl paper figures and tables.

Run from the paper/ directory:
  julia --project=.. generate_paper_data.jl

Output: data/*.csv
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using UTDKernels
using SpecialFunctions: erfc, erfcx
using ForwardDiff
using CSV, DataFrames

const DATA_DIR = joinpath(@__DIR__, "data")
mkpath(DATA_DIR)

# ────────────────────────────────────────────────────────────
# Exact Sommerfeld half-plane solution (for validation)
# ────────────────────────────────────────────────────────────
"""
Sommerfeld exact solution for PEC half-plane diffraction.

Uses the Pauli–Clemmow form (exp(+iωt) convention):
  V(ρ,ψ) = e^{+ikρ cos ψ} · erfc(-ξ e^{+iπ/4}) / 2
where ξ = √(2kρ) · cos(ψ/2).

Total field:
  u_s = V(φ-φ') - V(φ+φ')   (soft/Dirichlet)
  u_h = V(φ-φ') + V(φ+φ')   (hard/Neumann)
"""
function sommerfeld_V(k, rho, psi)
    xi = sqrt(2 * k * rho) * cos(psi / 2)
    z  = -xi * exp(+im * π/4)
    return exp(+im * k * rho * cos(psi)) * erfc(z) / 2
end

function sommerfeld_exact(k, rho, phi, phip; pol=:soft)
    V1 = sommerfeld_V(k, rho, phi - phip)
    V2 = sommerfeld_V(k, rho, phi + phip)
    return pol == :soft ? V1 - V2 : V1 + V2
end

# ────────────────────────────────────────────────────────────
# UTD total field for half-plane (GO + diffracted)
# ────────────────────────────────────────────────────────────
function utd_total_halfplane(k, rho, phi, phip; pol=:soft)
    u_i    = exp(+im * k * rho * cos(phi - phip))
    u_r_bare = exp(+im * k * rho * cos(phi + phip))
    R      = pol == :soft ? -1 : +1

    isb = π + phip
    rsb = π - phip

    u_GO = zero(ComplexF64)
    if phi < isb
        u_GO += u_i
    end
    if phi < rsb
        u_GO += R * u_r_bare
    end

    w   = Wedge(2π)
    ang = RayAngles(phi, phip)
    L   = rho   # plane wave: L = s = ρ
    Ds, Dh = pec_wedge_DsDh(w, ang, k, L)
    D   = pol == :soft ? Ds : Dh
    A   = 1 / sqrt(rho)
    u_d = D * A * exp(-im * k * rho)

    return (total = u_GO + u_d, go = u_GO, diff = u_d)
end

# ────────────────────────────────────────────────────────────
# Figure 1: Transition function F(x)
# ────────────────────────────────────────────────────────────
function gen_fig1()
    xs = 10 .^ range(-4, 3, length=500)
    df = DataFrame(
        x       = xs,
        abs_F   = [abs(F_utd(x))  for x in xs],
        arg_F   = [angle(F_utd(x))/π for x in xs],
        Re_F    = [real(F_utd(x)) for x in xs],
        Im_F    = [imag(F_utd(x)) for x in xs],
    )
    CSV.write(joinpath(DATA_DIR, "transition_function.csv"), df)
    println("  transition_function: ($(nrow(df)) rows)")
end

# ────────────────────────────────────────────────────────────
# Figure 2: Half-plane UTD vs exact Sommerfeld
# ────────────────────────────────────────────────────────────
function gen_fig2()
    phip = π / 4
    k    = 1.0
    phis = range(0.05, 2π - 0.05, length=720)

    for (krho, label) in [(5.0, "krho5"), (50.0, "krho50")]
        rho = krho / k
        rows = NamedTuple[]
        for phi in phis
            u_exact_s = sommerfeld_exact(k, rho, phi, phip; pol=:soft)
            u_exact_h = sommerfeld_exact(k, rho, phi, phip; pol=:hard)
            u_utd_s   = utd_total_halfplane(k, rho, phi, phip; pol=:soft).total
            u_utd_h   = utd_total_halfplane(k, rho, phi, phip; pol=:hard).total
            push!(rows, (
                phi_deg        = rad2deg(phi),
                u_exact_s_abs  = abs(u_exact_s),
                u_utd_s_abs    = abs(u_utd_s),
                u_exact_h_abs  = abs(u_exact_h),
                u_utd_h_abs    = abs(u_utd_h),
            ))
        end
        df = DataFrame(rows)
        CSV.write(joinpath(DATA_DIR, "halfplane_$(label).csv"), df)
        println("  halfplane_$(label): ($(nrow(df)) rows)")
    end
end

# ────────────────────────────────────────────────────────────
# Figure 3: Shadow boundary detail (GO + diffracted → total)
# ────────────────────────────────────────────────────────────
function gen_fig3()
    phip = π / 4
    k    = 1.0
    krho = 10.0
    rho  = krho / k

    isb = π + phip
    phis = range(isb - 0.8, isb + 0.8, length=400)

    rows = NamedTuple[]
    for phi in phis
        for pol in [:soft, :hard]
            r = utd_total_halfplane(k, rho, phi, phip; pol=pol)
            u_exact = sommerfeld_exact(k, rho, phi, phip; pol=pol)
            push!(rows, (
                phi_deg       = rad2deg(phi),
                pol           = string(pol),
                abs_total     = abs(r.total),
                abs_go        = abs(r.go),
                abs_diff      = abs(r.diff),
                abs_exact     = abs(u_exact),
            ))
        end
    end
    df = DataFrame(rows)
    CSV.write(joinpath(DATA_DIR, "shadow_boundary.csv"), df)
    println("  shadow_boundary: ($(nrow(df)) rows)")
end

# ────────────────────────────────────────────────────────────
# Figure 4: GTD convergence — relative error vs kL
# ────────────────────────────────────────────────────────────
function gen_fig4()
    alphas = [2π, 3π/2, 5π/4]
    alpha_labels = ["2pi", "3pi2", "5pi4"]
    phip = π / 4

    rows = NamedTuple[]
    kLs  = 10 .^ range(0, 5, length=100)

    for (alpha, alabel) in zip(alphas, alpha_labels)
        w = Wedge(alpha)
        n = wedge_n(w)
        # Choose phi safely away from all boundaries
        phi = min(alpha - 0.3, π/2)
        ang = RayAngles(phi, phip)

        # GTD reference (F=1)
        phi_w  = UTDKernels.wrap_angle(phi, alpha)
        phip_w = UTDKernels.wrap_angle(phip, alpha)
        terms  = UTDKernels.kp_four_terms(phi_w, phip_w, n)
        C_gtd(k) = -exp(-im * π/4) / (2 * n * sqrt(2π * k))

        for kL in kLs
            k = 1.0
            L = kL / k
            Ds, Dh = pec_wedge_DsDh(w, ang, k, L)

            C = C_gtd(k)
            Ds_gtd = C * sum(UTDKernels.PEC_SIGMA_SOFT[j] * cot(terms.psi[j]) for j in 1:4)
            Dh_gtd = C * sum(UTDKernels.PEC_SIGMA_HARD[j] * cot(terms.psi[j]) for j in 1:4)

            err_s = abs(Ds - Ds_gtd) / abs(Ds_gtd)
            err_h = abs(Dh - Dh_gtd) / abs(Dh_gtd)

            push!(rows, (kL=kL, alpha=alabel, err_s=err_s, err_h=err_h))
        end
    end
    df = DataFrame(rows)
    CSV.write(joinpath(DATA_DIR, "gtd_convergence.csv"), df)
    println("  gtd_convergence: ($(nrow(df)) rows)")
end

# ────────────────────────────────────────────────────────────
# Figure 5: Multi-wedge diffraction patterns
# ────────────────────────────────────────────────────────────
function gen_fig5()
    phip = π / 4
    k    = 10.0
    L    = 1.0

    alphas = [5π/4, 3π/2, 2π]
    alpha_labels = ["5pi4", "3pi2", "2pi"]

    rows = NamedTuple[]
    for (alpha, alabel) in zip(alphas, alpha_labels)
        w = Wedge(alpha)
        phis = range(0.05, alpha - 0.05, length=500)
        for phi in phis
            ang = RayAngles(phi, phip)
            Ds, Dh = pec_wedge_DsDh(w, ang, k, L)
            push!(rows, (
                phi_deg  = rad2deg(phi),
                alpha    = alabel,
                Ds_abs   = abs(Ds),
                Dh_abs   = abs(Dh),
                Ds_dB    = 20*log10(max(abs(Ds), 1e-30)),
                Dh_dB    = 20*log10(max(abs(Dh), 1e-30)),
            ))
        end
    end
    df = DataFrame(rows)
    CSV.write(joinpath(DATA_DIR, "wedge_patterns.csv"), df)
    println("  wedge_patterns: ($(nrow(df)) rows)")
end

# ────────────────────────────────────────────────────────────
# Table I: Transition function values
# ────────────────────────────────────────────────────────────
function gen_table1()
    xs = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    rows = NamedTuple[]
    for x in xs
        Fval = F_utd(x)
        # erfc reference
        sqx = sqrt(Complex(x))
        z   = exp(+im * π/4) * sqx
        ref = sqrt(π * Complex(x)) * exp(+im * (π/4 + x)) * erfc(z)
        push!(rows, (
            x        = x,
            Re_F     = real(Fval),
            Im_F     = imag(Fval),
            abs_F    = abs(Fval),
            Re_ref   = real(ref),
            Im_ref   = imag(ref),
            abs_ref  = abs(ref),
            rel_err  = abs(Fval - ref) / max(abs(ref), 1e-30),
        ))
    end
    df = DataFrame(rows)
    CSV.write(joinpath(DATA_DIR, "table1_transition_values.csv"), df)
    println("  table1: transition function values  ($(nrow(df)) rows)")
end

# ────────────────────────────────────────────────────────────
# Table II: UTD vs exact Sommerfeld errors
# ────────────────────────────────────────────────────────────
function gen_table2()
    phip = π / 4
    k    = 1.0
    phi_tests = [π/6, π/3, π/2, 2π/3, π, 5π/4 - 0.2, 5π/4 + 0.2, 3π/2, 7π/4]
    krho_vals = [3.0, 5.0, 10.0, 20.0, 50.0, 100.0]

    rows = NamedTuple[]
    for krho in krho_vals
        rho = krho / k
        for phi in phi_tests
            u_exact_s = sommerfeld_exact(k, rho, phi, phip; pol=:soft)
            u_exact_h = sommerfeld_exact(k, rho, phi, phip; pol=:hard)
            u_utd_s   = utd_total_halfplane(k, rho, phi, phip; pol=:soft).total
            u_utd_h   = utd_total_halfplane(k, rho, phi, phip; pol=:hard).total

            denom_s = max(abs(u_exact_s), 1e-30)
            denom_h = max(abs(u_exact_h), 1e-30)
            push!(rows, (
                krho     = krho,
                phi_deg  = rad2deg(phi),
                err_s    = abs(u_utd_s - u_exact_s) / denom_s,
                err_h    = abs(u_utd_h - u_exact_h) / denom_h,
            ))
        end
    end
    df = DataFrame(rows)
    CSV.write(joinpath(DATA_DIR, "table2_utd_vs_exact.csv"), df)
    println("  table2: UTD vs Sommerfeld errors  ($(nrow(df)) rows)")
end

# ────────────────────────────────────────────────────────────
# Table III: Reciprocity verification
# ────────────────────────────────────────────────────────────
function gen_table3()
    rows = NamedTuple[]
    configs = [
        (Wedge(2π),   "2pi"),
        (Wedge(3π/2), "3pi2"),
        (Wedge(5π/4), "5pi4"),
        (Wedge(π),    "pi"),    # Flat plane: D=0 (validation of degenerate case)
    ]
    k = 10.0
    L = 1.0
    angle_pairs = [(π/4, π/3), (π/6, π/2), (0.5, 1.0), (π/5, 2π/5)]

    for (w, alabel) in configs
        for (phi, phip) in angle_pairs
            if phi >= w.alpha || phip >= w.alpha
                continue
            end
            Ds_fwd, Dh_fwd = pec_wedge_DsDh(w, RayAngles(phi, phip), k, L)
            Ds_rev, Dh_rev = pec_wedge_DsDh(w, RayAngles(phip, phi), k, L)
            push!(rows, (
                alpha    = alabel,
                phi_deg  = rad2deg(phi),
                phip_deg = rad2deg(phip),
                Ds_fwd_abs = abs(Ds_fwd),
                Ds_rev_abs = abs(Ds_rev),
                Ds_recip_err = abs(Ds_fwd - Ds_rev) / max(abs(Ds_fwd), 1e-30),
                Dh_fwd_abs = abs(Dh_fwd),
                Dh_rev_abs = abs(Dh_rev),
                Dh_recip_err = abs(Dh_fwd - Dh_rev) / max(abs(Dh_fwd), 1e-30),
            ))
        end
    end
    df = DataFrame(rows)
    CSV.write(joinpath(DATA_DIR, "table3_reciprocity.csv"), df)
    println("  table3: reciprocity verification  ($(nrow(df)) rows)")
end

# ────────────────────────────────────────────────────────────
# Figure 6: AD gradient validation — ForwardDiff vs finite diff
# ────────────────────────────────────────────────────────────
function gen_fig6()
    k    = 10.0
    L    = 1.0
    phip = π / 4

    # --- (a) Derivative wrt observation angle φ (half-plane) ---
    w = Wedge(2π)
    phis = range(0.1, 2π - 0.1, length=600)

    f_Ds(phi) = abs(pec_wedge_DsDh(w, RayAngles(phi, phip), k, L)[1])
    f_Dh(phi) = abs(pec_wedge_DsDh(w, RayAngles(phi, phip), k, L)[2])

    h_fd = 1e-7
    rows_phi = NamedTuple[]
    for phi in phis
        ad_s = ForwardDiff.derivative(f_Ds, phi)
        ad_h = ForwardDiff.derivative(f_Dh, phi)
        fd_s = (f_Ds(phi + h_fd) - f_Ds(phi - h_fd)) / (2h_fd)
        fd_h = (f_Dh(phi + h_fd) - f_Dh(phi - h_fd)) / (2h_fd)
        push!(rows_phi, (
            phi_deg = rad2deg(phi),
            ad_dDs  = ad_s,  fd_dDs = fd_s,
            ad_dDh  = ad_h,  fd_dDh = fd_h,
            err_s   = abs(ad_s - fd_s) / max(abs(ad_s), 1e-30),
            err_h   = abs(ad_h - fd_h) / max(abs(ad_h), 1e-30),
        ))
    end
    CSV.write(joinpath(DATA_DIR, "ad_vs_fd_phi.csv"), DataFrame(rows_phi))
    println("  ad_vs_fd_phi: ($(length(rows_phi)) rows)")

    # --- (b) Derivative wrt wavenumber k ---
    phi0 = π / 3
    ks = 10 .^ range(-1, 3, length=200)

    f_Ds_k(k_) = abs(pec_wedge_DsDh(w, RayAngles(phi0, phip), k_, L)[1])

    rows_k = NamedTuple[]
    for kval in ks
        h_k = kval * 1e-7
        ad = ForwardDiff.derivative(f_Ds_k, kval)
        fd = (f_Ds_k(kval + h_k) - f_Ds_k(kval - h_k)) / (2h_k)
        push!(rows_k, (
            k = kval,  ad_dDs = ad,  fd_dDs = fd,
            err = abs(ad - fd) / max(abs(ad), 1e-30),
        ))
    end
    CSV.write(joinpath(DATA_DIR, "ad_vs_fd_k.csv"), DataFrame(rows_k))
    println("  ad_vs_fd_k: ($(length(rows_k)) rows)")

    # --- (c) Derivative wrt effective distance L ---
    Ls = 10 .^ range(-2, 3, length=200)

    f_Ds_L(L_) = abs(pec_wedge_DsDh(w, RayAngles(phi0, phip), k, L_)[1])

    rows_L = NamedTuple[]
    for Lval in Ls
        h_L = Lval * 1e-7
        ad = ForwardDiff.derivative(f_Ds_L, Lval)
        fd = (f_Ds_L(Lval + h_L) - f_Ds_L(Lval - h_L)) / (2h_L)
        push!(rows_L, (
            L = Lval,  ad_dDs = ad,  fd_dDs = fd,
            err = abs(ad - fd) / max(abs(ad), 1e-30),
        ))
    end
    CSV.write(joinpath(DATA_DIR, "ad_vs_fd_L.csv"), DataFrame(rows_L))
    println("  ad_vs_fd_L: ($(length(rows_L)) rows)")
end

# ────────────────────────────────────────────────────────────
# Figure 7: Branch-safety — |D| finiteness through shadow boundary
# ────────────────────────────────────────────────────────────
function gen_fig7()
    k    = 10.0
    L    = 1.0
    phip = π / 4

    # Sweep φ finely through the ISB at φ = π + φ' = 225°
    isb = π + phip
    # Include the exact boundary and extreme close-ups
    phis_coarse = collect(range(isb - 0.3, isb + 0.3, length=1000))
    phis_fine = [isb + ε for ε in [-1e-6, -1e-8, -1e-10, -1e-12, -1e-14, -1e-15, 0.0, 1e-15, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6]]
    phis = sort(unique(vcat(phis_coarse, phis_fine)))

    w = Wedge(2π)
    n = wedge_n(w)

    rows = NamedTuple[]
    for phi in phis
        ang = RayAngles(phi, phip)
        Ds, Dh = pec_wedge_DsDh(w, ang, k, L)

        # Also compute naïve (unregularized) cot·F for comparison
        phi_w  = UTDKernels.wrap_angle(phi, 2π)
        phip_w = UTDKernels.wrap_angle(phip, 2π)
        terms  = UTDKernels.kp_four_terms(phi_w, phip_w, n)
        C = UTDKernels.pec_wedge_prefactor(k, n)
        naive_s = zero(ComplexF64)
        naive_h = zero(ComplexF64)
        for j in 1:4
            psi_j = terms.psi[j]
            a_j   = terms.aj[j]
            X_j   = k * L * a_j
            cotF  = cot(psi_j) * F_utd(X_j)  # naïve: may be Inf*0
            naive_s += UTDKernels.PEC_SIGMA_SOFT[j] * cotF
            naive_h += UTDKernels.PEC_SIGMA_HARD[j] * cotF
        end
        naive_Ds = C * naive_s
        naive_Dh = C * naive_h

        push!(rows, (
            phi_deg     = rad2deg(phi),
            Ds_reg_abs  = abs(Ds),
            Dh_reg_abs  = abs(Dh),
            Ds_naive_abs = abs(naive_Ds),
            Dh_naive_abs = abs(naive_Dh),
            Ds_reg_finite  = isfinite(abs(Ds)),
            Ds_naive_finite = isfinite(abs(naive_Ds)),
        ))
    end
    df = DataFrame(rows)
    CSV.write(joinpath(DATA_DIR, "branch_safety.csv"), df)
    n_inf = count(.!df.Ds_naive_finite)
    println("  branch_safety: ($(nrow(df)) rows, $n_inf naïve NaN/Inf)")
end

# ────────────────────────────────────────────────────────────
# Table IV: AD accuracy summary
# ────────────────────────────────────────────────────────────
function gen_table4()
    k = 10.0; L = 1.0; phip = π/4
    w = Wedge(2π)

    rows = NamedTuple[]

    # Derivatives wrt φ at several angles
    f_phi(phi) = abs(pec_wedge_DsDh(w, RayAngles(phi, phip), k, L)[1])
    for phi_deg in [30.0, 60.0, 90.0, 120.0, 150.0, 224.0, 226.0, 270.0, 315.0]
        phi = deg2rad(phi_deg)
        ad = ForwardDiff.derivative(f_phi, phi)
        h = 1e-7
        fd = (f_phi(phi + h) - f_phi(phi - h)) / (2h)
        push!(rows, (
            variable = "phi", value_deg = phi_deg,
            ad = ad, fd = fd,
            abs_err = abs(ad - fd),
            rel_err = abs(ad - fd) / max(abs(ad), 1e-30),
        ))
    end

    # Derivatives wrt k
    f_k(k_) = abs(pec_wedge_DsDh(w, RayAngles(π/3, phip), k_, L)[1])
    for kval in [1.0, 10.0, 100.0]
        ad = ForwardDiff.derivative(f_k, kval)
        h = kval * 1e-7
        fd = (f_k(kval + h) - f_k(kval - h)) / (2h)
        push!(rows, (
            variable = "k", value_deg = kval,
            ad = ad, fd = fd,
            abs_err = abs(ad - fd),
            rel_err = abs(ad - fd) / max(abs(ad), 1e-30),
        ))
    end

    # Derivatives wrt L
    f_L(L_) = abs(pec_wedge_DsDh(w, RayAngles(π/3, phip), k, L_)[1])
    for Lval in [0.1, 1.0, 10.0]
        ad = ForwardDiff.derivative(f_L, Lval)
        h = Lval * 1e-7
        fd = (f_L(Lval + h) - f_L(Lval - h)) / (2h)
        push!(rows, (
            variable = "L", value_deg = Lval,
            ad = ad, fd = fd,
            abs_err = abs(ad - fd),
            rel_err = abs(ad - fd) / max(abs(ad), 1e-30),
        ))
    end

    df = DataFrame(rows)
    CSV.write(joinpath(DATA_DIR, "table4_ad_accuracy.csv"), df)
    println("  table4: AD accuracy summary  ($(nrow(df)) rows)")
end

# ────────────────────────────────────────────────────────────
# Error analysis: max relative error vs kρ, and angular profile near ISB
# ────────────────────────────────────────────────────────────
function gen_error_analysis()
    phip = π / 4
    k    = 1.0

    # (a) Max relative error vs kρ
    krho_vals = 10 .^ range(log10(0.5), 3, length=200)
    rows_krho = NamedTuple[]
    for krho in krho_vals
        rho = krho / k
        phis = range(0.05, 2π - 0.05, length=720)
        max_abs_s = 0.0; max_abs_h = 0.0
        max_rel_s = 0.0; max_rel_h = 0.0
        for phi in phis
            u_exact_s = sommerfeld_exact(k, rho, phi, phip; pol=:soft)
            u_exact_h = sommerfeld_exact(k, rho, phi, phip; pol=:hard)
            u_utd_s   = utd_total_halfplane(k, rho, phi, phip; pol=:soft).total
            u_utd_h   = utd_total_halfplane(k, rho, phi, phip; pol=:hard).total
            abs_s = abs(u_utd_s - u_exact_s)
            abs_h = abs(u_utd_h - u_exact_h)
            max_abs_s = max(max_abs_s, abs_s)
            max_abs_h = max(max_abs_h, abs_h)
            if abs(u_exact_s) > 0.01
                max_rel_s = max(max_rel_s, abs_s / abs(u_exact_s))
            end
            if abs(u_exact_h) > 0.01
                max_rel_h = max(max_rel_h, abs_h / abs(u_exact_h))
            end
        end
        push!(rows_krho, (krho=krho, max_abs_s=max_abs_s, max_abs_h=max_abs_h,
                          max_rel_s=max_rel_s, max_rel_h=max_rel_h))
    end
    CSV.write(joinpath(DATA_DIR, "error_vs_krho.csv"), DataFrame(rows_krho))
    println("  error_vs_krho: ($(length(rows_krho)) rows)")

    # (b) Absolute error angular profile near ISB
    isb = π + phip
    rows_isb = NamedTuple[]
    for krho in [3.0, 10.0, 50.0]
        rho = krho / k
        phis = range(isb - 0.5, isb + 0.5, length=200)
        for phi in phis
            u_exact_s = sommerfeld_exact(k, rho, phi, phip; pol=:soft)
            u_utd_s   = utd_total_halfplane(k, rho, phi, phip; pol=:soft).total
            push!(rows_isb, (krho=krho, phi_deg=rad2deg(phi),
                             abs_err=abs(u_utd_s - u_exact_s)))
        end
    end
    CSV.write(joinpath(DATA_DIR, "error_angular_isb.csv"), DataFrame(rows_isb))
    println("  error_angular_isb: ($(length(rows_isb)) rows)")
end

# ────────────────────────────────────────────────────────────
# Run all
# ────────────────────────────────────────────────────────────
println("Generating paper data …")
gen_fig1()
gen_fig2()
gen_fig3()
gen_fig4()
gen_fig5()
gen_fig6()
gen_fig7()
gen_error_analysis()
gen_table1()
gen_table2()
gen_table3()
gen_table4()
println("Done.  Output in $(DATA_DIR)/")
