#!/usr/bin/env julia
"""
Generate figures for the UTDKernels.jl SoftwareX paper.

Run from the package root:
  julia --project=paper paper/plot_paper.jl

Input:  data/*.csv  (from generate_paper_data.jl)
Output: figs/*.pdf

Backend: PlotlySupply.jl.
Wong colorblind palette + distinct dash patterns.
"""

using Pkg
Pkg.activate(@__DIR__)

using CSV, DataFrames, PlotlySupply, UTDKernels

const DATA = joinpath(@__DIR__, "data")
const FIGS = joinpath(@__DIR__, "figs")
const WDC_REF = joinpath(@__DIR__, "..", "validation", "data", "wdc_reference.csv")
mkpath(FIGS)

# Wong colorblind palette + dash set (per plotting-guide.md)
const C_BLUE   = "#0072B2"
const C_ORANGE = "#D55E00"
const C_GREEN  = "#009E73"
const C_PINK   = "#CC79A7"
const C_BLACK  = "#000000"

# 2-panel figure rendered at 0.85*\linewidth in single-column elsarticle
# (~5.5in wide). Use double-column-ish width with subfigure font doubling.
const FIG_W = 1008
const FIG_H = 380
const FS    = 12  # figure* (double-col) — no font doubling per plotting-guide

posclip(x; lo=1e-30) = max(x, lo)

function smart_err(a::Complex, b::Complex; abs_floor::Float64=1e-4)
    abs(a - b) / max(abs(b), abs(a), abs_floor)
end

function boundary_distance_deg(phi_deg, phip_deg, n)
    isb = 180.0 + phip_deg
    rsb = 180.0 - phip_deg
    max_angle = n * 180.0
    min(abs(phi_deg - isb), abs(phi_deg - rsb), phi_deg, max_angle - phi_deg)
end

function compute_utd_di(n, R, phi_deg, phip_deg)
    wedge = Wedge(n * π)
    ang = RayAngles(deg2rad(phi_deg), deg2rad(phip_deg))
    return pec_wedge_DsDh(wedge, ang, 2π, R, R, R; Rs=0, Rh=0)[1]
end

# ═══════════════════════════════════════════════════════════════
# Figure: Half-plane total field — UTD vs exact Sommerfeld (k\rho=50)
# ═══════════════════════════════════════════════════════════════
function fig_halfplane()
    df = CSV.read(joinpath(DATA, "halfplane_krho50.csv"), DataFrame)

    sf = subplots(1, 2; sync=false, title="", per_subplot_legends=true)

    subplot!(sf, 1, 1)
    plot_scatter!(sf, df.phi_deg, df.u_exact_s_abs;
        mode="lines", color=C_BLUE, dash="solid", linewidth=2,
        legend="Exact (Sommerfeld)",
        xlabel=raw"$\phi\;[\mathrm{deg}]$", ylabel=raw"$|u_{\mathrm{s}}|$",
        fontsize=FS)
    plot_scatter!(sf, df.phi_deg, df.u_utd_s_abs;
        mode="lines", color=C_ORANGE, dash="dash", linewidth=2,
        legend="UTD", fontsize=FS)

    subplot!(sf, 1, 2)
    plot_scatter!(sf, df.phi_deg, df.u_exact_h_abs;
        mode="lines", color=C_BLUE, dash="solid", linewidth=2,
        legend="Exact (Sommerfeld)",
        xlabel=raw"$\phi\;[\mathrm{deg}]$", ylabel=raw"$|u_{\mathrm{h}}|$",
        fontsize=FS)
    plot_scatter!(sf, df.phi_deg, df.u_utd_h_abs;
        mode="lines", color=C_ORANGE, dash="dash", linewidth=2,
        legend="UTD", fontsize=FS)

    subplot_legends!(sf; position=:topright)
    savefig(sf.fig, joinpath(FIGS, "halfplane_krho50.pdf"); width=FIG_W, height=FIG_H)
    println("  Saved halfplane_krho50.pdf")
end

# ═══════════════════════════════════════════════════════════════
# Figure: Error analysis — max rel err vs k\rho ; abs err near ISB
# ═══════════════════════════════════════════════════════════════
function fig_error_analysis()
    df_krho = CSV.read(joinpath(DATA, "error_vs_krho.csv"), DataFrame)
    df_isb  = CSV.read(joinpath(DATA, "error_angular_isb.csv"), DataFrame)

    sf = subplots(1, 2; sync=false, title="", per_subplot_legends=true)

    subplot!(sf, 1, 1)
    plot_scatter!(sf, df_krho.krho, posclip.(df_krho.max_rel_s);
        mode="lines", color=C_BLUE, dash="solid", linewidth=2,
        legend=raw"Soft $|u_{\mathrm{s}}|$",
        xlabel=raw"$k\rho$", ylabel="Max relative error",
        xscale="log", yscale="log", fontsize=FS)
    plot_scatter!(sf, df_krho.krho, posclip.(df_krho.max_rel_h);
        mode="lines", color=C_ORANGE, dash="dash", linewidth=2,
        legend=raw"Hard $|u_{\mathrm{h}}|$",
        xscale="log", yscale="log", fontsize=FS)
    plot_scatter!(sf, df_krho.krho, fill(2.2e-16, length(df_krho.krho));
        mode="lines", color=C_BLACK, dash="dot", linewidth=1,
        legend=raw"$\epsilon_{\mathrm{mach}}$",
        xscale="log", yscale="log", fontsize=FS)

    subplot!(sf, 1, 2)
    krho_styles = [(3.0, C_BLUE, "solid"), (10.0, C_ORANGE, "dash"), (50.0, C_GREEN, "dashdot")]
    first = true
    for (krho_val, clr, ds) in krho_styles
        sub = df_isb[df_isb.krho .== krho_val, :]
        plot_scatter!(sf, sub.phi_deg, posclip.(sub.abs_err);
            mode="lines", color=clr, dash=ds, linewidth=2,
            legend="kρ=$(Int(krho_val))",
            xlabel=first ? raw"$\phi\;[\mathrm{deg}]$" : "",
            ylabel=first ? "Absolute error" : "",
            yscale="log", yrange=[-18.0, -12.0], fontsize=FS)
        first = false
    end

    subplot_legends!(sf; position=:right)
    # Override right subplot legend to topright
    let leg2 = sf.plot.layout.fields[:legend2]
        leg2[:x] = 0.99
        leg2[:y] = 0.97
        leg2[:xanchor] = "right"
        leg2[:yanchor] = "top"
    end
    savefig(sf.fig, joinpath(FIGS, "error_analysis.pdf"); width=FIG_W, height=FIG_H)
    println("  Saved error_analysis.pdf")
end

# ═══════════════════════════════════════════════════════════════
# Figure: Shadow boundary — GO + diffracted -> total
# ═══════════════════════════════════════════════════════════════
function fig_shadow_boundary()
    df  = CSV.read(joinpath(DATA, "shadow_boundary.csv"), DataFrame)
    dfs = df[df.pol .== "soft", :]
    dfh = df[df.pol .== "hard", :]

    sf = subplots(1, 2; sync=false, title="", per_subplot_legends=true)

    for (col, dd, ylab) in ((1, dfs, raw"$|u_{\mathrm{s}}|$"),
                             (2, dfh, raw"$|u_{\mathrm{h}}|$"))
        subplot!(sf, 1, col)
        plot_scatter!(sf, dd.phi_deg, dd.abs_exact;
            mode="lines", color=C_BLUE, dash="solid", linewidth=2,
            legend="Exact (Sommerfeld)",
            xlabel=raw"$\phi\;[\mathrm{deg}]$", ylabel=ylab, fontsize=FS)
        plot_scatter!(sf, dd.phi_deg, dd.abs_total;
            mode="lines", color=C_ORANGE, dash="dash", linewidth=2,
            legend="Total (UTD)", fontsize=FS)
        plot_scatter!(sf, dd.phi_deg, dd.abs_go;
            mode="lines", color=C_GREEN, dash="dot", linewidth=2,
            legend="GO", fontsize=FS)
        plot_scatter!(sf, dd.phi_deg, dd.abs_diff;
            mode="lines", color=C_PINK, dash="dashdot", linewidth=2,
            legend="Diffracted", fontsize=FS)
    end

    subplot_legends!(sf; position=:topright)
    savefig(sf.fig, joinpath(FIGS, "shadow_boundary.pdf"); width=FIG_W, height=FIG_H)
    println("  Saved shadow_boundary.pdf")
end

# ═══════════════════════════════════════════════════════════════
# Figure: GTD convergence — relative error vs kL
# ═══════════════════════════════════════════════════════════════
function fig_gtd_convergence()
    df = CSV.read(joinpath(DATA, "gtd_convergence.csv"), DataFrame)

    alpha_styles = [
        ("2pi",  "Half-plane (α=2π)", C_BLUE,   "solid"),
        ("3pi2", "α=3π/2",            C_ORANGE, "dash"),
        ("5pi4", "α=5π/4",            C_GREEN,  "dashdot"),
    ]

    sf = subplots(1, 2; sync=false, title="", per_subplot_legends=true)

    for (col, errcol, ylab) in ((1, :err_s, raw"$|D_{\mathrm{s}}^{\mathrm{UTD}}-D_{\mathrm{s}}^{\mathrm{GTD}}|/|D_{\mathrm{s}}^{\mathrm{GTD}}|$"),
                                 (2, :err_h, raw"$|D_{\mathrm{h}}^{\mathrm{UTD}}-D_{\mathrm{h}}^{\mathrm{GTD}}|/|D_{\mathrm{h}}^{\mathrm{GTD}}|$"))
        subplot!(sf, 1, col)
        first = true
        for (akey, lbl, clr, ds) in alpha_styles
            sub = df[df.alpha .== akey, :]
            plot_scatter!(sf, sub.kL, posclip.(sub[!, errcol]);
                mode="lines", color=clr, dash=ds, linewidth=2,
                legend=lbl,
                xlabel=first ? raw"$kL$" : "",
                ylabel=first ? ylab : "",
                xscale="log", yscale="log", fontsize=FS)
            first = false
        end
        kLref = [1.0, 1e5]
        plot_scatter!(sf, kLref, 1.0 ./ kLref;
            mode="lines", color=C_BLACK, dash="dot", linewidth=1,
            legend="O(1/kL)",
            xscale="log", yscale="log", fontsize=FS)
    end

    subplot_legends!(sf; position=:bottomleft)
    savefig(sf.fig, joinpath(FIGS, "gtd_convergence.pdf"); width=FIG_W, height=FIG_H)
    println("  Saved gtd_convergence.pdf")
end

# ═══════════════════════════════════════════════════════════════
# Figure: Branch-safety — regularised vs naïve near boundary
# ═══════════════════════════════════════════════════════════════
function fig_branch_safety()
    df = CSV.read(joinpath(DATA, "branch_safety.csv"), DataFrame; comment="#")

    ds_naive = copy(df.Ds_naive_abs)
    dh_naive = copy(df.Dh_naive_abs)
    ds_naive[.!isfinite.(ds_naive)] .= NaN
    dh_naive[.!isfinite.(dh_naive)] .= NaN

    sf = subplots(1, 2; sync=false, title="", per_subplot_legends=true)

    function _isb_lims(v)
        finite = v[isfinite.(v)]
        if isempty(finite)
            return (0.0, 1.0)
        end
        return (minimum(finite), maximum(finite))
    end

    subplot!(sf, 1, 1)
    plot_scatter!(sf, df.phi_deg, df.Ds_reg_abs;
        mode="lines", color=C_BLUE, dash="solid", linewidth=2,
        legend="Regularised",
        xlabel=raw"$\phi\;[\mathrm{deg}]$", ylabel=raw"$|D_{\mathrm{s}}|$",
        fontsize=FS)
    plot_scatter!(sf, df.phi_deg, ds_naive;
        mode="lines", color=C_ORANGE, dash="dash", linewidth=2,
        legend="Naive cot·F", fontsize=FS)
    let (ymin, ymax) = _isb_lims(df.Ds_reg_abs)
        plot_scatter!(sf, [225.0, 225.0], [ymin, ymax];
            mode="lines", color=C_BLACK, dash="dot", linewidth=1,
            legend="ISB", fontsize=FS)
    end

    subplot!(sf, 1, 2)
    plot_scatter!(sf, df.phi_deg, df.Dh_reg_abs;
        mode="lines", color=C_BLUE, dash="solid", linewidth=2,
        legend="Regularised",
        xlabel=raw"$\phi\;[\mathrm{deg}]$", ylabel=raw"$|D_{\mathrm{h}}|$",
        fontsize=FS)
    plot_scatter!(sf, df.phi_deg, dh_naive;
        mode="lines", color=C_ORANGE, dash="dash", linewidth=2,
        legend="Naive cot·F", fontsize=FS)
    let (ymin, ymax) = _isb_lims(df.Dh_reg_abs)
        plot_scatter!(sf, [225.0, 225.0], [ymin, ymax];
            mode="lines", color=C_BLACK, dash="dot", linewidth=1,
            legend="ISB", fontsize=FS)
    end

    subplot_legends!(sf; position=:topleft)
    savefig(sf.fig, joinpath(FIGS, "branch_safety.pdf"); width=FIG_W, height=FIG_H)
    println("  Saved branch_safety.pdf")
end

# ═══════════════════════════════════════════════════════════════
# Figure: AD gradient validation — ForwardDiff vs finite diff
# ═══════════════════════════════════════════════════════════════
function fig_ad_validation()
    df = CSV.read(joinpath(DATA, "ad_vs_fd_phi.csv"), DataFrame)

    sf = subplots(1, 2; sync=false, title="", per_subplot_legends=true)

    subplot!(sf, 1, 1)
    plot_scatter!(sf, df.phi_deg, df.ad_dDs;
        mode="lines", color=C_BLUE, dash="solid", linewidth=2,
        legend="AD (ForwardDiff)",
        xlabel=raw"$\phi\;[\mathrm{deg}]$",
        ylabel=raw"$\partial|D_{\mathrm{s}}|/\partial\phi$",
        fontsize=FS)
    plot_scatter!(sf, df.phi_deg, df.fd_dDs;
        mode="lines", color=C_ORANGE, dash="dash", linewidth=2,
        legend="Finite difference", fontsize=FS)

    subplot!(sf, 1, 2)
    plot_scatter!(sf, df.phi_deg, posclip.(df.err_s);
        mode="lines", color=C_BLUE, dash="solid", linewidth=2,
        legend=raw"$|D_{\mathrm{s}}|$",
        xlabel=raw"$\phi\;[\mathrm{deg}]$",
        ylabel="Relative error (AD vs FD)",
        yscale="log", fontsize=FS)
    plot_scatter!(sf, df.phi_deg, posclip.(df.err_h);
        mode="lines", color=C_ORANGE, dash="dash", linewidth=2,
        legend=raw"$|D_{\mathrm{h}}|$",
        yscale="log", fontsize=FS)

    subplot_legends!(sf; position=:bottomleft)
    savefig(sf.fig, joinpath(FIGS, "ad_validation.pdf"); width=FIG_W, height=FIG_H)
    println("  Saved ad_validation.pdf")
end

# ═══════════════════════════════════════════════════════════════
# Figure: Balanis WDC.m cross-validation
# ═══════════════════════════════════════════════════════════════
function fig_balanis_validation()
    df = CSV.read(WDC_REF, DataFrame)
    bal_abs = Float64[]
    utd_abs = Float64[]
    errs = Float64[]

    for row in eachrow(df)
        d_bnd = boundary_distance_deg(row.phi_deg, row.phip_deg, row.n)
        if row.n == 1.0 || d_bnd < 2.0
            continue
        end
        di_ref = complex(row.Re_Di, row.Im_Di)
        abs(di_ref) < 1e-14 && continue
        di_utd = compute_utd_di(row.n, row.R, row.phi_deg, row.phip_deg)
        push!(bal_abs, abs(di_ref))
        push!(utd_abs, abs(di_utd))
        push!(errs, smart_err(di_utd, di_ref))
    end

    sample_count = min(length(errs), 5000)
    sample_idx = unique(round.(Int, range(1, length(errs); length=sample_count)))
    sorted_errs = sort(errs)
    pct = 100 .* collect(1:length(sorted_errs)) ./ length(sorted_errs)

    sf = subplots(1, 2; sync=false, title="", per_subplot_legends=true)

    subplot!(sf, 1, 1)
    plot_scatter!(sf, bal_abs[sample_idx], utd_abs[sample_idx];
        mode="markers", color=C_BLUE, marker_size=3,
        legend="Away-boundary cases",
        xlabel=raw"$|D_i|$ from Balanis WDC.m",
        ylabel=raw"$|D_i|$ from UTDKernels.jl",
        xscale="log", yscale="log", fontsize=FS)
    lims = [minimum(vcat(bal_abs, utd_abs)), maximum(vcat(bal_abs, utd_abs))]
    plot_scatter!(sf, lims, lims;
        mode="lines", color=C_BLACK, dash="dash", linewidth=1,
        legend="1:1", xscale="log", yscale="log", fontsize=FS)

    subplot!(sf, 1, 2)
    plot_scatter!(sf, pct, sorted_errs;
        mode="lines", color=C_ORANGE, dash="solid", linewidth=2,
        legend="Empirical CDF",
        xlabel="Percentile [%]",
        ylabel=raw"Relative error in $D_i$",
        yscale="log", fontsize=FS)
    plot_scatter!(sf, [0.0, 100.0], [0.025, 0.025];
        mode="lines", color=C_BLACK, dash="dash", linewidth=1,
        legend="2.5% tolerance", yscale="log", fontsize=FS)

    subplot_legends!(sf; position=:bottomright)
    savefig(sf.fig, joinpath(FIGS, "balanis_validation.pdf"); width=FIG_W, height=FIG_H)
    println("  Saved balanis_validation.pdf")
end

const ALL_FIGS = Dict(
    "halfplane"      => fig_halfplane,
    "error"          => fig_error_analysis,
    "shadow"         => fig_shadow_boundary,
    "gtd"            => fig_gtd_convergence,
    "branch"         => fig_branch_safety,
    "ad"             => fig_ad_validation,
    "balanis"        => fig_balanis_validation,
)

println("Generating figures …")
if length(ARGS) == 0 || ARGS[1] == "all"
    for k in ("halfplane", "error", "shadow", "gtd", "branch", "ad", "balanis")
        ALL_FIGS[k]()
    end
else
    ALL_FIGS[ARGS[1]]()
end
println("Done.  Output in $(FIGS)/")
flush(stdout); flush(stderr)
exit(0)
