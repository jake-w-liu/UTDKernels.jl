#!/usr/bin/env julia
"""
Generate figures for UTDKernels.jl validation.

Run from the package root:
  julia --project=validation validation/plot_figures.jl

Input: data/*.csv  (from generate_data.jl)
Output: figs/*.pdf

Style: GR backend, linewidth=2, framestyle=:box
"""

using CSV, DataFrames, Plots
gr()

const DATA = joinpath(@__DIR__, "data")
const FIGS = joinpath(@__DIR__, "figs")
mkpath(FIGS)

# ── Global defaults (plotting_style.md) ──────────────────────
default(
    linewidth    = 2,
    framestyle   = :box,
    grid         = true,
    minorgrid    = false,
    legendfontsize  = 10,
    guidefontsize   = 12,
    tickfontsize    = 10,
    titlefontsize   = 10,
    legend_background_color = RGBA(1,1,1,0.85),
    markersize   = 5,
    dpi          = 300,
)

posclip(x; lo=1e-30) = max(x, lo)

# ═══════════════════════════════════════════════════════════════
# Figure 1: Transition function F(x) — magnitude and phase
# ═══════════════════════════════════════════════════════════════
function fig_transition()
    df = CSV.read(joinpath(DATA, "fig1_transition_function.csv"), DataFrame)

    p1 = plot(df.x, df.abs_F;
        label  = raw"$|F(x)|$",
        xlabel = raw"$x$",
        ylabel = raw"$|F(x)|$",
        xscale = :log10,
        color  = :blue,
        legend = :bottomright,
        size   = (600, 400),
    )
    hline!(p1, [1.0]; color=:black, linestyle=:dash, linewidth=1, label=raw"GTD limit $F=1$")

    p2 = plot(df.x, df.arg_F;
        label  = raw"$\arg F(x)\;/\;\pi$",
        xlabel = raw"$x$",
        ylabel = raw"$\arg F(x) \;/\;\pi$",
        xscale = :log10,
        color  = :red,
        legend = :topright,
        size   = (600, 400),
    )
    hline!(p2, [0.0]; color=:black, linestyle=:dash, linewidth=1, label="")

    p = plot(p1, p2; layout=(1,2), size=(1100, 400), margin=5Plots.mm)
    savefig(p, joinpath(FIGS, "fig1_transition_function.pdf"))
    println("  Saved fig1_transition_function.pdf")
end

# ═══════════════════════════════════════════════════════════════
# Figure 2: Half-plane total field — UTD vs exact Sommerfeld
# ═══════════════════════════════════════════════════════════════
function fig_halfplane()
    for (krho, label) in [("krho5", raw"$k\rho=5$"), ("krho50", raw"$k\rho=50$")]
        df = CSV.read(joinpath(DATA, "fig2_halfplane_$(krho).csv"), DataFrame)

        p1 = plot(df.phi_deg, df.u_exact_s_abs;
            label     = "Exact (Sommerfeld)",
            xlabel    = raw"$\phi$ (deg)",
            ylabel    = raw"$|u_{\mathrm{s}}|$",
            title     = "Soft, $(label)",
            color     = :blue,
            linestyle = :solid,
            legend    = :topright,
            size      = (600, 450),
        )
        plot!(p1, df.phi_deg, df.u_utd_s_abs;
            label     = "UTD",
            color     = :red,
            linestyle = :dash,
        )

        p2 = plot(df.phi_deg, df.u_exact_h_abs;
            label     = "Exact (Sommerfeld)",
            xlabel    = raw"$\phi$ (deg)",
            ylabel    = raw"$|u_{\mathrm{h}}|$",
            title     = "Hard, $(label)",
            color     = :blue,
            linestyle = :solid,
            legend    = :topright,
            size      = (600, 450),
        )
        plot!(p2, df.phi_deg, df.u_utd_h_abs;
            label     = "UTD",
            color     = :red,
            linestyle = :dash,
        )

        p = plot(p1, p2; layout=(1,2), size=(1100, 450), margin=5Plots.mm)
        savefig(p, joinpath(FIGS, "fig2_halfplane_$(krho).pdf"))
        println("  Saved fig2_halfplane_$(krho).pdf")
    end
end

# ═══════════════════════════════════════════════════════════════
# Figure 3: Shadow boundary detail — GO + diffracted → total
# ═══════════════════════════════════════════════════════════════
function fig_shadow_boundary()
    df = CSV.read(joinpath(DATA, "fig3_shadow_boundary.csv"), DataFrame)
    dfs = df[df.pol .== "soft", :]
    dfh = df[df.pol .== "hard", :]

    # Plot exact (solid) first, then UTD total (dash) on top so dashes are visible
    p1 = plot(dfs.phi_deg, dfs.abs_exact;
        label     = "Exact (Sommerfeld)",
        xlabel    = raw"$\phi$ (deg)",
        ylabel    = "Magnitude",
        title     = raw"Soft pol., ISB at $\phi=225°$",
        color     = :blue,
        linestyle = :solid,
        legend    = :topright,
        size      = (600, 450),
    )
    plot!(p1, dfs.phi_deg, dfs.abs_total;
        label = "Total (UTD)", color = :red, linestyle = :dash)
    plot!(p1, dfs.phi_deg, dfs.abs_go;
        label = "GO", color = :green, linestyle = :dot)
    plot!(p1, dfs.phi_deg, dfs.abs_diff;
        label = "Diffracted", color = :orange, linestyle = :dashdot)

    p2 = plot(dfh.phi_deg, dfh.abs_exact;
        label     = "Exact (Sommerfeld)",
        xlabel    = raw"$\phi$ (deg)",
        ylabel    = "Magnitude",
        title     = raw"Hard pol., ISB at $\phi=225°$",
        color     = :blue,
        linestyle = :solid,
        legend    = :topright,
        size      = (600, 450),
    )
    plot!(p2, dfh.phi_deg, dfh.abs_total;
        label = "Total (UTD)", color = :red, linestyle = :dash)
    plot!(p2, dfh.phi_deg, dfh.abs_go;
        label = "GO", color = :green, linestyle = :dot)
    plot!(p2, dfh.phi_deg, dfh.abs_diff;
        label = "Diffracted", color = :orange, linestyle = :dashdot)

    p = plot(p1, p2; layout=(1,2), size=(1100, 450), margin=5Plots.mm)
    savefig(p, joinpath(FIGS, "fig3_shadow_boundary.pdf"))
    println("  Saved fig3_shadow_boundary.pdf")
end

# ═══════════════════════════════════════════════════════════════
# Figure 4: GTD convergence — relative error vs kL
# ═══════════════════════════════════════════════════════════════
function fig_gtd_convergence()
    df = CSV.read(joinpath(DATA, "fig4_gtd_convergence.csv"), DataFrame)

    alpha_map = Dict(
        "2pi"  => (raw"Half-plane ($\alpha=2\pi$)", :blue),
        "3pi2" => (raw"$\alpha=3\pi/2$", :red),
        "5pi4" => (raw"$\alpha=5\pi/4$", :green),
    )

    p1 = plot(;
        xlabel = raw"$kL$",
        ylabel = raw"$|D_{\mathrm{s}}^{\mathrm{UTD}} - D_{\mathrm{s}}^{\mathrm{GTD}}|/|D_{\mathrm{s}}^{\mathrm{GTD}}|$",
        title  = "Soft polarization",
        xscale = :log10, yscale = :log10,
        legend = :bottomleft,
        size   = (600, 450),
    )
    p2 = plot(;
        xlabel = raw"$kL$",
        ylabel = raw"$|D_{\mathrm{h}}^{\mathrm{UTD}} - D_{\mathrm{h}}^{\mathrm{GTD}}|/|D_{\mathrm{h}}^{\mathrm{GTD}}|$",
        title  = "Hard polarization",
        xscale = :log10, yscale = :log10,
        legend = :bottomleft,
        size   = (600, 450),
    )

    for akey in ["2pi", "3pi2", "5pi4"]
        sub = df[df.alpha .== akey, :]
        lbl, clr = alpha_map[akey]
        plot!(p1, sub.kL, posclip.(sub.err_s); label=lbl, color=clr)
        plot!(p2, sub.kL, posclip.(sub.err_h); label=lbl, color=clr)
    end

    # O(1/kL) reference slope
    kLref = [1.0, 1e5]
    plot!(p1, kLref, 1.0 ./ kLref; label=raw"$O(1/kL)$", color=:black, linestyle=:dash, linewidth=1)
    plot!(p2, kLref, 1.0 ./ kLref; label=raw"$O(1/kL)$", color=:black, linestyle=:dash, linewidth=1)

    p = plot(p1, p2; layout=(1,2), size=(1100, 450), margin=5Plots.mm)
    savefig(p, joinpath(FIGS, "fig4_gtd_convergence.pdf"))
    println("  Saved fig4_gtd_convergence.pdf")
end

# ═══════════════════════════════════════════════════════════════
# Figure 5: Multi-wedge diffraction patterns
# ═══════════════════════════════════════════════════════════════
function fig_wedge_patterns()
    df = CSV.read(joinpath(DATA, "fig5_wedge_patterns.csv"), DataFrame)

    alpha_map = Dict(
        "5pi4" => (raw"$\alpha=5\pi/4$", :green, :solid),
        "3pi2" => (raw"$\alpha=3\pi/2$", :red, :dash),
        "2pi"  => (raw"$\alpha=2\pi$ (half-plane)", :blue, :dot),
    )

    p1 = plot(;
        xlabel = raw"$\phi$ (deg)",
        ylabel = raw"$|D_{\mathrm{s}}|$",
        title  = "Soft polarization",
        legend = :topright,
        size   = (600, 450),
    )
    p2 = plot(;
        xlabel = raw"$\phi$ (deg)",
        ylabel = raw"$|D_{\mathrm{h}}|$",
        title  = "Hard polarization",
        legend = :topright,
        size   = (600, 450),
    )

    for akey in ["5pi4", "3pi2", "2pi"]
        sub = df[df.alpha .== akey, :]
        lbl, clr, ls = alpha_map[akey]
        plot!(p1, sub.phi_deg, sub.Ds_abs; label=lbl, color=clr, linestyle=ls)
        plot!(p2, sub.phi_deg, sub.Dh_abs; label=lbl, color=clr, linestyle=ls)
    end

    p = plot(p1, p2; layout=(1,2), size=(1100, 450), margin=5Plots.mm)
    savefig(p, joinpath(FIGS, "fig5_wedge_patterns.pdf"))
    println("  Saved fig5_wedge_patterns.pdf")
end

# ═══════════════════════════════════════════════════════════════
# Figure 6: AD gradient validation — ForwardDiff vs finite diff
# ═══════════════════════════════════════════════════════════════
function fig_ad_validation()
    # (a) Gradient wrt φ and agreement error
    df = CSV.read(joinpath(DATA, "fig6a_ad_vs_fd_phi.csv"), DataFrame)

    p1 = plot(df.phi_deg, df.ad_dDs;
        label     = raw"AD: $\partial|D_{\mathrm{s}}|/\partial\phi$",
        xlabel    = raw"$\phi$ (deg)",
        ylabel    = raw"$\partial|D_{\mathrm{s}}|/\partial\phi$",
        title     = "Derivative comparison",
        color     = :blue,
        linestyle = :solid,
        legend    = :topright,
        size      = (600, 450),
    )
    plot!(p1, df.phi_deg, df.fd_dDs;
        label = "Finite difference", color = :red, linestyle = :dash)

    p2 = plot(df.phi_deg, posclip.(df.err_s);
        label  = raw"$|D_{\mathrm{s}}|$",
        xlabel = raw"$\phi$ (deg)",
        ylabel = "Relative error (AD vs FD)",
        title  = "AD-FD relative error",
        color  = :blue,
        yscale = :log10,
        legend = :topright,
        size   = (600, 450),
    )
    plot!(p2, df.phi_deg, posclip.(df.err_h);
        label = raw"$|D_{\mathrm{h}}|$", color = :red, linestyle = :dash)

    p = plot(p1, p2; layout=(1,2), size=(1100, 450), margin=5Plots.mm)
    savefig(p, joinpath(FIGS, "fig6_ad_validation.pdf"))
    println("  Saved fig6_ad_validation.pdf")
end

# ═══════════════════════════════════════════════════════════════
# Figure 7: Branch-safety — regularized vs naïve near boundary
# ═══════════════════════════════════════════════════════════════
function fig_branch_safety()
    df = CSV.read(joinpath(DATA, "fig7_branch_safety.csv"), DataFrame)

    # Replace Inf/NaN in naïve with a sentinel for plotting
    ds_naive = copy(df.Ds_naive_abs)
    dh_naive = copy(df.Dh_naive_abs)
    ds_naive[.!isfinite.(ds_naive)] .= NaN
    dh_naive[.!isfinite.(dh_naive)] .= NaN

    p1 = plot(df.phi_deg, df.Ds_reg_abs;
        label     = "Regularized",
        xlabel    = raw"$\phi$ (deg)",
        ylabel    = raw"$|D_{\mathrm{s}}|$",
        title     = "Soft polarization near ISB",
        color     = :blue,
        linestyle = :solid,
        legend    = :topright,
        size      = (600, 450),
    )
    plot!(p1, df.phi_deg, ds_naive;
        label = raw"Naive $\cot\cdot F$", color = :red, linestyle = :dash)
    vline!(p1, [225.0]; color=:black, linestyle=:dot, linewidth=1, label="ISB")

    p2 = plot(df.phi_deg, df.Dh_reg_abs;
        label     = "Regularized",
        xlabel    = raw"$\phi$ (deg)",
        ylabel    = raw"$|D_{\mathrm{h}}|$",
        title     = "Hard polarization near ISB",
        color     = :blue,
        linestyle = :solid,
        legend    = :topright,
        size      = (600, 450),
    )
    plot!(p2, df.phi_deg, dh_naive;
        label = raw"Naive $\cot\cdot F$", color = :red, linestyle = :dash)
    vline!(p2, [225.0]; color=:black, linestyle=:dot, linewidth=1, label="ISB")

    p = plot(p1, p2; layout=(1,2), size=(1100, 450), margin=5Plots.mm)
    savefig(p, joinpath(FIGS, "fig7_branch_safety.pdf"))
    println("  Saved fig7_branch_safety.pdf")
end

# ═══════════════════════════════════════════════════════════════
# Run all
# ═══════════════════════════════════════════════════════════════
println("Generating figures …")
fig_transition()
fig_halfplane()
fig_shadow_boundary()
fig_gtd_convergence()
fig_wedge_patterns()
fig_ad_validation()
fig_branch_safety()
println("Done.  Output in $(FIGS)/")
