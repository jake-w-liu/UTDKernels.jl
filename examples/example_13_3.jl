using UTDKernels
using PlotlySupply
using Printf

include(joinpath(@__DIR__, "common.jl"))

# Problem (Balanis Example 13-3):
# For a plane wave normally incident on a PEC half-plane at ρ = λ and
# φ′ = 30°, compute GO, incident-diffracted, reflected-diffracted, and total
# fields (soft and hard polarizations) versus observation angle φ.

function run_example_13_3(; save_png = true)
    λ = 1.0
    k = 2π / λ
    rho = λ
    phip = deg2rad(30.0)

    phi_deg = collect(0.125:0.25:359.875)
    phi_rad = deg2rad.(phi_deg)

    go_s = ComplexF64[]
    go_h = ComplexF64[]
    id_pkg = ComplexF64[]
    rd_pkg_s = ComplexF64[]
    rd_pkg_h = ComplexF64[]
    total_pkg_s = ComplexF64[]
    total_pkg_h = ComplexF64[]
    total_txt_s = ComplexF64[]
    total_txt_h = ComplexF64[]

    phase_factor = exp(-im * k * rho) / sqrt(rho)

    for phi in phi_rad
        go = go_halfplane_components(phi, phip, k, rho)
        pkg = halfplane_package_coeffs(phi, phip, k, rho)
        txt = halfplane_textbook_coeffs(phi, phip, k, rho)

        idp = pkg.Di * phase_factor
        rsp = (pkg.Ds - pkg.Di) * phase_factor
        rhp = (pkg.Dh - pkg.Di) * phase_factor

        ids_txt = txt.Di * phase_factor
        rds_txt = (txt.Ds - txt.Di) * phase_factor
        rdh_txt = (txt.Dh - txt.Di) * phase_factor

        tp_s = go.soft + idp + rsp
        tp_h = go.hard + idp + rhp
        tt_s = go.soft + ids_txt + rds_txt
        tt_h = go.hard + ids_txt + rdh_txt

        push!(go_s, go.soft)
        push!(go_h, go.hard)
        push!(id_pkg, idp)
        push!(rd_pkg_s, rsp)
        push!(rd_pkg_h, rhp)
        push!(total_pkg_s, tp_s)
        push!(total_pkg_h, tp_h)
        push!(total_txt_s, tt_s)
        push!(total_txt_h, tt_h)
    end

    err_soft = max_rel_err(total_pkg_s, total_txt_s)
    err_hard = max_rel_err(total_pkg_h, total_txt_h)

    sf = subplots(
        2,
        2;
        sync = false,
        width = 1240,
        height = 860,
        title = "",
        subplot_titles = reshape(
            [
                "Example 13-3 Soft: |Field|",
                "Example 13-3 Hard: |Field|",
                "Example 13-3 Soft: Phase(total)",
                "Example 13-3 Hard: Phase(total)",
            ],
            2,
            2,
        ),
    )

    addtraces!(
        sf,
        scatter(
            x = phi_deg,
            y = abs.(go_s),
            mode = "lines",
            name = "GO (soft)",
            line = attr(color = "#7f7f7f", width = 1.8, dash = "dash"),
        );
        row = 1,
        col = 1,
    )

    addtraces!(
        sf,
        scatter(
            x = phi_deg,
            y = abs.(id_pkg),
            mode = "lines",
            name = "Incident diffraction",
            line = attr(color = "#1f77b4", width = 1.6, dash = "dot"),
        );
        row = 1,
        col = 1,
    )

    addtraces!(
        sf,
        scatter(
            x = phi_deg,
            y = abs.(rd_pkg_s),
            mode = "lines",
            name = "Reflected diffraction (soft)",
            line = attr(color = "#d62728", width = 1.6, dash = "dashdot"),
        );
        row = 1,
        col = 1,
    )

    addtraces!(
        sf,
        scatter(
            x = phi_deg,
            y = abs.(total_pkg_s),
            mode = "lines",
            name = "Total (UTDKernels)",
            line = attr(color = "#111111", width = 2.5),
        );
        row = 1,
        col = 1,
    )

    addtraces!(
        sf,
        scatter(
            x = phi_deg,
            y = abs.(total_txt_s),
            mode = "lines",
            name = "Total (Textbook Eq.)",
            line = attr(color = "#ff7f0e", width = 2.0, dash = "dash"),
        );
        row = 1,
        col = 1,
    )

    addtraces!(
        sf,
        scatter(
            x = phi_deg,
            y = abs.(go_h),
            mode = "lines",
            name = "GO (hard)",
            line = attr(color = "#7f7f7f", width = 1.8, dash = "dash"),
            showlegend = false,
        );
        row = 1,
        col = 2,
    )

    addtraces!(
        sf,
        scatter(
            x = phi_deg,
            y = abs.(id_pkg),
            mode = "lines",
            name = "Incident diffraction",
            line = attr(color = "#1f77b4", width = 1.6, dash = "dot"),
            showlegend = false,
        );
        row = 1,
        col = 2,
    )

    addtraces!(
        sf,
        scatter(
            x = phi_deg,
            y = abs.(rd_pkg_h),
            mode = "lines",
            name = "Reflected diffraction (hard)",
            line = attr(color = "#d62728", width = 1.6, dash = "dashdot"),
            showlegend = false,
        );
        row = 1,
        col = 2,
    )

    addtraces!(
        sf,
        scatter(
            x = phi_deg,
            y = abs.(total_pkg_h),
            mode = "lines",
            name = "Total (UTDKernels)",
            line = attr(color = "#111111", width = 2.5),
            showlegend = false,
        );
        row = 1,
        col = 2,
    )

    addtraces!(
        sf,
        scatter(
            x = phi_deg,
            y = abs.(total_txt_h),
            mode = "lines",
            name = "Total (Textbook Eq.)",
            line = attr(color = "#ff7f0e", width = 2.0, dash = "dash"),
            showlegend = false,
        );
        row = 1,
        col = 2,
    )

    addtraces!(
        sf,
        scatter(
            x = phi_deg,
            y = rad2deg.(angle.(total_pkg_s)),
            mode = "lines",
            name = "Soft total phase (UTDKernels)",
            line = attr(color = "#111111", width = 2.2),
            showlegend = false,
        );
        row = 2,
        col = 1,
    )

    addtraces!(
        sf,
        scatter(
            x = phi_deg,
            y = rad2deg.(angle.(total_txt_s)),
            mode = "lines",
            name = "Soft total phase (Textbook)",
            line = attr(color = "#ff7f0e", width = 1.8, dash = "dash"),
            showlegend = false,
        );
        row = 2,
        col = 1,
    )

    addtraces!(
        sf,
        scatter(
            x = phi_deg,
            y = rad2deg.(angle.(total_pkg_h)),
            mode = "lines",
            name = "Hard total phase (UTDKernels)",
            line = attr(color = "#111111", width = 2.2),
            showlegend = false,
        );
        row = 2,
        col = 2,
    )

    addtraces!(
        sf,
        scatter(
            x = phi_deg,
            y = rad2deg.(angle.(total_txt_h)),
            mode = "lines",
            name = "Hard total phase (Textbook)",
            line = attr(color = "#ff7f0e", width = 1.8, dash = "dash"),
            showlegend = false,
        );
        row = 2,
        col = 2,
    )

    p = sf.plot
    relayout!(
        p,
        xaxis = attr(title = "ϕ (deg)", range = [0, 360]),
        xaxis2 = attr(title = "ϕ (deg)", range = [0, 360]),
        xaxis3 = attr(title = "ϕ (deg)", range = [0, 360]),
        xaxis4 = attr(title = "ϕ (deg)", range = [0, 360]),
        yaxis = attr(title = "|E| (linear)", range = [0, 2.5], dtick = 0.5),
        yaxis2 = attr(title = "|E| (linear)", range = [0, 2.5], dtick = 0.5),
        yaxis3 = attr(title = "Phase (deg)", range = [-180, 180], dtick = 90),
        yaxis4 = attr(title = "Phase (deg)", range = [-180, 180], dtick = 90),
        legend = attr(x = 0.01, y = 0.99),
        margin = attr(l = 70, r = 20, t = 80, b = 60),
        annotations = vcat(p.layout[:annotations], [
            attr(
                x = 0.99,
                y = 0.01,
                xref = "paper",
                yref = "paper",
                xanchor = "right",
                yanchor = "bottom",
                align = "right",
                showarrow = false,
                text = @sprintf("max rel err soft = %.3e<br>max rel err hard = %.3e", err_soft, err_hard),
                font = attr(size = 12),
                bgcolor = "rgba(255,255,255,0.85)",
                bordercolor = "rgba(0,0,0,0.3)",
                borderwidth = 1,
            ),
        ]),
    )

    if save_png
        out = fig_path("example13_3_halfplane_components.png")
        save_plot_png(p, out; width = 1240, height = 860)
    end

    println(@sprintf("Example 13-3: max rel err soft=%.3e hard=%.3e", err_soft, err_hard))
    return (max_rel_soft = err_soft, max_rel_hard = err_hard)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_example_13_3()
end
