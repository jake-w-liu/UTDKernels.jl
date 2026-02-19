using UTDKernels
using PlotlySupply
using Printf

include(joinpath(@__DIR__, "common.jl"))

# Problem (Balanis Example 13-4):
# For a line source above a finite PEC strip (w = 2λ, h = 0.5λ), compute the
# principal-plane GTD pattern from GO plus edge diffraction and compare with
# the infinite-strip GO reference.
#
# Note:
# The diffraction function V_B for this line-source construction carries the
# amplitude factor exp(-jks')/sqrt(s'). Using /s' would create nonphysical
# boundary kinks.

psi1_strip(phi) = phi <= π ? (π - phi) : (3π - phi)
psi2_strip(phi) = phi

function run_example_13_4(; save_png = true)
    λ = 1.0
    k = 2π / λ
    w = 2.0 * λ
    h = 0.5 * λ
    alpha = atan(2h / w)
    sprime = hypot(h, w / 2)
    vb_edge_factor = exp(-im * k * sprime) / sqrt(sprime)

    phi_deg = collect(0.125:0.25:359.875)
    phi_rad = deg2rad.(phi_deg)

    go_inf = ComplexF64[]
    go_fin = ComplexF64[]
    total_pkg = ComplexF64[]
    total_txt = ComplexF64[]

    for phi in phi_rad
        direct_present = (0.0 < phi < (π + alpha)) || ((2π - alpha) < phi < 2π)
        direct = direct_present ? exp(im * k * h * sin(phi)) : 0.0 + 0.0im
        refl_present = alpha < phi < (π - alpha)
        reflected = refl_present ? -exp(-im * k * h * sin(phi)) : 0.0 + 0.0im

        go_i = exp(im * k * h * sin(phi)) - exp(-im * k * h * sin(phi))
        go_f = direct + reflected

        psi1 = psi1_strip(phi)
        psi2 = psi2_strip(phi)

        c1_pkg = halfplane_package_coeffs(psi1, alpha, k, sprime).Ds
        c2_pkg = halfplane_package_coeffs(psi2, alpha, k, sprime).Ds
        # Example 13-4 uses V_B(s', ·) (diffraction function), not only D(·).
        # In this line-source GTD form, V_B contributes exp(-j k s') / sqrt(s').
        d_pkg = vb_edge_factor * (
            c1_pkg * exp(im * k * (w / 2) * cos(phi)) +
            c2_pkg * exp(-im * k * (w / 2) * cos(phi))
        )

        c1_txt = halfplane_textbook_coeffs(psi1, alpha, k, sprime).Ds
        c2_txt = halfplane_textbook_coeffs(psi2, alpha, k, sprime).Ds
        d_txt = vb_edge_factor * (
            c1_txt * exp(im * k * (w / 2) * cos(phi)) +
            c2_txt * exp(-im * k * (w / 2) * cos(phi))
        )

        push!(go_inf, go_i)
        push!(go_fin, go_f)
        push!(total_pkg, go_f + d_pkg)
        push!(total_txt, go_f + d_txt)
    end

    err_max = max_rel_err(total_pkg, total_txt)
    err_rms = rms_rel_err(total_pkg, total_txt)

    go_inf_db = normalize_db(abs.(go_inf); floor_db = -30.0)
    go_fin_db = normalize_db(abs.(go_fin); floor_db = -30.0)
    total_pkg_db = normalize_db(abs.(total_pkg); floor_db = -30.0)
    total_txt_db = normalize_db(abs.(total_txt); floor_db = -30.0)

    p = plot_scatterpolar(
        phi_deg,
        [total_pkg_db, go_fin_db, go_inf_db, total_txt_db];
        trange = [0, 360],
        rrange = [-30, 0.0],
        legend = [
            "GTD (UTDKernels)",
            "GO (w = 2λ, h = 0.5λ)",
            "GO (w = ∞, h = 0.5λ)",
            "GTD (Textbook Eq., standalone)",
        ],
        color = ["#1f77b4", "#6b6b6b", "#9e9e9e", "#ff7f0e"],
        dash = ["", "dash", "dot", "dash"],
        mode = "lines",
        title = "Example 13-4: Line Source Above Finite Strip (Polar Pattern)",
        width = 960,
        height = 860,
    )

    relayout!(
        p,
        polar = attr(
            radialaxis = attr(title = "Normalized |E| (dB)", range = [-30, 0.0], dtick = 10),
            angularaxis = attr(direction = "counterclockwise", rotation = 0, dtick = 30),
        ),
        legend = attr(x = 0.5, y = 0.98, xanchor = "center", yanchor = "top"),
        margin = attr(l = 45, r = 45, t = 90, b = 55),
        annotations = [
            attr(
                x = 0.99,
                y = 0.01,
                xref = "paper",
                yref = "paper",
                xanchor = "right",
                yanchor = "bottom",
                align = "right",
                showarrow = false,
                text = @sprintf("α = %.2f deg<br>s′ = %.3f λ<br>max rel err = %.3e<br>rms rel err = %.3e", rad2deg(alpha), sprime / λ, err_max, err_rms),
                font = attr(size = 12),
                bgcolor = "rgba(255,255,255,0.85)",
                bordercolor = "rgba(0,0,0,0.3)",
                borderwidth = 1,
            ),
        ],
    )

    if save_png
        out = fig_path("example13_4_strip_line_source.png")
        save_plot_png(p, out; width = 960, height = 860)
    end

    println(@sprintf("Example 13-4: max rel err=%.3e rms rel err=%.3e", err_max, err_rms))
    return (max_rel = err_max, rms_rel = err_rms)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_example_13_4()
end
