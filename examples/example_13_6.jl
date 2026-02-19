using UTDKernels
using PlotlySupply
using Printf

include(joinpath(@__DIR__, "common.jl"))

# Problem (Balanis Example 13-6):
# Compute and plot the principal elevation-plane pattern of a λ/4 monopole
# above a finite square PEC ground plane (w = 4 ft at f = 1 GHz), comparing
# GO only and GO+GTD diffraction contributions from the two principal edges.

function xi1_monopole(theta)
    # Observation angle for edge 1 (near edge): always θ + π/2.
    # Crosses the ISB at θ = 90° (ξ₁ = π) — a real shadow boundary
    # compensated by the GO field drop.
    return theta + π / 2
end

function xi2_monopole(theta)
    # Observation angle for edge 2 (far edge): |π/2 - θ|.
    # The raw angle π/2 - θ crosses zero at θ = 90°, which in the
    # half-plane model corresponds to crossing the PEC face. But the
    # ground plane is finite — the face ends at edge 2 — so this
    # boundary is unphysical. Using the absolute value reflects the
    # observer across the non-existent face, giving a continuous
    # diffraction contribution from edge 2.
    return abs(π / 2 - theta)
end

function run_example_13_6(; save_png = true)
    # Example parameters from text: w = 4 ft = 1.22 m, f = 1 GHz
    c0 = 299_792_458.0
    f = 1.0e9
    lambda = c0 / f
    k = 2π / lambda
    w = 1.22
    edge_distance = w / 2
    vb_edge_factor = exp(-im * k * edge_distance) / sqrt(edge_distance)

    # Avoid exact transition samples (θ = 90°) where one-point asymptotic
    # values are branch dependent; compare smooth one-sided limits instead.
    theta_deg = collect(0.25:0.5:179.75)
    theta_rad = deg2rad.(theta_deg)

    go = Float64[]
    total_pkg = ComplexF64[]
    total_txt = ComplexF64[]

    for theta in theta_rad
        sθ = sin(theta)
        go_infinite = cos((π / 2) * cos(theta)) / max(sθ, 1e-12)
        go_theta = theta <= (π / 2) ? go_infinite : 0.0

        xi1 = xi1_monopole(theta)
        xi2 = xi2_monopole(theta)

        di1_pkg = halfplane_package_coeffs(xi1, 0.0, k, edge_distance).Di
        di2_pkg = halfplane_package_coeffs(xi2, 0.0, k, edge_distance).Di
        d_pkg = vb_edge_factor * (
            di1_pkg * exp(im * k * edge_distance * sθ) -
            di2_pkg * exp(-im * k * edge_distance * sθ)
        )

        di1_txt = halfplane_textbook_coeffs(xi1, 0.0, k, edge_distance).Di
        di2_txt = halfplane_textbook_coeffs(xi2, 0.0, k, edge_distance).Di
        d_txt = vb_edge_factor * (
            di1_txt * exp(im * k * edge_distance * sθ) -
            di2_txt * exp(-im * k * edge_distance * sθ)
        )

        push!(go, go_theta)
        push!(total_pkg, go_theta + d_pkg)
        push!(total_txt, go_theta + d_txt)
    end

    err_max = max_rel_err(total_pkg, total_txt)
    err_rms = rms_rel_err(total_pkg, total_txt)

    go_db = normalize_db(abs.(go); floor_db = -30.0)
    pkg_db = normalize_db(abs.(total_pkg); floor_db = -30.0)
    txt_db = normalize_db(abs.(total_txt); floor_db = -30.0)

    theta_full = vcat(theta_deg, reverse(360 .- theta_deg))
    go_full = vcat(go_db, reverse(go_db))
    pkg_full = vcat(pkg_db, reverse(pkg_db))
    txt_full = vcat(txt_db, reverse(txt_db))

    p = plot_scatterpolar(
        theta_full,
        [pkg_full, go_full, txt_full];
        trange = [0, 360],
        rrange = [-30, 0.0],
        legend = [
            "GO + diffraction (UTDKernels)",
            "GO only (infinite ground plane)",
            "GO + diffraction (Textbook Eq., standalone)",
        ],
        color = ["#1f77b4", "#6b6b6b", "#ff7f0e"],
        dash = ["", "dash", "dash"],
        mode = "lines",
        title = "Example 13-6: λ/4 Monopole on Square Ground Plane (Polar)",
        width = 980,
        height = 900,
    )

    relayout!(
        p,
        polar = attr(
            radialaxis = attr(title = "Relative power (dB below maximum)", range = [-30, 0.0], dtick = 10),
            angularaxis = attr(
                rotation = 90,
                direction = "clockwise",
                tickmode = "array",
                tickvals = collect(0:30:330),
                ticktext = ["0°", "30°", "60°", "90°", "120°", "150°", "180°", "150°", "120°", "90°", "60°", "30°"],
            ),
        ),
        legend = attr(x = 0.5, y = 0.98, xanchor = "center", yanchor = "top"),
        margin = attr(l = 45, r = 45, t = 95, b = 55),
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
                text = @sprintf(
                    "w = %.2f m (%.2f λ at 1 GHz)<br>max rel err = %.3e<br>rms rel err = %.3e",
                    w,
                    w / lambda,
                    err_max,
                    err_rms,
                ),
                font = attr(size = 12),
                bgcolor = "rgba(255,255,255,0.85)",
                bordercolor = "rgba(0,0,0,0.3)",
                borderwidth = 1,
            ),
        ],
    )

    if save_png
        out = fig_path("example13_6_monopole_square_ground.png")
        save_plot_png(p, out; width = 980, height = 900)
    end

    println(@sprintf("Example 13-6: max rel err=%.3e rms rel err=%.3e", err_max, err_rms))
    return (max_rel = err_max, rms_rel = err_rms)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_example_13_6()
end
