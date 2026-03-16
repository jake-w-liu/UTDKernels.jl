using UTDKernels
using PlotlySupply
using Printf

include(joinpath(@__DIR__, "common.jl"))

# Problem (Balanis Example 13-7):
# Compute and plot the principal elevation-plane pattern of a λ/4 monopole
# above a finite circular PEC ground plane (diameter = 4.064λ), using the
# textbook two-point diffraction approximation and comparing with GO only.

function xi1_curved(theta)
    # Observation angle for edge 1 (near edge): always θ + π/2.
    # Crosses the ISB at θ = 90° (ξ₁ = π) — a real shadow boundary
    # compensated by the GO field drop.
    return theta + π / 2
end

function xi2_curved(theta)
    # Geometric observation angle for edge 2 (far edge).
    # When xi_raw < 0, the observer has crossed the PEC face in the local
    # half-plane model. But the ground plane is finite — the face ends at
    # edge 2 — so this shadow boundary is fictitious. Reflecting the angle
    # (xi → -xi) maps the observer to the lit side of the local half-plane,
    # which is physically correct for the finite structure.
    # This is equivalent to abs() for the symmetric case but the physical
    # reasoning (fictitious ISB reflection) generalizes to 3D/asymmetric
    # geometries where φ → 2nπ - φ is used instead.
    xi_raw = π / 2 - theta
    return xi_raw >= 0 ? xi_raw : -xi_raw
end

function run_example_13_7(; save_png = true)
    # Example text: diameter = 4.064 lambda -> radius a = 2.032 lambda
    lambda = 1.0
    k = 2π / lambda
    a = 2.032 * lambda
    vb_edge_factor = exp(-im * k * a) / sqrt(a)

    # Stay away from exact boundary points where one-point asymptotic values
    # are branch dependent. Keep textbook two-point validity window.
    theta_deg = collect(10.25:0.5:169.75)
    theta_rad = deg2rad.(theta_deg)

    go = Float64[]
    total_pkg = ComplexF64[]
    total_txt = ComplexF64[]

    for theta in theta_rad
        sθ = sin(theta)
        go_infinite = cos((π / 2) * cos(theta)) / max(sθ, 1e-12)
        go_theta = theta <= (π / 2) ? go_infinite : 0.0

        xi1 = xi1_curved(theta)
        xi2 = xi2_curved(theta)

        di1_pkg = halfplane_package_coeffs(xi1, 0.0, k, a).Di
        di2_pkg = halfplane_package_coeffs(xi2, 0.0, k, a).Di
        # Balanis Ex. 13-7 (Fig. 13-38/13-39 derivation):
        #   E_d1 ∝ exp(+jka sinθ)/sqrt(sinθ)
        #   E_d2 ∝ -exp(-jka sinθ)/sqrt(-sinθ)
        # Using 1/sinθ (instead of square roots) creates incorrect edge scaling.
        d_pkg = vb_edge_factor * (
            di1_pkg * exp(im * k * a * sθ) / sqrt(complex(sθ, 0.0)) -
            di2_pkg * exp(-im * k * a * sθ) / sqrt(complex(-sθ, 0.0))
        )

        di1_txt = halfplane_textbook_coeffs(xi1, 0.0, k, a).Di
        di2_txt = halfplane_textbook_coeffs(xi2, 0.0, k, a).Di
        d_txt = vb_edge_factor * (
            di1_txt * exp(im * k * a * sθ) / sqrt(complex(sθ, 0.0)) -
            di2_txt * exp(-im * k * a * sθ) / sqrt(complex(-sθ, 0.0))
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

    # Break the curve across the invalid two-point region near 0°/180°
    # (where equivalent-current/ring-radiator modeling is required).
    theta_full = vcat(theta_deg, missing, reverse(360 .- theta_deg))
    go_full = vcat(go_db, missing, reverse(go_db))
    pkg_full = vcat(pkg_db, missing, reverse(pkg_db))
    txt_full = vcat(txt_db, missing, reverse(txt_db))

    p = plot_scatterpolar(
        theta_full,
        [pkg_full, go_full, txt_full];
        trange = [0, 360],
        rrange = [-30, 0.0],
        legend = [
            "GO + two-point diffraction (UTDKernels)",
            "GO only",
            "GO + two-point diffraction (Textbook Eq., standalone)",
        ],
        color = ["#1f77b4", "#6b6b6b", "#ff7f0e"],
        dash = ["", "dash", "dash"],
        mode = "lines",
        title = "Example 13-7: λ/4 Monopole on Circular Ground Plane (Polar, two-point region)",
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
                    "a = %.3f λ (diameter = 4.064 λ)<br>validity window: 10.25° ≤ θ ≤ 169.75°<br>max rel err = %.3e<br>rms rel err = %.3e",
                    a / lambda,
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
        out = fig_path("example13_7_monopole_circular_ground.png")
        save_plot_png(p, out; width = 980, height = 900)
    end

    println(@sprintf("Example 13-7: max rel err=%.3e rms rel err=%.3e", err_max, err_rms))
    return (max_rel = err_max, rms_rel = err_rms)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_example_13_7()
end
