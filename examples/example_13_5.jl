using UTDKernels
using PlotlySupply
using Printf

include(joinpath(@__DIR__, "common.jl"))

# Problem (Balanis Example 13-5):
# For soft polarization on a finite strip (w = 2λ), compute the monostatic
# 2D scattering width and plot (1) 10log10(σ2D/λ) and (2) 10log10(σ2D [m]) at
# f = 10 GHz for φ = φ′.

function sigma_textbook(phi, k, w, lambda)
    x = k * w * cos(phi)
    term = cos(x) + im * (k * w) * sinc_ratio(x)
    return lambda / (2π) * abs2(term)
end

function sigma_package(phi, k, w)
    c1 = halfplane_package_coeffs(π - phi, π - phi, k, Inf).Ds
    c2 = halfplane_package_coeffs(phi, phi, k, Inf).Ds
    dsum = c1 * exp(im * k * w * cos(phi)) + c2 * exp(-im * k * w * cos(phi))
    return 2π * abs2(dsum)
end

function run_example_13_5(; save_png = true)
    # Avoid exact 90 deg where the textbook closed form is finite but each
    # individual half-plane term is singular.
    phi_deg = collect(0.125:0.25:179.875)
    phi_rad = deg2rad.(phi_deg)

    # Case A: normalized sigma/lambda in dB for w = 2 lambda
    lambda_a = 1.0
    k_a = 2π / lambda_a
    w_a = 2.0 * lambda_a

    sigma_pkg_a = [sigma_package(phi, k_a, w_a) for phi in phi_rad]
    sigma_txt_a = [sigma_textbook(phi, k_a, w_a, lambda_a) for phi in phi_rad]

    # Case B: sigma in dB/m at f = 10 GHz, w = 2 lambda
    c0 = 299_792_458.0
    f = 10.0e9
    lambda_b = c0 / f
    k_b = 2π / lambda_b
    w_b = 2.0 * lambda_b

    sigma_pkg_b = [sigma_package(phi, k_b, w_b) for phi in phi_rad]
    sigma_txt_b = [sigma_textbook(phi, k_b, w_b, lambda_b) for phi in phi_rad]

    err_a_max = max_rel_err(sigma_pkg_a, sigma_txt_a)
    err_a_rms = rms_rel_err(sigma_pkg_a, sigma_txt_a)
    err_b_max = max_rel_err(sigma_pkg_b, sigma_txt_b)
    err_b_rms = rms_rel_err(sigma_pkg_b, sigma_txt_b)

    y1_pkg = db10.(sigma_pkg_a ./ lambda_a)
    y1_txt = db10.(sigma_txt_a ./ lambda_a)
    y2_pkg = db10.(sigma_pkg_b)
    y2_txt = db10.(sigma_txt_b)

    sf = subplots(
        2,
        1;
        sync = false,
        width = 1160,
        height = 860,
        title = "",
        subplot_titles = reshape(
            [
                "Example 13-5: Monostatic 2D Scattering Width (σ2D/λ)",
                "Example 13-5: Monostatic 2D Scattering Width (dB/m) at f = 10 GHz",
            ],
            2,
            1,
        ),
    )

    addtraces!(
        sf,
        scatter(
            x = phi_deg,
            y = y1_pkg,
            mode = "lines",
            name = "UTDKernels",
            line = attr(color = "#1f77b4", width = 2.6),
        );
        row = 1,
        col = 1,
    )

    addtraces!(
        sf,
        scatter(
            x = phi_deg,
            y = y1_txt,
            mode = "lines",
            name = "Textbook Eq. 13-5",
            line = attr(color = "#ff7f0e", width = 2.0, dash = "dash"),
        );
        row = 1,
        col = 1,
    )

    addtraces!(
        sf,
        scatter(
            x = phi_deg,
            y = y2_pkg,
            mode = "lines",
            name = "UTDKernels (10 GHz)",
            line = attr(color = "#1f77b4", width = 2.6),
            showlegend = false,
        );
        row = 2,
        col = 1,
    )

    addtraces!(
        sf,
        scatter(
            x = phi_deg,
            y = y2_txt,
            mode = "lines",
            name = "Textbook Eq. (10 GHz)",
            line = attr(color = "#ff7f0e", width = 2.0, dash = "dash"),
            showlegend = false,
        );
        row = 2,
        col = 1,
    )

    p = sf.plot
    relayout!(
        p,
        xaxis = attr(title = "ϕ = ϕ′ (deg)", range = [0, 180]),
        xaxis2 = attr(title = "ϕ = ϕ′ (deg)", range = [0, 180]),
        yaxis = attr(title = "10log10(σ2D/λ)", range = [-10, 20], dtick = 5),
        yaxis2 = attr(title = "10log10(σ2D [m])", range = [-25, 0], dtick = 5),
        legend = attr(x = 0.01, y = 0.99),
        margin = attr(l = 80, r = 25, t = 90, b = 65),
        annotations = vcat(p.layout.fields[:annotations], [
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
                    "σ/λ case: max=%.3e rms=%.3e<br>10 GHz case: max=%.3e rms=%.3e",
                    err_a_max,
                    err_a_rms,
                    err_b_max,
                    err_b_rms,
                ),
                font = attr(size = 12),
                bgcolor = "rgba(255,255,255,0.85)",
                bordercolor = "rgba(0,0,0,0.3)",
                borderwidth = 1,
            ),
        ]),
    )

    if save_png
        out = fig_path("example13_5_monostatic_strip_sw.png")
        save_plot_png(p, out; width = 1160, height = 860)
    end

    println(@sprintf("Example 13-5: max rel err σ/λ=%.3e, 10GHz=%.3e", err_a_max, err_b_max))
    return (
        max_rel_sigma_over_lambda = err_a_max,
        rms_rel_sigma_over_lambda = err_a_rms,
        max_rel_10ghz = err_b_max,
        rms_rel_10ghz = err_b_rms,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_example_13_5()
end
