#!/usr/bin/env julia
# PlotlySupply figures generated only from persisted paper/data CSV files.
# evidence_min_rows: 6
# evidence_density_justification: bandlimited statistics use 256 probes at
#   seven kappa values and 5,376 bound checks; half-plane and curved maps use
#   eight and seven logarithmic kappa values; the Sommerfeld benchmark spans
#   six logarithmic kappa levels and sets the declared minimum; exact
#   derivative verification contains 336 values; switch, multiplier, and
#   sensitivity profiles use 1,400, 800, and 1,800 coordinates; the sampling
#   study uses 44 grids up to 65,536 panels; the translated objective uses
#   241 bandwidth values.

using PlotlySupply
using Statistics

include(joinpath(@__DIR__, "..", "..", "common.jl"))
using .PaperReproductionPaths
const PAPER = output_directory(170)
const DATA = joinpath(PAPER, "data")
const FIGS = joinpath(PAPER, "figs")
const LEGEND_AUDIT = get(ENV, "LEGEND_AUDIT", "0") == "1"
const LEGEND_AUDIT_DIR = joinpath(PAPER, "..", "tmp", "legend-audit-170")

const COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#000000",
                "#E69F00"]
const DASHES = ["solid", "dash", "dashdot", "dot", "longdash", "longdashdot"]
const INSIDE_POSITIONS = [:topright, :topleft, :bottomright, :bottomleft,
                          :top, :bottom, :right, :left]
const IEEE_SINGLE_COL_W = 504
const IEEE_SINGLE_COL_H = 360
const IEEE_DOUBLE_COL_W = 1008
const IEEE_DOUBLE_COL_H = 420
const FS = 12

mkpath(FIGS)
LEGEND_AUDIT && mkpath(LEGEND_AUDIT_DIR)

function load_columns(path::AbstractString)
    lines = filter(!isempty, readlines(path))
    lines = filter(line -> !startswith(lstrip(line), '#'), lines)
    isempty(lines) && error("CSV is empty: $path")
    header = split(lines[1], ',')
    rows = [parse.(Float64, split(line, ',')) for line in lines[2:end]]
    columns = Dict{String,Vector{Float64}}()
    for (index, name) in enumerate(header)
        columns[String(name)] = [row[index] for row in rows]
    end
    return columns
end

plot_object(figure) = figure isa PlotlySupply.SubplotFigure ?
                      plot_object(figure.fig) :
                      hasproperty(figure, :plot) ? figure.plot : figure

function tight_margins!(figure)
    plot_object(figure).layout[:margin] =
        Dict(:l => 74, :r => 18, :t => 18, :b => 58)
    return figure
end

function power_ticks!(figure)
    plot = plot_object(figure)
    for key in keys(plot.layout)
        name = String(key)
        if startswith(name, "xaxis") || startswith(name, "yaxis")
            plot.layout[key][:exponentformat] = "power"
            plot.layout[key][:minexponent] = 3
        end
    end
    return figure
end

positive(values) = max.(values, 1e-18)

function audit_legend_positions!(figure, stem::AbstractString;
                                 width::Integer, height::Integer)
    LEGEND_AUDIT || return figure
    for position in INSIDE_POSITIONS
        set_legend!(figure; position=position)
        savefig(figure, joinpath(LEGEND_AUDIT_DIR,
                                 "$(stem)_$(position).pdf");
                width=width, height=height)
    end
    return figure
end

function compact_horizontal_legend!(figure; position::Symbol)
    set_legend!(figure; position=position)
    legend = plot_object(figure).layout[:legend]
    legend[:orientation] = "h"
    legend[:font] = Dict(:size => 9)
    return figure
end

function compact_subplot_legends!(figure, count::Integer)
    plot = plot_object(figure)
    for index in 1:count
        legend_key = index == 1 ? :legend : Symbol("legend$index")
        haskey(plot.layout, legend_key) ||
            throw(ArgumentError("compact legend $index has no matching subplot"))
        plot.layout[legend_key][:orientation] = "h"
        plot.layout[legend_key][:font] = Dict(:size => 9)
    end
    return figure
end

function inside_legend_anchor(xdom, ydom, inset, position::Symbol)
    position in INSIDE_POSITIONS ||
        throw(ArgumentError("unsupported inside legend position: $position"))
    xpad = clamp(Float64(inset[1]), 0.0, 0.49)
    ypad = clamp(Float64(inset[2]), 0.0, 0.49)
    xleft = xdom[1] + (xdom[2] - xdom[1]) * xpad
    xright = xdom[2] - (xdom[2] - xdom[1]) * xpad
    xcenter = (xdom[1] + xdom[2]) / 2
    ybottom = ydom[1] + (ydom[2] - ydom[1]) * ypad
    ytop = ydom[2] - (ydom[2] - ydom[1]) * ypad
    ycenter = (ydom[1] + ydom[2]) / 2
    position === :topright && return (xright, ytop, "right", "top")
    position === :topleft && return (xleft, ytop, "left", "top")
    position === :bottomright && return (xright, ybottom, "right", "bottom")
    position === :bottomleft && return (xleft, ybottom, "left", "bottom")
    position === :top && return (xcenter, ytop, "center", "top")
    position === :bottom && return (xcenter, ybottom, "center", "bottom")
    position === :right && return (xright, ycenter, "right", "middle")
    return (xleft, ycenter, "left", "middle")
end

function subplot_legends_at!(figure, positions::Vector{Symbol})
    isempty(positions) && throw(ArgumentError("at least one panel legend position is required"))
    inset = (0.02, 0.03)
    background = "rgba(255,255,255,0.72)"
    border = "rgba(0,0,0,0.15)"
    subplot_legends!(figure; position=first(positions), legend_inset=inset,
                     legend_bgcolor=background, legend_bordercolor=border,
                     legend_borderwidth=1.0)
    plot = plot_object(figure)
    for (index, position) in enumerate(positions)
        xaxis = index == 1 ? :xaxis : Symbol("xaxis$index")
        yaxis = index == 1 ? :yaxis : Symbol("yaxis$index")
        legend_key = index == 1 ? :legend : Symbol("legend$index")
        all(haskey(plot.layout, key) for key in (xaxis, yaxis, legend_key)) ||
            throw(ArgumentError("legend position $index has no matching subplot"))
        x_domain = Tuple(Float64.(plot.layout[xaxis][:domain]))
        y_domain = Tuple(Float64.(plot.layout[yaxis][:domain]))
        x, y, x_anchor, y_anchor =
            inside_legend_anchor(x_domain, y_domain, inset, position)
        legend = plot.layout[legend_key]
        legend[:x] = x
        legend[:y] = y
        legend[:xanchor] = x_anchor
        legend[:yanchor] = y_anchor
    end
    return figure
end

function audit_subplot_legends!(figure, stem::AbstractString,
                                selected::Vector{Symbol};
                                width::Integer, height::Integer)
    LEGEND_AUDIT || return figure
    for panel in eachindex(selected), position in INSIDE_POSITIONS
        trial = copy(selected)
        trial[panel] = position
        subplot_legends_at!(figure, trial)
        savefig(figure, joinpath(LEGEND_AUDIT_DIR,
                                 "$(stem)_panel$(panel)_$(position).pdf");
                width=width, height=height)
    end
    return figure
end

# figure_evidence_begin
# figure_outputs: fig_canonical_multiplier.pdf
# evidence_sources: canonical_switch_profiles.csv,multiplier_phase.csv
# evidence_min_rows: 800
# evidence_density_justification: 1400 switch samples and 800 multiplier samples resolve narrowing physical profiles and the exact quadratic spectral phase.
function figure_canonical_multiplier()
    switch_data = load_columns(joinpath(DATA, "canonical_switch_profiles.csv"))
    multiplier_data = load_columns(joinpath(DATA, "multiplier_phase.csv"))
    figure = subplots(1, 2; show=false, sync=false, title="",
                      per_subplot_legends=true,
                      legend_position=:topright)
    subplot!(figure, 1, 1)
    switch_series = (
        ("W_kappa8_real_nd", raw"$\mathrm{Re}\,W,\ \kappa=8$"),
        ("W_kappa8_imag_nd", raw"$\mathrm{Im}\,W,\ \kappa=8$"),
        ("W_kappa32_real_nd", raw"$\mathrm{Re}\,W,\ \kappa=32$"),
        ("W_kappa32_imag_nd", raw"$\mathrm{Im}\,W,\ \kappa=32$"),
    )
    for (index, (name, legend)) in enumerate(switch_series)
        plot_scatter!(figure, switch_data["s_nd"], switch_data[name];
                      mode="lines", color=COLORS[div(index - 1, 2) + 1],
                      dash=DASHES[isodd(index) ? 1 : 2], legend=legend,
                      linewidth=2, fontsize=FS)
    end
    xlabel!(figure, raw"$\mathit{signed\ coordinate}\ s$")
    ylabel!(figure, raw"$\mathit{switch\ component}$")
    subplot!(figure, 1, 2)
    for (index, kappa) in enumerate((8, 32, 128))
        plot_scatter!(figure, multiplier_data["xi_nd"],
                      multiplier_data["phase_kappa$(kappa)_rad"];
                      mode="lines", color=COLORS[index], dash=DASHES[index],
                      legend="\$\\kappa=$kappa\$", linewidth=2, fontsize=FS)
    end
    xlabel!(figure, raw"$\mathit{spatial\ frequency}\ \xi$")
    ylabel!(figure, raw"$\mathit{multiplier\ phase}\ [\mathrm{rad}]$")
    selected = [:topleft, :top]
    audit_subplot_legends!(figure, "canonical_multiplier", selected;
                           width=IEEE_DOUBLE_COL_W,
                           height=IEEE_DOUBLE_COL_H)
    # The switch rises toward the right, leaving the upper-left least
    # obstructed. The multiplier parabolas are lowest at the center, leaving
    # the top center with the smallest visible overlap.
    subplot_legends_at!(figure, selected)
    savefig(figure, joinpath(FIGS, "fig_canonical_multiplier.pdf");
            width=IEEE_DOUBLE_COL_W, height=IEEE_DOUBLE_COL_H)
end
# figure_evidence_end

# figure_evidence_begin
# figure_outputs: fig_bandlimited.pdf
# evidence_sources: bandlimited_summary.csv
# evidence_min_rows: 7
# evidence_density_justification: seven logarithmic kappa levels resolve the distinct first-, second-, and third-order action-error slopes reported in the manuscript.
function figure_bandlimited()
    data = load_columns(joinpath(DATA, "bandlimited_summary.csv"))
    series = (("median_delta_nd", raw"$\mathit{Dirac}$"),
              ("median_first_nd", raw"$\mathit{first}\ \mathrm{UTD}$"),
              ("median_second_nd", raw"$\mathit{second}\ \mathrm{UTD}$"),
              ("median_gaussian_nd", raw"$\mathit{real\ Gaussian}$"))
    first_name, first_legend = first(series)
    figure = plot_scatter(data["kappa_nd"], positive(data[first_name]);
                          xlabel=raw"$\mathit{large\ parameter}\ \kappa$",
                          ylabel=raw"$\mathit{median\ relative\ action\ error}$",
                          mode="lines+markers", color=COLORS[1],
                          dash=DASHES[1], marker_size=5,
                          legend=first_legend, linewidth=2,
                          xscale="log", yscale="log", fontsize=FS,
                          show=false)
    for (offset, (name, legend)) in enumerate(series[2:end])
        index = offset + 1
        plot_scatter!(figure, data["kappa_nd"], positive(data[name]);
                      mode="lines+markers", color=COLORS[index],
                      dash=DASHES[index], marker_size=5,
                      legend=legend, linewidth=2)
    end
    audit_legend_positions!(figure, "bandlimited";
                            width=IEEE_SINGLE_COL_W,
                            height=IEEE_SINGLE_COL_H)
    # All error curves descend toward the right. The lower-left corner lies
    # below their high-k-independent starting values.
    set_legend!(figure; position=:bottomleft)
    tight_margins!(figure)
    power_ticks!(figure)
    savefig(figure, joinpath(FIGS, "fig_bandlimited.pdf");
            width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
end
# figure_evidence_end

# figure_evidence_begin
# figure_evidence_end

# figure_evidence_begin
# figure_outputs: fig_halfplane.pdf
# evidence_sources: halfplane_pullback.csv
# evidence_min_rows: 8
# evidence_density_justification: eight logarithmic kappa levels resolve the three successive half-plane pullback orders over the fitted tail.
function figure_halfplane()
    data = load_columns(joinpath(DATA, "halfplane_pullback.csv"))
    series = (("error_leading_nd", raw"$\mathit{leading}$"),
              ("error_first_nd", raw"$\mathit{first\ correction}$"),
              ("error_second_nd", raw"$\mathit{second\ correction}$"))
    first_name, first_legend = first(series)
    figure = plot_scatter(data["kappa_nd"], positive(data[first_name]);
                          xlabel=raw"$k\rho$", ylabel=raw"$\mathit{absolute\ action\ error}$",
                          mode="lines+markers", color=COLORS[1],
                          dash=DASHES[1], marker_size=5,
                          legend=first_legend, linewidth=2,
                          xscale="log", yscale="log", fontsize=FS,
                          show=false)
    for (offset, (name, legend)) in enumerate(series[2:end])
        index = offset + 1
        plot_scatter!(figure, data["kappa_nd"], positive(data[name]);
                      mode="lines+markers", color=COLORS[index],
                      dash=DASHES[index], marker_size=5,
                      legend=legend, linewidth=2)
    end
    audit_legend_positions!(figure, "halfplane";
                            width=IEEE_SINGLE_COL_W,
                            height=IEEE_SINGLE_COL_H)
    # The curves start high on the left and descend. The lower-left corner is
    # below all three starting values and is least obstructive.
    set_legend!(figure; position=:bottomleft)
    tight_margins!(figure)
    power_ticks!(figure)
    savefig(figure, joinpath(FIGS, "fig_halfplane.pdf");
            width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
end
# figure_evidence_end

# figure_evidence_begin
# figure_outputs: fig_curved.pdf
# evidence_sources: curved_coordinate.csv
# evidence_min_rows: 7
# evidence_density_justification: seven logarithmic kappa levels distinguish width-only, curvature-complete, and second-order nonlinear-coordinate behavior.
function figure_curved()
    data = load_columns(joinpath(DATA, "curved_coordinate.csv"))
    series = (("error_leading_nd", raw"$\mathit{leading}$"),
              ("error_width_only_nd", raw"$\mathit{width\ only}$"),
              ("error_full_first_nd", raw"$\mathit{complete\ first}$"),
              ("error_full_second_nd", raw"$\mathit{complete\ second}$"))
    first_name, first_legend = first(series)
    figure = plot_scatter(data["kappa_nd"], positive(data[first_name]);
                          xlabel=raw"$\mathit{large\ parameter}\ \kappa$",
                          ylabel=raw"$\mathit{absolute\ action\ error}$",
                          mode="lines+markers", color=COLORS[1],
                          dash=DASHES[1], marker_size=5,
                          legend=first_legend, linewidth=2,
                          xscale="log", yscale="log", fontsize=FS,
                          show=false)
    for (offset, (name, legend)) in enumerate(series[2:end])
        index = offset + 1
        plot_scatter!(figure, data["kappa_nd"], positive(data[name]);
                      mode="lines+markers", color=COLORS[index],
                      dash=DASHES[index], marker_size=5,
                      legend=legend, linewidth=2)
    end
    audit_legend_positions!(figure, "curved";
                            width=IEEE_SINGLE_COL_W,
                            height=IEEE_SINGLE_COL_H)
    # The four curves start above the lower axis and descend. The lower-left
    # corner is the least obstructive inside position.
    set_legend!(figure; position=:bottomleft)
    tight_margins!(figure)
    power_ticks!(figure)
    savefig(figure, joinpath(FIGS, "fig_curved.pdf");
            width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
end
# figure_evidence_end

# figure_evidence_begin
# figure_outputs: fig_sensitivity_profiles.pdf
# evidence_sources: sensitivity_profiles.csv
# evidence_min_rows: 1800
# evidence_density_justification: 1800 signed-coordinate samples at two electrical scales resolve the oscillatory derivative profiles and tails.
function figure_sensitivity_profiles()
    data = load_columns(joinpath(DATA, "sensitivity_profiles.csv"))
    figure = subplots(1, 2; show=false, sync=false, title="",
                      per_subplot_legends=true,
                      legend_position=:topright)
    subplot!(figure, 1, 1)
    for (index, kappa) in enumerate((16, 64))
        plot_scatter!(figure, data["s_rad"],
                      data["d1_kappa$(kappa)_real_nd"];
                      mode="lines", color=COLORS[index], dash=DASHES[index],
                      legend="\$\\kappa=$kappa\$", linewidth=2, fontsize=FS)
    end
    xlabel!(figure, raw"$s=\pi-\psi\ [\mathrm{rad}]$")
    ylabel!(figure, raw"$\operatorname{Re}(\widetilde V'/\sqrt{\kappa})$")
    subplot!(figure, 1, 2)
    for (index, kappa) in enumerate((16, 64))
        plot_scatter!(figure, data["s_rad"],
                      data["d1_kappa$(kappa)_imag_nd"];
                      mode="lines", color=COLORS[index], dash=DASHES[index],
                      legend="\$\\kappa=$kappa\$", linewidth=2, fontsize=FS)
    end
    xlabel!(figure, raw"$s=\pi-\psi\ [\mathrm{rad}]$")
    ylabel!(figure, raw"$\operatorname{Im}(\widetilde V'/\sqrt{\kappa})$")
    selected = [:top, :top]
    # The oscillatory traces fill the natural range, so the y-ceilings are
    # raised to free a top band; the compact legends sit there clear of data.
    yrange!(figure, [-0.48, 0.82]; row = 1, col = 1)
    yrange!(figure, [-0.48, 0.82]; row = 1, col = 2)
    audit_subplot_legends!(figure, "sensitivity_profiles", selected;
                           width=IEEE_DOUBLE_COL_W,
                           height=IEEE_DOUBLE_COL_H)
    subplot_legends_at!(figure, selected)
    compact_subplot_legends!(figure, 2)
    savefig(figure, joinpath(FIGS, "fig_sensitivity_profiles.pdf");
            width=IEEE_DOUBLE_COL_W, height=IEEE_DOUBLE_COL_H)
end
# figure_evidence_end

# figure_evidence_begin
# figure_outputs: fig_sampling.pdf
# evidence_sources: sampling_burden.csv
# evidence_min_rows: 44
# evidence_density_justification: 11 panel counts at each of four kappa values resolve the fixed-interval sampling burden and accepted thresholds.
function figure_sampling()
    data = load_columns(joinpath(DATA, "sampling_burden.csv"))
    kappas = (16.0, 64.0, 256.0, 1024.0)
    mask = data["kappa_nd"] .== first(kappas)
    figure = plot_scatter(data["N_panels_nd"][mask],
                          positive(data["absolute_error_nd"][mask]);
                          xlabel=raw"$\mathit{uniform\ trapezoidal\ panels}$",
                          ylabel=raw"$\mathit{absolute\ action\ error}$",
                          mode="lines+markers", color=COLORS[1],
                          dash=DASHES[1], marker_size=4,
                          legend=raw"$\kappa=16$", linewidth=2,
                          xscale="log", yscale="log", fontsize=FS,
                          show=false)
    for (offset, kappa) in enumerate(kappas[2:end])
        index = offset + 1
        mask = data["kappa_nd"] .== kappa
        plot_scatter!(figure, data["N_panels_nd"][mask],
                      positive(data["absolute_error_nd"][mask]);
                      mode="lines+markers", color=COLORS[index],
                      dash=DASHES[index], marker_size=4,
                      legend="\$\\kappa=$(Int(kappa))\$", linewidth=2)
    end
    audit_legend_positions!(figure, "sampling";
                            width=IEEE_SINGLE_COL_W,
                            height=IEEE_SINGLE_COL_H)
    # Error drops occur progressively farther right. Beyond the final drop,
    # every curve lies near the lower axis, leaving the upper-right corner
    # unobstructed.
    set_legend!(figure; position=:topright)
    tight_margins!(figure)
    power_ticks!(figure)
    savefig(figure, joinpath(FIGS, "fig_sampling.pdf");
            width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
end
# figure_evidence_end

# figure_evidence_begin
# figure_outputs: fig_translated.pdf
# evidence_sources: translated_objective.csv
# evidence_min_rows: 241
# evidence_density_justification: 241 carrier frequencies resolve exact, asymptotic, and Gaussian translated-boundary actions through the low-bandwidth regime and failure boundary.
function figure_translated()
    data = load_columns(joinpath(DATA, "translated_objective.csv"))
    figure = subplots(1, 2; show=false, sync=false, title="",
                      per_subplot_legends=true,
                      legend_position=:topright)
    subplot!(figure, 1, 1)
    phase_series = (("exact_phase_deg", raw"$\mathit{exact}\ \mathrm{UTD}$"),
                    ("first_phase_deg", raw"$\mathit{first\ correction}$"),
                    ("gaussian_phase_deg", raw"$\mathit{real\ Gaussian}$"))
    for (index, (name, legend)) in enumerate(phase_series)
        plot_scatter!(figure, data["zeta_nd"], data[name];
                      mode="lines", color=COLORS[index], dash=DASHES[index],
                      legend=legend, linewidth=2, fontsize=FS)
    end
    xlabel!(figure, raw"$\zeta=\nu^2/(4\kappa)$")
    ylabel!(figure, raw"$\mathit{action\ phase}\ [\mathrm{deg}]$")
    subplot!(figure, 1, 2)
    error_series = (("error_delta_nd", raw"$\mathit{Dirac}$"),
                    ("error_first_nd", raw"$\mathit{first\ correction}$"),
                    ("error_gaussian_nd", raw"$\mathit{real\ Gaussian}$"))
    for (index, (name, legend)) in enumerate(error_series)
        plot_scatter!(figure, data["zeta_nd"], positive(data[name]);
                      mode="lines", color=COLORS[index], dash=DASHES[index],
                      legend=legend, linewidth=2, yscale="log", fontsize=FS)
    end
    xlabel!(figure, raw"$\zeta=\nu^2/(4\kappa)$")
    ylabel!(figure, raw"$\mathit{relative\ action\ error}$")
    selected = [:topleft, :bottomright]
    audit_subplot_legends!(figure, "translated", selected;
                           width=IEEE_DOUBLE_COL_W,
                           height=IEEE_DOUBLE_COL_H)
    # The phase panel grows toward the right, leaving its upper-left open.
    # All error curves finish high on the right, leaving the lower-right of
    # the logarithmic error panel unobstructed.
    subplot_legends_at!(figure, selected)
    power_ticks!(figure)
    savefig(figure, joinpath(FIGS, "fig_translated.pdf");
            width=IEEE_DOUBLE_COL_W, height=IEEE_DOUBLE_COL_H)
end
# figure_evidence_end

# figure_evidence_begin
# figure_outputs: fig_sommerfeld.pdf
# evidence_sources: sommerfeld_benchmark.csv
# evidence_min_rows: 6
# evidence_density_justification: six logarithmic kappa levels resolve the demodulated-observable convergence orders and the misfit nonuniformity across two decades.
function figure_sommerfeld()
    data = load_columns(joinpath(DATA, "sommerfeld_benchmark.csv"))
    figure = subplots(1, 2; show=false, sync=false, title="",
                      per_subplot_legends=true,
                      legend_position=:topright)
    subplot!(figure, 1, 1)
    s1_series = (("s1_err_dirac_nd", raw"$\mathit{Dirac}$"),
                 ("s1_err_first_nd", raw"$\mathit{first\ UTD}$"),
                 ("s1_err_second_nd", raw"$\mathit{second\ UTD}$"),
                 ("s1_err_gauss_nd", raw"$\mathit{real\ Gaussian}$"))
    for (index, (name, legend)) in enumerate(s1_series)
        plot_scatter!(figure, data["kappa_nd"], positive(data[name]);
                      mode="lines+markers", color=COLORS[index],
                      dash=DASHES[index], marker_size=5,
                      legend=legend, linewidth=2, fontsize=FS,
                      xscale="log", yscale="log")
    end
    xlabel!(figure, raw"$\mathit{large\ parameter}\ \kappa$")
    ylabel!(figure, raw"$\mathit{absolute\ }S_1\ \mathit{action\ error}$")
    subplot!(figure, 1, 2)
    j2_series = (("j2_err_dirac_nd", raw"$\mathit{Dirac}$"),
                 ("j2_err_first_nd", raw"$\mathit{first\ UTD}$"),
                 ("j2_err_second_nd", raw"$\mathit{second\ UTD}$"),
                 ("j2_err_gauss_nd", raw"$\mathit{real\ Gaussian}$"))
    for (index, (name, legend)) in enumerate(j2_series)
        plot_scatter!(figure, data["kappa_nd"], positive(data[name]);
                      mode="lines+markers", color=COLORS[index],
                      dash=DASHES[index], marker_size=5,
                      legend=legend, linewidth=2, fontsize=FS,
                      xscale="log", yscale="log")
    end
    xlabel!(figure, raw"$\mathit{large\ parameter}\ \kappa$")
    ylabel!(figure, raw"$\mathit{absolute\ }J_2'\ \mathit{action\ error}$")
    selected = [:bottomleft, :bottomleft]
    audit_subplot_legends!(figure, "sommerfeld", selected;
                           width=IEEE_DOUBLE_COL_W,
                           height=IEEE_DOUBLE_COL_H)
    # Both error families descend toward the right, leaving the lower-left
    # corner of each log-log panel the least covered inside position.
    subplot_legends_at!(figure, selected)
    compact_subplot_legends!(figure, 2)
    power_ticks!(figure)
    savefig(figure, joinpath(FIGS, "fig_sommerfeld.pdf");
            width=IEEE_DOUBLE_COL_W, height=IEEE_DOUBLE_COL_H)
end
# figure_evidence_end

function main()
    figure_canonical_multiplier()
    figure_bandlimited()
    figure_halfplane()
    figure_curved()
    figure_sensitivity_profiles()
    figure_sampling()
    figure_translated()
    figure_sommerfeld()
    println("Wrote PlotlySupply figures to $FIGS")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
