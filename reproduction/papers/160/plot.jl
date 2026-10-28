if !isdefined(@__MODULE__, :PaperReproductionPaths)
    include(joinpath(@__DIR__, "..", "..", "common.jl"))
end
using .PaperReproductionPaths: output_directory, selected_backend

# PlotlySupply figures from persisted CSVs (paper/data). Run in an environment
# where the registered PlotlySupply package is available.
# This producer emits several figures. Each figure-evidence block below binds
# its output file to the CSV literals used by that exact code block.

using PlotlySupply

const PAPER = output_directory(160)
const DATA = joinpath(PAPER, "data")
const FIGS = joinpath(PAPER, "figs")
mkpath(FIGS)

const COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#000000", "#E69F00"]
const DASHES = ["solid", "dash", "dashdot", "dot", "longdash", "longdashdot"]
const IEEE_SINGLE_COL_W = 504
const IEEE_SINGLE_COL_H = 360
const IEEE_DOUBLE_COL_W = 1008
const IEEE_DOUBLE_COL_H = 380
const FS = 12

function load_cols(path)
    lines = filter(!isempty, readlines(path))
    lines = filter(l -> !startswith(lstrip(l), '#'), lines)
    header = split(lines[1], ',')
    rows = [parse.(Float64, split(line, ',')) for line in lines[2:end]]
    cols = Dict{String,Vector{Float64}}()
    for (i, name) in enumerate(header)
        cols[String(name)] = [row[i] for row in rows]
    end
    return cols
end

pos(x) = max.(x, 1e-18)
finite_or_nan(x) = map(v -> isfinite(v) ? v : NaN, x)

# Tighten plot margins so a single-panel figure's axes box fills the column,
# matching the plot area of the tightly-packed subplot panels in the two-panel
# figures. Legends are placed inside, so no external legend margin is needed.
tight_margins!(fig) = (plot_obj(fig).layout[:margin] = Dict(:l => 72, :r => 14, :t => 14, :b => 56); fig)

function plot_obj(fig)
    fig isa PlotlySupply.SubplotFigure && return plot_obj(fig.fig)
    hasproperty(fig, :plot) && return fig.plot
    return fig
end

function power_ticks!(fig)
    p = plot_obj(fig)
    for key in keys(p.layout)
        ks = String(key)
        if startswith(ks, "xaxis") || startswith(ks, "yaxis")
            p.layout[key][:exponentformat] = "power"
            p.layout[key][:minexponent] = 3
        end
    end
    return fig
end

const INSIDE_LEGEND_POSITIONS = (
    :topright, :topleft, :bottomright, :bottomleft,
    :top, :bottom, :right, :left,
)

function inside_legend_anchor(xdom, ydom, inset, position::Symbol)
    position in INSIDE_LEGEND_POSITIONS ||
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

function panel_legends!(sf, positions::Vector{Symbol})
    isempty(positions) && throw(ArgumentError("at least one panel legend position is required"))
    inset = (0.02, 0.03)
    bgcolor = "rgba(255,255,255,0.72)"; bordercolor = "rgba(0,0,0,0.15)"; borderwidth = 1.0
    # The public PlotlySupply API assigns traces to per-subplot legends and
    # supplies their style. Plotly's public layout schema then permits each
    # legend to use the separately inspected inside position listed here.
    subplot_legends!(sf; position=first(positions), legend_inset=inset,
        legend_bgcolor=bgcolor, legend_bordercolor=bordercolor, legend_borderwidth=borderwidth)
    p = plot_obj(sf)
    for (i, position) in enumerate(positions)
        xaxis = i == 1 ? :xaxis : Symbol("xaxis$i")
        yaxis = i == 1 ? :yaxis : Symbol("yaxis$i")
        legend_key = i == 1 ? :legend : Symbol("legend$i")
        all(haskey(p.layout, key) for key in (xaxis, yaxis, legend_key)) ||
            throw(ArgumentError("legend position $i has no matching subplot"))
        xdom = Tuple(Float64.(p.layout[xaxis][:domain]))
        ydom = Tuple(Float64.(p.layout[yaxis][:domain]))
        x, y, xanchor, yanchor = inside_legend_anchor(xdom, ydom, inset, position)
        legend = p.layout[legend_key]
        legend[:x] = x
        legend[:y] = y
        legend[:xanchor] = xanchor
        legend[:yanchor] = yanchor
    end
    return sf
end

function panel_labels!(sf, labels::Vector{String})
    p = plot_obj(sf)
    annotations = haskey(p.layout, :annotations) ? collect(p.layout[:annotations]) : Any[]
    for (i, label) in enumerate(labels)
        suffix = i == 1 ? "" : string(i)
        push!(annotations, Dict(
            :text => "\$\\mathbf{$(label)}\$", :showarrow => false,
            :xref => "x$(suffix) domain", :yref => "y$(suffix) domain",
            :x => 0.015, :y => 0.985, :xanchor => "left", :yanchor => "top",
            :font => Dict(:size => FS, :color => "#000000"),
            :bgcolor => "rgba(255,255,255,0.78)", :borderpad => 1,
        ))
    end
    p.layout[:annotations] = annotations
    return sf
end

# MathJax cannot rotate or correctly position colorbar titles, so the title
# rides as a paper-space annotation just above the bar. The bar sits outside
# the axes domain; the right margin is widened so bar, ticks, and title fit.
function set_heatmap_colorbar!(fig, title)
    p = plot_obj(fig)
    cb = Dict(:thickness => 16, :len => 0.88, :y => 0.46, :x => 1.03,
        :outlinewidth => 1, :outlinecolor => "#444444", :ticks => "outside")
    !isempty(p.data) && (p.data[1][:colorbar] = cb)
    haskey(p.layout, :coloraxis) && (p.layout[:coloraxis][:colorbar] = cb)
    p.layout[:margin] = merge(Dict(:l => 72, :t => 14, :b => 56),
        Dict(:r => 92))
    anns = get(p.layout, :annotations, Any[])
    p.layout[:annotations] = vcat(collect(anns),
        [Dict(:x => 1.02, :y => 0.99, :xref => "paper", :yref => "paper",
              :text => title, :showarrow => false, :xanchor => "left",
              :yanchor => "top", :font => Dict(:size => 11))])
    return fig
end

function cdf(vals)
    ev = sort(pos(vals))
    p = collect(range(1, length(ev); length=length(ev))) ./ length(ev)
    return ev, p
end

# figure_evidence_begin
# figure_outputs: fig_coalescence.pdf
# evidence_sources: coalescence.csv
# evidence_min_rows: 76
# evidence_density_justification: 76 log-spaced offsets resolve the coalescing kernels and their difference across 15 decades.
function fig_coalescence()
    c = load_cols(joinpath(DATA, "coalescence.csv"))
    fig = plot_scatter(c["h_rad"], pos(c["abs_G_minus"]);
        xlabel=raw"$\mathit{grazing\ offset}\ h\ [\mathrm{rad}]$", ylabel=raw"$\mathit{kernel\ magnitude}$", mode="lines",
        color=COLORS[1], dash=DASHES[1], legend=raw"$|G(\phi-h)|$", linewidth=2,
        xscale="log", yscale="log", fontsize=FS, show=false)
    plot_scatter!(fig, c["h_rad"], pos(c["abs_G_plus"]); mode="lines", color=COLORS[2], dash=DASHES[2], legend=raw"$|G(\phi+h)|$", linewidth=2)
    plot_scatter!(fig, c["h_rad"], pos(c["abs_difference"]); mode="lines", color=COLORS[3], dash=DASHES[3], legend=raw"$|G(\phi-h)-G(\phi+h)|$", linewidth=2)
    # Least-obstruction: at large h the difference is above the lower-right legend box, while the two kernels remain near the top.
    set_legend!(fig; position=:bottomright)
    power_ticks!(fig)
    tight_margins!(fig)
    savefig(fig, joinpath(FIGS, "fig_coalescence.pdf"); width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
end
# figure_evidence_end

# figure_evidence_begin
# figure_outputs: fig_canonical.pdf
# evidence_sources: canonical_sweep.csv
# evidence_min_rows: 151
# evidence_density_justification: 151 log-spaced offsets resolve the inverse-offset condition slope, provide a multi-level GL4 order fit, and locate the binary64 floor.
function fig_canonical()
    c = load_cols(joinpath(DATA, "canonical_sweep.csv"))
    h = c["h_rad"]
    kappa = finite_or_nan(c["condition_number"])   # condition number diverges at h->0
    hf = h[isfinite.(kappa)]
    anchor = maximum(kappa[isfinite.(kappa)]) * minimum(hf)  # O(1/h) guide anchored to the peak
    sf = subplots(1, 2; show=false, sync=false, title="", per_subplot_legends=true, legend_position=:topright)
    subplot!(sf, 1, 1)
    plot_scatter!(sf, h, kappa; mode="lines", color=COLORS[1], dash=DASHES[1], legend=raw"$\kappa_s(h)$", linewidth=2, xscale="log", yscale="log", fontsize=FS)
    plot_scatter!(sf, h, anchor ./ h; mode="lines", color=COLORS[2], dash=DASHES[2], legend=raw"$O(h^{-1})$", linewidth=2)
    xlabel!(sf, raw"$\mathit{grazing\ offset}\ h\ [\mathrm{rad}]$"); ylabel!(sf, raw"$\mathit{subtraction\ condition\ number}$")
    subplot!(sf, 1, 2)
    series = (("direct_rel_error",raw"$\mathit{direct}$",COLORS[1],DASHES[1]),
              ("utdkernels_rel_error",raw"$\mathit{regularized}$",COLORS[2],DASHES[2]),
              ("linear_rel_error",raw"$\mathit{linear}$",COLORS[3],DASHES[3]),
              ("gl2_rel_error",raw"$\mathrm{GL2}$",COLORS[4],DASHES[4]),
              ("gl4_rel_error",raw"$\mathrm{GL4}$",COLORS[5],DASHES[5]),
              ("gl8_rel_error",raw"$\mathrm{GL8}$",COLORS[6],DASHES[6]))
    first = true
    for (key, lab, col, dash) in series
        yv = pos(c[key])
        if first
            plot_scatter!(sf, h, yv; mode="lines", color=col, dash=dash, legend=lab, linewidth=2, xscale="log", yscale="log", fontsize=FS); first = false
        else
            plot_scatter!(sf, h, yv; mode="lines", color=col, dash=dash, legend=lab, linewidth=2)
        end
    end
    plot_scatter!(sf, h, finite_or_nan(pos(c["unit_roundoff_bound"])); mode="markers",
        color="#888888", marker_symbol="x", marker_size=4, legend=raw"$u\kappa_s$")
    xlabel!(sf, raw"$\mathit{grazing\ offset}\ h\ [\mathrm{rad}]$"); ylabel!(sf, raw"$\mathit{relative\ error}$")
    # Eight-position inspection: top-right minimizes overlap in both panels; in
    # (b) its translucent box avoids the primary descending and floor-level data
    # and intersects only the high-h tails of the order curves.
    panel_legends!(sf, [:topright, :topright])
    panel_labels!(sf, ["(a)", "(b)"])
    power_ticks!(sf)
    savefig(sf, joinpath(FIGS, "fig_canonical.pdf"); width=IEEE_DOUBLE_COL_W, height=IEEE_DOUBLE_COL_H)
end
# figure_evidence_end

# All three regimes: GL8 stays at the floor while direct subtraction collapses.
# figure_evidence_begin
# figure_outputs: fig_regimes.pdf
# evidence_sources: regime_sweep.csv
# evidence_min_rows: 228
# evidence_density_justification: 76 offsets in each of three regimes resolve exterior, interior, and infinite-distance behavior.
function fig_regimes()
    c = load_cols(joinpath(DATA, "regime_sweep.csv"))
    reg = c["regime"]; h = c["h_rad"]
    labels = Dict(1.0 => raw"\mathit{exterior}", 2.0 => raw"\mathit{interior}", 3.0 => raw"L\to\infty")
    direct_dash = Dict(1.0 => DASHES[1], 2.0 => DASHES[3], 3.0 => DASHES[5])
    stable_dash = Dict(1.0 => DASHES[2], 2.0 => DASHES[4], 3.0 => DASHES[6])
    fig = nothing
    first = true
    for r in (1.0, 2.0, 3.0)
        m = reg .== r
        # direct (warm) and GL8 (green) for this regime
        if first
            fig = plot_scatter(h[m], pos(c["direct_rel_error"][m]);
                xlabel=raw"$\mathit{grazing\ offset}\ h\ [\mathrm{rad}]$", ylabel=raw"$\mathit{relative\ error}$", mode="lines",
                color=COLORS[2], dash=direct_dash[r], legend="\$\\mathit{direct},\\ $(labels[r])\$", linewidth=2,
                xscale="log", yscale="log", fontsize=FS, show=false); first = false
        else
            plot_scatter!(fig, h[m], pos(c["direct_rel_error"][m]); mode="lines", color=COLORS[2], dash=direct_dash[r], legend="\$\\mathit{direct},\\ $(labels[r])\$", linewidth=2)
        end
        plot_scatter!(fig, h[m], pos(c["gl8_rel_error"][m]); mode="lines", color=COLORS[3], dash=stable_dash[r], legend="\$$(r == 3.0 ? "\\mathit{closed\\ form}" : "\\mathrm{GL8}"),\\ $(labels[r])\$", linewidth=2)
    end
    # Least-obstruction: the direct curves sweep the upper-left (small h, high
    # error) and everything falls to the lower-right, so the top-right corner
    # (large h, near-unit error) is empty -- the least obstructive inside spot.
    set_legend!(fig; position=:topright)
    power_ticks!(fig)
    tight_margins!(fig)
    # Single-column, same panel size as the subpanels of the two-panel figures.
    savefig(fig, joinpath(FIGS, "fig_regimes.pdf"); width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
end
# figure_evidence_end

# figure_evidence_begin
# figure_outputs: fig_ensemble_electrical.pdf
# evidence_sources: ensemble.csv,ensemble_farfield.csv,electrical_sweep.csv
# evidence_min_rows: 101
# evidence_density_justification: 800 finite cases, 400 infinite-distance cases, and 101 electrical-distance samples resolve the distributions and sweep.
function fig_ensemble_electrical()
    e = load_cols(joinpath(DATA, "ensemble.csv"))
    ff = load_cols(joinpath(DATA, "ensemble_farfield.csv"))
    el = load_cols(joinpath(DATA, "electrical_sweep.csv"))
    sf = subplots(1, 2; show=false, sync=false, title="", per_subplot_legends=true, legend_position=:bottomright)
    subplot!(sf, 1, 1)
    x1, p1 = cdf(e["direct_rel_error"]); plot_scatter!(sf, x1, p1; mode="lines", color=COLORS[1], dash=DASHES[1], legend=raw"$\mathit{direct}$", linewidth=2, xscale="log", fontsize=FS)
    x2, p2 = cdf(e["utdkernels_rel_error"]); plot_scatter!(sf, x2, p2; mode="lines", color=COLORS[2], dash=DASHES[2], legend=raw"$\mathit{regularized}$", linewidth=2)
    x3, p3 = cdf(e["stable_rel_error"]); plot_scatter!(sf, x3, p3; mode="lines", color=COLORS[3], dash=DASHES[3], legend=raw"$\mathrm{GL8}$", linewidth=2)
    x4, p4 = cdf(ff["stable_rel_error"]); plot_scatter!(sf, x4, p4; mode="lines", color=COLORS[6], dash=DASHES[4], legend=raw"$\mathit{closed\ form},\ L\to\infty$", linewidth=2)
    xlabel!(sf, raw"$\mathit{relative\ error}$"); ylabel!(sf, raw"$\mathit{empirical\ cumulative\ fraction}$")
    subplot!(sf, 1, 2)
    plot_scatter!(sf, el["kL"], pos(el["direct_rel_error"]); mode="lines", color=COLORS[1], dash=DASHES[1], legend=raw"$\mathit{direct}$", linewidth=2, xscale="log", yscale="log", fontsize=FS)
    plot_scatter!(sf, el["kL"], pos(el["stable_rel_error"]); mode="lines", color=COLORS[3], dash=DASHES[3], legend=raw"$\mathrm{GL8}$", linewidth=2)
    xlabel!(sf, raw"$\mathit{electrical\ distance}\ kL$"); ylabel!(sf, raw"$\mathit{relative\ error}$")
    # Least-obstruction: the CDF bottom right is below its high-error tail, and the sweep right-center lies between its separated error bands.
    panel_legends!(sf, [:bottomright, :right])
    panel_labels!(sf, ["(a)", "(b)"])
    power_ticks!(sf)
    savefig(sf, joinpath(FIGS, "fig_ensemble_electrical.pdf"); width=IEEE_DOUBLE_COL_W, height=IEEE_DOUBLE_COL_H)
end
# figure_evidence_end

# figure_evidence_begin
# figure_outputs: fig_error_map_direct.pdf,fig_error_map_stable.pdf
# evidence_sources: error_map.csv
# evidence_min_rows: 1089
# evidence_density_justification: the 33 by 33 domain grid resolves the joint grazing-offset and electrical-distance structure.
function fig_error_map()
    c = load_cols(joinpath(DATA, "error_map.csv"))
    hs = sort(unique(c["h_rad"])); kLs = sort(unique(c["kL"]))
    ih = Dict(v => i for (i, v) in enumerate(hs)); ik = Dict(v => j for (j, v) in enumerate(kLs))
    Ud = fill(NaN, length(hs), length(kLs)); Us = fill(NaN, length(hs), length(kLs))
    for n in eachindex(c["h_rad"])
        i = ih[c["h_rad"][n]]; j = ik[c["kL"][n]]
        Ud[i, j] = log10(max(c["direct_rel_error"][n], 1e-18))
        Us[i, j] = log10(max(c["stable_rel_error"][n], 1e-18))
    end
    fd = plot_heatmap(hs, kLs, Ud; xlabel=raw"$\mathit{grazing\ offset}\ h\ [\mathrm{rad}]$", ylabel=raw"$\mathit{electrical\ distance}\ kL$",
        xscale="log", yscale="log", zrange=[-16.0, 1.0], fontsize=FS, title=raw"$\mathit{direct\ difference}$")
    power_ticks!(fd); set_heatmap_colorbar!(fd, raw"$\log_{10}\ \mathit{rel.\ err.}$")
    savefig(fd, joinpath(FIGS, "fig_error_map_direct.pdf"); width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
    fs = plot_heatmap(hs, kLs, Us; xlabel=raw"$\mathit{grazing\ offset}\ h\ [\mathrm{rad}]$", ylabel=raw"$\mathit{electrical\ distance}\ kL$",
        xscale="log", yscale="log", zrange=[-16.0, 1.0], fontsize=FS, title=raw"$\mathrm{GL8}\ \mathit{continuation}$")
    power_ticks!(fs); set_heatmap_colorbar!(fs, raw"$\log_{10}\ \mathit{rel.\ err.}$")
    savefig(fs, joinpath(FIGS, "fig_error_map_stable.pdf"); width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
end
# figure_evidence_end

if abspath(PROGRAM_FILE) == @__FILE__
    fig_coalescence()
    fig_canonical()
    fig_regimes()
    fig_ensemble_electrical()
    fig_error_map()
    println("figures written to ", FIGS)
end
