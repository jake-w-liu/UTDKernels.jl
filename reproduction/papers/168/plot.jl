# PlotlySupply figures from persisted CSVs (paper/data).
# evidence_min_rows: 30
# evidence_density_justification: the null-layer maps are 181x161 (Re z, Lambda)
#   samples on [-3,3]x[-4,4], resolving the O(1) joint-null ridge in Lambda
#   and the O(1) saddle-pole transition in Re z; the Lambda-section has 161
#   samples; convergence uses 30 log-spaced k over 5.5 decades; zero-order
#   scaling uses the same 30 k values; moment profiles use 401 samples;
#   stability uses 150 log-spaced |z| from 10^{0.3} to 10^{12}; the wedge
#   sweep uses 160 observation angles over 100 deg. These densities resolve
#   the order-one error band, k^{-1/2} slopes, even/odd moment structure,
#   the recurrence-cancellation onset, and the wedge null/transition bands.
using PlotlySupply

include(joinpath(@__DIR__, "..", "..", "common.jl"))
using .PaperReproductionPaths
const PAPER = output_directory(168)
const DATA = joinpath(PAPER, "data")
const FIGS = joinpath(PAPER, "figs")
mkpath(FIGS)

const COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#000000", "#E69F00"]
const DASHES = ["solid", "dash", "dashdot", "dot", "longdash", "longdashdot"]
const INSIDE_POSITIONS = [:topright, :topleft, :bottomright, :bottomleft,
                          :top, :bottom, :right, :left]
const IEEE_SINGLE_COL_W = 504
const IEEE_SINGLE_COL_H = 360
const IEEE_DOUBLE_COL_W = 1008
const IEEE_DOUBLE_COL_H = 380
const FS = 12
# Larger font for panels displayed three-across at ~0.31 textwidth:
# a 500-px export shown at ~154 pt keeps the effective size near 6 pt.
const FS_PANEL = 15

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

plot_obj(fig) = fig isa PlotlySupply.SubplotFigure ? plot_obj(fig.fig) :
                hasproperty(fig, :plot) ? fig.plot : fig

function tight_margins!(fig)
    plot_obj(fig).layout[:margin] = Dict(:l => 72, :r => 14, :t => 14, :b => 56)
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

function set_heatmap_colorbar!(fig, title; index=1)
    p = plot_obj(fig)
    cb = Dict(:thickness => 18, :len => 0.88, :y => 0.46, :x => 1.02,
              :outlinewidth => 1, :outlinecolor => "#444444",
              :ticks => "outside", :tickfont => Dict(:size => 10),
              :tickmode => "linear", :tick0 => -5.0, :dtick => 1.0)
    length(p.data) >= index && (p.data[index][:colorbar] = cb)
    haskey(p.layout, :coloraxis) && (p.layout[:coloraxis][:colorbar] = cb)
    anns = get(p.layout, :annotations, Any[])
    p.layout[:annotations] = vcat(collect(anns),
        [Dict(:x => 1.02, :y => 0.93, :xref => "paper", :yref => "paper",
              :text => title, :showarrow => false, :xanchor => "left",
              :yanchor => "bottom", :font => Dict(:size => 11))])
    m = get(p.layout, :margin, Dict{Symbol,Int}())
    rmar = max(get(m, :r, 0), 150)
    p.layout[:margin] = Dict(:l => 72, :r => rmar, :t => 18, :b => 56)
    return fig
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

function panel_legends!(sf, positions::Vector{Symbol})
    isempty(positions) && throw(ArgumentError("at least one panel legend position is required"))
    inset = (0.02, 0.03)
    bgcolor = "rgba(255,255,255,0.72)"
    bordercolor = "rgba(0,0,0,0.15)"
    borderwidth = 1.0
    subplot_legends!(sf; position=first(positions), legend_inset=inset,
                     legend_bgcolor=bgcolor, legend_bordercolor=bordercolor,
                     legend_borderwidth=borderwidth)
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

function grid_from_long(cols, xname, yname, zname)
    xs = sort(unique(cols[xname]))
    ys = sort(unique(cols[yname]))
    ix = Dict(v => i for (i, v) in enumerate(xs))
    iy = Dict(v => j for (j, v) in enumerate(ys))
    U = fill(NaN, length(xs), length(ys))
    for n in eachindex(cols[xname])
        U[ix[cols[xname][n]], iy[cols[yname][n]]] = cols[zname][n]
    end
    return xs, ys, U
end

pos(x) = max.(x, 1e-18)

function log10_error_grid(cols, zname)
    xs, ys, U = grid_from_long(cols, "z_real_nd", "lambda_nd", zname)
    return xs, ys, log10.(max.(U, 1e-8))
end

function error_heatmap(zname, outfile)
    c = load_cols(joinpath(DATA, "null_layer_map.csv"))
    xs, ys, U = log10_error_grid(c, zname)
    fig = plot_heatmap(xs, ys, U;
                       xlabel=raw"$\mathrm{Re}\,z$",
                       ylabel=raw"$\Lambda$",
                       fontsize=FS_PANEL, equalar=false, colorscale="Viridis",
                       xrange=[-3.0, 3.0], yrange=[-4.0, 4.0],
                       zrange=[-5.0, 1.0], show=false)
    set_heatmap_colorbar!(fig, raw"$\log_{10}\ \mathit{error}$")
    savefig(fig, joinpath(FIGS, outfile); width=500, height=405)
end

# figure_evidence_begin
# figure_outputs: fig_king_standard.pdf
# evidence_sources: null_layer_map.csv
# evidence_min_rows: 29141
# evidence_density_justification: the 181 by 161 canonical grid resolves the moving null layer across transition coordinate and null offset.
fig_king_standard() =
    error_heatmap("one_moment_error_nd", "fig_king_standard.pdf")
# figure_evidence_end

# figure_evidence_begin
# figure_outputs: fig_king_corrected.pdf
# evidence_sources: null_layer_map.csv
# evidence_min_rows: 29141
# evidence_density_justification: the same dense canonical grid resolves the corrected two-moment residual without interpolation from another source.
fig_king_corrected() =
    error_heatmap("two_moment_error_nd", "fig_king_corrected.pdf")
# figure_evidence_end

# figure_evidence_begin
# figure_outputs: fig_king_poleext.pdf
# evidence_sources: null_layer_map.csv
# evidence_min_rows: 29141
# evidence_density_justification: the classical pole-extraction baseline is evaluated on the identical 181 by 161 grid at matched normalization.
fig_king_poleext() =
    error_heatmap("pole_extraction_error_nd", "fig_king_poleext.pdf")
# figure_evidence_end

# figure_evidence_begin
# figure_outputs: fig_king_separated.pdf
# evidence_sources: null_layer_map.csv
# evidence_min_rows: 29141
# evidence_density_justification: the separated-slope baseline is evaluated on the identical 181 by 161 grid at matched normalization.
fig_king_separated() =
    error_heatmap("slope_separated_error_nd", "fig_king_separated.pdf")
# figure_evidence_end

# figure_evidence_begin
# figure_outputs: fig_king_section.pdf
# evidence_sources: null_layer_section.csv
# evidence_min_rows: 161
# evidence_density_justification: 161 null-offset samples resolve the fixed-transition slice for all four comparators.
function fig_king_section()
    s = load_cols(joinpath(DATA, "null_layer_section.csv"))
    fig = plot_scatter(s["lambda_nd"], pos(s["one_moment_error_nd"]);
                       xlabel=raw"$\Lambda$",
                       ylabel=raw"$\mathit{scale\text{-}normalized\ error}$",
                       mode="lines", color=COLORS[1], dash=DASHES[1],
                       legend=raw"$\mathit{one\text{-}moment}$", linewidth=2,
                       yscale="log", fontsize=FS_PANEL, show=false)
    plot_scatter!(fig, s["lambda_nd"], pos(s["two_moment_error_nd"]);
                  mode="lines", color=COLORS[2], dash=DASHES[2],
                  legend=raw"$\mathit{two\text{-}moment}$", linewidth=2)
    plot_scatter!(fig, s["lambda_nd"],
                  pos(s["pole_extraction_error_nd"]);
                  mode="lines", color=COLORS[3], dash=DASHES[3],
                  legend=raw"$\mathit{pole\ extraction}$", linewidth=2)
    plot_scatter!(fig, s["lambda_nd"],
                  pos(s["slope_separated_error_nd"]);
                  mode="lines", color=COLORS[5], dash=DASHES[4],
                  legend=raw"$\mathit{separated\ slope}$", linewidth=2)
    set_legend!(fig; position=:left)
    tight_margins!(fig)
    power_ticks!(fig)
    savefig(fig, joinpath(FIGS, "fig_king_section.pdf");
            width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
end
# figure_evidence_end

# figure_evidence_begin
# figure_outputs: fig_convergence.pdf
# evidence_sources: convergence.csv,zero_order_scaling.csv
# evidence_min_rows: 30
# evidence_density_justification: 30 logarithmic large-parameter samples resolve each half-order convergence law and exact-zero field scaling.
function fig_convergence()
    c = load_cols(joinpath(DATA, "convergence.csv"))
    z = load_cols(joinpath(DATA, "zero_order_scaling.csv"))
    sf = subplots(1, 2; show=false, sync=false, title="",
                  per_subplot_legends=true, legend_position=:topright)
    subplot!(sf, 1, 1)
    plot_scatter!(sf, c["k_nd"], pos(c["error_M0_nd"]);
                  mode="lines", color=COLORS[1], dash=DASHES[1],
                  legend=raw"$\mathit{through}\ \mathcal{W}_0$", linewidth=2,
                  xscale="log", yscale="log", fontsize=FS)
    plot_scatter!(sf, c["k_nd"], pos(c["error_M1_nd"]);
                  mode="lines", color=COLORS[2], dash=DASHES[2],
                  legend=raw"$\mathit{through}\ \mathcal{W}_1$", linewidth=2)
    plot_scatter!(sf, c["k_nd"], pos(c["error_M2_nd"]);
                  mode="lines", color=COLORS[3], dash=DASHES[3],
                  legend=raw"$\mathit{through}\ \mathcal{W}_2$", linewidth=2)
    plot_scatter!(sf, c["k_nd"], pos(c["error_M3_nd"]);
                  mode="lines", color=COLORS[4], dash=DASHES[4],
                  legend=raw"$\mathit{through}\ \mathcal{W}_3$", linewidth=2)
    plot_scatter!(sf, c["k_nd"], pos(c["error_pole_extraction_nd"]);
                  mode="lines", color=COLORS[5], dash=DASHES[5],
                  legend=raw"$\mathit{pole\ extraction}$", linewidth=2)
    xlabel!(sf, raw"$\mathit{large\ parameter}\ k$")
    ylabel!(sf, raw"$\mathit{scale\text{-}normalized\ error}$")
    subplot!(sf, 1, 2)
    plot_scatter!(sf, z["k_nd"], pos(z["abs_order1_nd"]);
                  mode="lines", color=COLORS[1], dash=DASHES[1],
                  legend=raw"$\mathit{zero\ order}\ 1$", linewidth=2,
                  xscale="log", yscale="log", fontsize=FS)
    plot_scatter!(sf, z["k_nd"], pos(z["abs_order2_nd"]);
                  mode="lines", color=COLORS[2], dash=DASHES[2],
                  legend=raw"$\mathit{zero\ order}\ 2$", linewidth=2)
    plot_scatter!(sf, z["k_nd"], pos(z["abs_order3_nd"]);
                  mode="lines", color=COLORS[3], dash=DASHES[3],
                  legend=raw"$\mathit{zero\ order}\ 3$", linewidth=2)
    xlabel!(sf, raw"$\mathit{large\ parameter}\ k$")
    ylabel!(sf, raw"$|I|$")
    # (a) curves descend left to right; the lowered floor keeps the
    # bottom-left strip empty below the leftmost W3 start.
    p = plot_obj(sf)
    p.layout[:yaxis][:range] = [-16.0, 0.8]
    # (b) all power laws start high on the left; bottom-left is free.
    panel_legends!(sf, [:bottomleft, :bottomleft])
    power_ticks!(sf)
    savefig(sf, joinpath(FIGS, "fig_convergence.pdf");
            width=IEEE_DOUBLE_COL_W, height=IEEE_DOUBLE_COL_H)
end
# figure_evidence_end

# figure_evidence_begin
# figure_outputs: fig_stability.pdf
# evidence_sources: moment_profile.csv,stability.csv
# evidence_min_rows: 401
# evidence_density_justification: 401 profile samples and 3000 stability cases (150 radii on 5 sheets) resolve moment parity, zeros, and the recurrence-to-asymptotic transition across the Stokes sheet.
function fig_stability()
    p = load_cols(joinpath(DATA, "moment_profile.csv"))
    s = load_cols(joinpath(DATA, "stability.csv"))
    sf = subplots(1, 2; show=false, sync=false, title="",
                  per_subplot_legends=true, legend_position=:topright)
    subplot!(sf, 1, 1)
    plot_scatter!(sf, p["z_real_nd"], pos(p["abs_W0_nd"]);
                  mode="lines", color=COLORS[1], dash=DASHES[1],
                  legend=raw"$|\mathcal{W}_0|$", linewidth=2,
                  yscale="log", yrange=[-1.8, 2.6], fontsize=FS)
    plot_scatter!(sf, p["z_real_nd"], pos(p["abs_W1_nd"]);
                  mode="lines", color=COLORS[2], dash=DASHES[2],
                  legend=raw"$|\mathcal{W}_1|$", linewidth=2)
    plot_scatter!(sf, p["z_real_nd"], pos(p["abs_W2_nd"]);
                  mode="lines", color=COLORS[3], dash=DASHES[3],
                  legend=raw"$|\mathcal{W}_2|$", linewidth=2)
    plot_scatter!(sf, p["z_real_nd"], pos(p["abs_W3_nd"]);
                  mode="lines", color=COLORS[4], dash=DASHES[4],
                  legend=raw"$|\mathcal{W}_3|$", linewidth=2)
    xlabel!(sf, raw"$\mathrm{Re}\,z\ \mathit{at}\ \mathrm{Im}\,z=0.35$")
    ylabel!(sf, raw"$\mathit{magnitude}$")
    subplot!(sf, 1, 2)
    rec_leg = (raw"$\mathit{rec}\ \mathcal{W}_1$",
               raw"$\mathit{rec}\ \mathcal{W}_2$",
               raw"$\mathit{rec}\ \mathcal{W}_3$",
               raw"$\mathit{rec}\ \mathcal{W}_4$")
    st_leg = (raw"$\mathit{st}\ \mathcal{W}_1$",
              raw"$\mathit{st}\ \mathcal{W}_2$",
              raw"$\mathit{st}\ \mathcal{W}_3$",
              raw"$\mathit{st}\ \mathcal{W}_4$")
    # Worst-sheet envelope at each swept Re z (radius parameter): the CSV
    # carries five Im z sheets, so per-(m, radius) maxima keep the curves
    # single-valued while covering the lower-half-plane Stokes sheet.
    for m in 1:4
        rmask = s["m_nd"] .== m
        rs = s["radius_nd"][rmask]
        rec = s["recurrence_error_nd"][rmask]
        sta = s["stable_error_nd"][rmask]
        radii = sort(unique(rs))
        rec_env = [maximum(rec[rs .== r]) for r in radii]
        sta_env = [maximum(sta[rs .== r]) for r in radii]
        plot_scatter!(sf, radii, pos(rec_env);
                      mode="lines", color=COLORS[m], dash=DASHES[2],
                      legend=rec_leg[m], linewidth=1.8,
                      xscale="log", yscale="log", fontsize=FS)
        plot_scatter!(sf, radii, pos(sta_env);
                      mode="lines", color=COLORS[m], dash=DASHES[1],
                      legend=st_leg[m], linewidth=1.8)
    end
    xlabel!(sf, raw"$\mathrm{Re}\,z\ \mathit{(worst\ over\ sheets)}$")
    ylabel!(sf, raw"$\mathit{relative\ error}$")
    # (a) peaks occupy the center; top-left tail is least obstructive.
    # (b) recurrence fills the upper-right; top-left is least obstructive.
    panel_legends!(sf, [:topleft, :topleft])
    power_ticks!(sf)
    savefig(sf, joinpath(FIGS, "fig_stability.pdf");
            width=IEEE_DOUBLE_COL_W, height=IEEE_DOUBLE_COL_H)
end
# figure_evidence_end

# fig_wedge lives in plot_wedge.jl: its declared evidence minimum differs
# from this file's (it mixes the 160-angle sweep with the 3-row exact-angle
# diagnostic table). Include after the shared constants/helpers above.
include(joinpath(@__DIR__, "plot_wedge.jl"))

function main()
    fig_king_standard()
    fig_king_corrected()
    fig_king_poleext()
    fig_king_separated()
    fig_king_section()
    fig_convergence()
    fig_wedge()
    fig_stability()
    println("figures written to ", FIGS)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
