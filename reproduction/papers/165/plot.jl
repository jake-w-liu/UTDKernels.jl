# PlotlySupply figures from persisted CSVs (paper/data).
# evidence_min_rows: 13
# evidence_density_justification: log-spaced facet counts (13 turning-angle
#   samples from N=12 to 512, and 13 ellipse N values) resolve the 2 delta/pi
#   first-order bias and the O(h)/O(h^2) split; the aliasing map is 29 x 97;
#   the pattern uses 721 observation angles; alias identity uses 600 cases.

include(joinpath(@__DIR__, "..", "..", "common.jl"))
using .PaperReproductionPaths: output_directory, selected_backend

using PlotlySupply

const PAPER = output_directory(165)
const DATA = joinpath(PAPER, "data")
const FIGS = joinpath(PAPER, "figs")
mkpath(FIGS)

const COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#000000", "#E69F00"]
const DASHES = ["solid", "dash", "dashdot", "dot", "longdash", "solid"]
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

function plot_obj(fig)
    fig isa PlotlySupply.SubplotFigure && return plot_obj(fig.fig)
    hasproperty(fig, :plot) && return fig.plot
    return fig
end

tight_margins!(fig) = (plot_obj(fig).layout[:margin] = Dict(:l => 72, :r => 14, :t => 14, :b => 56); fig)

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

function apply_equal_aspect!(fig)
    p = plot_obj(fig)
    p.layout[:yaxis][:scaleanchor] = "x"
    p.layout[:yaxis][:scaleratio] = 1
    p.layout[:xaxis][:constrain] = "domain"
    p.layout[:yaxis][:constrain] = "domain"
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
    isempty(positions) &&
        throw(ArgumentError("at least one panel legend position is required"))
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
        x, y, xanchor, yanchor =
            inside_legend_anchor(xdom, ydom, inset, position)
        legend = p.layout[legend_key]
        legend[:x] = x
        legend[:y] = y
        legend[:xanchor] = xanchor
        legend[:yanchor] = yanchor
    end
    return sf
end

function set_heatmap_colorbar!(fig, title)
    p = plot_obj(fig)
    cb = Dict(
        :thickness => 14,
        :len => 0.88,
        :y => 0.46,
        :x => 1.02,
        :outlinewidth => 1,
        :outlinecolor => "#444444",
        :ticks => "outside",
        :tickfont => Dict(:size => 10),
    )
    for tr in p.data
        t = String(get(tr, :type, ""))
        if t == "heatmap"
            tr[:colorbar] = cb
        end
    end
    haskey(p.layout, :coloraxis) && (p.layout[:coloraxis][:colorbar] = cb)
    anns = get(p.layout, :annotations, Any[])
    p.layout[:annotations] = vcat(collect(anns),
        [Dict(:x => 1.04, :y => 0.93, :xref => "paper", :yref => "paper",
              :text => title, :showarrow => false, :xanchor => "center",
              :yanchor => "bottom", :font => Dict(:size => 11))])
    p.layout[:margin] = Dict(:l => 72, :r => 88, :t => 14, :b => 56)
    return fig
end

function hide_quiver_legends!(fig)
    p = plot_obj(fig)
    for tr in p.data
        t = String(get(tr, :type, ""))
        # PlotlySupply quivers are cone/scatter traces without a user legend.
        if t == "cone" || t == "scatter" && get(tr, :mode, "") == "lines" && get(tr, :name, "") == ""
            tr[:showlegend] = false
        end
        nm = String(get(tr, :name, ""))
        if startswith(nm, "trace")
            tr[:showlegend] = false
        end
    end
    return fig
end

# figure_evidence_begin
# figure_outputs: fig_king_geometry.pdf
# evidence_sources: king_circle.csv,king_polygon.csv
# evidence_min_rows: 14
# evidence_density_justification: 600 smooth-boundary samples and 14 polygon vertices resolve the geometric limit and every turning-angle arrow.
function fig_king_geometry()
    circ = load_cols(joinpath(DATA, "king_circle.csv"))
    poly = load_cols(joinpath(DATA, "king_polygon.csv"))
    px = vcat(poly["x_m"], poly["x_m"][1])
    py = vcat(poly["y_m"], poly["y_m"][1])
    fig = plot_scatter(circ["x_m"], circ["y_m"];
        xlabel=raw"$x\ [\mathrm{m}]$", ylabel=raw"$y\ [\mathrm{m}]$", mode="lines",
        color=COLORS[1], dash=DASHES[1], legend=raw"$\mathit{smooth\ circle}$", linewidth=2,
        fontsize=FS, show=false)
    plot_scatter!(fig, px, py; mode="lines+markers", color=COLORS[2], dash=DASHES[2],
        legend=raw"$\mathit{faceted\ boundary}$", linewidth=2, marker_size=6)
    # Turning-angle arrows at every other vertex.
    ax = Float64[]; ay = Float64[]; au = Float64[]; av = Float64[]
    for i in eachindex(poly["arrow_u_m"])
        if poly["arrow_u_m"][i] != 0 || poly["arrow_v_m"][i] != 0
            push!(ax, poly["x_m"][i]); push!(ay, poly["y_m"][i])
            push!(au, poly["arrow_u_m"][i]); push!(av, poly["arrow_v_m"][i])
        end
    end
    plot_quiver!(fig, ax, ay, au, av; color=COLORS[3])
    hide_quiver_legends!(fig)
    apply_equal_aspect!(fig)
    # Least obstruction: the closed curve fills the disk; the lower-left
    # interior is the emptiest corner around the polygon.
    set_legend!(fig; position=:bottomleft)
    tight_margins!(fig)
    savefig(fig, joinpath(FIGS, "fig_king_geometry.pdf"); width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
end
# figure_evidence_end

# figure_evidence_begin
# figure_outputs: fig_king_pattern.pdf
# evidence_sources: smooth_pattern.csv
# evidence_min_rows: 2163
# evidence_density_justification: 721 observation angles at each of three facet counts resolve the complete periodic pattern and its continuum reference.
function fig_king_pattern()
    c = load_cols(joinpath(DATA, "smooth_pattern.csv"))
    phi_deg(p) = p .* (180 / pi)
    fig = nothing
    first = true
    for (N, col, dash, lab) in (
        (12.0, COLORS[2], DASHES[2], raw"$\mathit{debiased}\ N=12$"),
        (24.0, COLORS[3], DASHES[3], raw"$\mathit{debiased}\ N=24$"),
        (48.0, COLORS[4], DASHES[4], raw"$\mathit{debiased}\ N=48$"),
    )
        m = c["N_count"] .== N
        if first
            fig = plot_scatter(phi_deg(c["phi_rad"][m]), c["debiased_abs_1"][m];
                xlabel=raw"$\mathit{observation\ angle}\ [\mathrm{deg}]$", ylabel=raw"$\mathit{normalized\ magnitude}\ [\mathrm{1}]$",
                mode="lines", color=col, dash=dash, legend=lab, linewidth=2,
                fontsize=FS, show=false)
            first = false
        else
            plot_scatter!(fig, phi_deg(c["phi_rad"][m]), c["debiased_abs_1"][m];
                mode="lines", color=col, dash=dash, legend=lab, linewidth=2)
        end
    end
    m12 = c["N_count"] .== 12.0
    plot_scatter!(fig, phi_deg(c["phi_rad"][m12]), c["exact_abs_1"][m12];
        mode="lines", color=COLORS[1], dash=DASHES[1], legend=raw"$\mathit{curvature\ integral}$", linewidth=2.5)
    m48 = c["N_count"] .== 48.0
    plot_scatter!(fig, phi_deg(c["phi_rad"][m48]), c["raw_abs_1"][m48];
        mode="lines", color=COLORS[5], dash=DASHES[4], legend=raw"$\mathit{raw}\ N=48$", linewidth=1.5)
    # Least obstruction: the exact curve peaks near ±90 deg; the upper-right
    # shoulder after the main lobe is the least crowded inside region.
    set_legend!(fig; position=:topright)
    tight_margins!(fig)
    savefig(fig, joinpath(FIGS, "fig_king_pattern.pdf"); width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
end
# figure_evidence_end

# figure_evidence_begin
# figure_outputs: fig_king_error.pdf
# evidence_sources: harmonic_convergence.csv
# evidence_min_rows: 15
# evidence_density_justification: 15 logarithmically spaced facet counts resolve local-bias and aliasing convergence from the coarse regime to the numerical floor.
function fig_king_error()
    c = load_cols(joinpath(DATA, "harmonic_convergence.csv"))
    N = c["N_count"]
    fig = plot_scatter(N, pos(c["raw_rel_error_1"]);
        xlabel=raw"$\mathit{number\ of\ facets}\ N\ [\mathrm{count}]$", ylabel=raw"$\mathit{relative\ error}\ [\mathrm{1}]$",
        mode="lines+markers", color=COLORS[1], dash=DASHES[1], legend=raw"$\mathit{raw\ seam\ sum}$",
        linewidth=2, marker_size=7, xscale="log", yscale="log", fontsize=FS, show=false)
    plot_scatter!(fig, N, pos(c["local_bias_rel_1"]); mode="lines", color=COLORS[2],
        dash=DASHES[2], legend=raw"$\mathit{local\ wedge\ bias}$", linewidth=2)
    plot_scatter!(fig, N, pos(c["debiased_rel_error_1"]); mode="lines+markers",
        color=COLORS[3], dash=DASHES[3], legend=raw"$\mathit{debiased\ /\ aliasing}$", linewidth=2, marker_size=7)
    refN = [30.0, 90.0]
    idx = findfirst(==(48.0), N)
    anchor = c["raw_rel_error_1"][idx] .* (N[idx] ./ refN)
    plot_scatter!(fig, refN, anchor; mode="lines", color=COLORS[5], dash=DASHES[4],
        legend=raw"$N^{-1}$", linewidth=2)
    # Least obstruction: after N~40 the debiased curve drops to the floor and
    # the upper-right is occupied by the raw N^{-1} trend; bottom-left is empty.
    set_legend!(fig; position=:bottomleft)
    power_ticks!(fig)
    tight_margins!(fig)
    savefig(fig, joinpath(FIGS, "fig_king_error.pdf"); width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
end
# figure_evidence_end

# figure_evidence_begin
# figure_outputs: fig_king_aliasing.pdf
# evidence_sources: aliasing_map.csv
# evidence_min_rows: 2813
# evidence_density_justification: The 29 by 97 facet-count and electrical-radius grid resolves the aliasing boundary across the plotted domain.
function fig_king_aliasing()
    c = load_cols(joinpath(DATA, "aliasing_map.csv"))
    Ns = sort(unique(c["N_count"]))
    zs = sort(unique(c["z_1"]))
    ih = Dict(v => i for (i, v) in enumerate(Ns))
    iz = Dict(v => j for (j, v) in enumerate(zs))
    U = fill(NaN, length(Ns), length(zs))
    for n in eachindex(c["N_count"])
        i = ih[c["N_count"][n]]
        j = iz[c["z_1"][n]]
        U[i, j] = log10(max(c["rel_error_1"][n], 1e-16))
    end
    fig = plot_heatmap(Ns, zs, U;
        xlabel=raw"$\mathit{number\ of\ facets}\ N\ [\mathrm{count}]$", ylabel=raw"$\mathit{electrical\ radius}\ z\ [\mathrm{1}]$",
        fontsize=FS, title="", zrange=[-16.0, 1.0])
    nline = [n for n in Ns if 2.0 <= n - 2 <= 50.0]
    plot_scatter!(fig, nline, nline .- 2; mode="lines", color="#FFFFFF", dash=DASHES[2],
        legend=raw"$z \approx N-2$", linewidth=2)
    p = plot_obj(fig)
    p.layout[:xaxis][:range] = [8.0, 64.0]
    p.layout[:yaxis][:range] = [2.0, 50.0]
    set_heatmap_colorbar!(fig, raw"$\log_{10}\ \mathit{relative\ aliasing\ error}$")
    # Least obstruction: the high-error region is the upper-left (small N, large
    # z); the N-2 guide lives on the diagonal, so the top-left of the resolved
    # band is the least obstructive inside spot for the short legend.
    set_legend!(fig; position=:topleft)
    savefig(fig, joinpath(FIGS, "fig_king_aliasing.pdf"); width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
end
# figure_evidence_end

# figure_evidence_begin
# figure_outputs: fig_local_bias.pdf
# evidence_sources: local_bias.csv
# evidence_min_rows: 13
# evidence_density_justification: 13 logarithmic turning angles resolve the exact local bias and its first-order asymptote.
function fig_local_bias()
    c = load_cols(joinpath(DATA, "local_bias.csv"))
    fig = plot_scatter(c["delta_rad"], pos(c["abs_beta_minus_1_1"]);
        xlabel=raw"$\mathit{turning\ angle}\ \delta\ [\mathrm{rad}]$", ylabel=raw"$|\beta(\delta)-1|\ [\mathrm{1}]$",
        mode="lines+markers", color=COLORS[1], dash=DASHES[1], legend=raw"$\mathit{exact\ local\ bias}$",
        linewidth=2, marker_size=7, xscale="log", yscale="log", fontsize=FS, show=false)
    dd = exp10.(range(-3, -0.6; length=100))
    plot_scatter!(fig, dd, 2 .* dd ./ pi; mode="lines", color=COLORS[2], dash=DASHES[2],
        legend=raw"$2\delta/\pi$", linewidth=2)
    # Least obstruction: the data occupy the upper-right; the upper-left above
    # the 2 delta/pi guide is empty.
    set_legend!(fig; position=:topleft)
    power_ticks!(fig)
    tight_margins!(fig)
    savefig(fig, joinpath(FIGS, "fig_local_bias.pdf"); width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
end
# figure_evidence_end

# A 1x3 PlotlySupply subplot at IEEE double-column width overlays y-axis
# labels across adjacent panels. Export three independent panels.
function fig_resolution()
    # figure_evidence_begin
    # figure_outputs: fig_resolution_count.pdf
    # evidence_sources: resolution_curves.csv
    # evidence_min_rows: 531
    # evidence_density_justification: 531 tolerance, harmonic-order, and electrical-radius cases resolve the minimum-facet count curves.
    r = load_cols(joinpath(DATA, "resolution_curves.csv"))
    fig_a = nothing
    first = true
    for (tol, col, dash, lab) in (
        (1e-3, COLORS[1], DASHES[1], raw"$\mathit{tol.}\ 10^{-3}$"),
        (1e-6, COLORS[2], DASHES[2], raw"$\mathit{tol.}\ 10^{-6}$"),
        (1e-9, COLORS[3], DASHES[3], raw"$\mathit{tol.}\ 10^{-9}$"),
    )
        m = (r["m_1"] .== 3.0) .& (r["tol_1"] .== tol)
        if first
            fig_a = plot_scatter(r["z_1"][m], r["N_min_count"][m]; mode="lines", color=col,
                dash=dash, legend=lab, linewidth=2, fontsize=FS, show=false,
                xlabel=raw"$\mathit{electrical\ radius}\ z\ [\mathrm{1}]$", ylabel=raw"$\mathit{minimum\ facet\ count}\ [\mathrm{count}]$")
            first = false
        else
            plot_scatter!(fig_a, r["z_1"][m], r["N_min_count"][m]; mode="lines", color=col,
                dash=dash, legend=lab, linewidth=2)
        end
    end
    m3 = (r["m_1"] .== 3.0) .& (r["tol_1"] .== 1e-3)
    plot_scatter!(fig_a, r["z_1"][m3], r["z_1"][m3] .+ 3; mode="lines", color=COLORS[5],
        dash=DASHES[4], legend=raw"$N=z+3$", linewidth=2)
    # Rising curves leave the upper-left empty.
    set_legend!(fig_a; position=:topleft)
    tight_margins!(fig_a)
    savefig(fig_a, joinpath(FIGS, "fig_resolution_count.pdf");
            width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
    # figure_evidence_end

    # figure_evidence_begin
    # figure_outputs: fig_resolution_pattern.pdf
    # evidence_sources: resolution_pattern_l2.csv
    # evidence_min_rows: 29
    # evidence_density_justification: 29 facet counts resolve the global pattern-error transition on both sides of the N=z+P onset.
    p = load_cols(joinpath(DATA, "resolution_pattern_l2.csv"))
    fig_b = plot_scatter(p["N_count"], pos(p["relative_L2_error_1"]); mode="lines+markers",
        color=COLORS[1], dash=DASHES[1], legend=raw"$\mathit{smooth\ amplitude}$", linewidth=2,
        marker_size=6, yscale="log", fontsize=FS, show=false,
        xlabel=raw"$\mathit{number\ of\ facets}\ N\ [\mathrm{count}]$", ylabel=raw"$\mathit{relative}\ L_2\ \mathit{pattern\ error}\ [\mathrm{1}]$")
    plot_scatter!(fig_b, [23.0, 23.0], [1e-16, 10.0]; mode="lines", color=COLORS[2],
        dash=DASHES[2], legend=raw"$N=z+P$", linewidth=2)
    # Drop after N=z+P leaves top-right empty.
    set_legend!(fig_b; position=:topright)
    power_ticks!(fig_b)
    tight_margins!(fig_b)
    savefig(fig_b, joinpath(FIGS, "fig_resolution_pattern.pdf");
            width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
    # figure_evidence_end

    # figure_evidence_begin
    # figure_outputs: fig_resolution_partition.pdf
    # evidence_sources: nonuniform_partition.csv
    # evidence_min_rows: 13
    # evidence_density_justification: 13 nonuniform refinements resolve the distinct first- and second-order partition-error slopes.
    nu = load_cols(joinpath(DATA, "nonuniform_partition.csv"))
    fig_c = plot_scatter(nu["h_max_rad"], pos(nu["raw_rel_error_1"]); mode="lines+markers",
        color=COLORS[1], dash=DASHES[1], legend=raw"$\mathit{raw\ local\ coefficient}$", linewidth=2,
        marker_size=6, xscale="log", yscale="log", fontsize=FS, show=false,
        xlabel=raw"$\mathit{maximum\ turning\ cell}\ h\ [\mathrm{rad}]$", ylabel=raw"$\mathit{relative\ error}\ [\mathrm{1}]$")
    plot_scatter!(fig_c, nu["h_max_rad"], pos(nu["debiased_rel_error_1"]); mode="lines+markers",
        color=COLORS[3], dash=DASHES[2], legend=raw"$\mathit{exact\ local\ debiasing}$", linewidth=2, marker_size=6)
    hh = [0.04, 0.35]
    i0 = length(nu["h_max_rad"]) - 3
    h0 = nu["h_max_rad"][i0]
    plot_scatter!(fig_c, hh, nu["raw_rel_error_1"][i0] .* (hh ./ h0); mode="lines",
        color=COLORS[5], dash=DASHES[4], legend=raw"$h$", linewidth=2)
    plot_scatter!(fig_c, hh, nu["debiased_rel_error_1"][i0] .* (hh ./ h0) .^ 2; mode="lines",
        color=COLORS[2], dash=DASHES[3], legend=raw"$h^2$", linewidth=2)
    # Small-h data sit at the lower-left; top-left is empty above the raw O(h) trend.
    set_legend!(fig_c; position=:topleft)
    power_ticks!(fig_c)
    tight_margins!(fig_c)
    savefig(fig_c, joinpath(FIGS, "fig_resolution_partition.pdf");
            width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
    # figure_evidence_end
end

# figure_evidence_begin
# figure_outputs: fig_joint.pdf
# evidence_sources: joint_refinement.csv
# evidence_min_rows: 117
# evidence_density_justification: 39 electrical radii for each of three refinement rules resolve the fixed- and adaptive-facet error laws.
function fig_joint()
    c = load_cols(joinpath(DATA, "joint_refinement.csv"))
    fig = nothing
    first = true
    for (id, col, dash, lab) in (
        (1.0, COLORS[1], DASHES[1], raw"$\mathit{fixed}\ N=48$"),
        (2.0, COLORS[2], DASHES[2], raw"$N=\lceil z+16\rceil$"),
        (3.0, COLORS[3], DASHES[3], raw"$N=\lceil 1.5z+8\rceil$"),
    )
        m = c["rule_id_1"] .== id
        if first
            fig = plot_scatter(c["z_1"][m], pos(c["relative_L2_error_1"][m]);
                xlabel=raw"$\mathit{electrical\ radius}\ z\ [\mathrm{1}]$", ylabel=raw"$\mathit{relative}\ L_2\ \mathit{pattern\ error}\ [\mathrm{1}]$",
                mode="lines+markers", color=col, dash=dash, legend=lab, linewidth=2,
                marker_size=6, yscale="log", fontsize=FS, show=false)
            first = false
        else
            plot_scatter!(fig, c["z_1"][m], pos(c["relative_L2_error_1"][m]);
                mode="lines+markers", color=col, dash=dash, legend=lab, linewidth=2, marker_size=6)
        end
    end
    # Least obstruction: the fixed-N curve occupies the upper-right; the
    # electrically refined curves stay low, leaving the lower-left crowded, so
    # top-left (small z, before the fixed-N blow-up) is least obstructive.
    set_legend!(fig; position=:topleft)
    power_ticks!(fig)
    tight_margins!(fig)
    savefig(fig, joinpath(FIGS, "fig_joint.pdf"); width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
end
# figure_evidence_end

# figure_evidence_begin
# figure_outputs: fig_scaling.pdf
# evidence_sources: absolute_scaling.csv
# evidence_min_rows: 23
# evidence_density_justification: 23 facet counts spanning the asymptotic range resolve fixed-distance saturation and geometry-scaled decay.
function fig_scaling()
    c = load_cols(joinpath(DATA, "absolute_scaling.csv"))
    fig = plot_scatter(c["N_count"], pos(c["fixed_L_total_m05"]);
        xlabel=raw"$\mathit{number\ of\ facets}\ N\ [\mathrm{count}]$", ylabel=raw"$\mathit{absolute\ seam\ strength}\ [\mathrm{m}^{1/2}]$",
        mode="lines+markers", color=COLORS[1], dash=DASHES[1], legend=raw"$\mathit{fixed\ local\ distance}$",
        linewidth=2, marker_size=6, xscale="log", yscale="log", fontsize=FS, show=false)
    plot_scatter!(fig, c["N_count"], pos(c["geometry_scaled_total_m05"]);
        mode="lines+markers", color=COLORS[2], dash=DASHES[2],
        legend=raw"$L_N=R\sin(\pi/N)$", linewidth=2, marker_size=6)
    refN = [2e5, 5e7]
    i0 = length(c["N_count"]) - 6
    anchor = c["geometry_scaled_total_m05"][i0] .* (refN ./ c["N_count"][i0]) .^ (-0.5)
    plot_scatter!(fig, refN, anchor; mode="lines", color=COLORS[5], dash=DASHES[4],
        legend=raw"$N^{-1/2}$", linewidth=2)
    # Least obstruction: the fixed-L curve is nearly flat at the top; the
    # geometry-scaled curve falls through the middle; bottom-left is empty.
    set_legend!(fig; position=:bottomleft)
    power_ticks!(fig)
    tight_margins!(fig)
    savefig(fig, joinpath(FIGS, "fig_scaling.pdf"); width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
end
# figure_evidence_end

# figure_evidence_begin
# figure_outputs: fig_monostatic.pdf
# evidence_sources: monostatic_convergence.csv
# evidence_min_rows: 13
# evidence_density_justification: 13 facet counts resolve the raw/debiased refinement orders against the matched and fixed arcs.
function fig_monostatic()
    conv = load_cols(joinpath(DATA, "monostatic_convergence.csv"))
    sf = subplots(1, 1; show=false, sync=false, title="", per_subplot_legends=true,
        legend_position=:topleft)
    subplot!(sf, 1, 1)
    plot_scatter!(sf, conv["N_count"], pos(conv["raw_matched_rel_error_1"]); mode="lines+markers",
        color=COLORS[1], dash=DASHES[1], legend=raw"$\mathit{raw,\ matched\ arc}$", linewidth=2,
        marker_size=6, xscale="log", yscale="log", fontsize=FS)
    plot_scatter!(sf, conv["N_count"], pos(conv["debiased_matched_rel_error_1"]); mode="lines+markers",
        color=COLORS[3], dash=DASHES[2], legend=raw"$\mathit{debiased,\ matched\ arc}$", linewidth=2, marker_size=6)
    plot_scatter!(sf, conv["N_count"], pos(conv["debiased_fixed_rel_error_1"]); mode="lines+markers",
        color=COLORS[4], dash=DASHES[3], legend=raw"$\mathit{debiased,\ fixed\ arc}$", linewidth=2, marker_size=6)
    plot_scatter!(sf, conv["N_count"], pos(conv["domain_rel_error_1"]); mode="lines",
        color=COLORS[6], dash=DASHES[5], legend=raw"$\mathit{visibility/domain}$", linewidth=2)
    refN = [40.0, 500.0]
    i0 = length(conv["N_count"]) - 3
    n0 = conv["N_count"][i0]
    plot_scatter!(sf, refN, conv["raw_matched_rel_error_1"][i0] .* (refN ./ n0) .^ (-1.0); mode="lines",
        color=COLORS[5], dash=DASHES[4], legend=raw"$N^{-1}$", linewidth=2)
    plot_scatter!(sf, refN, conv["debiased_matched_rel_error_1"][i0] .* (refN ./ n0) .^ (-2.0); mode="lines",
        color=COLORS[2], dash=DASHES[3], legend=raw"$N^{-2}$", linewidth=2)
    xlabel!(sf, raw"$\mathit{number\ of\ facets}\ N\ [\mathrm{count}]$")
    ylabel!(sf, raw"$\mathit{relative\ field\ error}\ [\mathrm{1}]$")

    # Convergence: all four error curves descend toward the lower-right and
    # the fixed-arc/domain curves level off mid-panel at the right edge, so
    # the empty lower-left triangle is the least obstructive legend site.
    panel_legends!(sf, [:bottomleft])
    power_ticks!(sf)
    savefig(sf, joinpath(FIGS, "fig_monostatic.pdf"); width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
end
# figure_evidence_end

# figure_evidence_begin
# figure_outputs: fig_ellipse.pdf
# evidence_sources: ellipse_convergence.csv,ellipse_curve.csv,ellipse_polygon.csv
# evidence_min_rows: 13
# evidence_density_justification: 800 smooth-curve samples, 24 polygon vertices, and 13 refinements resolve both geometry and convergence panels.
function fig_ellipse()
    curve = load_cols(joinpath(DATA, "ellipse_curve.csv"))
    poly = load_cols(joinpath(DATA, "ellipse_polygon.csv"))
    conv = load_cols(joinpath(DATA, "ellipse_convergence.csv"))
    sf = subplots(1, 2; show=false, sync=false, title="", per_subplot_legends=true,
        legend_position=:bottomleft)
    subplot!(sf, 1, 1)
    plot_scatter!(sf, curve["x_m"], curve["y_m"]; mode="lines", color=COLORS[1],
        dash=DASHES[1], legend=raw"$\mathit{smooth\ ellipse}$", linewidth=2, fontsize=FS)
    px = vcat(poly["x_m"], poly["x_m"][1])
    py = vcat(poly["y_m"], poly["y_m"][1])
    plot_scatter!(sf, px, py; mode="lines+markers", color=COLORS[2], dash=DASHES[2],
        legend=raw"$N=24$ polygon", linewidth=2, marker_size=6)
    ax = Float64[]; ay = Float64[]; au = Float64[]; av = Float64[]
    for i in eachindex(poly["arrow_u_m"])
        if poly["arrow_u_m"][i] != 0 || poly["arrow_v_m"][i] != 0
            push!(ax, poly["x_m"][i]); push!(ay, poly["y_m"][i])
            push!(au, poly["arrow_u_m"][i]); push!(av, poly["arrow_v_m"][i])
        end
    end
    plot_quiver!(sf, ax, ay, au, av; color=COLORS[3])
    hide_quiver_legends!(sf)
    xlabel!(sf, raw"$x\ [\mathrm{m}]$"); ylabel!(sf, raw"$y\ [\mathrm{m}]$")
    p = plot_obj(sf)
    # Equal scale on the same-unit geometry panel (xaxis / yaxis of subplot 1).
    p.layout[:yaxis][:scaleanchor] = "x"
    p.layout[:yaxis][:scaleratio] = 1

    subplot!(sf, 1, 2)
    plot_scatter!(sf, conv["h_max_rad"], pos(conv["raw_rel_error_1"]); mode="lines+markers",
        color=COLORS[1], dash=DASHES[1], legend=raw"$\mathit{raw\ intrinsic\ coefficient}$", linewidth=2,
        marker_size=6, xscale="log", yscale="log", fontsize=FS)
    plot_scatter!(sf, conv["h_max_rad"], pos(conv["debiased_rel_error_1"]); mode="lines+markers",
        color=COLORS[3], dash=DASHES[2], legend=raw"$\mathit{exact\ local\ debiasing}$", linewidth=2, marker_size=6)
    hh = [0.025, 0.18]
    i0 = length(conv["h_max_rad"]) - 3
    h0 = conv["h_max_rad"][i0]
    plot_scatter!(sf, hh, conv["raw_rel_error_1"][i0] .* (hh ./ h0); mode="lines",
        color=COLORS[5], dash=DASHES[4], legend=raw"$h$", linewidth=2)
    plot_scatter!(sf, hh, conv["debiased_rel_error_1"][i0] .* (hh ./ h0) .^ 2; mode="lines",
        color=COLORS[2], dash=DASHES[3], legend=raw"$h^2$", linewidth=2)
    xlabel!(sf, raw"$\mathit{maximum\ polygon\ turning\ angle}\ h\ [\mathrm{rad}]$")
    ylabel!(sf, raw"$\mathit{relative\ field\ error}\ [\mathrm{1}]$")
    # Geometry: lower-left interior of the ellipse is emptiest. Convergence:
    # both errors fall to the lower-left, so top-right is least obstructive.
    # Geometry: lower-left interior of the ellipse is emptiest. Convergence:
    # coarse-grid oscillations occupy the upper-right; top-left is empty above
    # the raw O(h) trend.
    panel_legends!(sf, [:bottomleft, :topleft])
    power_ticks!(sf)
    savefig(sf, joinpath(FIGS, "fig_ellipse.pdf"); width=IEEE_DOUBLE_COL_W, height=IEEE_DOUBLE_COL_H)
end
# figure_evidence_end

if abspath(PROGRAM_FILE) == @__FILE__
    fig_king_geometry()
    fig_king_pattern()
    fig_king_error()
    fig_king_aliasing()
    fig_local_bias()
    fig_resolution()
    fig_joint()
    fig_scaling()
    fig_monostatic()
    fig_ellipse()
    println("figures written to ", FIGS)
end
