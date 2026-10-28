# PlotlySupply figures from persisted CSVs (paper/data).
# evidence_min_rows: 120
# evidence_density_justification: accuracy/condition/coplanar sweeps have >=120
#   log-spaced defects so linear slope and 1/eps conditioning resolve; electrical
#   220 points; diagnostic map 120x140; polygon 150 N values; angle factor 360.

include(joinpath(@__DIR__, "..", "..", "common.jl"))
using .PaperReproductionPaths: output_directory, selected_backend

using PlotlySupply

const PAPER = output_directory(162)
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
    rawrows = [split(line, ',') for line in lines[2:end]]
    cols = Dict{String,Vector{Any}}()
    for (i, name) in enumerate(header)
        vals = Any[r[i] for r in rawrows]
        numeric = all(v -> !isempty(v) &&
                          (tryparse(Float64, v) !== nothing), vals)
        cols[String(name)] = numeric ? Float64[parse(Float64, v) for v in vals] :
                                     String[String(v) for v in vals]
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

# MathJax cannot rotate or correctly position colorbar titles, so the title
# rides as a paper-space annotation just above the bar. The bar sits outside
# the axes domain; the right margin is widened so bar, ticks, and title fit.
function set_heatmap_colorbar!(fig, title)
    p = plot_obj(fig)
    cb = Dict(:thickness => 16, :len => 0.80, :y => 0.44, :x => 1.03,
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

# figure_evidence_begin
# figure_outputs: fig_coplanar_scaling.pdf
# evidence_sources: coplanar_scaling.csv
# evidence_min_rows: 540
# evidence_density_justification: 540 samples resolve the compensated coplanar limit and its fitted scaling range.
function fig_coplanar_scaling()
    c = load_cols(joinpath(DATA, "coplanar_scaling.csv"))
    fig = nothing
    first = true
    for (kL, col, dash) in ((0.01, COLORS[1], DASHES[1]), (1.0, COLORS[2], DASHES[2]), (100.0, COLORS[3], DASHES[3]))
        m = c["kL_nd"] .== kL
        lab = kL == 0.01 ? raw"$\eta_e\ (kL=0.01)$" : kL == 1.0 ? raw"$\eta_e\ (kL=1)$" : raw"$\eta_e\ (kL=100)$"
        if first
            fig = plot_scatter(c["eps_nd"][m], pos(c["intrinsic_score_nd"][m]);
                xlabel=raw"$\mathit{coplanar\ defect}\ \epsilon$", ylabel=raw"$\mathit{dimensionless\ score}$",
                mode="lines", color=col, dash=dash, legend=lab, linewidth=2,
                xscale="log", yscale="log", fontsize=FS, show=false)
            first = false
        else
            plot_scatter!(fig, c["eps_nd"][m], pos(c["intrinsic_score_nd"][m]);
                mode="lines", color=col, dash=dash, legend=lab, linewidth=2)
        end
    end
    m100 = c["kL_nd"] .== 100.0
    plot_scatter!(fig, c["eps_nd"][m100], pos(c["raw_score_nd"][m100]);
        mode="lines", color=COLORS[4], dash=DASHES[4], legend=raw"$\mathit{raw}\ (kL=100)$", linewidth=2)
    # Least-obstruction: raw plateau occupies the top; intrinsic traces fill the
    # lower-right, so top-left in the empty band under the raw line is emptiest.
    set_legend!(fig; position=:topleft)
    power_ticks!(fig)
    tight_margins!(fig)
    savefig(fig, joinpath(FIGS, "fig_coplanar_scaling.pdf"); width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
end
# figure_evidence_end

# figure_evidence_begin
# figure_outputs: fig_accuracy_loss.pdf
# evidence_sources: accuracy_curve.csv
# evidence_min_rows: 360
# evidence_density_justification: 360 samples resolve direct-subtraction accuracy loss and the compensated reference across the tested offset range.
function fig_accuracy_loss()
    c = load_cols(joinpath(DATA, "accuracy_curve.csv"))
    fig = nothing
    first = true
    series = (
        (0.2, COLORS[1], DASHES[1], DASHES[2]),
        (0.5, COLORS[2], DASHES[3], DASHES[4]),
        (0.8, COLORS[3], DASHES[1], DASHES[5]),
    )
    for (d, col, dash_n, dash_c) in series
        m = c["delta_over_pi_nd"] .== d
        labn = d == 0.2 ? raw"$\mathit{direct}\ 0.2$" : d == 0.5 ? raw"$\mathit{direct}\ 0.5$" : raw"$\mathit{direct}\ 0.8$"
        labc = d == 0.2 ? raw"$\mathit{comp.}\ 0.2$" : d == 0.5 ? raw"$\mathit{comp.}\ 0.5$" : raw"$\mathit{comp.}\ 0.8$"
        if first
            fig = plot_scatter(c["eps_nd"][m], pos(c["naive_rel_error_nd"][m]);
                xlabel=raw"$\mathit{coplanar\ defect}\ \epsilon$", ylabel=raw"$\mathit{relative\ residual\ error}$",
                mode="lines", color=col, dash=dash_n, legend=labn, linewidth=2,
                xscale="log", yscale="log", fontsize=FS, show=false)
            first = false
        else
            plot_scatter!(fig, c["eps_nd"][m], pos(c["naive_rel_error_nd"][m]);
                mode="lines", color=col, dash=dash_n, legend=labn, linewidth=2)
        end
        plot_scatter!(fig, c["eps_nd"][m], pos(c["compensated_rel_error_nd"][m]);
            mode="lines", color=col, dash=dash_c, legend=labc, linewidth=2)
    end
    # Direct traces occupy the upper-left diagonal; compensated sit on the floor.
    # Top-right (large eps, near-unit error) is the emptiest inside corner; :top
    # sat on the diagonal and was worse.
    set_legend!(fig; position=:topright)
    power_ticks!(fig)
    tight_margins!(fig)
    savefig(fig, joinpath(FIGS, "fig_accuracy_loss.pdf"); width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
end
# figure_evidence_end

# figure_evidence_begin
# figure_outputs: fig_conditioning.pdf
# evidence_sources: condition_sweep.csv
# evidence_min_rows: 150
# evidence_density_justification: 150 logarithmic condition samples resolve the cancellation onset and stabilized plateau.
function fig_conditioning()
    c = load_cols(joinpath(DATA, "condition_sweep.csv"))
    fig = plot_scatter(c["eps_nd"], pos(c["eps_mach_kappa_nd"]);
        xlabel=raw"$\mathit{coplanar\ defect}\ \epsilon$", ylabel=raw"$\mathit{relative\ scale}$",
        mode="lines", color=COLORS[1], dash=DASHES[1], legend=raw"$\epsilon_{\mathrm{mach}}\kappa_S$", linewidth=2,
        xscale="log", yscale="log", fontsize=FS, show=false)
    plot_scatter!(fig, c["eps_nd"], pos(c["naive_rel_error_nd"]);
        mode="lines", color=COLORS[2], dash=DASHES[2], legend=raw"$\mathit{direct\ error}$", linewidth=2)
    plot_scatter!(fig, c["eps_nd"], pos(c["compensated_rel_error_nd"]);
        mode="lines", color=COLORS[3], dash=DASHES[3], legend=raw"$\mathit{compensated\ error}$", linewidth=2)
    # Direct and eps_mach kappa occupy the upper-left diagonal; compensated is
    # on the floor. Top-right (large eps, near-unit error) is empty.
    set_legend!(fig; position=:topright)
    power_ticks!(fig)
    tight_margins!(fig)
    savefig(fig, joinpath(FIGS, "fig_conditioning.pdf"); width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
end
# figure_evidence_end

# figure_evidence_begin
# figure_outputs: fig_asymmetry_sweep.pdf
# evidence_sources: asymmetry_sweep.csv
# evidence_min_rows: 540
# evidence_density_justification: 540 samples resolve the asymmetric face-edge split over the full perturbation sweep.
function fig_asymmetry_sweep()
    c = load_cols(joinpath(DATA, "asymmetry_sweep.csv"))
    fig = nothing
    first = true
    for (e, col, dash, lab) in (
        (1e-2, COLORS[1], DASHES[1], raw"$\mathit{exact}\ 10^{-2}$"),
        (1e-4, COLORS[2], DASHES[2], raw"$\mathit{exact}\ 10^{-4}$"),
        (1e-6, COLORS[3], DASHES[3], raw"$\mathit{exact}\ 10^{-6}$"),
    )
        m = c["eps_nd"] .== e
        if first
            fig = plot_scatter(c["delta_over_pi_nd"][m], pos(c["scaled_exact_sqrtm"][m]);
                xlabel=raw"$\mathit{asymmetry}\ \Delta/\pi$", ylabel=raw"$|D_{\mathrm{e}}|/\epsilon$",
                mode="lines", color=col, dash=dash, legend=lab, linewidth=2,
                yscale="log", fontsize=FS, show=false)
            first = false
        else
            plot_scatter!(fig, c["delta_over_pi_nd"][m], pos(c["scaled_exact_sqrtm"][m]);
                mode="lines", color=col, dash=dash, legend=lab, linewidth=2)
        end
    end
    m6 = c["eps_nd"] .== 1e-6
    plot_scatter!(fig, c["delta_over_pi_nd"][m6], pos(c["scaled_first_order_sqrtm"][m6]);
        mode="lines", color=COLORS[4], dash=DASHES[4], legend=raw"$\mathit{first\text{-}order\ limit}$", linewidth=2)
    # Curves span most of the panel; the upper-right (large Delta, high scaled
    # residual) is the most occupied, so top-left over the flatter left side.
    set_legend!(fig; position=:topleft)
    power_ticks!(fig)
    tight_margins!(fig)
    savefig(fig, joinpath(FIGS, "fig_asymmetry_sweep.pdf"); width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
end
# figure_evidence_end

# figure_evidence_begin
# figure_outputs: fig_angle_factor.pdf
# evidence_sources: angle_factor.csv
# evidence_min_rows: 360
# evidence_density_justification: 360 angular samples resolve the face and edge factors through their limiting orientations.
function fig_angle_factor()
    c = load_cols(joinpath(DATA, "angle_factor.csv"))
    fig = plot_scatter(c["eps_nd"], c["angle_factor_nd"];
        xlabel=raw"$\mathit{coplanar\ defect}\ \epsilon$", ylabel=raw"$\mathit{angular\ factor}\ g(n)$",
        mode="lines", color=COLORS[1], dash=DASHES[1], legend=raw"$g(n)$", linewidth=2,
        fontsize=FS, show=false)
    plot_scatter!(fig, c["eps_nd"], c["first_order_nd"];
        mode="lines", color=COLORS[2], dash=DASHES[2], legend=raw"$\pi\epsilon/2$", linewidth=2)
    # The first-order line rises above g(n); the lower-right under g(n) is empty.
    set_legend!(fig; position=:bottomright)
    tight_margins!(fig)
    savefig(fig, joinpath(FIGS, "fig_angle_factor.pdf"); width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
end
# figure_evidence_end

# figure_evidence_begin
# figure_outputs: fig_electrical_length.pdf
# evidence_sources: electrical_length.csv
# evidence_min_rows: 880
# evidence_density_justification: 880 electrical-length samples resolve the direct and compensated decompositions across the tested regimes.
function fig_electrical_length()
    c = load_cols(joinpath(DATA, "electrical_length.csv"))
    fig = nothing
    first = true
    for (e, col, dash, lab) in (
        (0.005, COLORS[1], DASHES[1], raw"$\epsilon=0.005$"),
        (0.02, COLORS[2], DASHES[2], raw"$\epsilon=0.02$"),
        (0.1, COLORS[3], DASHES[3], raw"$\epsilon=0.1$"),
        (0.3, COLORS[4], DASHES[4], raw"$\epsilon=0.3$"),
    )
        m = c["eps_nd"] .== e
        if first
            fig = plot_scatter(c["kL_nd"][m], pos(c["intrinsic_score_nd"][m]);
                xlabel=raw"$\mathit{electrical\ distance}\ kL$", ylabel=raw"$\mathit{intrinsic\ score}\ \eta_e$",
                mode="lines", color=col, dash=dash, legend=lab, linewidth=2,
                xscale="log", yscale="log", fontsize=FS, show=false)
            first = false
        else
            plot_scatter!(fig, c["kL_nd"][m], pos(c["intrinsic_score_nd"][m]);
                mode="lines", color=col, dash=dash, legend=lab, linewidth=2)
        end
    end
    # Traces rise L->R into plateaus; the upper-left (small kL, high score) is empty.
    set_legend!(fig; position=:topleft)
    power_ticks!(fig)
    tight_margins!(fig)
    savefig(fig, joinpath(FIGS, "fig_electrical_length.pdf"); width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
end
# figure_evidence_end

# figure_evidence_begin
# figure_outputs: fig_polygon_refinement.pdf
# evidence_sources: polygon_refinement.csv
# evidence_min_rows: 149
# evidence_density_justification: 149 refinement samples resolve convergence with polygon order.
function fig_polygon_refinement()
    c = load_cols(joinpath(DATA, "polygon_refinement.csv"))
    N = c["N_count"]
    fig = plot_scatter(N, pos(c["raw_score_nd"]);
        xlabel=raw"$\mathit{regular\text{-}polygon\ side\ count}\ N$", ylabel=raw"$\mathit{score}$",
        mode="lines", color=COLORS[1], dash=DASHES[1], legend=raw"$\mathit{raw\ UTD\ score}$", linewidth=2,
        xscale="log", yscale="log", fontsize=FS, show=false)
    plot_scatter!(fig, N, pos(c["intrinsic_score_nd"]);
        mode="lines", color=COLORS[2], dash=DASHES[2], legend=raw"$\mathit{intrinsic\ UTD\ score}$", linewidth=2)
    plot_scatter!(fig, N, pos(c["geometric_score_nd"]);
        mode="lines", color=COLORS[3], dash=DASHES[3], legend=raw"$\mathit{geometric\ score}$", linewidth=2)
    plot_scatter!(fig, N, pos(c["structural_reference_nd"]);
        mode="lines", color=COLORS[4], dash=DASHES[4], legend=raw"$90\text{-}\mathit{deg\ edge}$", linewidth=2)
    # Raw and 90-degree lines occupy the top; falling traces reach bottom-right,
    # so the empty lower-left corner is least obstructive.
    set_legend!(fig; position=:bottomleft)
    power_ticks!(fig)
    tight_margins!(fig)
    savefig(fig, joinpath(FIGS, "fig_polygon_refinement.pdf"); width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
end
# figure_evidence_end

# figure_evidence_begin
# figure_outputs: fig_polygon_distance_scaling.pdf
# evidence_sources: polygon_refinement.csv
# evidence_min_rows: 149
# evidence_density_justification: 149 distance/refinement samples resolve the polygon-distance scaling law.
function fig_polygon_distance()
    c = load_cols(joinpath(DATA, "polygon_refinement.csv"))
    N = c["N_count"]
    fig = plot_scatter(N, pos(c["intrinsic_score_nd"]);
        xlabel=raw"$\mathit{regular\text{-}polygon\ side\ count}\ N$", ylabel=raw"$\mathit{score}$",
        mode="lines", color=COLORS[1], dash=DASHES[1], legend=raw"$\mathit{intrinsic,\ fixed}\ kL$", linewidth=2,
        xscale="log", yscale="log", fontsize=FS, show=false)
    plot_scatter!(fig, N, pos(c["raw_score_nd"]);
        mode="lines", color=COLORS[2], dash=DASHES[2], legend=raw"$\mathit{raw,\ fixed}\ kL$", linewidth=2)
    plot_scatter!(fig, N, pos(c["physical_intrinsic_score_nd"]);
        mode="lines", color=COLORS[3], dash=DASHES[3], legend=raw"$\mathit{intrinsic},\ L\sim 1/N$", linewidth=2)
    plot_scatter!(fig, N, pos(c["physical_raw_score_nd"]);
        mode="lines", color=COLORS[4], dash=DASHES[4], legend=raw"$\mathit{raw},\ L\sim 1/N$", linewidth=2)
    # The rising raw family occupies the top-right; falling traces leave the
    # lower-left corner (small N, low score) empty.
    set_legend!(fig; position=:bottomleft)
    power_ticks!(fig)
    tight_margins!(fig)
    savefig(fig, joinpath(FIGS, "fig_polygon_distance_scaling.pdf"); width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
end
# figure_evidence_end

# figure_evidence_begin
# figure_outputs: fig_boundary_transition.pdf
# evidence_sources: boundary_sweep.csv
# evidence_min_rows: 800
# evidence_density_justification: 840 hard-polarization path samples resolve
#   the total field through both reflection boundaries and the midpoint gap.
function fig_boundary_transition()
    c = load_cols(joinpath(DATA, "boundary_sweep.csv"))
    m = (c["eps_nd"] .== 1e-3) .& (c["polarization"] .== "hard")
    fig = plot_scatter(c["c_nd"][m], c["u_go_abs_nd"][m];
        xlabel=raw"$\mathit{path\ coordinate}\ c$",
        ylabel=raw"$\mathit{field\ magnitude}$",
        mode="lines", color=COLORS[2], dash=DASHES[2],
        legend=raw"$|u_{\mathrm{GO}}|$", linewidth=2,
        fontsize=FS, show=false)
    plot_scatter!(fig, c["c_nd"][m], c["u_full_abs_nd"][m];
        mode="lines", color=COLORS[1], dash=DASHES[1],
        legend=raw"$|u_{\mathrm{GO+UTD}}|$", linewidth=2)
    # GO steps down inside the gap 0<c<2 while the total stays smooth;
    # the flat traces occupy the top band; bottom-right is empty of curves.
    set_legend!(fig; position=:bottomright)
    tight_margins!(fig)
    savefig(fig, joinpath(FIGS, "fig_boundary_transition.pdf");
        width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
end
# figure_evidence_end

# figure_evidence_begin
# figure_outputs: fig_gap_loss.pdf
# evidence_sources: boundary_sweep.csv
# evidence_min_rows: 800
# evidence_density_justification: 840 hard-polarization samples show the
#   delete error plateau inside the gap versus the preserve error floor.
function fig_gap_loss()
    c = load_cols(joinpath(DATA, "boundary_sweep.csv"))
    m = (c["eps_nd"] .== 1e-3) .& (c["polarization"] .== "hard")
    fig = plot_scatter(c["c_nd"][m], pos(c["delete_err_nd"][m]);
        xlabel=raw"$\mathit{path\ coordinate}\ c$",
        ylabel=raw"$\mathit{absolute\ field\ error}$",
        mode="lines", color=COLORS[2], dash=DASHES[2],
        legend=raw"$\mathit{delete}$", linewidth=2,
        yscale="log", fontsize=FS, show=false)
    plot_scatter!(fig, c["c_nd"][m], pos(c["preserve_err_nd"][m]);
        mode="lines", color=COLORS[1], dash=DASHES[1],
        legend=raw"$\mathit{preserve}$", linewidth=2)
    # Delete plateau occupies the top band inside the gap; preserve sits on
    # the floor. Lower-left outside the gap is empty.
    set_legend!(fig; position=:bottomleft)
    power_ticks!(fig)
    tight_margins!(fig)
    savefig(fig, joinpath(FIGS, "fig_gap_loss.pdf");
        width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
end
# figure_evidence_end

# figure_evidence_begin
# figure_outputs: fig_path_limit.pdf
# evidence_sources: path_limit.csv
# evidence_min_rows: 1300
# evidence_density_justification: 1350 path samples over three defects
#   collapse the reflection-pair magnitude onto the boxcar limit.
function fig_path_limit()
    c = load_cols(joinpath(DATA, "path_limit.csv"))
    fig = nothing
    first = true
    for (e, col, dash) in ((1e-2, COLORS[1], DASHES[1]),
                           (1e-4, COLORS[2], DASHES[2]),
                           (1e-6, COLORS[3], DASHES[3]))
        m = c["eps_nd"] .== e
        lab = e == 1e-2 ? raw"$\epsilon=10^{-2}$" :
              e == 1e-4 ? raw"$\epsilon=10^{-4}$" : raw"$\epsilon=10^{-6}$"
        if first
            fig = plot_scatter(c["c_nd"][m], c["Df_over_sqrtL_abs_nd"][m];
                xlabel=raw"$\mathit{path\ coordinate}\ c$",
                ylabel=raw"$|D_{\mathrm{f}}|/\sqrt{L}$",
                mode="lines", color=col, dash=dash, legend=lab, linewidth=2,
                fontsize=FS, show=false)
            first = false
        else
            plot_scatter!(fig, c["c_nd"][m], c["Df_over_sqrtL_abs_nd"][m];
                mode="lines", color=col, dash=dash, legend=lab, linewidth=2)
        end
    end
    # Boxcar curves overlap on 0<c<2; legend goes in the empty upper-left.
    set_legend!(fig; position=:topleft)
    tight_margins!(fig)
    savefig(fig, joinpath(FIGS, "fig_path_limit.pdf");
        width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
end
# figure_evidence_end

# figure_evidence_begin
# figure_outputs: fig_polygon_field.pdf
# evidence_sources: polygon_field.csv
# evidence_min_rows: 700
# evidence_density_justification: 720 observation samples cover the full
#   circle for the polygon-versus-cylinder comparison.
function fig_polygon_field()
    c = load_cols(joinpath(DATA, "polygon_field.csv"))
    m = c["polarization"] .== "soft"
    fig = plot_scatter(c["phi_over_pi_nd"][m], c["u_cylinder_abs_nd"][m];
        xlabel=raw"$\mathit{observation\ angle}\ \phi/\pi$",
        ylabel=raw"$\mathit{total\ field\ magnitude}$",
        mode="lines", color=COLORS[3], dash=DASHES[3],
        legend=raw"$\mathit{cylinder\ series}$", linewidth=2,
        fontsize=FS, show=false)
    plot_scatter!(fig, c["phi_over_pi_nd"][m], c["u_polygon_abs_nd"][m];
        mode="lines", color=COLORS[1], dash=DASHES[1],
        legend=raw"$\mathit{polygon\ GO+UTD}$", linewidth=2)
    # Place the legend below the curves.
    set_legend!(fig; position=:bottom)
    tight_margins!(fig)
    savefig(fig, joinpath(FIGS, "fig_polygon_field.pdf");
        width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
end
# figure_evidence_end

# figure_evidence_begin
# figure_outputs: fig_selection.pdf
# evidence_sources: selection_study.csv
# evidence_min_rows: 60
# evidence_density_justification: 66 retained-count samples over three rules
#   resolve the matched-tolerance error curves on the 128-gon.
function fig_selection()
    c = load_cols(joinpath(DATA, "selection_study.csv"))
    m = (c["N_count"] .== 128) .& (c["beta_over_pi_nd"] .== 1.0)
    series = [
        ("intrinsic", "preserve_max_err_nd", COLORS[1], DASHES[1],
         raw"$\mathit{intrinsic,\ preserve}$"),
        ("angle", "preserve_max_err_nd", COLORS[2], DASHES[2],
         raw"$\mathit{angle,\ preserve}$"),
        ("raw", "preserve_max_err_nd", COLORS[3], DASHES[3],
         raw"$\mathit{raw,\ preserve}$"),
        ("intrinsic", "delete_max_err_nd", COLORS[4], DASHES[4],
         raw"$\mathit{intrinsic,\ delete}$"),
    ]
    ymin = Inf
    ymax = 0.0
    for (rule, ykey, _, _, _) in series
        yv = c[ykey][m .& (c["rule"] .== rule)]
        posv = yv[yv .> 0]
        !isempty(posv) && (ymin = min(ymin, minimum(posv)))
        ymax = max(ymax, maximum(yv))
    end
    # Exact zeros cannot sit on a log axis: they are flagged with x markers
    # in a dedicated off-scale band below the smallest positive value.
    yfloor = ymin / 100
    fig = nothing
    first = true
    zeroshown = false
    for (rule, ykey, col, dash, lab) in series
        mm = m .& (c["rule"] .== rule)
        x = c["retained_count"][mm]
        y = c[ykey][mm]
        nz = y .> 0
        if first
            fig = plot_scatter(x[nz], y[nz];
                xlabel=raw"$\mathit{retained\ edges}$",
                ylabel=raw"$\mathit{max\ field\ error}$",
                mode="lines+markers", color=col, dash=dash, legend=lab,
                linewidth=2, marker_size=5,
                yscale="log",
                yrange=[log10(yfloor) - 0.35, log10(ymax) + 0.25],
                fontsize=FS, show=false)
            first = false
        else
            plot_scatter!(fig, x[nz], y[nz];
                mode="lines+markers", color=col, dash=dash, legend=lab,
                linewidth=2, marker_size=5)
        end
        if any(.!nz)
            plot_scatter!(fig, x[.!nz], fill(yfloor, count(.!nz));
                mode="markers", color=col, marker_symbol="x",
                marker_size=8,
                legend=zeroshown ? "" :
                       raw"$\mathit{exact\ 0\ (off\ scale)}$",
                showlegend=!zeroshown)
            zeroshown = true
        end
    end
    # Linear count axis keeps the all-pruned m = 0 endpoint visible; every
    # marker is a sampled count, not an interpolation.
    set_legend!(fig; position=:bottomleft)
    power_ticks!(fig)
    tight_margins!(fig)
    savefig(fig, joinpath(FIGS, "fig_selection.pdf");
        width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
end
# figure_evidence_end

# figure_evidence_begin
# figure_outputs: fig_incident_shadow.pdf
# evidence_sources: incident_shadow_sweep.csv
# evidence_min_rows: 1500
# evidence_density_justification: 1200 samples per polarization across the
#   interior incident boundary resolve the razor-thin shadow sliver and the
#   safeguard switch-on band.
function fig_incident_shadow()
    c = load_cols(joinpath(DATA, "incident_shadow_sweep.csv"))
    x = c["delta_minus_pi_over_pi_nd"] .* pi
    series = [
        ("hard", "pres_refl_err_nd", COLORS[4], DASHES[1],
         raw"$\mathit{reflection\mbox{-}only,\ hard}$"),
        ("soft", "pres_refl_err_nd", COLORS[4], DASHES[4],
         raw"$\mathit{reflection\mbox{-}only,\ soft}$"),
        ("hard", "preserve_err_nd", COLORS[1], DASHES[1],
         raw"$\mathit{safeguarded,\ hard}$"),
        ("soft", "preserve_err_nd", COLORS[1], DASHES[4],
         raw"$\mathit{safeguarded,\ soft}$"),
        ("hard", "delete_err_nd", COLORS[5], DASHES[2],
         raw"$\mathit{delete,\ hard}$"),
    ]
    ymin = Inf
    ymax = 0.0
    for (pol, ykey, _, _, _) in series
        yv = c[ykey][c["polarization"] .== pol]
        posv = yv[yv .> 0]
        !isempty(posv) && (ymin = min(ymin, minimum(posv)))
        ymax = max(ymax, maximum(yv))
    end
    yfloor = ymin / 100
    fig = nothing
    first = true
    zeroshown = false
    for (pol, ykey, col, dash, lab) in series
        mm = c["polarization"] .== pol
        y = c[ykey][mm]
        nz = y .> 0
        if first
            fig = plot_scatter(x[mm][nz], y[nz];
                xlabel=raw"$\phi-\phi'-\pi\ \mathit{(rad)}$",
                ylabel=raw"$\mathit{field\ error}$",
                mode="lines", color=col, dash=dash, legend=lab,
                linewidth=2, yscale="log",
                yrange=[log10(yfloor) - 0.35, log10(ymax) + 0.25],
                fontsize=FS, show=false)
            first = false
        else
            plot_scatter!(fig, x[mm][nz], y[nz];
                mode="lines", color=col, dash=dash, legend=lab,
                linewidth=2)
        end
        if any(.!nz)
            plot_scatter!(fig, x[mm][.!nz],
                fill(yfloor, count(.!nz));
                mode="markers", color=col, marker_symbol="x",
                marker_size=6,
                legend=zeroshown ? "" :
                       raw"$\mathit{exact\ 0\ (off\ scale)}$",
                showlegend=!zeroshown)
            zeroshown = true
        end
    end
    set_legend!(fig; position=:bottomleft)
    power_ticks!(fig)
    tight_margins!(fig)
    savefig(fig, joinpath(FIGS, "fig_incident_shadow.pdf");
        width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
end
# figure_evidence_end

function _map_grid(c, zkey)
    es = sort(unique(c["eps_nd"]))
    xs = sort(unique(c["kL_nd"]))
    ie = Dict(v => i for (i, v) in enumerate(es))
    ix = Dict(v => j for (j, v) in enumerate(xs))
    U = fill(NaN, length(xs), length(es))
    for n in eachindex(c["eps_nd"])
        i = ix[c["kL_nd"][n]]
        j = ie[c["eps_nd"][n]]
        U[i, j] = log10(max(c[zkey][n], 1e-18))
    end
    return xs, es, U
end

# figure_evidence_begin
# figure_outputs: fig_diagnostic_score_map.pdf,fig_diagnostic_ratio_map.pdf
# evidence_sources: diagnostic_map.csv
# evidence_min_rows: 16800
# evidence_density_justification: the 16800-point diagnostic grid resolves both score and ratio structure over the full two-parameter domain.
function fig_diagnostic_maps()
    c = load_cols(joinpath(DATA, "diagnostic_map.csv"))
    xs, es, Ueta = _map_grid(c, "intrinsic_score_nd")
    _, _, Uratio = _map_grid(c, "raw_to_intrinsic_nd")
    # Axes are log(kL) and log(eps): different physical quantities, not equal-aspect.
    fs = plot_heatmap(xs, es, Ueta;
        xlabel=raw"$\mathit{electrical\ distance}\ kL$", ylabel=raw"coplanar defect $\epsilon$",
        xscale="log", yscale="log", zrange=[-11.0, 0.0], fontsize=FS, title="")
    power_ticks!(fs)
    set_heatmap_colorbar!(fs, raw"$\log_{10}\,\eta_e$")
    savefig(fs, joinpath(FIGS, "fig_diagnostic_score_map.pdf"); width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)

    fr = plot_heatmap(xs, es, Uratio;
        xlabel=raw"$\mathit{electrical\ distance}\ kL$", ylabel=raw"coplanar defect $\epsilon$",
        xscale="log", yscale="log", zrange=[0.0, 11.0], fontsize=FS, title="")
    power_ticks!(fr)
    set_heatmap_colorbar!(fr, raw"$\log_{10}(\mathit{raw}/\eta_e)$")
    savefig(fr, joinpath(FIGS, "fig_diagnostic_ratio_map.pdf"); width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
end
# figure_evidence_end

if abspath(PROGRAM_FILE) == @__FILE__
    fig_coplanar_scaling()
    fig_accuracy_loss()
    fig_conditioning()
    fig_asymmetry_sweep()
    fig_angle_factor()
    fig_electrical_length()
    fig_polygon_refinement()
    fig_polygon_distance()
    fig_diagnostic_maps()
    fig_boundary_transition()
    fig_gap_loss()
    fig_path_limit()
    fig_polygon_field()
    fig_selection()
    fig_incident_shadow()
    println("figures written to ", FIGS)
end
