# PlotlySupply wedge figure from persisted CSVs (paper/data).
# Included by plot_paper.jl, which defines DATA, FIGS, COLORS, DASHES, FS,
# IEEE_DOUBLE_COL_W/H, load_cols, subplots, pos, panel_legends!, power_ticks!,
# and savefig. The import is repeated here so this file's provenance is
# self-contained; `using` is idempotent. Not run standalone.
using PlotlySupply
# evidence_min_rows: 3
# evidence_density_justification: fig_wedge combines two declared sources.
#   wedge_halfplane.csv is the primary 160-angle sweep over 100 deg that
#   resolves the exact diffracted-field null at phi=180 deg and the
#   transition neighborhoods near 170 and 190 deg. wedge_diagnostics.csv
#   is a secondary 3-row table of principal-value diagnostics at exactly
#   the reflection boundary, diffracted-field null, and shadow boundary;
#   it is overlaid as three open markers and is three rows by design.

# figure_evidence_begin
# figure_outputs: fig_wedge_a.pdf,fig_wedge_b.pdf
# evidence_sources: wedge_halfplane.csv
# evidence_aux_sources: wedge_diagnostics.csv
# evidence_min_rows: 160
# evidence_density_justification: 160 observation angles over 100 deg resolve
#   the exact diffracted-field null at phi=180 deg and the geometrical-optics
#   transition neighborhoods near 170 and 190 deg; the three distinguished
#   angles are overlayed from the separate principal-value diagnostics.

# GO boundary marks at the reflection (170 deg) and shadow (190 deg)
# transition angles, drawn as dotted vertical lines across the axes.
function go_boundary_lines!(fig)
    p = plot_obj(fig)
    p.layout[:xaxis][:range] = [140.0, 240.0]
    shapes = get(p.layout, :shapes, Any[])
    new_shapes = Any[]
    for xb in (170.0, 190.0)
        push!(new_shapes, Dict(:type => "line",
            :xref => "x", :yref => "paper",
            :x0 => xb, :x1 => xb,
            :y0 => 0.0, :y1 => 1.0,
            :line => Dict(:color => "#888888", :width => 1,
                          :dash => "dot")))
    end
    p.layout[:shapes] = vcat(collect(shapes), new_shapes)
    return fig
end

function fig_wedge()
    w = load_cols(joinpath(DATA, "wedge_halfplane.csv"))
    dg = load_cols(joinpath(DATA, "wedge_diagnostics.csv"))
    phi = w["phi_deg_nd"]
    mag = sqrt.(w["ud_re_nd"] .^ 2 .+ w["ud_im_nd"] .^ 2)
    fig_a = plot_scatter(phi, pos(mag);
                  mode="lines", color=COLORS[1], dash=DASHES[1],
                  legend=raw"$|\psi^{\mathrm{d}}|\ \mathit{(exact)}$",
                  linewidth=2, yscale="log", fontsize=FS, show=false,
                  xlabel=raw"$\mathit{observation\ angle}\ \phi\ [\mathrm{deg}]$",
                  ylabel=raw"$|\psi^{\mathrm{d}}|$")
    go_boundary_lines!(fig_a)
    # (a) field has a deep null at 180 deg; legend sits bottom-left where
    # the curve stays above ~1e-1 until 160 deg.
    set_legend!(fig_a; position=:bottomleft)
    power_ticks!(fig_a)
    tight_margins!(fig_a)
    savefig(fig_a, joinpath(FIGS, "fig_wedge_a.pdf");
            width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)

    fig_b = nothing
    for (col, color, dash, leg) in (
            ("one_moment_error_nd", COLORS[1], DASHES[1],
             raw"$\mathit{one\text{-}moment}$"),
            ("two_moment_error_nd", COLORS[2], DASHES[2],
             raw"$\mathit{two\text{-}moment}$"),
            ("pole_extraction_error_nd", COLORS[3], DASHES[3],
             raw"$\mathit{pole\ extraction}$"),
            ("slope_separated_error_nd", COLORS[5], DASHES[4],
             raw"$\mathit{separated\ slope}$"))
        if fig_b === nothing
            fig_b = plot_scatter(phi, pos(w[col]);
                      mode="lines", color=color, dash=dash,
                      legend=leg, linewidth=2,
                      xscale="linear", yscale="log", fontsize=FS, show=false,
                      xlabel=raw"$\mathit{observation\ angle}\ \phi\ [\mathrm{deg}]$",
                      ylabel=raw"$\mathit{scale\text{-}normalized\ error}$")
        else
            plot_scatter!(fig_b, phi, pos(w[col]);
                      mode="lines", color=color, dash=dash,
                      legend=leg, linewidth=2)
        end
    end
    # Exact-angle diagnostics (principal-value convention): one- and
    # two-moment errors at 170/180/190 deg as open markers matching the
    # curve colors; they are not sweep samples.
    plot_scatter!(fig_b, dg["phi_deg_nd"], pos(dg["one_moment_error_nd"]);
                  mode="markers", color=COLORS[1],
                  marker_symbol="circle-open", marker_size=7,
                  legend=raw"$\mathit{one\text{-}moment\ diag.}$",
                  fontsize=FS)
    plot_scatter!(fig_b, dg["phi_deg_nd"], pos(dg["two_moment_error_nd"]);
                  mode="markers", color=COLORS[2],
                  marker_symbol="square-open", marker_size=7,
                  legend=raw"$\mathit{two\text{-}moment\ diag.}$")
    go_boundary_lines!(fig_b)
    # (b) mid-left sits in the wide free band between the comparator curves
    # above ~1e-1 and the diagnostic floor below ~1e-9.
    set_legend!(fig_b; position=:left)
    power_ticks!(fig_b)
    tight_margins!(fig_b)
    savefig(fig_b, joinpath(FIGS, "fig_wedge_b.pdf");
            width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
end
# figure_evidence_end
