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
# figure_outputs: fig_wedge.pdf
# evidence_sources: wedge_halfplane.csv,wedge_diagnostics.csv
# evidence_min_rows: 160
# evidence_density_justification: 160 observation angles over 100 deg resolve
#   the exact diffracted-field null at phi=180 deg and the geometrical-optics
#   transition neighborhoods near 170 and 190 deg; the three distinguished
#   angles are overlayed from the separate principal-value diagnostics.
function fig_wedge()
    w = load_cols(joinpath(DATA, "wedge_halfplane.csv"))
    dg = load_cols(joinpath(DATA, "wedge_diagnostics.csv"))
    phi = w["phi_deg_nd"]
    mag = sqrt.(w["ud_re_nd"] .^ 2 .+ w["ud_im_nd"] .^ 2)
    sf = subplots(1, 2; show=false, sync=false, title="",
                  per_subplot_legends=true, legend_position=:bottomleft)
    subplot!(sf, 1, 1)
    plot_scatter!(sf, phi, pos(mag);
                  mode="lines", color=COLORS[1], dash=DASHES[1],
                  legend=raw"$|\psi^{\mathrm{d}}|\ \mathit{(exact)}$",
                  linewidth=2, yscale="log", fontsize=FS)
    xlabel!(sf, raw"$\mathit{observation\ angle}\ \phi\ [\mathrm{deg}]$")
    ylabel!(sf, raw"$|\psi^{\mathrm{d}}|$")
    subplot!(sf, 1, 2)
    for (col, color, dash, leg) in (
            ("one_moment_error_nd", COLORS[1], DASHES[1],
             raw"$\mathit{one\text{-}moment}$"),
            ("two_moment_error_nd", COLORS[2], DASHES[2],
             raw"$\mathit{two\text{-}moment}$"),
            ("pole_extraction_error_nd", COLORS[3], DASHES[3],
             raw"$\mathit{pole\ extraction}$"),
            ("slope_separated_error_nd", COLORS[5], DASHES[4],
             raw"$\mathit{separated\ slope}$"))
        plot_scatter!(sf, phi, pos(w[col]);
                      mode="lines", color=color, dash=dash,
                      legend=leg, linewidth=2,
                      xscale="linear", yscale="log", fontsize=FS)
    end
    # Exact-angle diagnostics (principal-value convention): one- and
    # two-moment errors at 170/180/190 deg as open markers matching the
    # curve colors; they are not sweep samples.
    plot_scatter!(sf, dg["phi_deg_nd"], pos(dg["one_moment_error_nd"]);
                  mode="markers", color=COLORS[1],
                  marker_symbol="circle-open", marker_size=7,
                  legend=raw"$\mathit{one\text{-}moment\ diag.}$",
                  fontsize=FS)
    plot_scatter!(sf, dg["phi_deg_nd"], pos(dg["two_moment_error_nd"]);
                  mode="markers", color=COLORS[2],
                  marker_symbol="square-open", marker_size=7,
                  legend=raw"$\mathit{two\text{-}moment\ diag.}$")
    xlabel!(sf, raw"$\mathit{observation\ angle}\ \phi\ [\mathrm{deg}]$")
    ylabel!(sf, raw"$\mathit{scale\text{-}normalized\ error}$")
    # GO boundary marks at the reflection (170 deg) and shadow (190 deg)
    # transition angles, drawn as dotted vertical lines on both panels.
    p = plot_obj(sf)
    p.layout[:xaxis][:range] = [140.0, 240.0]
    p.layout[:xaxis2][:range] = [140.0, 240.0]
    shapes = get(p.layout, :shapes, Any[])
    new_shapes = Any[]
    xdom1 = Tuple(Float64.(p.layout[:xaxis][:domain]))
    xdom2 = Tuple(Float64.(p.layout[:xaxis2][:domain]))
    for dom in (xdom1, xdom2)
        for xb in (170.0, 190.0)
            frac = (xb - 140.0) / 100.0
            push!(new_shapes, Dict(:type => "line",
                :xref => "paper", :yref => "paper",
                :x0 => dom[1] + frac * (dom[2] - dom[1]),
                :x1 => dom[1] + frac * (dom[2] - dom[1]),
                :y0 => 0.0, :y1 => 1.0,
                :line => Dict(:color => "#888888", :width => 1,
                              :dash => "dot")))
        end
    end
    p.layout[:shapes] = vcat(collect(shapes), new_shapes)
    # (a) field has a deep null at 180 deg; legend sits bottom-left where
    # the curve stays above ~1e-1 until 160 deg.
    # (b) the band between the separated-slope floor and the pole-extraction
    # ceiling at the right edge is free.
    panel_legends!(sf, [:bottomleft, :right])
    power_ticks!(sf)
    savefig(sf, joinpath(FIGS, "fig_wedge.pdf");
            width=IEEE_DOUBLE_COL_W, height=IEEE_DOUBLE_COL_H)
end
# figure_evidence_end
