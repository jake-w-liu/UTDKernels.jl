"""
# KP Integer Tracking Diagnostic (Balanis Figure 13-19)

This script visualizes the angular boundaries for the Kouyoumjian-Pathak (KP) 
boundary-tracking integers N_j. 

In UTD, the integer N satisfies the condition:
    2nπN - (φ ± φ') = ±π

The script identifies regions where N equals a target value (`val`) for either 
the "+" or "-" transition terms (`sym`). This is critical for ensuring 
total-field continuity across shadow and reflection boundaries.

## Usage:
    julia examples/fig_13_19.jl [val] [sym]
    
    val: Integer target (default: 0)
    sym: :p for N+ terms, :m for N- terms (default: :m)
"""

using PlotlySupply
using Printf

"""
    plot_kp_regimes(val::Int, sym::Symbol)

Generate a scatter plot showing the regions of (φ ± φ')/π vs n where the 
KP integer N matches `val` for the specified transition term (`sym`).
"""
function plot_kp_regimes(val::Int = 0, sym::Symbol = :m)
    # n-axis: wedge parameter α/π from 1.0 (flat) to 2.0 (half-plane)
    n_range = 1.0:0.02:2.0
    
    fig = plot_scatter([0.0], [1.0], 
        showlegend = false, 
        yrange = [1.0, 2.0], 
        xrange = [-2.5, 4.5],
        title = "KP Integer Tracking: N$((sym == :p ? "^+" : "^-")) = $val",
        xlabel = "(φ ± φ') / π",
        ylabel = "n = α / π")

    for nm in n_range
        # Physical angular range for the given wedge
        phi_range  = LinRange(0, nm * π, 20)
        phip_range = LinRange(0, nm * π, 20)

        # Point buckets for visualization
        # Grey: all valid (φ ± φ') samples
        # Orange: samples where N_i (incident) == val
        # Purple: samples where N_r (reflected) == val
        grey_x   = Float64[]
        orange_x = Float64[]
        purple_x = Float64[]

        for φi in phi_range, φj in phip_range
            ξi = φi - φj  # Incident angular difference
            ξr = φi + φj  # Reflected angular difference

            push!(grey_x, ξi / π, ξr / π)

            if sym == :p
                # Condition: 2nπN - ξ = +π  =>  N = (ξ + π) / (2nπ)
                Ni = round(Int, (ξi + π) / (2 * nm * π))
                Nr = round(Int, (ξr + π) / (2 * nm * π))
            else
                # Condition: 2nπN - ξ = -π  =>  N = (ξ - π) / (2nπ)
                Ni = round(Int, (ξi - π) / (2 * nm * π))
                Nr = round(Int, (ξr - π) / (2 * nm * π))
            end

            Ni == val && push!(orange_x, ξi / π)
            Nr == val && push!(purple_x, ξr / π)
        end

        ny = fill(nm, length(grey_x))
        
        # Background: all tested points
        isempty(grey_x)   || plot_scatter!(fig, grey_x, ny, 
                                mode = "markers", color = "lightgrey", marker_size = 3, showlegend = false)
        
        # Highlighted: target N regions
        isempty(orange_x) || plot_scatter!(fig, orange_x, fill(nm, length(orange_x)), 
                                mode = "markers", marker_symbol = "square", color = "orange", marker_size = 5, showlegend = false)
        
        isempty(purple_x) || plot_scatter!(fig, purple_x, fill(nm, length(purple_x)), 
                                mode = "markers", marker_symbol = "square", color = "purple", marker_size = 5, showlegend = false)
    end

    display(fig)
end

# CLI entry point
if abspath(PROGRAM_FILE) == @__FILE__
    # Parse val (Int)
    target_val = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 0
    
    # Parse sym (Symbol)
    target_sym = length(ARGS) >= 2 ? Symbol(ARGS[2]) : :m
    
    @info "Plotting KP regimes for N = $target_val, sym = $target_sym"
    plot_kp_regimes(target_val, target_sym)
end
