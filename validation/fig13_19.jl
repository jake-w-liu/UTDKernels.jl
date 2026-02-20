using PlotlySupply

function main()
    fig = plot_scatter([0], [0], showlegend = false, yrange = [1, 2])

    n = 1:0.01:2
    val = 0
    sym = :m 

    for nm in n
        phi  = LinRange(0, nm * π, 25)
        phip = LinRange(0, nm * π, 25)

        # Pre-allocate point buckets
        grey_x   = Float64[]
        orange_x = Float64[]
        purple_x = Float64[]

        for φi in phi, φj in phip
            ξi = φi - φj
            ξr = φi + φj

            push!(grey_x, ξi / π, ξr / π)

            if sym == :p
                Npi = round(Int, (ξi + π) / (2nm * π))
                Npr = round(Int, (ξr + π) / (2nm * π))

                Npi == val && push!(orange_x, ξi / π)
                Npr == val && push!(purple_x, ξr / π)
            elseif sym == :m
                Nmi = round(Int, (ξi - π) / (2nm * π))
                Nmr = round(Int, (ξr - π) / (2nm * π))

                Nmi == val && push!(orange_x, ξi / π)
                Nmr == val && push!(purple_x, ξr / π)
            end
        end

        ny = fill(nm, length(grey_x))
        isempty(grey_x)   || plot_scatter!(fig, grey_x,   ny,                         mode = "markers", color = "grey",   showlegend = false)
        isempty(orange_x) || plot_scatter!(fig, orange_x, fill(nm, length(orange_x)), mode = "markers", marker_symbol = "square", color = "orange", showlegend = false)
        isempty(purple_x) || plot_scatter!(fig, purple_x, fill(nm, length(purple_x)), mode = "markers", marker_symbol = "square", color = "purple", showlegend = false)
    end

    display(fig)
end

main()