using Pkg
cd(@__DIR__) do
    Pkg.develop(PackageSpec(path=".."))
    Pkg.instantiate()
end

using Documenter
using UTDKernels

makedocs(
    sitename = "UTDKernels.jl",
    modules = [UTDKernels],
    pages = [
        "Home" => "index.md",
        "Tutorial" => [
            "tutorial/maxwell.md",
            "tutorial/wedge.md",
            "tutorial/transition.md",
            "tutorial/kp_coefficients.md",
            "tutorial/numerical.md",
            "tutorial/ad.md",
            "tutorial/validation.md",
            "tutorial/impedance.md",
            "tutorial/maliuzhinets.md",
        ],
        "API Reference" => "api.md",
    ],
    format = Documenter.HTML(
        prettyurls = false,
        mathengine = MathJax3(),
        example_size_threshold = nothing,
        size_threshold_warn = 512 * 2^10,
        size_threshold = 1024 * 2^10,
    ),
    checkdocs = :exports,
    doctest = true,
    warnonly = false,
)
