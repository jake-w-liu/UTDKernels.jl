using Pkg
Pkg.develop(PackageSpec(path=joinpath(@__DIR__, "..")))
Pkg.instantiate()

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
        ],
        "API Reference" => "api.md",
    ],
    format = Documenter.HTML(
        prettyurls = false,
        mathengine = MathJax3(),
    ),
    checkdocs = :exports,
)
