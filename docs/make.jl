using Pkg
cd(@__DIR__) do
    Pkg.develop(PackageSpec(path=".."))
    Pkg.instantiate()
end

using Documenter
using UTDKernels

const PACKAGE_ROOT = normpath(joinpath(@__DIR__, ".."))
const PACKAGE_REMOTE = Documenter.Remotes.GitHub("jake-w-liu", "UTDKernels.jl")
const PACKAGE_REVISION = "main"
const PACKAGE_IS_GIT_CHECKOUT = ispath(joinpath(PACKAGE_ROOT, ".git"))

makedocs(
    sitename = "UTDKernels.jl",
    modules = [UTDKernels],
    remotes = Dict(".." => (PACKAGE_REMOTE, PACKAGE_REVISION)),
    pages = [
        "Home" => "index.md",
        "Tutorial" => [
            "tutorial/maxwell.md",
            "tutorial/wedge.md",
            "tutorial/transition.md",
            "tutorial/finite_edge.md",
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
        disable_git = !PACKAGE_IS_GIT_CHECKOUT,
        edit_link = PACKAGE_IS_GIT_CHECKOUT ? PACKAGE_REVISION : nothing,
        repolink = "https://github.com/jake-w-liu/UTDKernels.jl",
        mathengine = MathJax3(),
        example_size_threshold = nothing,
        size_threshold_warn = 512 * 2^10,
        size_threshold = 1024 * 2^10,
    ),
    checkdocs = :exports,
    doctest = true,
    warnonly = false,
)
