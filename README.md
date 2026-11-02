# UTDKernels.jl

Julia kernels for uniform geometrical theory of diffraction, with explicit
branch conventions, stable boundary limits and ForwardDiff support.

Requires Julia 1.12 or newer. The CI matrix targets Julia 1.12 and the latest
stable Julia on Linux, macOS and Windows.

## Install

```julia
using Pkg
Pkg.add("UTDKernels")
```

For local development:

```julia
Pkg.develop(url="https://github.com/jake-w-liu/UTDKernels.jl")
```

## Quick start

```julia
using UTDKernels

wedge = Wedge(2pi)
angles = RayAngles(pi/2, pi/4)
k, L = 10.0, 2.0
Ds, Dh = wedge_DsDh(wedge, angles, k, L)

# The same router handles a small incident face offset.
Ds, Dh = wedge_DsDh(wedge, RayAngles(pi/2, 1e-14), k, L)

# Apply coefficients to incident soft/hard components.
Es, Eh = pec_wedge_apply_sh(Ds, Dh, 1.0, 0.0, k, L, Inf)
```

Fields use the `exp(+iωt)` convention: outgoing waves have phase `exp(-iks)`.
For plane-wave incidence the effective distance is the finite observation
range; `L = Inf` selects the local GTD coefficient limit.

## Features

- PEC and impedance-wedge coefficients, with grazing continuation and
  face–edge decomposition on the midpoint slice.
- Finite-edge, passive complex, correlated bivariate, clustered-pole and
  continuous-order transition functions.
- Curvature-measure seam sums, cylindrical monostatic assembly and harmonic
  reference models.
- Null-uniform Faddeeva moments, amplitude jets and half-plane fields.
- Shadow-boundary sensitivities, nonlinear pullbacks and receiver actions.
- Independent high-precision, quadrature and eigenfunction references.

The feature modules are `FaceEdge`, `CurvatureMeasure` (including
`MonostaticAssembly`), `NullUniform` and `ShadowSensitivity`. They share the
parent kernels; the production package does not depend on DiffUTD or plotting.

See the [API reference](docs/src/api.md),
[model guide](docs/src/study_models.md) and
[ForwardDiff tutorial](docs/src/tutorial/ad.md) for signatures, supported
numeric types, domains and branch-local differentiation contracts.

## Reproduction, examples and tests

The optional [reproduction environment](reproduction/README.md) generates study
data, figures and manuscript macros with either UTDKernels or DiffUTD. It requires
adjacent checkouts and access to DiffUTD. The [textbook examples](examples/README.md)
provide smaller formula comparisons.

Run the package tests from a checkout:

```sh
julia --startup-file=no --project=. -e 'using Pkg; Pkg.test()'
```

## Citation and license

Jake W. Liu, “UTDKernels.jl: A branch-safe and differentiable Julia library for
Kouyoumjian–Pathak wedge diffraction kernels,” *SoftwareX*, 35, 102815 (2026).
[DOI: 10.1016/j.softx.2026.102815](https://doi.org/10.1016/j.softx.2026.102815).

[MIT license](LICENSE.md).
