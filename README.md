# UTDKernels.jl

A branch-safe and differentiable implementation of the Uniform Theory of Diffraction (UTD) for perfectly electrically conducting (PEC) wedges.

## Features

- **Branch-safe transition function**: Evaluates F(x) via erfcx to avoid overflow
- **Regularised cot-F product**: Eliminates infinity-times-zero singularity at shadow boundaries
- **Automatic differentiation**: ForwardDiff.jl extension for end-to-end gradients
- **Principal-branch consistency**: All square roots use a documented single branch

## Installation

```julia
using Pkg
Pkg.add("UTDKernels")
```

Or for development:
```julia
Pkg.develop(path="path/to/UTDKernels.jl")
```

## Quick Start

```julia
using UTDKernels

# Define a half-plane wedge (alpha = 2pi)
w = Wedge(2pi)

# Set ray angles: observation phi, incident phi'
ang = RayAngles(pi/2, pi/4)

# Wavenumber and effective distance
k = 10.0
L = 1.0

# Compute soft and hard diffraction coefficients
Ds, Dh = pec_wedge_DsDh(w, ang, k, L)
```

## API Reference

### Types

- `Wedge(alpha)`: Wedge with exterior angle alpha in (0, 2pi]
- `RayAngles(phi, phip)`: Observation and incident azimuths
- `Distances(s, sp)`: Edge-to-observer and source-to-edge distances
- `PhasorConvention`: Time-harmonic convention (exp(+iwt) supported)

### Functions

- `F_utd(x)`: UTD transition function
- `pec_wedge_DsDh(w, ang, k, L)`: Soft/hard diffraction coefficients
- `pec_wedge_apply_sh(Ds, Dh, E_soft, E_hard)`: Apply dyadic to field
- `spreading_factor(s, sp)`: UTD spreading factor A(s, s')
- `effective_L(d::Distances)`: Compute L = s*s'/(s+s')

### Utilities

- `wrap_angle(phi, alpha)`: Normalise angle to [0, alpha)
- `wedge_n(w)`, `wedge_nu(w)`: Wedge parameters n = alpha/pi, nu = pi/alpha
- `inspect_kp_terms(w, ang)`: Diagnostic for KP four-term structure

## Automatic Differentiation

When `ForwardDiff` is loaded, the package extension enables differentiation:

```julia
using UTDKernels, ForwardDiff

w = Wedge(2pi)
f(phi) = abs(pec_wedge_DsDh(w, RayAngles(phi, pi/4), 10.0, 1.0)[1])

# Compute gradient
grad = ForwardDiff.derivative(f, pi/2)
```

## Directory Structure

```
UTDKernels.jl/
├── src/
│   ├── UTDKernels.jl          # Main module
│   ├── common/
│   │   ├── Types.jl           # Core types
│   │   ├── AngleWrap.jl       # Angle normalisation
│   │   ├── Branches.jl        # Branch-safe sqrt
│   │   └── Numerics.jl        # Numerical utilities
│   ├── transition/
│   │   └── TransitionF.jl     # F(x) via erfcx
│   ├── wedge/
│   │   ├── WedgeGeometry.jl   # KP four-term structure
│   │   ├── WedgePEC.jl        # PEC coefficients
│   │   ├── WedgeDyadic.jl     # Dyadic application
│   │   └── Regimes.jl         # Regime classification
│   └── utils/
│       └── Diagnostics.jl     # inspect_kp_terms
├── ext/
│   └── UTDKernelsForwardDiffExt.jl  # ForwardDiff extension
├── test/
│   ├── runtests.jl
│   ├── test_transition.jl
│   ├── test_wedge_pec_continuity.jl
│   ├── test_wedge_pec_limits.jl
│   ├── test_symmetry.jl
│   └── test_ad.jl
└── Project.toml
```

## Running Tests

```julia
using Pkg
Pkg.test("UTDKernels")
```

## Reproducing Results

```bash
# Requirements: Julia 1.9+ with SpecialFunctions.jl
cd /path/to/UTDKernels.jl/..

# Generate all CSV data files
julia generate_paper_data.jl

# Generate all PDF figures
julia plot_paper.jl
```

**Julia version**: All development and testing performed on Julia 1.12.
**Required packages**: SpecialFunctions.jl (v2+), and optionally ForwardDiff.jl for AD tests.

## License

MIT License. See LICENSE file for details.

<!-- ## Citation

If you use this package in your research, please cite:

```bibtex
@article{liu2024utdkernels,
  title={UTDKernels.jl: A Branch-Safe and Differentiable Implementation of
         Uniform Theory of Diffraction for Electromagnetic Wedges},
  author={Liu, Jake W.},
  journal={Computer Physics Communications},
  year={2024}
}
```

## References

1. R. G. Kouyoumjian and P. H. Pathak, "A uniform geometrical theory of diffraction for an edge in a perfectly conducting surface," Proc. IEEE, vol. 62, no. 11, pp. 1448-1461, Nov. 1974.

2. J. B. Keller, "Geometrical theory of diffraction," J. Opt. Soc. Am., vol. 52, no. 2, pp. 116-130, 1962. -->
