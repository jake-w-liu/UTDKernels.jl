# UTDKernels.jl

**Branch-safe and differentiable UTD diffraction coefficients for PEC and impedance wedges.**

## Overview

UTDKernels.jl provides a numerically robust Julia implementation of the Kouyoumjian--Pathak (KP) uniform theory of diffraction (UTD) for both **perfectly electrically conducting (PEC)** and **impedance** wedges. The package evaluates the UTD transition function via the scaled complementary error function (`erfcx`) to avoid overflow/underflow cancellation in practical regimes, and includes a regularised cotangent--transition-function product that eliminates the ``\infty \cdot 0`` singularity at shadow boundaries.

Key features:

- **PEC wedge diffraction** via the four-term KP structure with overflow-free `erfcx` and regularised cot-``F`` product
- **Impedance wedge diffraction** via the Holm (2000) heuristic with face-specific Fresnel reflection coefficients and incident-term product weights
- **Maliuzhinets exact solution** for validation: spectral function method with adaptive quadrature
- **Fresnel reflection coefficients** for TE/TM polarisations with complex permittivity support
- **Forward-mode automatic differentiation** via a ForwardDiff.jl package extension (for smooth points)
- **Documented principal-branch policy** for branch-sensitive square roots
- **Certified face-grazing continuation** that avoids soft-coefficient
  cancellation, handles both faces and the infinite-distance ``F \to 1`` limit, and preserves the
  physical one-sided source-angle derivative

## Installation

```julia
using Pkg
Pkg.add("UTDKernels")
```

For development:

```julia
Pkg.develop(path="path/to/UTDKernels.jl")
```

## Quick start

### PEC wedge

```@example quickstart
using UTDKernels

# Define a half-plane wedge (exterior angle α = 2π)
w = Wedge(2π)

# Set ray angles: observation φ = 90°, incident φ' = 45°
ang = RayAngles(π/2, π/4)

# Wavenumber and effective distance
k = 10.0
L = 1.0

# Compute soft and hard diffraction coefficients with the recommended router
Ds, Dh = wedge_DsDh(w, ang, k, L)
println("Ds = $Ds")
println("Dh = $Dh")
```

### Impedance wedge (Holm heuristic)

```@example quickstart
# 270° wedge with ε_r = 10 dielectric on both faces
mat = WedgeFaceMaterial(10.0 + 0.0im)
iw = ImpedanceWedge(1.5π, mat)

Ds_imp, Dh_imp = impedance_wedge_DsDh(iw, RayAngles(π/2, π/4), 10.0, 1.0)
println("Ds (impedance) = $Ds_imp")
println("Dh (impedance) = $Dh_imp")
```

### Maliuzhinets exact solution (validation reference)

```@example quickstart
# Exact impedance-wedge coefficients via spectral function method
Ds_mal, Dh_mal = maliuzhinets_DsDh(1.5π, 10.0, 10.0, π/2, π/4, 2π)
println("|Ds| exact = $(abs(Ds_mal))")
println("|Dh| exact = $(abs(Dh_mal))")
```

## First-pass workflow (recommended)

For first-time users with basic EM background, this sequence is the fastest
path from geometry to a correct UTD field:

1. Define wedge geometry and angles: `Wedge(alpha)`, `RayAngles(phi, phip)`.
2. Compute coefficients: `Ds, Dh = wedge_DsDh(...)`.
3. Build the physical diffracted field using spreading and phase:
   `D * A(s,sp) * exp(-im*k*s)`.
4. Add the appropriate GO components for your lit/shadow regions.
5. Validate against a canonical case (half-plane / Sommerfeld or WDC sweep)
   before applying to complex scenarios.

Common pitfalls:

- The original `pec_wedge_DsDh` four-term form is useful for formula validation,
  but `wedge_DsDh` is safer for general evaluation because it avoids loss of
  significance near face grazing.
- Exact transition boundaries have explicit package conventions; use one-sided
  samples when a plot must represent a particular physical side.
- Do not mix phasor conventions: this package is `exp(+iωt)` only.
- For broad regression, use `validation/compare_wdc.jl` in addition to unit tests.

## Tutorial outline

This documentation develops the package conventions, equations, numerical
methods, and validation examples needed to use and understand the implementation.

1. **[Maxwell's Equations and the Helmholtz Equation](@ref maxwell)** -- Time-harmonic convention, frequency-domain Maxwell's equations, and the scalar Helmholtz equation for 2D diffraction problems.
2. **[Wedge Geometry and Geometrical Optics](@ref wedge)** -- Canonical PEC wedge, shadow/reflection boundaries, ray geometry, spreading factor, and diffraction dyadic.
3. **[The UTD Transition Function](@ref transition)** -- Full step-by-step derivation of ``F(x)`` from the Fresnel integral to the `erfc` form to the numerically stable `erfcx` form.
4. **[Kouyoumjian--Pathak Diffraction Coefficients](@ref kp)** -- The four-term KP structure: cotangent arguments, boundary-tracking integers, distance parameters, sign factors, and the full ``D_{s/h}`` formula.
5. **[Numerical Methods](@ref numerical)** -- Five numerical challenges: transition-function conditioning, cotangent cancellation, branch cuts, angle seams, and face-grazing loss of significance.
6. **[Automatic Differentiation](@ref ad)** -- Derivation of the `erfcx` derivative rule, the complex chain rule for ForwardDiff, and gradient examples away from non-smooth boundary points.
7. **[Validation](@ref validation)** -- Comparison with the exact Sommerfeld half-plane solution, GTD convergence, reciprocity, shadow-boundary continuity, and broad WDC-reference regression.
8. **[Impedance Wedge Diffraction](@ref impedance)** -- Fresnel reflection coefficients, the Holm (2000) heuristic, material specification, PEC convergence, and ForwardDiff examples.
9. **[Maliuzhinets Exact Solution](@ref maliuzhinets)** -- The Maliuzhinets function, auxiliary product, impedance angles, spectral function approach, and validation against the Holm heuristic.
