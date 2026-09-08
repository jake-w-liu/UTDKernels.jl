# UTDKernels.jl

**Branch-safe and differentiable UTD wedge kernels and finite-edge canonical integrals.**

## Overview

UTDKernels.jl provides a numerically robust Julia implementation of the Kouyoumjian--Pathak (KP) uniform theory of diffraction (UTD) for both **perfectly electrically conducting (PEC)** and **impedance** wedges. The package evaluates the UTD transition function via the scaled complementary error function (`erfcx`) to avoid overflow/underflow cancellation in practical regimes, and includes a regularised cotangent--transition-function product that eliminates the ``\infty \cdot 0`` singularity at shadow boundaries.

Key features:

- **PEC wedge diffraction** via the four-term KP structure with overflow-free `erfcx` and regularised cot-``F`` product
- **Impedance wedge diffraction** via the Holm (2000) heuristic with face-specific Fresnel reflection coefficients and incident-term product weights
- **Maliuzhinets exact solution** for validation: spectral function method with adaptive quadrature
- **Fresnel reflection coefficients** for TE/TM polarisations with complex permittivity support
- **Forward-mode automatic differentiation** via a ForwardDiff.jl package extension (for smooth points)
- **Documented square-root policies**: principal roots for mathematical kernels
  and the passive lower-bank limit for material waves on the negative-real cut
- **Domain-certified, adaptively refined face-grazing continuation** that avoids soft-coefficient
  cancellation, handles both faces and the infinite-distance ``F \to 1`` limit, and preserves the
  physical one-sided source-angle derivative
- **Finite-edge endpoint-uniform evaluation** using the exact propagation
  phase coordinate and three closed Fresnel moments
- **Reflection-boundary face--edge decomposition** that separates the
  cancellation-prone intrinsic incident pair from the finite PEC face
  transition without changing the complete coefficient
- **Passive-sheet complex transition derivatives** through second order, with
  a scale-aware sector contract and cancellation-safe large-argument residuals
- **Correlation-aware bivariate Fresnel transitions** with certified adaptive
  correlation quadrature, optional bounded fixed rules, and robust Hessian maps
- **Curvature-measure seam limits** that expose and remove the local
  near-coplanar wedge bias for supplied positive turning partitions
- **Clustered-pole multipole transitions** that continue Faddeeva divided
  differences across partial and complete pole coalescence
- **Coordinate-invariant Hessian metrics** for signed astigmatic transition
  coordinates and direction-dependent effective distances
- **Null-uniform Faddeeva moments** that retain co-leading amplitude
  derivatives when a regular factor vanishes on the saddle scale
- **Continuous-order saddle--endpoint transitions** for positive algebraic
  endpoint order, with certified adaptive quadrature and a scaled saddle branch

For interior wedges, the four-term pairing follows the
[Hutchins--Kouyoumjian arbitrary-angle nearest-integer construction](https://doi.org/10.21236/AD0699228)
in KP transition-function form.

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
4. **[Passive-Sheet Complex Transitions](@ref passive_transition)** -- Passive-sector mapping, stable residuals and derivatives, principal-root policy, and lossy-wavenumber construction.
5. **[Bivariate Fresnel Transitions](@ref bivariate_transition)** -- Correlation integral, four-region weights, adaptive/fixed evaluation, and normalized Hessian mapping.
6. **[Curvature-Measure Seam Limits](@ref curvature_measure)** -- Intrinsic seam scaling, turning-angle bias, supplied partitions, and circular harmonic reference.
7. **[Clustered-Pole Multipole Transitions](@ref multipole_transition_tutorial)** -- Direct divided differences, centered confluent expansion, basis selection, and scope limits.
8. **[Coordinate-Invariant Hessian Metrics](@ref hessian_metric_tutorial)** -- Dual metric, effective distance, coordinate invariance, and principal-radius projection.
9. **[Null-Uniform Faddeeva Moments](@ref null_uniform_moments_tutorial)** -- Canonical recurrence, certified large-argument routing, analytic-amplitude hierarchy, and shifted-null basis.
10. **[Continuous-Order Saddle--Endpoint Transitions](@ref continuous_order_tutorial)** -- Algebraic endpoint order, certified quadrature, scaled saddle evaluation, and analytic-amplitude moments.
11. **[Finite-Edge Endpoint Transitions](@ref finite_edge)** -- Exact phase coordinates, low-order Fresnel moments, endpoint derivatives, and scope limits.
12. **[Reflection-Boundary Face--Edge Decomposition](@ref reflection_boundary)** -- Exact PEC split, compensated intrinsic residual, branch contract, and diagnostic scope.
13. **[Kouyoumjian--Pathak Diffraction Coefficients](@ref kp)** -- The four-term KP structure: cotangent arguments, boundary-tracking integers, distance parameters, sign factors, and the full ``D_{s/h}`` formula.
14. **[Numerical Methods](@ref numerical)** -- Five numerical challenges: transition-function conditioning, cotangent cancellation, branch cuts, angle seams, and face-grazing loss of significance.
15. **[Automatic Differentiation](@ref ad)** -- Derivation of the `erfcx` derivative rule, the complex chain rule for ForwardDiff, and gradient examples away from non-smooth boundary points.
16. **[Validation](@ref validation)** -- Comparison with the exact Sommerfeld half-plane solution, GTD convergence, reciprocity, shadow-boundary continuity, and broad WDC-reference regression.
17. **[Impedance Wedge Diffraction](@ref impedance)** -- Fresnel reflection coefficients, the Holm (2000) heuristic, material specification, PEC convergence, and ForwardDiff examples.
18. **[Maliuzhinets Exact Solution](@ref maliuzhinets)** -- The Maliuzhinets function, auxiliary product, impedance angles, spectral function approach, and validation against the Holm heuristic.
