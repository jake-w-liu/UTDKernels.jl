# UTDKernels.jl

**A branch-safe and differentiable implementation of the Uniform Theory of Diffraction for PEC wedges.**

## Overview

UTDKernels.jl provides a numerically robust Julia implementation of the Kouyoumjian--Pathak (KP) uniform theory of diffraction (UTD) for perfectly electrically conducting (PEC) wedges. The package evaluates the UTD transition function via the scaled complementary error function (`erfcx`) to avoid overflow/underflow cancellation in practical regimes, and includes a regularised cotangent--transition-function product that eliminates the ``\infty \cdot 0`` singularity at shadow boundaries.

Key features:

- **Overflow-free transition function** via the `erfcx` identity
- **Regularised cot--``F`` product** for finite values at shadow boundaries
- **Documented principal-branch policy** for all square roots
- **Grazing-incidence handling** that collapses ``\phi'`` to zero while keeping ``\phi`` in ``[0, \alpha)``
- **Forward-mode automatic differentiation** via a ForwardDiff.jl package extension (for smooth points)

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

```@example quickstart
using UTDKernels

# Define a half-plane wedge (exterior angle α = 2π)
w = Wedge(2π)

# Set ray angles: observation φ = 90°, incident φ' = 45°
ang = RayAngles(π/2, π/4)

# Wavenumber and effective distance
k = 10.0
L = 1.0

# Compute soft and hard diffraction coefficients
Ds, Dh = pec_wedge_DsDh(w, ang, k, L)
println("Ds = $Ds")
println("Dh = $Dh")
```

## Tutorial outline

This documentation is structured as a self-contained tutorial that follows and expands on the companion paper. Every equation is derived in full detail with no intermediate steps omitted.

1. **[Maxwell's Equations and the Helmholtz Equation](@ref maxwell)** -- Time-harmonic convention, frequency-domain Maxwell's equations, and the scalar Helmholtz equation for 2D diffraction problems.
2. **[Wedge Geometry and Geometrical Optics](@ref wedge)** -- Canonical PEC wedge, shadow/reflection boundaries, ray geometry, spreading factor, and diffraction dyadic.
3. **[The UTD Transition Function](@ref transition)** -- Full step-by-step derivation of ``F(x)`` from the Fresnel integral to the `erfc` form to the numerically stable `erfcx` form.
4. **[Kouyoumjian--Pathak Diffraction Coefficients](@ref kp)** -- The four-term KP structure: cotangent arguments, boundary-tracking integers, distance parameters, sign factors, and the full ``D_{s/h}`` formula.
5. **[Numerical Methods](@ref numerical)** -- The four numerical challenges (overflow, singularity, branch cuts, and grazing-angle seam aliasing) and their solutions.
6. **[Automatic Differentiation](@ref ad)** -- Derivation of the `erfcx` derivative rule, the complex chain rule for ForwardDiff, and gradient examples away from non-smooth boundary points.
7. **[Validation](@ref validation)** -- Comparison with the exact Sommerfeld half-plane solution, GTD convergence, reciprocity, and shadow-boundary continuity.
