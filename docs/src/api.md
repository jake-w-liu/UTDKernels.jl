# API Reference

## Types

```@docs
EXP_IWT
PhasorConvention
Wedge
RayAngles
Distances
```

## Wedge parameters

```@docs
wedge_n
wedge_nu
effective_L
```

## Angle utilities

```@docs
wrap_angle
```

## Transition function

```@docs
F_utd
```

## Diffraction coefficients

```@docs
pec_wedge_DsDh
pec_wedge_apply_sh
spreading_factor
```

`pec_wedge_DsDh` provides both:
- a single-`L` API for standard KP usage, and
- a three-distance (`Li`, `Lro`, `Lrn`) API for separated incident/reflection transition distances.

### Practical constraints and edge cases

- `Wedge(alpha)` requires `0 < alpha <= 2π`.
- `pec_wedge_DsDh(wedge, ang, k, Li, Lro, Lrn)` requires positive finite
  distances or `Inf`; zero, negative, and `NaN` distances are rejected.
- The single-`L` API supports finite positive `L` and `L = Inf` (far-field limit).
- For advanced analytic-continuation workflows (e.g., complex-source beams), the single-`L` API also accepts finite nonzero complex `L`.
- At exact transition-boundary samples:
  - finite `L` uses a one-sided surrogate evaluation,
  - `L = Inf` uses the far-field midpoint-safe handling for singular terms.
- All angles are internally wrapped to `[0, alpha)`. Near grazing incidence,
  the effective source angle has zero value at the grazed face while retaining
  its signed local offset, so the seam is value-continuous and ForwardDiff
  returns the physical interior-side derivative.
- `WedgeFaceMaterial` requires finite relative permittivity and conductivity;
  conductivity must be nonnegative, and frequency-dependent material evaluation
  requires a finite positive frequency.
- Matched media (`ε_r = 1`, zero conductivity) return exactly zero TE and TM
  Fresnel reflection, including at grazing incidence.

See also:
- [Wedge Geometry and Geometrical Optics](@ref wedge)
- [Numerical Methods](@ref numerical)

## Regime detection

```@docs
wedge_transition_args
```

## Diagnostics

```@docs
inspect_kp_terms
```

## Fresnel reflection coefficients

```@docs
WedgeFaceMaterial
fresnel_te
fresnel_tm
```

`WedgeFaceMaterial` can be constructed in two ways:
- From a complex permittivity: `WedgeFaceMaterial(eps_r)` where `eps_r` may be complex.
- From real permittivity + conductivity + frequency: `WedgeFaceMaterial(eps_r_real, sigma, freq)`.
  The effective permittivity is ``\varepsilon_{r,\text{eff}} = \varepsilon_r - i\sigma/(\omega\varepsilon_0)`` under the ``\exp(+i\omega t)`` convention.

The Fresnel coefficients use **grazing angle** ``\psi`` measured from the surface (not the normal).
The relation to the conventional incidence angle is ``\psi = \pi/2 - \theta_i``.

## Impedance wedge diffraction (Holm heuristic)

```@docs
ImpedanceWedge
impedance_wedge_DsDh
```

`impedance_wedge_DsDh` provides both:
- a single-`L` API for standard usage, and
- a three-distance (`Li`, `Lro`, `Lrn`) API for separated incident/reflection transition distances.

The Holm (2000) heuristic modifies the PEC four-term KP structure by replacing
the fixed sign factors on terms 3 and 4 with face-specific Fresnel reflection
coefficients.  Terms 1--2 (incident shadow) are unchanged.  This provides a
practical impedance-wedge diffraction coefficient that is ForwardDiff-compatible
and reduces exactly to the PEC result as ``|\varepsilon_r| \to \infty``.

See also:
- [Impedance Wedge Diffraction](@ref impedance) for derivation and examples.

## Maliuzhinets exact solution

```@docs
psi_Phi
maliuzhinets_DsDh
```

The Maliuzhinets spectral function method provides the **exact** impedance-wedge
diffraction coefficients for normal incidence with uniform (but not identical)
surface impedances on each face.  It serves as a validation reference for the
Holm heuristic.

`psi_Phi(w, Phi)` evaluates the Maliuzhinets function ``\psi_\Phi(w)`` via
adaptive Gauss--Kronrod quadrature within the convergence strip, extended
beyond via the functional relation.

`maliuzhinets_DsDh` computes the exact ``D_s`` and ``D_h`` from the spectral
function approach of Kotelnikov et al. (2013).  It accepts raw angles and
permittivities (not wrapped in struct types) for flexibility.

### Practical constraints

- `maliuzhinets_DsDh` requires `alpha ∈ (π, 2π)` (exterior wedge only).
- Computation involves adaptive quadrature and is significantly slower than `impedance_wedge_DsDh`.
- The soft coefficient ``D_s`` is validated only for finite impedance; for PEC, use `pec_wedge_DsDh`.
- The hard coefficient ``D_h`` reduces exactly to PEC as ``\varepsilon_r \to \infty``.

See also:
- [Maliuzhinets Exact Solution](@ref maliuzhinets) for theory and examples.
