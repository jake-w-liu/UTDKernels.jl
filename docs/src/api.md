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
- `pec_wedge_DsDh(wedge, ang, k, Li, Lro, Lrn)` requires `Li > 0`, `Lro > 0`, `Lrn > 0`.
- The single-`L` API supports finite positive `L` and `L = Inf` (far-field limit).
- At exact transition-boundary samples:
  - finite `L` uses a one-sided surrogate evaluation,
  - `L = Inf` uses the far-field midpoint-safe handling for singular terms.
- All angles are internally wrapped to `[0, alpha)`; near grazing incidence, `phip` is collapsed to zero to avoid seam aliasing.

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
