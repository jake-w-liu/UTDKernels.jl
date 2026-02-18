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

## Regime detection

```@docs
wedge_transition_args
```

## Diagnostics

```@docs
inspect_kp_terms
```
