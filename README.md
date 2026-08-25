# UTDKernels.jl

A branch-safe and differentiable Julia implementation of the Uniform Theory of
Diffraction (UTD) for PEC wedges, with impedance-wedge and exact-reference
evaluators.

This package accompanies:

> J. W. Liu, "UTDKernels.jl: A branch-safe and differentiable Julia library for
> Kouyoumjian–Pathak wedge diffraction kernels," *SoftwareX*, vol. 35, 102815,
> 2026. [doi:10.1016/j.softx.2026.102815](https://doi.org/10.1016/j.softx.2026.102815)

## Features

- **Overflow-free transition function**: Evaluates F(x) = sqrt(pi*x) * exp(+i*pi/4) * erfcx(exp(+i*pi/4)*sqrt(x)) via the scaled complementary error function, including the real `x = +Inf` GTD limit without overflow
- **Regularised cot-F product**: Eliminates the infinity-times-zero singularity at shadow and reflection boundaries
- **Face-grazing continuation**: `pec_wedge_DsDh_grazing` evaluates the same PEC pairing without the soft G(φ−h)−G(φ+h) cancellation, for interior and exterior wedges and the infinite-distance `F → 1` limit `L = Inf` (an exact closed form). For interior wedges, the pairing follows the [Hutchins–Kouyoumjian arbitrary-angle nearest-integer construction](https://doi.org/10.21236/AD0699228) in KP transition-function form. `pec_wedge_DsDh` is unchanged. `wedge_DsDh` is the recommended entry point: it uses reciprocity to auto-select the certified continuation when either the incident or observation direction approaches a face, and uses the four-term form otherwise. The continuation is refused for impedance wedges, unequal L, and uncertified intervals. Plane-wave incidence alone has `sp = Inf` and therefore `L = s`, generally finite.
- **Automatic differentiation**: ForwardDiff.jl package extension for end-to-end gradients of diffraction coefficients with respect to angle, wavenumber, and distance
- **Principal-branch consistency**: Branch-sensitive square roots use a single documented branch via `safe_sqrt`, ensuring AD compatibility
- **Validated**: Tested against the exact Sommerfeld half-plane solution, GTD convergence, reciprocity, independent formula reconstructions, and automatic-differentiation finite differences

## Convention

All fields use the exp(+iωt) phasor convention:

- Outgoing waves: exp(-iks)
- Incident plane wave: exp(+ikr cos(φ - φ'))
- Maxwell equations: curl E = -iωμH, curl H = +iωεE

## Installation

```julia
using Pkg
Pkg.add("UTDKernels")
```

## Quick Start

```julia
using UTDKernels

# Half-plane wedge (exterior angle = 2pi)
w = Wedge(2pi)

# Observation angle phi = 90 deg, incident angle phi' = 45 deg
ang = RayAngles(pi/2, pi/4)

# Compute soft and hard diffraction coefficients with the recommended router
k = 10.0   # wavenumber
L = 1.0    # effective distance parameter
Ds, Dh = wedge_DsDh(w, ang, k, L)

# Near face grazing, the same API selects the cancellation-free continuation.
Ds, Dh = wedge_DsDh(w, RayAngles(pi/2, 1e-14), k, L)   # tiny incident offset

# The original four-term pairing remains available as a validation baseline.
Ds_four, Dh_four = pec_wedge_DsDh(w, ang, k, L)

# Transition function
F = F_utd(1.0)   # F(1) ~ 0.81 + 0.23i, |F| ~ 0.84
```

### Diffracted field computation

```julia
# Full diffracted field: E^d = D * E^i * A(s,s') * exp(-iks)
Es_i, Eh_i = 1.0, 0.0   # incident field in soft/hard basis
s, sp = 2.0, Inf         # distances (plane-wave incidence)

Es_d, Eh_d = pec_wedge_apply_sh(Ds, Dh, Es_i, Eh_i, k, s, sp)
```

### Automatic differentiation

```julia
using ForwardDiff

w = Wedge(2pi)
f(phi) = abs(wedge_DsDh(w, RayAngles(phi, pi/4), 10.0, 1.0)[1])

# Gradient of |Ds| with respect to observation angle
dDs_dphi = ForwardDiff.derivative(f, pi/2)
```

## API

### Geometry and convention

- `PhasorConvention`, `EXP_IWT` -- Time-harmonic convention type and the supported exp(+iωt) constant
- `Wedge(alpha)`, `wedge_n(w)`, `wedge_nu(w)` -- Exterior-wedge geometry and KP parameters
- `RayAngles(phi, phip)` -- Observation and incident azimuths
- `Distances(s, sp)`, `effective_L(d)` -- Ray distances and overflow-safe effective distance
- `wrap_angle(phi, alpha)` -- Robust periodic normalization to `[0, alpha)`

### Transition functions

- `F_utd(x)` -- UTD transition function via `erfcx`
- `F_utd_prime(x)` -- Stable transition-function derivative
- `F_utd_minus_one(x)` -- Cancellation-free `F(x) - 1` at large real `x`

### PEC coefficients and fields

- `wedge_DsDh(w, ang, k, L)` -- Recommended router: cancellation-free continuation near grazing and the four-term form elsewhere
- `pec_wedge_DsDh(w, ang, k, L...)` -- Original four-term pairing, including the separate-distance form
- `pec_wedge_DsDh_grazing(w, ang, k, L)` -- Certified face-grazing continuation
- `pec_wedge_Ds_linear(w, ang, k, L)` -- Leading soft Taylor term for comparison
- `grazing_local_angles(w, ang)`, `grazing_interval_report(w, ang, k, L)` -- Face-local mapping and continuation certificate
- `GrazingIntervalReport`, `GrazingDomainError` -- Certificate result and typed domain failure
- `two_term_kernel(beta, w, k, L)`, `two_term_kernel_derivative(beta, w, k, L)` -- Branch-local paired kernel and derivative
- `pec_wedge_apply_sh(...)`, `spreading_factor(s, sp)` -- Soft/hard field application and spreading

### Impedance and exact-reference coefficients

- `WedgeFaceMaterial`, `ImpedanceWedge` -- Face material and impedance-wedge types
- `fresnel_te(psi, eps_r)`, `fresnel_tm(psi, eps_r)` -- Grazing-angle reflection coefficients
- `impedance_wedge_DsDh(iw, ang, k, L...)` -- Holm impedance-wedge evaluator
- `psi_Phi(w, Phi)`, `maliuzhinets_DsDh(...)` -- Maliuzhinets special function and exact-reference coefficients

### Classification and inspection

- `wedge_transition_args(w, ang, k, L)` -- Four-term transition arguments and regimes
- `inspect_kp_terms(w, ang, k, L)` -- Printed KP term summary

The [API reference](docs/src/api.md) gives
complete signatures, keyword defaults, domains, and return fields.

## Package Structure

```
UTDKernels.jl/
├── src/
│   ├── UTDKernels.jl              # Module entry point
│   ├── common/
│   │   ├── Types.jl               # Wedge, RayAngles, Distances, PhasorConvention
│   │   ├── AngleWrap.jl           # wrap_angle
│   │   ├── Branches.jl            # safe_sqrt (principal branch)
│   │   └── Numerics.jl            # DEFAULT_TRANSITION_TOL
│   ├── transition/
│   │   ├── TransitionF.jl         # F_utd(x) via erfcx
│   │   └── TransitionFPrime.jl    # F_utd_prime, F_utd_minus_one
│   ├── fresnel/
│   │   └── Fresnel.jl             # materials and TE/TM reflection
│   ├── wedge/
│   │   ├── WedgeGeometry.jl       # KP four-term structure (psi_j, N_j, a_j)
│   │   ├── WedgePEC.jl            # pec_wedge_DsDh, _cot_F_regularized
│   │   ├── WedgeDyadic.jl         # pec_wedge_apply_sh, spreading_factor
│   │   ├── WedgeImpedance.jl      # Holm impedance coefficient
│   │   ├── WedgeGrazing.jl        # certified grazing continuation and router
│   │   └── Regimes.jl             # regime detection (:lit, :shadow, :transition)
│   ├── maliuzhinets/
│   │   ├── MaliuzhinetsFunction.jl # psi_Phi
│   │   └── MaliuzhinetsExact.jl    # exact impedance-wedge reference
│   └── utils/
│       └── Diagnostics.jl         # inspect_kp_terms
├── ext/
│   └── UTDKernelsForwardDiffExt.jl  # ForwardDiff AD rule for erfcx
├── examples/
│   ├── README.md                    # Balanis GTD examples (13-3 to 13-7)
│   ├── run_all.jl                   # Run all textbook validation examples
│   └── example_13_*.jl              # Individual example scripts + PlotlySupply plots
├── test/
│   ├── runtests.jl
│   ├── test_transition.jl           # transition values, derivatives, limits
│   ├── test_wedge_pec_*.jl          # boundaries, distances, limits, grazing
│   ├── test_wedge_impedance.jl      # Fresnel and Holm checks
│   ├── test_maliuzhinets.jl         # exact-reference identities and limits
│   ├── test_ad.jl                   # ForwardDiff vs finite differences
│   └── test_robustness.jl           # extreme-scale and invalid-input checks
├── validation/
│   ├── generate_wdc_reference.m    # Balanis WDC.m MATLAB reference (no toolboxes)
│   ├── compare_wdc.jl              # Julia-vs-MATLAB cross-validation
│   └── data/wdc_reference.csv      # Locally generated MATLAB cross-validation set
├── paper/
│   ├── generate_paper_data.jl      # Regenerates all in-paper CSV data
│   ├── plot_paper.jl               # Regenerates all in-paper figures
│   ├── data/*.csv                  # Figure/table source data
│   └── figs/*.pdf                  # Publication figures
├── docs/                           # Documenter.jl documentation
└── Project.toml
```

## Reproducing the paper

The `paper/` directory regenerates every figure and table in the
accompanying SoftwareX article. It has a dedicated, resolved Julia 1.12
environment that declares all data and plotting dependencies.

```bash
# Run from the package root.
julia --startup-file=no --project=paper -e 'using Pkg; Pkg.instantiate()'

# Regenerate the in-paper data and figures.
julia --startup-file=no --project=paper paper/generate_paper_data.jl
julia --startup-file=no --project=paper paper/plot_paper.jl
```

The Balanis MATLAB cross-validation set is produced locally as
`validation/data/wdc_reference.csv` by `generate_wdc_reference.m` (no MATLAB
toolboxes required) and checked against the Julia kernel by `compare_wdc.jl`.
The generated CSV is intentionally not tracked in the repository.

## Textbook Example Plots (Balanis GTD)

The `examples/` folder includes executable reproductions of Balanis chapter examples
13-3 through 13-7, with side-by-side pattern comparisons between:

- `UTDKernels` API evaluations, and
- direct textbook-equation reconstructions.

Plots are generated with `PlotlySupply` and exported to `examples/figs/`.

```bash
julia --project=examples examples/run_all.jl
```

## Running Tests

```bash
julia --startup-file=no --project=. -e 'using Pkg; Pkg.test()'
```

The test suite covers transition accuracy, boundary continuity, GTD convergence,
reciprocity, grazing continuation, impedance and exact-reference behavior,
extreme-scale inputs, and automatic-differentiation gradients.

## Requirements

- **Julia**: 1.12+
- **SpecialFunctions.jl**: v2+ (provides `erfcx`)
- **ForwardDiff.jl**: optional, for automatic differentiation

The package root tracks `Project.toml` only. A root `Manifest.toml` is local
resolver output and is ignored; the dedicated `docs/`, `examples/`, `paper/`,
and `validation/` environments retain their own `Project.toml` and
`Manifest.toml` for reproducible tooling.

## License

MIT License. See [LICENSE](LICENSE.md) for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@article{liu2026utdkernels,
  title   = {{UTDKernels.jl}: A branch-safe and differentiable Julia library for
             Kouyoumjian--Pathak wedge diffraction kernels},
  author  = {Liu, Jake W.},
  journal = {SoftwareX},
  volume  = {35},
  pages   = {102815},
  year    = {2026},
  doi     = {10.1016/j.softx.2026.102815}
}
```

## References

1. R. G. Kouyoumjian and P. H. Pathak, "A uniform geometrical theory of diffraction for an edge in a perfectly conducting surface," *Proc. IEEE*, vol. 62, no. 11, pp. 1448--1461, Nov. 1974.
2. J. B. Keller, "Geometrical theory of diffraction," *J. Opt. Soc. Am.*, vol. 52, no. 2, pp. 116--130, 1962.
