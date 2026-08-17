# UTDKernels.jl

A branch-safe and differentiable implementation of the Uniform Theory of Diffraction (UTD) for PEC wedges in Julia, with impedance-wedge and exact-reference validation routines.

<!-- This package accompanies the paper:

> J. W. Liu, "UTDKernels.jl: A Branch-Safe and Differentiable Implementation of Uniform Theory of Diffraction for Electromagnetic Wedges," *Computer Physics Communications*, 2025. -->

## Features

- **Overflow-free transition function**: Evaluates F(x) = sqrt(pi*x) * exp(+i*pi/4) * erfcx(exp(+i*pi/4)*sqrt(x)) via the scaled complementary error function, including the real `x = +Inf` GTD limit without overflow
- **Regularised cot-F product**: Eliminates the infinity-times-zero singularity at shadow and reflection boundaries
- **Face-grazing continuation**: `pec_wedge_DsDh_grazing` evaluates the same PEC pairing without the soft G(φ−h)−G(φ+h) cancellation. `pec_wedge_DsDh` is unchanged. The continuation is refused for impedance wedges, unequal L, and uncertified intervals.
- **Automatic differentiation**: ForwardDiff.jl package extension for end-to-end gradients of diffraction coefficients with respect to angle, wavenumber, and distance
- **Principal-branch consistency**: Branch-sensitive square roots use a single documented branch via `safe_sqrt`, ensuring AD compatibility
- **Validated**: Tested against the exact Sommerfeld half-plane solution, GTD convergence at O(1/kL), reciprocity to machine precision, and over one thousand automated test assertions

## Convention

All fields use the exp(+iwt) phasor convention:
- Outgoing waves: exp(-iks)
- Incident plane wave: exp(+ikr*cos(phi - phi'))
- Maxwell equations: curl E = -iwu*H, curl H = +iwe*E

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

# Compute soft and hard diffraction coefficients
k = 10.0   # wavenumber
L = 1.0    # effective distance parameter
Ds, Dh = pec_wedge_DsDh(w, ang, k, L)

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
f(phi) = abs(pec_wedge_DsDh(w, RayAngles(phi, pi/4), 10.0, 1.0)[1])

# Gradient of |Ds| with respect to observation angle
dDs_dphi = ForwardDiff.derivative(f, pi/2)
```

## API

### Types

- `Wedge(alpha)` -- Wedge with exterior angle alpha in (0, 2pi]
- `RayAngles(phi, phip)` -- Observation and incident azimuths
- `Distances(s, sp)` -- Edge-to-observer and source-to-edge distances
- `PhasorConvention` -- Time-harmonic convention (`EXP_IWT` for exp(+iwt))

### Functions

- `F_utd(x)` -- UTD transition function via erfcx
- `pec_wedge_DsDh(w, ang, k, L)` -- Soft/hard scalar diffraction coefficients (four-term pairing)
- `pec_wedge_DsDh_grazing(w, ang, k, L)` -- Same PEC coefficient via certified Gauss–Legendre continuation of G'
- `grazing_interval_report(w, ang, k, L)` -- Analytic interval certificate for that continuation
- `F_utd_prime(x)` -- Transition-function derivative, with a large-x series branch
- `pec_wedge_apply_sh(Ds, Dh, Es, Eh, k, s, sp)` -- Apply diffraction dyadic to field components
- `impedance_wedge_DsDh(iw, ang, k, L)` -- Impedance-wedge diffraction coefficients
- `spreading_factor(s, sp)` -- UTD spreading factor A(s,s') = sqrt(s'/(s(s+s')))
- `effective_L(d::Distances)` -- Effective distance L = s*s'/(s+s')
- `wedge_n(w)` / `wedge_nu(w)` -- Wedge parameters n = alpha/pi, nu = pi/alpha
- `wrap_angle(phi, alpha)` -- Normalise angle to [0, alpha)
- `fresnel_te(psi, eps_r)` / `fresnel_tm(psi, eps_r)` -- Grazing-angle Fresnel reflection coefficients
- `psi_Phi(w, Phi)` / `maliuzhinets_DsDh(alpha, eps_r_o, eps_r_n, phi, phip, k)` -- Exact-reference spectral-function routines
- `inspect_kp_terms(w, ang, k, L)` -- Diagnostic printout of KP four-term structure

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
│   │   └── TransitionF.jl         # F_utd(x) via erfcx
│   ├── wedge/
│   │   ├── WedgeGeometry.jl       # KP four-term structure (psi_j, N_j, a_j)
│   │   ├── WedgePEC.jl            # pec_wedge_DsDh, _cot_F_regularized
│   │   ├── WedgeDyadic.jl         # pec_wedge_apply_sh, spreading_factor
│   │   └── Regimes.jl             # Regime detection (:lit, :shadow, :transition)
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
│   ├── test_transition.jl           # F(x) limits, monotonicity, complex args
│   ├── test_reference_values.jl     # Reference value verification
│   ├── test_wedge_pec_continuity.jl # Shadow/reflection boundary continuity
│   ├── test_wedge_pec_limits.jl     # GTD high-frequency recovery
│   ├── test_symmetry.jl            # Reciprocity D(phi,phi') = D(phi',phi)
│   └── test_ad.jl                  # ForwardDiff vs finite differences
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
accompanying SoftwareX article. The data and plotting scripts activate the
package environment automatically; `CSV`, `DataFrames`, `ForwardDiff`, and
`PlotlySupply` must be available (e.g. in the default Julia environment).

```bash
# Regenerate the in-paper data (CSV files in paper/data/)
cd paper
julia --project=.. generate_paper_data.jl

# Regenerate the paper figures (paper/figs/*.pdf)
julia --project=.. plot_paper.jl
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

```julia
using Pkg
Pkg.test("UTDKernels")
```

The full test suite passes, covering transition function accuracy, shadow-boundary continuity, GTD convergence, reciprocity, and AD gradient correctness.

## Requirements

- **Julia**: 1.9+
- **SpecialFunctions.jl**: v2+ (provides `erfcx`)
- **ForwardDiff.jl**: optional, for automatic differentiation

## License

MIT License. See [LICENSE](LICENSE) for details.

<!-- ## Citation

If you use this package in your research, please cite:

```bibtex
@article{liu2025utdkernels,
  title   = {{UTDKernels.jl}: A Branch-Safe and Differentiable Implementation of
             Uniform Theory of Diffraction for Electromagnetic Wedges},
  author  = {Liu, Jake W.},
  journal = {Computer Physics Communications},
  year    = {2025}
}
``` -->

## References

1. R. G. Kouyoumjian and P. H. Pathak, "A uniform geometrical theory of diffraction for an edge in a perfectly conducting surface," *Proc. IEEE*, vol. 62, no. 11, pp. 1448--1461, Nov. 1974.
2. J. B. Keller, "Geometrical theory of diffraction," *J. Opt. Soc. Am.*, vol. 52, no. 2, pp. 116--130, 1962.
