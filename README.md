# UTDKernels.jl

A branch-safe and differentiable Julia implementation of Uniform Theory of
Diffraction (UTD) wedge kernels and finite-edge canonical integrals, with
impedance-wedge and exact-reference evaluators.

This package accompanies:

> J. W. Liu, "UTDKernels.jl: A branch-safe and differentiable Julia library for
> Kouyoumjian–Pathak wedge diffraction kernels," *SoftwareX*, vol. 35, 102815,
> 2026. [doi:10.1016/j.softx.2026.102815](https://doi.org/10.1016/j.softx.2026.102815)

## Features

- **Overflow-free transition function**: Evaluates F(x) = sqrt(pi*x) * exp(+i*pi/4) * erfcx(exp(+i*pi/4)*sqrt(x)) via the scaled complementary error function, including the real `x = +Inf` GTD limit without overflow
- **Regularised cot-F product**: Eliminates the infinity-times-zero singularity at shadow and reflection boundaries
- **Face-grazing continuation**: `pec_wedge_DsDh_grazing` evaluates the same PEC pairing without the soft G(φ−h)−G(φ+h) cancellation, for interior and exterior wedges and the infinite-distance `F → 1` limit `L = Inf` (an exact closed form). Finite-distance integration refines the Gauss–Legendre order until the highest-order estimate agrees with two lower-order checks. For interior wedges, the pairing follows the [Hutchins–Kouyoumjian arbitrary-angle nearest-integer construction](https://doi.org/10.21236/AD0699228) in KP transition-function form. `pec_wedge_DsDh` is unchanged. `wedge_DsDh` is the recommended entry point: it uses reciprocity to auto-select the domain-certified, quadrature-checked continuation when either the incident or observation direction approaches a face, and uses the four-term form otherwise. The continuation is refused for impedance wedges, unequal L, and uncertified intervals. Plane-wave incidence alone has `sp = Inf` and therefore `L = s`, generally finite.
- **Finite-edge endpoint transition**: Exact source–edge–observer phase coordinates and three closed Fresnel moments evaluate a smooth scalar finite straight-edge integral continuously as its stationary point crosses an endpoint.
- **Reflection-boundary face–edge split**: Forms the cancellation-free intrinsic PEC incident pair and the finite face transition separately, with an explicit nearest-pole branch contract.
- **Passive complex transitions**: Evaluates `F-1`, `F'`, and `F''` without large-argument cancellation throughout the physical passive sector, using principal mathematical roots.
- **Bivariate Fresnel transitions**: Retains correlation between two simultaneous canonical boundaries through a finite Plackett integral, four-region weights, and a robust mixed-Hessian map.
- **Curvature-measure seam limits**: Reuses the intrinsic PEC face–edge component to form raw and locally debiased sums over supplied exterior turning angles.
- **Clustered-pole multipole transitions**: Evaluates Faddeeva divided differences from separated poles through exact coalescence with a bounded automatic basis switch.
- **Coordinate-invariant Hessian metrics**: Reduces an astigmatic saddle--pole neighborhood to a signed dual-norm coordinate and effective distance without forming a matrix inverse.
- **Null-uniform Faddeeva moments**: Promotes amplitude derivatives near saddle-scale zeros through certified moment recurrences, large-argument asymptotics, and shifted-null bases.
- **Continuous-order saddle--endpoint transitions**: Resolves algebraic
  endpoints with certified log-domain half-line quadrature, an overflow-safe
  scaled saddle branch, and analytic-amplitude moments.
- **Distributional shadow-boundary sensitivity**: Supplies the exact Fresnel
  switch derivative, dispersive Fourier multiplier, finite expansion, and
  nonlinear-coordinate pullback coefficients.
- **Automatic differentiation**: ForwardDiff.jl package extension for end-to-end gradients of diffraction coefficients with respect to angle, wavenumber, and distance
- **Explicit square-root policies**: Mathematical kernels use the principal branch; Fresnel and material-wave roots use the passive lower-bank limit on the negative-real cut
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
- `Wedge(alpha)`, `wedge_n(w)`, `wedge_nu(w)` -- Exterior-wedge geometry and KP parameters, with a type-local inclusive `2π` endpoint
- `RayAngles(phi, phip)` -- Observation and incident azimuths
- `Distances(s, sp)`, `effective_L(d)` -- Ray distances and overflow-safe effective distance
- `wrap_angle(phi, alpha)` -- Robust periodic normalization to `[0, alpha)`

### Transition functions

- `F_utd(x)` -- UTD transition function via `erfcx`
- `F_utd_prime(x)` -- Stable transition-function derivative
- `F_utd_minus_one(x)` -- Cancellation-free `F(x) - 1` at large real `x`
- `F_utd_second(x)` -- Second transition derivative on the positive real axis or passive complex sheet
- `is_passive_transition_argument(x)`, `passive_wavenumber(k0, attenuation)` -- Passive-sector classification and lossy-wavenumber construction

### Finite-edge integral

- `FiniteEdgeGeometry`, `FiniteEdgeAmplitude` -- Straight-edge geometry and local smooth-amplitude data
- `finite_edge_phase_coordinate`, `finite_edge_coordinate_derivative` -- Exact rationalized phase map and derivative
- `finite_edge_fresnel_moments` -- Closed moments through quadratic order
- `finite_edge_epm`, `finite_edge_endpoint_derivative` -- Endpoint-uniform field and moving-endpoint derivative
- `finite_edge_stationary_phase` -- Leading infinite-edge stationary-phase limit

### Bivariate transition

- `bivariate_fresnel_transition(xi, eta, rho)` -- Adaptive correlated two-boundary switch with an optional fixed Gauss--Legendre order
- `bivariate_mechanism_weights(xi, eta, rho)` -- Four correlated canonical-region weights
- `bivariate_transition_hessian(a, b, c, u, v; k)` -- Positive-definite quadratic-phase map to `(xi, eta, rho)`

### Curvature-measure seam limit

- `intrinsic_seam_coefficient(delta, k, L)` -- Symmetric near-coplanar intrinsic PEC coefficient for one positive exterior turn
- `curvature_measure_sum(turning, amplitudes)` -- Normalized raw turning measure with exact local wedge bias
- `debiased_curvature_measure_sum(turning, amplitudes)` -- Turning measure after removing that local bias
- `curvature_continuum_harmonic(m, z, phi)` -- Closed circular Bessel-harmonic reference

### Clustered-pole multipole transition

- `faddeeva_divided_difference(nodes)`, `faddeeva_divided_difference_with_condition(nodes)` -- Direct distinct-node Faddeeva divided difference, with a separate typed diagnostic form
- `multipole_transition(nodes)`, `multipole_transition_with_info(nodes)` -- Automatic separated/confluent evaluation, with a separate typed diagnostic form, for validated orders 1 through 7
- `MultipoleEvaluationInfo` -- Typed diagnostics with `:single`, `:direct`, or `:cluster` method identifiers

### Coordinate-invariant Hessian metric

- `hessian_metric_q2(H, g)` -- Cholesky-evaluated dual metric `g' * (H \ g)`
- `hessian_effective_L(H, g)` -- Extreme-scale-safe reciprocal dual metric
- `hessian_transition_coordinate(delta, H, g)` -- Signed canonical offset
- `hessian_transition_argument(k, delta, H, g)` -- Nonnegative argument for `F_utd`
- `directional_effective_L(R1, R2, beta)` -- Stable principal-radius harmonic projection

### Null-uniform Faddeeva moments

- `faddeeva_moments(z, max_order)` -- Certified canonical moments sharing the central Faddeeva owner
- `null_uniform_transition(k, z, coefficients)` -- Ascending analytic-amplitude hierarchy
- `shifted_null_transition(k, z, Lambda, zero_order, coefficients)` -- Moving higher-order-null hierarchy

### Continuous-order saddle--endpoint transitions

- `continuous_order_transition(mu, zeta)` -- Coalescence-normalized
  continuous-order canonical family, with exact unit-order Faddeeva reuse
- `scaled_continuous_order_transition(mu, zeta)` -- Direct real saddle-side
  evaluation of `exp(-zeta^2) * B_mu(zeta)`
- `continuous_order_moment(nu, index, k, h, tau, coefficient)` -- One physical
  analytic-amplitude moment; a coefficient-vector overload evaluates the
  bounded hierarchy without a term array

### Distributional shadow-boundary sensitivity

- `shadow_switch(q)` -- Signed canonical Fresnel switch for the supported
  `exp(+i*omega*t)` convention
- `shadow_sensitivity_kernel(q)` -- Exact canonical derivative; the
  `(s, kappa)` overload evaluates `sqrt(kappa) K(sqrt(kappa) s)`
- `shadow_sensitivity_multiplier(xi, kappa; terms=nothing)` -- Exact
  unit-modulus Fourier multiplier or a bounded finite Taylor expansion
- `shadow_sensitivity_pullback(...)` -- Leading, first, and second
  nonlinear-coordinate actions from caller-supplied local derivatives

### PEC coefficients and fields

- `wedge_DsDh(w, ang, k, L)` -- Recommended router: cancellation-free continuation near grazing and the four-term form elsewhere
- `pec_wedge_DsDh(w, ang, k, L...)` -- Original four-term pairing, including the separate-distance form
- `pec_wedge_face_edge(w, delta, k, L)` -- Reflection-boundary decomposition into `Ds`, `Dh`, intrinsic `edge`, and `face`
- `pec_wedge_intrinsic_score(w, delta, k, L)` -- Dimensionless local candidate-edge score
- `pec_wedge_DsDh_grazing(w, ang, k, L)` -- Domain-certified, adaptively refined face-grazing continuation
- `pec_wedge_Ds_linear(w, ang, k, L)` -- Leading soft Taylor term for comparison
- `grazing_local_angles(w, ang)`, `grazing_interval_report(w, ang, k, L)` -- Face-local mapping and continuation certificate
- `GrazingIntervalReport`, `GrazingDomainError` -- Domain-certificate result and typed continuation failure
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
│   │   ├── Branches.jl            # principal and passive material roots
│   │   ├── Numerics.jl            # DEFAULT_TRANSITION_TOL
│   │   └── Quadrature.jl          # bounded shared Gauss–Legendre cache
│   ├── transition/
│   │   ├── FaddeevaCore.jl        # shared scaled-erfc/Faddeeva identity
│   │   ├── TransitionF.jl         # F_utd(x) via erfcx
│   │   ├── TransitionFPrime.jl    # real F_utd_prime, F_utd_minus_one
│   │   ├── PassiveTransition.jl   # passive complex residual and derivatives
│   │   ├── BivariateTransition.jl # correlated two-boundary canonical factor
│   │   ├── MultipoleTransition.jl # clustered Faddeeva divided differences
│   │   ├── NullUniformMoments.jl  # amplitude-null moment hierarchy
│   │   └── ContinuousOrderTransition.jl # algebraic endpoint family
│   ├── finite_edge/
│   │   └── FiniteEdge.jl          # exact phase map and endpoint moments
│   ├── geometry/
│   │   ├── CurvatureMeasure.jl    # intrinsic turning-angle seam measure
│   │   └── HessianMetric.jl       # astigmatic dual metric and distance
│   ├── sensitivity/
│   │   └── ShadowBoundarySensitivity.jl # dispersive shadow derivative
│   ├── fresnel/
│   │   └── Fresnel.jl             # materials and TE/TM reflection
│   ├── wedge/
│   │   ├── WedgeGeometry.jl       # KP four-term structure (psi_j, N_j, a_j)
│   │   ├── WedgePEC.jl            # pec_wedge_DsDh, _cot_F_regularized
│   │   ├── WedgeFaceEdge.jl       # reflection-boundary face–edge split
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
│   └── UTDKernelsForwardDiffExt.jl  # ForwardDiff rules for complex erfc/erfcx
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

## Building the documentation

The generated `docs/build/` site is intentionally not tracked. Rebuild it from
the package root with `julia --project=docs docs/make.jl`.

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
The generated CSV, paper datasets, and paper figures are intentionally not
tracked in the repository.

## Textbook Example Plots (Balanis GTD)

The `examples/` folder includes executable reproductions of Balanis chapter examples
13-3 through 13-7, with side-by-side pattern comparisons between:

- `UTDKernels` API evaluations, and
- direct textbook-equation reconstructions.

Plots are generated with `PlotlySupply` and exported to `examples/figs/`. These
generated PNG files are intentionally not tracked.

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
