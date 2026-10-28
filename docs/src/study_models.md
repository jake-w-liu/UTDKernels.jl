# Canonical study models

The scalar models and study algorithms are owned by the following UTDKernels
modules. They do not import paper-local packages or DiffUTD.

| Module | Algorithms |
|:--|:--|
| `UTDKernels.FaceEdge` | Reflection-boundary geometry, compensated diagnostics, general-angle incident/reflection pairs, guarded edge selection, polygon GO and diffracted fields |
| `UTDKernels.CurvatureMeasure` | Turning-angle bias, regular and nonuniform seam quadrature, harmonic aliasing, ellipse and smooth-amplitude studies |
| `UTDKernels.CurvatureMeasure.MonostaticAssembly` | Visible seam geometry and cylindrical line-source monostatic fields |
| `UTDKernels.NullUniform` | Factored amplitude jets, shifted-null hierarchies, half-plane spectral reduction and established asymptotic comparators |
| `UTDKernels.ShadowSensitivity` | Bandlimited actions, nonlinear pullbacks, complete Sommerfeld fields, receiver-weighted sensitivities and local test jets |

Existing parent kernels remain the owners of compensated face–edge evaluation,
curvature measure accumulation, error-checked Faddeeva moments, and canonical shadow
switches and pullbacks. The study modules call these kernels.

## Geometry and fields

`FaceEdge.BoundaryGeometry(eps, Delta, k, L)` stores the coplanar defect `eps`
separately from the rounded wedge angle. At general angles,
`FaceEdge.wedge_coefficients(phi, phip, n, k, L)` returns incident and reflection
components `De`, `Df` before assembling `Ds = De-Df` and `Dh = De+Df`.
The general-angle pair API accepts `0 < n <= 2`, finite `k > 0`, and
`L > 0` finite or `+Inf`.
`FaceEdge.wedge_pair_terms` includes `1/sqrt(k)` in both returned sums;
its remaining common prefactor is `-exp(-im*pi/4)/(2n*sqrt(2pi))`.
`wedge_cartesian_detunings` supplies local pole offsets from an oriented
Cartesian edge chart. Coefficient evaluation and `wedge_go_visibility` consume
the same offsets, including the positive-pole convention at an exact boundary.

The `:preserve` policy retains the reflection pair and retains the incident pair
when its transition guard or declared per-edge field tolerance is active.

`FaceEdge.polygon_go_rays` supplies shared visibility decisions and facet anchors
for scalar and Cartesian field assemblers. A reflected ray's phase matches the
incident phase at its facet anchor; translating a facet therefore preserves the
expected plane-wave phase covariance.

`CurvatureMeasure.MonostaticAssembly` uses cylindrical incident illumination
`exp(-im*k*rho)/sqrt(rho)` and outgoing spreading `1/sqrt(rho)`, giving the
round-trip factor `exp(-2im*k*rho)/rho` and effective distance `rho/2`.

`ShadowSensitivity.sommerfeld_field` supports `polarization=:soft` or `:hard`
(default `:soft`). `sommerfeld_cosine_branch_with_derivatives` returns branch
`value`, `dx`, and `dkappa`; `sommerfeld_field_with_derivatives` returns complete
field `value`, `ds`, `dp`, and `dkappa` for Cartesian field reconstruction.

## Numeric and approximation contracts

Geometry constructors promote real inputs and retain supported ForwardDiff
seeds. Differentiability holds inside fixed visibility, pole-sheet and selection
branches. These branch decisions are discrete. The underlying kernel's numeric
support still applies: in particular, finite BigFloat `F_utd` is unsupported
because its complex-erfcx backend is unavailable. Independent high-precision
reference functions provide that validation capability without silent demotion.

Principal scalar FaceEdge coefficient pairs, NullUniform models and moment
fields, and Sommerfeld fields and derivatives retain promoted Float32 or
Float64 inputs, including supported ForwardDiff seeds. Mixed inputs promote.
Validation quadratures, independent comparators and geometry factories retain
their documented Julia promotion behavior.

Real first-order and nested ForwardDiff shadow switch/kernel calls use analytic
chain rules for binary32/binary64 tapes. Per-coefficient range and cancellation
guards retain the bounded wide evaluator for unsafe products, including
subnormal seeds and cancellation-sensitive mixed derivatives. Complex tapes
continue through the general evaluator. Supported finite wave-phase products
also retain their original factors when native multiplication overflows or
would lose an AD coefficient.

The half-plane null model uses analytic pole extraction and binomial Taylor
coefficients. Its independent Cauchy reference is
`NullUniform.wedge_cauchy_coefficients`; it evaluates the unrearranged cotangent
kernel rather than the production algebraic amplitude.

Sensitivity actions based on fourth-order local jets support `order_M=1:3`.
Larger orders throw an error rather than silently returning the third-order
answer. Window exponent jets use an exact rational recurrence, including near
the compact support boundary.
Complete misfit-test jets preserve the exact zero-bearing identity `D(s)=0`
and the zero test outside compact receiver support before evaluating separate
large field terms. Unrepresentable nonzero jets raise a domain error.

`ShadowSensitivity.receiver_quadrature` integrates finite scalar or array
receiver callbacks with their promoted element type. The panel count must lie
in `1:max_panels` (default maximum ten million), and Gauss order in `1:2048`.
Callbacks must be pure; a midpoint sample initializes the accumulator before
quadrature nodes are evaluated. Non-finite samples raise a domain error.

`ShadowSensitivity.spectral_moment` scans the complete requested finite band.
Its `moment` and `l1` values are numerical trapezoidal estimates; the additional
error estimates combine observed frequency-grid and Fourier-quadrature changes.
They do not certify the omitted infinite-band tail. Shifted or separated lobes
are retained, and unresolved Fourier-quadrature refinement raises an error.
Callers must refine `frequency_step` for narrow features or a noninteger-power
cusp at zero. These diagnostics do not replace the full weighted moment required
by the mathematical remainder theorem.

## Independent references and reproduction

`NullUniform.classical_pole_term` evaluates the exact amplitude divided
difference at every nonzero pole coordinate. At the exact zero-pole limit,
its analytic amplitude callback must preserve the widened working precision
and AD coefficients. For a constant amplitude use `t -> one(t)` or
`t -> (1+im)*one(t)`; a callback that casts its result to `ComplexF64` is
rejected at that limit. Unresolved numerical derivative refinement raises an
error. `linearized_pole_term(k, z, a0, 0)` also supplies the constant-amplitude
case without a callback.

High-precision Fresnel/Faddeeva integrals, direct quadrature, Gauss–Hermite,
Cauchy coefficient extraction and cylinder eigenfunction references are retained
for validation. They do not substitute expected datasets for executed models.
The `reproduction/` environment runs the five complete study drivers against the
chosen canonical backend and writes CSV data, quantitative figures, and generated
manuscript evidence. See the reproduction runner's usage for output selection.

The complete manuscript result inventory is:

| Study | CSV datasets | Quantitative figure assets | Table environments |
|:--|--:|--:|--:|
| 160 | 17 | 6 | 8 |
| 162 | 20 | 16 | 6 |
| 165 | 18 | 12 | 3 |
| 168 | 11 | 8 | 6 |
| 170 | 16 | 8 | 4 |
| Total | 82 | 50 | 27 |

The table count includes analytical and experimental-design tables in addition
to numerical result tables. Numerical table values are supplied by generated
manuscript macros and their CSV sources. Schematics drawn directly in LaTeX are
separate from the quantitative figure-asset count.

## Model API

```@autodocs
Modules = [UTDKernels.FaceEdge, UTDKernels.CurvatureMeasure,
           UTDKernels.CurvatureMeasure.MonostaticAssembly,
           UTDKernels.NullUniform, UTDKernels.ShadowSensitivity]
Private = false
```
