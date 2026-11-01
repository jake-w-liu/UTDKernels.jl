# UTDKernels development notes

## Development Ledger

| Item | Contract and evidence |
|---|---|
| Objective | Improve native classical-pole normalization and moderate positive-real transition derivatives without changing the physical formulas or existing public APIs. |
| Source changes | Reassociate the classical regular term; use a factored continued fraction for binary64 F′ on `[16,35)`; validate finite ordinary/nested AD seeds before stable propagation. |
| Inputs and outputs | Preserve positive-finite derivative domains, scalar/complex dispatch and the existing positive-infinity value limit of F. The independent classical integral requires `Im(z)>0`. |
| Independent oracles | Gaussian moments, polynomial real-line quadrature, convergent high-precision erf series, separate decaying real-axis integrals and exact-input binary64 provenance. |
| Test plan | Retain all previous tests and tolerances, including the strict `1e-17` regression; add interval-neighbor, ordinary/nested AD, finite-seed, precision-agreement and old-literal rejection controls. |
| Baseline meaning | Classical pole extraction retains the full pole amplitude and first regular saddle term. Its nonzero quartic remainder is intentional and is never added to the comparator. |
| Regeneration trigger | Rerun the component experiment after changes to classical normalization, hierarchy, amplitudes or moments; rerun derivative qualification after transition/AD changes. |
| Reproducibility | Deterministic grids; recorded source/test/experiment/Manifest hashes; exact native input bits; separate raw logs for Julia 1.13.3 and 1.12.7. |
| Qualification | Both runtimes pass complete `Pkg.test()`, the scalar experiment and the canonical development harness. These component checks do not establish manuscript-level accuracy. |

## Implementation Correctness Gate

For ascending quartic coefficients, let `h=1/sqrt(k)`, `q=z*h`, and
`W[m+1]=z*W[m]-i*mu[m]/pi`, with Gaussian moments `mu[m]`. The hierarchy
and classical comparator satisfy

```text
H = sum(a[m]*h^m*W[m], m=0:4)
C = a(q)*W[0] - i*((a(q)-a[0])/q)/(sqrt(pi)*sqrt(k))
H-C = -i*h^3*(a[3]+a[4]*q)/(2*sqrt(pi)).
```

The regular term now uses `(numerator/z)/sqrt(pi)`, avoiding redundant
rounded normalization through q and sqrt(k). The callback still receives q;
moments, recovery guards and the comparator's retained order are unchanged.
The original strict fixture at `k=400`, `z=0.8+0.35im` remains below `1e-17`,
alongside the newer upstream high-precision/ULP and mutation checks.

## Moderate positive-real transition derivative

The raw erfcx primal evaluator is unchanged. For Float64 primal x in
`16 <= x < 35`, including ForwardDiff inputs, the first derivative uses a
factored form of [DLMF 7.9.1](https://dlmf.nist.gov/7.9#E1). With
`z=sqrt(x/2)*(1+i)`, factor the first two levels of the erfcx fraction:

```text
T = (3/2)/(z + 2/(z + (5/2)/(z + ...)))
w = 1/(z+T), D = z*(z+w)+1/2
F = z*(z+w)/D = 1-1/(2D).
```

Since `z^2=i*x`, substitution into `F'=(i+1/(2*x))*F-i` gives
`F'=z*w/(2*x*D)` without subtracting nearly equal terms. The implementation
uses 64 half-step numerators. The infinite-fraction identity alone does not
certify fixed depth: native 32/64/96-depth comparisons and independent
high-precision references were checked on a 159-point grid. At finite depth
the expression is the ODE transform of the truncated fraction, rather than
exactly its differentiated truncation; ordinary/nested AD was checked
independently.

Below 16 and at/above 35, the original ODE and inverse-power paths remain.
Non-Float64 real types and complex/passive inputs retain their previous
dispatch. A native before/after probe at x=20 found exactly identical values,
output types or rejection behavior for Float32, integer, rational, BigFloat,
positive-real complex and passive-complex cases. This is not a claim that
non-Float64 ODE evaluation has the new branch's accuracy.

Within the new interval the largest observed relative first-derivative error
was `6.62e-16` for the public API, bundle and AD path. Bundle second/third
ODE recurrences remain, with measured errors up to `9.97e-14` and `1.15e-12`.
Independent AD second/third errors were `1.19e-15` and `1.52e-15`. These are
measured grid results, not uniform correct-rounding bounds.

## Exact-input residual expectations

Two legacy complex residual literals were independently disproved. Native
input bits were recorded before reconstructing either oracle:

| Case | epsilon bits | Delta bits | Old literal relative error |
|---|---|---|---:|
| g1 | `3d719799812dea11` | `40041b2f769cf0e0` | `8.30674e-14` |
| g2 | `3e45798ee2308c3a` | `3ff921fb54442d18` | `2.73802e-14` |

The corrected literals equal the nearest binary64 values of their separate
120-digit references. Tests retain the original `2e-14` and `5e-14` gates,
require 80/120-digit agreement, check the exact input bits and reject the
old literals. All other legacy residual/coefficient/score/polygon fixtures
were audited together and retained unchanged.

For g1, native F′ and delta-B errors were `1.773e-12`; substituting only
oracle delta-B diagnostically reduced residual error from `8.28955e-14` to
`2.34791e-15`. For g2, corrected production error is `1.324e-15`, whereas
the old ODE output was `2.742e-14` from the reference. Independent decaying
real-axis integrals agreed with the Julia high-precision series oracle to
approximately `7.15e-119` and `5.86e-123` relative. These checks distinguish
production arithmetic error from inaccurate expectations.

## Invalid AD seeds

A pre-fix 48-case native audit rejected only 12 nonfinite ordinary/nested
AD seed inputs; stable paths returned nonfinite derivatives in the other
36 cases. Recursive finite-value/partial guards now reject every case.
The existing F(+Inf)=1 endpoint with a finite seed remains accepted, with
zero derivative. The derivative and derivative-bundle domains remain finite
positive x.

## Component experiment and completed checks

`julia --startup-file=no --project=. examples/experiments.jl` checks nine
polynomial cases using two independently controlled real-line quadrature
resolutions. Sign, factor and retained-order mutations must exceed ten
times the same fixed common budget. Only an ignored validation CSV is
written; the experiment performs no electromagnetic solve and produces no
figures.

Each native runtime passes 30,244 complete-package assertions. A separate
bounded run on each runtime passes 11,078 assertions, including the entire
legacy FaceEdgeUTD suite and dense derivative checks; these executions
overlap and their counts are not a count of unique tests. The canonical
harness reports 162 passes, three warnings and zero failures on both
runtimes. Detailed limits and evidence are in TEST_REPORT.md.
