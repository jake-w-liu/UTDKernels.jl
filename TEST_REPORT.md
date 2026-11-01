# Verification report

## Completed native qualification

The final numerical source, tests, scalar experiment and local Manifest
were frozen through both complete harness runs. Exact Julia 1.13.3 and
1.12.7 each completed:

| Check | Result per runtime |
|---|---|
| Complete `Pkg.test()` | 30,244 passed; zero failures/errors |
| `examples/experiments.jl` | Nine scalar cases passed |
| Canonical development harness | 162 passed; three warnings; zero failures |
| Separate bounded native verification | 11,078 passed |

The bounded execution contains 90 derivative, 50 finite-seed, 14 exact-input
provenance, 9,346 original FaceEdgeUTD and 1,578 dense-grid assertions. It
overlaps the full suite; the two totals must not be added as unique tests.
The three harness warnings flag existing loose tolerances and empty
manuscript data/figure directories in the package clone. Every original
test and tolerance is retained. In particular,
`abs((exact-cl)-residual)<1e-17` is restored alongside newer upstream
high-precision/ULP/mutation checks and passes.

Both dense derivative CSVs are bit-identical, as are both nine-case
component experiment CSVs. Full raw test logs retain each runtime's actual
resolved test environment; the component CSV's local Manifest digest alone
does not describe that complete runtime environment. These checks qualify
package components, not manuscript conclusions or complete optional
reproduction campaigns.

## Classical-pole contract and independent experiment

The correction reassociates the regular normalization without changing the
classical comparator's approximation or adding its omitted quartic term.
Existing tests cover classical/linearized formulas, zero-pole limits,
callback precision, AD jets, Float32, small/large scales, finite-range
recovery, allocation and adapter delegation. The independent experiment
integrates

```text
(i/pi) * integral[a(u/sqrt(k))*exp(-u^2)/(z-u), u=-Inf:Inf]
```

with explicit ascending polynomial powers, avoiding the production Horner
and moment evaluators. The nine cases use `k=(16,400,1600)`,
`z=(0.4+0.3im,0.8+0.35im,1.2+0.6im)` and `Lambda=1.2`.

Coarse `(rtol=1e-8, atol=5e-12)` and tight `(rtol=2e-13, atol=5e-15)`
requests each must satisfy their own returned quadrature estimate.
Agreement alone cannot excuse a missed target. Native budgets are fixed
from independent absolute Gaussian-integral, pole, saddle and remainder
scales: 128 eps for the main terms, 32 eps for remainder arithmetic and a
separate subtraction allowance. They are engineering acceptance budgets
for this bounded experiment, not correct-rounding theorems.

Independent CSV readback checks nine rows/33 columns, the exact grid,
finite values, source/experiment/Manifest hashes, each quadrature target,
convergence, the original strict fixture and all value/control budgets.
The largest error/budget ratio is `0.001884`. The weakest sign/factor/order
control exceeds `6.09e7` times the unchanged common budget. The generated
CSV is an ignored component artifact, not manuscript data or a reference
cache.

## Moderate real transition derivatives

A 159-point native grid covers `[16,35)` in 0.125 increments, 16 and its
neighbors, the exact g1 x0, 20 and nextfloat(20), a central point through
prevfloat(35), and 35 neighbors. Independent convergent high-precision erf
series give F and its derivatives without using the production fraction,
binary64 erfcx or binary64 ODE. Fixed-depth 32/64/96 comparisons separately
check the finite fraction; depth 64 and 96 results agree bitwise on this
grid.

| Quantity within the new interval | Maximum relative error |
|---|---:|
| Public/bundle/AD first derivative | `6.62e-16` |
| AD second derivative | `1.19e-15` |
| Nested AD third derivative | `1.52e-15` |
| Bundle ODE second derivative | `9.97e-14` |
| Bundle ODE third derivative | `1.15e-12` |

The bundle's higher derivatives retain their ODE recurrences and original
budgets. Neither the table nor the new branch establishes uniform full
precision for all derivative orders, scalar types or complex arguments.
Below 16 and at/above 35, the original branches and oracle budgets remain.
A native before/after probe confirms exactly identical type/value/error
behavior at x=20 for Float32, integer, rational, BigFloat rejection,
positive-real complex and passive-complex inputs. Non-Float64 real inputs
retain the previous ODE evaluation below the asymptotic crossover.

The finite-seed regression covers 48 ordinary/nested Dual calls with NaN
or Inf seeds at x=15,20,35,100. Every call now requires DomainError; the
pre-fix probe rejected only 12. Two additional assertions preserve F's
positive-infinity value and zero-derivative endpoint with a finite seed.

## Residual fixture provenance and causal controls

Native Julia versions produced identical input bits and pre-fix g1
intermediates. F′/delta-B relative error was `1.773e-12`, with residual error
`8.28955e-14`. Substituting only an independent delta-B reduced the final
error to `2.34791e-15`, isolating the arithmetic amplification. The old g1
literal itself was `8.30674e-14` from the independent exact-input value.

A completed intermediate full run passed 30,226 assertions and failed only
the g2 old expectation at its unchanged `2e-14` threshold. The old literal
was `2.738e-14` from truth; corrected production was `1.324e-15` away.
Separate decaying real-axis integrals used observed native binary64 inputs:
their 80/120-target precision agreements were approximately `5.93e-110`
(g1) and `1.50e-114` (g2), with agreement against the separate Julia series
oracle of `7.15e-119` and `5.86e-123` relative.

Both corrected residual literals equal ComplexF64 of their 120-digit
reference. Exact bits, precision agreement, correct rounding and rejection
of the old literals are tested. All other legacy scalar residual,
coefficient, score, symmetric and polygon expectations were audited and
remain unchanged within their original gates. Final full-suite results
above include these corrections. An earlier deliberately superseded run
is excluded from completed-check totals.
