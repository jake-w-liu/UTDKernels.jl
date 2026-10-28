# Wedge transition and derivative corrections

## Numerical contract

For finite wavenumber k, effective distance L and signed detuning delta, the
local transition product is cot(delta) F(2 k L sin(n delta)^2). It has finite
one-sided values at delta=0. Preserve the existing positive-side convention at
exactly zero, its one-sided derivatives and the separate divergent L=Inf
contract. Public PEC and Holm impedance coefficients must remain finite when
the final result is representable, including when an unscaled intermediate is
not. Preserve the diffraction formulation and public API.

For positive delta, analytical cancellation gives

    n sqrt(pi) (1+i) sqrt(kL) cos(delta)
      sinc(n delta)/sinc(delta)
      erfcx((1+i) sqrt(kL) sin(n delta)).

Here sinc(t)=sin(t)/t. The negative side negates the prefactor and erfcx
argument. At zero the first derivative is -4i n^2 kL, and the second is the
boundary value times [4i n^2 kL-(n^2+2)/3]. Multiplication by the PEC prefactor
reduces the exact positive-side coefficient to -sqrt(L)/2 for positive real k,L.

## Confirmed defects and corrections

1. Registered 0.4.0 replaces zero detuning by sqrt(eps). The resulting scaled
   transition coordinate depends on kL and is not the exact boundary value.
   Of 318 new independent checks, 186 fail before correction. Analytical
   cancellation preserves exact and subnormal detunings. Scaled square-root
   products retain representable results when kL itself overflows/underflows.
2. At k=L=1e308, the unscaled transition amplitude overflows while the public
   single-pole coefficient -5e153 is representable. Applying the common PEC
   prefactor before that amplitude corrects PEC and impedance evaluation.
   Four of twelve new public-scale assertions error before the correction.
3. Differentiating the real erfcx product loses small derivatives at large
   arguments. At x=1e8, the registered derivative's relative error is 5.436
   against an independent high-precision oracle. Ordinary wedge terms and
   the local cotangent representation have the same cancellation. They now
   use the transition API, whose ForwardDiff extension propagates the stable analytic F_prime
   above its established positive-real threshold. Its inverse-power series
   retains small large-argument derivatives and nested sensitivities. Passive complex
   arguments use the existing sector-wide backend. Normal-product guards
   retain the scaled-root path when a product underflows.

Ordinary real value evaluation keeps the erfcx implementation. Evaluating the
full derivative bundle for these calls had a measured 3–4x cost in small/large
argument sweeps and was removed before release. The replacement benchmark has
zero allocations and the original value evaluator's cost. No fitted field,
smoothing envelope or geometry-specific correction is introduced.

## Independent expectations and regression scope

The added tests use rotated-contour quadrature, high-precision transition and
wedge evaluations, exact one-sided limits/derivatives and the public
half-incident-field contribution. They cover both sides, zero and subnormal
detunings, extreme product scales, real and passive-complex arguments, first
and second automatic derivatives, and PEC/impedance public paths. Four older
complex boundary snapshots contained the artificial-offset error; independent
384/512-bit calculations established their correction at the original 5e-14
relative tolerance. The high-precision oracle's iteration limit was increased
without changing its convergence tolerance.

The final focused run passed 13,602 assertions, including 107 transition and
wedge sensitivity checks. The same 107 checks pass with ForwardDiff 0.10.39;
the full suite uses ForwardDiff 1.4.6. A separate 12-case Float16/Float32/Float64
comparison found no increased derivative error. Four exploratory checks that
incorrectly imposed a Float64 tolerance on low-precision paths reproduce
byte-identical values in 0.4.0 and 0.4.1; they are not release regressions.

All six validation/example stages and the article figure reproduction pass.
The WDC comparison passes all 45,855 samples selected by its existing criteria
(8,465 excluded as before). Its immutable reference SHA256 remains
8d6eb8858f364686764fb567361554a08343af08a8ff3a1338d08e3737fae3ba.
The 31 CSV comparisons preserve schemas and finite-value patterns. Ordinary
wedge field changes are at most 3.34e-13; half-plane fields change by at most
4.44e-16. The exact-boundary correction changes the corresponding coefficient
samples by at most 5.991e-8. Fixed-step finite-difference diagnostic ratios are
not treated as derivative oracles: at their three worst tail samples,
independent 384-bit calculations bound the new saved AD relative error by
1.1e-12, versus up to 7.9e-5 before correction.

All eight validation PDFs, seven article PDFs and five example figures were
regenerated and visually checked. Article figures use Julia and PlotlySupply.
The generic manuscript figure audit cannot process this figure-only article
reproduction directory because it has no TeX source; producer execution and
artifact inspection were verified directly. Documentation and doctests pass.
The final bounds-checked full package run passes all 16,210 assertions on
Julia 1.12.7 with ForwardDiff 1.4.6. Its runtime digest is unchanged before
and after testing:
70708c165d9206610ac9402de1f6a4e7babc58718769d217c426f72bd1203696.
Run the full suite with
`julia --check-bounds=yes --project=. -e 'using Pkg; Pkg.test()'`.
Run the dedicated validation, examples and article scripts using their
documented environments.

The first cross-platform CI run exposed a test-environment defect: the CLI
test selected the separate validation manifest, whose pinned dependencies
were not installed by Pkg.test. Linux, macOS and Windows each passed 16,209
assertions and failed the same valid-fixture subprocess check. The child now
inherits the active test environment. Both fixtures also check the CLI output,
so a package-loading error cannot pass the deliberately corrupted-data test.
The revised harness passes all 16 assertions locally. This change leaves the
numerical source and its runtime digest unchanged.


## 0.4.3 grazing-evaluator allocation correction (2026-10-28)

Version 0.4.2 introduced a nested fallback helper whose `ds` and `dh`
bindings also belonged to the outer evaluator. The resulting inferred return
was `Tuple{Any,Any,...}`. In a warmed Julia 1.12.7 comparison using the same
DiffUTD source, ordinary and near-grazing field calls allocated 768 and 1,248
bytes, respectively, while registered 0.4.0 allocated zero. The corresponding
kernel probes allocated 48 and 528 bytes. Distinct fallback-local bindings
restore concrete coefficient returns without changing numerical expressions,
error types, fallback conditions, or diagnostic traces.

The regression covers finite and infinite distances at exact grazing, a small
face offset, and an ordinary offset. All six type-inference checks and all six
zero-allocation checks fail or error under 0.4.2. The corrected local 0.4.3
three-dimensional probes allocate zero bytes and return the same field values
as 0.4.2. A separate ten-assertion check forces adaptive nonconvergence:
the default throws `GrazingDomainError`, the opt-in four-term fallback equals
the baseline, and diagnostics preserve the attempted orders, call count,
valid domain certificate, and nonconverged status.

The focused grazing suite passed 403/403 assertions. The full package run
passed 16,258/16,258 assertions on Julia 1.12.7 at four threads with bounds
checking enabled:
`JULIA_NUM_THREADS=4 julia +1.12 --startup-file=no --check-bounds=yes --project=. -e 'using Pkg; Pkg.test()'`.
The project, corrected source, and regression-test SHA-256 digests were unchanged
after the run. An independent diff review found no additional issue in this
change. These checks cover the correction and its existing package regressions;
they are not a proof for every possible input or a new throughput benchmark.
