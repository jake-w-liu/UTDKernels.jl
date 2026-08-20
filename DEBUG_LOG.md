# Debug Log

## 2026-08-20 — Holm incident weights were omitted

Symptom: `impedance_wedge_DsDh` differed from the cited Holm four-term
coefficient by 25.2% (soft) and 9.75% (hard) in a finite-impedance probe below
the incident-angle midpoint. A second-half probe also differed for both
polarizations. Existing PEC-limit tests did not expose the defect because the
missing incident weights approach one in that limit.

Root cause: both public overloads assembled the incident terms as `c1 + c2`.
Holm instead assigns the piecewise weights `W_n` and `W_0`, one of which is the
product of the two face reflection coefficients. The standard single-distance
coefficient also specifies `G=1/2` at exact face-grazing incidence.

Fix: added the polarization-specific product weights to both overloads and the
exact-point grazing factor to the standard single-distance overload. The
separate-distance extension keeps its one-sided continuous value at exact face
grazing because applying the isolated standard factor with unequal distances
introduces a jump. Updated the independent transcription tests, matched-medium
expectations, source documentation, and generated documentation.

Reverification correction: the first edit also applied `G=1/2` to the
separate-distance overload. The full suite then reported four failures in the
existing exact-grazing derivative check, with finite differences dominated by
the introduced jump. Restricting the factor to the published single-distance
coefficient restored the continuous generalized extension.

Files modified:

- `src/wedge/WedgeImpedance.jl`
- `test/test_wedge_impedance.jl`
- `docs/src/index.md`
- `docs/src/api.md`
- `docs/src/tutorial/impedance.md`
- generated files under `docs/build/`

Verification:

- Independent pre-fix probes demonstrated nonzero formula errors in both
  incident-angle halves.
- The corrected implementation matched the weighted oracle exactly in the
  same probes.
- `Pkg.test()` passed all 11,578 tests after the correction.
- `julia --project=docs --startup-file=no --color=no docs/make.jl` completed
  with no Documenter warnings or errors.
- Strict prose-firewall scans reported zero findings in the edited source
  documentation.

Lesson: a PEC-limit oracle cannot detect finite-impedance weights that collapse
to unity. Formula-transcription tests must use finite, asymmetric face
materials and exercise both sides of every piecewise angular definition.

## 2026-08-20 — Maliuzhinets recurrence could stall indefinitely

Symptom: `psi_Phi(2.0, nextfloat(0.0))` did not terminate. At `Float64`
precision, subtracting `4Phi` from the working argument produced the same
floating-point value on every recurrence step.

Root cause: the convergence-strip recurrence used an unbounded `while` loop
without checking that `wc - 4Phi` changed `wc`. For sufficiently small positive
`Phi`, the loop condition remained true while the recurrence made no numerical
progress.

Fix: compute the next recurrence argument before storing its cotangent factor,
and throw `DomainError` if the step cannot change the working value. Added a
regression test for the smallest positive `Float64` half-angle.

Files modified:

- `src/maliuzhinets/MaliuzhinetsFunction.jl`
- `test/test_robustness.jl`

Verification:

- A pre-fix arithmetic probe confirmed that the loop condition was true and
  `wc - 4Phi == wc` for the failing input.
- The corrected function throws `DomainError` for that input and still returns
  the prior finite value for a normal in-strip probe.
- `Pkg.test()` passed all 11,579 tests.

Lesson: every floating-point recurrence needs an explicit progress invariant;
mathematical positivity does not guarantee a representable numerical step.

## 2026-08-20 — Seam tolerance broke angular periodicity

Symptom: shifting an angle by one wedge period changed the PEC coefficient when
the wrapped angle was within the transition tolerance of zero. For an incident
offset of `1e-9` rad on a `3pi/2` wedge, the hard-coefficient difference between
`phip=delta` and `phip=alpha+delta` was `0.6304634230136278`. The same defect
affected observation angles and the automatic grazing router.

Root cause: the exact raw `alpha` seam was disambiguated with an absolute
tolerance. Consequently, `alpha+delta` was classified as the n-face while its
periodic counterpart `delta` was classified as the o-face. The duplicated
grazing-router logic used the same tolerance test.

Fix: added a dependency-free primal-value zero predicate and limited raw-seam
disambiguation to an angle whose primal value is exactly `alpha`. Applied the
same predicate to the four-term evaluator and the grazing router. Added
behavioral periodicity tests for observation angles, incident angles, and
automatic routing; the existing exact-seam AD tests remain unchanged.

Files modified:

- `src/common/Numerics.jl`
- `src/wedge/WedgePEC.jl`
- `src/wedge/WedgeGrazing.jl`
- `test/test_symmetry.jl`
- `docs/src/tutorial/kp_coefficients.md`
- generated files under `docs/build/`

Verification:

- Pre-fix probes showed `O(1)` hard-coefficient differences for both source and
  observation angles below the `1.4901161193847656e-8` seam tolerance.
- Post-fix source coefficients were identical in the same probe; observation
  coefficients differed by at most `1.97e-17` from wrapping roundoff.
- The focused symmetry suite passed all 22 tests.
- `Pkg.test()` passed all 11,587 tests, including the exact n-face ForwardDiff
  regression suite.

Lesson: a tolerance may detect numerical proximity, but it cannot establish
discrete seam identity when nearby periodic values must remain distinct.

## 2026-08-20 — WDC validation reported exclusions as passes and exited cleanly on failure

Symptom: a one-row exact-boundary fixture returned
`(passed=1, failed=0, skipped=0)` even though that category was described as
excluded. The command-line entry point also printed nonzero failure counts
without returning a nonzero exit code. Malformed CSV rows were silently
discarded, and the pass criterion covered only the incident component `Di`.

Root cause: the category condition combined `isinf(tol)` with the pass branch,
the script did not call `exit` with its result, and the CSV loader continued past
wrong-width rows. The component oracle omitted the reflected term even though
the reference file contains it independently.

Fix: count excluded regimes as skipped, validate the CSV header, row width,
numeric fields, finiteness, and nonempty payload, and return an enforced CLI
exit code. Compare `Di` and `Dr`, reconstructing the latter from both `Ds` and
`Dh` so either polarization can expose a regression. Enforce the
transition-table tolerance and add mutation-sensitive fixture and subprocess
tests. Declare `Printf` as a test-only standard-library dependency.

Files modified:

- `Project.toml`
- `validation/compare_wdc.jl`
- `validation/README.md`
- `test/runtests.jl`
- `test/test_validation_harness.jl`
- `docs/src/tutorial/validation.md`
- generated documentation under `docs/build/`

Verification:

- The focused validation-harness suite passed all 14 tests, including actual
  process exit codes `0` for a reference row and `1` for a corrupted reflected
  component.
- The complete 54,320-row WDC dataset reported 45,855 tested and passed, zero
  failed, and 8,465 skipped; the excluded cases no longer contribute to passes.
- Reconstructing the reflected component from the hard coefficient introduced
  no failures; its largest included-category error was `1.5506497287336935e-2`.
- The transition-table maximum relative error was `8.58e-3`, below the enforced
  `1e-2` tolerance.
- `Pkg.test()` passed all 11,601 tests in its isolated test environment.

Lesson: a validation report must fail closed at input, oracle, accounting, and
command-exit boundaries; printed warnings alone do not protect automation.

## 2026-08-20 — Integer Maliuzhinets arguments failed during strip reduction

Symptom: `psi_Phi(10, 2)` and `psi_Phi(10 + 2im, 2)` raised `InexactError`
when the functional recurrence reduced the argument into its convergence strip.

Root cause: `Complex(w)` preserved the integer component type, and the recurrence
allocated `cot_factors` as `Complex{Int}[]`. Its generally noninteger cotangent
factor could not be stored in that vector.

Fix: promote the argument and half-angle to a floating computation type before
allocating recurrence storage. Apply the same normalization to direct internal
strip-integral calls, and cover real and complex integer arguments in the
Maliuzhinets tests.

Verification:

- The focused Maliuzhinets suites passed all 65 tests, including four new
  type-and-value checks.
- Integer and floating calls returned identical values for both probes.
- A ForwardDiff derivative agreed with a centered finite difference to
  `2.64e-11` absolute error.
- `Pkg.test()` passed all 11,605 tests.

Lesson: parametric recurrence storage must use the computation type after
promotion, not the literal input type.
