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
