# Reproduction of studies 160, 162, 165, 168, and 170

This optional environment shares complete experiment definitions between
UTDKernels and DiffUTD. Production kernels do not depend on the plotting or
artifact-validation packages used here. Julia 1.12 is the minimum supported
runtime; use an explicit latest-stable channel for current reproduction.

The two central checkouts must be adjacent and named `UTDKernels.jl` and
`DiffUTD.jl`. From the UTDKernels root:

```sh
julia +1.13.3 --startup-file=no --project=reproduction -e 'using Pkg; Pkg.instantiate()'
julia +1.13.3 --startup-file=no --project=reproduction reproduction/run.jl --paper all --backend kernels
julia +1.13.3 --startup-file=no --project=reproduction reproduction/run.jl --paper all --backend diffutd
```

For unpublished dependency changes, `paper_entry.jl` develops both checkouts
together. The active manuscripts' `paper/scripts/reproduce.jl` entries perform
this bootstrap automatically and run both backends by default. An external
manuscript source bundle can set `EMPIRE_CODES_ROOT` to the directory containing
the two checkouts.

| Study | Datasets | Referenced PDF assets | Manuscript tables | Coverage |
|---|---:|---:|---:|---|
| 160 | 17 | 6 | 8 | [Grazing coefficients, evaluators, cost, sensitivity, and spatial validation](papers/160/README.md) |
| 162 | 20 | 16 | 6 | [Face–edge decomposition, selection, and polygon fields](papers/162/README.md) |
| 165 | 18 | 12 | 3 | [Curvature sums, smooth limits, and cylindrical monostatic seams](papers/165/README.md) |
| 168 | 11 | 8 | 6 | [Null-uniform moments and half-plane fields](papers/168/README.md) |
| 170 | 16 | 8 | 4 | [Distributional shadow sensitivity and frozen receiver objectives](papers/170/README.md) |

These are inventory counts, not a claim that every table is a generated file.
Per-study coverage records identify data, mathematical identities, fit windows,
and the tests supporting each result. Physical groups use Cartesian fields and
geometry in DiffUTD. Abstract complex-pole or spectral groups use shared scalar
kernels and retain that mathematical scope.

`--paper ID` selects one study. `--output /absolute/path` selects an output
directory; for multiple studies it contains `ID/paper/` subdirectories.
`--no-plots` validates data and macros without claiming PDF coverage.
`--stage SCRIPT.jl` selects a listed stage for one study and produces a
single-stage receipt rather than a complete-reproduction receipt.

Complete runs start with an empty staging directory. The runner checks every
declared CSV's name, columns and row count, rejects unexpected nonfinite values,
verifies copied macros and PDF headers, and then publishes the files while
preserving unrelated manuscript files. Numerical correctness is checked by the
independent references and invariants in each driver; the artifact inventory
does not replace those checks. A failure retains the staged logs and writes a
failed receipt. Child process diagnostics record the stage, exit status and
signal without serializing the inherited environment.

Each complete receipt records the selected runtime manifest, both source paths,
versions, commits and implementation hashes, stage logs, and artifact hashes.
Complete runs also append the actual Julia and package versions, backend and
thread count to the manuscript macros. This step checks that the generated
macro copies agree before updating them.
Source mutation during a run fails the run. Generated results and local
resolver manifests are ignored by Git.

Run the orchestration contract tests with:

```sh
julia +1.13.3 --startup-file=no --project=reproduction reproduction/test/runtests.jl
```
