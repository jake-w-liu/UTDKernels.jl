# Reproduce the diffraction studies

This optional environment runs the complete experiment definitions with either
UTDKernels scalar models or DiffUTD spatial fields. The production package does
not depend on this environment's plotting and artifact tools.

Use Julia 1.12 or newer. Place the two checkouts beside one another, named
`UTDKernels.jl` and `DiffUTD.jl`; access to DiffUTD is required even for the
kernel run because provenance records both packages. From the UTDKernels root:

```sh
julia --startup-file=no --project=reproduction -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --project=reproduction reproduction/run.jl --study all --backend kernels
julia --startup-file=no --project=reproduction reproduction/run.jl --study all --backend diffutd
```

Select an explicit installed Juliaup channel when comparing runtimes. For
unregistered local dependency changes, `study_entry.jl` develops both checkouts
before running the same runner. Its `CentralStudyEntry.central_study_main`
entry accepts a study slug, output/manuscript environment directory and the
parent directory containing both checkouts.

| Study | Datasets | PDF assets | Tables | Coverage |
|---|---:|---:|---:|---|
| `face-grazing` | 17 | 6 | 6 | [Coefficients, evaluators, cost and sensitivity](studies/face-grazing/README.md) |
| `face-edge` | 20 | 16 | 6 | [Decomposition, selection and polygon fields](studies/face-edge/README.md) |
| `curvature-measure` | 18 | 12 | 3 | [Seam quadrature and cylindrical monostatic fields](studies/curvature-measure/README.md) |
| `null-uniform` | 11 | 8 | 6 | [Moment hierarchies and half-plane fields](studies/null-uniform/README.md) |
| `shadow-sensitivity` | 16 | 8 | 4 | [Pullbacks and frozen receiver objectives](studies/shadow-sensitivity/README.md) |

Coverage records distinguish generated numerical tables from analytical and
experiment-design tables. Physical groups use Cartesian fields and geometry
in DiffUTD. Arbitrary complex-pole and spectral groups retain their scalar
mathematical scope.

## Select a run

- `--study SLUG` selects one listed study; `--study all` selects all five.
- `--backend kernels|diffutd` selects the production backend.
- `--output /absolute/path` overrides the output directory. With all studies,
  outputs are grouped under `/absolute/path/<slug>/`.
- `--no-plots` checks data and macros without claiming PDF coverage.
- `--stage SCRIPT.jl` selects one declared stage for a single study and writes
  a single-stage receipt rather than a complete-run receipt.

The default output is `reproduction/results/<backend>/<slug>/`, containing
`data/`, `figs/`, macro files and `reproduction-logs/`. Direct stage invocation
uses `UTD_REPRO_BACKEND=kernels|diffutd` and an absolute `UTD_REPRO_OUTPUT`;
the runner sets both for its children.

## Results and receipts

Complete runs begin in an empty staging directory. The runner checks the
coverage record's CSV names, columns and row counts, permitted diagnostic
nonfinite values, matching macro copies and PDF headers before publication.
Independent references and invariants in the drivers check numerical behavior;
artifact inventory alone is not a scientific validation.

Receipts retain the `study` slug, actual runtime manifest, both source paths,
versions, commits and implementation hashes, stage logs and artifact hashes.
They also record the Julia and package versions in generated manuscript
macros. A failure retains staged logs and a failed receipt. Source mutation
fails the run. Publication preserves unrelated files in the destination.

## Development checks

```sh
julia --startup-file=no --project=reproduction reproduction/test/runtests.jl
```

The [development harness guide](DEVELOPMENT_HARNESS.md) describes full-package,
source-binding, launcher and receipt checks. Generated results and local
resolver manifests are ignored by Git.
