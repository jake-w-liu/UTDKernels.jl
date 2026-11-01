# Development harness

The harness requires complete package tests and complete study reproduction
through both backends. It can reuse evidence only when source files, resolved
dependencies, runtime options, logs and artifacts match.

Use an adjacent pair of checkouts, and select the Julia runtime explicitly when
comparing the supported baseline with the latest stable release. From the
UTDKernels checkout:

```sh
julia --startup-file=no --threads=4 --check-bounds=yes --project=reproduction \
  reproduction/development_harness.jl --project /path/to/manuscript-project \
  --codes /path/to/checkouts
```

A manuscript project has its environment under `paper/Project.toml` and thin
launchers under `paper/scripts/`. `CentralStudyEntry.study_launcher_sources`
defines that launcher contract using a feature slug. The launcher environment
variable `UTD_REPRO_CODES_ROOT` selects the directory containing both checkouts.
The harness verifies the complete launcher inventory and both package bindings;
archived implementations cannot qualify as active sources.

Full package tests use four threads and forced bounds checks. Experiments use
one thread and Julia's normal bounds policy. Both profiles retain the executable
hash, build and compiler options. Failed or incomplete commands fail the harness
and retain their logs.

## Full package gate

```sh
julia --startup-file=no --threads=4 --check-bounds=yes --project=reproduction \
  reproduction/full_test_gate.jl UTDKernels /path/to/checkouts
```

Replace `UTDKernels` with `DiffUTD` for the other package. The observer captures
the actual `Pkg.test` temporary Project and Manifest at entry and exit.
Dependency precompilation cannot supply test-process observations.

## Receipts and reuse

Receipts live under `reproduction/results/development-harness/` by default.
`--receipts DIR` selects another shared cache. A directory lock prevents
concurrent duplicate full tests; the harness never signals processes or removes
another invocation's lock.

`--reuse-only` checks completed evidence without launching missing tests or
experiments. It fails when matching full-package or complete reproduction
receipts are absent. Single-stage receipts do not qualify as complete runs.

`--outputs DIR` checks study outputs under `DIR/<backend>/<slug>/`. Otherwise,
the kernel output is the manuscript's `paper/` directory and the spatial output
is `paper/validation/diffutd/`. Reproduction reuse checks every declared dataset,
figure, macro, numerical receipt and stage log against its recorded hash.

## Contract tests

These tests exercise rejected source, launcher, receipt and observer mutations
without running the scientific experiments:

```sh
julia --startup-file=no --check-bounds=yes --threads=4 --project=reproduction \
  -e 'for f in ("test_harness_receipts.jl", "test_harness_entries.jl", "test_harness_reproduction.jl", "test_harness_observer.jl"); include(joinpath("reproduction",f)); end'
```

The precompile regression runs a real `Pkg.test` on local fixture packages:

```sh
julia --startup-file=no --project=reproduction reproduction/test_harness_precompile.jl
```

Its receipts are fixture evidence. Full production-package evidence comes from
`full_test_gate.jl`.
