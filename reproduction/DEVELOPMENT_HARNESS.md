# Active-paper development audit

Run from an active project with both central checkouts available:

```sh
bash .agents/scripts/dev-harness-audit.sh .
```

The dispatcher selects Julia 1.13.3 explicitly. `DEV_HARNESS_JULIA_VERSION=1.12`
selects the supported baseline channel without changing the default Julia.
`EMPIRE_CODES_ROOT` points to the directory containing `UTDKernels.jl` and
`DiffUTD.jl`; paths beginning with `~/` are expanded.

The audit checks the paper's complete thin-launcher inventory and both central
source bindings. It then requires full `Pkg.test` gates for both production
packages, plus complete data and plot runs through both paper backends.
Archived implementations cannot supply active evidence. Failed or incomplete
commands fail the audit and retain their logs.

Full tests use four threads and forced bounds checks. Paper experiments use
one thread and Julia's normal bounds policy, including the timing studies.
Both profiles record the selected executable hash, build and compiler options.

Receipts and logs live under `reproduction/results/development-harness/` by
default. Five paper audits share completed central package tests through this
directory. A receipt qualifies only when production/test sources (including
non-Julia fixtures and validation helpers), actual temporary test dependencies,
runtime, full-tier selector, source boundaries, completion marker, and log
hashes match. Otherwise the default audit runs a fresh full test. A shared
directory lock prevents duplicate concurrent tests; it never signals processes
or removes another invocation's lock. Test-only receipt fixtures do not qualify
as actual package-test evidence.

To run the full gate directly, under the central reproduction environment:

```sh
julia +1.13.3 --startup-file=no --threads=4 --check-bounds=yes \
  --project=reproduction reproduction/full_test_gate.jl DiffUTD /path/to/codes
```

Use `UTDKernels` for the other owner. The observer captures the actual
`Pkg.test` temporary Project/Manifest at entry and exit. Historical logs lacking
these observations are retained as historical evidence and cannot be promoted
to reusable receipts.

For a read-only check of completed evidence, append `--reuse-only`. It fails
when a matching full-test or reproduction receipt is absent. `--receipts DIR`
selects another shared audit/test cache. `--outputs DIR` checks complete paper
runs under `DIR/<backend>/<paper>/paper`; otherwise it checks the active paper
directory and `paper/validation/diffutd`. Full reproduction reuse requires every
declared dataset, figure, macro, additional numerical receipt, stage log and
their content hashes to match the current source and environment.

Isolated mutation tests (no full package tests or paper experiments):

```sh
julia +1.13.3 --startup-file=no --check-bounds=yes --threads=4 \
  --project=reproduction -e 'for f in ("test_harness_receipts.jl", "test_harness_entries.jl", "test_harness_reproduction.jl", "test_harness_observer.jl"); include(joinpath("reproduction",f)); end'
```

The precompile integration regression runs a real `Pkg.test` on fresh local
fixture packages. It proves that dependency image workers leave `Main` and the
observer nonce untouched, while the actual test process supplies both observed
PIDs. Its complete evidence is retained under
`reproduction/results/harness-precompile-controls/`.

```sh
julia +1.13.3 --startup-file=no --project=reproduction \
  reproduction/test_harness_precompile.jl
```

Full production-package evidence still comes from `full_test_gate.jl`.
