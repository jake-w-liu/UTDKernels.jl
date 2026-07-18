# Documentation Verification Report

Date: 2026-07-18

## Result

The documentation build exits successfully with doctests enabled and all
exported bindings covered. All executable textbook examples agree with their
independent equation reconstructions. The rich plot outputs are embedded inline
under explicit 512 KiB warning and 1 MiB failure limits; the largest measured
tutorial page is 367.05 KiB. The final Documenter build emits no warnings or
errors. The plotted content, dimensions, and layout are unchanged.

## Checks

- `julia --startup-file=no --project=docs docs/make.jl`: passed with
  `doctest=true`, `checkdocs=:exports`, and `warnonly=false`; emitted no warning
  or error.
- `julia --startup-file=no --project=examples -e
  'include("examples/run_all.jl"); run_all_examples(save_png=false)'`: passed.
  No figure files were rewritten.
- Maximum textbook-example relative discrepancies:
  - Example 13-3: `3.2783347249556617e-13` (soft),
    `6.0179891194571895e-16` (hard).
  - Example 13-4: `2.576629622246064e-15`.
  - Example 13-5: `5.4937103029836513e-14` and
    `5.404365112666579e-14`.
  - Example 13-6: `3.0146630170193015e-15`.
  - Example 13-7: `1.6476815913570673e-15`.
- `julia --startup-file=no --project=. validation/generate_data.jl`: passed and
  regenerated all ignored validation tables without producing a quantitative
  figure.
- The broad Balanis `WDC.m` comparison used a fixture regenerated during this
  audit with MATLAB R2025b from `validation/generate_wdc_reference.m` (SHA-256
  `8d6eb8858f364686764fb567361554a08343af08a8ff3a1338d08e3737fae3ba`).
  The regenerated 54,320-row file is byte-for-byte identical to the historical
  MATLAB fixture from repository commit `bd1516e`.
  It tested 48,023 nondegenerate cases: 48,023 passed and 0 failed; 6,297
  zero-reference flat-plane rows were skipped as defined by the validator.

## External-tool note

MATLAB R2025b was found at `/Applications/MATLAB_R2025b.app/bin/matlab` and was
used directly for the reference regeneration. No alternate implementation was
substituted.

Docs complete: 1 issue detected → 1 confirmed → 1 fixed, 0 require user action.
