# Paper 170 reproduction contract

`generate.jl` regenerates all 16 CSVs and `results_macros.tex`; `plot.jl`
regenerates all eight figure PDFs. The shared root runner snapshots both
central sources and the environment and runs generation and plotting in
fresh processes. `UTD_PAPER_OUTPUT` is the absolute paper output directory.

Run from the UTDKernels checkout:

```sh
julia +1.13.3 --startup-file=no --threads=1 --check-bounds=auto --project=reproduction reproduction/run.jl --paper 170 --backend kernels
julia +1.13.3 --startup-file=no --threads=1 --check-bounds=auto --project=reproduction reproduction/run.jl --paper 170 --backend diffutd
```

| Evidence | Complete CSV output | Acceptance reference or fit |
| --- | --- | --- |
| 256 seeded bandlimited probes, seven scales, three bounds | bandlimited_probe_errors, bandlimited_summary, bandlimit_bound_effectivity | Exact Fourier multiplier; same 640-node spectra for every comparator; all seven OLS samples |
| Exact half-plane coordinate | halfplane_pullback | Phase-adapted quadrature; four largest scales fitted |
| Nonlinear receiver coordinate | curved_coordinate | Complete map jets versus width-only correction; four largest scales fitted |
| 336 derivative comparisons | derivative_validation | Five-point differences of independent 256-bit local Fresnel integrals |
| Oscillatory point-sampling burden | sampling_burden | First sampled grid below absolute 1e-6, no claim of a proven minimum |
| Translated Gaussian objective | translated_objective | Exact complex Gaussian action and multiplier; 241 bandwidths |
| Switch, multiplier and sensitivity profiles | canonical_switch_profiles, multiplier_phase, sensitivity_profiles | Central canonical functions; physical 3D branch derivatives for the spatial half-plane profiles |
| Linear and nonlinear transformed-test bounds | transformed_bound, transformed_bound_nonlinear | Independent spectral actions/moments and explicit cutoff/refinement protocol |
| Carrier-restoration diagnostic | carrier_uniformity | Exact carrier cancellation and explicit finite-order constants |
| Complete physical Sommerfeld field/objective | sommerfeld_benchmark, sommerfeld_offset_scan | Nine scales, four largest fitted; frozen receiver window; independent finite difference of the misfit objective; 57 offsets |

The four manuscript tables map to the transfer multipliers (central
canonical formulas), spectral-probe settings (`bandlimited_study`),
coordinate/probe settings (`halfplane_study`, `curved_coordinate_study`), and
predicted/fitted orders (`bandlimited_summary` and generated macros).

Kernel backend: `UTDKernels.ShadowSensitivity` owns complete production
families, and the parent module owns canonical switches/multipliers and
pullbacks. Spatial backend: `DiffUTD.ShadowSensitivityStudy` realizes the
half-plane and curved coordinates on world receiver curves; derives map
jets through `receiver_shadow_pullback`; projects the complete Cartesian
E/H solution; retains the incident demodulating carrier; and evaluates
fixed-world-window exact, transition, smooth-remainder and local-truncation
actions. Generic `receiver_*` APIs additionally handle oblique incidence,
variable electrical scale, transported receiving bases and full-vector
objectives. Abstract spectral tests share canonical primitives; no arbitrary
Fourier probe is mislabeled as a physical 3D scene.

For the physical objective the center is frozen at `p0+π`; it does not follow
`p` in a derivative. Local approximations differentiate their complete
effective tests, including the actual field factor in the misfit. The code
preserves failures of nonuniform carrier-restored truncations and the
finite-range qualification on misfit fits.

## Nonlinear moment controls

For case3 of `transformed_bound.csv`, the legacy `bound_physical_nd` and `bound_canonical_nd` columns are finite-band Fourier-moment proxies. They do not bound the omitted Fourier tails. `data/spectral_norm_controls.toml` separately records the analytic whole-line Sobolev inequality, independently evaluated derivative norms, quadrature/precision diagnostics, exact scale identity, and observed action margins. Its numerical norm constants are not interval enclosures.

The auxiliary `carrier_uniformity.csv` uses the unit canonical box on |y| <= 0.5 (integral one). The manuscript's smooth flat-top example is an analytic counterexample for any nonzero test integral; that smooth example is not relabeled as the box quadrature.
