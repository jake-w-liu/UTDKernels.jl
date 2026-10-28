# Paper 168 reproduction contract

`generate.jl` regenerates all 11 CSVs and `results_macros.tex`; `plot.jl`
regenerates all eight PDF assets used by the four manuscript figures.
Use the root `reproduction/run.jl` with paper 168 and either backend. The
shared runner verifies both central source paths and snapshots the source
and environment hashes. `UTD_PAPER_OUTPUT` is an absolute **paper directory**;
its outputs are `data/`, `figs/`, and `results_macros.tex`.

Run from the UTDKernels checkout:

```sh
julia +1.13.3 --startup-file=no --threads=1 --check-bounds=auto --project=reproduction reproduction/run.jl --paper 168 --backend kernels
julia +1.13.3 --startup-file=no --threads=1 --check-bounds=auto --project=reproduction reproduction/run.jl --paper 168 --backend diffutd
```

| Evidence | Complete data output | Reference and interpretation |
| --- | --- | --- |
| Joint null layer, 181×161 grid and 161-point section | null_layer_map, null_layer_section | Exactly factored quartic amplitude; all retained moment and classical comparators; local term-scale normalization |
| Four truncation slopes and three exact-zero slopes | convergence, zero_order_scaling | All 30 log-spaced scales in the OLS fit; distinct field and normalized-error orders |
| Seeded 320-case amplitude ensemble | random_quadrature | Independent 2000-point Gauss–Hermite rule |
| 36 direct moment integrals | direct_integral | Independent real-line QuadGK integration |
| Four moment profiles | moment_profile | Canonical identities and outer orders and polynomial completions |
| 3000 stability comparisons | stability | 384-bit oracle, five sheets including Stokes continuation |
| 64 nonpolynomial amplitudes | nonpolynomial_benchmark | Completed-square closed form versus quadrature, Taylor orders 0:5 |
| 160-angle physical PEC sweep | wedge_halfplane | Eigenfunction minus GO reference, independent steepest-descent contour; one/two/four moments and both classical baselines |
| Reflection, null, shadow diagnostics | wedge_diagnostics | Symmetric PV and half-weight GO; complete total-field residual |

The six manuscript tables map to: canonical moment structure (identities
and outer orders in the central moment kernels); map statistics
(`null_layer_map`); all-scale fitted orders (`convergence`,
`zero_order_scaling`); boundary diagnostics (`wedge_diagnostics`); and
independent-reference maxima (random, direct, stability, nonpolynomial,
and wedge data). Numeric cells are emitted by `result_macros`; structural
identities are verified by the canonical moment test suite.

Kernel backend: `UTDKernels.NullUniform` is the production owner.
Spatial backend: `DiffUTD.NullUniformStudy` uses shared canonical primitives
and independent references for arbitrary complex canonical ensembles, and
actual `HalfPlaneWave` geometry, transported amplitudes and full Cartesian
E/H reconstruction for the real PEC half-plane family. Complex canonical
inputs are not assigned fabricated real geometries. The two mapped saddles
and all boundary residues remain visible in the diagnostics.

The figures are the four joint-layer maps, joint-layer section, convergence,
stability, and wedge assets. Plotting reads only persisted CSVs. There is no
archived research-package import or global plotting-environment dependency.

`tab:canonical-settings` records the generator grids/distributions and reference settings for the null map, convergence, randomized/direct moment checks, and stability sweep; it is a supporting reproducibility table.
