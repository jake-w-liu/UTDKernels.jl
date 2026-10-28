# Paper 162 reproduction contract

Run `generate.jl`, then `plot.jl`, in the reproduction environment. The shared
`UTD_PAPER_BACKEND=kernels|diffutd` selects the central implementation;
`UTD_PAPER_OUTPUT` is an absolute paper output directory. No archived paper
implementation is imported. `coverage.json` enumerates all 20 CSV schemas and
16 figure assets. Both copies of `results_macros.tex` are generated from the
same measured datasets. The six manuscript tables cover independent ensembles,
accuracy quantiles, direct-evaluation wins, measured cost, limits, and failure
domains; the last two also require the mathematical/domain tests.

UTDKernels owns scalar transition/wedge mathematics, the compensated midpoint
split, polygon geometry/visibility, and independent high-precision and cylinder
references. The Diff backend embeds the study in Cartesian 3D geometry. It
derives ray-fixed angles, preserves the small face-vector turn separately from
the rounded exterior angle, and projects electric-soft and magnetic-hard
channels after physical transport. Geometry generation and reference oracles
are shared; production field calls enter DiffUTD.

| Dataset | Main acceptance obligation | Figure or table evidence |
|---|---|---|
| high_precision_accuracy | 600 seeded cases; compare compensated residual against the independent guarded-precision unrearranged pair, retain direct wins | Accuracy-quantile and ensemble tables |
| decomposition_ensemble | 1500 seeded cases; full soft/hard agreement and parity identities, with conditioning-aware denominators | Independent-ensemble table |
| coplanar_scaling | Three electrical lengths; linear residual, finite face limit; fit epsilon <=1e-4 | fig_coplanar_scaling |
| electrical_length | Four defects; low/high electrical-length regimes and declared windows | fig_electrical_length |
| asymmetry_sweep | Three defects; independently arranged first-order comparison, branch restriction | fig_asymmetry_sweep |
| polygon_refinement | 149 unique regular counts; fixed and geometry-scaled distance slopes and sampled threshold | fig_polygon_refinement, fig_polygon_distance_scaling |
| angle_factor | Monotonic geometric factor and endpoint limit | fig_angle_factor |
| condition_sweep | Inverse-defect conditioning and both actual numerical-error curves | fig_conditioning |
| accuracy_curve | Three asymmetries; compensated versus direct errors, preserve all direct wins | fig_accuracy_loss, direct-win table |
| diagnostic_map | Complete 120 by 140 grid; intrinsic score and raw/intrinsic ratio | fig_diagnostic_score_map, fig_diagnostic_ratio_map |
| counterexample | Soft and hard full/preserve/delete fields at the reflection-gap example | Text and failure-mode table |
| boundary_sweep | Both reflection switches, both polarizations; absolute unit-incident field errors | fig_boundary_transition, fig_gap_loss |
| path_limit | Three defects; finite boxcar limit of normalized reflection pair | fig_path_limit |
| incident_shadow_cex | General-angle incident pair, independent HP comparison, and full/reflection-only safeguard error | Text and failure-mode table |
| incident_shadow_sweep | Both polarizations; preserved GO compensation and omitted-field guard | fig_incident_shadow |
| polygon_field | All 720 complex-field comparisons; phase matched on actual facet planes, cylinder model discrepancy retained | fig_polygon_field |
| cylinder_refinement | Seven regular polygons and both polarizations; no presumed convergence | Refinement paragraph/macros |
| selection_study | Three geometries, three stable orderings, every sampled count; full/pruned complex error and gap probes | fig_selection |
| selection_vertices | All 128 vertex defects, diagnostic distances, ranks and isolated omitted-pair probes | Selection protocol |
| evaluation_timing | Every evaluator; actual one/two/four transition-call count, warmup plus seven batches of 200 calls, median/min/max and actual host metadata | Cost table |

The original isolated phase error was reproduced with a non-origin facet and
translation covariance. Boundary acceptance additionally requires exact and
one-sided reflection/incident handoffs, endpoint visibility, scale covariance,
and high-precision held-out angular checks. A matching CSV alone is not proof
of those properties. Focused Diff tests live in `test/paper162_integration.jl`;
canonical scalar/domain and geometry tests live in the UTDKernels suite.

Timing compares the selected production score/pair routes, independent direct
reference evaluators, and general-angle coefficient/guard calls. The Diff
production route includes frame construction; those times must not be presented
as a pruning speedup. Independent-reference times are not a second production
implementation. Full end-to-end field throughput should be reported separately
when comparing the backends.
