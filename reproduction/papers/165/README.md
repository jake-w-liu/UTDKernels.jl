# Paper 165 reproduction contract

Run `generate.jl`, then `plot.jl`, in the reproduction environment with
`UTD_PAPER_BACKEND=kernels|diffutd`. `UTD_PAPER_OUTPUT` must be an absolute
paper output directory. `coverage.json` lists every schema: 18 CSVs and 12
figure assets. The driver writes both manuscript macro files. Its three tables
describe fit windows, resolution counts, and the error decomposition/budget.
No archived implementation package is imported.

UTDKernels owns intrinsic coefficients, local bias, curvature accumulation,
canonical geometry and independent Fourier–Bessel/quadrature/cylinder
references. DiffUTD derives turns from a planar curve embedded in arbitrary 3D
coordinates and evaluates phases with Cartesian dot products. Its nonuniform
adapter constructs the polygon from the turning cells' unit support lines.
The monostatic path uses an actual infinite cylindrical line source, local
3D frames, separate electric-soft/magnetic-hard transport, physical ranges,
and cylindrical spreading. Equal-direction incident pairs use the compensated
intrinsic primitive before soft/hard face cancellation. This scalar even-pair
diagnostic is not the complete electromagnetic field of the smooth cylinder.

| Dataset | Main acceptance obligation | Figure or table evidence |
|---|---|---|
| local_bias | Exact local bias at 13 counts; fit last seven, N>=64 | fig_local_bias, fit-window table |
| harmonic_convergence | Raw/debiased complex errors, exact local/alias split; fit raw only after alias <1% local term | fig_king_error, error-budget table |
| alias_identity_samples | All 600 seeded samples; direct curvature quadrature versus independent Bessel aliases, including near-zero denominators | Alias identity and reported maximum absolute error |
| absolute_scaling | 23 counts through 2^27; exact regular symmetry N times one physical seam, fixed and geometry-scaled distance; fit last six N>=2^22 | fig_scaling |
| smooth_pattern | All 2163 rows; prescribed smooth amplitude with tested signs; errors of complex fields | fig_king_pattern |
| aliasing_map | Complete 29 by 97 grid, angular bandwidth and fixed harmonic | fig_king_aliasing |
| nonuniform_partition | Thirteen positive turning partitions; raw/debiased error and hmax; last six counts N=96..512 | fig_resolution_partition |
| resolution_curves | All 531 first-satisfying integer counts; three harmonics, 59 electrical lengths, three tolerances and explicit no-hit sentinel | fig_resolution_count, resolution-count table |
| resolution_pattern_l2 | All 29 even counts; complex-pattern L2 over 721 endpoint-inclusive observation angles | fig_resolution_pattern |
| joint_refinement | All 117 rows; fixed and two growing-count rules, 181 endpoint-exclusive complex patterns; retain fixed-mesh failure | fig_joint |
| ellipse_convergence | Thirteen actual chord-turn geometries and phases; independent dense curvature-density reference, raw O(h)/debiased O(h²) windows | fig_ellipse, fit-window table |
| monostatic_convergence | Thirteen polygon counts, physical cylindrical normalization, full pair diagnostics, matched and fixed arcs, exact complex error split | fig_monostatic, error-budget/fit-window tables |
| monostatic_seams | Every N=48 seam; visibility and full soft/hard/even/face diagnostics, NaN only for explicitly invisible seams | Monostatic protocol |
| monostatic_pattern | All 361 aspects; both complete-cylinder reference polarizations under the stated source normalization, with diagnostic/component distinction | Monostatic scale comparison |
| king_circle | 600 independent geometry-panel points | fig_king_geometry |
| king_polygon | Fourteen vertices and declared turning-arrow locations | fig_king_geometry |
| ellipse_curve | 800 smooth-ellipse geometry points | fig_ellipse |
| ellipse_polygon | Twenty-four vertices with actual largest-turn arrows | fig_ellipse |

`test/paper165_integration.jl` checks complex field reductions, cylindrical
source amplitude/range/spreading, separate transported soft/hard channels,
arbitrary rigid transformations, derivatives and invalid inputs. The scalar
suite checks independent fixtures, aliases, asymptotic orders, compensated
small-turn limits and complete error identities. Relative errors against
near-zero harmonic values are roundoff-sensitive: compare absolute identity
residuals and the underlying complex values before interpreting backend
differences in those derived ratios. Resolution integer counts must match.

The physical scope is an extruded convex cross section with line and seam axes
parallel. Arbitrary Cartesian rotation/translation is supported. The continuum
claim remains planar, positive-curvature, normal-incidence/equal-azimuth; it
does not extend by coordinate embedding to a torsional curve or a full-field
convergence theorem. Large regular counts use exact symmetry, not allocated
arrays with 2^27 seams. Runtime claims require separately measured warm
coefficient, prepared-geometry and complete-field costs.
