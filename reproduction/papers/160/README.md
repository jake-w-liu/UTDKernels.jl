# Paper 160 reproduction contract

The complete workflow is `generate.jl`, `evaluator.jl`, `cost.jl`,
`sensitivity.jl`, `spatial_validation.jl`, `macros.jl`, then `plot.jl`.
Select `UTD_PAPER_BACKEND=kernels|diffutd` and an absolute
`UTD_PAPER_OUTPUT` paper directory. The 17 manuscript CSV schemas and six
figure assets are enumerated in `coverage.json`. Eight manuscript tables draw
from the saved error, cost, evaluator, reference-precision and fitting evidence.
The spatial receipt is TOML and does not add a fictitious manuscript dataset.

UTDKernels remains the mathematical owner of F/F', G/G', branch certificates,
fixed/adaptive GL integration and infinite-distance factorization. Independent
erfc-series/asymptotic and Laplace-contour references remain separate from those
production functions. Mathematical transition/derivative/precision experiments
are shared honestly. On the Diff backend every production coefficient route
constructs a 3D ray frame, transports its polarization dyad and recovers the
coefficient by projection. No archived paper package is imported.

Retained angular parameters are explicit precision carriers validated against
the Cartesian geometry. The oracle still receives the exact original binary64
inputs; representation error is not relabeled as kernel error. The `L=Inf`
population is the local F-to-one coefficient limit. It is not interpreted as a
finite-distance field or a finite-edge far-field extraction.

| Dataset | Main acceptance obligation | Evidence role |
|---|---|---|
| canonical_sweep | All 151 offsets; direct/four-term/linear/GL2/4/8, saved complex coefficients, exact independent references and declared fits | Coalescence/accuracy figures, absolute-error table |
| coalescence | All 76 offsets and both large kernel terms; preserve the lost difference | Canonical coalescence panel |
| ensemble | All 800 accepted finite-distance cases and rejection design; every production coefficient uses selected backend | Empirical error/condition figure and quantiles |
| ensemble_farfield | All 400 accepted local infinite-distance cases; factored limit, independent wavenumber replay, no finite-field substitution | Infinite-distance panel and regime evidence |
| derivative_verification | All 400 accepted G' samples against independent differentiation; shared canonical mathematical primitive | Derivative error table |
| electrical_sweep | All 101 electrical lengths with exact inputs and independent coefficient references | Electrical-length panel |
| error_map | Complete 33 by 33 offset/electrical-length grid | Joint error figure |
| regime_sweep | All 228 interior/exterior/infinite-distance rows under explicit opt-ins | Regime figure |
| transition_oracle | All 17 held-out F/F' arguments through 2e8 against independent contour quadrature | Derivative/reference accuracy evidence |
| transition_scaled | Seven scaled/unscaled erfc comparisons against the independent contour | Transition evaluation evidence |
| derivative_condition | Four one-sided summation-conditioning cases using independently rounded terms | Formula-level conditioning limit |
| derivative_floor | Seven actual collapsed-node cases; selected coefficient, linear and correctly rounded derivative substitutions | Derivative-floor explanation |
| precision_convergence | All nine original/max-condition/crossover geometries; 320/431 working bits and original threshold | Reference-precision table |
| evaluator_validation | All 19 adaptive, limit, mirrored-input, quantization, whole-interval, refinement and fallback checks | Evaluator tables and guarded-domain evidence |
| cost_accuracy | All 194 rows; matched evaluator outputs, actual warmup/minimum protocol, tolerance-control settings and backend spatial overhead | Cost/accuracy table |
| adaptive_order_stats | All 300 fresh accepted-population draws, reported accepted orders/calls/times | Adaptive-order table |
| sensitivity_sweep | All 16 complex sensitivity references and matched continuation/four-term/linear/hybrid AD plus FD | Sensitivity figure and use-case evidence |

`spatial_validation.jl` replays all 800 finite and 400 infinite populations,
checks exact stored coefficient reductions, scores finite physical fields with
the independent oracle at their actual represented L, and performs held-out
3D dyad rotations. Its receipt separates coefficient error, angular encoding
and propagation-phase rounding. `test/paper160_spatial.jl` additionally checks
both polarizations, oblique finite fields, rigid covariance, explicit carrier
and derivative failures, uncomputed hard channels, domain rejection and the
infinite-distance API distinction.

The fixed-order timed route uses the canonical interval certificate without
optional G/G' diagnostics, then evaluates the cached canonical GL rule.
Reported integration-call counts describe the selected fixed rule.
Timing headers distinguish the Diff spatial work from scalar-kernel timing.
