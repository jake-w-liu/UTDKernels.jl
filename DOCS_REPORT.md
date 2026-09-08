# Documentation verification

## Unified canonical-transition API verification

Version: 0.4.0 development tree
Verified: 2026-09-08

| Check | Result |
|---|---:|
| Exported symbols represented by Documenter `checkdocs=:exports` | 85/85 |
| Finite-edge tutorial included in navigation | Passed |
| Reflection-boundary tutorial included in navigation | Passed |
| Passive-transition tutorial included in navigation | Passed |
| Bivariate-transition tutorial included in navigation | Passed |
| Curvature-measure tutorial included in navigation | Passed |
| Multipole-transition tutorial included in navigation | Passed |
| Hessian-metric tutorial included in navigation | Passed |
| Null-uniform-moment tutorial included in navigation | Passed |
| Continuous-order tutorial included in navigation | Passed |
| Shadow-sensitivity tutorial included in navigation | Passed |
| Documenter build | Passed |
| Doctests | Passed |
| Broken cross-references | 0 |
| Documenter warnings and errors | 0 |
| Package tests (`--threads=1`) | 15,530/15,530 |
| Package tests (`--threads=4`) | 15,530/15,530 |
| Package tests (`--threads=8`) | 15,530/15,530 |

The README, API reference, home page, and phase-specific tutorials document the
finite-edge, PEC reflection-boundary, passive complex, bivariate,
curvature-measure, multipole, Hessian-metric, and null-uniform contracts. The
Hessian tutorial defines the dual metric, signed coordinate, stable effective
distance, type/allocation behavior, coordinate and pole-equation invariance,
and the separation between a canonical argument and physical amplitudes. The
curvature tutorial records the `Float16`, stored-precision `BigFloat`,
automatic-differentiation, and harmonic-order domains. The multipole tutorial
defines the distinct-node and automatic APIs, diagnostics, validated order and
series bounds, type and allocation behavior, stable intermediate- and
large-argument derivative routes, and the separation between scalar canonical
factors and geometry-specific mechanism assembly.
The null-moment tutorial defines recurrence/asymptotic routing, bounded precision
recovery, hierarchy scaling, shifted-null coefficients, type/allocation
behavior, and the separation between a canonical scalar basis and physical
vector matching. The continuous-order tutorial defines the normalized
half-line integral, certified quadrature, scaled saddle branch, physical
moment hierarchy, numeric types, resource caps, and geometry-owned scope. The
shadow-sensitivity tutorial defines the exact switch derivative, dispersive
multiplier, finite hierarchy, pullback coefficients, resource bounds, and
phase-demodulated real-branch scope.

## Version 0.3.2 release verification

Version: 0.3.2
Verified: 2026-08-27

| Check | Result |
|---|---:|
| Exported symbols with docstrings | 33/33 |
| Exported symbols included in the API reference | 33/33 |
| Documenter build | Passed |
| Doctests | Passed |
| Broken cross-references | 0 |
| Documenter warnings and errors | 0 |

The updated documentation distinguishes the analytic grazing-domain certificate
from quadrature convergence, records the adaptive quadrature controls, and
separates principal mathematical roots from passive material-wave roots. The
README, API reference, numerical-methods tutorial, impedance tutorial, and
Maliuzhinets tutorial agree with the v0.3.2 implementation.

Documentation complete: every check above passed; no documentation issue remains open.
