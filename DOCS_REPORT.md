# Documentation verification

## Unified canonical-transition API verification

Version: 0.3.2 development tree
Verified: 2026-09-07

| Check | Result |
|---|---:|
| Exported symbols represented by Documenter `checkdocs=:exports` | 61/61 |
| Finite-edge tutorial included in navigation | Passed |
| Reflection-boundary tutorial included in navigation | Passed |
| Passive-transition tutorial included in navigation | Passed |
| Bivariate-transition tutorial included in navigation | Passed |
| Documenter build | Passed |
| Doctests | Passed |
| Broken cross-references | 0 |
| Documenter warnings and errors | 0 |

The README, API reference, home page, and phase-specific tutorials document the
finite-edge, PEC reflection-boundary, passive complex, and bivariate canonical
contracts. The bivariate tutorial distinguishes the correlated canonical
factor from a complete double-diffraction coefficient and records adaptive
certification, fixed-rule bounds, mechanism-weight identities, Hessian
normalization, AD scope, and near-rank-one limitations.

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

Docs complete: 4 issues detected → 4 confirmed → 4 fixed, 0 require user action.
