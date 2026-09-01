# Documentation verification

## Unified finite-edge API verification

Version: 0.3.2 development tree
Verified: 2026-09-01

| Check | Result |
|---|---:|
| Exported symbols represented by Documenter `checkdocs=:exports` | 51/51 |
| Finite-edge tutorial included in navigation | Passed |
| Documenter build | Passed |
| Doctests | Passed |
| Broken cross-references | 0 |
| Documenter warnings and errors | 0 |

The README, API reference, home page, and finite-edge tutorial document the
exact phase map, local amplitude contract, three Fresnel moments, EPM orders,
endpoint and parameter derivatives, ForwardDiff behavior, and physical scope.
The docs distinguish the smooth scalar/componentwise finite-edge kernel from a
complete vector ITD or full-wave model.

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
