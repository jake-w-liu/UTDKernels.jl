# Active central harness verification — 29 October 2026

This records the implementation checkpoint before the authoritative package and paper runs. Running/pending labels refer to this checkpoint; completed outcomes require the generated full-test and reproduction receipts.

| Surface | Actual checks | Status |
|---|---|---|
| Central paper contract | Generated-source AST, UUID and source checks; actual five projects with 9/5/5/6/5 entry files | Passed |
| Launcher generation | Archive-first preflight; no partial writes on a conflicting implementation | 12 mutation checks passed |
| Full-test worker entry | Load worker safely; reject missing arguments and an unknown package before launching tests | 2 checks passed |
| Full-test receipts | Source/fixture/runtime/dependency/selector/observer/log checks; stale, missing, failed and tampered cases | 16 checks plus 1 parse check passed |
| Reproduction receipts | Complete stages, identities, source/dependency/runtime profiles, log hashes and full inventory | 15 mutation checks passed |
| Test subprocess observer | Actual environment capture, nested-process overwrite guard, and empty-project/restricted-LOAD_PATH bootstrap; no `Pkg.test` invoked | 26 checks passed |
| Runtime matrix | 19 receipt/worker controls, 12 launcher controls, 15 reproduction controls and 26 observer controls on Julia 1.12.7 and 1.13.3 | All 72 controls passed on both versions |
| Five dispatchers | Latest channel, full flags/argument forwarding, child failure, missing entry and tilde path | 16 initial assertions plus 5 missing-entry dot-path mutations passed |
| Mirrors | Temporary regeneration; only changed script copied; unrelated 82/84 mirror files preserved | Matching generated script |
| Workflow architecture | Existing validator in all five active projects | 5/5 terminal 0 |
| Actual audit smoke | Real 168 entries/source bindings; strict reuse-only branch without a qualifying UTD receipt | Correctly failed; no `Pkg.test` launched |
| Real Pkg.test precompile regression | Fresh local target/dependency images; clean precompile Main/nonce, actual test PID and exit observations | 16 end-to-end controls plus 5 fixture-package assertions passed on both versions |
| Actual full package tests | Strict worker and observer supplied; unqualified historical receipts rejected | Final-source attempt 4 is running; no passed full receipt claimed here |
| Actual full reproduction | Complete data and plots required for both backends; partial runs cannot qualify | Authoritative latest runs pending |

Current 72-control matrix: `/tmp/central-harness-post-precompile-{baseline,latest}.log`. Real-Pkg.test integration controls: `/tmp/central-observer-real-pkg-{baseline,latest}.log`. Exact guard diff, independently matching pre-fix builder hash, actual fixture test PIDs and after-completion tool-session exit observations are in `reproduction/results/harness-precompile-controls/observer-precompile-proof.json`. The fixture proof is separate from full production-package receipts.
Dispatcher and workflow records: `/tmp/central-harness-dispatch-tests.json`, `/tmp/central-harness-dot-missing-green.json` and `/tmp/central-harness-workflow-{160,162,165,168,170}.log`.
The real failed smoke is `/tmp/central-harness-reuse-only-smoke.log`; its receipt is `reproduction/results/development-harness/paper168-DgeGVs/audit.toml`.

No fixture receipt is actual package-test evidence. The controls did not modify production projects, sources or tests. The first full-worker launch failed before `Pkg.test`; the second reached sandbox setup but failed during its CacheFlags preload because named stdlib imports were unavailable. The worker-entry and restricted-environment regressions cover both repairs. The third attempt exposed observer loading in dependency image workers; the precompile guard now runs before any nonce mutation or module include. A real local-package regression reproduced that failure before repair and passed afterward. The observer imports stdlibs by UUID without changing LOAD_PATH or declaring extra target test dependencies. A successful final-source full test remains required.
