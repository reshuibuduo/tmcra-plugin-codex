# Release-candidate scanner review

The official HOL scanner currently blocks this release's marketplace security gate.
The failure is retained; no runtime files or rules were excluded to obtain a pass.
The threshold remains 80 with high-severity findings fatal, and Cisco scanning
remains enabled.

Locally reproduced with `plugin-scanner` 3.0.17, 3.0.65 and 3.0.94. The latter
matches the official action pinned in this repository. Its
`DANGEROUS_DYNAMIC_EXECUTION` detector reports `.eval()` calls in these Python files:

- `runtime/memory-api/tmcra_v3_online_runtime.py`
- `runtime/memory-api/tmcra_v3_reranker.py`
- `runtime/memory-api/tmp_tmcra_v2_lme_pipeline.py`
- `runtime/memory-api/core/gru_text_generator.py`
- `runtime/memory-api/core/natural_layout.py`
- `runtime/memory-api/core/policy_network.py`
- `runtime/memory-api/core/scene_line_generator.py`
- `runtime/memory-api/core/tri_maze_neural_trainer.py`

The flagged expressions are model methods (`model.eval()`, `self.model.eval()`,
`self.cross_model.eval()`, `self.fusion.eval()`, `proposal.eval()`, `ranker.eval()`,
`gen.model.eval()` and `policy.model.eval()`). They take no source-code argument.
PyTorch's `Module.eval()` switches a module to inference mode. These calls are
preserved in the bundled backend, whose SHA-256 inventory is verified during build.
This review covers these specific findings, not a blanket claim that the entire
application or its dependencies have no security issues.

The account-free setup entry now has no placeholder API credential. It exposes
an empty, read-only setup dashboard and rejects all memory operations until the
user reopens an authenticated workspace after installation. Regression tests
verify that it creates no memory-control state.

The remaining scanner finding needs an upstream language-aware detector fix or
explicit marketplace maintainer adjudication. A passing functional test suite
must not be described as a passing marketplace security scan.

## Independent review and fixture remediation

The failed official run `33990306533` scanned merge commit
`0d46070f1e42e438871163d55ae43fb61fb8b7ef`, containing PR head
`00aba63c7f9642c277e257651295531a6c4fe8d5`. It reported the eight Python
files above and 15 test files with `HARDCODED_SECRET` (reported under both
plugin ecosystems). The Action does not trust repository-owned exclusions;
the pre-existing `tests/*` policy is not evidence of a passing official scan.

Source review traced the flagged credentials to local HTTP mock servers,
`MockTmcraServer` token allowlists, injected request implementations, or an
`example.invalid` demonstration workspace. The usage-attribution test has a
production-shaped URL but replaces `globalThis.fetch` before making its request.
The opt-in provider test obtains its real provider credential from an explicit
environment input; its flagged lease and service tokens belong to the local fake
service. No flagged committed value was found to be a production credential.
This conclusion is limited to the reported findings, not the entire repository.

These 15 files now generate ephemeral fixture credentials with Node's built-in
`randomUUID()`. Shared credential variables preserve authentication comparisons,
separate service/provider identities, and checks for redaction and isolation.
The memory-state leakage assertion checks the actual generated credential.
Redaction payloads remain in the tests so secret filtering is still exercised.
No dependency, runtime model file, scanner rule, threshold, exclusion, or
repository-policy trust setting changed.

Local validation uses the already installed Node 24.21.0: syntax checks for the
six `verify` entry points and all changed JavaScript files, plus
`node scripts/test_codex_e2e.mjs` in mock mode. Additional recovery and
usage-attribution tests are run separately. The opt-in real-provider test is
syntax-checked only; it requires explicit provider configuration and is not
needed for this fixture-only change. Windows checks and archive build require
the existing CI matrix.

Upstream `hashgraph-online/hol-guard#2805` is still open and unmerged, with no
submitted human reviews at the time of this review. The current scanner source
still uses a text regex that matches these inference calls. This downstream PR
does not execute the unmerged scanner or replace the pinned official Action.
The official high-severity gate remains blocking until a reviewed detector fix
is published and adopted, or a marketplace maintainer adjudicates the findings.
