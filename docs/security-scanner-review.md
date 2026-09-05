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

The current GitHub Action also ignores repository-owned scanner policy by default.
Consequently its raw CI report includes `HARDCODED_SECRET` findings for synthetic
mock credentials in `tests/`, while the local CLI respects the pre-existing
test-fixture exclusion in `.plugin-scanner.toml`. Those reports are different and
neither is a passing severity gate. The tests use mock/synthetic sentinel values;
they are excluded from the install ZIP by its explicit release-file inventory.
No new exclusions or policy-trust overrides were introduced in the Action.
