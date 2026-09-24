# Development guardrails

- Preserve every existing evaluation and frozen release. Never rerun an incomplete attempt automatically or overwrite its trajectories.
- Develop on `dev/local-verifier` or another explicit development branch. Preserve checkpoint tags; never force-push or rewrite published history.
- Commit each reviewed, tested logical change with an accurate message. Recovered snapshots are not an invented historical commit sequence.
- Use `scripts/dev_local.py` for new local evaluations: clean Git tree, pinned local model, frozen source and input snapshots. Commit code before starting an experiment, and record the commit/config/results together.
- Do not change the official House adapter defaults or already-submitted image when configuring local experiments.
- Do not commit credentials, team claims, submission ZIPs, environment files, weights, raw datasets or raw evaluation traces. Inspect staged files before any public push.
- Do not call engineering tests or connectivity checks pass@1. Report complete fixed-list model evaluations separately with their limitations.
