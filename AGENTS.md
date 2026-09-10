# Project instructions for coding agents

Before planning or changing this project, read `README.md` and `PROJECT_CONTEXT.md` completely.

## Project contract

- Treat the four-week execution gates in `PROJECT_CONTEXT.md` as hard scope controls.
- Optimize for one defensible main claim, one main table, one risk-overlay video, and one reproducible release.
- Keep Outcome Prediction separate from Impending Failure Detection in labels, metrics, tables, and claims.
- Do not claim held-out task generalization from the 2–3 task MVP.
- Do not add task-specific unexpected-contact whitelists to the MVP.
- Do not allow privileged simulator state into predictor features or normalization statistics.
- Do not treat an all-zero action as a universal safe stop.
- Do not weaken a strong baseline, merge regimes, cherry-pick seeds, or tune thresholds on the test set to obtain a positive result.
- If the Week-2 learned-signal gate fails, implement the documented pivot instead of expanding model complexity.

## Data and security

- Never read, copy, commit, summarize, or expose credential files, access tokens, private keys, cloud addresses, or competition secrets from adjacent directories.
- Store only public project configuration and explicitly sanitized context in this repository.
- Keep large rollout data, videos, model weights, and feature caches out of Git. Track them through manifests.

## Engineering expectations

- Logging must be in the rollout path from the first episode and use stable episode/step identifiers.
- Freeze train/validation/test manifests before generating temporal windows.
- Record exact checkpoint revision, resolved config, task, seed, initial state, Git commit, and environment/container version for each experiment.
- Prefer the smallest change that advances the current gate. π0.5, Transformers, RoboTwin, recovery, and adaptive chunking are not MVP prerequisites.
- Do not write experimental results or resume claims until they are backed by reproducible artifacts.

