# VLA-SafeBench Project Context

This file is a durable handoff for future Codex conversations. It contains project decisions and relevant history, but intentionally excludes all credentials, access tokens, private keys, IP addresses, and other secrets.

## User goal

Build a four-week, internship-ready VLA simulation project that is reproducible, technically honest, and independent of access to a physical robot. The target roles are VLA algorithm engineer and VLA systems/algorithm engineer.

The user prefers a clean result over a large but shallow experiment matrix. The final project must visibly demonstrate the work through a risk-overlay rollout video and one defensible main result table.

## Why this project exists

The user previously trained and remotely deployed a π0.5 policy for a dual-arm cup-hanging task in a physical-robot competition. The model server, authentication, GPU inference, network connection, action schema, and latency worked. A physical test video showed smooth arm motion but an incorrect grasp approach: the gripper remained open, contacted the cup body, knocked it over, and failed to recover.

That experience motivated a simulator-first project about distinguishing deployment faults, eventual task failure, and imminent unsafe events. The competition infrastructure and credentials are not dependencies of this project.

## Current project state

- Status (2026-09-11): Week 1 and the Week-2 learned-signal gate are complete; the project is on the Outcome Prediction main track.
- Canonical specification: `README.md` in this directory.
- Platform: SmolVLA + LIBERO `libero_spatial` task 4 on RTX 4090 D.
- Frozen natural cohort: 50 episodes, 18 successes / 32 failures; group-disjoint train/validation/test = 30/10/10 by `initial_state_id`.
- Formal event coverage: 1/32 failures has an available self-collision or joint-violation event (3.125%); therefore impending lead-time and safe-stop claims were dropped.
- State/action temporal MLP first reliably exceeds initial-proprio difficulty at step 120.
- Frozen dual-camera ResNet-50 checkpoint vision reliably exceeds initial-frame and initial-proprio difficulty at step 80 (test AUPRC 0.982, AUROC 0.952). This supports earlier outcome-discriminative signal, not impending unsafe-event detection.
- Test contains only 10 episodes and every failure reaches the 280-step horizon; retain both limitations in every claim.
- Next: freeze the main comparison, compute validation-thresholded offline decision utility, run only the main/wrist/dual-camera ablation, and prepare visualization. Do not upgrade to Transformer.

## Locked research definitions

1. A1 syntax/protocol faults, A2 semantic configuration faults, and B natural/emergent failures are separate regimes.
2. Outcome Prediction and Impending Failure Detection are different research questions:
   - Outcome predicts eventual episode success/failure and may learn task difficulty.
   - Impending detects a clearly timestamped unsafe event within a future horizon K.
3. Outcome metrics must never be presented as runtime failure-detection metrics.
4. Only impending risk may drive safe stop in the four-week Safety track.
5. Simulator privileged state may generate labels and evaluation diagnostics, but must not enter predictor inputs or normalization statistics.
6. The MVP excludes task-specific unexpected-contact labels and contact whitelists. Allowed unsafe events are self-collision, joint/workspace violation, and an optional clearly thresholded impact event.
7. Core generalization is held-out episode / seed / initial state. Held-out task/object is exploratory only.
8. Safe stop depends on control semantics. A zero vector is not universally a safe command.

## Four-week execution contract

### By Day 3

- Obtain at least one successful and one failed episode.
- Logging is built into the first rollout path, not added later.
- RGB, proprioception, predicted action/chunk, executed action, timestamps, latency, event labels, and MP4 frames share stable step IDs.
- Benchmark at least 20 episodes on the intended GPU and recording configuration.
- Record episodes/hour, steps/second, inference p50/p95/p99, recording overhead, storage per 100 episodes, and projected data cost.

### By end of Week 1

- Collect at least 50 natural episodes with at least 20 failures.
- Compute `p_event_given_failure` with raw numerator/denominator and per-event/task counts.
- Choose exactly one main track:
  - Safety/Impending only if `p_event_given_failure >= 20%` and the measured budget can produce at least 50/15/20 event-positive train/val/test episodes within 48 GPU-hours.
  - Otherwise choose Outcome and drop lead-time and safe-stop claims.

### By end of Week 2

- Freeze leak-free episode/seed/initial-state splits before generating windows.
- Compare a state/action temporal model and frozen-image-feature model against the appropriate strong baseline.
- Frozen vision is prepared during Week 2, not used as a late Week 3 rescue.
- If neither learned approach beats the strong baseline under episode bootstrap, pivot immediately to the framework/negative-result project. Do not spend two more weeks forcing a positive learned result.

### Week 3

- Complete held-out evaluation, calibration, and only the minimal ablations needed for the chosen main claim.
- If on the Safety track, quantify safe-stop cost and benefit.

### Week 4 acceptance

- One honest main result table.
- One failure video with risk, event time, and intervention time overlaid as applicable.
- One reproducible release with frozen configuration and known limitations.

## Main-table branches

### Safety / Impending

Compare a tuned rule monitor, state/action temporal MLP, and frozen-vision predictor using AUPRC, event recall at a fixed false-alarm budget, false alarms per 1,000 steps, lead time, and calibration. Safe-stop intervention is a supporting table.

### Outcome

Compare task-ID + initial-frame difficulty, state/action temporal, and frozen-vision predictors. Report early/mid/late episode AUPRC plus AUROC and calibration. Do not claim imminent failure detection or runtime safety intervention.

## Explicit fallback

If natural failure data, event labels, or learned-signal gates fail, ship the project as a reproducible VLA fault-injection, observability, and runtime-diagnostics framework. Valid fallback artifacts include synchronized rollout logging/video, A1 validation, selected A2 stress tests, command-effect consistency, action/risk curves, failure taxonomy, and negative-result analysis.

## Scope that must not block the four-week release

- π0.5 comparison
- Multimodal Transformer
- Full A2 fault matrix
- RoboTwin / dual-arm evaluation
- Held-out task generalization claims
- Recovery policy
- Adaptive action chunking
- Real-robot deployment

These remain possible follow-up work only after the MVP release is frozen.

## Completed implementation foundation

The recorder has been active from step zero since the first formal SmolVLA + LIBERO rollout. The Day-3 plumbing/cost gate, provenance capture, artifact validation, predictor input allowlist, frozen split, outcome dataset, initial-proprio baseline, temporal MLP, episode bootstrap, timing audit, and frozen-vision baseline are complete. Generated rollouts, videos, feature caches, and model files remain outside Git; tracked manifests and reports define their provenance.
