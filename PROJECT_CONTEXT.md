# VLA-SafeBench Project Context

This file is a durable handoff for future Codex conversations. It contains project decisions and relevant history, but intentionally excludes all credentials, access tokens, private keys, IP addresses, and other secrets.

## User goal

Build a four-week, internship-ready VLA simulation project that is reproducible, technically honest, and independent of access to a physical robot. The target roles are VLA algorithm engineer and VLA systems/algorithm engineer.

The user prefers a clean result over a large but shallow experiment matrix. The final project must visibly demonstrate the work through a risk-overlay rollout video and one defensible main result table.

## Why this project exists

The user previously trained and remotely deployed a π0.5 policy for a dual-arm cup-hanging task in a physical-robot competition. The model server, authentication, GPU inference, network connection, action schema, and latency worked. A physical test video showed smooth arm motion but an incorrect grasp approach: the gripper remained open, contacted the cup body, knocked it over, and failed to recover.

That experience motivated a simulator-first project about distinguishing deployment faults, eventual task failure, and imminent unsafe events. The competition infrastructure and credentials are not dependencies of this project.

## Current project state

- Status (2026-09-26): Phase 1 and the Phase-2 frozen pickup-stall confirmation are complete. Remaining work is documentation and presentation, not further model tuning.
- Canonical specification: `README.md` in this directory.
- Platform: SmolVLA + LIBERO `libero_spatial` task 4.
- v0.1 natural cohort: 50 episodes, 18 successes / 32 failures; group-disjoint train/validation/test = 30/10/10 by `initial_state_id`.
- Formal event coverage in v0.1 is 1/32 failures (3.125%); impending lead-time and safe-stop claims remain dropped.
- v0.1 test: frozen dual-camera vision at step 80 and temporal MLP at step 120 both reach AUPRC 0.982 / AUROC 0.952.
- v0.2 independent confirmation uses preset states 30–49 (20 episodes, 11 success / 9 failure). Frozen vision-step80 reaches AUPRC 0.939 / AUROC 0.919; temporal-step120 reaches 0.882 / 0.828.
- v0.3 was frozen before collection: preset states 30–49, five new seeds per state, 100/100 valid episodes (40 success / 60 failure). Privileged black-bowl, drawer, plate, and gripper state is analysis-only.
- On v0.3, vision-step80 raw AUPRC/AUROC is 0.926/0.863 and temporal-step120 is 0.937/0.876. Initial-state cluster bootstrap is used because each state has five repetitions.
- The stage-only baseline reaches AUROC 0.908 at step 80, exceeding vision. Leave-one-initial-state-out stage residualization reduces vision AUROC to 0.426 [0.260, 0.597] and temporal to 0.514 [0.334, 0.705].
- Same-state nearest-stage pairs agree: vision raw/stage-only/residual ordering is 13/16, 15/16, 7/16; temporal is 15/16, 13/16, 9/16. Median match distances remain 2.89/4.41 SD.
- Final Phase-1 interpretation: frozen predictors reliably rank eventual outcome, but measured task stage explains the signal. There is no reliable progress-independent precursor evidence.
- Phase-2 event: first target approach within 0.10 m followed by less than 0.04 m target displacement over 40 steps; predict confirmation from the 16-step history ending 20 steps before `t_event`.
- Strict Phase-2 cohort: 86 aligned samples, 27 positives. Test has 19 samples, 7 positives, and 4 held-out initial-state groups; every test positive has a same-state negative.
- Strict stage baseline uses privileged first-approach time and checkpoint target displacement for analysis only. Temporal MLP reaches AUPRC/AUROC 0.938/0.964, with increments 0.402 [0.037, 0.784] / 0.321 [0.042, 0.750]; same-state pairs are 7/7.
- Frozen vision reaches 0.982/0.988, but incremental confidence intervals touch zero; treat it as suggestive. Temporal ECE is 0.231, so no probability-threshold or safe-stop claim is supported.
- v0.5 predeclared confirmation: states 0–29 × 6 new seeds, 180/180 valid episodes (70 success / 110 failure). Frozen strict labeling keeps 144 samples, 61 pickup-stall positives, 83 negatives, and 29 initial-state groups.
- Full confirmation stage-only/temporal/vision AUROC is 0.643/0.838/0.936. Vision minus stage-only AUROC is 0.293 [0.164, 0.431]; temporal is 0.195 [0.044, 0.355].
- The mixed-initial-state subset has 73 samples, 36 positives, and 13 groups. Stage-only/temporal/vision AUROC is 0.589/0.715/0.892; vision increment is 0.302 [0.147, 0.456] and same-state pair ranking is 69/77. Temporal AUROC increment crosses zero on this subset.
- The predeclared support gate fails because only 36/61 positives (59%) have a same-state negative, below 80%. The ranking gate passes but the combined formal claim gate remains false.
- Phase-2 claim boundary: frozen vision contains progress-independent signal for one operational pickup-stall event after 20 observed development steps. Do not generalize this to all failures, pre-attempt prediction, safe stop, or held-out tasks.
- A1/A2 supporting results remain bounded: A1 11/11 predefined cases; action-axis swap command-effect 10/10 with normal false alarms 1/10; camera swap 7/10 is only an indirect response.
- Do not reopen ordinary outcome model selection, alter the frozen v0.4/v0.5 protocols, tune on confirmation results, or add post-hoc episodes to repair the failed support gate. Any future extension must predeclare a new sampling design that increases mixed-state coverage.

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
