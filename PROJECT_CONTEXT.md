# VLA-SafeBench Project Context

This file is a durable handoff for future Codex conversations. It contains project decisions and relevant history, but intentionally excludes all credentials, access tokens, private keys, IP addresses, and other secrets.

## User goal

Build a four-week, internship-ready VLA simulation project that is reproducible, technically honest, and independent of access to a physical robot. The target roles are VLA algorithm engineer and VLA systems/algorithm engineer.

The user prefers a clean result over a large but shallow experiment matrix. The final project must visibly demonstrate the work through a risk-overlay rollout video and one defensible main result table.

## Why this project exists

The user previously trained and remotely deployed a π0.5 policy for a dual-arm cup-hanging task in a physical-robot competition. The model server, authentication, GPU inference, network connection, action schema, and latency worked. A physical test video showed smooth arm motion but an incorrect grasp approach: the gripper remained open, contacted the cup body, knocked it over, and failed to recover.

That experience motivated a simulator-first project about distinguishing deployment faults, eventual task failure, and imminent unsafe events. The competition infrastructure and credentials are not dependencies of this project.

## Current project state

- Status (2026-09-20): the bounded four-week v0.1 release remains frozen and checksum-verified (21/21 files). A post-release v0.2 audit verified initial-state identity, ablated the temporal checkpoint feature, and evaluated the frozen predictors on 20 previously unused task-4 preset states.
- Canonical specification: `README.md` in this directory.
- Platform: SmolVLA + LIBERO `libero_spatial` task 4 on RTX 4090 D.
- Frozen natural cohort: 50 episodes, 18 successes / 32 failures; group-disjoint train/validation/test = 30/10/10 by `initial_state_id`.
- Formal event coverage: 1/32 failures has an available self-collision or joint-violation event (3.125%); therefore impending lead-time and safe-stop claims were dropped.
- State/action temporal MLP first reliably exceeds initial-proprio difficulty at step 120.
- Frozen dual-camera ResNet-50 checkpoint vision reliably exceeds initial-frame and initial-proprio difficulty at step 80 (test AUPRC 0.982, AUROC 0.952), so the Week-2 learned-signal gate remains GO. Frozen RGB/error and LOEO audits classify the cue as B: vision mainly reads outcome-associated execution progress / proximity to a successful configuration, not an independent early failure precursor. Vision-step80 and temporal-step120 assign different probabilities but misorder the same failure/success pair, have identical LOEO ranking metrics, and are highly correlated across the 10 test episodes (Pearson 0.9998; Spearman 0.9515). Treat them as likely measurements of the same latent progress signal, not independent evidence chains.
- The shared false negative `smolvla-20260911T013117405943Z` looks success-like at step 80, reaches a near-complete configuration around steps 120–160, then makes small adjustments until the 280-step timeout without a usable self-collision or joint-violation event. Its first reliable divergence from successful trajectories is around steps 120–160, not step 80.
- Test contains only 10 episodes and every failure reaches the 280-step horizon; retain both limitations in every claim.
- The main/wrist/dual-camera ablation is complete. At step 80, main/wrist/dual test AUPRC is 0.844/1.000/0.982 and AUROC is 0.667/1.000/0.952. Paired bootstrap supports lower Brier/ECE for wrist and dual versus main, but does not support a reliable wrist-versus-dual difference. Keep the preregistered dual-camera pipeline as the main model.
- Validation selected an offline outcome threshold of 0.9998072982 under a zero-sacrificed-validation-success constraint. On test it identified 4/7 failures, observed 0/3 sacrificed successes, and counterfactually saved 756 recorded steps (52.95 seconds at the measured 70.042 ms/step). This is an offline efficiency/faster-confirmation result, not a safety intervention.
- The test risk plot and a 360x360, 20 FPS, 280-frame outcome-risk overlay video are complete. The overlay explicitly marks the step-80 trigger as offline counterfactual and continues the recorded trajectory.
- Progress control uses a train-only, outcome-label-free frozen-feature time axis. Progress-only test AUROC is 0.905 at vision-step80 and 1.000 at temporal-step120; progress-residualized AUROC is 0.714 and 0.381 respectively. Vision residual AUROC CI [0.111, 1.000] and partial-correlation CI [-0.232, 1.000] are inconclusive. Only two vision success/failure pairs are within 0.5 train SD; no temporal pairs are close. Final interpretation: progress explains much of the observed ranking, while the current 10-episode test cannot determine whether vision retains progress-independent information.
- The recorded `initial_state_id` matches the LIBERO preset index for all 50 v0.1 episodes, with stable preset fingerprints across collector sessions. Removing `checkpoint_step / 280` leaves temporal test AUPRC/AUROC unchanged at steps 80 and 120; the feature affects probability scale rather than within-checkpoint ranking.
- Independent confirmation: preset states 30–49 produced 20 episodes (11 success / 9 failure). With all v0.1 weights and normalization frozen, vision-step80 reaches AUPRC 0.939 / AUROC 0.919 and temporal-step120 reaches AUPRC 0.882 / AUROC 0.828. This confirms same-task outcome association, not a progress-independent failure precursor or held-out-task generalization.
- Next: preserve the frozen v0.1/v0.2 artifacts and claim boundary. A second task and learned A2 monitor remain out of scope. Do not reopen model selection, frame outcome-based termination as a safety intervention, or upgrade to Transformer.

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
