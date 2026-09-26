# VLA-SafeBench outcome MVP release artifacts

This file indexes the frozen task-4 Outcome Prediction release. Large rollout data,
videos, feature caches, and model weights remain outside Git. Paths below are relative
to the project root on the experiment host.

## Frozen identity and data

- Split manifest: `docs/manifests/week1_task4_split.json`
- Rollouts: `artifacts/week1/task4/episodes/`
- Outcome dataset: `artifacts/datasets/week1_task4_outcome.npz`
- Frozen ResNet-50 feature cache: `artifacts/features/week1_task4_frozen_vision.npz`
- Cohort: 50 episodes, 18 success / 32 failure
- Split: 30 train / 10 validation / 10 test, grouped by `initial_state_id`
- Checkpoints: steps 0, 40, 80, 120

The manifest records the task, seeds, initial states, project revision, LeRobot tree
digest, and policy-weights digest. Predictor inputs exclude label-only simulator state.

## Frozen result artifacts

- Initial-proprio baseline: `artifacts/results/week1_task4_difficulty_baseline.json`
- Temporal MLP: `artifacts/results/week1_task4_temporal_mlp.json`
- Frozen dual-camera vision: `artifacts/results/week1_task4_frozen_vision.json`
- Temporal bootstrap: `artifacts/results/week1_task4_temporal_vs_difficulty_bootstrap.json`
- Vision bootstrap: `artifacts/results/week1_task4_frozen_vision_bootstrap.json`
- Camera ablation reports: `artifacts/results/week1_task4_frozen_vision_{main,wrist,dual}.json`
- Camera ablation bootstrap: `artifacts/results/week1_task4_camera_ablation_bootstrap.json`
- Offline decision utility: `artifacts/results/week1_task4_outcome_utility.json`
- Progress-control analysis: `artifacts/results/week1_task4_progress_control.json`
- Timing audit: `artifacts/results/week1_task4_timing_audit.json`
- RGB/error and LOEO audit: `artifacts/audits/week2_step80/`
- False-negative audit: `artifacts/audits/week2_fn_013117405943Z/`

## RQ1 supporting study

- A1 protocol case table: `artifacts/results/rq1_a1_monitor.json`
- A2 paired rollouts: `artifacts/rq1/task4/{none,action_swap_xy,camera_swap}/`
- A2 command-effect report: `artifacts/results/rq1_task4_a2_consistency.json`
- Protocol: 20 paired initial states; first 10 normal pairs calibrate the threshold;
  the remaining 10 pairs are evaluated over the first 40 actions.
- A1 predefined cases: 11/11 passed.
- A2 evaluation: normal 1/10, action-swap 10/10, and camera-swap 7/10
  command-effect alarms. The camera result is an indirect closed-loop anomaly signal,
  not direct camera-mapping classification.

## Display artifacts

- Test risk curves: `artifacts/visualizations/week1_task4/test_outcome_risk_curves.png`
- Risk-overlay video:
  `artifacts/visualizations/week1_task4/smolvla-20260911T014308102939Z_outcome_risk_overlay.mp4`
- Render metadata: `artifacts/visualizations/week1_task4/render_metadata.json`

The overlay video is 360×360, 20 FPS, and 280 frames. Its trigger is an offline
counterfactual marker; the recorded rollout continued and no closed-loop intervention
was performed.

## v0.2 post-release audit and confirmation

v0.1 remains frozen. The following artifacts audit its integrity and apply its frozen
predictors to 20 previously unused task-4 preset states (30–49):

- Confirmation manifest: `docs/manifests/v02_task4_confirmation.json`
- Initial-state identity audit: `artifacts/results/v02_initial_state_identity.json`
- Temporal model without explicit checkpoint progress:
  `artifacts/results/v02_task4_temporal_no_checkpoint_progress.json`
- Paired temporal ablation:
  `artifacts/results/v02_task4_temporal_progress_ablation_bootstrap.json`
- Frozen confirmation evaluation:
  `artifacts/results/v02_task4_frozen_confirmation.json`
- Large-artifact checksum: `artifacts/v02/release-sha256.txt`

The confirmation cohort contains 20 episodes (11 success / 9 failure). Frozen
vision at step 80 reaches AUPRC 0.939 and AUROC 0.919; frozen temporal MLP at
step 120 reaches AUPRC 0.882 and AUROC 0.828. These results confirm outcome
association within the same task, not a progress-independent failure precursor.

## v0.3 predeclared stage-control audit

v0.3 freezes a 100-episode mechanism cohort before outcome inspection: preset
states 30–49, five new seeds per state, 40 successes and 60 failures. All 100
episodes passed sidecar/video validation. Privileged scene poses are analysis-only
and never enter either frozen predictor.

- Predeclared protocol: `docs/manifests/v03_task4_stage_protocol.json`
- Frozen cohort manifest: `docs/manifests/v03_task4_stage_cohort.json`
- Rollouts: `artifacts/v03/task4_stage_cohort/`
- Temporal dataset: `artifacts/v03/datasets/task4_stage_outcome.npz`
- Frozen dual-camera features: `artifacts/v03/features/task4_stage_frozen_vision.npz`
- Frozen predictor evaluation: `artifacts/results/v03_task4_frozen_stage_cohort.json`
- Stage-aligned analysis: `artifacts/results/v03_task4_stage_aligned_signal.json`
- Checksum: `artifacts/v03/release-sha256.txt`

At step 80, frozen vision reaches AUPRC 0.926 / AUROC 0.863, while the
privileged stage-only baseline reaches AUROC 0.908. Leave-one-initial-state-out
stage control reduces vision AUROC to 0.426 [0.260, 0.597]. Temporal step 120
falls to 0.514 [0.334, 0.705]. Both intervals include chance; Phase 1 finds no
reliable progress-independent outcome signal.

## v0.4 strict pickup-stall pilot

Phase 2 defines an explicit online event: after the first end-effector approach
within 0.10 m of the target bowl, pickup stall is confirmed if target displacement
remains below 0.04 m for 40 steps. Prediction uses the 16-step history ending 20
steps before confirmation. Negatives that already crossed 0.04 m by that checkpoint
are excluded.

- Frozen protocol: `docs/manifests/v04_task4_pickup_stall_protocol.json`
- Strict pilot manifest: `docs/manifests/v04_task4_pickup_stall_pilot_strict.json`
- Temporal dataset: `artifacts/v04/datasets/task4_pickup_stall_pilot_strict.npz`
- Frozen dual-camera features:
  `artifacts/v04/features/task4_pickup_stall_frozen_vision_strict.npz`
- Strict result: `artifacts/results/v04_task4_pickup_stall_pilot_strict.json`
- Model bundle: `artifacts/results/v04_task4_pickup_stall_pilot_strict.pt`
- Physical-review contact sheets:
  `artifacts/visualizations/v04_pickup_stall_first_approach_review/`
- Release checksum: `artifacts/v04/release-sha256.txt`

The strict cohort contains 86 aligned samples and 27 positives. Group-disjoint test
contains 19 samples, 7 positives, and 4 initial-state clusters. Temporal MLP reaches
AUPRC/AUROC 0.938/0.964 and improves over the privileged stage baseline by
0.402 [0.037, 0.784] / 0.321 [0.042, 0.750]. Frozen vision reaches 0.982/0.988,
but its incremental intervals touch zero. This is pilot evidence for predicting
stall confirmation after 20 observed development steps, not a safe-stop result or
a general failure detector.

## v0.5 frozen pickup-stall confirmation

The confirmation protocol was frozen before collection: preset states 0–29, six
new seeds per state, and 180 episodes total. Frozen v0.4 event rules and model
weights were applied once without confirmation-time tuning.

- Confirmation protocol: `docs/manifests/v05_task4_pickup_stall_confirmation_protocol.json`
- Complete cohort: `docs/manifests/v05_task4_pickup_stall_confirmation_cohort.json`
- Strict event manifest: `docs/manifests/v05_task4_pickup_stall_confirmation.json`
- Temporal metadata: `artifacts/v05/datasets/task4_pickup_stall_confirmation.json`
- Vision-feature metadata:
  `artifacts/v05/features/task4_pickup_stall_confirmation_frozen_vision.json`
- Frozen confirmation result:
  `artifacts/results/v05_task4_pickup_stall_frozen_confirmation.json`
- Release checksum: `artifacts/v05/release-sha256.txt`

All 180 episodes passed validation. The strict manifest retains 144 samples with
61 positives. On the full set, frozen vision reaches AUPRC/AUROC 0.922/0.936 and
improves over stage-only by 0.299 [0.146, 0.468] / 0.293 [0.164, 0.431]. On the
13-group mixed-state subset, vision reaches 0.899/0.892; its increments remain
0.255 [0.112, 0.425] / 0.302 [0.147, 0.456], with 69/77 same-state pairs ranked
correctly. Only 36/61 positives have a same-state negative, below the predeclared
80% support gate, so the combined formal claim gate remains false.

## Reproduce evaluation from frozen inputs

Activate the recorded environment and expose the source package:

```bash
source /root/autodl-tmp/envs/vlasafe312/bin/activate
export PYTHONPATH=/root/autodl-tmp/vlasafe/src
cd /root/autodl-tmp/vlasafe
```

Rebuild the outcome windows and train the baselines:

```bash
python scripts/build_outcome_dataset.py \
  docs/manifests/week1_task4_split.json \
  --output artifacts/datasets/week1_task4_outcome.npz \
  --window-size 16 --checkpoint-steps 0 40 80 120

python scripts/train_difficulty_baseline.py \
  artifacts/datasets/week1_task4_outcome.npz \
  --output artifacts/results/week1_task4_difficulty_baseline.json

python scripts/train_temporal_mlp.py \
  artifacts/datasets/week1_task4_outcome.npz \
  --output artifacts/results/week1_task4_temporal_mlp.json \
  --device cuda

python scripts/train_frozen_vision_baseline.py \
  artifacts/features/week1_task4_frozen_vision.npz \
  --cameras dual \
  --output artifacts/results/week1_task4_frozen_vision.json \
  --device cuda
```

Reproduce the offline utility and display artifacts:

```bash
python scripts/evaluate_outcome_utility.py \
  artifacts/results/week1_task4_frozen_vision.json \
  docs/manifests/week1_task4_split.json \
  --output artifacts/results/week1_task4_outcome_utility.json \
  --checkpoint-steps 0 40 80 120 \
  --milliseconds-per-step 70.042

python scripts/render_outcome_risk.py \
  artifacts/results/week1_task4_frozen_vision.json \
  docs/manifests/week1_task4_split.json \
  artifacts/results/week1_task4_outcome_utility.json \
  --output-dir artifacts/visualizations/week1_task4 \
  --camera main_camera.mp4

python scripts/analyze_progress_control.py \
  artifacts/features/week1_task4_frozen_vision.npz \
  artifacts/results/week1_task4_frozen_vision.json \
  artifacts/results/week1_task4_temporal_mlp.json \
  --output artifacts/results/week1_task4_progress_control.json \
  --samples 10000 --seed 20260914
```

Reproduce the RQ1 reports from the recorded paired cohort:

```bash
python scripts/evaluate_a1_monitor.py \
  --output artifacts/results/rq1_a1_monitor.json

python scripts/evaluate_a2_consistency.py \
  --normal-root artifacts/rq1/task4/none \
  --action-swap-root artifacts/rq1/task4/action_swap_xy \
  --camera-swap-root artifacts/rq1/task4/camera_swap \
  --calibration-count 10 --detection-horizon 40 \
  --output artifacts/results/rq1_task4_a2_consistency.json
```

Apply the frozen v0.1 predictors to the already collected confirmation cohort:

```bash
python scripts/build_outcome_dataset.py \
  docs/manifests/v02_task4_confirmation.json \
  --output artifacts/v02/datasets/task4_confirmation_outcome.npz

python scripts/extract_frozen_vision_features.py \
  docs/manifests/v02_task4_confirmation.json \
  --output artifacts/v02/features/task4_confirmation_frozen_vision.npz \
  --device cuda

python scripts/evaluate_frozen_confirmation.py \
  artifacts/v02/datasets/task4_confirmation_outcome.npz \
  artifacts/v02/features/task4_confirmation_frozen_vision.npz \
  --temporal-model artifacts/results/week1_task4_temporal_mlp.pt \
  --vision-model artifacts/results/week1_task4_frozen_vision.pt \
  --reference-report artifacts/results/week1_task4_temporal_mlp.json \
  --output artifacts/results/v02_task4_frozen_confirmation.json \
  --device cuda
```

Rebuild and evaluate the v0.3 stage cohort from its frozen manifest:

```bash
python scripts/build_outcome_dataset.py \
  docs/manifests/v03_task4_stage_cohort.json \
  --checkpoint-steps 80 120 \
  --output artifacts/v03/datasets/task4_stage_outcome.npz

python scripts/extract_frozen_vision_features.py \
  docs/manifests/v03_task4_stage_cohort.json \
  --checkpoint-steps 80 \
  --cameras main_camera wrist_camera \
  --output artifacts/v03/features/task4_stage_frozen_vision.npz \
  --device cuda

python scripts/evaluate_frozen_stage_cohort.py \
  artifacts/v03/datasets/task4_stage_outcome.npz \
  artifacts/v03/features/task4_stage_frozen_vision.npz \
  docs/manifests/v03_task4_stage_cohort.json \
  --temporal-model artifacts/results/week1_task4_temporal_mlp.pt \
  --vision-model artifacts/results/week1_task4_frozen_vision.pt \
  --output artifacts/results/v03_task4_frozen_stage_cohort.json \
  --device cuda

python scripts/analyze_stage_aligned_signal.py \
  docs/manifests/v03_task4_stage_cohort.json \
  artifacts/results/v03_task4_frozen_stage_cohort.json \
  --output artifacts/results/v03_task4_stage_aligned_signal.json
```

Re-evaluate the frozen v0.4 pickup-stall models on the already built v0.5 inputs:

```bash
python scripts/evaluate_frozen_pickup_stall_confirmation.py \
  artifacts/v05/datasets/task4_pickup_stall_confirmation.npz \
  artifacts/v05/features/task4_pickup_stall_confirmation_frozen_vision.npz \
  --pilot-temporal artifacts/v04/datasets/task4_pickup_stall_pilot_strict.npz \
  --pilot-report artifacts/results/v04_task4_pickup_stall_pilot_strict.json \
  --model artifacts/results/v04_task4_pickup_stall_pilot_strict.pt \
  --confirmation-manifest docs/manifests/v05_task4_pickup_stall_confirmation.json \
  --output artifacts/results/v05_task4_pickup_stall_frozen_confirmation.json \
  --bootstrap-samples 10000 --device cuda
```

## Integrity check before publishing

Run this on the experiment host after all artifacts are frozen, store the output with
the release, and do not edit the listed artifacts afterward:

```bash
sha256sum \
  docs/manifests/week1_task4_split.json \
  artifacts/datasets/week1_task4_outcome.npz \
  artifacts/features/week1_task4_frozen_vision.npz \
  artifacts/results/week1_task4_*.json \
  artifacts/visualizations/week1_task4/* \
  > artifacts/release-sha256.txt
```

The final 2026-09-14 manifest contains 21 entries and passed `sha256sum -c` for
all 21 files.

## Claim boundary

Phase 1 supports reproducible same-task outcome association, an offline efficiency
analysis, a bounded A1/A2 supporting study, and a task-progress confounding result.
Phase 2 adds independently confirmed visual ranking signal for one operational
pickup-stall event after controlling current stage and initial state. The
predeclared support gate remains unmet because same-state negative coverage is 59%.
It does not establish held-out task generalization, general impending unsafe-event
detection, safe-stop effectiveness, learned A2 monitoring, or coverage beyond the
explicitly tested event and fault cases.
