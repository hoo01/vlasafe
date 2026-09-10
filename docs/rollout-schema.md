# Rollout recording contract

VLA-SafeBench keeps LeRobot's native recording as the canonical RGB/action dataset and writes a synchronized sidecar for runtime-safety fields that the standard evaluator does not capture.

## Episode lifecycle

An active episode is stored as `<episode_id>.incomplete`. It becomes a finalized episode only after `result.json` and `COMPLETE` are written and the directory is atomically renamed to `<episode_id>`. Training and evaluation manifests must select only directories containing `COMPLETE`.

## Alignment

For the MVP, `frame_id == step_id`. A row in `steps.jsonl` describes the observation used for inference, the predicted action chunk, the action actually executed, the resulting outcome fields, and label-only simulator diagnostics. IDs must be contiguous from zero.

## Information boundary

`proprioception`, actions, RGB referenced by `frame_id`, and `deploy_metadata` may be used by predictors. `label_only` may be used only for label generation and evaluation. Predictor dataloaders must explicitly reject `label_only` rather than silently ignoring arbitrary extra fields.

## Native LeRobot fields reused

LeRobot evaluation recording already stores visual observations, action, reward, success, done, task description, timestamps, frame indices, and episode indices. The sidecar adds predicted-versus-executed action separation, inference/control timestamps and latency, checkpoint/config provenance, simulator event labels, and an atomic completion marker.

## Artifact validation

Run the validator before adding an episode to any manifest:

```bash
PYTHONPATH=src python scripts/validate_episode.py artifacts/episodes/<episode_id>
```

The validator rejects incomplete episodes, missing required files or fields, malformed/non-finite records, mismatched episode IDs, noncontiguous step/frame IDs, nonmonotonic timestamps, inconsistent `result.num_steps`, and MP4/JSONL frame-count mismatches. Video inspection requires `ffprobe`.
