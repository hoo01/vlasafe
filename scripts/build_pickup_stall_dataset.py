"""Build deploy-input temporal windows for the frozen pickup-stall pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vlasafe.outcome_dataset import build_history_window


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=16)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records: dict[str, list] = {
        key: []
        for key in (
            "state", "action", "timing", "mask", "frame_id", "pickup_stall",
            "checkpoint_step", "first_approach_step", "event_step", "episode_id",
            "checkpoint_target_movement_m", "episode_path", "initial_state_id", "split",
        )
    }
    for split, samples in manifest["splits"].items():
        for sample in samples:
            path = Path(sample["path"])
            rows = [
                json.loads(line)
                for line in (path / "steps.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            checkpoint = int(sample["checkpoint_step"])
            window = build_history_window(rows, checkpoint, args.window_size)
            for key in ("state", "action", "timing", "mask", "frame_id"):
                records[key].append(window[key])
            records["pickup_stall"].append(int(sample["pickup_stall"]))
            records["checkpoint_step"].append(checkpoint)
            records["first_approach_step"].append(int(sample["first_approach_step"]))
            records["event_step"].append(
                -1 if sample["event_step"] is None else int(sample["event_step"])
            )
            records["checkpoint_target_movement_m"].append(
                float(sample["checkpoint_target_movement_m"])
            )
            records["episode_id"].append(str(sample["episode_id"]))
            records["episode_path"].append(path.as_posix())
            records["initial_state_id"].append(int(sample["initial_state_id"]))
            records["split"].append(split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        **{
            key: np.stack(values) if key in {"state", "action", "timing", "mask", "frame_id"}
            else np.asarray(values)
            for key, values in records.items()
        },
    )
    metadata = {
        "schema_version": "0.1.0",
        "source_manifest": args.manifest.as_posix(),
        "target": "pickup_stall_confirmed_within_next_20_steps",
        "window_size": args.window_size,
        "samples": len(records["pickup_stall"]),
        "positives": int(sum(records["pickup_stall"])),
        "feature_shapes": {"state": [args.window_size, 25], "action": [args.window_size, 7], "timing": [args.window_size, 2]},
        "absolute_checkpoint_feature": False,
        "privileged_predictor_inputs": [],
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("PICKUP STALL DATASET", args.output)
    print("samples", metadata["samples"], "positives", metadata["positives"])
    print("state/action/timing", np.stack(records["state"]).shape, np.stack(records["action"]).shape, np.stack(records["timing"]).shape)


if __name__ == "__main__":
    main()
