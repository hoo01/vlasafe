"""Build causal outcome samples from a frozen episode split manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vlasafe.outcome_dataset import build_history_window, flatten_proprioception
from vlasafe.predictor_inputs import select_deploy_inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--checkpoint-steps", type=int, nargs="+", default=[0, 40, 80, 120])
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records: dict[str, list] = {
        "state": [],
        "action": [],
        "timing": [],
        "mask": [],
        "frame_id": [],
        "initial_state": [],
        "failure": [],
        "checkpoint_step": [],
        "episode_id": [],
        "episode_path": [],
        "split": [],
    }
    skipped: list[dict[str, object]] = []

    for split_name, episodes in manifest["splits"].items():
        for episode in episodes:
            episode_path = Path(episode["path"])
            rows = [
                json.loads(line)
                for line in (episode_path / "steps.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            initial_deploy = select_deploy_inputs(rows[0], fields=("proprioception",))
            initial_state = flatten_proprioception(initial_deploy["proprioception"])
            for checkpoint_step in args.checkpoint_steps:
                if checkpoint_step >= len(rows):
                    skipped.append(
                        {
                            "episode_id": episode["episode_id"],
                            "checkpoint_step": checkpoint_step,
                            "num_steps": len(rows),
                        }
                    )
                    continue
                window = build_history_window(rows, checkpoint_step, args.window_size)
                for key in ("state", "action", "timing", "mask", "frame_id"):
                    records[key].append(window[key])
                records["initial_state"].append(initial_state)
                records["failure"].append(not episode["success"])
                records["checkpoint_step"].append(checkpoint_step)
                records["episode_id"].append(episode["episode_id"])
                records["episode_path"].append(episode_path.as_posix())
                records["split"].append(split_name)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        state=np.stack(records["state"]),
        action=np.stack(records["action"]),
        timing=np.stack(records["timing"]),
        mask=np.stack(records["mask"]),
        frame_id=np.stack(records["frame_id"]),
        initial_state=np.stack(records["initial_state"]),
        failure=np.asarray(records["failure"], dtype=np.int64),
        checkpoint_step=np.asarray(records["checkpoint_step"], dtype=np.int64),
        episode_id=np.asarray(records["episode_id"]),
        episode_path=np.asarray(records["episode_path"]),
        split=np.asarray(records["split"]),
    )
    metadata_path = args.output.with_suffix(".json")
    metadata = {
        "source_manifest": str(args.manifest),
        "window_size": args.window_size,
        "checkpoint_steps": args.checkpoint_steps,
        "num_samples": len(records["failure"]),
        "skipped": skipped,
        "feature_shapes": {
            "state": [args.window_size, 25],
            "action": [args.window_size, 7],
            "timing": [args.window_size, 2],
            "mask": [args.window_size],
        },
        "target": "episode_failure",
        "privileged_inputs": [],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"OUTCOME DATASET {args.output}")
    print(f"METADATA {metadata_path}")
    print(f"samples: {metadata['num_samples']}")
    print(f"skipped: {len(skipped)}")
    for split_name in manifest["splits"]:
        indexes = [i for i, name in enumerate(records["split"]) if name == split_name]
        labels = [records["failure"][i] for i in indexes]
        print(
            split_name,
            "samples=", len(indexes),
            "episodes=", len({records["episode_id"][i] for i in indexes}),
            "failure_rate=", sum(labels) / len(labels),
        )


if __name__ == "__main__":
    main()
