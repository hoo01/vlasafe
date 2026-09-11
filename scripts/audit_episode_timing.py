"""Audit where fixed outcome checkpoints fall within recorded episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def summarize(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None, "mean": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "min": int(array.min()),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "max": int(array.max()),
        "mean": float(array.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-steps", type=int, nargs="+", default=[0, 40, 80, 120])
    args = parser.parse_args()

    manifest: dict[str, Any] = json.loads(args.manifest.read_text(encoding="utf-8"))
    episodes = [
        {**episode, "split": split}
        for split, split_episodes in manifest["splits"].items()
        for episode in split_episodes
    ]
    if not episodes:
        raise ValueError("manifest contains no episodes")

    groups: dict[str, list[dict[str, Any]]] = {
        "all": episodes,
        "success": [episode for episode in episodes if episode["success"]],
        "failure": [episode for episode in episodes if not episode["success"]],
    }
    report: dict[str, Any] = {
        "manifest": str(args.manifest),
        "remaining_steps_definition": "num_steps - (checkpoint_step + 1), after observing checkpoint step",
        "termination_steps": {
            name: summarize([int(episode["num_steps"]) for episode in selected])
            for name, selected in groups.items()
        },
        "splits": {},
        "checkpoints": {},
    }
    for split in manifest["splits"]:
        selected = [episode for episode in episodes if episode["split"] == split]
        report["splits"][split] = {
            "all": summarize([int(episode["num_steps"]) for episode in selected]),
            "success": summarize(
                [int(episode["num_steps"]) for episode in selected if episode["success"]]
            ),
            "failure": summarize(
                [int(episode["num_steps"]) for episode in selected if not episode["success"]]
            ),
        }

    for checkpoint in sorted(set(args.checkpoint_steps)):
        if checkpoint < 0:
            raise ValueError("checkpoint steps must be non-negative")
        checkpoint_result: dict[str, Any] = {}
        for name, selected in groups.items():
            at_risk = [episode for episode in selected if int(episode["num_steps"]) > checkpoint]
            remaining = [int(episode["num_steps"]) - checkpoint - 1 for episode in at_risk]
            checkpoint_result[name] = {
                "episodes": len(selected),
                "at_risk": len(at_risk),
                "at_risk_fraction": len(at_risk) / len(selected) if selected else None,
                "remaining_steps": summarize(remaining),
            }
        report["checkpoints"][str(checkpoint)] = checkpoint_result

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"TIMING AUDIT {args.output}")
    for name, values in report["termination_steps"].items():
        print("termination", name, values)
    for checkpoint, values in report["checkpoints"].items():
        print(
            "step", checkpoint,
            "at_risk", values["all"]["at_risk"], "/", values["all"]["episodes"],
            "success_remaining", values["success"]["remaining_steps"],
            "failure_remaining", values["failure"]["remaining_steps"],
        )


if __name__ == "__main__":
    main()
