"""Audit outcome-independent candidate failure events in the v0.3 cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    from analyze_stage_aligned_signal import (
        PLATE_BODY,
        _body_position,
        select_target,
    )
except ModuleNotFoundError:
    from scripts.analyze_stage_aligned_signal import (
        PLATE_BODY,
        _body_position,
        select_target,
    )


def first_persistent(condition: np.ndarray, persistence: int) -> int | None:
    if persistence <= 0:
        raise ValueError("persistence must be positive")
    run = 0
    for index, value in enumerate(condition):
        run = run + 1 if bool(value) else 0
        if run >= persistence:
            return index - persistence + 1
    return None


def trajectory(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [row["label_only"] for row in rows]
    target = select_target(labels[0])
    bowl = np.stack([_body_position(label, target) for label in labels])
    plate = np.stack([_body_position(label, PLATE_BODY) for label in labels])
    eef = np.stack(
        [np.asarray(label["eef_position"], dtype=np.float64) for label in labels]
    )
    start = bowl[0]
    return {
        "target": target,
        "bowl": bowl,
        "plate": plate,
        "eef": eef,
        "lifted": bowl[:, 2] - start[2] > 0.04,
        "bowl_eef_distance": np.linalg.norm(bowl - eef, axis=1),
        "plate_xy_distance": np.linalg.norm((bowl - plate)[:, :2], axis=1),
        "running_max_height": np.maximum.accumulate(bowl[:, 2]),
        "success": np.asarray([bool(row["success"]) for row in rows]),
    }


def detect_drop(data: dict[str, Any], fall: float, persistence: int) -> int | None:
    ever_lifted = np.maximum.accumulate(data["lifted"])
    condition = (
        ever_lifted
        & (data["running_max_height"] - data["bowl"][:, 2] >= fall)
        & (data["plate_xy_distance"] >= 0.10)
    )
    return first_persistent(condition, persistence)


def detect_grasp_loss(
    data: dict[str, Any], close: float, separation: float, persistence: int
) -> int | None:
    associated = data["lifted"] & (data["bowl_eef_distance"] <= close)
    ever_associated = np.maximum.accumulate(associated)
    condition = (
        ever_associated
        & (data["bowl_eef_distance"] >= separation)
        & (data["plate_xy_distance"] >= 0.10)
    )
    return first_persistent(condition, persistence)


def detect_failed_placement(
    data: dict[str, Any], entered: float, exited: float, persistence: int
) -> int | None:
    ever_entered = np.maximum.accumulate(data["plate_xy_distance"] <= entered)
    condition = ever_entered & (data["plate_xy_distance"] >= exited)
    return first_persistent(condition, persistence)


def detect_pickup_stall(
    data: dict[str, Any], approach: float, wait: int, movement: float
) -> int | None:
    """First close approach not followed by target movement in a fixed window."""
    if wait <= 0:
        raise ValueError("wait must be positive")
    if movement <= 0:
        raise ValueError("movement must be positive")
    approached = np.flatnonzero(data["bowl_eef_distance"] <= approach)
    if not len(approached):
        return None
    # Pickup is a one-shot stage transition. Searching later close approaches
    # can relabel transport/placement as pickup stalls after a valid pickup.
    start = int(approached[0])
    end = start + wait
    if end >= len(data["bowl"]):
        return None
    displacement = np.linalg.norm(
        data["bowl"][start : end + 1] - data["bowl"][start], axis=1
    )
    if float(np.max(displacement)) < movement:
        return end
    return None


def detect_transport_stall(
    data: dict[str, Any], window: int, min_improvement: float
) -> int | None:
    """After lift, detect a window with too little progress toward the plate."""
    if window <= 0:
        raise ValueError("window must be positive")
    distance = data["plate_xy_distance"]
    for end in range(window, len(distance)):
        start = end - window
        if not np.any(data["lifted"][: start + 1]):
            continue
        if distance[end] < 0.10:
            continue
        improvement = float(distance[start] - np.min(distance[start : end + 1]))
        if improvement < min_improvement:
            return end
    return None


def detect_placement_timeout(
    data: dict[str, Any], entered: float, wait: int
) -> int | None:
    """After entering the plate neighborhood, detect non-completion for ``wait`` steps."""
    if wait <= 0:
        raise ValueError("wait must be positive")
    for start in np.flatnonzero(data["plate_xy_distance"] <= entered):
        end = int(start) + wait
        if end >= len(data["success"]):
            continue
        if not np.any(data["success"][start : end + 1]):
            return end
    return None


def candidate_rules() -> list[tuple[str, str, dict[str, Any], Callable[[dict[str, Any]], int | None]]]:
    rules = []
    for fall in (0.04, 0.06, 0.08):
        for persistence in (3, 5, 10):
            parameters = {"fall": fall, "persistence": persistence, "outside_plate_xy": 0.10}
            rules.append(
                (
                    f"drop_fall{fall:.2f}_p{persistence}",
                    "object_drop",
                    parameters,
                    lambda data, fall=fall, persistence=persistence: detect_drop(
                        data, fall, persistence
                    ),
                )
            )
    for separation in (0.12, 0.15, 0.18):
        for persistence in (3, 5, 10):
            parameters = {
                "prior_close": 0.10,
                "separation": separation,
                "persistence": persistence,
                "outside_plate_xy": 0.10,
            }
            rules.append(
                (
                    f"grasp_loss_sep{separation:.2f}_p{persistence}",
                    "grasp_loss",
                    parameters,
                    lambda data, separation=separation, persistence=persistence: detect_grasp_loss(
                        data, 0.10, separation, persistence
                    ),
                )
            )
    for exited in (0.12, 0.15):
        for persistence in (5, 10):
            parameters = {"entered": 0.10, "exited": exited, "persistence": persistence}
            rules.append(
                (
                    f"failed_placement_exit{exited:.2f}_p{persistence}",
                    "failed_placement",
                    parameters,
                    lambda data, exited=exited, persistence=persistence: detect_failed_placement(
                        data, 0.10, exited, persistence
                    ),
                )
            )
    for approach in (0.08, 0.10, 0.12):
        for wait in (20, 40):
            parameters = {
                "bowl_eef_approach": approach,
                "wait": wait,
                "minimum_target_movement": 0.04,
            }
            rules.append(
                (
                    f"pickup_stall_approach{approach:.2f}_w{wait}_move0.04",
                    "pickup_stall",
                    parameters,
                    lambda data, approach=approach, wait=wait: detect_pickup_stall(
                        data, approach, wait, 0.04
                    ),
                )
            )
    for window in (20, 40):
        for improvement in (0.01, 0.02):
            parameters = {
                "window": window,
                "minimum_plate_xy_improvement": improvement,
                "outside_plate_xy": 0.10,
            }
            rules.append(
                (
                    f"transport_stall_w{window}_imp{improvement:.2f}",
                    "transport_stall",
                    parameters,
                    lambda data, window=window, improvement=improvement: detect_transport_stall(
                        data, window, improvement
                    ),
                )
            )
    for entered in (0.08, 0.10):
        for wait in (20, 40):
            parameters = {"plate_xy_entered": entered, "wait_without_success": wait}
            rules.append(
                (
                    f"placement_timeout_enter{entered:.2f}_w{wait}",
                    "placement_timeout",
                    parameters,
                    lambda data, entered=entered, wait=wait: detect_placement_timeout(
                        data, entered, wait
                    ),
                )
            )
    return rules


def summarize_rule(
    name: str,
    event_type: str,
    parameters: dict[str, Any],
    detector: Callable[[dict[str, Any]], int | None],
    episodes: list[dict[str, Any]],
    history: int,
    horizon: int,
    margin: int,
) -> dict[str, Any]:
    detections = []
    for episode in episodes:
        event_step = detector(episode["trajectory"])
        if event_step is not None:
            detections.append(
                {
                    "episode_id": episode["episode_id"],
                    "initial_state_id": episode["initial_state_id"],
                    "failure": episode["failure"],
                    "event_step": int(event_step),
                    "full_h_k_m_available": bool(
                        event_step >= history + horizon + margin
                    ),
                }
            )
    failure_detections = [row for row in detections if row["failure"]]
    success_detections = [row for row in detections if not row["failure"]]
    eligible = [row for row in detections if row["full_h_k_m_available"]]
    return {
        "name": name,
        "event_type": event_type,
        "parameters": parameters,
        "outcome_independent_rule": True,
        "event_episodes": len(detections),
        "failure_event_episodes": len(failure_detections),
        "success_trigger_episodes": len(success_detections),
        "success_trigger_rate": len(success_detections)
        / sum(not episode["failure"] for episode in episodes),
        "initial_state_groups": len({row["initial_state_id"] for row in detections}),
        "failed_initial_state_groups": len(
            {row["initial_state_id"] for row in failure_detections}
        ),
        "full_h_k_m_available": len(eligible),
        "full_h_k_m_fraction": len(eligible) / len(detections) if detections else None,
        "event_step_median": float(np.median([row["event_step"] for row in detections]))
        if detections
        else None,
        "detections": detections,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--margin", type=int, default=20)
    args = parser.parse_args()
    if min(args.history, args.horizon, args.margin) < 0 or args.horizon == 0:
        raise ValueError("history/margin must be non-negative and horizon positive")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = [row for split in manifest["splits"].values() for row in split]
    episodes = []
    for row in rows:
        episode_rows = [
            json.loads(line)
            for line in (Path(row["path"]) / "steps.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        episodes.append(
            {
                "episode_id": str(row["episode_id"]),
                "initial_state_id": int(row["initial_state_id"]),
                "failure": int(not row["success"]),
                "trajectory": trajectory(episode_rows),
            }
        )

    rules = [
        summarize_rule(
            name,
            event_type,
            parameters,
            detector,
            episodes,
            args.history,
            args.horizon,
            args.margin,
        )
        for name, event_type, parameters, detector in candidate_rules()
    ]
    output = {
        "schema_version": "0.1.0",
        "analysis_type": "exploratory failure-event feasibility audit; no predictor training",
        "manifest": args.manifest.as_posix(),
        "episodes": len(episodes),
        "successes": sum(not episode["failure"] for episode in episodes),
        "failures": sum(episode["failure"] for episode in episodes),
        "label_window": {
            "history_h": args.history,
            "future_horizon_k": args.horizon,
            "gray_zone_margin_m": args.margin,
            "positive": "0 < t_event - t <= K",
            "ignore": "K < t_event - t <= K + M",
            "negative": "t_event - t > K + M or no event, subject to stage matching",
        },
        "candidate_rules": rules,
        "selection_warning": "Do not select a formal event rule from outcome performance. Review physical validity and success-trigger videos, then freeze one rule before pilot modeling.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("FAILURE EVENT FEASIBILITY", args.output)
    print("episodes", len(episodes), "success/failure", f"{output['successes']}/{output['failures']}")
    for rule in rules:
        print(
            rule["name"],
            "events", rule["event_episodes"],
            "failure", rule["failure_event_episodes"],
            "success_trigger", rule["success_trigger_episodes"],
            "groups", rule["failed_initial_state_groups"],
            "eligible", rule["full_h_k_m_available"],
        )


if __name__ == "__main__":
    main()
