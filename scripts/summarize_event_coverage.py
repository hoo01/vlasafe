"""Summarize unsafe-event coverage for a frozen episode split manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


EVENTS = ("self_collision", "joint_violation", "workspace_violation", "impact")


def _episode_events(episode_dir: Path) -> tuple[dict[str, bool], Counter[str], dict[str, bool]]:
    rows = [
        json.loads(line)
        for line in (episode_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    episode_events = {event: False for event in EVENTS}
    event_steps: Counter[str] = Counter()
    availability = {event: False for event in EVENTS}
    for row in rows:
        labels = row["label_only"]
        declared = labels.get("event_availability", {})
        for event in EVENTS:
            availability[event] |= declared.get(event) is True
            if labels.get(event) is True:
                episode_events[event] = True
                event_steps[event] += 1
    return episode_events, event_steps, availability


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest: dict[str, Any] = json.loads(args.manifest.read_text(encoding="utf-8"))
    episode_counts: Counter[str] = Counter()
    failure_episode_counts: Counter[str] = Counter()
    step_counts: Counter[str] = Counter()
    availability_counts: Counter[str] = Counter()
    split_rows: dict[str, dict[str, int]] = {}
    failures = 0
    failures_with_any_event = 0

    for split_name, episodes in manifest["splits"].items():
        split_failures = 0
        split_failures_with_event = 0
        for episode in episodes:
            is_failure = not episode["success"]
            failures += int(is_failure)
            split_failures += int(is_failure)
            events, event_steps, availability = _episode_events(Path(episode["path"]))
            has_any_event = any(events.values())
            failures_with_any_event += int(is_failure and has_any_event)
            split_failures_with_event += int(is_failure and has_any_event)
            for event in EVENTS:
                episode_counts[event] += int(events[event])
                failure_episode_counts[event] += int(is_failure and events[event])
                step_counts[event] += event_steps[event]
                availability_counts[event] += int(availability[event])
        split_rows[split_name] = {
            "episodes": len(episodes),
            "failures": split_failures,
            "failures_with_any_event": split_failures_with_event,
        }

    result = {
        "manifest": str(args.manifest),
        "episodes": sum(len(items) for items in manifest["splits"].values()),
        "failures": failures,
        "failures_with_any_event": failures_with_any_event,
        "p_event_given_failure": failures_with_any_event / failures if failures else None,
        "event_episode_counts": dict(episode_counts),
        "failure_episode_counts": dict(failure_episode_counts),
        "event_step_counts": dict(step_counts),
        "event_available_episode_counts": dict(availability_counts),
        "splits": split_rows,
    }
    output = args.output or Path("artifacts/reports") / f"{args.manifest.stem}-event-coverage.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"EVENT COVERAGE {output}")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
