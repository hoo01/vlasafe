"""Generate a deterministic A1 runtime-monitor case table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vlasafe.monitors import (
    ActionProtocolError,
    TimestampProtocolError,
    validate_action_for_execution,
    validate_observation_timestamp,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    low, high = np.full(7, -1.0), np.full(7, 1.0)
    cases = []

    def action_case(name: str, action: object, expected: str) -> None:
        try:
            _, result = validate_action_for_execution(action, low, high)
            observed = "range_intercepted" if result["range_violation"] else "accepted"
        except ActionProtocolError:
            observed = "rejected"
        cases.append({"case": name, "expected": expected, "observed": observed, "pass": observed == expected})

    action_case("valid", np.zeros((1, 7), dtype=np.float32), "accepted")
    action_case("wrong_shape", np.zeros((1, 6), dtype=np.float32), "rejected")
    action_case("non_numeric_dtype", np.asarray([["x"] * 7]), "rejected")
    nan = np.zeros((1, 7)); nan[0, 0] = np.nan
    action_case("nan", nan, "rejected")
    inf = np.zeros((1, 7)); inf[0, 0] = np.inf
    action_case("inf", inf, "rejected")
    high_range = np.zeros((1, 7)); high_range[0, 2] = 2.0
    action_case("above_range", high_range, "range_intercepted")
    low_range = np.zeros((1, 7)); low_range[0, 2] = -2.0
    action_case("below_range", low_range, "range_intercepted")

    for name, current, previous, expected in (
        ("timestamp_valid", 100, None, "accepted"),
        ("timestamp_wrong_type", 100.0, None, "rejected"),
        ("timestamp_negative", -1, None, "rejected"),
        ("timestamp_regressed", 99, 100, "rejected"),
    ):
        try:
            validate_observation_timestamp(current, previous)
            observed = "accepted"
        except TimestampProtocolError:
            observed = "rejected"
        cases.append({"case": name, "expected": expected, "observed": observed, "pass": observed == expected})

    output = {
        "regime": "A1 syntax/protocol",
        "cases": cases,
        "passed": sum(row["pass"] for row in cases),
        "total": len(cases),
        "all_passed": all(row["pass"] for row in cases),
        "range_semantics": "intercept and clip before env.step; structural/non-finite errors are rejected",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("A1 MONITOR", args.output, f"{output['passed']}/{output['total']}")
    if not output["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
