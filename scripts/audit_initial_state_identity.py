"""Audit whether recorded initial_state_id values identify actual LIBERO reset states."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig


def _simulator(env: Any) -> Any:
    wrapper = env.envs[0]
    libero_env = getattr(wrapper, "_env", wrapper)
    sim = getattr(libero_env, "sim", None)
    if sim is None:
        raise RuntimeError("could not locate LIBERO simulator through env.envs[0]._env.sim")
    return sim


def _flat_state(sim: Any) -> np.ndarray:
    if hasattr(sim, "get_state"):
        state = sim.get_state()
        if hasattr(state, "flatten"):
            return np.asarray(state.flatten(), dtype=np.float64)
    data = sim.data
    parts = [np.asarray(data.qpos), np.asarray(data.qvel)]
    if hasattr(data, "act") and data.act is not None:
        parts.append(np.asarray(data.act))
    return np.concatenate([part.reshape(-1) for part in parts]).astype(np.float64)


def _fingerprint(state: np.ndarray, decimals: int) -> str:
    rounded = np.round(np.asarray(state, dtype="<f8"), decimals=decimals)
    return hashlib.sha256(rounded.tobytes()).hexdigest()


def _first_frame(path: Path) -> np.ndarray:
    reader = imageio.get_reader(path)
    try:
        return np.asarray(reader.get_data(0))
    finally:
        reader.close()


def _frame_error(recorded: np.ndarray, replayed: np.ndarray) -> dict[str, float]:
    left = recorded.astype(np.float64)
    right = replayed.astype(np.float64)
    if left.shape != right.shape:
        raise ValueError(f"frame shape mismatch: recorded={left.shape}, replayed={right.shape}")
    difference = np.abs(left - right)
    return {
        "mae": float(difference.mean()),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "max_abs": float(difference.max()),
    }


def _manifest_rows(path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for split, episodes in manifest["splits"].items():
        for episode in episodes:
            rows.append({**episode, "split": split})
    return sorted(rows, key=lambda row: str(row["episode_id"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-id", type=int, default=4)
    parser.add_argument("--state-decimals", type=int, default=8)
    parser.add_argument(
        "--limit",
        type=int,
        help="Audit only the first N manifest rows for a smoke test.",
    )
    args = parser.parse_args()

    rows = _manifest_rows(args.manifest)
    if args.limit is not None:
        rows = rows[: args.limit]
    cfg = LiberoEnvConfig(
        task="libero_spatial",
        task_ids=[args.task_id],
        observation_height=360,
        observation_width=360,
        episode_length=280,
    )
    env = cfg.create_envs(n_envs=1, use_async_envs=False)["libero_spatial"][args.task_id]
    audited = []
    try:
        for row in rows:
            observation, _ = env.reset(seed=int(row["seed"]))
            state = _flat_state(_simulator(env))
            episode_path = Path(row["path"])
            main_error = _frame_error(
                _first_frame(episode_path / "main_camera.mp4"),
                np.asarray(observation["pixels"]["image"][0]),
            )
            wrist_error = _frame_error(
                _first_frame(episode_path / "wrist_camera.mp4"),
                np.asarray(observation["pixels"]["image2"][0]),
            )
            audited.append(
                {
                    "episode_id": row["episode_id"],
                    "split": row["split"],
                    "seed": int(row["seed"]),
                    "claimed_initial_state_id": int(row["initial_state_id"]),
                    "replay_state_fingerprint": _fingerprint(state, args.state_decimals),
                    "replay_state_size": int(state.size),
                    "recorded_vs_replay_main_frame": main_error,
                    "recorded_vs_replay_wrist_frame": wrist_error,
                }
            )
    finally:
        env.close()

    by_claimed: dict[int, set[str]] = {}
    by_fingerprint: dict[str, set[int]] = {}
    for row in audited:
        claimed = int(row["claimed_initial_state_id"])
        fingerprint = str(row["replay_state_fingerprint"])
        by_claimed.setdefault(claimed, set()).add(fingerprint)
        by_fingerprint.setdefault(fingerprint, set()).add(claimed)
    output = {
        "schema_version": "0.1.0",
        "purpose": "privileged audit only; fingerprints are not predictor inputs",
        "manifest": str(args.manifest),
        "task": "libero_spatial",
        "task_id": args.task_id,
        "state_fingerprint_round_decimals": args.state_decimals,
        "episodes": len(audited),
        "claimed_ids": len(by_claimed),
        "replay_fingerprints": len(by_fingerprint),
        "claimed_ids_with_multiple_replay_fingerprints": {
            str(key): sorted(value) for key, value in by_claimed.items() if len(value) > 1
        },
        "replay_fingerprints_with_multiple_claimed_ids": {
            key: sorted(value) for key, value in by_fingerprint.items() if len(value) > 1
        },
        "details": audited,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("INITIAL STATE IDENTITY AUDIT", args.output)
    print(
        "episodes", output["episodes"],
        "claimed_ids", output["claimed_ids"],
        "replay_fingerprints", output["replay_fingerprints"],
        "claimed_ids_with_multiple_fingerprints",
        len(output["claimed_ids_with_multiple_replay_fingerprints"]),
        "fingerprints_with_multiple_claimed_ids",
        len(output["replay_fingerprints_with_multiple_claimed_ids"]),
    )
    if audited:
        print(
            "frame_mae_max main",
            max(row["recorded_vs_replay_main_frame"]["mae"] for row in audited),
            "wrist",
            max(row["recorded_vs_replay_wrist_frame"]["mae"] for row in audited),
        )


if __name__ == "__main__":
    main()
