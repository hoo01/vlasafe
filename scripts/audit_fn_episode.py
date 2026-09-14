"""Extract post-checkpoint frames and diagnostics for the Week-2 false negative."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def read_frame(path: Path, index: int) -> Image.Image:
    reader = imageio.get_reader(str(path))
    try:
        frame = np.asarray(reader.get_data(index))[:, :, :3]
    finally:
        reader.close()
    return Image.fromarray(frame.astype(np.uint8)).convert("RGB")


def flatten_proprio(row: dict) -> np.ndarray:
    values = []
    for key in sorted(row["proprioception"]):
        values.extend(np.asarray(row["proprioception"][key], dtype=float).reshape(-1))
    return np.asarray(values)


def active_diagnostics(value, prefix="") -> list[str]:
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else key
            found.extend(active_diagnostics(child, name))
    elif isinstance(value, bool) and value:
        found.append(prefix)
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episode",
        type=Path,
        default=Path("artifacts/week1/task4/episodes/smolvla-20260911T013117405943Z"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/audits/week2_fn_013117405943Z"),
    )
    parser.add_argument("--steps", type=int, nargs="+", default=[80, 120, 160, 200, 240, 279])
    args = parser.parse_args()

    rows = [json.loads(line) for line in (args.episode / "steps.jsonl").read_text().splitlines()]
    result = json.loads((args.episode / "result.json").read_text())
    if not rows:
        raise ValueError("empty steps.jsonl")
    steps = [step for step in args.steps if 0 <= step < len(rows)]
    if not steps:
        raise ValueError("no requested steps exist")

    actions = np.asarray([row["executed_action"] for row in rows], dtype=float)
    action_norm = np.linalg.norm(actions, axis=1)
    action_delta = np.r_[0.0, np.linalg.norm(np.diff(actions, axis=0), axis=1)]
    states = np.stack([flatten_proprio(row) for row in rows])
    state_delta = np.r_[0.0, np.linalg.norm(np.diff(states, axis=0), axis=1)]
    active = [active_diagnostics(row.get("label_only", {})) for row in rows]
    event_steps = [(index, names) for index, names in enumerate(active) if names]
    clipped_steps = [
        index for index, row in enumerate(rows)
        if bool(row.get("deploy_metadata", {}).get("action_clipped", False))
    ]

    window = 16
    rolling_state_motion = np.asarray([
        float(state_delta[max(1, index - window + 1): index + 1].sum())
        for index in range(len(rows))
    ])
    post80 = np.arange(len(rows)) >= 80
    post80_indexes = np.flatnonzero(post80)
    lowest_motion = post80_indexes[np.argsort(rolling_state_motion[post80])[:10]].tolist()

    print("=== EPISODE RESULT ===")
    print(json.dumps(result, indent=2))
    print("RECORDED_STEPS", len(rows))
    print("LAST_RECORDED_STEP", len(rows) - 1)
    print("NOTE requested step 280 is outside a 280-frame episode; frame 279 is used")
    print("ACTIVE_DIAGNOSTIC_STEPS", json.dumps(event_steps))
    print("ACTION_CLIPPED_COUNT", len(clipped_steps))
    print("ACTION_CLIPPED_FIRST_LAST", clipped_steps[:1], clipped_steps[-1:] if clipped_steps else [])
    print("LOWEST_POST80_16STEP_STATE_MOTION_ENDPOINTS", lowest_motion)
    print("\n=== CHECKPOINT DIAGNOSTICS ===")
    print("step,reward,success,terminated,truncated,action_norm,action_delta,state_delta,rolling16_state_motion,active_diagnostics,executed_action")
    diagnostics = []
    for step in steps:
        row = rows[step]
        item = {
            "step": step,
            "reward": row.get("reward"),
            "success": row.get("success"),
            "terminated": row.get("terminated"),
            "truncated": row.get("truncated"),
            "action_norm": float(action_norm[step]),
            "action_delta": float(action_delta[step]),
            "state_delta": float(state_delta[step]),
            "rolling16_state_motion": float(rolling_state_motion[step]),
            "active_diagnostics": active[step],
            "executed_action": actions[step].tolist(),
        }
        diagnostics.append(item)
        print(
            f"{step},{item['reward']},{item['success']},{item['terminated']},{item['truncated']},"
            f"{item['action_norm']:.6f},{item['action_delta']:.6f},{item['state_delta']:.6f},"
            f"{item['rolling16_state_motion']:.6f},{json.dumps(item['active_diagnostics'])},"
            f"{json.dumps(item['executed_action'])}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cell_width, cell_height, caption_height = 720, 720, 45
    sheet = Image.new("RGB", (cell_width * 2, (cell_height + caption_height) * len(steps)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for row_index, step in enumerate(steps):
        for column, (title, filename) in enumerate((("main", "main_camera.mp4"), ("wrist", "wrist_camera.mp4"))):
            frame = read_frame(args.episode / filename, step)
            frame.thumbnail((cell_width, cell_height))
            x = column * cell_width
            y = row_index * (cell_height + caption_height)
            sheet.paste(frame, (x, y))
            draw.text(
                (x + 5, y + cell_height + 4),
                f"{title} step={step} action_norm={action_norm[step]:.4f} state_delta={state_delta[step]:.4f}",
                fill="black", font=font,
            )
    sheet_path = args.output_dir / "fn_post80_contact_sheet.jpg"
    sheet.save(sheet_path, quality=95)
    report_path = args.output_dir / "fn_post80_diagnostics.json"
    report_path.write_text(json.dumps({
        "episode": str(args.episode),
        "result": result,
        "recorded_steps": len(rows),
        "checkpoints": diagnostics,
        "active_diagnostic_steps": event_steps,
        "action_clipped_steps": clipped_steps,
        "lowest_post80_16step_state_motion_endpoints": lowest_motion,
    }, indent=2))
    print("\n=== OUTPUTS ===")
    print(sheet_path)
    print(report_path)


if __name__ == "__main__":
    main()
