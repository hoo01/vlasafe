"""Render deterministic contact sheets for physical review of one event rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont


def choose_review_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Keep all success triggers, then span states and event-time quantiles."""
    success = sorted(
        (row for row in rows if not row["failure"]),
        key=lambda row: (row["event_step"], row["episode_id"]),
    )
    failures = sorted(
        (row for row in rows if row["failure"]),
        key=lambda row: (row["event_step"], row["episode_id"]),
    )
    selected = list(success)
    remaining = max(0, limit - len(selected))
    if not remaining or not failures:
        return selected[:limit]

    # First maximize independent-state coverage. Within each state, choose the
    # median event time so the result is not biased toward unusually early events.
    by_state: dict[int, list[dict[str, Any]]] = {}
    for row in failures:
        by_state.setdefault(int(row["initial_state_id"]), []).append(row)
    representatives = [
        values[len(values) // 2]
        for _, values in sorted(by_state.items())
    ]
    if len(representatives) > remaining:
        indexes = {
            round(index * (len(representatives) - 1) / max(remaining - 1, 1))
            for index in range(remaining)
        }
        representatives = [representatives[index] for index in sorted(indexes)]
    selected.extend(representatives[:remaining])

    if len(selected) < limit:
        used = {row["episode_id"] for row in selected}
        pool = [row for row in failures if row["episode_id"] not in used]
        needed = min(limit - len(selected), len(pool))
        indexes = {
            round(index * (len(pool) - 1) / max(needed - 1, 1))
            for index in range(needed)
        }
        selected.extend(pool[index] for index in sorted(indexes))
    return selected[:limit]


def load_frames(path: Path, indexes: list[int]) -> list[Image.Image]:
    reader = imageio.get_reader(path)
    try:
        return [Image.fromarray(reader.get_data(index)).convert("RGB") for index in indexes]
    finally:
        reader.close()


def contact_sheet(
    episode_path: Path,
    row: dict[str, Any],
    wait: int,
    num_steps: int,
    output: Path,
) -> None:
    event = int(row["event_step"])
    approach = event - wait
    indexes = sorted(
        {
            min(num_steps - 1, max(0, step))
            for step in (approach - 5, approach, approach + wait // 2, event, event + 10)
        }
    )
    cameras = ["main_camera.mp4", "wrist_camera.mp4"]
    frames = [load_frames(episode_path / camera, indexes) for camera in cameras]
    tile_w, tile_h = frames[0][0].size
    header = 54
    sheet = Image.new("RGB", (tile_w * len(indexes), header + tile_h * 2), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    outcome = "FAILURE" if row["failure"] else "SUCCESS TRIGGER"
    draw.text(
        (8, 6),
        f"{row['episode_id']} | state {row['initial_state_id']} | {outcome}",
        fill="black",
        font=font,
    )
    draw.text(
        (8, 25),
        f"approach step {approach}; pickup-stall confirmed at {event} after wait={wait}",
        fill="black",
        font=font,
    )
    for camera_index, camera_frames in enumerate(frames):
        for column, (step, frame) in enumerate(zip(indexes, camera_frames, strict=True)):
            x, y = column * tile_w, header + camera_index * tile_h
            sheet.paste(frame, (x, y))
            ImageDraw.Draw(sheet, "RGBA").rectangle((x, y, x + 92, y + 18), fill=(0, 0, 0, 180))
            ImageDraw.Draw(sheet).text((x + 4, y + 3), f"{cameras[camera_index][:5]} s={step}", fill="white", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("feasibility_report", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--rule", default="pickup_stall_approach0.10_w40")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.limit <= 0:
        raise ValueError("limit must be positive")

    report = json.loads(args.feasibility_report.read_text(encoding="utf-8"))
    rule = next((row for row in report["candidate_rules"] if row["name"] == args.rule), None)
    if rule is None:
        raise ValueError(f"rule not found: {args.rule}")
    if rule["event_type"] != "pickup_stall":
        raise ValueError("this renderer currently supports pickup_stall rules only")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    episodes = {
        str(row["episode_id"]): row
        for split in manifest["splits"].values()
        for row in split
    }
    selected = choose_review_rows(rule["detections"], args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    review_rows = []
    wait = int(rule["parameters"]["wait"])
    for index, row in enumerate(selected):
        episode = episodes[str(row["episode_id"])]
        image_name = f"{index:02d}_{row['episode_id']}.jpg"
        contact_sheet(
            Path(episode["path"]),
            row,
            wait,
            int(episode["num_steps"]),
            args.output_dir / image_name,
        )
        review_rows.append({**row, "contact_sheet": image_name})
    review = {
        "schema_version": "0.1.0",
        "purpose": "physical-validity review only; not model selection",
        "rule": args.rule,
        "selection": "all success triggers, then deterministic state/time coverage",
        "rows": review_rows,
    }
    path = args.output_dir / "review.json"
    path.write_text(json.dumps(review, indent=2), encoding="utf-8")
    print("EVENT REVIEW", path, "sheets", len(review_rows))
    print("success triggers", sum(not row["failure"] for row in review_rows))
    print("failure states", len({row["initial_state_id"] for row in review_rows if row["failure"]}))


if __name__ == "__main__":
    main()
