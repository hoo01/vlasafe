"""Render test outcome-risk curves and one offline-trigger overlay video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def prediction_index(report: dict[str, Any], split: str = "test") -> dict[str, dict[int, float]]:
    indexed: dict[str, dict[int, float]] = {}
    for row in report["predictions"]:
        if row["split"] != split:
            continue
        indexed.setdefault(str(row["episode_id"]), {})[
            int(row["checkpoint_step"])
        ] = float(row["checkpoint_vision_probability"])
    return indexed


def test_episode_index(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(episode["episode_id"]): episode
        for episode in manifest["splits"]["test"]
    }


def risk_chart(
    predictions: dict[str, dict[int, float]],
    episodes: dict[str, dict[str, Any]],
    threshold: float,
    output: Path,
) -> None:
    width, height = 1280, 760
    left, top, right, bottom = 100, 70, 40, 100
    plot_w, plot_h = width - left - right, height - top - bottom
    checkpoints = sorted({step for values in predictions.values() for step in values})
    max_step = max(checkpoints)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    def xy(step: int, risk: float) -> tuple[float, float]:
        x = left + plot_w * step / max_step
        y = top + plot_h * (1.0 - risk)
        return x, y

    for risk in np.linspace(0, 1, 6):
        _, y = xy(0, float(risk))
        draw.line((left, y, width - right, y), fill=(225, 225, 225), width=1)
        draw.text((25, y - 7), f"{risk:.1f}", fill="black", font=font)
    for step in checkpoints:
        x, _ = xy(step, 0)
        draw.line((x, top, x, height - bottom), fill=(235, 235, 235), width=1)
        draw.text((x - 10, height - bottom + 12), str(step), fill="black", font=font)
    _, threshold_y = xy(0, threshold)
    draw.line((left, threshold_y, width - right, threshold_y), fill=(180, 0, 180), width=2)
    draw.text((left + 8, threshold_y - 18), f"validation threshold {threshold:.6f}", fill=(150, 0, 150), font=font)

    colors = {True: (190, 35, 35), False: (35, 120, 55)}
    for episode_id in sorted(predictions):
        failure = not bool(episodes[episode_id]["success"])
        points = [xy(step, predictions[episode_id][step]) for step in checkpoints]
        draw.line(points, fill=colors[failure], width=3)
        for x, y in points:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=colors[failure])
        x, y = points[-1]
        draw.text((x + 5, y - 7), episode_id[-8:], fill=colors[failure], font=font)

    draw.text((left, 24), "Frozen dual-camera outcome risk on held-out test episodes", fill="black", font=font)
    draw.text((left, height - 45), "green = eventual success; red = eventual failure; risk updates only at frozen checkpoints", fill="black", font=font)
    draw.text((width // 2 - 40, height - 45), "checkpoint step", fill="black", font=font)
    draw.text((10, 10), "failure probability", fill="black", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def render_overlay(
    video_path: Path,
    output: Path,
    episode_id: str,
    probabilities: dict[int, float],
    threshold: float,
    trigger_step: int,
) -> int:
    reader = imageio.get_reader(video_path)
    metadata = reader.get_meta_data()
    fps = float(metadata.get("fps", 20.0))
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        output,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    font = ImageFont.load_default()
    checkpoints = sorted(probabilities)
    frames = 0
    try:
        for step, frame in enumerate(reader):
            observed = [checkpoint for checkpoint in checkpoints if checkpoint <= step]
            current_checkpoint = max(observed) if observed else checkpoints[0]
            risk = probabilities[current_checkpoint]
            triggered = step >= trigger_step
            image = Image.fromarray(np.asarray(frame)).convert("RGB")
            draw = ImageDraw.Draw(image, "RGBA")
            draw.rectangle((8, 8, image.width - 8, 92), fill=(0, 0, 0, 175))
            draw.text((18, 16), f"episode {episode_id}", fill="white", font=font)
            draw.text(
                (18, 34),
                f"step {step} | frozen-checkpoint risk {risk:.6f} (checkpoint {current_checkpoint})",
                fill="white",
                font=font,
            )
            status = (
                f"OFFLINE TRIGGER at step {trigger_step}; recorded rollout continues for analysis"
                if triggered
                else f"validation threshold {threshold:.6f} | no trigger yet"
            )
            draw.text(
                (18, 54),
                status,
                fill=(255, 210, 40) if triggered else (210, 210, 210),
                font=font,
            )
            draw.text((18, 72), "final outcome: FAILURE (offline efficiency analysis, not safe stop)", fill=(255, 110, 110), font=font)
            writer.append_data(np.asarray(image))
            frames += 1
    finally:
        reader.close()
        writer.close()
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("vision_report", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("utility_report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera", default="main_camera.mp4")
    args = parser.parse_args()

    vision = load_json(args.vision_report)
    manifest = load_json(args.manifest)
    utility = load_json(args.utility_report)
    predictions = prediction_index(vision)
    episodes = test_episode_index(manifest)
    threshold = float(utility["threshold_selection"]["selected_threshold"])
    utility_rows = utility["test"]["episodes_detail"]
    candidates = [
        row for row in utility_rows
        if row["failure"] and row["stop_step"] is not None
    ]
    if not candidates:
        raise ValueError("test utility has no detected failure to render")
    chosen = max(candidates, key=lambda row: (row["saved_steps"], -row["stop_step"], row["episode_id"]))
    episode_id = str(chosen["episode_id"])
    video_path = Path(episodes[episode_id]["path"]) / args.camera
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    chart_path = args.output_dir / "test_outcome_risk_curves.png"
    video_output = args.output_dir / f"{episode_id}_outcome_risk_overlay.mp4"
    risk_chart(predictions, episodes, threshold, chart_path)
    frames = render_overlay(
        video_path,
        video_output,
        episode_id,
        predictions[episode_id],
        threshold,
        int(chosen["stop_step"]),
    )
    metadata = {
        "vision_report": str(args.vision_report),
        "manifest": str(args.manifest),
        "utility_report": str(args.utility_report),
        "episode_id": episode_id,
        "source_video": str(video_path),
        "output_video": str(video_output),
        "risk_chart": str(chart_path),
        "frames": frames,
        "offline_trigger_step": int(chosen["stop_step"]),
        "threshold": threshold,
        "risk_semantics": "probability of eventual episode failure",
        "intervention_semantics": "offline counterfactual only; recorded rollout was not stopped",
    }
    metadata_path = args.output_dir / "render_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("RISK CHART", chart_path)
    print("OVERLAY VIDEO", video_output)
    print("METADATA", metadata_path)
    print("episode", episode_id, "trigger_step", chosen["stop_step"], "frames", frames)


if __name__ == "__main__":
    main()
