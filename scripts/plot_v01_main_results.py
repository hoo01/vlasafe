"""Render the compact v0.1 AUPRC/AUROC figure used by the README."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STEPS = [0, 40, 80, 120]
DOCUMENTED_SNAPSHOT = {
    "Train prevalence": {
        "auprc": [0.700, 0.700, 0.700, 0.700],
        "auroc": [0.500, 0.500, 0.500, 0.500],
    },
    "Initial proprio": {
        "auprc": [0.799, 0.799, 0.799, 0.799],
        "auroc": [0.452, 0.452, 0.452, 0.452],
    },
    "Temporal MLP": {
        "auprc": [0.609, 0.856, 0.909, 0.982],
        "auroc": [0.286, 0.571, 0.762, 0.952],
    },
    "Frozen dual-camera vision": {
        "auprc": [0.652, 0.856, 0.982, 0.982],
        "auroc": [0.190, 0.571, 0.952, 0.952],
    },
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_reports(
    difficulty_path: Path, temporal_path: Path, vision_path: Path
) -> dict[str, dict[str, list[float]]]:
    """Load the plotted metrics directly from the three frozen v0.1 reports."""
    difficulty = _read(difficulty_path)["results"]["test"]
    temporal = _read(temporal_path)["results"]["test"]
    vision = _read(vision_path)["results"]["test"]

    def series(report: dict[str, Any], key: str, metric: str) -> list[float]:
        return [float(report[str(step)][key][metric]) for step in STEPS]

    return {
        "Train prevalence": {
            metric: [float(difficulty["prevalence"][metric])] * len(STEPS)
            for metric in ("auprc", "auroc")
        },
        "Initial proprio": {
            metric: [float(difficulty["initial_proprio_logistic"][metric])] * len(STEPS)
            for metric in ("auprc", "auroc")
        },
        "Temporal MLP": {
            metric: series(temporal, "temporal_mlp", metric)
            for metric in ("auprc", "auroc")
        },
        "Frozen dual-camera vision": {
            metric: series(vision, "checkpoint_vision", metric)
            for metric in ("auprc", "auroc")
        },
    }


def render(data: dict[str, dict[str, list[float]]], output_base: Path) -> None:
    colors = {
        "Train prevalence": "#6B7280",
        "Initial proprio": "#CC79A7",
        "Temporal MLP": "#0072B2",
        "Frozen dual-camera vision": "#D55E00",
    }
    styles = {
        "Train prevalence": "--",
        "Initial proprio": ":",
        "Temporal MLP": "-",
        "Frozen dual-camera vision": "-",
    }
    markers = {
        "Train prevalence": "o",
        "Initial proprio": "s",
        "Temporal MLP": "^",
        "Frozen dual-camera vision": "D",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharex=True, sharey=True)
    for axis, metric, title in zip(
        axes, ("auprc", "auroc"), ("AUPRC", "AUROC"), strict=True
    ):
        for name, values in data.items():
            axis.plot(
                STEPS,
                values[metric],
                label=name,
                color=colors[name],
                linestyle=styles[name],
                marker=markers[name],
                linewidth=2.2,
                markersize=6,
            )
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel("Checkpoint step")
        axis.set_xticks(STEPS)
        axis.set_ylim(0.0, 1.04)
        axis.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        axis.grid(axis="y", alpha=0.25, linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Score")
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("VLA-SafeBench v0.1 outcome ranking", fontweight="bold", fontsize=14)
    fig.text(0.5, 0.015, "Frozen held-out initial-state test, n=10", ha="center", color="#4B5563")
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))

    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.with_suffix(".png")
    svg_path = output_base.with_suffix(".svg")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    # Matplotlib emits trailing spaces in SVG path data; normalize for clean Git diffs.
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty-report", type=Path)
    parser.add_argument("--temporal-report", type=Path)
    parser.add_argument("--vision-report", type=Path)
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path("docs/figures/v01_main_ranking"),
        help="Output path without extension; both PNG and SVG are written.",
    )
    args = parser.parse_args()

    report_paths = [args.difficulty_report, args.temporal_report, args.vision_report]
    if any(report_paths) and not all(report_paths):
        parser.error("provide all three frozen reports, or omit all to use the documented snapshot")
    if all(report_paths):
        data = load_reports(*report_paths)
        source = {"mode": "frozen_reports", "paths": [str(path) for path in report_paths]}
    else:
        data = DOCUMENTED_SNAPSHOT
        source = {
            "mode": "documented_snapshot",
            "note": "Values match README section 6.1 and the frozen v0.1 result reports.",
        }

    render(data, args.output_base)
    metadata = {
        "title": "VLA-SafeBench v0.1 outcome ranking",
        "test_episodes": 10,
        "checkpoint_steps": STEPS,
        "source": source,
        "metrics": data,
        "outputs": [
            str(args.output_base.with_suffix(".png")),
            str(args.output_base.with_suffix(".svg")),
        ],
    }
    metadata_path = args.output_base.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("MAIN RESULT FIGURE", args.output_base.with_suffix(".png"))
    print("VECTOR FIGURE", args.output_base.with_suffix(".svg"))
    print("METADATA", metadata_path)


if __name__ == "__main__":
    main()
