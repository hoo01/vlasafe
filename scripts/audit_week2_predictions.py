"""Audit frozen Week-2 predictions, LOEO stability, and step-80 RGB cases."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vlasafe.outcome_metrics import binary_metrics


def report_index(report: dict, step: int, probability_key: str) -> dict[str, dict]:
    return {
        row["episode_id"]: {
            "failure": int(row["failure"]),
            "probability": float(row[probability_key]),
        }
        for row in report["predictions"]
        if row["split"] == "test" and int(row["checkpoint_step"]) == step
    }


def misordered_pairs(ids, labels, probabilities):
    failures = [(eid, p) for eid, y, p in zip(ids, labels, probabilities, strict=True) if y]
    successes = [(eid, p) for eid, y, p in zip(ids, labels, probabilities, strict=True) if not y]
    wrong = []
    ties = []
    for failure_id, failure_p in failures:
        for success_id, success_p in successes:
            row = {
                "failure_episode": failure_id,
                "failure_risk": float(failure_p),
                "success_episode": success_id,
                "success_risk": float(success_p),
            }
            if failure_p < success_p:
                wrong.append(row)
            elif failure_p == success_p:
                ties.append(row)
    return wrong, ties


def loeo(ids, labels, probabilities):
    full = binary_metrics(labels, probabilities)
    rows = []
    for index, eid in enumerate(ids):
        keep = np.arange(len(ids)) != index
        metrics = binary_metrics(labels[keep], probabilities[keep])
        rows.append({
            "removed_episode": eid,
            "removed_outcome": "failure" if labels[index] else "success",
            "auprc": metrics["auprc"],
            "auroc": metrics["auroc"],
            "delta_auprc_vs_full": metrics["auprc"] - full["auprc"],
            "delta_auroc_vs_full": metrics["auroc"] - full["auroc"],
        })
    return full, rows


def read_frame(path: Path, index: int) -> Image.Image:
    reader = imageio.get_reader(str(path))
    try:
        frame = np.asarray(reader.get_data(index))[:, :, :3]
    finally:
        reader.close()
    return Image.fromarray(frame.astype(np.uint8)).convert("RGB")


def print_loeo(name, full, rows):
    print(f"\n=== LOEO {name} ===")
    print("removed_episode,outcome,auprc,auroc,delta_auprc,delta_auroc")
    for row in rows:
        print(
            f"{row['removed_episode']},{row['removed_outcome']},"
            f"{row['auprc']:.9f},{row['auroc']:.9f},"
            f"{row['delta_auprc_vs_full']:+.9f},{row['delta_auroc_vs_full']:+.9f}"
        )
    worst_pr = min(rows, key=lambda row: row["delta_auprc_vs_full"])
    worst_roc = min(rows, key=lambda row: row["delta_auroc_vs_full"])
    print(f"FULL,AUPRC={full['auprc']:.9f},AUROC={full['auroc']:.9f}")
    print(f"MAX_AUPRC_DROP,{worst_pr['removed_episode']},{worst_pr['delta_auprc_vs_full']:+.9f}")
    print(f"MAX_AUROC_DROP,{worst_roc['removed_episode']},{worst_roc['delta_auroc_vs_full']:+.9f}")
    print(
        f"RANGE,AUPRC={min(r['auprc'] for r in rows):.9f}..{max(r['auprc'] for r in rows):.9f},"
        f"AUROC={min(r['auroc'] for r in rows):.9f}..{max(r['auroc'] for r in rows):.9f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision", type=Path, default=Path("artifacts/results/week1_task4_frozen_vision.json"))
    parser.add_argument("--temporal", type=Path, default=Path("artifacts/results/week1_task4_temporal_mlp.json"))
    parser.add_argument("--manifest", type=Path, default=Path("docs/manifests/week1_task4_split.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/audits/week2_step80"))
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    vision_report = json.loads(args.vision.read_text(encoding="utf-8"))
    temporal_report = json.loads(args.temporal.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    vision = report_index(vision_report, 80, "checkpoint_vision_probability")
    temporal = report_index(temporal_report, 120, "temporal_mlp_probability")
    episodes = {row["episode_id"]: row for row in manifest["splits"]["test"]}
    if set(vision) != set(temporal) or set(vision) != set(episodes):
        raise ValueError("test episode sets differ across reports or manifest")

    ids = [row["episode_id"] for row in manifest["splits"]["test"]]
    labels = np.asarray([vision[eid]["failure"] for eid in ids], dtype=np.int64)
    vision_p = np.asarray([vision[eid]["probability"] for eid in ids])
    temporal_p = np.asarray([temporal[eid]["probability"] for eid in ids])
    if any(vision[eid]["failure"] != temporal[eid]["failure"] for eid in ids):
        raise ValueError("labels differ across reports")

    print("=== PREDICTIONS ===")
    print("episode_id,true_outcome,vision_step80,temporal_step120,absolute_difference")
    prediction_rows = []
    for eid, y, vp, tp in zip(ids, labels, vision_p, temporal_p, strict=True):
        row = {
            "episode_id": eid,
            "true_outcome": "failure" if y else "success",
            "vision_step80": float(vp),
            "temporal_step120": float(tp),
        }
        prediction_rows.append(row)
        row["absolute_difference"] = float(abs(vp - tp))
        print(f"{eid},{row['true_outcome']},{vp:.9f},{tp:.9f},{abs(vp - tp):.9f}")

    vision_wrong, vision_ties = misordered_pairs(ids, labels, vision_p)
    temporal_wrong, temporal_ties = misordered_pairs(ids, labels, temporal_p)
    print("\n=== MISORDERED PAIRS ===")
    print("VISION", json.dumps(vision_wrong, ensure_ascii=False))
    print("VISION_TIES", json.dumps(vision_ties, ensure_ascii=False))
    print("TEMPORAL", json.dumps(temporal_wrong, ensure_ascii=False))
    print("TEMPORAL_TIES", json.dumps(temporal_ties, ensure_ascii=False))
    vision_pair_ids = {(r["failure_episode"], r["success_episode"]) for r in vision_wrong}
    temporal_pair_ids = {(r["failure_episode"], r["success_episode"]) for r in temporal_wrong}
    print("SAME_MISORDERED_PAIRS", vision_pair_ids == temporal_pair_ids)
    print("PROBABILITY_VECTORS_IDENTICAL", bool(np.array_equal(vision_p, temporal_p)))
    print("PROBABILITY_VECTORS_ALLCLOSE", bool(np.allclose(vision_p, temporal_p)))
    pearson = float(np.corrcoef(vision_p, temporal_p)[0, 1])
    vision_rank = np.argsort(np.argsort(vision_p, kind="stable"), kind="stable")
    temporal_rank = np.argsort(np.argsort(temporal_p, kind="stable"), kind="stable")
    spearman = float(np.corrcoef(vision_rank, temporal_rank)[0, 1])
    print("PEARSON_R", f"{pearson:.9f}")
    print("SPEARMAN_RHO", f"{spearman:.9f}")

    vision_full, vision_loeo = loeo(ids, labels, vision_p)
    temporal_full, temporal_loeo = loeo(ids, labels, temporal_p)
    print_loeo("VISION_STEP80", vision_full, vision_loeo)
    print_loeo("TEMPORAL_STEP120", temporal_full, temporal_loeo)

    failures = [eid for eid in ids if vision[eid]["failure"]]
    successes = [eid for eid in ids if not vision[eid]["failure"]]
    groups = {
        "high-risk failure": sorted(failures, key=lambda eid: vision[eid]["probability"], reverse=True)[:2],
        "low-risk success": sorted(successes, key=lambda eid: vision[eid]["probability"])[:2],
        "false positive @ threshold": [eid for eid in successes if vision[eid]["probability"] >= args.threshold],
        "false negative @ threshold": [eid for eid in failures if vision[eid]["probability"] < args.threshold],
    }
    selected = []
    reasons: dict[str, list[str]] = {}
    for reason, group in groups.items():
        for eid in group:
            if eid not in selected:
                selected.append(eid)
            reasons.setdefault(eid, []).append(reason)
    print("\n=== RGB SELECTION ===")
    print("THRESHOLD", args.threshold)
    for name, group in groups.items():
        print(name.upper(), group)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        ("main t=0", "main_camera.mp4", 0),
        ("main t=40", "main_camera.mp4", 40),
        ("main t=80", "main_camera.mp4", 80),
        ("wrist t=0", "wrist_camera.mp4", 0),
        ("wrist t=40", "wrist_camera.mp4", 40),
        ("wrist t=80", "wrist_camera.mp4", 80),
    ]
    cell_width, image_height, caption_height = 360, 360, 58
    sheet = Image.new("RGB", (cell_width * len(columns), (image_height + caption_height) * len(selected)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    audit_rows = []
    for row_index, eid in enumerate(selected):
        outcome = "failure" if vision[eid]["failure"] else "success"
        risk = vision[eid]["probability"]
        reason = " | ".join(reasons[eid])
        audit_rows.append({
            "episode_id": eid,
            "true_outcome": outcome,
            "vision_step80_risk": risk,
            "selection_reason": reason,
            "visible_execution_state": "FILL_AFTER_REVIEW",
            "possible_visual_cue": "FILL_AFTER_REVIEW",
        })
        root = Path(episodes[eid]["path"])
        for column_index, (title, filename, frame_index) in enumerate(columns):
            frame = read_frame(root / filename, frame_index)
            frame.thumbnail((cell_width, image_height))
            x = column_index * cell_width
            y = row_index * (image_height + caption_height)
            sheet.paste(frame, (x, y))
            draw.multiline_text(
                (x + 3, y + image_height + 2),
                f"{eid[-12:]} {outcome} p={risk:.4f}\n{title}\n{reason}",
                fill="black", font=font, spacing=1,
            )
    sheet_path = args.output_dir / "step80_rgb_audit_contact_sheet.jpg"
    sheet.save(sheet_path, quality=95)

    with (args.output_dir / "rgb_audit_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit_rows[0].keys())
        writer.writeheader()
        writer.writerows(audit_rows)
    result = {
        "threshold_for_fp_fn": args.threshold,
        "predictions": prediction_rows,
        "vision_misordered_pairs": vision_wrong,
        "temporal_misordered_pairs": temporal_wrong,
        "same_misordered_pairs": vision_pair_ids == temporal_pair_ids,
        "probability_vectors_identical": bool(np.array_equal(vision_p, temporal_p)),
        "probability_vectors_allclose": bool(np.allclose(vision_p, temporal_p)),
        "pearson_r": pearson,
        "spearman_rho": spearman,
        "vision_full": vision_full,
        "temporal_full": temporal_full,
        "vision_loeo": vision_loeo,
        "temporal_loeo": temporal_loeo,
        "rgb_selection": groups,
    }
    result_path = args.output_dir / "numeric_results.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\n=== OUTPUTS ===")
    print(result_path)
    print(sheet_path)
    print(args.output_dir / "rgb_audit_table.csv")


if __name__ == "__main__":
    main()
