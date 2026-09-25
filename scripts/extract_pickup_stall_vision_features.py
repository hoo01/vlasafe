"""Extract frozen dual-camera features at per-episode pickup-stall checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torchvision.models import ResNet50_Weights

try:
    from extract_frozen_vision_features import build_encoder, encode_batch, load_frames
except ModuleNotFoundError:
    from scripts.extract_frozen_vision_features import build_encoder, encode_batch, load_frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cameras", nargs="+", default=["main_camera", "wrist_camera"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = [
        {**sample, "split": split}
        for split, samples in manifest["splits"].items()
        for sample in samples
    ]
    device = torch.device(args.device)
    encoder, transform = build_encoder(device)
    features = np.empty((len(records), len(args.cameras), 2048), dtype=np.float32)
    pending_frames: list[np.ndarray] = []
    pending_positions: list[tuple[int, int]] = []

    def flush() -> None:
        if not pending_frames:
            return
        encoded = encode_batch(pending_frames, encoder, transform, device)
        for value, position in zip(encoded, pending_positions, strict=True):
            features[position] = value
        pending_frames.clear()
        pending_positions.clear()

    for record_index, record in enumerate(records):
        checkpoint = int(record["checkpoint_step"])
        for camera_index, camera in enumerate(args.cameras):
            frame = load_frames(Path(record["path"]) / f"{camera}.mp4", [checkpoint])[0]
            pending_frames.append(frame)
            pending_positions.append((record_index, camera_index))
            if len(pending_frames) >= args.batch_size:
                flush()
    flush()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=features,
        episode_id=np.asarray([row["episode_id"] for row in records]),
        split=np.asarray([row["split"] for row in records]),
        pickup_stall=np.asarray([row["pickup_stall"] for row in records], dtype=np.int64),
        checkpoint_step=np.asarray([row["checkpoint_step"] for row in records], dtype=np.int64),
        initial_state_id=np.asarray([row["initial_state_id"] for row in records], dtype=np.int64),
        camera=np.asarray(args.cameras),
    )
    metadata = {
        "schema_version": "0.1.0",
        "manifest": args.manifest.as_posix(),
        "encoder": "torchvision/resnet50 IMAGENET1K_V2",
        "weights_url": ResNet50_Weights.IMAGENET1K_V2.url,
        "encoder_frozen": True,
        "target": "pickup_stall_confirmed_within_next_20_steps",
        "camera_order": args.cameras,
        "records": len(records),
        "feature_shape": list(features.shape),
        "device": str(device),
        "privileged_predictor_inputs": [],
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("PICKUP STALL VISION FEATURES", args.output)
    print("records", len(records), "shape", features.shape, "finite", bool(np.isfinite(features).all()))


if __name__ == "__main__":
    main()
