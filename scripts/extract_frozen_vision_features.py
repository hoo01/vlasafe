"""Extract frozen dual-camera ResNet-50 features at fixed checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
import torchvision
from torchvision.models import ResNet50_Weights, resnet50


def load_frames(video_path: Path, frame_indexes: list[int]) -> list[np.ndarray]:
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    reader = imageio.get_reader(str(video_path))
    try:
        frames = [np.asarray(reader.get_data(index)) for index in frame_indexes]
    finally:
        reader.close()
    for frame in frames:
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError(f"{video_path}: expected an RGB frame, got {frame.shape}")
    return [np.ascontiguousarray(frame[:, :, :3], dtype=np.uint8) for frame in frames]


def build_encoder(device: torch.device) -> tuple[torch.nn.Module, Any]:
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    encoder = torch.nn.Sequential(*list(model.children())[:-1]).to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder, weights.transforms()


def encode_batch(
    frames: list[np.ndarray],
    encoder: torch.nn.Module,
    transform: Any,
    device: torch.device,
) -> np.ndarray:
    tensors = [
        transform(torch.from_numpy(frame).permute(2, 0, 1))
        for frame in frames
    ]
    inputs = torch.stack(tensors).to(device)
    with torch.inference_mode():
        features = encoder(inputs).flatten(1)
    return features.cpu().numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-steps", type=int, nargs="+", default=[0, 40, 80, 120])
    parser.add_argument("--cameras", nargs="+", default=["main_camera", "wrist_camera"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    checkpoints = sorted(set(args.checkpoint_steps))
    if not checkpoints or min(checkpoints) < 0:
        raise ValueError("checkpoint steps must be non-negative")
    if len(set(args.cameras)) != len(args.cameras):
        raise ValueError("camera names must be unique")

    manifest: dict[str, Any] = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = []
    episode_records = []
    for split, episodes in manifest["splits"].items():
        for episode in episodes:
            episode_records.append(episode)
            for checkpoint in checkpoints:
                if checkpoint >= int(episode["num_steps"]):
                    raise ValueError(
                        f"{episode['episode_id']}: checkpoint {checkpoint} outside "
                        f"{episode['num_steps']} recorded steps"
                    )
                records.append(
                    {
                        "episode_id": episode["episode_id"],
                        "episode_path": Path(episode["path"]),
                        "split": split,
                        "failure": int(not episode["success"]),
                        "checkpoint_step": checkpoint,
                    }
                )

    device = torch.device(args.device)
    encoder, transform = build_encoder(device)
    feature_batches = []
    pending_frames: list[np.ndarray] = []
    pending_positions: list[tuple[int, int]] = []
    features = np.empty((len(records), len(args.cameras), 2048), dtype=np.float32)

    def flush() -> None:
        if not pending_frames:
            return
        encoded = encode_batch(pending_frames, encoder, transform, device)
        feature_batches.append(len(encoded))
        for value, (record_index, camera_index) in zip(encoded, pending_positions, strict=True):
            features[record_index, camera_index] = value
        pending_frames.clear()
        pending_positions.clear()

    position = {
        (record["episode_id"], record["checkpoint_step"]): index
        for index, record in enumerate(records)
    }
    for episode in episode_records:
        for camera_index, camera in enumerate(args.cameras):
            video_path = Path(episode["path"]) / f"{camera}.mp4"
            for checkpoint, frame in zip(
                checkpoints, load_frames(video_path, checkpoints), strict=True
            ):
                pending_frames.append(frame)
                pending_positions.append(
                    (position[(episode["episode_id"], checkpoint)], camera_index)
                )
                if len(pending_frames) >= args.batch_size:
                    flush()
    flush()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=features,
        episode_id=np.asarray([record["episode_id"] for record in records]),
        split=np.asarray([record["split"] for record in records]),
        failure=np.asarray([record["failure"] for record in records], dtype=np.int64),
        checkpoint_step=np.asarray(
            [record["checkpoint_step"] for record in records], dtype=np.int64
        ),
        camera=np.asarray(args.cameras),
    )
    metadata_path = args.output.with_suffix(".json")
    metadata = {
        "manifest": str(args.manifest),
        "output": str(args.output),
        "encoder": "torchvision/resnet50 IMAGENET1K_V2",
        "weights_url": ResNet50_Weights.IMAGENET1K_V2.url,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "encoder_frozen": True,
        "camera_order": args.cameras,
        "checkpoint_steps": checkpoints,
        "records": len(records),
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "device": str(device),
        "batch_size": args.batch_size,
        "encoded_batches": feature_batches,
        "image_orientation": "as recorded; no outcome-dependent correction",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"FROZEN VISION FEATURES {args.output}")
    print(f"METADATA {metadata_path}")
    print("encoder:", metadata["encoder"])
    print("records:", len(records))
    print("feature shape:", features.shape)
    print("finite:", bool(np.isfinite(features).all()))


if __name__ == "__main__":
    main()
