"""Validate one finalized VLA-SafeBench episode artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from vlasafe.rollout.validator import ArtifactValidationError, validate_episode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode_dir", type=Path)
    args = parser.parse_args()
    try:
        summary = validate_episode(args.episode_dir)
    except ArtifactValidationError as exc:
        print(f"INVALID: {args.episode_dir}\n{exc}")
        return 1
    print(
        f"VALID episode={summary.episode_id} steps={summary.num_steps} "
        f"video_frames={summary.video_frames} success={summary.success} "
        f"clipped_steps={summary.clipped_steps}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
