"""Atomic JSONL sidecar recorder layered beside a LeRobot recording."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .schema import EpisodeMetadata, StepRecord


class SidecarRecorder:
    """Write one episode; incomplete runs never appear as finalized episodes."""

    def __init__(self, root: str | Path, metadata: EpisodeMetadata) -> None:
        self.root = Path(root)
        self.metadata = metadata
        self.partial_dir = self.root / f"{metadata.episode_id}.incomplete"
        self.final_dir = self.root / metadata.episode_id
        if self.partial_dir.exists() or self.final_dir.exists():
            raise FileExistsError(f"episode already exists: {metadata.episode_id}")
        self.partial_dir.mkdir(parents=True)
        self._write_json(self.partial_dir / "metadata.json", metadata.to_dict())
        self._steps = (self.partial_dir / "steps.jsonl").open("x", encoding="utf-8")
        self._next_step_id = 0
        self._finalized = False

    @staticmethod
    def _write_json(path: Path, value: dict) -> None:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")

    def add_step(self, record: StepRecord) -> None:
        if self._finalized:
            raise RuntimeError("cannot append to a finalized episode")
        if record.episode_id != self.metadata.episode_id:
            raise ValueError("step episode_id does not match metadata")
        if record.step_id != self._next_step_id:
            raise ValueError(f"expected step_id {self._next_step_id}, got {record.step_id}")
        self._steps.write(json.dumps(record.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n")
        self._steps.flush()
        self._next_step_id += 1

    def finalize(
        self,
        *,
        termination_reason: str,
        success: bool,
        metrics: dict | None = None,
    ) -> Path:
        if self._finalized:
            raise RuntimeError("episode already finalized")
        self._steps.flush()
        os.fsync(self._steps.fileno())
        self._steps.close()
        result = {
            "num_steps": self._next_step_id,
            "success": success,
            "termination_reason": termination_reason,
        }
        if metrics is not None:
            result["metrics"] = metrics
        self._write_json(self.partial_dir / "result.json", result)
        (self.partial_dir / "COMPLETE").touch(exist_ok=False)
        os.replace(self.partial_dir, self.final_dir)
        self._finalized = True
        return self.final_dir

    def abort(self) -> Path:
        """Close files but deliberately retain the `.incomplete` evidence."""
        if not self._steps.closed:
            self._steps.close()
        return self.partial_dir
