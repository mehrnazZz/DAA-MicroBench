from __future__ import annotations

from collections import deque
from typing import Any


class EpisodeRingBuffer:
    def __init__(self, max_frames: int):
        self.max_frames = max(1, int(max_frames))
        self._buf: deque[dict[str, Any]] = deque(maxlen=self.max_frames)

    def push(self, frame: dict[str, Any]) -> None:
        self._buf.append(frame)

    def snapshot(self) -> list[dict[str, Any]]:
        return list(self._buf)

    def __len__(self) -> int:
        return len(self._buf)
