"""Cut a stream of VAD-scored audio frames into speech segments (pure, testable).

Feed fixed-size frames with their speech probability; a segment opens when the
probability crosses ``start_p``, keeps ``pre_roll_s`` of audio from before
that (so the first syllable isn't clipped), and closes after ``end_silence_s``
below ``end_p`` or at ``max_s``. Segments with less than ``min_speech_s`` of
speech are dropped (coughs, clicks, motor bumps).
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np


class Segmenter:
    def __init__(self, *, rate: int = 16000, frame: int = 512, start_p: float = 0.5,
                 end_p: float = 0.35, end_silence_s: float = 0.7, pre_roll_s: float = 0.3,
                 min_speech_s: float = 0.3, max_s: float = 15.0) -> None:
        self.rate = rate
        self.frame = frame
        fs = frame / rate
        self._start_p = start_p
        self._end_p = end_p
        self._end_frames = max(1, round(end_silence_s / fs))
        self._min_speech = max(1, round(min_speech_s / fs))
        self._max_frames = max(1, round(max_s / fs))
        self._pre: deque[np.ndarray] = deque(maxlen=max(1, round(pre_roll_s / fs)))
        self._seg: Optional[list[np.ndarray]] = None
        self._speech = 0
        self._silence = 0

    @property
    def active(self) -> bool:
        return self._seg is not None

    def feed(self, frame: np.ndarray, p: float) -> Optional[np.ndarray]:
        """Add one frame; returns a finished segment (float32 mono) or None."""
        if self._seg is None:
            if p >= self._start_p:
                self._seg = list(self._pre) + [frame]
                self._pre.clear()
                self._speech, self._silence = 1, 0
            else:
                self._pre.append(frame)
            return None
        self._seg.append(frame)
        if p >= self._end_p:
            self._speech += 1
            self._silence = 0
        else:
            self._silence += 1
        if self._silence >= self._end_frames or len(self._seg) >= self._max_frames:
            return self._close()
        return None

    def flush(self) -> Optional[np.ndarray]:
        """End any open segment now (mic muted / stream gap)."""
        self._pre.clear()
        return self._close() if self._seg is not None else None

    def _close(self) -> Optional[np.ndarray]:
        seg, speech = self._seg, self._speech
        self._seg, self._speech, self._silence = None, 0, 0
        if seg is None or speech < self._min_speech:
            return None
        return np.concatenate(seg).astype(np.float32)
