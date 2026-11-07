"""Kokoro-82M text-to-speech (runs in .venv-voice — see desktop/voice/README.md).

``pitch`` raises the voice without changing its pace: synthesize at
``speed / pitch`` and declare ``24000 * pitch`` as the sample rate, so the
Pi's resampler (pw-cat) does the shift for free.
"""

from __future__ import annotations

import logging
from typing import Iterator

import numpy as np

logger = logging.getLogger(__name__)

KOKORO_RATE = 24000


def pick_device(gpu_match: str) -> str:
    """CUDA device whose name contains ``gpu_match`` (never the vLLM GPU), else CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                if gpu_match and gpu_match.lower() in torch.cuda.get_device_name(i).lower():
                    return f"cuda:{i}"
    except Exception:
        pass
    logger.warning("TTS on CPU: no GPU name contains %r", gpu_match)
    return "cpu"


def to_pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


class KokoroTts:
    def __init__(self, *, voice: str = "af_heart", pitch: float = 1.0, speed: float = 1.0,
                 gpu_match: str = "5060") -> None:
        from kokoro import KPipeline

        self.voice = voice
        self.pitch = max(0.5, min(2.0, float(pitch)))
        self.speed = float(speed)
        self.device = pick_device(gpu_match)
        self._pipe = KPipeline(lang_code=voice[0], device=self.device, repo_id="hexgrad/Kokoro-82M")
        logger.info("Kokoro voice=%s pitch=%.2f on %s", voice, self.pitch, self.device)

    @property
    def sample_rate(self) -> int:
        return int(round(KOKORO_RATE * self.pitch))

    def warm_up(self) -> None:
        for _ in self.synth("Ready."):
            pass

    def synth(self, text: str) -> Iterator[bytes]:
        """s16le PCM per sentence-ish segment, as soon as each is ready."""
        for result in self._pipe(text, voice=self.voice, speed=self.speed / self.pitch):
            audio = result.audio
            if audio is None:
                continue
            yield to_pcm16(audio.detach().cpu().numpy())
