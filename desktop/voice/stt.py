"""Speech detection (Silero VAD, CPU) and transcription (Whisper, 5060 Ti).

Both run in .venv-voice. Audio in is 16 kHz mono float32 in [-1, 1].
"""

from __future__ import annotations

import logging

import numpy as np

from desktop.voice.tts import pick_device

logger = logging.getLogger(__name__)

RATE = 16000
VAD_FRAME = 512   # Silero's fixed window at 16 kHz (32 ms)


class SileroVad:
    """Per-frame speech probability. Stateful: call reset() after a stream gap."""

    def __init__(self) -> None:
        import torch
        from silero_vad import load_silero_vad

        torch.set_num_threads(1)
        self._torch = torch
        self._model = load_silero_vad()

    def __call__(self, frame: np.ndarray) -> float:
        with self._torch.no_grad():
            return float(self._model(self._torch.from_numpy(frame), RATE).item())

    def reset(self) -> None:
        self._model.reset_states()


class WhisperStt:
    def __init__(self, model: str = "openai/whisper-large-v3-turbo", gpu_match: str = "5060",
                 language: str = "en") -> None:
        import torch
        from transformers import pipeline

        self.device = pick_device(gpu_match)
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self._pipe = pipeline("automatic-speech-recognition", model=model,
                              dtype=dtype, device=self.device)
        self._kw = {"language": language, "task": "transcribe"}
        logger.info("Whisper %s on %s", model, self.device)

    def __call__(self, audio: np.ndarray) -> str:
        out = self._pipe({"raw": audio.astype(np.float32), "sampling_rate": RATE},
                         generate_kwargs=self._kw)
        return str(out.get("text", "")).strip()

    def warm_up(self) -> None:
        self(np.zeros(RATE, dtype=np.float32))
