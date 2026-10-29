"""Perception: slow, advisory VLM queries over pose-tagged camera frames.

Nothing here is in the real-time safety loop. Answers arrive seconds after
the frame was taken; callers may only use them to add caution (wait, stop,
ask the operator), never to relax a Pi-side veto.
"""

from .frames import FrameGrabber, PosedFrame
from .vlm_worker import VlmJob, VlmResult, VlmWorker

__all__ = ["FrameGrabber", "PosedFrame", "VlmJob", "VlmResult", "VlmWorker"]
