"""Object tasks: ``goto <obj>``, ``find <obj>``, ``follow <obj>``.

Plan: ~/.claude/plans/velvety-exploring-penguin.md. Phase 1 (this package so
far) is look-only: parse a command, ask the VLM where the object is in a
posed frame, turn the box into a bearing / range / world point, and remember
it. Nothing here moves the robot yet.
"""
from .commands import Command, parse
from .locate import (CameraModel, Sighting, bbox_to_bearing, depth_range_m,
                     sighting_from_answer)
from .memory import ObjectMemory

__all__ = [
    "Command", "parse",
    "CameraModel", "Sighting", "bbox_to_bearing", "depth_range_m",
    "sighting_from_answer",
    "ObjectMemory",
]
