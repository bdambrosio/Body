"""Standalone world-map editor.

Loads a saved world-map snapshot (``layers.npz``), lets the operator
paint/erase occupancy on the global map, and saves the edited map back
to a snapshot the live PF stack can load. Optionally (when a bot is
reachable) overlays the live lidar scan transformed by a read-only
particle-filter pose so the operator can correct the map against ground
truth.

The editor *never fuses* — the operator's brush is the only writer of
the grid. See ``docs`` / the Stage-C plan for rationale.
"""
