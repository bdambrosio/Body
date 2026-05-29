"""Tier-3 drive debug console (desktop ↔ Pi).

A thin operator UI for bringing up and debugging the Pi-side reactive
driver (`body.local_drive`) in isolation: view the live body-frame
local_map, click a goal, watch the Pi drive to it. No PF, no global map,
no planner — the operator plays the upper tiers by hand. See
docs/drive_tier3_spec.md.
"""
