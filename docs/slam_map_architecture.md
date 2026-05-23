# SLAM & Map Architecture (2026)

Navigation uses a **two-phase map-and-localize** design:

| Phase | Entry | Output |
|-------|--------|--------|
| Mapping | `desktop/.venv/bin/python -m desktop.mapping` | `reference_map.npz` (2D occupancy + likelihood field) |
| Navigation | `desktop/.venv/bin/python -m desktop.nav --map PATH` | MCL pose + static costmap |

**Environment:** Desktop tools use the venv at `desktop/.venv` (not the repo-root `.venv` used on the Pi). From the repo root:

```bash
export PYTHONPATH="$(pwd)"
desktop/.venv/bin/pip install -r desktop/requirements.txt   # once
desktop/.venv/bin/python -m desktop.mapping --router tcp/PI:7447
desktop/.venv/bin/python -m desktop.nav --router tcp/PI:7447 --map path/to/reference_map.npz
```

## Packages

- `desktop/reference_map/` — frozen map schema, load/save, legacy `layers.npz` converter
- `desktop/mapping/` — log-odds occupancy builder during teleop mapping drives
- `desktop/localization/` — MCL particle filter against a read-only reference map
- `desktop/nav/` — autonomy shell (planner, follower, safety); Pi `local_2p5d` stays body-frame only

## Deprecated for nav

The online `WorldGrid` fusion loop (`FuserController`, dual particle filters, scan-match vs `block_votes`) is superseded for production navigation. It remains under `desktop/world_map/` for reference and migration tooling.

See the redesign plan in `.cursor/plans/` and the original critique in [bayesian_localization_redesign.md](bayesian_localization_redesign.md).
