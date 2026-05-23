#!/usr/bin/env python3
"""Convert legacy fuser layers.npz to reference_map.npz."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from desktop.reference_map.legacy_convert import convert_layers_npz


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("layers_npz", help="Path to layers.npz from save_snapshot_bundle")
    p.add_argument(
        "-o", "--output", required=True,
        help="Output reference_map.npz path",
    )
    args = p.parse_args()
    convert_layers_npz(args.layers_npz, out_path=args.output)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
