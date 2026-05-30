"""Short git SHA of the running build — shared by the Pi runtime and the
desktop so the operator can spot a stale Pi (Pi sha ≠ desktop sha after a
deploy that wasn't restarted)."""
from __future__ import annotations

import os
import subprocess
from typing import Optional

_CACHE: Optional[str] = None
_RESOLVED = False


def git_sha() -> Optional[str]:
    """Short SHA of HEAD for the repo this file lives in, or None. Cached."""
    global _CACHE, _RESOLVED
    if _RESOLVED:
        return _CACHE
    _RESOLVED = True
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL, timeout=2.0,
        )
        _CACHE = out.decode().strip() or None
    except Exception:
        _CACHE = None
    return _CACHE
