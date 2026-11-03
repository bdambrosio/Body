"""Which torch device the particle filter runs on.

The desktop has two GPUs. PyTorch orders them fastest-first, so plain
``cuda`` / ``cuda:0`` is the RTX PRO 6000 that vLLM fills; on 11-03 a filter
kernel starved behind vLLM for > 7 s, the driver killed the context (Xid 8)
and localization died mid-patrol. The filter belongs on the RTX 5060 Ti.

``resolve_pf_device(choice)``:
  * ``cpu`` / ``cuda:N``   — as given;
  * ``auto`` / ``cuda``    — the CUDA device whose name contains
    ``$BODY_PF_GPU`` (default ``5060``); if none matches, ``auto`` falls back
    to CPU and ``cuda`` raises (never silently the vLLM GPU).
"""
from __future__ import annotations

import logging
import os
from typing import List, Tuple

logger = logging.getLogger(__name__)

DEFAULT_GPU_MATCH = "5060"


def _cuda_devices() -> List[Tuple[int, str]]:
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        return [(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
    except Exception:
        return []


def resolve_pf_device(choice: str, *, devices=None, match=None) -> str:
    if choice == "cpu" or choice.startswith("cuda:"):
        return choice
    match = (match if match is not None else os.environ.get("BODY_PF_GPU", DEFAULT_GPU_MATCH)).lower()
    devs = _cuda_devices() if devices is None else devices
    for i, name in devs:
        if match and match in name.lower():
            logger.info("particle filter on cuda:%d (%s)", i, name)
            return f"cuda:{i}"
    if choice == "cuda":
        raise RuntimeError(
            f"--pf-device cuda: no GPU name contains {match!r} among {devs}; "
            "pass cuda:N or set BODY_PF_GPU")
    logger.warning("particle filter on CPU: no GPU name contains %r among %s", match, devs)
    return "cpu"
