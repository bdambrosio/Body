"""DINOv2 feature extractor for Phase 6 VPR.

Single-frame in, L2-normalized feature vector out. Either offline
(bank-build time) or online (runtime observer); same code path so
features match bit-for-bit.

The model itself is injected at construction so tests can pass a
stub. ``load_default_extractor()`` does the torch.hub fetch and
returns a ready-to-use extractor on the chosen device.

Image input contract
--------------------
Accepts any of:
- ``numpy.ndarray`` HxWx3 ``uint8`` in **RGB** order
- ``PIL.Image`` (mode "RGB" — converted if not)

The Pi publishes JPEG over ``body/oakd/rgb``; decode via
``cv2.imdecode(..., IMREAD_COLOR)`` returns BGR — swap channels
before calling. Pillow's ``Image.open`` returns RGB; pass straight
through.

Output
------
A 1-D ``torch.float32`` tensor of size ``feature_dim``
(768 for ViT-B/14). L2-normalized so cosine similarity is a dot
product. Lives on the extractor's device; ``.cpu().numpy()`` to
serialize for bank storage.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ImageNet normalization, as used by DINOv2's reference pre-processing.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ExtractorConfig:
    # DINOv2 model name. ViT-B/14 is the Phase 6 default (paper-quality
    # VPR, ~85 MB weights, ~1 GB VRAM end-to-end with PF). ViT-S/14 is
    # ~4× faster and ~22 MB if extraction latency hurts gating.
    model_name: str = "dinov2_vitb14"

    # Square input side. Must be a multiple of patch_size (14 for
    # all ViT/14 variants). 518 = 37×37 patches, the DINOv2 reference
    # eval resolution. 224 = 16×16 patches, faster but noticeably
    # weaker on outdoor/large-scene VPR per the DINOv2 paper.
    input_size: int = 518

    # Patch size of the backbone. ViT-B/14 → 14. Kept explicit so
    # ``input_size`` validation doesn't have to introspect the model.
    patch_size: int = 14

    # 'cpu' or 'cuda'. Mirrors FuserConfig.pf_device naming.
    device: str = "cpu"

    # Cast model to fp16 on GPU. Roughly 2× throughput, ~half VRAM,
    # cosine similarity loss is negligible per the DINOv2 paper
    # (their public eval uses fp16). No effect on CPU.
    use_half_on_cuda: bool = True

    def __post_init__(self) -> None:
        if self.input_size % self.patch_size != 0:
            raise ValueError(
                f"input_size={self.input_size} must be a multiple of "
                f"patch_size={self.patch_size}"
            )


ImageLike = Union[np.ndarray, "Image.Image"]  # noqa: F821  (PIL forward ref)


class DinoV2Extractor:
    """Stateless feature extractor. Thread-safe for read-only use
    once constructed (torch.nn.Module forward is reentrant under the
    same lock-free assumptions as any inference path)."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: Optional[ExtractorConfig] = None,
    ) -> None:
        self._cfg = config or ExtractorConfig()
        self._device = torch.device(self._cfg.device)
        self._use_half = (
            self._cfg.use_half_on_cuda and self._device.type == "cuda"
        )

        model = model.to(self._device)
        if self._use_half:
            model = model.half()
        model.eval()
        self._model = model

        mean = torch.tensor(_IMAGENET_MEAN, device=self._device)
        std = torch.tensor(_IMAGENET_STD, device=self._device)
        # (3, 1, 1) for broadcasting over (3, H, W).
        self._mean = mean.view(3, 1, 1)
        self._std = std.view(3, 1, 1)

    @property
    def config(self) -> ExtractorConfig:
        return self._cfg

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def feature_dim(self) -> int:
        # DINOv2 backbones expose embed_dim; fall back to a forward
        # probe if a stub model doesn't.
        embed_dim = getattr(self._model, "embed_dim", None)
        if isinstance(embed_dim, int):
            return embed_dim
        with torch.inference_mode():
            probe = torch.zeros(
                1, 3, self._cfg.input_size, self._cfg.input_size,
                device=self._device,
                dtype=torch.float16 if self._use_half else torch.float32,
            )
            out = self._model(probe)
        return int(out.shape[-1])

    # ── Public API ────────────────────────────────────────────────────

    @torch.inference_mode()
    def extract(self, image: ImageLike) -> torch.Tensor:
        """Return a 1-D L2-normalized feature tensor on ``self.device``."""
        batch = self.extract_batch([image])
        return batch[0]

    @torch.inference_mode()
    def extract_batch(self, images: list) -> torch.Tensor:
        """Batched variant. Returns (B, D) tensor on ``self.device``."""
        if not images:
            raise ValueError("extract_batch: empty image list")
        tensor = torch.stack([self._preprocess(img) for img in images], dim=0)
        if self._use_half:
            tensor = tensor.half()
        feats = self._model(tensor)
        feats = feats.float()
        feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        return feats

    # ── Preprocess ────────────────────────────────────────────────────

    def _preprocess(self, image: ImageLike) -> torch.Tensor:
        """RGB image → (3, S, S) float32 normalized tensor on device."""
        rgb = _to_rgb_uint8(image)
        s = self._cfg.input_size
        rgb_resized = _resize_rgb(rgb, s, s)
        # uint8 HWC → float32 CHW in [0, 1]
        chw = torch.from_numpy(rgb_resized).to(self._device)
        chw = chw.permute(2, 0, 1).contiguous().float().div_(255.0)
        chw.sub_(self._mean).div_(self._std)
        return chw


# ── Helpers ──────────────────────────────────────────────────────────


def _to_rgb_uint8(image: ImageLike) -> np.ndarray:
    """Normalize input to HxWx3 uint8 RGB numpy."""
    try:
        from PIL import Image as PilImage
    except ImportError:
        PilImage = None  # type: ignore[assignment]

    if PilImage is not None and isinstance(image, PilImage.Image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        return np.asarray(image, dtype=np.uint8)

    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            raise TypeError(
                f"extract expected uint8 RGB array, got dtype={image.dtype}"
            )
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"extract expected HxWx3 RGB array, got shape={image.shape}"
            )
        return image

    raise TypeError(
        f"extract: unsupported image type {type(image).__name__} "
        "(want numpy uint8 HxWx3 RGB or PIL Image)"
    )


def _resize_rgb(rgb: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize an HxWx3 uint8 RGB image. Prefer PIL (BICUBIC, matches
    DINOv2 reference); fall back to OpenCV; raise if neither present.
    """
    if rgb.shape[:2] == (height, width):
        return rgb
    try:
        from PIL import Image as PilImage
        # PIL expects RGB.
        pil = PilImage.fromarray(rgb, mode="RGB")
        pil = pil.resize((width, height), PilImage.BICUBIC)
        # np.asarray on PIL is read-only; copy so torch.from_numpy
        # downstream doesn't warn about non-writable storage.
        return np.array(pil, dtype=np.uint8)
    except ImportError:
        pass
    try:
        import cv2
        # cv2 expects (width, height) and is channel-agnostic.
        return cv2.resize(
            rgb, (width, height), interpolation=cv2.INTER_CUBIC,
        ).astype(np.uint8, copy=False)
    except ImportError as e:
        raise RuntimeError(
            "VPR extractor needs Pillow or OpenCV for image resize. "
            "Install pillow (already a desktop dep) or opencv-python-headless."
        ) from e


# ── Default factory ──────────────────────────────────────────────────


def load_default_extractor(
    config: Optional[ExtractorConfig] = None,
    *,
    hub_repo: str = "facebookresearch/dinov2",
) -> DinoV2Extractor:
    """Fetch the configured DINOv2 backbone via torch.hub and wrap it.

    First call downloads weights into ``$TORCH_HOME/hub`` (default
    ``~/.cache/torch/hub``). Subsequent calls hit the cache. No
    network needed once cached.
    """
    cfg = config or ExtractorConfig()
    logger.info(
        "vpr.extractor: loading %s from torch.hub (%s) to device=%s",
        cfg.model_name, hub_repo, cfg.device,
    )
    model = torch.hub.load(hub_repo, cfg.model_name, pretrained=True)
    return DinoV2Extractor(model=model, config=cfg)
