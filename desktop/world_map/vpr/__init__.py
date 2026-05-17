"""Visual place recognition (Phase 6 of the localization redesign).

Module layout:
- ``extractor``: DINOv2 feature extractor wrapper (offline + runtime).
- (later) ``bank``: feature-bank file format + cosine query.
- (later) ``observer``: runtime observation stream into ParticleFilterPose.

See ``docs/bayesian_localization_redesign.md`` §"Phase 6" for the plan.
"""

from .bank import (
    QueryResult,
    VPRBank,
    mixture_observation_from_query,
)
from .extractor import (
    DinoV2Extractor,
    ExtractorConfig,
    load_default_extractor,
)
from .shadow_driver import (
    ShadowVPRConfig,
    ShadowVPRDriver,
)

__all__ = [
    "DinoV2Extractor",
    "ExtractorConfig",
    "QueryResult",
    "ShadowVPRConfig",
    "ShadowVPRDriver",
    "VPRBank",
    "load_default_extractor",
    "mixture_observation_from_query",
]
