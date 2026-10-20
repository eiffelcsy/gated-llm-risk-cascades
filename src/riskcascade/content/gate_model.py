"""Tier 1 for content: does this image contain text a human could read?

It does NOT read the text. Frozen DINOv3 backbone + small adapter + linear
head. Must be fast: report ms/image.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


class TextPresenceGate:
    def __init__(self, backbone: str = "facebook/dinov3-vitb16-pretrain-lvd1689m", device: str = "cuda"):
        self.backbone_name = backbone
        self.device = device
        self.model = None  # TODO: load backbone, attach adapter + head

    def fit(self, items: Sequence[dict], val_items: Sequence[dict] | None = None) -> "TextPresenceGate":
        """Train adapter + head on train split only. Add degradation augmentation
        (JPEG, blur, resize)."""
        raise NotImplementedError

    def score(self, items: Sequence[dict]) -> np.ndarray:
        """Return P(contains smuggled-looking text) per item. Batch on GPU."""
        raise NotImplementedError
