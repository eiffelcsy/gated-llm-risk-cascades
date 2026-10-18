"""Tier 2 for content: restore, read, then judge policy.

1. Restore: contrast normalisation, upscaling, de-stylisation.
2. Read: OCR (easyocr) and the MLLM's own reading.
3. Judge: MLLM (e.g. Qwen2.5-VL-7B) decides scam/gambling promotion or not.

Evidence features for the override gate (keep them cheap and numeric):
  - OCR confidence
  - OCR/MLLM text agreement (e.g. normalised edit distance)
  - self-consistency: fraction of k sampled verdicts that agree
  - whether restoration changed the reading
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


class OcrMllmJudge:
    def __init__(self, mllm: str = "Qwen/Qwen2.5-VL-7B-Instruct", k_samples: int = 3):
        self.mllm_name = mllm
        self.k = k_samples

    def judge(self, items: Sequence[dict], tier1_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return verdicts (n,) and evidence features (n, 4)."""
        raise NotImplementedError
