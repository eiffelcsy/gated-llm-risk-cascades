"""Content-track data.

Each item: {"image_path": str, "label": 0/1, "family": str}
family is the smuggling technique, or "benign" for clean/stylised negatives.
The family field is what splits.held_out_family uses.
"""
from __future__ import annotations

from pathlib import Path


def load_smugglebench(root: str | Path) -> list[dict]:
    """SmuggleBench (arXiv 2604.06950): 1,700 attack instances.

    TODO: map each instance to a family. Start with the paper's two pathways
    (perceptual blindness vs. reasoning blockade) and their sub-techniques.
    """
    raise NotImplementedError("download SmuggleBench into data/raw/smugglebench and parse it here")


def load_benign_stylised(root: str | Path) -> list[dict]:
    """Hard negatives: memes, decorative captions, product packaging.

    These set the false-positive cost. Without them the gate learns
    'any text = risky' and the FPR number is meaningless.
    """
    raise NotImplementedError


def render_smuggled(text: str, family: str, background_path: str | Path, out_path: str | Path) -> None:
    """Synthetic attacks for families SmuggleBench under-covers
    (e.g. low-contrast, fragmented, tiny watermark). Use PIL.

    Keep rendering families separate from test families so held-out results stay honest.
    """
    raise NotImplementedError
