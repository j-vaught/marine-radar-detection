"""Load the full PPI corpus into VRAM as a batch tensor, plus split helpers."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

SITES = {"MANV": 0, "DARC": 1, "DAAN": 2}
SITE_NAMES = {v: k for k, v in SITES.items()}
PPI_CENTER = (531, 515)
PPI_RADIUS = 466


@dataclass
class Corpus:
    images: torch.Tensor                 # [N, 1, H, W] float32 in [0, 1]
    circle_mask: torch.Tensor            # [1, 1, H, W] float32 {0, 1}
    image_ids: np.ndarray                # [N] int64 — COCO image id
    file_names: list[str]                # [N] PNG filenames
    site_tags: np.ndarray                # [N] int8 — 0 MANV, 1 DARC, 2 DAAN
    coco_ann_path: str                   # path to source COCO json
    coco: dict                           # parsed COCO json

    @property
    def N(self) -> int:
        return self.images.shape[0]

    @property
    def device(self) -> torch.device:
        return self.images.device

    def indices_for_site(self, site: str) -> np.ndarray:
        return np.where(self.site_tags == SITES[site])[0]

    def indices_for_sites(self, sites: list[str]) -> np.ndarray:
        tags = [SITES[s] for s in sites]
        return np.where(np.isin(self.site_tags, tags))[0]


def _site_from_filename(name: str) -> int:
    prefix = name.split("_")[0]
    return SITES.get(prefix, -1)


def load_corpus(
    ann_path: str = "data/processed/annotations/all.json",
    image_dir: str = "data/processed/images",
    device: str | torch.device = "cuda:1",
    dtype: torch.dtype = torch.float32,
) -> Corpus:
    """Load every image referenced by the COCO file into VRAM."""
    device = torch.device(device)
    with open(ann_path) as f:
        coco = json.load(f)

    infos = coco["images"]
    N = len(infos)

    first = cv2.imread(str(Path(image_dir) / infos[0]["file_name"]), cv2.IMREAD_GRAYSCALE)
    H, W = first.shape

    stack = np.zeros((N, H, W), dtype=np.uint8)
    image_ids = np.zeros(N, dtype=np.int64)
    file_names: list[str] = []
    site_tags = np.zeros(N, dtype=np.int8)

    for i, info in enumerate(infos):
        img = cv2.imread(str(Path(image_dir) / info["file_name"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(info["file_name"])
        if img.shape != (H, W):
            raise ValueError(f"Non-uniform image shape at {info['file_name']}: {img.shape}")
        stack[i] = img
        image_ids[i] = info["id"]
        file_names.append(info["file_name"])
        site_tags[i] = _site_from_filename(info["file_name"])

    circle = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(circle, PPI_CENTER, PPI_RADIUS, 1, -1)
    stack = (stack.astype(np.float32) / 255.0) * circle[None, :, :]

    images = torch.from_numpy(stack).to(device=device, dtype=dtype).unsqueeze(1)
    circle_mask = torch.from_numpy(circle.astype(np.float32)).to(device=device, dtype=dtype)[None, None]

    return Corpus(
        images=images,
        circle_mask=circle_mask,
        image_ids=image_ids,
        file_names=file_names,
        site_tags=site_tags,
        coco_ann_path=ann_path,
        coco=coco,
    )


# ── split helpers ────────────────────────────────────────────────────────

def pooled_split(N: int, seed: int, ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    n_tune = int(round(ratios[0] * N))
    n_val = int(round(ratios[1] * N))
    return idx[:n_tune], idx[n_tune:n_tune + n_val], idx[n_tune + n_val:]


def site_permutation_split(corpus: Corpus, tune: str, val: str, test: str
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (corpus.indices_for_site(tune),
            corpus.indices_for_site(val),
            corpus.indices_for_site(test))


def site_internal_split(corpus: Corpus, site: str, seed: int = 0,
                        ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = corpus.indices_for_site(site)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(idx)
    n = len(idx)
    n_tune = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    return idx[:n_tune], idx[n_tune:n_tune + n_val], idx[n_tune + n_val:]
