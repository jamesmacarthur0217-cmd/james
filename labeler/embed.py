"""Image and text embeddings with SigLIP 2, for text search and look-alike search."""
from __future__ import annotations

import logging
import math
import threading

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

log = logging.getLogger(__name__)


def _vec(out) -> torch.Tensor:
    # transformers 5 returns an output object; older versions return the tensor itself.
    return out if torch.is_tensor(out) else out.pooler_output


class Embedder:
    def __init__(self, cfg: dict):
        self.model_name = cfg["model"]
        self.device = cfg["device"] if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device != "cpu" else torch.float32
        log.info("loading embedding model %s", self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name, dtype=dtype).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.dtype = dtype
        self.scale = float(self.model.logit_scale.detach().exp())
        self.bias = float(self.model.logit_bias.detach())
        # The pipeline thread embeds crops while web requests embed search text.
        self._lock = threading.Lock()

    @torch.no_grad()
    def images(self, crops: list[np.ndarray]) -> np.ndarray:
        """Normalized embeddings, one row per BGR crop."""
        pil = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops]
        with self._lock:
            inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
            out = _vec(self.model.get_image_features(pixel_values=inputs["pixel_values"].to(self.dtype)))
        out = out.float()
        return (out / out.norm(dim=-1, keepdim=True)).cpu().numpy()

    def track_vector(self, crops: list[np.ndarray]) -> np.ndarray:
        """One vector per person: the mean of their best crops, renormalized."""
        mean = self.images(crops).mean(axis=0)
        return mean / max(np.linalg.norm(mean), 1e-8)

    @torch.no_grad()
    def text(self, query: str) -> np.ndarray:
        query = query.strip().lower()
        if not query.startswith(("a photo", "a picture")):
            query = f"a photo of {query}"
        with self._lock:
            inputs = self.processor(text=[query], padding="max_length", max_length=64,
                                    return_tensors="pt").to(self.device)
            out = _vec(self.model.get_text_features(**inputs)).float()
        out = out / out.norm(dim=-1, keepdim=True)
        return out[0].cpu().numpy()

    def match_probability(self, cosine: float) -> float:
        """SigLIP's calibrated text–image match probability for a cosine similarity."""
        return 1.0 / (1.0 + math.exp(-(self.scale * cosine + self.bias)))
