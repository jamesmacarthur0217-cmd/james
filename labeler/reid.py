"""Person re-identification (OSNet-AIN): a vector for "is this the same person?", trained for exactly that,
unlike SigLIP 2 (labeler/embed.py), which describes what a picture shows and stays in use for text search and the
fast classifiers. Only measured so far (tools/relink_check.py --reid): on MOT16 it was not better than SigLIP 2
inside the pipeline (look-alikes 24/74 vs 34/74; README "One person, one record"), so the app does not use it;
measure it again on footage from far or overhead cameras, where it did better (a view from a bus: 18/33 vs 10/33).

The network below is OSNet-AIN x1.0 from Torchreid (https://github.com/KaiyangZhou/deep-person-reid,
torchreid/models/osnet_ain.py), MIT License, Copyright (c) 2018 Kaiyang Zhou; only the layers needed to run it are
kept (no training head, no downloading). Reference: Zhou et al., Learning Generalisable Omni-Scale
Representations for Person Re-Identification, TPAMI 2021. The weights (`reid.weights`, 8.9 MB, MIT) are the
Torchreid multi-source checkpoint (trained on MSMT17, DukeMTMC-reID and CUHK03: research datasets, so
non-commercial use only; README "Licenses").
"""
from __future__ import annotations

import logging
import threading

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import resolve

log = logging.getLogger(__name__)

MODEL_NAME = "osnet_ain_x1_0"
SIZE = (128, 256)   # width, height the network was trained on
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


# --- the network (Torchreid, MIT) ---------------------------------------------------------------------

class ConvLayer(nn.Module):
    def __init__(self, cin, cout, k, stride=1, padding=0, IN=False):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride=stride, padding=padding, bias=False)
        self.bn = nn.InstanceNorm2d(cout, affine=True) if IN else nn.BatchNorm2d(cout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1Linear(nn.Module):
    def __init__(self, cin, cout, bn=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout) if bn else None

    def forward(self, x):
        x = self.conv(x)
        return self.bn(x) if self.bn is not None else x


class LightConv3x3(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 1, bias=False)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False, groups=cout)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class LightConvStream(nn.Module):
    def __init__(self, cin, cout, depth):
        super().__init__()
        self.layers = nn.Sequential(LightConv3x3(cin, cout), *[LightConv3x3(cout, cout) for _ in range(depth - 1)])

    def forward(self, x):
        return self.layers(x)


class ChannelGate(nn.Module):
    def __init__(self, cin, reduction=16):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(cin, cin // reduction, 1, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(cin // reduction, cin, 1, bias=True)
        self.gate_activation = nn.Sigmoid()

    def forward(self, x):
        g = self.gate_activation(self.fc2(self.relu(self.fc1(self.global_avgpool(x)))))
        return x * g


class OSBlock(nn.Module):
    """Omni-scale block; `IN` adds instance normalization inside the residual (OSBlockINin upstream)."""

    def __init__(self, cin, cout, IN=False, reduction=4, T=4):
        super().__init__()
        mid = cout // reduction
        self.conv1 = Conv1x1(cin, mid)
        self.conv2 = nn.ModuleList([LightConvStream(mid, mid, t) for t in range(1, T + 1)])
        self.gate = ChannelGate(mid)
        self.conv3 = Conv1x1Linear(mid, cout, bn=not IN)
        self.downsample = Conv1x1Linear(cin, cout) if cin != cout else None
        self.IN = nn.InstanceNorm2d(cout, affine=True) if IN else None

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = 0
        for conv2_t in self.conv2:
            x2 = x2 + self.gate(conv2_t(x1))
        x3 = self.conv3(x2)
        if self.IN is not None:
            x3 = self.IN(x3)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return F.relu(x3 + identity)


class OSNetAIN(nn.Module):
    """OSNet-AIN x1.0 without the identity classifier: an image in, a 512-number vector out."""

    def __init__(self, channels=(64, 256, 384, 512), feature_dim=512):
        super().__init__()
        c = channels
        self.conv1 = ConvLayer(3, c[0], 7, stride=2, padding=3, IN=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = nn.Sequential(OSBlock(c[0], c[1], IN=True), OSBlock(c[1], c[1], IN=True))
        self.pool2 = nn.Sequential(Conv1x1(c[1], c[1]), nn.AvgPool2d(2, stride=2))
        self.conv3 = nn.Sequential(OSBlock(c[1], c[2]), OSBlock(c[2], c[2], IN=True))
        self.pool3 = nn.Sequential(Conv1x1(c[2], c[2]), nn.AvgPool2d(2, stride=2))
        self.conv4 = nn.Sequential(OSBlock(c[2], c[3], IN=True), OSBlock(c[3], c[3]))
        self.conv5 = Conv1x1(c[3], c[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(c[3], feature_dim), nn.BatchNorm1d(feature_dim), nn.ReLU())

    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.conv5(self.conv4(x))
        return self.fc(self.global_avgpool(x).flatten(1))


# --- use ------------------------------------------------------------------------------------------------

class ReIDEmbedder:
    """Same interface as embed.Embedder for pictures: images(crops) and track_vector(crops), normalized."""

    def __init__(self, cfg: dict):
        self.model_name = MODEL_NAME
        self.device = cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
        path = resolve(cfg["weights"])
        log.info("loading re-identification model %s", path)
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = {k.removeprefix("module."): v for k, v in state.get("state_dict", state).items()
                 if not k.removeprefix("module.").startswith("classifier.")}
        self.model = OSNetAIN()
        self.model.load_state_dict(state)   # strict: a wrong file fails here, not with silent bad vectors
        self.model.to(self.device).eval()
        self._lock = threading.Lock()

    @staticmethod
    def prepare(crops: list[np.ndarray]) -> torch.Tensor:
        batch = []
        for c in crops:
            rgb = cv2.cvtColor(cv2.resize(c, SIZE, interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
            batch.append(((rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1))
        return torch.from_numpy(np.stack(batch))

    @torch.no_grad()
    def images(self, crops: list[np.ndarray]) -> np.ndarray:
        """Normalized vectors, one row per BGR crop."""
        with self._lock:
            out = self.model(self.prepare(crops).to(self.device)).float()
        return F.normalize(out, dim=1).cpu().numpy()

    def track_vector(self, crops: list[np.ndarray]) -> np.ndarray:
        """One vector per person: the mean of their best crops, renormalized."""
        mean = self.images(crops).mean(axis=0)
        return mean / max(np.linalg.norm(mean), 1e-8)
