"""Re-identification model wrapper (labeler/reid.py): loading, vectors. Random weights, so no download is needed."""
import numpy as np
import pytest
import torch

from labeler.reid import MODEL_NAME, OSNetAIN, ReIDEmbedder


@pytest.fixture
def weights(tmp_path):
    torch.manual_seed(0)
    state = OSNetAIN().state_dict()
    state["classifier.weight"] = torch.zeros(10, 512)   # the training head in the original files is skipped
    path = tmp_path / "osnet.pt"
    torch.save({f"module.{k}": v for k, v in state.items()}, path)
    return path


def _crop(seed, h=300, w=120):
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_vectors_are_normalized_and_deterministic(weights):
    r = ReIDEmbedder({"weights": str(weights), "device": "cpu"})
    assert r.model_name == MODEL_NAME
    v = r.images([_crop(1), _crop(2, 80, 40)])            # any crop size: resized to 128x256
    assert v.shape == (2, 512)
    assert np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-5)
    assert np.allclose(r.images([_crop(1)])[0], v[0], atol=1e-5)   # eval mode: same picture, same vector
    t = r.track_vector([_crop(1), _crop(3)])
    assert t.shape == (512,) and abs(np.linalg.norm(t) - 1) < 1e-5


def test_a_wrong_weights_file_fails_loudly(tmp_path):
    path = tmp_path / "other.pt"
    torch.save({"conv1.conv.weight": torch.zeros(1)}, path)
    with pytest.raises(RuntimeError):
        ReIDEmbedder({"weights": str(path), "device": "cpu"})


def test_prepare_uses_rgb_and_imagenet_normalization():
    crop = np.zeros((256, 128, 3), np.uint8)
    crop[..., 2] = 255                                      # pure red in BGR
    x = ReIDEmbedder.prepare([crop])
    assert x.shape == (1, 3, 256, 128)
    assert torch.allclose(x[0, 0], torch.full((256, 128), (1 - 0.485) / 0.229), atol=1e-4)   # red channel first
    assert torch.allclose(x[0, 2], torch.full((256, 128), (0 - 0.406) / 0.225), atol=1e-4)
