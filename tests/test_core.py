"""Fast unit tests: no model weights, no private photos."""
import numpy as np
from PIL import Image
from whitebg import models, download
from whitebg.common import place


def test_manifest_has_pipeline_models():
    m = download.manifest()
    for n in ("birefnet-general", "isnet-general-use", "u2net"):
        assert n in m and len(m[n]["sha256"]) == 64 and m[n]["size"] > 0 and m[n]["url"].startswith("https://")


def test_normalize_shapes():
    img = Image.fromarray((np.random.default_rng(0).random((50, 70, 3)) * 255).astype(np.uint8))
    for n, (mean, std, size, _) in models.SPECS.items():
        x = models.normalize(img, mean, std, size)
        assert x.shape == (1, 3, size[1], size[0]) and x.dtype == np.float32


def test_postprocess_range():
    raw = np.random.default_rng(1).normal(size=(1, 1, 32, 32)).astype(np.float32)
    for post in ("sigmoid", "plain", "clip"):
        m = np.asarray(models.postprocess(raw, post, (40, 20)))
        assert m.shape == (20, 40) and m.max() <= 255


def test_place_square_and_white():
    out = np.full((300, 400, 3), 255.0, np.float32)
    out[100:200, 150:250] = 30
    img, pl = place(out)
    assert img.shape[0] == img.shape[1] and img[0, 0].tolist() == [255, 255, 255]
    assert abs(pl["fill_pct"] - 88) < 1


def test_find_model_missing(tmp_path):
    assert download.find_model("u2net", tmp_path) is None
