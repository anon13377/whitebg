"""Model sessions on onnxruntime directly (no rembg dependency).

Pre and post processing replicate rembg 2.0.86 exactly (sessions/base.py normalize, birefnet_general.py,
dis_general_use.py, u2net.py) so masks match what the original pipeline was tuned on.
"""
import os
import numpy as np
from PIL import Image

from .download import ensure_model

# name -> (mean, std, input size, post)
SPECS = {
    "birefnet-general": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), (1024, 1024), "sigmoid"),
    "isnet-general-use": ((0.5, 0.5, 0.5), (1.0, 1.0, 1.0), (1024, 1024), "plain"),
    "u2net": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), (320, 320), "clip"),
}

_SESSIONS = {}


def _session(name, models_dir=None):
    if name not in _SESSIONS:
        import onnxruntime as ort
        path = ensure_model(name, models_dir)
        opts = ort.SessionOptions()
        threads = os.environ.get("WHITEBG_THREADS")
        if threads:
            opts.intra_op_num_threads = int(threads)
        _SESSIONS[name] = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
    return _SESSIONS[name]


def normalize(img, mean, std, size):
    """rembg BaseSession.normalize, line for line"""
    im = img.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    a = np.array(im)
    a = a / max(np.max(a), 1e-6)
    t = np.zeros((a.shape[0], a.shape[1], 3))
    for c in range(3):
        t[:, :, c] = (a[:, :, c] - mean[c]) / std[c]
    return np.expand_dims(t.transpose((2, 0, 1)), 0).astype(np.float32)


def postprocess(raw, post, size):
    pred = raw[:, 0, :, :]
    if post == "sigmoid":
        pred = 1 / (1 + np.exp(-pred))
    ma, mi = np.max(pred), np.min(pred)
    pred = np.squeeze((pred - mi) / (ma - mi))
    if post == "clip":
        pred = pred.clip(0, 1)
    mask = Image.fromarray((pred * 255).astype("uint8"), mode="L")
    return mask.resize(size, Image.Resampling.LANCZOS)


def predict_mask(img, name, models_dir=None):
    """img: PIL RGB image (already EXIF transposed). Returns a PIL 'L' mask at img.size."""
    mean, std, size, post = SPECS[name]
    s = _session(name, models_dir)
    x = normalize(img, mean, std, size)
    out = s.run(None, {s.get_inputs()[0].name: x})
    return postprocess(out[0], post, img.size)
