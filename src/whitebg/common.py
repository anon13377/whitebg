"""Shared helpers. Generalized from the original 2026-10-24 product photo pipeline."""
import os, json, hashlib
import numpy as np, cv2
from PIL import Image

MODELS = ["birefnet-general", "isnet-general-use", "u2net"]


def disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def normconv(img, w, sigma):
    """weighted gaussian average of img with weights w -> (avg, weightsum)"""
    k = int(sigma * 3) * 2 + 1
    ws = cv2.GaussianBlur(w.astype(np.float32), (k, k), sigma)
    if img.ndim == 3:
        num = cv2.GaussianBlur(img * w[..., None], (k, k), sigma)
        return num / np.maximum(ws, 1e-6)[..., None], ws
    num = cv2.GaussianBlur(img * w, (k, k), sigma)
    return num / np.maximum(ws, 1e-6), ws


def load_rgb(path):
    """read only; EXIF orientation applied so masks and outputs match what the user sees"""
    from PIL import ImageOps
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    return np.asarray(im.convert("RGB"))


def src_key(path):
    st = os.stat(path)
    h = hashlib.sha1(f"{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:12]
    return os.path.splitext(os.path.basename(path))[0] + "_" + h


def load_masks(path, cache, models_dir=None):
    """3 model masks (float 0..1), cached per source file under cache/"""
    os.makedirs(cache, exist_ok=True)
    key = src_key(path)
    out = []
    need = [m for m in MODELS if not os.path.exists(os.path.join(cache, f"{key}_{m}.png"))]
    if need:
        from .models import predict_mask
        im = Image.fromarray(load_rgb(path))
        for m in need:
            predict_mask(im, m, models_dir).save(os.path.join(cache, f"{key}_{m}.png"))
            print("mask", m, key, flush=True)
    for m in MODELS:
        out.append(np.asarray(Image.open(os.path.join(cache, f"{key}_{m}.png")).convert("L")).astype(np.float32) / 255)
    return out


def rect_mask(shape, rects):
    m = np.zeros(shape, bool)
    for r in rects:
        x0, y0, x1, y1 = r[:4]
        m[max(y0, 0):y1, max(x0, 0):x1] = True
    return m


def poly_mask(shape, polys):
    m = np.zeros(shape, np.uint8)
    for p in polys:
        cv2.fillPoly(m, [np.array(p, np.int32)], 1)
    return m.astype(bool)


def geodesic(seed, allowed, maxit=600):
    cur = seed & allowed
    k = np.ones((3, 3), np.uint8)
    for _ in range(maxit):
        nxt = cv2.dilate(cur.astype(np.uint8), k).astype(bool) & allowed
        if (nxt == cur).all():
            break
        cur = nxt
    return cur


def place(out, alpha=None, fill=0.88, cap=2048):
    """The approved listing look. bbox = union of (any channel < 245) and (alpha > 0.05) so white item parts are never
    trimmed; side = round(max(w,h)/fill) capped at cap; item at 1:1 unless it cannot fit; centred; white pad.
    An item that runs off the photo edge is padded, never cropped (the bbox just reaches the edge)."""
    dark = (out < 245).any(2)
    if alpha is not None:
        dark |= alpha > 0.05
    ys, xs = np.nonzero(dark)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    bw, bh = x1 - x0, y1 - y0
    m = max(bw, bh)
    side = int(round(m / fill))
    s = 1.0
    if side > cap:
        side = cap
        s = cap * fill / m
    crop = out[y0:y1, x0:x1]
    if s < 1.0:
        nw, nh = int(round(bw * s)), int(round(bh * s))
        crop = cv2.resize(crop.astype(np.float32), (nw, nh), interpolation=cv2.INTER_AREA)
    else:
        nw, nh = bw, bh
    canvas = np.full((side, side, 3), 255.0, np.float32)
    ox, oy = (side - nw) // 2, (side - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = crop
    img = np.clip(np.round(canvas), 0, 255).astype(np.uint8)
    h, w = out.shape[:2]
    touches = [bool(x0 == 0), bool(y0 == 0), bool(x1 == w), bool(y1 == h)]
    return img, {"bbox_src_xywh": [int(x0), int(y0), int(bw), int(bh)], "side": side, "scale": round(s, 5),
                 "placed_xywh": [int(ox), int(oy), int(nw), int(nh)], "fill_pct": round(100 * max(nw, nh) / side, 2),
                 "touches_photo_edge_LTRB": touches,
                 "coord_rule": "final = (src - bbox_xy) * scale + placed_xy"}


def save_jpeg(arr, path, quality=95):
    if os.path.exists(path):
        raise SystemExit(f"refusing to overwrite {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr).save(path, quality=quality, subsampling=0)


def save_new(path, writer):
    """writer(path) only if path does not exist (never overwrite)"""
    if os.path.exists(path):
        raise SystemExit(f"refusing to overwrite {path}")
    writer(path)


def load_cfg(path):
    if not path or not os.path.exists(path):
        return {}
    return json.load(open(path))
