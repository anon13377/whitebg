"""Compare two square finals at review size (shared by selftest and the golden harness)."""
import numpy as np, cv2
from PIL import Image, ImageOps

REVIEW = 1000


def _load(p):
    return np.asarray(ImageOps.exif_transpose(Image.open(p)).convert("RGB"))


def item_mask(a, thr=245):
    return (a < thr).any(2)


def compare_finals(new_path, ref_path, size=REVIEW):
    a, b = _load(new_path), _load(ref_path)
    ra = cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA)
    rb = cv2.resize(b, (size, size), interpolation=cv2.INTER_AREA)
    ma, mb = item_mask(ra), item_mask(rb)
    inter, union = int((ma & mb).sum()), int((ma | mb).sum())
    d = np.abs(ra.astype(int) - rb.astype(int)).max(2)
    return {"iou": round(inter / max(union, 1), 5),
            "px_diff_pct": round(100 * float((ma ^ mb).mean()), 4),
            "mean_abs_diff": round(float(d.mean()), 3),
            "p99_abs_diff": int(np.percentile(d, 99)),
            "new_side": a.shape[0], "ref_side": b.shape[0]}
