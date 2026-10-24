"""Already-white photos: AUDIT ONLY, no cutout. Trim canvas to the approved look straight from the RAW.
usage: whitebg audit-white SRC.jpg OUT.jpg
item = pixels with any channel < 245 (grouped, isolated far specks ignored); background outside hull+8 px set to 255;
square side = round(max/0.88) capped 2048; saved ONCE at JPEG q100 4:4:4 (never re-save repeatedly at q95, that
drifted up to 21 levels on boxed product photos)."""
import sys, os, json
import numpy as np, cv2
from PIL import Image
from .common import load_rgb


def main(argv=None):
    argv = [None] + list(sys.argv[1:] if argv is None else argv)
    src, dst = argv[1], argv[2]
    if os.path.exists(dst):
        raise SystemExit("refusing to overwrite " + dst)
    I = load_rgb(src)
    dark = (I < 245).any(2).astype(np.uint8)
    grp = cv2.dilate(dark, np.ones((25, 25), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(grp, 8)
    big = 1 + int(np.argmax(st[1:, 4]))
    keep = (lab == big) & (dark > 0)
    hull = cv2.convexHull(cv2.findNonZero(keep.astype(np.uint8)))
    hm = np.zeros(dark.shape, np.uint8); cv2.fillPoly(hm, [hull], 1)
    near = cv2.dilate(hm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))).astype(bool)
    out = I.copy(); out[~near] = 255
    ys, xs = np.nonzero(keep)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    bw, bh = x1 - x0, y1 - y0; m = max(bw, bh)
    side = int(round(m / 0.88)); s = 1.0
    if side > 2048:
        side, s = 2048, 2048 * 0.88 / m
    crop = out[y0:y1, x0:x1]
    if s < 1:
        crop = cv2.resize(crop, (int(round(bw * s)), int(round(bh * s))), interpolation=cv2.INTER_AREA)
    nh, nw = crop.shape[:2]
    canvas = np.full((side, side, 3), 255, np.uint8)
    ox, oy = (side - nw) // 2, (side - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = crop
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    Image.fromarray(canvas).save(dst, quality=100, subsampling=0)
    print(json.dumps({"bbox_src_xywh": [int(x0), int(y0), int(bw), int(bh)], "side": side, "scale": round(s, 5),
                      "placed_xywh": [int(ox), int(oy), int(nw), int(nh)], "stray_px_ignored": int(((dark > 0) & ~keep).sum()),
                      "item_identical_to_raw": bool((out[near] == I[near]).all())}))


if __name__ == "__main__":
    main()
