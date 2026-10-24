"""Independent verifier helper: register a final back to its raw (SIFT + similarity RANSAC) and compare.
usage: whitebg register RAW.jpg FINAL.jpg OUTDIR
Prints scale/rotation/shift (expect scale = placement scale, rot ~0), inlier median residual, and the item
pixel difference after warping (JPEG noise only expected). Writes OUTDIR/reg_diff.png (abs diff x4) and
OUTDIR/reg_side.jpg. Read only on inputs; refuses to overwrite."""
import sys, os, json
import numpy as np, cv2
from .common import load_rgb


def main(argv=None):
    argv = [None] + list(sys.argv[1:] if argv is None else argv)
    raw, fin, od = argv[1:4]
    os.makedirs(od, exist_ok=True)
    for f in ("reg_diff.png", "reg_side.jpg"):
        if os.path.exists(os.path.join(od, f)):
            raise SystemExit("refusing to overwrite " + f)
    a = cv2.cvtColor(load_rgb(raw), cv2.COLOR_RGB2BGR); b = cv2.imread(fin)
    sift = cv2.SIFT_create(8000)
    ka, da = sift.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None)
    kb, db = sift.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None)
    good = [x for x, y in cv2.BFMatcher().knnMatch(da, db, k=2) if x.distance < 0.75 * y.distance]
    pa = np.float32([ka[g.queryIdx].pt for g in good]); pb = np.float32([kb[g.trainIdx].pt for g in good])
    M, inl = cv2.estimateAffinePartial2D(pa, pb, ransacReprojThreshold=1.5, maxIters=20000)
    res = np.linalg.norm((pa @ M[:, :2].T + M[:, 2]) - pb, axis=1)[inl.ravel() == 1]
    w = cv2.warpAffine(a, M, (b.shape[1], b.shape[0]), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    item = (b < 245).any(2)
    item = cv2.erode(item.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    d = np.abs(w.astype(int) - b.astype(int)).max(2)
    cv2.imwrite(os.path.join(od, "reg_diff.png"), np.clip(d * 4, 0, 255).astype(np.uint8))
    h = 900; sa = cv2.resize(a, (int(a.shape[1] * h / a.shape[0]), h)); sb = cv2.resize(b, (h, h))
    cv2.imwrite(os.path.join(od, "reg_side.jpg"), np.hstack([sa, np.full((h, 20, 3), 128, np.uint8), sb]))
    print(json.dumps({"scale": round(float(np.hypot(M[0, 0], M[1, 0])), 5), "rot_deg": round(float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))), 4),
                      "tx": round(float(M[0, 2]), 2), "ty": round(float(M[1, 2]), 2), "inliers": int(inl.sum()), "matches": len(good),
                      "median_residual_px": round(float(np.median(res)), 3),
                      "item_interior_diff_p99": int(np.percentile(d[item], 99)), "item_interior_diff_max": int(d[item].max())}))


if __name__ == "__main__":
    main()
