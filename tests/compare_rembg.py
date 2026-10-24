"""Developer check: whitebg masks (onnxruntime direct) vs rembg masks on the same photos.
usage: python tests/compare_rembg.py PHOTO.jpg [PHOTO.jpg ...] [--models-dir DIR]
Needs rembg installed (dev only, not a runtime dependency). Pass = under 1 percent of pixels differ at 0.5."""
import sys, json, argparse
import numpy as np
from PIL import Image
from whitebg.common import load_rgb
from whitebg.models import predict_mask, SPECS


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("photos", nargs="+"); ap.add_argument("--models-dir")
    a = ap.parse_args()
    import warnings; warnings.filterwarnings("ignore")
    import rembg
    ok = True
    for p in a.photos:
        im = Image.fromarray(load_rgb(p))
        for m in SPECS:
            ours = np.asarray(predict_mask(im, m, a.models_dir)).astype(int)
            ref = np.asarray(rembg.remove(im, session=rembg.new_session(m), only_mask=True)).astype(int)
            diff_pct = 100 * float(((ours > 127) != (ref > 127)).mean())
            r = {"photo": p, "model": m, "binary_diff_pct": round(diff_pct, 5), "max_abs": int(np.abs(ours - ref).max()),
                 "identical": bool((ours == ref).all())}
            ok &= diff_pct < 1.0
            print(json.dumps(r), flush=True)
    print("MASK MATCH", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
