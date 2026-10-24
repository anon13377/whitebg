"""selftest: rerun loop 1 on reference raws and compare to approved finals at review size.

usage: whitebg selftest --raw-dir DIR --final-dir DIR [--names img_005 img_009] [--out DIR] [--models-dir DIR]
                        [--min-iou 0.98]
Paths can also come from a local, never committed whitebg_local.json (see README developer section):
  {"selftest": {"raw_dir": "...", "final_dir": "...", "names": ["img_005", "img_009"]}}
Writes only into a new timestamped folder under --out (default ./selftest_runs). Exit code 1 on FAIL.
"""
import os, sys, json, time, argparse
from .cutout import run_one
from .compare import compare_finals


def local_cfg():
    for p in (os.environ.get("WHITEBG_LOCAL"), "whitebg_local.json"):
        if p and os.path.exists(p):
            return json.load(open(p, encoding="utf-8"))
    return {}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="whitebg selftest")
    ap.add_argument("--raw-dir"); ap.add_argument("--final-dir")
    ap.add_argument("--names", nargs="*")
    ap.add_argument("--out", default="selftest_runs")
    ap.add_argument("--models-dir", default=None)
    ap.add_argument("--min-iou", type=float, default=0.98)
    a = ap.parse_args(argv)
    lc = local_cfg().get("selftest", {})
    raw_dir = a.raw_dir or lc.get("raw_dir"); fin_dir = a.final_dir or lc.get("final_dir")
    names = a.names or lc.get("names") or ["img_005", "img_009"]
    if not raw_dir or not fin_dir:
        raise SystemExit("selftest needs --raw-dir and --final-dir (or whitebg_local.json)")
    run = os.path.join(a.out, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run, exist_ok=False)
    results, ok = {}, True
    for n in names:
        t0 = time.time()
        log = run_one(os.path.join(raw_dir, n + ".jpg"), os.path.join(run, n, "loop1"),
                      cache=os.path.join(a.out, "_cache"), models_dir=a.models_dir, quiet=True)
        r = compare_finals(os.path.join(run, n, "loop1", "final.jpg"), os.path.join(fin_dir, n + ".jpg"))
        r["alpha1_px_changed"] = log["alpha1_px_changed"]
        r["seconds"] = round(time.time() - t0, 1)
        r["pass"] = r["iou"] >= a.min_iou and log["alpha1_px_changed"] == 0
        ok &= r["pass"]
        results[n] = r
        print(n, json.dumps(r), flush=True)
    json.dump({"min_iou": a.min_iou, "results": results, "pass": ok}, open(os.path.join(run, "selftest.json"), "w"), indent=1)
    print("SELFTEST", "PASS" if ok else "FAIL", "->", run)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
