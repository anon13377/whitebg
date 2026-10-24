"""Golden IoU harness (local only; photos are never committed).

usage: python tests/golden_iou.py tests/golden_local.json [--out DIR] [--models-dir DIR] [--min-iou 0.995] [--only name ...]

The manifest is a JSON list (see tests/golden_manifest.example.json):
  [{"name": "...", "raw": "path/raw.jpg", "final": "path/approved_final.jpg",
    "config_from_log": "path/log.json (optional: reuse the exact config that produced the approved final)",
    "config": {... optional inline config ...}}]
For each entry the harness reruns the cutout into a new folder, compares the new final to the approved final at
review size (IoU of item pixels, any channel < 245) and checks alpha1_px_changed == 0 (item pixels untouched).
Release gate target (council agenda 3): IoU >= 0.995 and alpha1_px_changed == 0 on every verified final.
Entries without the config used for the approved final measure loop 1 only and are reported, not gated.
"""
import os, sys, json, time, argparse
from whitebg.cutout import run_one
from whitebg.compare import compare_finals


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest"); ap.add_argument("--out", default="golden_runs")
    ap.add_argument("--models-dir", default=None); ap.add_argument("--min-iou", type=float, default=0.995)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args(argv)
    items = json.load(open(a.manifest, encoding="utf-8"))
    run = os.path.join(a.out, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run, exist_ok=False)
    rows, gated_ok = [], True
    for it in items:
        if a.only and it["name"] not in a.only:
            continue
        cfg = it.get("config")
        if cfg is None and it.get("config_from_log"):
            cfg = json.load(open(it["config_from_log"], encoding="utf-8"))["config"]
        loop = os.path.join(run, it["name"].replace("/", "_"))
        try:
            log = run_one(it["raw"], loop, cfg_dict=cfg or {}, cache=os.path.join(a.out, "_cache"),
                          models_dir=a.models_dir, quiet=True)
        except Exception as e:  # keep going; report the failure
            rows.append({"name": it["name"], "error": repr(e)[:300]}); gated_ok = False
            print(it["name"], "ERROR", e, flush=True); continue
        r = compare_finals(os.path.join(loop, "final.jpg"), it["final"])
        r.update(name=it["name"], gated=cfg is not None, alpha1_px_changed=log["alpha1_px_changed"])
        r["pass"] = r["iou"] >= a.min_iou and log["alpha1_px_changed"] == 0
        if r["gated"]:
            gated_ok &= r["pass"]
        rows.append(r)
        print(json.dumps(r), flush=True)
    json.dump({"min_iou": a.min_iou, "rows": rows, "gated_pass": gated_ok}, open(os.path.join(run, "golden.json"), "w"), indent=1)
    print("GOLDEN gated", "PASS" if gated_ok else "FAIL", "| loop1-only rows reported:", sum(1 for r in rows if not r.get("gated")), "->", run)
    return 0 if gated_ok else 1


if __name__ == "__main__":
    sys.exit(main())
