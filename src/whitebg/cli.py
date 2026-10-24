"""whitebg command line.

  whitebg run PHOTO_OR_FOLDER... -o OUT_DIR     loop 1 cutout for each photo, finals copied to OUT_DIR
  whitebg review OUT.html ITEMS.json            static raw vs final review page (P1 stub; real UI in P5)
  whitebg audit-white SRC.jpg OUT.jpg           already white photo: trim to the approved square look, no cutout
  whitebg selftest --raw-dir D --final-dir D    rerun loop 1 on reference raws, compare to approved finals
  whitebg models [--models-dir D] [--download]  show or fetch the model files
Advanced: cutout, evidence, batch, register (same arguments as the original scripts).
"""
import os, sys, json, glob, shutil, argparse

EXTS = (".jpg", ".jpeg", ".png", ".webp", ".heic")


def cmd_run(argv):
    from .cutout import run_one
    from . import evidence
    ap = argparse.ArgumentParser(prog="whitebg run")
    ap.add_argument("inputs", nargs="+"); ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--config", default=None, help="JSON config applied to every photo")
    ap.add_argument("--models-dir", default=None); ap.add_argument("--no-evidence", action="store_true")
    a = ap.parse_args(argv)
    photos = []
    for p in a.inputs:
        if os.path.isdir(p):
            photos += sorted(f for f in glob.glob(os.path.join(p, "*")) if f.lower().endswith(EXTS))
        else:
            photos.append(p)
    work = os.path.join(a.out, "_work")
    for p in photos:
        stem = os.path.splitext(os.path.basename(p))[0]
        loop = os.path.join(work, stem, "loop1")
        dst = os.path.join(a.out, stem + "_white.jpg")
        if os.path.exists(dst):
            print("skip (exists, never overwritten):", dst); continue
        log = run_one(p, loop, a.config, os.path.join(work, "_cache"), a.models_dir, quiet=True)
        if not a.no_evidence:
            evidence.main([p, loop])
        shutil.copyfile(os.path.join(loop, "final.jpg"), dst)
        print("done", dst, "fill", log["placement"]["fill_pct"], "alpha1_px_changed", log["alpha1_px_changed"], flush=True)


def cmd_models(argv):
    from .download import manifest, find_model, ensure_model, default_dir
    ap = argparse.ArgumentParser(prog="whitebg models")
    ap.add_argument("--models-dir", default=None); ap.add_argument("--download", action="store_true")
    a = ap.parse_args(argv)
    print("models folder:", a.models_dir or default_dir())
    for n, s in manifest().items():
        p = ensure_model(n, a.models_dir) if a.download else find_model(n, a.models_dir)
        print(f"  {n:20s} {s['size'] >> 20:5d} MB  {'OK ' + str(p) if p else 'missing'}")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__); return
    cmd, rest = argv[0], argv[1:]
    if cmd == "run":
        return cmd_run(rest)
    if cmd == "models":
        return cmd_models(rest)
    mods = {"review": "review", "audit-white": "whitebox", "selftest": "selftest", "cutout": "cutout",
            "evidence": "evidence", "batch": "batch", "register": "register"}
    if cmd not in mods:
        raise SystemExit(f"unknown command {cmd}\n{__doc__}")
    import importlib
    return importlib.import_module("whitebg." + mods[cmd]).main(rest)


if __name__ == "__main__":
    main()
