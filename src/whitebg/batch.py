"""Run one loop for many photos. usage: whitebg batch SOURCES.json ROOT LOOPNAME [--hunt] [--models-dir DIR] [--only key ...]
SOURCES.json maps "<category>/<item>" -> source photo path. Per item config: ROOT/<key>/cfg_<LOOPNAME>.json (optional).
Writes ROOT/<key>/<LOOPNAME>/ via cutout then evidence. Never overwrites."""
import sys, os, json
from .cutout import run_one
from . import evidence


def main(argv=None):
    argv = [None] + list(sys.argv[1:] if argv is None else argv)
    src, root, loop = argv[1], argv[2], argv[3]
    hunt = "--hunt" in argv
    models_dir = argv[argv.index("--models-dir") + 1] if "--models-dir" in argv else None
    if "--models-dir" in argv:
        i = argv.index("--models-dir"); del argv[i:i + 2]
    only = argv[argv.index("--only") + 1:] if "--only" in argv else None
    for key, p in json.load(open(src)).items():
        if only and key not in only:
            continue
        d = os.path.join(root, key, loop)
        cfg = os.path.join(root, key, f"cfg_{loop}.json")
        try:
            log = run_one(p, d, cfg if os.path.exists(cfg) else None, os.path.join(root, "_cache"), models_dir, quiet=True)
            print("==", key, json.dumps({k: v for k, v in log.items() if k not in ("config", "signature")})[:400], flush=True)
            evidence.main([p, d] + (["--hunt"] if hunt else []))
        except SystemExit as e:
            print("==", key, "SKIPPED:", e, flush=True)


if __name__ == "__main__":
    main()
