"""Resumable, hash checked model downloader.

Models live in a folder chosen in this order:
  1. the models_dir argument (CLI --models-dir)
  2. the WHITEBG_MODELS environment variable
  3. %LOCALAPPDATA%\\whitebg\\models (Windows) or ~/.local/share/whitebg/models

Lookup also accepts an existing rembg cache layout (<dir>/<name>/<name>.onnx), so pointing --models-dir at
~/.rembg/models reuses files already on disk instead of downloading 1.2 GB again.
A file is used only after its SHA-256 matches models.json; the verified hash is remembered in a small
<file>.verified stamp (size + mtime) so the 1 GB file is not rehashed on every start.
"""
import os, json, hashlib, urllib.request
from pathlib import Path

MANIFEST = Path(__file__).with_name("models.json")


def manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))["models"]


def default_dir():
    env = os.environ.get("WHITEBG_MODELS")
    if env:
        return Path(env)
    base = os.environ.get("LOCALAPPDATA")
    if base:
        return Path(base) / "whitebg" / "models"
    return Path.home() / ".local" / "share" / "whitebg" / "models"


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _verified(p, spec):
    st = p.stat()
    if st.st_size != spec["size"]:
        return False
    stamp = Path(str(p) + ".verified")
    key = f'{spec["sha256"]}|{st.st_size}|{int(st.st_mtime)}'
    try:
        if stamp.read_text().strip() == key:
            return True
    except OSError:
        pass
    if sha256_file(p) != spec["sha256"]:
        return False
    try:
        stamp.write_text(key)
    except OSError:
        pass  # read only cache folder: fine, we just rehash next time
    return True


def candidates(name, d):
    d = Path(d)
    return [d / f"{name}.onnx", d / name / f"{name}.onnx"]


def find_model(name, models_dir=None):
    spec = manifest()[name]
    for p in candidates(name, models_dir or default_dir()):
        if p.exists() and _verified(p, spec):
            return p
    return None


def download(name, models_dir=None, progress=True):
    spec = manifest()[name]
    d = Path(models_dir or default_dir())
    d.mkdir(parents=True, exist_ok=True)
    dst = d / f"{name}.onnx"
    part = d / f"{name}.onnx.part"
    have = part.stat().st_size if part.exists() else 0
    if have > spec["size"]:
        raise SystemExit(f"{part} is larger than expected; delete it by hand and retry")
    if have < spec["size"]:
        req = urllib.request.Request(spec["url"], headers={"Range": f"bytes={have}-"} if have else {})
        with urllib.request.urlopen(req, timeout=60) as r:
            if have and r.status != 206:
                have = 0  # server ignored the range; start over
            mode = "ab" if have else "wb"
            with open(part, mode) as f:
                done, last = have, -1
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
                    done += len(b)
                    pct = int(100 * done / spec["size"])
                    if progress and pct != last and pct % 5 == 0:
                        print(f"  {name}: {pct}% ({done >> 20} of {spec['size'] >> 20} MB)", flush=True)
                        last = pct
    got = sha256_file(part)
    if got != spec["sha256"]:
        raise SystemExit(f"{name}: SHA-256 mismatch ({got}); the partial file was kept at {part}. Delete it and retry.")
    if dst.exists():
        raise SystemExit(f"refusing to overwrite {dst}")
    part.rename(dst)
    _verified(dst, spec)
    return dst


def ensure_model(name, models_dir=None):
    p = find_model(name, models_dir)
    if p:
        return p
    print(f"downloading {name} ({manifest()[name]['size'] >> 20} MB) to {Path(models_dir or default_dir())}", flush=True)
    return download(name, models_dir)
