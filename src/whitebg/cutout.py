"""Generalized white background cutout (one photo, one loop).

usage: whitebg cutout SRC.jpg LOOP_DIR [CONFIG.json] [--cache DIR] [--models-dir DIR]

Writes ONLY into LOOP_DIR (refuses to run if LOOP_DIR/final.jpg exists):
  final.jpg            square #FFFFFF canvas, item 1:1, ~88 percent fill (JPEG q95 4:4:4)
  work_full.png        full-resolution composite (source geometry)
  work_alpha.png       alpha 0..255
  work_layers.npz      masks: base, added, carved, shadow, protect, itemlike, shadowlike
  log.json             every step's pixel counts + placement + the derived item colour signature

Method (original product photo pipeline, generalized):
 1. birefnet-general base matte (only model that cuts holes and slots); isnet + u2net as second opinions.
 2. local background field Bg from pixels outside the union of all models (handles grey sweeps and gradients).
 3. ITEM COLOUR SIGNATURE derived per photo (replaces the hardcoded green ratio rule):
      k = (I.Bg)/(Bg.Bg) brightness of a pixel relative to the local background
      res = |I - k Bg| / |I| chromatic deviation from "background lit differently"
    floor and shadow have res near 0 with k < 1; the item core is clustered (k-means, Lab) and every pixel
    gets d_item (Lab distance to nearest item cluster) and d_bg (Lab distance to k Bg).
    itemlike = d_item + margin < d_bg ; shadowlike = res < res_max and k < k_max and not itemlike.
    Clusters that are themselves shadowlike (white, grey, black items on a neutral floor) are flagged
    "bg_chroma_item": automatic shadow carving is then limited to thin bands and the loop must decide by eye.
 4. add back: pixels near the matte that isnet or u2net call item and that are strongly itemlike.
 5. shadow carve: thick shadowlike bodies connected to background within maxdist (config "shadow").
 6. config carves: hole seeds, rect_carves, poly_carves (hand-verified), protect rects/polys, addback rects.
 7. edge band colour-line matting against the local Bg; interior pixels copied untouched; exterior = 255.
 8. place on the approved square canvas (common.place).
"""
import os, json, argparse
import numpy as np, cv2
from scipy import ndimage as ndi
from .common import (disk, normconv, load_rgb, load_masks, rect_mask, poly_mask, geodesic, place, save_jpeg, load_cfg)

DEFAULT = {
    "base_thr": 0.5,
    "bg_sigma": 25,
    "k_clusters": 8,
    "margin": 6.0,          # Lab units: itemlike / shadowlike decision margin
    "res_max": 0.075,       # chromatic deviation below which a pixel is "background lit differently"
    "k_max": 0.97,          # darker than local background -> possible shadow
    "add": {"reach": 20, "min_area": 6, "res_min": 0.10},
    "shadow": {"maxdist": 40, "open": 2, "extend": 2, "island": 25, "vmax_k": 0.92},
    "band_in": 2, "band_out": 2,
    "hole_seeds": [], "rect_carves": [], "poly_carves": [], "protect_rects": [], "protect_polys": [],
    "addback_rects": [], "addback_polys": [], "force_bg_rects": [],
    "union_rects": [],      # inside these rects use the union of all 3 models (a model dropped a part)
    "jpeg_q": 95,
}


def lab(I):
    return cv2.cvtColor((I / 255.0).astype(np.float32), cv2.COLOR_RGB2Lab)


def merge(a, b):
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


def signature(I, L, Bg, core, c, rng):
    """k-means item colour clusters on the sure-item core; returns centres + per-cluster bg-chroma flag"""
    pts = np.nonzero(core)
    n = len(pts[0])
    idx = rng.choice(n, min(n, 40000), replace=False)
    X = L[pts[0][idx], pts[1][idx]].astype(np.float32)
    K = int(min(c["k_clusters"], max(2, len(X) // 50)))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    _, lbl, ctr = cv2.kmeans(X, K, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    lbl = lbl.ravel()
    Ic = I[pts[0][idx], pts[1][idx]]
    Bc = Bg[pts[0][idx], pts[1][idx]]
    kk = (Ic * Bc).sum(1) / np.maximum((Bc * Bc).sum(1), 1)
    res = np.linalg.norm(Ic - kk[:, None] * Bc, axis=1) / np.maximum(np.linalg.norm(Ic, axis=1), 1)
    info = []
    for j in range(K):
        sel = lbl == j
        if not sel.any():
            continue
        r_med = float(np.median(res[sel]))
        info.append({"lab": [round(float(v), 1) for v in ctr[j]], "share": round(float(sel.mean()), 3),
                     "res_med": round(r_med, 3), "k_med": round(float(np.median(kk[sel])), 3),
                     "bg_chroma_item": bool(r_med < c["res_max"])})
    return ctr, info


def build(src, c, cache, models_dir=None):
    I8 = load_rgb(src)
    I = I8.astype(np.float32)
    h, w = I.shape[:2]
    bA, iso, u2 = load_masks(src, cache, models_dir)
    log = {"src": src, "size": [w, h]}
    B = bA > c["base_thr"]
    U = (bA > 0.5) | (iso > 0.5) | (u2 > 0.5)
    for r in c["union_rects"]:
        B |= U & rect_mask((h, w), [r])
    # 2. background field
    ext = ~cv2.dilate(U.astype(np.uint8), disk(15)).astype(bool)
    Bg, wb = normconv(I, ext.astype(np.float32), c["bg_sigma"])
    # coarse fill for deep item interiors: inpaint the background at 1/8 scale (normconv alone divides by ~0 there)
    f8 = 8
    sm_ = cv2.resize(I, (w // f8, h // f8), interpolation=cv2.INTER_AREA)
    hole8 = cv2.resize((~ext).astype(np.uint8), (w // f8, h // f8), interpolation=cv2.INTER_NEAREST)
    hole8 = cv2.dilate(hole8, np.ones((3, 3), np.uint8))
    inp = cv2.inpaint(np.clip(sm_, 0, 255).astype(np.uint8), hole8, 25, cv2.INPAINT_TELEA).astype(np.float32)
    Bg2 = cv2.GaussianBlur(cv2.resize(inp, (w, h), interpolation=cv2.INTER_LINEAR), (0, 0), 8)
    t = np.clip(wb / 0.2, 0, 1)[..., None]
    Bg = t * Bg + (1 - t) * Bg2
    Bg = np.maximum(Bg, 1)
    log["bg_median_rgb"] = [int(v) for v in np.median(I[ext], 0)] if ext.any() else None
    # 3. signature
    L = lab(I)
    kmap = (I * Bg).sum(2) / np.maximum((Bg * Bg).sum(2), 1)
    res = np.linalg.norm(I - kmap[..., None] * Bg, axis=2) / np.maximum(np.linalg.norm(I, axis=2), 1)
    core = cv2.erode(((bA > 0.5) & (iso > 0.5) & (u2 > 0.5)).astype(np.uint8), disk(6)).astype(bool)
    if core.sum() < 200:
        core = cv2.erode(B.astype(np.uint8), disk(4)).astype(bool)
    rng = np.random.default_rng(0)
    ctr, sig = signature(I, L, Bg, core, c, rng)
    log["signature"] = sig
    log["bg_chroma_item_share"] = round(sum(s["share"] for s in sig if s["bg_chroma_item"]), 3)
    Lsh = lab(np.clip(kmap[..., None] * Bg, 0, 255))
    d_bg = np.linalg.norm(L - Lsh, axis=2)
    d_item = np.full((h, w), 1e9, np.float32)
    for cc in ctr:
        d_item = np.minimum(d_item, np.linalg.norm(L - cc[None, None, :], axis=2))
    itemlike = d_item + c["margin"] < d_bg
    shadowlike = (res < c["res_max"]) & (kmap < c["k_max"]) & ~itemlike
    protect = rect_mask((h, w), c["protect_rects"]) | poly_mask((h, w), c["protect_polys"])
    # 4. add back
    a = c["add"]
    near = cv2.dilate(B.astype(np.uint8), disk(a["reach"])).astype(bool)
    other = (iso > 0.5) | (u2 > 0.5)
    cand = near & ~B & other & itemlike & (res > a["res_min"])
    labc, n = ndi.label(cand | B)
    keep = set(np.unique(labc[B])) - {0}
    added = cand & np.isin(labc, list(keep))
    lab2, n2 = ndi.label(added)
    if n2:
        sizes = ndi.sum(added, lab2, range(1, n2 + 1))
        small = np.isin(lab2, [i + 1 for i, s in enumerate(sizes) if s < a["min_area"]])
        added &= ~small
    added |= (rect_mask((h, w), c["addback_rects"]) | poly_mask((h, w), c["addback_polys"])) & U
    # majority restore (vetting fix: Scorpios Rex white box panel on a white floor was dropped by birefnet):
    # isnet AND u2net call it item, it is outside birefnet's outline (not an enclosed hole, holes stay
    # birefnet's call), and it touches the matte.
    if c.get("majority_restore", True):
        nb = (~B).astype(np.uint8)
        ff = np.zeros((h + 2, w + 2), np.uint8)
        outside = np.zeros((h, w), bool)
        nlab, labo = cv2.connectedComponents(nb, connectivity=4)
        border_ids = set(np.unique(np.concatenate([labo[0], labo[-1], labo[:, 0], labo[:, -1]]))) - {0}
        outside = np.isin(labo, list(border_ids)) & (nb > 0)
        # final-hunt fix (Mario Kart card: textured black floor under the hang tab was restored): never restore
        # pixels that look like the local background (Lab distance to Bg small) or like its shadow.
        d_raw = np.linalg.norm(L - lab(Bg), axis=2)
        maj = (iso > 0.5) & (u2 > 0.5) & outside & ~rect_mask((h, w), c["force_bg_rects"])
        maj &= (d_raw > c.get("maj_bg_dist", 12)) & ~shadowlike
        labm, nm = ndi.label(maj | B)
        keepm = set(np.unique(labm[B])) - {0}
        maj &= np.isin(labm, list(keepm))
        maj = cv2.morphologyEx(maj.astype(np.uint8), cv2.MORPH_OPEN, disk(2)).astype(bool)
        added |= maj
        log["majority_restore_px"] = int(maj.sum())
    log["added_px"] = int(added.sum())
    M = B | added
    # 5. shadow carve
    carved = np.zeros((h, w), bool)
    sc = c.get("shadow")
    if sc and sc.get("on", True):
        bgm = ~M
        dist = cv2.distanceTransform(M.astype(np.uint8), cv2.DIST_L2, 3)
        cond = M & shadowlike & (dist <= sc["maxdist"]) & (kmap < sc["vmax_k"]) & ~protect & ~added
        # model-support safeguard (vetting fix 2026-10-24: Scorpios Rex white box front panel and the
        # Catch and Release / Ordinary box edges were carved as shadow while birefnet AND u2net called them item):
        # carve only where at least 2 of the 3 models call the pixel background, or birefnet itself is unsure.
        nbg = (bA < 0.5).astype(np.uint8) + (iso < 0.5) + (u2 < 0.5)
        # loop 3 fix (Catch and Release black case rim: birefnet sure item, isnet + u2net both said bg, carved):
        # birefnet's confident item pixels are never auto-carved; floor shadow there needs a hand-verified
        # rect_carve / poly_carve (exactly how the original golden frames were finished), unless "trust_votes" is set.
        if sc.get("trust_votes", False):
            support = (nbg >= sc.get("min_bg_votes", 2)) | (bA < sc.get("unsure", 0.6))
        else:
            support = (nbg >= sc.get("min_bg_votes", 2)) & (bA < sc.get("unsure", 0.9))
        log["shadow_blocked_by_models_px"] = int((cond & ~support).sum())
        cond &= support
        # neutral item safeguard (vetting fix 2026-10-24, black DVD case frame was carved as shadow):
        # a shadowlike pixel that is close to a bg-chroma item cluster (white/grey/black item parts) is only
        # carved where the models disagree (some model calls it background or birefnet is unsure).
        bgc = [np.array(s["lab"], np.float32) for s in sig if s["bg_chroma_item"]]
        if bgc:
            d_bgc = np.full((h, w), 1e9, np.float32)
            for cc in bgc:
                d_bgc = np.minimum(d_bgc, np.linalg.norm(L - cc[None, None, :], axis=2))
            disagree = (iso < 0.5) & (u2 < 0.5)
            amb = (d_bgc < sc.get("amb_dist", 12)) & ~disagree
            log["shadow_blocked_neutral_px"] = int((cond & amb).sum())
            cond &= ~amb
        if sc.get("rects"):
            cond &= rect_mask((h, w), sc["rects"])
        ker = disk(sc["open"])
        body = cv2.morphologyEx(cond.astype(np.uint8), cv2.MORPH_OPEN, ker).astype(bool)
        seed = body & cv2.dilate(bgm.astype(np.uint8), disk(sc["open"] + 1)).astype(bool)
        grown = geodesic(seed, body)
        if sc["extend"]:
            grown = geodesic(grown, cond & cv2.dilate(grown.astype(np.uint8), disk(sc["extend"])).astype(bool), maxit=sc["extend"])
        inv = (~(grown | bgm)).astype(np.uint8)
        nI, labI, st, _ = cv2.connectedComponentsWithStats(inv, 8)
        isl = np.zeros((h, w), bool)
        for i in range(1, nI):
            if st[i][4] <= sc["island"]:
                comp = labI == i
                if shadowlike[comp].mean() > 0.5 and not (comp & protect).any():
                    isl |= comp
        grown |= isl
        carved |= grown
        log["shadow_carved_px"] = int(grown.sum())
    # 6. config carves
    k3 = np.ones((3, 3), np.uint8)
    sm = cv2.medianBlur(I8, 3)
    for sd in c["hole_seeds"]:
        x, y = int(sd[0]), int(sd[1])
        tol = sd[2] if len(sd) > 2 else 8
        ffm = np.zeros((h + 2, w + 2), np.uint8)
        ffm[1:-1, 1:-1] = (~M).astype(np.uint8)
        cv2.floodFill(sm.copy(), ffm, (x, y), (0, 0, 0), (tol,) * 3, (tol,) * 3, 4 | cv2.FLOODFILL_MASK_ONLY | (2 << 8))
        reg = (ffm[1:-1, 1:-1] == 2) & ~protect
        reg = cv2.dilate(reg.astype(np.uint8), k3).astype(bool) & M & ~itemlike & ~protect | reg
        carved |= reg
        log.setdefault("holes", []).append({"seed": [x, y], "tol": tol, "px": int(reg.sum())})
    for rc in c["rect_carves"]:
        reg = rect_mask((h, w), [rc["rect"]])
        cond = reg & ~protect & M
        if rc.get("shadowlike", True):
            cond &= shadowlike | (res < rc.get("res_max", c["res_max"]))
        if "kmax" in rc:
            cond &= kmap < rc["kmax"]
        if "vmax" in rc:
            cond &= I.max(2) < rc["vmax"]
        seed = cond & cv2.dilate(((~M | carved) & reg).astype(np.uint8), k3).astype(bool)
        for sx, sy in rc.get("seeds", []):
            seed[sy, sx] = True
        g = geodesic(seed, cond | seed)
        carved |= g
        log.setdefault("rect_carves", []).append({"rect": rc["rect"], "px": int(g.sum()), "note": rc.get("note", "")})
    for pc in c["poly_carves"]:
        pm = poly_mask((h, w), [pc["poly"]]) & ~protect & M
        if "res_max" in pc:
            pm &= res < pc["res_max"]
        if "vmax" in pc:
            pm &= I.max(2) < pc["vmax"]
        if pc.get("not_itemlike"):
            pm &= ~itemlike
        carved |= pm
        log.setdefault("poly_carves", []).append({"px": int(pm.sum()), "note": pc.get("note", "")})
    for r in c["force_bg_rects"]:
        fb = rect_mask((h, w), [r]) & ~protect
        carved |= fb & M
    carved &= ~(protect & ~rect_mask((h, w), c["force_bg_rects"]))
    log["carved_px_total"] = int(carved.sum())
    M = M & ~carved
    # drop detached specks (not connected to the main item) smaller than island_max
    labM, nM = ndi.label(M, np.ones((3, 3), bool))
    if nM > 1:
        sizes = ndi.sum(M, labM, range(1, nM + 1))
        mx = sizes.max()
        drop = [i + 1 for i, s in enumerate(sizes) if s < c.get("island_max", 400) and s < mx]
        dm = np.isin(labM, drop) & ~protect & ~added
        M &= ~dm
        log["detached_specks_dropped_px"] = int(dm.sum())
    # 7. edge band matting
    Mu = M.astype(np.uint8)
    sureF = cv2.erode(Mu, disk(c["band_in"])).astype(bool)
    sureB = ~cv2.dilate(Mu, disk(c["band_out"])).astype(bool)
    farB = ~cv2.dilate(Mu, disk(c["band_out"] + 3)).astype(bool)
    farF = cv2.erode(Mu, disk(c["band_in"] + 2)).astype(bool)
    Bl, wbl = normconv(I, farB.astype(np.float32), 6)
    Bl2, _ = normconv(I, farB.astype(np.float32), 20)
    Bl = np.where((wbl < 0.02)[..., None], Bl2, Bl)
    Fg, wf = normconv(I, farF.astype(np.float32), 3)
    Fg2, wf2 = normconv(I, farF.astype(np.float32), 10)
    Fg = np.where((wf < 0.02)[..., None], Fg2, Fg)
    d = Fg - Bl
    dn2 = (d * d).sum(-1)
    ac = np.clip(((I - Bl) * d).sum(-1) / np.maximum(dn2, 1), 0, 1)
    wgt = np.clip((np.sqrt(dn2) - 25) / 50, 0, 1)
    wgt = np.where(wf2 < 0.005, 0, wgt)
    am = np.where(M, np.maximum(bA, 0.5), np.minimum(bA, 0.5))
    am[carved] = 0
    mix = wgt * ac + (1 - wgt) * am
    alpha = np.where(sureF, 1.0, np.where(sureB, 0.0, mix)).astype(np.float32)
    alpha[carved & ~cv2.dilate(M.astype(np.uint8), k3).astype(bool)] = 0
    alpha[alpha < 0.03] = 0
    alpha[alpha > 0.97] = 1
    # recomposite: I = aF + (1-a)B  ->  out = I + (1-a)(255-B)
    out = I + (1 - alpha)[..., None] * (255 - Bl)
    out = np.clip(out, 0, 255)
    out[alpha == 0] = 255
    out[alpha == 1] = I[alpha == 1]
    layers = dict(base=B, added=added, carved=carved, protect=protect, itemlike=itemlike, shadowlike=shadowlike, M=M)
    return I8, out, alpha, layers, log


def run_one(src, loop, cfg_path=None, cache=None, models_dir=None, cfg_dict=None, quiet=False):
    """one photo, one loop folder; returns the log dict"""
    if os.path.exists(os.path.join(loop, "final.jpg")):
        raise SystemExit("loop folder already has final.jpg; use a new loop folder (never overwrite)")
    os.makedirs(loop, exist_ok=True)
    cfg = merge(DEFAULT, cfg_dict if cfg_dict is not None else load_cfg(cfg_path))
    cache = cache or os.path.join(os.path.dirname(os.path.abspath(loop)), "_cache")
    I8, out, alpha, layers, log = build(src, cfg, cache, models_dir)
    a = argparse.Namespace(loop=loop)
    img, pl = place(out, alpha)
    log["placement"] = pl
    o8 = np.round(out).astype(np.uint8)
    log["alpha1_px_changed"] = int(((alpha == 1) & (o8 != I8).any(2)).sum())
    log["alpha0_px_not255"] = int(((alpha == 0) & (o8 != 255).any(2)).sum())
    log["config"] = cfg
    save_jpeg(img, os.path.join(a.loop, "final.jpg"), cfg["jpeg_q"])
    cv2.imwrite(os.path.join(a.loop, "work_full.png"), cv2.cvtColor(o8, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(a.loop, "work_alpha.png"), np.round(alpha * 255).astype(np.uint8))
    np.savez_compressed(os.path.join(a.loop, "work_layers.npz"), **layers)
    json.dump(log, open(os.path.join(a.loop, "log.json"), "w"), indent=1)
    if not quiet:
        print(json.dumps({k: v for k, v in log.items() if k not in ("config", "signature")}))
        print("signature:", json.dumps(log["signature"]))
    return log


def main(argv=None):
    ap = argparse.ArgumentParser(prog="whitebg cutout")
    ap.add_argument("src"); ap.add_argument("loop"); ap.add_argument("cfg", nargs="?")
    ap.add_argument("--cache", default=None); ap.add_argument("--models-dir", default=None)
    a = ap.parse_args(argv)
    run_one(a.src, a.loop, a.cfg, a.cache, a.models_dir)


if __name__ == "__main__":
    main()
