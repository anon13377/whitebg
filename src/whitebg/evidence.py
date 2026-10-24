"""Evidence + numeric checks for one loop folder.

usage: whitebg evidence SRC.jpg LOOP_DIR [--points "name:x:y" ...] [--hunt]

Writes LOOP_DIR/evidence/ (new files only):
  1_side.jpg      raw | final on grey
  2_overlay.jpg   magenta = removed, cyan = KEPT pixels that look like floor/shadow, red = REMOVED pixels that
                  look like item colour (possible missing part), yellow = added back
  3_crops_N.png   100 percent crops (shown at 200 percent): raw | result | overlay. Auto picks the worst red and
                  cyan clusters plus any --points. Positions printed in source AND final coordinates.
  4_hunt_N.png    (--hunt, loop 3 adversarial) the ENTIRE item outline tiled at 100 percent, raw | result
  checks.json     square, fill, border, bg purity, interior changed, item-colour kept pct, suspect clusters
A person (or reviewer) must LOOK at every image written here.
"""
import os, json, argparse
import numpy as np, cv2
from scipy import ndimage as ndi
from .common import load_rgb


def lbl(img, t, y=18, s=0.5):
    cv2.putText(img, t, (5, y), cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 0), 3)
    cv2.putText(img, t, (5, y), cv2.FONT_HERSHEY_SIMPLEX, s, (255, 0, 255), 1)


def to_final(pl, x, y):
    bx, by, _, _ = pl["bbox_src_xywh"]
    px, py, _, _ = pl["placed_xywh"]
    return int(round((x - bx) * pl["scale"] + px)), int(round((y - by) * pl["scale"] + py))


def clusters(mask, n=6, min_px=8):
    lab, k = ndi.label(mask, np.ones((3, 3), bool))
    if not k:
        return []
    sizes = ndi.sum(mask, lab, range(1, k + 1))
    objs = ndi.find_objects(lab)
    order = np.argsort(-sizes)
    out = []
    for i in order[:n]:
        if sizes[i] < min_px:
            break
        sl = objs[i]
        out.append({"px": int(sizes[i]), "cx": (sl[1].start + sl[1].stop) // 2, "cy": (sl[0].start + sl[0].stop) // 2,
                    "bbox": [sl[1].start, sl[0].start, sl[1].stop, sl[0].stop]})
    return out


def crop3(I, O, Ov, cx, cy, r=128):
    h, w = I.shape[:2]
    cx = int(min(max(cx, r), w - r)); cy = int(min(max(cy, r), h - r))
    sl = np.s_[cy - r:cy + r, cx - r:cx + r]
    return [x[sl].copy() for x in (I, O, Ov)], (cx, cy)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="whitebg evidence")
    ap.add_argument("src"); ap.add_argument("loop")
    ap.add_argument("--points", nargs="*", default=[])
    ap.add_argument("--hunt", action="store_true")
    a = ap.parse_args(argv)
    ev = os.path.join(a.loop, "evidence")
    os.makedirs(ev, exist_ok=True)
    I = cv2.cvtColor(load_rgb(a.src), cv2.COLOR_RGB2BGR)
    O = cv2.imread(os.path.join(a.loop, "work_full.png"))
    F = cv2.imread(os.path.join(a.loop, "final.jpg"))
    A = cv2.imread(os.path.join(a.loop, "work_alpha.png"), 0).astype(np.float32) / 255
    Ly = np.load(os.path.join(a.loop, "work_layers.npz"))
    log = json.load(open(os.path.join(a.loop, "log.json")))
    pl = log["placement"]
    h, w = I.shape[:2]
    itemlike, shadowlike, added = Ly["itemlike"], Ly["shadowlike"], Ly["added"]
    kept = A >= 0.5
    # suspects
    near = cv2.dilate(kept.astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool)
    lost_item = itemlike & ~kept & near
    lost_item = cv2.morphologyEx(lost_item.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)).astype(bool)
    edge = kept & ~cv2.erode(kept.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
    kept_shadow = shadowlike & kept & edge
    kept_shadow = cv2.morphologyEx(kept_shadow.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    # overlay
    r = (1 - A)[..., None]
    Ov = (I.astype(np.float32) * (1 - 0.6 * r) + np.array([255, 0, 255], np.float32) * 0.6 * r).astype(np.uint8)
    Ov[kept_shadow] = (255, 255, 0)
    Ov[lost_item] = (0, 0, 255)
    Ov[added] = (0, 255, 255)
    ch = {"frame_loop": a.loop}
    Fi = F.astype(np.int16)
    side = F.shape[0]
    ch["final_size"] = [F.shape[1], F.shape[0]]
    ch["square"] = F.shape[0] == F.shape[1]
    ch["fill_pct"] = pl["fill_pct"]
    ch["scale"] = pl["scale"]
    ch["touches_photo_edge_LTRB"] = pl.get("touches_photo_edge_LTRB")
    ch["border_min"] = int(min(Fi[0].min(), Fi[-1].min(), Fi[:, 0].min(), Fi[:, -1].min()))
    dark = (Fi < 245).any(2).astype(np.uint8)
    dist = cv2.distanceTransform(1 - dark, cv2.DIST_L2, 3)
    ch["final_bg_non255_gt4px_from_item"] = int(((Fi.min(2) < 255) & (dist > 4)).sum())
    ch["final_bg_below250_gt4px_from_item"] = int(((Fi.min(2) < 250) & (dist > 4)).sum())
    ch["alpha1_px_changed"] = log["alpha1_px_changed"]
    ch["alpha0_px_not255"] = log["alpha0_px_not255"]
    il = itemlike & near
    ch["itemlike_src_px"] = int(il.sum())
    ch["itemlike_kept_pct"] = round(100 * float((il & kept).sum()) / max(1, il.sum()), 3)
    ch["suspect_lost_item_px"] = int(lost_item.sum())
    ch["suspect_kept_shadow_px"] = int(kept_shadow.sum())
    lc = clusters(lost_item); sc = clusters(kept_shadow)
    for c_ in lc + sc:
        c_["final_xy"] = list(to_final(pl, c_["cx"], c_["cy"]))
    ch["suspect_lost_item_clusters"] = lc
    ch["suspect_kept_shadow_clusters"] = sc
    ch["bg_chroma_item_share"] = log.get("bg_chroma_item_share")
    # 1 side by side
    H = 900
    a1 = cv2.resize(I, (int(w * H / h), H), interpolation=cv2.INTER_AREA)
    a2 = cv2.resize(F, (H, H), interpolation=cv2.INTER_AREA)
    sb = np.full((H + 20, a1.shape[1] + a2.shape[1] + 60, 3), 128, np.uint8)
    sb[10:10 + H, 20:20 + a1.shape[1]] = a1
    sb[10:10 + H, 40 + a1.shape[1]:40 + a1.shape[1] + H] = a2
    cv2.imwrite(os.path.join(ev, "1_side.jpg"), sb, [cv2.IMWRITE_JPEG_QUALITY, 90])
    s = min(1.0, 1600 / max(h, w))
    cv2.imwrite(os.path.join(ev, "2_overlay.jpg"), cv2.resize(Ov, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 90])
    # 3 crops
    pts = []
    for c_ in lc[:4]:
        pts.append(("LOSTitem", c_["cx"], c_["cy"]))
    for c_ in sc[:4]:
        pts.append(("KEPTshadow", c_["cx"], c_["cy"]))
    for p in a.points:
        n_, x_, y_ = p.split(":")
        pts.append((n_, int(x_), int(y_)))
    if not pts:
        ys, xs = np.nonzero(edge)
        if len(xs):
            for q in (0.1, 0.35, 0.6, 0.85):
                i = int(q * (len(xs) - 1)); pts.append(("edge", int(xs[i]), int(ys[i])))
    rows = []; ch["crops"] = []
    for (nm, cx, cy) in pts:
        (c1, c2, c3), (cx2, cy2) = crop3(I, O, Ov, cx, cy)
        fx, fy = to_final(pl, cx2, cy2)
        t = f"{nm} src({cx2},{cy2}) fin({fx},{fy})"
        lbl(c1, t, s=0.42); lbl(c2, "result"); lbl(c3, "overlay")
        row = np.hstack([c1, np.full((256, 4, 3), 128, np.uint8), c2, np.full((256, 4, 3), 128, np.uint8), c3])
        row = cv2.resize(row, (row.shape[1] * 2, 512), interpolation=cv2.INTER_NEAREST)
        rows.append(row); ch["crops"].append(t)
    for k in range(0, len(rows), 4):
        cv2.imwrite(os.path.join(ev, f"3_crops_{k // 4 + 1}.png"), np.vstack(rows[k:k + 4]))
    # 4 hunt: whole outline at 100 percent
    if a.hunt:
        ring = kept & ~cv2.erode(kept.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        ring |= (A > 0) & (A < 1)
        T = 200
        tiles = []
        for ty in range(0, h, T):
            for tx in range(0, w, T):
                if ring[ty:ty + T, tx:tx + T].sum() < 30:
                    continue
                t1 = np.full((T, T, 3), 128, np.uint8); t2 = t1.copy()
                pi = I[ty:ty + T, tx:tx + T]; po = O[ty:ty + T, tx:tx + T]
                t1[:pi.shape[0], :pi.shape[1]] = pi; t2[:po.shape[0], :po.shape[1]] = po
                lbl(t1, f"{tx},{ty}", 14, 0.4)
                tiles.append(np.hstack([t1, np.full((T, 3, 3), 60, np.uint8), t2]))
        ch["hunt_tiles"] = len(tiles)
        per = 12
        for k in range(0, len(tiles), per):
            grp = tiles[k:k + per]
            while len(grp) % 3:
                grp.append(np.full_like(tiles[0], 128))
            g = np.vstack([np.hstack([x for t_ in grp[i:i + 3] for x in (t_, np.full((T, 10, 3), 40, np.uint8))]) for i in range(0, len(grp), 3)])
            cv2.imwrite(os.path.join(ev, f"4_hunt_{k // per + 1}.png"), g)
    json.dump(ch, open(os.path.join(ev, "checks.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in ch.items() if k not in ("suspect_lost_item_clusters", "suspect_kept_shadow_clusters", "crops")}))
    print("lost_item:", [(c_["px"], c_["cx"], c_["cy"]) for c_ in lc])
    print("kept_shadow:", [(c_["px"], c_["cx"], c_["cy"]) for c_ in sc])


if __name__ == "__main__":
    main()
