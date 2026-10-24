"""Build a review page: grey page, raw left, final right, optional zoom images under each row.
usage: whitebg review OUT.html ITEMS.json
ITEMS.json: [{"title": "...", "raw": "path", "final": "path", "note": "...", "zooms": ["path", ...]}, ...]
Paths are written relative to OUT.html. Refuses to overwrite. P1 stub: the interactive review UI comes in P5."""
import sys, os, json, html


def main(argv=None):
    argv = [None] + list(sys.argv[1:] if argv is None else argv)
    out, items = argv[1], json.load(open(argv[2]))
    if os.path.exists(out):
        raise SystemExit("refusing to overwrite " + out)
    base = os.path.dirname(os.path.abspath(out))
    rel = lambda p: os.path.relpath(os.path.abspath(p), base).replace("\\", "/")
    rows = []
    for it in items:
        z = "".join(f'<div class="cell"><img loading="lazy" src="{html.escape(rel(p))}"></div>' for p in it.get("zooms", []))
        rows.append(f'''<h2>{html.escape(it["title"])}</h2><p class="note">{html.escape(it.get("note", ""))}</p>
    <div class="row"><div class="cell"><img loading="lazy" src="{html.escape(rel(it["raw"]))}"><div class="cap">Raw</div></div>
    <div class="cell"><img loading="lazy" src="{html.escape(rel(it["final"]))}"><div class="cap">Final</div></div></div>{('<div class="zoom">' + z + '</div>') if z else ''}''')
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>White Background Review</title><style>
    :root{{--bg:#d9d9d9;--card:#f4f4f4;--ink:#1d1d1f;--muted:#555}}
    body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,Segoe UI,Arial,sans-serif}}
    main{{max-width:1200px;margin:0 auto;padding:20px 16px 60px}}h1{{font-size:22px;margin:0 0 4px}}h2{{font-size:18px;margin:30px 0 4px}}
    p.note{{color:var(--muted);margin:0 0 12px}}.row{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:12px}}
    .cell{{background:var(--card);padding:10px;border-radius:8px}}.cell img{{width:100%;display:block;outline:1px solid #999}}
    .cap{{font-size:13px;margin-top:6px}}.zoom{{display:grid;grid-template-columns:1fr;gap:12px}}
    @media(max-width:700px){{.row{{grid-template-columns:1fr}}}}</style></head><body><main>
    <h1>White background review</h1><p class="note">Grey page on purpose so you can see where each white photo ends. Raw on the left, final on the right.</p>
    {"".join(rows)}</main></body></html>'''
    open(out, "w", encoding="utf-8").write(page)
    print("wrote", out)


if __name__ == "__main__":
    main()
