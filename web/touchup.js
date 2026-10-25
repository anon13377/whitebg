'use strict';
// Touch up: circle leftover background (or paint with Erase / Keep), then recompose on pure white.
// Edits change the mask only; item pixels are never painted, and compose() reframes the item at 88%.
(() => {
  const $ = id => document.getElementById(id);
  const W = window.whitebg;
  const ed = $('editor'), wrap = $('ewrap'), can = $('ecan'), ov = $('eov');
  let it = null, src = null, px = null, eff = null, w = 0, h = 0, ctx = null;
  let tool = 'circle', undo = [], changes = 0, opening = false, saving = false, token = 0, back = null, s = 1, fit = 1, tx = 0, ty = 0;
  let stroke = null, pinch = null;
  const UNDO_MAX = 6;   // each step is a full mask copy (about 3 MB), so keep the iPhone budget small
  const BRUSH = {S: 22, M: 44, L: 80};   // brush diameter in screen points, same size at any zoom
  let brush = BRUSH.M;

  async function open() {
    if (opening || !ed.hidden) return;
    it = window.lotshot.active;
    if (!it || it.state !== 'done') return;
    if (!it.src || !it.mask) return window.lotshot.say('This photo was made with an older version. Remove it and add it again to touch it up.', true);
    opening = true; back = document.activeElement; const my = ++token;
    ed.hidden = false; document.body.classList.add('editing'); $('app').inert = true;
    $('ehelp').textContent = 'Opening...';
    try {
      const img = await createImageBitmap(it.src);
      if (my !== token) return img.close && img.close();   // closed while opening
      w = img.width; h = img.height;
      src = W.canvas(w, h); src.getContext('2d').drawImage(img, 0, 0); img.close && img.close();
      px = src.getContext('2d').getImageData(0, 0, w, h).data;
      const m = await W.pngToMask(it.mask, w, h);
      if (my !== token) return;
      eff = m;
      can.width = w; can.height = h; ctx = can.getContext('2d');
      undo = []; changes = 0; setTool('circle'); sizeOv(); fitView(); paint(0, 0, w, h); updUndo();
      $('ecancel').focus();
    } catch (_) {
      if (my !== token) return;
      close(); window.lotshot.say('Could not open this photo for touch up. Try again.', true);
    } finally { opening = false; }
  }

  function close() {
    token++; stroke = null; pinch = null; saving = false;
    $('ehelp').classList.remove('warn');
    ed.hidden = true; document.body.classList.remove('editing'); $('app').inert = false;
    if (back && back.isConnected) back.focus(); back = null;
    if (src) src.width = 0; can.width = can.height = 1;
    src = px = eff = null; undo = []; it = null; clearOv(); window.lotshot.render();
  }

  // composite of the photo on white using the current mask, for the visible rect only
  function paint(x0, y0, x1, y1) {
    x0 = Math.max(0, Math.floor(x0)); y0 = Math.max(0, Math.floor(y0));
    x1 = Math.min(w, Math.ceil(x1)); y1 = Math.min(h, Math.ceil(y1));
    const rw = x1 - x0, rh = y1 - y0; if (rw <= 0 || rh <= 0) return;
    const im = ctx.createImageData(rw, rh), d = im.data;
    for (let y = 0; y < rh; y++) for (let x = 0; x < rw; x++) {
      const p = (y + y0) * w + x + x0, i = p * 4, o = (y * rw + x) * 4, f = eff[p] / 255;
      d[o] = px[i] * f + 255 * (1 - f); d[o + 1] = px[i + 1] * f + 255 * (1 - f); d[o + 2] = px[i + 2] * f + 255 * (1 - f); d[o + 3] = 255;
    }
    ctx.putImageData(im, x0, y0);
  }

  function fitView() {
    const r = wrap.getBoundingClientRect();
    fit = s = Math.min(r.width / w, r.height / h);
    tx = (r.width - w * s) / 2; ty = (r.height - h * s) / 2; apply();
  }
  function apply() {
    can.style.width = w + 'px'; can.style.height = h + 'px';
    can.style.transform = `translate(${tx}px,${ty}px) scale(${s})`;
  }
  function toImg(cx, cy) { const r = wrap.getBoundingClientRect(); return [(cx - r.left - tx) / s, (cy - r.top - ty) / s]; }
  function toScr(x, y) { return [x * s + tx, y * s + ty]; }

  function pushUndo() { undo.push(eff.slice()); changes++; if (undo.length > UNDO_MAX) undo.shift(); updUndo(); }
  function updUndo() { $('eundo').disabled = !undo.length; }

  function setTool(t) {
    tool = t;
    ['circle', 'erase', 'keep'].forEach(k => $('t-' + k).setAttribute('aria-pressed', k === t ? 'true' : 'false'));
    $('esizes').hidden = t === 'circle'; $('ehelp').classList.remove('warn');
    $('ehelp').textContent = t === 'circle' ? 'Drag a finger around the leftover background. Pinch to zoom.'
      : t === 'erase' ? 'Paint over leftover background to turn it white. Pinch to zoom.'
      : 'Paint over part of the item that went missing to bring it back. Pinch to zoom.';
  }

  // overlay: the circle path while drawing, or the brush ring
  function clearOv() { const c = ov.getContext('2d'); c.clearRect(0, 0, ov.width, ov.height); }
  function sizeOv() { const r = wrap.getBoundingClientRect(), k = devicePixelRatio || 1; ov.width = Math.round(r.width * k); ov.height = Math.round(r.height * k); ov.getContext('2d').setTransform(k, 0, 0, k, 0, 0); }
  function drawPath(pts) {
    clearOv(); const c = ov.getContext('2d'); if (pts.length < 2) return;
    c.beginPath(); pts.forEach(([x, y], i) => { const [a, b] = toScr(x, y); i ? c.lineTo(a, b) : c.moveTo(a, b); });
    c.closePath(); c.fillStyle = 'rgba(255,45,140,.18)'; c.fill();
    c.lineWidth = 3; c.strokeStyle = '#ff2d8c'; c.setLineDash([8, 6]); c.stroke();
  }

  function stamp(x, y, val) {
    const r = brush / 2 / s, r2 = r * r;
    const x0 = Math.max(0, Math.floor(x - r)), x1 = Math.min(w - 1, Math.ceil(x + r));
    const y0 = Math.max(0, Math.floor(y - r)), y1 = Math.min(h - 1, Math.ceil(y + r));
    for (let yy = y0; yy <= y1; yy++) for (let xx = x0; xx <= x1; xx++)
      if ((xx - x) * (xx - x) + (yy - y) * (yy - y) <= r2) eff[yy * w + xx] = val;
    return [x0, y0, x1 + 1, y1 + 1];
  }
  function brushTo(x, y) {
    const val = tool === 'erase' ? 0 : 255, [lx, ly] = stroke.last || [x, y];
    const step = Math.max(1, brush / 4 / s), n = Math.ceil(Math.hypot(x - lx, y - ly) / step);
    let bx0 = w, by0 = h, bx1 = 0, by1 = 0;
    for (let i = 0; i <= n; i++) {
      const t = n ? i / n : 1, [a, b, c, d] = stamp(lx + (x - lx) * t, ly + (y - ly) * t, val);
      bx0 = Math.min(bx0, a); by0 = Math.min(by0, b); bx1 = Math.max(bx1, c); by1 = Math.max(by1, d);
    }
    stroke.last = [x, y]; paint(bx0, by0, bx1, by1);
  }

  // everything inside the circled shape becomes background
  function fillLasso(pts) {
    if (pts.length < 3) return false;
    let x0 = w, y0 = h, x1 = 0, y1 = 0;
    pts.forEach(([x, y]) => { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); });
    x0 = Math.max(0, Math.floor(x0)); y0 = Math.max(0, Math.floor(y0)); x1 = Math.min(w, Math.ceil(x1)); y1 = Math.min(h, Math.ceil(y1));
    const bw = x1 - x0, bh = y1 - y0;
    if (bw < 2 || bh < 2) { help('Draw on the photo itself, around the leftover.'); return false; }
    if (Math.max(bw, bh) * s < 12) { help('Keep your finger down and draw a loop around the leftover.'); return false; }
    const m = W.canvas(bw, bh), mc = m.getContext('2d');
    mc.beginPath(); pts.forEach(([x, y], i) => i ? mc.lineTo(x - x0, y - y0) : mc.moveTo(x - x0, y - y0));
    mc.closePath(); mc.fillStyle = '#000'; mc.fill();
    const md = mc.getImageData(0, 0, bw, bh).data; m.width = 0;
    pushUndo();
    let solid = 0;
    for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++)
      if (md[(y * bw + x) * 4 + 3] > 127) { const p = (y + y0) * w + x + x0; if (eff[p] > 200) solid++; eff[p] = 0; }
    paint(x0, y0, x1, y1); flash();
    // a big solid area inside the circle is usually part of the item, not leftover background
    if (solid > Math.max(400, w * h * 0.002)) help('Removed. If that took part of your item, tap Undo.', true);
    else help('Removed. Circle more, or tap Done.');
    return true;
  }
  function help(text, warn) { const e = $('ehelp'); e.textContent = text; e.classList.toggle('warn', !!warn); }
  function flash() { can.classList.remove('flash'); void can.offsetWidth; can.classList.add('flash'); }

  // one finger draws, two fingers pinch and pan
  function start1(cx, cy) {
    const p = toImg(cx, cy);
    if (tool === 'circle') stroke = {pts: [p]};
    else { pushUndo(); stroke = {pts: [p]}; brushTo(...p); }
  }
  function move1(cx, cy) {
    if (!stroke) return; const p = toImg(cx, cy);
    if (tool === 'circle') { stroke.pts.push(p); drawPath(stroke.pts); } else brushTo(...p);
  }
  function end1() {
    if (!stroke) return;
    if (tool === 'circle') {
      if (stroke.pts.length < 3) help('Keep your finger down and draw a loop around the leftover.');
      else fillLasso(stroke.pts);
      clearOv();
    }
    stroke = null;
  }
  function cancelStroke() {
    if (stroke && tool !== 'circle' && undo.length) { eff = undo.pop(); changes--; paint(0, 0, w, h); updUndo(); }
    stroke = null; clearOv();
  }
  const mid = (a, b) => [(a.clientX + b.clientX) / 2, (a.clientY + b.clientY) / 2];
  const dist = (a, b) => Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
  wrap.addEventListener('touchstart', e => {
    e.preventDefault();
    if (e.touches.length === 1) start1(e.touches[0].clientX, e.touches[0].clientY);
    else if (e.touches.length === 2) {
      cancelStroke();
      const r = wrap.getBoundingClientRect(), [mx, my] = mid(e.touches[0], e.touches[1]);
      pinch = {d: dist(e.touches[0], e.touches[1]), s: s, p: [(mx - r.left - tx) / s, (my - r.top - ty) / s]};
    }
  }, {passive: false});
  wrap.addEventListener('touchmove', e => {
    e.preventDefault();
    if (pinch && e.touches.length >= 2) {
      const r = wrap.getBoundingClientRect(), [mx, my] = mid(e.touches[0], e.touches[1]);
      s = Math.min(fit * 8, Math.max(fit, pinch.s * dist(e.touches[0], e.touches[1]) / pinch.d));
      tx = mx - r.left - pinch.p[0] * s; ty = my - r.top - pinch.p[1] * s; apply();
    } else if (!pinch && e.touches.length === 1) move1(e.touches[0].clientX, e.touches[0].clientY);
  }, {passive: false});
  wrap.addEventListener('touchend', e => {
    e.preventDefault();
    if (pinch) { if (!e.touches.length) pinch = null; return; }
    if (!e.touches.length) end1();
  }, {passive: false});
  wrap.addEventListener('touchcancel', () => { pinch = null; cancelStroke(); });
  // mouse on a computer: drag draws, wheel zooms
  wrap.addEventListener('pointerdown', e => { if (e.pointerType === 'mouse') { wrap.setPointerCapture(e.pointerId); start1(e.clientX, e.clientY); } });
  wrap.addEventListener('pointermove', e => { if (e.pointerType === 'mouse') move1(e.clientX, e.clientY); });
  wrap.addEventListener('pointerup', e => { if (e.pointerType === 'mouse') end1(); });
  wrap.addEventListener('wheel', e => {
    e.preventDefault(); const r = wrap.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    const p = [(mx - tx) / s, (my - ty) / s];
    s = Math.min(fit * 8, Math.max(fit, s * (e.deltaY < 0 ? 1.15 : 1 / 1.15))); tx = mx - p[0] * s; ty = my - p[1] * s; apply();
  }, {passive: false});

  $('touch').addEventListener('click', e => { e.stopPropagation(); open(); });
  ['circle', 'erase', 'keep'].forEach(k => $('t-' + k).onclick = () => setTool(k));
  [['S', 'sz-s'], ['M', 'sz-m'], ['L', 'sz-l']].forEach(([k, id]) => $(id).onclick = () => {
    brush = BRUSH[k]; ['sz-s', 'sz-m', 'sz-l'].forEach(x => $(x).setAttribute('aria-pressed', x === id ? 'true' : 'false'));
  });
  $('eundo').onclick = () => { if (!undo.length) return; eff = undo.pop(); changes--; paint(0, 0, w, h); updUndo(); setTool(tool); };
  $('efit').onclick = fitView;
  function cancel() { if (saving || opening) return; if (changes > 0 && !confirm('Throw away these touch ups?')) return; close(); }
  $('ecancel').onclick = cancel;
  ed.addEventListener('keydown', e => { if (e.key === 'Escape') { e.preventDefault(); cancel(); } });
  $('edone').onclick = async () => {
    if (saving) return;
    if (changes <= 0) { close(); return window.lotshot.say('No changes made.'); }
    saving = true; $('edone').disabled = true; $('ecancel').disabled = true; help('Saving...');
    const target = it, sw = w, sh = h, mask0 = eff;
    try {
      const out = W.compose(src, mask0), share = out.itemShare;
      const jpeg = await new Promise(res => out.toBlob(res, 'image/jpeg', 0.95)); out.width = 0;
      const mask = await W.maskToPng(mask0, sw, sh);
      close();
      window.lotshot.setResult(target, jpeg, mask, share);
      window.lotshot.say('Touch up saved.');
    } catch (err) {
      saving = false;
      help(/No item/.test(err.message) ? 'That removed the whole item. Tap Undo.' : 'Could not save. Try again.', true);
    } finally { $('edone').disabled = false; $('ecancel').disabled = false; }
  };
  window.addEventListener('resize', () => { if (!ed.hidden && src) { sizeOv(); if (s === fit) fitView(); else { const r = wrap.getBoundingClientRect(); fit = Math.min(r.width / w, r.height / h); s = Math.max(s, fit); apply(); } } });
  new ResizeObserver(() => { if (!ed.hidden) sizeOv(); }).observe(wrap);
})();
