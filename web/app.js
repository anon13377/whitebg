'use strict';
// whitebg PWA: photo -> mask (onnxruntime-web) -> pure #FFFFFF square, ~88% fill, no upscaling.
const MODELS = {
  u2netp: {url: 'models/u2netp.onnx', size: 320, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]},
};
const MAX_SIDE = 2000;   // eBay shows at most 1600 px; also bounds Safari canvas memory
const FILL = 0.88;
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

const sessions = {};   // name -> promise, so the model downloads once even if asked for twice
function getSession(name) {
  return sessions[name] ??= loadSession(name).catch(e => { delete sessions[name]; throw e; });
}
async function loadSession(name) {
  const t0 = performance.now();
  let buf, session;
  try {
    const res = await fetch(MODELS[name].url);
    if (!res.ok) throw new Error('model');
    buf = await res.arrayBuffer();
  } catch (_) { throw new Error('model'); }
  const t1 = performance.now();
  try {
    session = await ort.InferenceSession.create(buf, {executionProviders: ['wasm'], graphOptimizationLevel: 'all'});
  } catch (_) { throw new Error('model'); }
  return {session, loadMs: performance.now() - t0, fetchMs: t1 - t0, bytes: buf.byteLength};
}

function canvas(w, h) { const c = document.createElement('canvas'); c.width = w; c.height = h; return c; }

// small copy of the original, for hold to compare
function thumbOf(src, side = 1024) {
  const k = Math.min(1, side / Math.max(src.width, src.height));
  const c = canvas(Math.round(src.width * k), Math.round(src.height * k));
  c.getContext('2d').drawImage(src, 0, 0, c.width, c.height);
  return new Promise(res => c.toBlob(b => { c.width = 0; res(b); }, 'image/jpeg', 0.85));
}

async function loadImage(blob) {
  try { return await createImageBitmap(blob, {imageOrientation: 'from-image'}); }
  catch (_) {   // HEIC or odd formats: let Safari decode through an img element
    const img = new Image(), u = URL.createObjectURL(blob); img.src = u;
    try { await img.decode(); return img; } finally { URL.revokeObjectURL(u); }
  }
}

async function decode(blob, maxSide = MAX_SIDE) {
  const bmp = await loadImage(blob);
  const bw = bmp.naturalWidth || bmp.width, bh = bmp.naturalHeight || bmp.height;
  const k = Math.min(1, maxSide / Math.max(bw, bh));
  const w = Math.round(bw * k), h = Math.round(bh * k);
  const c = canvas(w, h), ctx = c.getContext('2d');
  ctx.fillStyle = '#FFFFFF'; ctx.fillRect(0, 0, w, h);   // transparent PNGs: white, not black
  ctx.drawImage(bmp, 0, 0, w, h); bmp.close && bmp.close();
  return c;
}

async function predictMask(src, name) {
  const m = MODELS[name], S = m.size;
  const {session, loadMs, fetchMs, bytes} = await getSession(name);
  const small = canvas(S, S), sctx = small.getContext('2d');
  sctx.imageSmoothingQuality = 'high'; sctx.drawImage(src, 0, 0, S, S);
  const px = sctx.getImageData(0, 0, S, S).data;
  // rembg normalisation: divide by max pixel value, then mean/std per channel
  let mx = 1; for (let i = 0; i < px.length; i += 4) mx = Math.max(mx, px[i], px[i + 1], px[i + 2]);
  const inp = new Float32Array(3 * S * S);
  for (let i = 0, p = 0; p < S * S; i += 4, p++)
    for (let ch = 0; ch < 3; ch++) inp[ch * S * S + p] = (px[i + ch] / mx - m.mean[ch]) / m.std[ch];
  const t0 = performance.now();
  const out = await session.run({[session.inputNames[0]]: new ort.Tensor('float32', inp, [1, 3, S, S])});
  const inferMs = performance.now() - t0;
  const d = out[session.outputNames[0]].data;
  let lo = Infinity, hi = -Infinity; for (let i = 0; i < S * S; i++) { lo = Math.min(lo, d[i]); hi = Math.max(hi, d[i]); }
  if (hi - lo < 0.1 || hi < 0.5) throw new Error('No item found in this photo.');   // flat prediction: blank photo
  const mc = canvas(S, S), mctx = mc.getContext('2d'), mi = mctx.createImageData(S, S);
  for (let i = 0; i < S * S; i++) { const v = Math.round(255 * (d[i] - lo) / ((hi - lo) || 1)); mi.data.set([v, v, v, 255], i * 4); }
  mctx.putImageData(mi, 0, 0); small.width = 0;
  // upscale mask to photo size (bilinear)
  const full = canvas(src.width, src.height), fctx = full.getContext('2d');
  fctx.imageSmoothingQuality = 'high'; fctx.drawImage(mc, 0, 0, src.width, src.height);
  const fd = fctx.getImageData(0, 0, src.width, src.height).data;
  full.width = 0; mc.width = 0;
  const alpha = new Uint8Array(src.width * src.height);
  for (let i = 0; i < alpha.length; i++) alpha[i] = fd[i * 4];
  return {alpha, inferMs, loadMs, fetchMs, bytes};
}

function compose(src, alpha) {
  const W = src.width, H = src.height;
  let x0 = W, y0 = H, x1 = -1, y1 = -1, n = 0;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if (alpha[y * W + x] > 127) {
    n++; if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
  if (x1 < 0) throw new Error('No item found in this photo.');
  const bw = x1 - x0 + 1, bh = y1 - y0 + 1;
  const side = Math.ceil(Math.max(bw, bh) / FILL);   // item at native resolution, never upscaled
  const sd = src.getContext('2d').getImageData(x0, y0, bw, bh);
  const p = sd.data;
  for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) {
    const a = alpha[(y + y0) * W + x + x0];
    if (a === 255) continue;                           // item pixel untouched
    const i = (y * bw + x) * 4, f = a / 255;
    p[i] = Math.round(p[i] * f + 255 * (1 - f));
    p[i + 1] = Math.round(p[i + 1] * f + 255 * (1 - f));
    p[i + 2] = Math.round(p[i + 2] * f + 255 * (1 - f));
    p[i + 3] = 255;
  }
  const out = canvas(side, side), octx = out.getContext('2d');
  octx.fillStyle = '#FFFFFF'; octx.fillRect(0, 0, side, side);
  octx.putImageData(sd, Math.floor((side - bw) / 2), Math.floor((side - bh) / 2));
  out.itemShare = n / (W * H);
  return out;
}

async function processBlob(blob, name, maxSide) {
  const t0 = performance.now();
  const src = await decode(blob, maxSide);
  const t1 = performance.now();
  let r, out;
  try {
    r = await predictMask(src, name);
    out = compose(src, r.alpha);
  } catch (e) { src.width = 0; throw e; }
  const t2 = performance.now();
  const jpeg = await new Promise(res => out.toBlob(res, 'image/jpeg', 0.92));
  const thumb = await thumbOf(src);
  const t3 = performance.now();
  const outSide = out.width, itemShare = out.itemShare, width = src.width, height = src.height;
  src.width = 0; out.width = 0;   // free Safari canvas memory now
  return {jpeg, thumb, itemShare, alpha: r.alpha, width, height, outSide,
    timings: {decodeMs: t1 - t0, maskMs: t2 - t1, inferMs: r.inferMs, modelLoadMs: r.loadMs, modelFetchMs: r.fetchMs,
      composeEncodeMs: t3 - t2, totalMs: t3 - t0}, modelBytes: r.bytes};
}
window.whitebg = {processBlob, MODELS, getSession};

// UI: a queue of photos, processed one after another in the order picked
const $ = id => document.getElementById(id);
const stage = $('stage');
const IOS = /iP(hone|ad|od)/.test(navigator.userAgent) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);

// finished photos are kept on the phone (IndexedDB) so they survive iOS closing the app
let dbp = null;
function db() {
  return dbp ??= new Promise((ok, no) => {
    const q = indexedDB.open('whitebg', 1);
    q.onupgradeneeded = () => q.result.createObjectStore('photos', {keyPath: 'id'});
    q.onsuccess = () => ok(q.result); q.onerror = () => no(q.error);
  });
}
function store(fn) {
  return db().then(d => new Promise(ok => {
    const t = d.transaction('photos', 'readwrite'); fn(t.objectStore('photos')); t.oncomplete = t.onerror = ok;
  })).catch(() => {});
}
const keep = it => store(s => s.put({id: it.id, jpeg: it.jpeg, thumb: it.thumb, flag: it.flag}));
const drop = it => store(s => s.delete(it.id));
function restore() {
  return db().then(d => new Promise(ok => {
    const q = d.transaction('photos').objectStore('photos').getAll();
    q.onsuccess = () => ok(q.result || []); q.onerror = () => ok([]);
  })).catch(() => []);
}
const items = [];
let busy = false, active = null, picked = false, sharing = false, chunk = Infinity, saveFrom = 0, dlNext = 0, avgMs = 1500, ready = false, nextId = 0;

function say(text, alert) {
  const s = $('status'); s.textContent = text; s.className = alert ? 'alert' : '';
  s.setAttribute('aria-live', alert ? 'assertive' : 'polite');
}
const pad = n => String(n).padStart(2, '0');
const fileOf = it => new File([it.jpeg], 'whitebg_' + pad(it.n) + '.jpg', {type: 'image/jpeg'});
const doneItems = () => items.filter(it => it.state === 'done');
const pending = () => items.filter(it => it.state === 'queued' || it.state === 'working').length;
const LABEL = {queued: 'waiting', working: 'working', done: 'ready', error: 'needs a new photo'};

function render() {
  const strip = $('strip');
  [...strip.children].forEach(b => { if (!items.some(it => it.tile === b)) b.remove(); });
  items.forEach((it, i) => {
    let b = it.tile;
    if (!b) {
      b = it.tile = document.createElement('button');
      b.onclick = () => { active = it; picked = true; render(); show(); };
    }
    if (strip.children[i] !== b) strip.insertBefore(b, strip.children[i] || null);
    b.className = 'tile ' + it.state + (it === active ? ' on' : '');
    b.setAttribute('aria-label', 'Photo ' + it.n + ', ' + LABEL[it.state]);
    b.setAttribute('aria-current', it === active ? 'true' : 'false');
    const bg = it.outUrl || it.srcUrl, css = bg ? `url("${bg}")` : '';
    if (b.dataset.bg !== css) { b.style.backgroundImage = css; b.dataset.bg = css; }
    const html = `<span class="num">${pad(it.n)}</span>` +
      (it.state === 'working' ? '<span class="spin"></span>' : it.state === 'error' ? '<span class="bad">!</span>' : '');
    if (b.dataset.html !== html) { b.innerHTML = html; b.dataset.html = html; }
  });
  if (active && active.tile) active.tile.scrollIntoView({block: 'nearest', inline: 'nearest'});
  $('stripwrap').hidden = !items.length;
  const d = doneItems().length;
  $('count').textContent = items.length ? `${d} of ${items.length} ready` : '';
  const b = $('saveall');
  if (saveFrom >= d) { saveFrom = 0; chunk = Infinity; }
  if (!b.classList.contains('ok')) b.textContent = chunk < d
    ? `Save photos ${saveFrom + 1} to ${Math.min(d, saveFrom + chunk)} of ${d}`
    : d > 1 ? `Save all ${d} to Photos` : 'Save to Photos';
  if (dlNext >= d) dlNext = 0;
  $('dl').textContent = IOS && d > 1 ? `Download photo ${dlNext + 1} of ${d}` : 'Download copies instead';
  $('saveall').disabled = !d || sharing;
  $('pickdock').hidden = !!items.length;
  $('resultdock').hidden = !items.length;
  $('bar').style.width = items.length ? (100 * (items.length - pending()) / items.length) + '%' : '0';
  $('bar').hidden = !pending();
  const left = pending();
  if (left && !ready) say('Downloading setup files, first time only. Keep this page open.');
  else if (left) say(`Working on ${items.length - left + 1} of ${items.length}, about ${Math.max(1, Math.round(left * avgMs / 1000))} s left`);
}

function show() {
  const it = active;
  const held = stage.classList.contains('orig') && it && it.state === 'done';
  stage.className = 'stage ' + (it ? it.state : 'empty') + (held ? ' orig' : '');
  if (!held) stage.setAttribute('aria-pressed', 'false');
  $('outimg').hidden = !(it && it.outUrl);
  $('srcimg').hidden = !(it && it.srcUrl);
  $('empty').hidden = !!it;
  if (it) $('outimg').alt = `Photo ${it.n} on a white background`;
  if (it && it.outUrl && $('outimg').getAttribute('src') !== it.outUrl) $('outimg').src = it.outUrl;
  if (it && it.srcUrl && $('srcimg').getAttribute('src') !== it.srcUrl) $('srcimg').src = it.srcUrl;
  $('hint').hidden = !(it && it.state === 'done');
  $('remove').disabled = !(it && it.state !== 'working');
  if (pending()) return;
  if (it && it.state === 'error') say(it.msg, true);
  else if (it && it.state === 'done')
    say(it.flag ? 'Needs a look: check the edges before you save.' : 'Done. Zoom in on the edges, then save.');
}

function add(files) {
  for (const f of files) items.push({id: ++nextId, n: items.length + 1, file: f, state: 'queued'});
  if (!active) active = items[0];
  render(); show(); run();
}

async function run() {
  if (busy) return; busy = true;
  let it;
  while ((it = items.find(x => x.state === 'queued'))) {
    it.state = 'working'; it.srcUrl = URL.createObjectURL(it.file);
    if (!picked && !stage.classList.contains('orig')) active = it;
    render(); show();
    await new Promise(r => setTimeout(r, 30));   // let the screen paint between photos
    const t0 = performance.now();
    try {
      const r = await whitebg.processBlob(it.file, 'u2netp');
      if (!items.includes(it)) continue;   // removed while it was working
      it.jpeg = r.jpeg; it.outUrl = URL.createObjectURL(r.jpeg);
      URL.revokeObjectURL(it.srcUrl); it.thumb = r.thumb; it.srcUrl = URL.createObjectURL(r.thumb);
      it.flag = r.itemShare > 0.9 || r.itemShare < 0.02;
      it.state = 'done'; it.file = null; ready = true; keep(it);
      avgMs = avgMs * 0.6 + (performance.now() - t0) * 0.4;
    } catch (err) {
      if (!items.includes(it)) continue;
      it.state = 'error'; it.code = err.message;
      it.msg = err.message === 'model'
        ? 'Could not download the setup files. It will try again by itself when you are back online.'
        : /No item/.test(err.message)
          ? 'Could not find the item. Try a plain background, more light, and fill most of the frame.'
          : 'This photo could not be opened. Try another photo or take a new one.';
    }
    render(); show();
  }
  busy = false;
  render(); show();
}

function forget(it) { drop(it); it.outUrl && URL.revokeObjectURL(it.outUrl); it.srcUrl && URL.revokeObjectURL(it.srcUrl); it.outUrl = it.srcUrl = null; }

function clearAll() {
  items.forEach(forget); store(s => s.clear()); saveFrom = 0; chunk = Infinity; dlNext = 0; items.length = 0; active = null; picked = false;
  render(); show(); say(ready ? 'Ready. Take or upload a photo.' : 'Getting ready...');
}

['cam', 'lib', 'cam2', 'lib2'].forEach(id => $(id).addEventListener('change', e => {
  const fs = [...e.target.files]; e.target.value = ''; if (fs.length) add(fs);
}));
$('clear').onclick = () => {
  if (items.length && !confirm(`Remove all ${items.length} photos and start over?`)) return;
  clearAll();
};
$('remove').onclick = () => {
  const it = active; if (!it || it.state === 'working') return;
  if (it.state === 'done' && !confirm(`Remove photo ${it.n}?`)) return;
  const at = items.indexOf(it);
  forget(it); items.splice(at, 1); items.forEach((x, i) => x.n = i + 1);
  if (!items.length) return clearAll();
  active = items[Math.min(at, items.length - 1)]; render(); show();
};
// the share sheet must open right inside the tap, with every file built already
$('saveall').onclick = () => {
  const all = doneItems(), files = all.slice(saveFrom, saveFrom + chunk).map(fileOf);
  if (!files.length || sharing) return;
  if (navigator.canShare && navigator.canShare({files})) {
    sharing = true; render();
    navigator.share({files}).then(() => {
      if (saveFrom + chunk < all.length) { saveFrom += chunk; say('Saved. Tap Save for the next group.'); }
      else { saveFrom = 0; chunk = Infinity; flash('Saved'); }
    })
      .catch(e => {
        if (e.name === 'AbortError') say('Not saved. Tap Save again when you are ready.');
        else if (e.name === 'NotAllowedError' && files.length > 10) {
          chunk = 10; saveFrom = 0; say('Too many photos at once for this phone. Tap Save to save them 10 at a time.');
        }
        else if (IOS) say('Saving to Photos did not work. Use Download below, one photo per tap.');
        else { say('Saving to Photos did not work, so the photos were downloaded instead.'); downloadAll(files); }
      })
      .finally(() => { sharing = false; render(); });
  } else if (IOS) say('Use Download below to save one photo per tap.');
  else { say('Downloading your photos.'); downloadAll(files); }
};
// iOS only takes one download per tap, so step through the photos
$('dl').onclick = () => {
  const all = doneItems(); if (!all.length) return;
  if (!IOS) return downloadAll(all.map(fileOf));
  downloadAll([fileOf(all[dlNext % all.length])]); dlNext++; render();
};
function downloadAll(files) {
  files.forEach((f, i) => setTimeout(() => {
    const a = document.createElement('a'), u = URL.createObjectURL(f);
    a.href = u; a.download = f.name; a.click(); setTimeout(() => URL.revokeObjectURL(u), 4000);
  }, i * 300));
}
function flash(t) {
  const b = $('saveall'); b.classList.add('ok'); b.textContent = '\u2713 ' + t;
  setTimeout(() => { b.classList.remove('ok'); render(); }, 1200);
}
// press and hold the picture to see the original
const hold = on => {
  const show = on && active && active.state === 'done';
  stage.classList.toggle('orig', !!show);
  stage.setAttribute('aria-pressed', show ? 'true' : 'false');
};
stage.addEventListener('pointerdown', () => { if (active && active.state === 'done') picked = true; hold(true); });
['pointerup', 'pointerleave', 'pointercancel'].forEach(ev => stage.addEventListener(ev, () => hold(false)));
stage.addEventListener('contextmenu', e => e.preventDefault());
stage.addEventListener('keydown', e => { if ((e.key === ' ' || e.key === 'Enter') && !e.repeat) { e.preventDefault(); hold(true); } });
stage.addEventListener('keyup', e => { if (e.key === ' ' || e.key === 'Enter') hold(false); });

// back online: retry photos that failed only because the setup files did not download
window.addEventListener('online', () => {
  const again = items.filter(it => it.state === 'error' && it.code === 'model');
  again.forEach(it => it.state = 'queued');
  if (again.length) { render(); run(); }
});

// warm the model while the page is idle so the first photo is quick
say('Getting ready...');
setTimeout(() => getSession('u2netp')
  .then(() => { ready = true; if (!items.length) say('Ready. Take or upload a photo.'); })
  .catch(() => { if (!items.length) say('Connect to the internet once to finish setup.', true); }), 50);
render(); show();
restore().then(saved => {
  if (!saved.length || items.length) return;
  saved.sort((a, b) => a.id - b.id).forEach(p => {
    nextId = Math.max(nextId, p.id);
    items.push({id: p.id, n: items.length + 1, state: 'done', jpeg: p.jpeg, thumb: p.thumb, flag: p.flag,
      outUrl: URL.createObjectURL(p.jpeg), srcUrl: URL.createObjectURL(p.thumb)});
  });
  active = items[items.length - 1]; ready = true; render(); show();
  say(`Your ${items.length} photo${items.length > 1 ? 's are' : ' is'} still here. Save them, or tap Start over.`);
});
if ('serviceWorker' in navigator) {
  const hadController = !!navigator.serviceWorker.controller;
  navigator.serviceWorker.register('sw.js');
  navigator.serviceWorker.addEventListener('controllerchange', () => {
    if (hadController && !items.length) say('App updated. Close and reopen to use the new version.');
  });
}
