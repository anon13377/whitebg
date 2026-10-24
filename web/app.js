'use strict';
// whitebg PWA spike: one photo -> mask (onnxruntime-web) -> pure #FFFFFF square, ~88% fill, no upscaling.
const MODELS = {
  u2netp: {url: 'models/u2netp.onnx', size: 320, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]},
  isnet: {url: 'models/isnet-general-use-int8.onnx', size: 1024, mean: [0.5, 0.5, 0.5], std: [1, 1, 1]},
};
const MAX_SIDE = 3000;   // cap decoded photo to bound Safari canvas memory
const FILL = 0.88;
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

const sessions = {};
async function getSession(name) {
  if (sessions[name]) return {session: sessions[name], loadMs: 0, bytes: 0};
  const t0 = performance.now();
  const buf = await (await fetch(MODELS[name].url)).arrayBuffer();
  const t1 = performance.now();
  const session = await ort.InferenceSession.create(buf, {executionProviders: ['wasm'], graphOptimizationLevel: 'all'});
  sessions[name] = session;
  return {session, loadMs: performance.now() - t0, fetchMs: t1 - t0, bytes: buf.byteLength};
}

function canvas(w, h) { const c = document.createElement('canvas'); c.width = w; c.height = h; return c; }

async function decode(blob) {
  const bmp = await createImageBitmap(blob, {imageOrientation: 'from-image'});
  const k = Math.min(1, MAX_SIDE / Math.max(bmp.width, bmp.height));
  const w = Math.round(bmp.width * k), h = Math.round(bmp.height * k);
  const c = canvas(w, h); c.getContext('2d').drawImage(bmp, 0, 0, w, h); bmp.close && bmp.close();
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
  const mc = canvas(S, S), mctx = mc.getContext('2d'), mi = mctx.createImageData(S, S);
  for (let i = 0; i < S * S; i++) { const v = Math.round(255 * (d[i] - lo) / ((hi - lo) || 1)); mi.data.set([v, v, v, 255], i * 4); }
  mctx.putImageData(mi, 0, 0);
  // upscale mask to photo size (bilinear)
  const full = canvas(src.width, src.height), fctx = full.getContext('2d');
  fctx.imageSmoothingQuality = 'high'; fctx.drawImage(mc, 0, 0, src.width, src.height);
  const fd = fctx.getImageData(0, 0, src.width, src.height).data;
  const alpha = new Uint8Array(src.width * src.height);
  for (let i = 0; i < alpha.length; i++) alpha[i] = fd[i * 4];
  return {alpha, inferMs, loadMs, fetchMs, bytes};
}

function compose(src, alpha) {
  const W = src.width, H = src.height;
  let x0 = W, y0 = H, x1 = -1, y1 = -1;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if (alpha[y * W + x] > 127) {
    if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
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
  return out;
}

async function processBlob(blob, name) {
  const t0 = performance.now();
  const src = await decode(blob);
  const t1 = performance.now();
  const r = await predictMask(src, name);
  const t2 = performance.now();
  const out = compose(src, r.alpha);
  const jpeg = await new Promise(res => out.toBlob(res, 'image/jpeg', 0.92));
  const t3 = performance.now();
  return {jpeg, alpha: r.alpha, width: src.width, height: src.height, outSide: out.width,
    timings: {decodeMs: t1 - t0, maskMs: t2 - t1, inferMs: r.inferMs, modelLoadMs: r.loadMs, modelFetchMs: r.fetchMs,
      composeEncodeMs: t3 - t2, totalMs: t3 - t0}, modelBytes: r.bytes};
}
window.whitebg = {processBlob, MODELS};

// UI
const $ = id => document.getElementById(id);
let lastBlob = null, lastName = 'whitebg.jpg';
$('pick').addEventListener('change', async e => {
  const f = e.target.files[0]; if (!f) return;
  $('status').textContent = 'Working... first run downloads the model.';
  $('result').hidden = true;
  try {
    const r = await processBlob(f, $('model').value);
    lastBlob = r.jpeg; lastName = (f.name || 'photo').replace(/\.[^.]+$/, '') + '_white.jpg';
    $('outimg').src = URL.createObjectURL(r.jpeg);
    $('result').hidden = false;
    $('status').textContent = `Done in ${(r.timings.totalMs / 1000).toFixed(1)} s (${r.outSide} x ${r.outSide}).`;
  } catch (err) { $('status').textContent = 'Error: ' + err.message; }
  e.target.value = '';
});
$('dl').addEventListener('click', () => {
  const a = document.createElement('a'); a.href = URL.createObjectURL(lastBlob); a.download = lastName; a.click();
});
$('share').addEventListener('click', async () => {
  const file = new File([lastBlob], lastName, {type: 'image/jpeg'});
  if (navigator.canShare && navigator.canShare({files: [file]})) {
    try { await navigator.share({files: [file]}); } catch (_) {}
  } else $('dl').click();
});
if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js');
