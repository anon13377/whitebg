// Offline cache. Bump VERSION when any shell file changes.
const VERSION = 'whitebg-v5';
const ORT = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
// small files: all must arrive for the install to finish
const SHELL = ['./', 'index.html', 'app.js', 'manifest.webmanifest',
  'icons/icon-192.png', 'icons/icon-512.png', 'icons/apple-touch-icon.png'];
// big files: each cached on its own, so one dropped download on bad signal does not undo the rest
const HEAVY = ['models/u2netp.onnx', ORT + 'ort.min.js', ORT + 'ort-wasm-simd-threaded.mjs', ORT + 'ort-wasm-simd-threaded.wasm'];
self.addEventListener('install', e => {
  e.waitUntil(caches.open(VERSION).then(async c => {
    await c.addAll(SHELL.map(u => new Request(u, {cache: 'reload'})));
    await Promise.all(HEAVY.map(u => c.match(u).then(hit => hit || fetch(u).then(r => r.ok && c.put(u, r))).catch(() => {})));
  }).then(() => self.skipWaiting()));
});
self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== VERSION).map(k => caches.delete(k))))
    .then(() => self.clients.claim()));
});
// Cache first; anything fetched once is stored for offline use.
self.addEventListener('fetch', e => {
  if (e.request.method !== 'GET') return;
  e.respondWith(caches.match(e.request, {ignoreSearch: true}).then(hit => hit || fetch(e.request).then(r => {
    if (r.ok && (r.type === 'basic' || r.type === 'cors')) {
      const copy = r.clone(); caches.open(VERSION).then(c => c.put(e.request, copy));
    }
    return r;
  })));
});
