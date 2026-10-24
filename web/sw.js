// Offline cache. Bump VERSION when any shell file changes.
const VERSION = 'whitebg-v1';
const ORT = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
const SHELL = ['./', 'index.html', 'app.js', 'manifest.webmanifest',
  'icons/icon-192.png', 'icons/icon-512.png', 'icons/apple-touch-icon.png',
  'models/u2netp.onnx', ORT + 'ort.min.js'];
self.addEventListener('install', e => {
  e.waitUntil(caches.open(VERSION).then(c => c.addAll(SHELL)).then(() => self.skipWaiting()));
});
self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== VERSION).map(k => caches.delete(k))))
    .then(() => self.clients.claim()));
});
// Cache first; anything fetched once (ort wasm, optional isnet model) is stored for offline use.
self.addEventListener('fetch', e => {
  if (e.request.method !== 'GET') return;
  e.respondWith(caches.match(e.request).then(hit => hit || fetch(e.request).then(r => {
    if (r.ok && (r.type === 'basic' || r.type === 'cors')) {
      const copy = r.clone(); caches.open(VERSION).then(c => c.put(e.request, copy));
    }
    return r;
  })));
});
