# P2 PWA spike report (2026-10-24)

## What was built (web/)
- `index.html`: one screen. "Take or pick a photo" (input accept=image/* capture=environment), model picker, result, Share / Save to Photos (Web Share API with files, falls back to download), Download JPEG. Apple meta tags: apple-mobile-web-app-capable, status bar style default, title, apple-touch-icon.
- `app.js`: decode with EXIF orientation (cap 3000 px long side), rembg identical preprocessing, onnxruntime-web 1.20.1 (pinned, jsDelivr, wasm EP, 1 thread unless cross-origin isolated), min-max normalised mask, bilinear upscale, compose on pure #FFFFFF square at 88 percent fill of the longer item side, never upscaled (canvas side = ceil(max(bw,bh)/0.88)). Pixels with alpha 255 are copied untouched; only partial alpha pixels are blended toward white. Exposes `window.whitebg.processBlob(blob, model)` for testing.
- `manifest.webmanifest` (standalone, scope ./, icons 192 + 512 maskable), `sw.js` (cache first offline: shell, icons, u2netp, ort.min.js; the ort wasm and the optional isnet model are cached the first time they are fetched).
- `icons/`: icon-192, icon-512, apple-touch-icon (180), PIL generated, white square with a green item silhouette.
- `models/u2netp.onnx`: rembg release v0.0.0 file (github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx), 4,574,861 bytes. U2-Net is Apache 2.0 per its repo; verify before release.
- `models/isnet-general-use-int8.onnx`: 46,360,717 bytes, made locally by onnxruntime dynamic UInt8 quantization of the 178.6 MB isnet-general-use. License must be verified before publishing; `.gitignore` excludes it and spike_results/.
- Note: the local model cache is `<local model cache>`, not `~/.u2net` (that folder does not exist). Local models: birefnet-general, birefnet-massive (973 MB each), isnet-general-use (179 MB), u2net (176 MB). silueta was not tested.

## How it was measured
`spike_results/measure.py`: local `python -m http.server` on 127.0.0.1:8765 plus its OWN headless Chrome (system chrome.exe launched by Playwright with a fresh temp profile; Playwright bundled browsers are not installed), closed at the end. Single thread wasm (no COOP/COEP headers, which GitHub Pages cannot set either, so this matches a Pages deployment). Host is the Windows Server VPS CPU, not an iPhone. Raw numbers in `spike_results/measurements.json`; outputs `*_u2netp.jpg`, `*_isnet.jpg`, `*_alpha.png`, `sheet_dvd_sample.jpg`.

Inputs: sample item raw `album/raw/sample photo` (2048x1536) and vetting raw DVD case sample (`2026-10-19_mixed3/raw/sample photo`, 1536x2048).
Ground truth: desktop approved alphas in raw coordinates, `album/white/finish/work/run/alpha_009.npy` and `whitebg_vetting/movies/dvd_sample/final_v2/work_alpha.png`. IoU at alpha > 0.5.

## Measurements
| | u2netp | isnet int8 |
|---|---|---|
| Model download | 4.6 MB | 46.4 MB |
| Session create (localhost fetch + compile) | 0.30 s | 0.41 s |
| Inference, warm | 0.87 to 0.89 s (320x320) | 4.52 to 4.54 s (1024x1024) |
| Total per photo, warm (decode + mask + compose + JPEG) | 0.96 to 1.0 s | 4.6 to 4.7 s |
| First photo, cold (includes session create) | 1.38 s | 5.31 s |
| JS heap after run | 45 to 85 MB | about 185 MB |
| IoU sample item | 0.978 | 0.995 |
| IoU DVD case sample | 0.727 | 0.674 |

Plus the onnxruntime wasm runtime (about 10 MB, fetched once from jsDelivr, then cached). Output corners are pure 255,255,255.

Quantization check (Python, DVD case sample): isnet fp32 IoU 0.678 vs int8 0.677, so int8 costs nothing measurable; the miss is the model, not the quantization.

DVD case sample failure: both small models segment the salient subject (the faces on the printed cover) instead of the whole DVD case (see `sheet_dvd_sample.jpg`). Offline test of a "solid item" post process (threshold soft mask at 40/255, fill holes, keep largest component): u2netp rises to 0.983 on DVD case sample but sample item drops to 0.952; isnet does not benefit (0.70) and its sample item IoU falls to 0.645 (the item is in more than one piece). So it must be a user toggle, not a default.

## What an iPhone can realistically handle (estimates, NOT measured on a device)
- Safari kills a tab somewhere around 1 to 1.5 GB on recent iPhones, lower on older ones, and caps canvas area (about 16.7 MP per canvas plus a total canvas budget). The 3000 px decode cap keeps each canvas under 9 MP; a full 12 MP photo is about 48 MB RGBA per copy, so the cap matters.
- wasm in Safari: single threaded on Pages, no guaranteed WebGPU on older iOS. Expect per photo roughly 1x to 3x the VPS figure: u2netp about 1 to 3 s, isnet int8 about 5 to 15 s with about 200 to 400 MB peak. Both should fit memory; isnet is slow but usable for a few photos.
- Storage: 4.6 MB model caches easily; 46 MB is fine, but iOS can evict caches of PWAs unused for weeks, so a first run after eviction redownloads.
- birefnet (973 MB) is not feasible on a phone, as the handoff predicted.

## Quality, honestly vs desktop
- Desktop uses birefnet plus isnet and u2net second opinions plus carving and add back logic, verified on 27 finals. The PWA uses one small model with no repair.
- Simple high contrast item on plain background (sample item): close to desktop (u2netp 0.978, isnet 0.995).
- Flat printed items (DVD cases, boxes, books): default small model output is not listable (0.67 to 0.73), it cuts out the cover art subject.
- u2netp's 320 px mask upscaled gives soft edges on fine detail; isnet at 1024 is sharper.

## Recommendation for PWA scope (P6)
1. Default model u2netp (4.6 MB, about 1 s, offline after first visit). Offer isnet int8 as an optional "Better edges" download only after its license is confirmed.
2. Add a "Solid box or case" toggle (low threshold, fill holes, largest component), default off; suggest it when the mask covers much less than its convex hull.
3. Keep the review concept simple on the phone: result over a tinted background with Keep / Redo / Solid toggle; no numbered spot carving in v1.
4. State in the UI that phone results are lower quality than the desktop app and must be checked before listing.
5. Before P6: the owner tests on a real iPhone (local network or Pages preview) to replace the estimates with real timings and confirm Add to Home Screen, share sheet to Photos, and offline relaunch.
