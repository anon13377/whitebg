# Models

whitebg downloads model weights on first use (about 1.2 GB) into `%LOCALAPPDATA%\whitebg\models`
(override with `--models-dir` or the `WHITEBG_MODELS` environment variable). Weights are never stored in this repo, with one exception: the small u2netp model (4.6 MB) ships in web/models/ for the iPhone web app, with its license in web/models/LICENSE-U2Net.txt.
URLs, sizes and SHA-256 hashes are in `src/whitebg/models.json`; a file is used only if its hash matches.

| Model | Role | Size | Where the ONNX file comes from | License |
|---|---|---|---|---|
| birefnet-general | base matte (cuts holes and slots) | 928 MB | rembg GitHub release v0.0.0 | TODO: UNVERIFIED. Check the BiRefNet upstream repo before the first public release. |
| isnet-general-use | second opinion | 170 MB | rembg GitHub release v0.0.0 | TODO: UNVERIFIED. Check the IS-Net (DIS) upstream repo before the first public release. |
| u2net | second opinion | 168 MB | rembg GitHub release v0.0.0 | Apache 2.0, verified 2026-10-24 at https://github.com/xuebinqin/U-2-Net/blob/master/LICENSE |
| u2netp (web app) | iPhone web app default | 4.6 MB | rembg GitHub release v0.0.0, shipped in web/models/ | Apache 2.0 (same U-2-Net repo), verified 2026-10-24; license copy in web/models/LICENSE-U2Net.txt |

birefnet-general and isnet-general-use licenses are UNVERIFIED. Do not describe them as verified until each one is checked at its original
source and the link is recorded in this file. Terms for downloading from the rembg release mirror are also UNVERIFIED.
