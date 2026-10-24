# whitebg

Turn product photos into clean, pure white (#FFFFFF) square photos for eBay, Etsy, Poshmark and other listings,
without cutting off any part of the item.

## For sellers (no coding needed)

What it does:
1. You give it your item photos.
2. It removes the floor or table behind the item and puts the item on pure white.
3. It centers the item on a square photo with a little white space around it.
4. Your original photos are never changed or deleted. Results go into a new folder.

Please look at every result before you list it. Tricky photos (white, grey or black items on a plain floor,
clear plastic on a dark floor) can need a fix by eye. A review screen that points out risky spots is coming.

The easy Windows installer and the iPhone web version are not ready yet. For now it runs from a command window:

```
whitebg run "C:\path\to\my photos" -o "C:\path\to\white results"
```

The first run downloads the AI models (about 1.2 GB, one time only).

Already shot on white? `whitebg audit-white photo.jpg photo_square.jpg` only trims and squares the photo, no cutout.

## For developers

```
python -m pip install -e .[dev]
whitebg models                    # where the models are and whether they are present
whitebg models --download         # fetch them (resumable, SHA-256 checked)
whitebg run PHOTOS... -o OUT      # loop 1 cutout + evidence images + final copy
whitebg review OUT.html ITEMS.json   # static raw vs final page (stub; real review UI later)
whitebg selftest --raw-dir R --final-dir F [--names a b]
python -m pytest tests            # fast unit tests, no models or photos needed
python tests/golden_iou.py tests/golden_local.json   # golden IoU harness on your own verified finals
```

- Models run on onnxruntime directly. Preprocessing replicates rembg 2.0.86 exactly, so masks match the pipeline
  this was tuned on. Already have a rembg cache? `--models-dir %USERPROFILE%\.rembg\models` reuses it (no download).
- Local selftest paths can go in `whitebg_local.json` (git ignored):
  `{"selftest": {"raw_dir": "...", "final_dir": "...", "names": ["img_005", "img_009"]}}`
- Golden manifest format: `tests/golden_manifest.example.json`. Your own `tests/golden_local.json` is git ignored.
- Never overwrite: every loop writes a new folder and refuses to replace an existing `final.jpg`.
- Advanced commands keep the original script arguments: `whitebg cutout`, `evidence`, `batch`, `register`.
- Model licenses: see MODELS.md (UNVERIFIED in this version).

License: MIT for the code in this repo. Each model keeps its own license.
