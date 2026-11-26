# palm-line-detector

Deterministic computer-vision palm-line detection, wrapped as an HTTP service.

Vendors the CV pipeline from [yeonsumia/palmistry](https://github.com/yeonsumia/palmistry)
(MediaPipe hand-landmark rectification → U-Net line segmentation → k-means line
classification) behind a small FastAPI service, for PalmInsight.

**Scope:** detects and traces the Heart, Head, and Life lines only, with a
`long`/`short` length classification per line (derived from fixed landmark
thresholds, same as upstream). There is no Fate line support and no rich
pattern vocabulary — that's intentional; the LLM in the main backend still
writes the descriptive pattern text and full narrative report, grounded in
these deterministic facts.

## API

`GET /health` → `{"ok": true}`

`POST /detect`
```json
{ "image": "data:image/jpeg;base64,..." }
```

Success:
```json
{
  "ok": true,
  "lines": {
    "heart": { "points": [{"x":0..1,"y":0..1}, ...], "lengthClass": "long" | "short" },
    "head":  { "points": [...], "lengthClass": "long" | "short" },
    "life":  { "points": [...], "lengthClass": "long" | "short" }
  }
}
```

Failure (no exception, still HTTP 200):
```json
{ "ok": false, "reason": "no_hand_detected" | "invalid_image" | "classification_failed" | "lines_not_found" }
```

Same image in → byte-identical output out (verified). Point order/direction per
line matches upstream (not guaranteed to be tip-to-base consistently across
lines — treat as a polyline, not a directed path).

## Local dev

```
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
MODEL_PATH=$(pwd)/checkpoint/checkpoint_aug_epoch70.pth .venv/Scripts/python -m uvicorn app.main:app --reload
```

## Notes on the vendored code

- `app/vendor/*.py` is copied near-verbatim from upstream's `code/`, minus
  `tools.py`'s HEIC conversion and matplotlib result-drawing (unneeded for an
  API).
- One compatibility fix applied: modern `scikit-image`'s `skeletonize()`
  returns a bool array; upstream's code (written against an older skimage)
  feeds it straight into `cv2.cvtColor`, which rejects bool. Wrapped to cast
  back to uint8 0/255.
- `mediapipe` is pinned to `0.10.14` — versions ≥1.0 dropped the legacy
  `mediapipe.solutions.hands` API this pipeline depends on.
