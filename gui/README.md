# LungGuard Web GUI

Chest X-ray classifier (COVID-19 / Normal / Pneumonia) with a FastAPI backend that runs your four PyTorch
models as a weighted ensemble, paired with an editorial web UI featuring Grad-CAM heatmaps.

## Quick start

1. Place the four `.pth` model weights (`resnet50.pth`, `vgg19.pth`, `efficientnet_b1.pth`, `mobilenet_v2.pth`)
   into the root `models/` folder (or `cli/models/`):

   The backend automatically searches `models/`, `../cli/models/`, `../cli/`, and `gui/backend/models/`.
   To use another location: `LUNGGUARD_MODELS=/path/to/weights python backend/app.py`

2. Run it (Linux, macOS, or Windows):

   ```bash
   python run.py
   ```

   or from the repository root:

   ```bash
   python run_gui.py
   ```

   or manually:

   ```bash
   cd backend
   pip install -r requirements.txt
   python app.py
   ```

3. Open <http://localhost:8000>. Interactive API docs are at <http://localhost:8000/docs>.

## Layout

```
run.py              Cross-platform launcher script
backend/
  app.py            FastAPI app: /api/predict, /api/health, /api/metrics + serves the frontend
  inference.py      model loading, soft-voting ensemble (0.5/0.3/0.1/0.1), Grad-CAM heatmap
  data/metrics.json test-set results shown on the Results / Models pages
frontend/           plain HTML + CSS + JS (no build step)
  assets/logo.svg   logo; favicon.ico / apple-touch-icon.png generated from it
```

## API

`POST /api/predict` (multipart): `file` (image, max 10 MB), `model` (`ensemble` default, or `resnet50`,
`vgg19`, `efficientnet_b1`, `mobilenet_v2`), `explain` (`true` returns a Grad-CAM heatmap).

```bash
curl -F "file=@xray.jpg" http://localhost:8000/api/predict
```

Uploads are processed in memory and never written to disk.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `LUNGGUARD_MODELS` | auto | Folder holding the `.pth` files |
| `PORT` / `HOST` | 8000 / 127.0.0.1 | Server address (use `HOST=0.0.0.0` to expose on your network) |

## Troubleshooting

- **Missing model weights**: run `python download_models.py` from the repository root.
- **Slow first prediction on CPU**: VGG19 is large; expect a second or two per image on CPU.
- Models use the class order `COVID19, NORMAL, PNEUMONIA` (alphabetical, as in `ImageFolder`).

Research project only. Not a medical device.
