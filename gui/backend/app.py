"""LungGuard API + static frontend.  Run:  python app.py"""
import io, json, os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from inference import WEIGHTS, Engine

ROOT = Path(__file__).resolve().parent
FRONT = ROOT.parent / "frontend"
MAX_BYTES = 10 * 1024 * 1024
engine = Engine()


@asynccontextmanager
async def lifespan(app):
    engine.load()
    print(f"[lungguard] device={engine.device} loaded={list(engine.models)} missing={engine.missing}")
    yield


app = FastAPI(title="LungGuard API", version="1.0", lifespan=lifespan)


@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response = await call_next(request)
    ct = response.headers.get("content-type", "")
    if "text/html" in ct or "javascript" in ct or "css" in ct:
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


@app.get("/api/health")
def health():
    return engine.info()


@app.get("/api/metrics")
def metrics():
    return json.loads((ROOT / "data" / "metrics.json").read_text())


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), model: str = Form("ensemble"), explain: bool = Form(True)):
    if not engine.ready:
        raise HTTPException(503, "No model weights found. Put resnet50.pth, vgg19.pth, efficientnet_b1.pth and "
                                 "mobilenet_v2.pth in a 'models' folder or set LUNGGUARD_MODELS.")
    if model != "ensemble" and model not in WEIGHTS:
        raise HTTPException(400, f"Unknown model '{model}'.")
    if model != "ensemble" and model not in engine.models:
        raise HTTPException(400, f"Weights for '{model}' are not loaded.")
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image is larger than 10 MB.")
    try:
        img = Image.open(io.BytesIO(data)); img.load()
    except (UnidentifiedImageError, OSError, Exception):
        raise HTTPException(415, "That file could not be read as an image. Use JPG, PNG, BMP or WebP.")
    if min(img.size) < 32:
        raise HTTPException(422, "Image is too small to analyse.")
    return await run_in_threadpool(engine.predict, img, model, explain)  # nothing is written to disk


# Clean URL routes
@app.get("/")
def route_index():
    return FileResponse(FRONT / "index.html")

@app.get("/index.html")
def route_index_html():
    return FileResponse(FRONT / "index.html")

@app.get("/analyze")
def route_analyze():
    return FileResponse(FRONT / "analyze.html")

@app.get("/analyze.html")
def route_analyze_html():
    return FileResponse(FRONT / "analyze.html")

@app.get("/models")
def route_models():
    return FileResponse(FRONT / "models.html")

@app.get("/models.html")
def route_models_html():
    return FileResponse(FRONT / "models.html")

@app.get("/results")
def route_results():
    return FileResponse(FRONT / "results.html")

@app.get("/results.html")
def route_results_html():
    return FileResponse(FRONT / "results.html")

@app.get("/method")
def route_method():
    return FileResponse(FRONT / "method.html")

@app.get("/method.html")
def route_method_html():
    return FileResponse(FRONT / "method.html")


app.mount("/", StaticFiles(directory=FRONT, html=True), name="site")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"\n  LungGuard running at http://localhost:{port}\n")
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=port)
