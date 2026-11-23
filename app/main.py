"""CIFAKE-Guard FastAPI backend.

Run:   uvicorn app.main:app --port 8000          (then open http://localhost:8000)
Env:   CIFAKE_RUN_DIR      run directory with checkpoints/ and results/ (default runs/cpu_experiment_robust_ensemble)
       CIFAKE_MAX_UPLOAD_MB upload limit (default 10)
"""
from __future__ import annotations

import os
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from cifake_guard import __version__
from cifake_guard.inference import InvalidImageError, Predictor, load_image_bytes
from cifake_guard.report import render_html_report
from cifake_guard.utils import get_logger

log = get_logger("api")
ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/gif", "image/tiff"}
MAX_REPORTS = 64


def _run_dir() -> Path:
    p = Path(os.environ.get("CIFAKE_RUN_DIR", "runs/cpu_experiment_robust_ensemble"))
    return p if p.is_absolute() else ROOT / p


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.reports = OrderedDict()
    app.state.predictor = None
    app.state.load_error = None
    try:
        app.state.predictor = Predictor(_run_dir())
    except Exception as e:                       # start anyway; /api/health reports the problem
        log.exception("Could not load models from %s", _run_dir())
        app.state.load_error = f"{type(e).__name__}: {e}"
    yield


app = FastAPI(title="CIFAKE-Guard", version=__version__, lifespan=lifespan,
              description="Detection of AI-generated images with calibrated ensembles and explanations.")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    log.info("%s %s -> %d (%.0f ms)", request.method, request.url.path, response.status_code,
             (time.perf_counter() - t0) * 1000)
    return response


def _predictor(request: Request) -> Predictor:
    pred = request.app.state.predictor
    if pred is None:
        raise HTTPException(503, f"Models are not loaded: {request.app.state.load_error}. "
                                 "Train and evaluate first (see README) or set CIFAKE_RUN_DIR.")
    return pred


@app.get("/api/health")
def health(request: Request):
    pred = request.app.state.predictor
    return {"status": "ok" if pred else "degraded", "version": __version__, "run_dir": str(_run_dir()),
            "models_loaded": list(pred.models) if pred else [], "error": request.app.state.load_error}


@app.get("/api/models")
def models(request: Request):
    pred = _predictor(request)
    return {"ensemble": pred.ensemble.to_dict(), "models": pred.model_info()}


@app.get("/api/results")
def results(request: Request):
    """Evaluation artefacts of the loaded run (metrics, robustness, unseen generators) for the dashboard."""
    import json

    import pandas as pd

    def records(df, orient):          # pandas' JSON writer turns NaN into null (plain json would fail)
        return json.loads(df.round(5).to_json(orient=orient))

    res = _run_dir() / "results"
    out = {}
    for key, fname in [("test_metrics", "test_metrics.csv"), ("dedup_metrics", "test_metrics_dedup.csv")]:
        if (res / fname).exists():
            out[key] = records(pd.read_csv(res / fname, index_col=0), "index")
    rob = res / "robustness.csv"
    if rob.exists():
        out["robustness"] = records(pd.read_csv(rob), "records")
    out["unseen"] = {}
    for f in sorted(res.glob("unseen_*.csv")):
        out["unseen"][f.stem.removeprefix("unseen_")] = records(pd.read_csv(f, index_col=0), "index")
    out["plots"] = sorted(p.name for p in (_run_dir() / "plots").glob("*.png"))
    return out


@app.get("/api/plots/{name}")
def plot(name: str):
    path = (_run_dir() / "plots" / name).resolve()
    if path.parent != (_run_dir() / "plots").resolve() or path.suffix != ".png" or not path.exists():
        raise HTTPException(404, "plot not found")
    return FileResponse(path, media_type="image/png")


@app.post("/api/predict")
async def predict(request: Request, file: UploadFile = File(...),
                  explain: bool = Query(True, description="compute explanation heatmaps")):
    pred = _predictor(request)
    max_mb = float(os.environ.get("CIFAKE_MAX_UPLOAD_MB", "10"))
    if file.content_type and file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, f"Unsupported content type {file.content_type!r}; upload JPEG, PNG, WebP, BMP, GIF or TIFF")
    data = await file.read(int(max_mb * 1024 * 1024) + 1)
    if len(data) > max_mb * 1024 * 1024:
        raise HTTPException(413, f"File larger than {max_mb:g} MB")
    try:
        image = load_image_bytes(data)
    except InvalidImageError as e:
        raise HTTPException(400, str(e)) from e
    result = await run_in_threadpool(pred.predict, image, "all" if explain else None)
    result["filename"] = file.filename
    result["report_id"] = rid = uuid.uuid4().hex
    reports: OrderedDict = request.app.state.reports
    reports[rid] = result
    while len(reports) > MAX_REPORTS:
        reports.popitem(last=False)
    return JSONResponse(result)


@app.get("/api/report/{report_id}")
def report(request: Request, report_id: str, format: str = Query("html", pattern="^(html|json)$")):
    result = request.app.state.reports.get(report_id)
    if result is None:
        raise HTTPException(404, "Report not found or expired (reports are kept in memory)")
    pred = _predictor(request)
    if format == "json":
        slim = {k: v for k, v in result.items() if k not in ("explanations", "visuals")}
        return JSONResponse(slim, headers={"Content-Disposition": f'attachment; filename="cifake_report_{report_id[:8]}.json"'})
    html = render_html_report(result, pred.model_info())
    return HTMLResponse(html, headers={"Content-Disposition": f'attachment; filename="cifake_report_{report_id[:8]}.html"'})


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


app.mount("/static", StaticFiles(directory=STATIC), name="static")
