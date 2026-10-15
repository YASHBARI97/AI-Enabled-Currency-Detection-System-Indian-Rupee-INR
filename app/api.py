"""FastAPI service. Run: uvicorn app.api:app --reload"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.inference import history
from src.inference.decision import DISCLAIMER
from src.inference.predictor import ModelLoadError, Predictor
from src.utils.config import Settings
from src.utils.logging_setup import get_logger

settings = Settings.from_env()
log = get_logger("api", settings.log_level)
state: dict = {"predictor": None, "error": None}


def load_model():
    try:
        state["predictor"] = Predictor.load(settings.model_path, settings.model_meta, settings.max_upload_mb)
        state["error"] = None
    except ModelLoadError as e:
        state["predictor"], state["error"] = None, str(e)


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield


app = FastAPI(title="AI Enabled Currency Detection System", version="1.0.0", lifespan=lifespan)

# Only needed when the browser calls this API directly. The Next.js app proxies /api/* so it is
# same-origin by default. Set CURRENCY_CORS_ORIGINS="http://localhost:3000,https://your.site" to allow origins.
if settings.cors_origins:
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                       allow_methods=["GET", "POST"], allow_headers=["*"])

HTTP_FOR_STATUS = {"empty_input": 400, "unsupported_format": 415, "unreadable_image": 422,
                   "file_too_large": 413, "inference_error": 500}


@app.get("/health")
def health():
    pr = state["predictor"]
    if pr is None:
        return {"status": "model_unavailable", "model_version": None, "classes": [], "threshold": None,
                "max_upload_mb": settings.max_upload_mb}
    return {"status": "ok", "model_version": pr.version, "classes": pr.class_names,
            "threshold": pr.threshold, "max_upload_mb": settings.max_upload_mb}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    pr = state["predictor"]
    if pr is None:
        return JSONResponse({"status": "model_unavailable", "message": state["error"] or "Model not loaded."}, 503)
    data = await file.read(int(settings.max_upload_mb * 1024 * 1024) + 1)  # bounded read
    result = pr.predict_bytes(data, file.filename).to_dict()
    result["disclaimer"] = DISCLAIMER
    if settings.enable_history and result["inference_ms"] is not None:
        history.record(settings.history_path, result)
    code = HTTP_FOR_STATUS.get(result["status"], 200)
    return JSONResponse(result, status_code=code)
