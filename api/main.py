"""
main.py
=======
FastAPI application — REST API for the Customer Churn Prediction System.

Endpoints
---------
GET  /health          Liveness + readiness probe
GET  /model-meta      Metadata about the currently loaded model
POST /predict         Score a single customer
POST /batch-predict   Score up to 1 000 customers per request

Design decisions
----------------
- ChurnPredictor is loaded once at startup via FastAPI's lifespan context
  and stored in app.state — safe for concurrent async requests.
- All inputs are validated by Pydantic schemas (api/schemas.py).
  Invalid payloads return HTTP 422 with field-level error details.
- All unhandled exceptions are caught by a global exception handler and
  returned as structured ErrorResponse JSON (HTTP 500).
- CORS is enabled for all origins — restrict in production.

Running locally
---------------
    uvicorn api.main:app --reload --port 8000

Swagger UI : http://localhost:8000/docs
ReDoc      : http://localhost:8000/redoc
"""

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import (
    BatchPredictItem,
    BatchPredictRequest,
    BatchPredictResponse,
    CustomerFeatures,
    ErrorResponse,
    HealthResponse,
    ModelMetaResponse,
    PredictionResponse,
)
from src.predict import ChurnPredictor

logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths — override via environment variables in production
# ---------------------------------------------------------------------------

MODEL_PATH    = Path("models/best_model.pkl")
METADATA_PATH = Path("models/model_metadata.json")

# ---------------------------------------------------------------------------
# Application startup / shutdown  (lifespan)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Load the ChurnPredictor once when the server starts, store it in
    app.state so all request handlers can share the same instance, and
    release resources when the server shuts down.

    Raises
    ------
    RuntimeError : If the model artefacts are missing (run training first).
    """
    logger.info("Loading model artefacts from '%s' …", MODEL_PATH)

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at '{MODEL_PATH}'. "
            "Run  python src/run_training.py  to train and save the model."
        )
    if not METADATA_PATH.exists():
        raise RuntimeError(
            f"Metadata not found at '{METADATA_PATH}'. "
            "Run  python src/run_training.py  to generate model metadata."
        )

    predictor = ChurnPredictor.load(
        model_path    = MODEL_PATH,
        metadata_path = METADATA_PATH,
    )

    # Load metadata for /model-meta endpoint
    with open(METADATA_PATH, "r") as fh:
        metadata = json.load(fh)

    app.state.predictor      = predictor
    app.state.metadata       = metadata
    app.state.start_time     = time.time()
    app.state.model_run_id   = metadata.get("best_model", {}).get("run_id", "unknown")

    logger.info("Model '%s' loaded and ready.", app.state.model_run_id)

    yield   # ── server is running ──

    logger.info("Shutting down — releasing model resources.")
    app.state.predictor  = None
    app.state.metadata   = None


# ---------------------------------------------------------------------------
# FastAPI app instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "Customer Churn Prediction API",
    description = (
        "Real-time and batch churn scoring powered by XGBoost + SMOTE.\n\n"
        "Train the model first:  `python src/run_training.py`\n\n"
        "Then start the API:     `uvicorn api.main:app --reload --port 8000`"
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# ── CORS ────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],    # restrict to specific domains in production
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ---------------------------------------------------------------------------
# Global exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all handler for unhandled server-side exceptions.
    Returns HTTP 500 with a structured ErrorResponse body.
    """
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
        content     = ErrorResponse(
            error       = "InternalServerError",
            detail      = f"An unexpected error occurred: {str(exc)}",
            status_code = 500,
        ).model_dump(),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    Handler for explicitly raised HTTPExceptions.
    Returns a structured ErrorResponse body.
    """
    return JSONResponse(
        status_code = exc.status_code,
        content     = ErrorResponse(
            error       = _http_error_label(exc.status_code),
            detail      = str(exc.detail),
            status_code = exc.status_code,
        ).model_dump(),
    )


def _http_error_label(status_code: int) -> str:
    """Map an HTTP status code to a short camelCase error label."""
    labels = {
        400: "BadRequest",
        401: "Unauthorized",
        403: "Forbidden",
        404: "NotFound",
        422: "ValidationError",
        503: "ServiceUnavailable",
    }
    return labels.get(status_code, f"HttpError{status_code}")


# ---------------------------------------------------------------------------
# Dependency helper
# ---------------------------------------------------------------------------

def _get_predictor(request: Request) -> ChurnPredictor:
    """
    Retrieve the shared ChurnPredictor from app.state.

    Raises HTTP 503 if the model failed to load during startup.
    """
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = (
                "Prediction model is not available. "
                "Check startup logs and ensure training artefacts exist."
            ),
        )
    return predictor


# ---------------------------------------------------------------------------
# Utility — convert Pydantic schema → raw dict for predict_single()
# ---------------------------------------------------------------------------

def _features_to_dict(features: CustomerFeatures) -> dict:
    """
    Convert a CustomerFeatures Pydantic model to a plain dict, stripping
    None values for optional fields so the feature engineering layer
    applies its own defaults.
    """
    raw = features.model_dump()
    # Remove optional fields that are None so imputation fires correctly
    return {k: v for k, v in raw.items() if v is not None or k in (
        "customer_id",  # always pass through even if None
    )}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model = HealthResponse,
    summary        = "Liveness and readiness probe",
    tags           = ["Operations"],
    responses      = {503: {"model": ErrorResponse}},
)
async def health(request: Request) -> HealthResponse:
    """
    Returns HTTP 200 when the API is running and the model is loaded.

    Returns HTTP 503 when the model failed to load at startup.

    Use this endpoint for:
    - Kubernetes liveness / readiness probes
    - Load balancer health checks
    """
    predictor    = getattr(request.app.state, "predictor", None)
    model_loaded = predictor is not None
    run_id       = getattr(request.app.state, "model_run_id", "unknown")
    uptime       = time.time() - getattr(request.app.state, "start_time", time.time())

    if not model_loaded:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = "Model is not loaded.",
        )

    return HealthResponse(
        status         = "ok",
        model_loaded   = model_loaded,
        model_run_id   = run_id,
        uptime_seconds = round(uptime, 2),
    )


@app.get(
    "/model-meta",
    response_model = ModelMetaResponse,
    summary        = "Retrieve metadata about the loaded model",
    tags           = ["Operations"],
    responses      = {503: {"model": ErrorResponse}},
)
async def model_meta(request: Request) -> ModelMetaResponse:
    """
    Returns full metadata for the currently loaded model, including:
    - Model name and variant (e.g. xgboost_smote)
    - Training timestamp
    - Cross-validation and test-set metrics (AUC, Precision, Recall, F1)
    - Feature names used by the model
    - Comparison table for all 6 trained variants
    """
    _get_predictor(request)   # 503 if model not loaded

    meta = getattr(request.app.state, "metadata", {})
    best = meta.get("best_model", {})

    return ModelMetaResponse(
        run_id          = best.get("run_id", "unknown"),
        model_name      = best.get("model_name", "unknown"),
        smote           = best.get("smote", False),
        trained_at      = meta.get("trained_at", "unknown"),
        cv_roc_auc      = best.get("cv_roc_auc", 0.0),
        cv_roc_auc_std  = best.get("cv_roc_auc_std", 0.0),
        test_roc_auc    = best.get("test_roc_auc", 0.0),
        test_precision  = best.get("test_precision", 0.0),
        test_recall     = best.get("test_recall", 0.0),
        test_f1         = best.get("test_f1", 0.0),
        feature_names   = meta.get("feature_names", []),
        n_features      = len(meta.get("feature_names", [])),
        all_runs        = meta.get("all_runs", []),
    )


@app.post(
    "/predict",
    response_model = PredictionResponse,
    summary        = "Score a single customer for churn risk",
    tags           = ["Prediction"],
    status_code    = status.HTTP_200_OK,
    responses      = {
        422: {"model": ErrorResponse, "description": "Validation error — invalid input fields"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
        500: {"model": ErrorResponse, "description": "Prediction failed"},
    },
)
async def predict(
    features: CustomerFeatures,
    request:  Request,
) -> PredictionResponse:
    """
    Score a single customer and return:
    - **churn_probability** — model's raw probability (0.0–1.0)
    - **churn_label** — binary prediction at 0.5 threshold
    - **risk_segment** — Low / Medium / High / Critical
    - **expected_revenue_loss** — `probability × monthly_charges × 6`
    - **retention_strategy** — rule-based recommended actions
    - **top_reasons** — top-3 SHAP-derived plain-English churn drivers

    All fields are validated — see the request schema for allowed values.
    """
    predictor  = _get_predictor(request)
    input_dict = _features_to_dict(features)

    try:
        result = predictor.predict_single(input_dict)
    except Exception as exc:
        logger.error("predict_single failed for customer '%s': %s", features.customer_id, exc)
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail      = f"Prediction failed: {str(exc)}",
        )

    return PredictionResponse(
        customer_id           = result.customer_id,
        churn_probability     = result.churn_probability,
        churn_label           = result.churn_label,
        risk_segment          = result.risk_segment,
        expected_revenue_loss = result.expected_revenue_loss,
        retention_strategy    = result.retention_strategy,
        top_reasons           = result.top_reasons,
    )


@app.post(
    "/batch-predict",
    response_model = BatchPredictResponse,
    summary        = "Score up to 1 000 customers in a single request",
    tags           = ["Prediction"],
    status_code    = status.HTTP_200_OK,
    responses      = {
        422: {"model": ErrorResponse, "description": "Validation error in one or more customers"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
        500: {"model": ErrorResponse, "description": "Batch prediction failed"},
    },
)
async def batch_predict(
    payload:  BatchPredictRequest,
    request:  Request,
) -> BatchPredictResponse:
    """
    Score a batch of customers (up to **1 000** per request).

    Each customer is scored independently. The response includes:
    - Per-customer predictions (same order as the request)
    - **high_risk_count** — customers in High or Critical segments
    - **total_expected_loss** — aggregate revenue at risk across the batch

    All customers in the batch must pass validation — if any field is
    invalid, the entire request is rejected with HTTP 422.
    """
    predictor   = _get_predictor(request)
    customer_list = [_features_to_dict(f) for f in payload.customers]

    try:
        results = predictor.predict_batch(customer_list)
    except Exception as exc:
        logger.error("predict_batch failed (batch size=%d): %s", len(customer_list), exc)
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail      = f"Batch prediction failed: {str(exc)}",
        )

    predictions = [
        BatchPredictItem(
            customer_id           = r.customer_id,
            churn_probability     = r.churn_probability,
            churn_label           = r.churn_label,
            risk_segment          = r.risk_segment,
            expected_revenue_loss = r.expected_revenue_loss,
            retention_strategy    = r.retention_strategy,
            top_reasons           = r.top_reasons,
        )
        for r in results
    ]

    high_risk_count    = sum(1 for p in predictions if p.risk_segment in ("High", "Critical"))
    total_expected_loss = round(sum(p.expected_revenue_loss for p in predictions), 2)

    logger.info(
        "Batch scored: total=%d  high_risk=%d  total_expected_loss=%.2f",
        len(predictions), high_risk_count, total_expected_loss,
    )

    return BatchPredictResponse(
        total               = len(predictions),
        predictions         = predictions,
        high_risk_count     = high_risk_count,
        total_expected_loss = total_expected_loss,
    )
