"""
schemas.py
==========
Pydantic v2 request and response models for the Churn Prediction REST API.

All schemas are used for:
  - Automatic input validation (FastAPI raises HTTP 422 on violation)
  - Auto-generated OpenAPI / Swagger documentation at /docs
  - Serialisation of all JSON responses

Schemas
-------
Requests
    CustomerFeatures        POST /predict
    BatchPredictRequest     POST /batch-predict

Responses
    HealthResponse          GET  /health
    ModelMetaResponse       GET  /model-meta
    PredictionResponse      POST /predict
    BatchPredictResponse    POST /batch-predict
    ErrorResponse           Any error envelope
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class CustomerFeatures(BaseModel):
    """
    Raw feature payload for a single customer churn prediction.

    Required fields mirror the columns produced by data/generate_data.py.
    Optional fields are imputed by the feature engineering pipeline when absent.
    """

    # ── Identifier (passed through, not used for prediction) ────────────────
    customer_id: Optional[str] = Field(
        default=None,
        description="Unique customer identifier (optional, passed through in response)",
        examples=["CUST-00042"],
    )

    # ── Required numeric features ────────────────────────────────────────────
    tenure: int = Field(
        ...,
        ge=0,
        le=720,
        description="Number of months the customer has been with the service",
        examples=[24],
    )
    monthly_charges: float = Field(
        ...,
        gt=0,
        le=10_000,
        description="Current monthly billing amount in local currency",
        examples=[750.0],
    )
    support_calls: int = Field(
        ...,
        ge=0,
        le=100,
        description="Number of support calls made by the customer",
        examples=[3],
    )

    # ── Required categorical features ────────────────────────────────────────
    contract_type: str = Field(
        ...,
        description="Contract type: 'Month-to-Month', 'One Year', or 'Two Year'",
        examples=["Month-to-Month"],
    )
    internet_service: str = Field(
        ...,
        description="Internet service type: 'DSL', 'Fiber Optic', or 'No'",
        examples=["Fiber Optic"],
    )
    payment_method: str = Field(
        ...,
        description=(
            "Payment method: 'Electronic Check', 'Mailed Check', "
            "'Bank Transfer', or 'Credit Card'"
        ),
        examples=["Electronic Check"],
    )

    # ── Optional numeric features (imputed if absent) ────────────────────────
    total_charges: Optional[float] = Field(
        default=None,
        ge=0,
        description="Total charges to date. Imputed as tenure × monthly_charges if absent.",
        examples=[18000.0],
    )
    number_of_logins: Optional[float] = Field(
        default=None,
        ge=0,
        le=1000,
        description="Number of product logins in the billing period. Defaults to 8 if absent.",
        examples=[15.0],
    )
    usage_hours: Optional[float] = Field(
        default=None,
        ge=0,
        le=8760,
        description="Product usage hours in the billing period. Defaults to 40.0 if absent.",
        examples=[80.0],
    )

    # ── Validators ───────────────────────────────────────────────────────────

    @field_validator("contract_type")
    @classmethod
    def validate_contract_type(cls, v: str) -> str:
        allowed = {"Month-to-Month", "One Year", "Two Year"}
        if v not in allowed:
            raise ValueError(
                f"contract_type '{v}' is not valid. "
                f"Allowed values: {sorted(allowed)}"
            )
        return v

    @field_validator("internet_service")
    @classmethod
    def validate_internet_service(cls, v: str) -> str:
        allowed = {"DSL", "Fiber Optic", "No"}
        if v not in allowed:
            raise ValueError(
                f"internet_service '{v}' is not valid. "
                f"Allowed values: {sorted(allowed)}"
            )
        return v

    @field_validator("payment_method")
    @classmethod
    def validate_payment_method(cls, v: str) -> str:
        allowed = {
            "Electronic Check", "Mailed Check",
            "Bank Transfer", "Credit Card",
        }
        if v not in allowed:
            raise ValueError(
                f"payment_method '{v}' is not valid. "
                f"Allowed values: {sorted(allowed)}"
            )
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "customer_id":      "CUST-00042",
                "tenure":           6,
                "monthly_charges":  950.0,
                "support_calls":    7,
                "contract_type":    "Month-to-Month",
                "internet_service": "Fiber Optic",
                "payment_method":   "Electronic Check",
                "total_charges":    5700.0,
                "number_of_logins": 3,
                "usage_hours":      12.5,
            }
        }
    }


class BatchPredictRequest(BaseModel):
    """Request body for POST /batch-predict — up to 1 000 customers."""

    customers: List[CustomerFeatures] = Field(
        ...,
        min_length=1,
        max_length=1_000,
        description="List of customer feature records to score (max 1 000 per request)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "customers": [
                    {
                        "customer_id":      "CUST-00001",
                        "tenure":           6,
                        "monthly_charges":  950.0,
                        "support_calls":    7,
                        "contract_type":    "Month-to-Month",
                        "internet_service": "Fiber Optic",
                        "payment_method":   "Electronic Check",
                    },
                    {
                        "customer_id":      "CUST-00002",
                        "tenure":           48,
                        "monthly_charges":  400.0,
                        "support_calls":    1,
                        "contract_type":    "Two Year",
                        "internet_service": "DSL",
                        "payment_method":   "Bank Transfer",
                    },
                ]
            }
        }
    }


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str = Field(
        description="Service health status: 'ok' or 'degraded'",
        examples=["ok"],
    )
    model_loaded: bool = Field(
        description="Whether the prediction model is loaded and ready"
    )
    model_run_id: str = Field(
        description="Identifier of the loaded model variant (e.g. 'xgboost_smote')",
        examples=["xgboost_smote"],
    )
    uptime_seconds: float = Field(
        description="Seconds since the API server started"
    )


class ModelMetaResponse(BaseModel):
    """Response for GET /model-meta — metadata about the loaded model."""

    run_id: str = Field(
        description="Model run identifier",
        examples=["xgboost_smote"],
    )
    model_name: str = Field(
        description="Base model class",
        examples=["xgboost"],
    )
    smote: bool = Field(
        description="Whether SMOTE was applied during training"
    )
    trained_at: str = Field(
        description="ISO-8601 UTC timestamp of when the model was trained",
        examples=["2024-06-01T10:30:00+00:00"],
    )
    cv_roc_auc: float = Field(
        description="Mean cross-validated ROC-AUC on training data"
    )
    cv_roc_auc_std: float = Field(
        description="Std-dev of cross-validated ROC-AUC scores"
    )
    test_roc_auc: float = Field(description="ROC-AUC on held-out test set")
    test_precision: float = Field(description="Precision on held-out test set")
    test_recall: float = Field(description="Recall on held-out test set")
    test_f1: float = Field(description="F1-score on held-out test set")
    feature_names: List[str] = Field(
        description="Ordered list of features used by the model"
    )
    n_features: int = Field(description="Number of input features")
    all_runs: List[Dict[str, Any]] = Field(
        description="Metrics summary for every model variant trained"
    )


class PredictionResponse(BaseModel):
    """Response for POST /predict — single customer scoring result."""

    customer_id: Optional[str] = Field(
        default=None,
        description="Passed-through customer identifier from the request",
    )
    churn_probability: float = Field(
        description="Model's predicted probability of churn, 0.0–1.0",
        examples=[0.82],
    )
    churn_label: int = Field(
        description="Binary churn prediction at 0.5 threshold (0=retain, 1=churn)",
        examples=[1],
    )
    risk_segment: str = Field(
        description="Risk band: 'Low' | 'Medium' | 'High' | 'Critical'",
        examples=["Critical"],
    )
    expected_revenue_loss: float = Field(
        description=(
            "Expected revenue at risk over 6 months: "
            "churn_probability × monthly_charges × 6"
        ),
        examples=[4560.0],
    )
    retention_strategy: List[str] = Field(
        description="Recommended retention actions based on customer profile"
    )
    top_reasons: List[str] = Field(
        description="Top-3 SHAP-derived plain-English churn drivers"
    )


class BatchPredictItem(BaseModel):
    """Single prediction result within a batch response."""

    customer_id: Optional[str] = None
    churn_probability: float
    churn_label: int
    risk_segment: str
    expected_revenue_loss: float
    retention_strategy: List[str]
    top_reasons: List[str]


class BatchPredictResponse(BaseModel):
    """Response for POST /batch-predict."""

    total: int = Field(description="Total number of customers scored")
    predictions: List[BatchPredictItem] = Field(
        description="Ordered list of prediction results (same order as request)"
    )
    high_risk_count: int = Field(
        description="Number of customers in 'High' or 'Critical' risk segments"
    )
    total_expected_loss: float = Field(
        description="Sum of expected_revenue_loss across all customers"
    )


class ErrorResponse(BaseModel):
    """Standardised error envelope returned on 4xx / 5xx responses."""

    error: str = Field(description="Short error category label")
    detail: str = Field(description="Human-readable explanation of the error")
    status_code: int = Field(description="HTTP status code")
