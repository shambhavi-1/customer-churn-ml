"""
predict.py
==========
Inference engine for the Customer Churn Prediction System.

Purpose
-------
Loads a serialised imblearn Pipeline (best_model.pkl) and exposes
predict_single() for real-time single-customer scoring.

Each prediction returns:
    - churn_probability       : float [0, 1]
    - churn_label             : int   {0, 1}
    - risk_segment            : str   "Low" | "Medium" | "High" | "Critical"
    - expected_revenue_loss   : float  churn_probability × monthly_charges × 6
    - retention_strategy      : list[str]  rule-based actions
    - top_reasons             : list[str]  top-3 SHAP-derived plain-English reasons

Business logic
--------------
Expected revenue loss
    Assumes a 6-month forward window:
        expected_loss = churn_probability × monthly_charges × 6

Retention rules (can fire simultaneously)
    1. high charges      : monthly_charges > 800     → offer discount / loyalty plan
    2. high support calls: support_calls   >= 5      → assign priority support agent
    3. month-to-month    : contract_type   contains "Month" → upsell annual plan
    4. low engagement    : engagement_score < 0.3    → trigger re-engagement campaign

Risk segments
    Critical : probability >= 0.75
    High     : probability >= 0.50
    Medium   : probability >= 0.25
    Low      : probability <  0.25

Explainability
--------------
Uses shap.TreeExplainer for Random Forest / XGBoost (fast, exact).
Falls back to shap.LinearExplainer for Logistic Regression.
Returns the top-3 features by absolute SHAP value as human-readable strings.

Dependencies
------------
    pip install shap xgboost imbalanced-learn scikit-learn numpy pandas

Usage
-----
    from src.predict import ChurnPredictor

    predictor = ChurnPredictor.load("models/best_model.pkl")
    result    = predictor.predict_single(customer_dict)
    print(result)
"""

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.data_processing import DataProcessor, DataProcessorConfig
from src.feature_engineering import engineer_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Risk segment thresholds
RISK_THRESHOLDS = {
    "Critical": 0.75,
    "High":     0.50,
    "Medium":   0.25,
}

# Retention rule thresholds
HIGH_CHARGES_THRESHOLD    = 800.0
HIGH_SUPPORT_THRESHOLD    = 5
LOW_ENGAGEMENT_THRESHOLD  = 0.3

# Revenue loss window (months)
REVENUE_LOSS_MONTHS = 6

# Number of SHAP reasons to surface
TOP_N_REASONS = 3

# Human-readable feature labels for SHAP explanations
FEATURE_LABELS: Dict[str, str] = {
    "tenure":                    "account tenure",
    "monthly_charges":           "monthly charges",
    "total_charges":             "total charges to date",
    "support_calls":             "number of support calls",
    "number_of_logins":          "login frequency",
    "usage_hours":               "usage hours",
    "customer_lifetime_value":   "customer lifetime value",
    "engagement_score":          "overall engagement score",
    "support_interaction_ratio": "support call rate",
    "charge_ratio":              "charge-to-total ratio",
    "tenure_group":              "tenure group",
    "contract_type_One Year":    "one-year contract status",
    "contract_type_Two Year":    "two-year contract status",
    "internet_service_Fiber Optic": "fiber optic service",
    "internet_service_No":       "no internet service",
    "payment_method_Credit Card":        "credit card payment",
    "payment_method_Electronic Check":   "electronic check payment",
    "payment_method_Mailed Check":       "mailed check payment",
}


# ---------------------------------------------------------------------------
# Prediction result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """
    Full structured output of a single customer churn prediction.

    Attributes
    ----------
    customer_id           : Passed-through from input (or None).
    churn_probability     : Model's raw probability of churning, 0.0–1.0.
    churn_label           : Binary prediction at 0.5 threshold (0=retained, 1=churn).
    risk_segment          : "Low" | "Medium" | "High" | "Critical".
    expected_revenue_loss : churn_probability × monthly_charges × 6 (£/$/₹).
    retention_strategy    : List of recommended retention actions.
    top_reasons           : Top-3 SHAP-driven plain-English churn drivers.
    raw_shap_values       : Raw SHAP values array (for downstream visualisation).
    feature_names         : Ordered feature names matching raw_shap_values.
    """

    customer_id:           Optional[str]
    churn_probability:     float
    churn_label:           int
    risk_segment:          str
    expected_revenue_loss: float
    retention_strategy:    List[str]
    top_reasons:           List[str]
    raw_shap_values:       Optional[np.ndarray]   = None
    feature_names:         Optional[List[str]]    = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (JSON-safe — SHAP array excluded)."""
        return {
            "customer_id":           self.customer_id,
            "churn_probability":     round(self.churn_probability, 4),
            "churn_label":           self.churn_label,
            "risk_segment":          self.risk_segment,
            "expected_revenue_loss": round(self.expected_revenue_loss, 2),
            "retention_strategy":    self.retention_strategy,
            "top_reasons":           self.top_reasons,
        }

    def __str__(self) -> str:
        lines = [
            "─" * 52,
            f"  Customer          : {self.customer_id or 'N/A'}",
            f"  Churn Probability : {self.churn_probability:.1%}",
            f"  Churn Label       : {'CHURN' if self.churn_label else 'RETAIN'}",
            f"  Risk Segment      : {self.risk_segment}",
            f"  Expected Loss     : ${self.expected_revenue_loss:,.2f}",
            "  Retention Actions :",
            *[f"    • {s}" for s in self.retention_strategy],
            "  Top Churn Reasons :",
            *[f"    {i+1}. {r}" for i, r in enumerate(self.top_reasons)],
            "─" * 52,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Business logic helpers
# ---------------------------------------------------------------------------

def _compute_risk_segment(probability: float) -> str:
    """
    Map a churn probability to a risk segment label.

    Thresholds
    ----------
    >= 0.75 → Critical
    >= 0.50 → High
    >= 0.25 → Medium
    <  0.25 → Low
    """
    if probability >= RISK_THRESHOLDS["Critical"]:
        return "Critical"
    if probability >= RISK_THRESHOLDS["High"]:
        return "High"
    if probability >= RISK_THRESHOLDS["Medium"]:
        return "Medium"
    return "Low"


def _compute_expected_revenue_loss(
    churn_probability: float, monthly_charges: float
) -> float:
    """
    Compute expected revenue at risk over the next 6 months.

    Formula
    -------
        expected_loss = churn_probability × monthly_charges × REVENUE_LOSS_MONTHS

    Parameters
    ----------
    churn_probability : Model output probability in [0, 1].
    monthly_charges   : Customer's current monthly billing amount.

    Returns
    -------
    Float — expected revenue loss in the same currency as monthly_charges.
    """
    return round(churn_probability * monthly_charges * REVENUE_LOSS_MONTHS, 2)


def _build_retention_strategy(raw_features: Dict[str, Any]) -> List[str]:
    """
    Apply rule-based retention logic to the raw (pre-engineering) feature dict
    and return a list of recommended actions.

    Rules (can fire simultaneously)
    --------------------------------
    1. monthly_charges > 800          → offer loyalty discount / reduced-rate plan
    2. support_calls  >= 5            → escalate to priority support agent
    3. contract_type contains 'Month' → upsell to annual or two-year contract
    4. engagement_score < 0.3 OR
       (number_of_logins <= 8 AND
        usage_hours <= 40)            → trigger personalised re-engagement campaign

    Parameters
    ----------
    raw_features : Dict of raw customer feature values (from the input dict).

    Returns
    -------
    List of plain-English retention recommendation strings.
    If no rule fires, returns a default watch-and-monitor message.
    """
    strategies: List[str] = []

    monthly_charges  = float(raw_features.get("monthly_charges", 0))
    support_calls    = int(raw_features.get("support_calls", 0))
    contract_type    = str(raw_features.get("contract_type", ""))
    number_of_logins = float(raw_features.get("number_of_logins", 8))
    usage_hours      = float(raw_features.get("usage_hours", 40))
    # engagement_score may come from engineered features or be estimated here
    engagement_score = float(raw_features.get("engagement_score", (number_of_logins / 50 + usage_hours / 200) / 2))

    # Rule 1 — High monthly charges
    if monthly_charges > HIGH_CHARGES_THRESHOLD:
        strategies.append(
            f"Offer a loyalty discount or reduced-rate plan "
            f"(current charge: ${monthly_charges:.0f}/mo)"
        )

    # Rule 2 — High support call volume
    if support_calls >= HIGH_SUPPORT_THRESHOLD:
        strategies.append(
            f"Escalate to a Priority Support Agent "
            f"({support_calls} support calls logged)"
        )

    # Rule 3 — Month-to-month contract
    if "month" in contract_type.lower():
        strategies.append(
            "Upsell to an Annual or Two-Year contract with incentive pricing"
        )

    # Rule 4 — Low engagement
    if engagement_score < LOW_ENGAGEMENT_THRESHOLD:
        strategies.append(
            "Trigger a personalised re-engagement campaign "
            "(low login frequency and usage hours detected)"
        )

    # Fallback — no specific rule fired
    if not strategies:
        strategies.append(
            "Monitor account health — no immediate escalation required"
        )

    return strategies


# ---------------------------------------------------------------------------
# SHAP explainability helpers
# ---------------------------------------------------------------------------

def _build_shap_explainer(pipeline, X_background: np.ndarray):
    """
    Construct the appropriate SHAP explainer for the classifier inside pipeline.

    Explainer selection
    -------------------
    XGBClassifier       → shap.TreeExplainer   (exact, fast)
    RandomForestClassifier → shap.TreeExplainer (exact, fast)
    LogisticRegression  → shap.LinearExplainer  (exact for linear models)
    All others          → shap.KernelExplainer  (model-agnostic, slow)

    Parameters
    ----------
    pipeline     : Fitted ImbPipeline. Classifier is the last step.
    X_background : Background dataset for KernelExplainer (ignored for Tree/Linear).

    Returns
    -------
    SHAP explainer instance.
    """
    clf = pipeline.steps[-1][1]

    if isinstance(clf, (XGBClassifier, RandomForestClassifier)):
        return shap.TreeExplainer(clf)

    if isinstance(clf, LogisticRegression):
        # LinearExplainer needs the data after all preprocessing steps except clf
        # We pass the background through the pipeline's non-clf steps
        transformer = _get_preprocessing_transform(pipeline)
        if transformer is not None:
            X_bg_transformed = transformer.transform(X_background)
        else:
            X_bg_transformed = X_background
        return shap.LinearExplainer(clf, X_bg_transformed, feature_perturbation="interventional")

    # Fallback — model-agnostic
    logger.warning(
        "Using slow KernelExplainer for %s. "
        "Consider switching to a tree-based model for faster explanations.",
        type(clf).__name__,
    )
    predict_fn = lambda x: pipeline.predict_proba(x)[:, 1]
    summary = shap.kmeans(X_background, 10)
    return shap.KernelExplainer(predict_fn, summary)


def _get_preprocessing_transform(pipeline):
    """
    Return a partial pipeline containing all steps before 'clf', or None.

    Used to transform data before passing to LinearExplainer.
    """
    from sklearn.pipeline import Pipeline as SkPipeline
    from imblearn.pipeline import Pipeline as ImbPipeline

    pre_steps = [(name, step) for name, step in pipeline.steps if name != "clf" and name != "smote"]
    if not pre_steps:
        return None
    return SkPipeline(pre_steps)


def _compute_shap_values(
    explainer,
    pipeline,
    X_row: np.ndarray,
) -> np.ndarray:
    """
    Compute SHAP values for a single observation.

    Handles the difference between TreeExplainer (returns array directly) and
    other explainers (returns Explanation objects).

    Parameters
    ----------
    explainer : Fitted SHAP explainer.
    pipeline  : Fitted pipeline (used to preprocess X_row for LinearExplainer).
    X_row     : 2-D numpy array, shape (1, n_features).

    Returns
    -------
    1-D numpy array of SHAP values, one per feature.
    """
    clf = pipeline.steps[-1][1]

    if isinstance(clf, (XGBClassifier, RandomForestClassifier)):
        # TreeExplainer — pass raw features directly to the classifier
        shap_values = explainer.shap_values(X_row)
        # RF returns list [class0, class1]; XGB returns 2-D array
        if isinstance(shap_values, list):
            shap_values = shap_values[1]   # class-1 (churn) SHAP values
        return np.array(shap_values).flatten()

    if isinstance(clf, LogisticRegression):
        transformer = _get_preprocessing_transform(pipeline)
        X_transformed = transformer.transform(X_row) if transformer else X_row
        shap_values = explainer.shap_values(X_transformed)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        return np.array(shap_values).flatten()

    # KernelExplainer
    shap_values = explainer.shap_values(X_row)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    return np.array(shap_values).flatten()


def _shap_to_reasons(
    shap_values: np.ndarray,
    feature_names: List[str],
    feature_values: np.ndarray,
    top_n: int = TOP_N_REASONS,
) -> List[str]:
    """
    Convert raw SHAP values into plain-English top-N churn driver sentences.

    Only features with *positive* SHAP values (i.e. pushing toward churn)
    are surfaced.  Falls back to top absolute-value features if all are negative.

    Sentence template
    -----------------
    "[Feature label] is [value] — [increases/decreases] churn risk [significantly/moderately/slightly]"

    Parameters
    ----------
    shap_values    : 1-D array of SHAP values (one per feature).
    feature_names  : Ordered list of feature names.
    feature_values : 1-D array of feature values for this customer.
    top_n          : Number of reasons to return.

    Returns
    -------
    List of plain-English reason strings, length <= top_n.
    """
    # Rank by absolute SHAP value, prefer positive contributors
    indexed = list(enumerate(shap_values))
    positive = [(i, v) for i, v in indexed if v > 0]
    if len(positive) >= top_n:
        ranked = sorted(positive, key=lambda x: abs(x[1]), reverse=True)
    else:
        ranked = sorted(indexed, key=lambda x: abs(x[1]), reverse=True)

    reasons: List[str] = []
    for i, shap_val in ranked[:top_n]:
        feat_name  = feature_names[i] if i < len(feature_names) else f"feature_{i}"
        feat_label = FEATURE_LABELS.get(feat_name, feat_name.replace("_", " "))
        feat_val   = feature_values[i] if i < len(feature_values) else "N/A"
        direction  = "increases" if shap_val > 0 else "decreases"

        abs_val = abs(shap_val)
        if abs_val >= 0.15:
            magnitude = "significantly"
        elif abs_val >= 0.07:
            magnitude = "moderately"
        else:
            magnitude = "slightly"

        # Format numeric values cleanly
        if isinstance(feat_val, (float, np.floating)):
            val_str = f"{feat_val:.2f}"
        else:
            val_str = str(feat_val)

        reasons.append(
            f"{feat_label.capitalize()} ({val_str}) "
            f"{direction} churn risk {magnitude}"
        )

    return reasons if reasons else ["Insufficient SHAP data to determine top reasons"]


# ---------------------------------------------------------------------------
# Main predictor class
# ---------------------------------------------------------------------------

class ChurnPredictor:
    """
    Production-ready inference engine for churn prediction.

    Load once at application startup via ChurnPredictor.load(), then call
    predict_single() per customer request.

    Parameters
    ----------
    pipeline          : Fitted ImbPipeline loaded from best_model.pkl.
    feature_names     : Ordered list of feature names (from DataProcessor).
    background_data   : Small numpy array used to initialise the SHAP explainer.
                        Typically 100–200 rows from the training set.
    """

    def __init__(
        self,
        pipeline:        Any,
        feature_names:   List[str],
        background_data: Optional[np.ndarray] = None,
    ) -> None:
        self.pipeline       = pipeline
        self.feature_names  = feature_names
        self._explainer     = None          # lazily initialised on first call
        self._background    = background_data

        # Internal DataProcessor used for feature engineering + encoding
        # Pre-fit encoders using known category values so single-row
        # predictions never trigger the "encoders not fitted" fallback
        self._processor = DataProcessor()
        self._fit_processor_encoders()

    def _fit_processor_encoders(self) -> None:
        """
        Fit the DataProcessor encoders using a minimal synthetic DataFrame
        that contains all known category values for each categorical column.
        This runs once at startup and takes <1ms — ensures encode(fit=False)
        always works for single-row inference without touching the dataset.
        """
        import pandas as pd
        seed_df = pd.DataFrame({
            "tenure":           [0, 12, 72],
            "monthly_charges":  [300.0, 650.0, 1200.0],
            "total_charges":    [0.0, 7800.0, 86400.0],
            "support_calls":    [0, 5, 10],
            "number_of_logins": [8.0, 25.0, 50.0],
            "usage_hours":      [40.0, 100.0, 200.0],
            "contract_type":    ["Month-to-Month", "One Year", "Two Year"],
            "internet_service": ["Fiber Optic", "DSL", "No"],
            "payment_method":   ["Electronic Check", "Mailed Check", "Credit Card"],
        })
        from src.feature_engineering import engineer_features
        seed_eng = engineer_features(seed_df)
        self._processor.encode(seed_eng, fit=True)

    # -----------------------------------------------------------------------
    # Factory
    # -----------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        model_path:        Union[str, Path] = "models/best_model.pkl",
        metadata_path:     Union[str, Path] = "models/model_metadata.json",
        background_data:   Optional[np.ndarray] = None,
    ) -> "ChurnPredictor":
        """
        Deserialise the model pipeline and feature names from disk.

        Parameters
        ----------
        model_path      : Path to best_model.pkl.
        metadata_path   : Path to model_metadata.json (provides feature_names).
        background_data : Optional background numpy array for SHAP.

        Returns
        -------
        Initialised ChurnPredictor, ready to call predict_single().

        Raises
        ------
        FileNotFoundError : If either artefact file does not exist.
        """
        model_path    = Path(model_path)
        metadata_path = Path(metadata_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"best_model.pkl not found at '{model_path}'. "
                "Run src/run_training.py first."
            )
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"model_metadata.json not found at '{metadata_path}'. "
                "Run src/run_training.py first."
            )

        with open(model_path, "rb") as fh:
            pipeline = pickle.load(fh)

        with open(metadata_path, "r") as fh:
            metadata = json.load(fh)

        feature_names = metadata.get("feature_names", [])
        logger.info(
            "Loaded model '%s' with %d features.",
            metadata.get("best_model", {}).get("run_id", "unknown"),
            len(feature_names),
        )

        return cls(
            pipeline=pipeline,
            feature_names=feature_names,
            background_data=background_data,
        )

    # -----------------------------------------------------------------------
    # Warm-up
    # -----------------------------------------------------------------------

    def warm_up(self, background_data: Optional[np.ndarray] = None) -> None:
        """
        Pre-initialise the SHAP explainer so the first prediction is fast.

        Parameters
        ----------
        background_data : numpy array used to fit the SHAP explainer.
                          Pass a representative sample (100–200 rows) from
                          the training set for best results.
        """
        if background_data is not None:
            self._background = background_data
        if self._background is not None and self._explainer is None:
            logger.info("Initialising SHAP explainer …")
            self._explainer = _build_shap_explainer(self.pipeline, self._background)
            logger.info("SHAP explainer ready.")

    # -----------------------------------------------------------------------
    # Core prediction
    # -----------------------------------------------------------------------

    def predict_single(self, customer: Dict[str, Any]) -> PredictionResult:
        """
        Score a single customer and return a fully populated PredictionResult.

        Pipeline
        --------
        1. Wrap the raw dict in a single-row DataFrame
        2. Apply feature engineering  (engineer_features)
        3. Drop customer_id, encode categoricals via fitted DataProcessor
        4. Align columns to the training schema (reindex)
        5. Run model.predict_proba()
        6. Compute risk segment, expected revenue loss, retention strategy
        7. Compute SHAP values → top-3 plain-English reasons

        Parameters
        ----------
        customer : Dict mapping raw feature names to values.
                   Must include all required fields:
                   tenure, monthly_charges, support_calls,
                   contract_type, internet_service, payment_method.
                   Optional: total_charges, number_of_logins, usage_hours,
                             customer_id.

        Returns
        -------
        PredictionResult

        Raises
        ------
        FeatureEngineeringError : If required columns are missing or invalid.
        RuntimeError            : If the feature matrix cannot be aligned.

        Example
        -------
        >>> predictor = ChurnPredictor.load()
        >>> result = predictor.predict_single({
        ...     "customer_id":      "CUST-00042",
        ...     "tenure":           6,
        ...     "monthly_charges":  950.0,
        ...     "support_calls":    7,
        ...     "contract_type":    "Month-to-Month",
        ...     "internet_service": "Fiber Optic",
        ...     "payment_method":   "Electronic Check",
        ... })
        >>> print(result)
        """
        customer_id = customer.get("customer_id")

        # ── Step 1: single-row DataFrame ────────────────────────────────────
        row_df = pd.DataFrame([customer])

        # ── Step 2: feature engineering ─────────────────────────────────────
        row_eng = engineer_features(row_df)

        # ── Step 3: drop non-feature columns, encode categoricals ───────────
        drop_cols = [c for c in ["customer_id", "churn"] if c in row_eng.columns]
        row_eng   = row_eng.drop(columns=drop_cols)
        row_enc   = self._processor.encode(row_eng, fit=False)

        # ── Step 4: align to training feature schema ────────────────────────
        X_row = self._align_features(row_enc)

        # ── Step 5: predict ──────────────────────────────────────────────────
        churn_probability = float(self.pipeline.predict_proba(X_row)[0, 1])
        churn_label       = int(churn_probability >= 0.5)

        # ── Step 6: business logic ───────────────────────────────────────────
        risk_segment          = _compute_risk_segment(churn_probability)
        monthly_charges       = float(customer.get("monthly_charges", 0))
        expected_revenue_loss = _compute_expected_revenue_loss(
            churn_probability, monthly_charges
        )
        retention_strategy    = _build_retention_strategy(customer)

        # ── Step 7: SHAP explanations ────────────────────────────────────────
        shap_values = None
        top_reasons: List[str] = []

        if self._explainer is not None:
            try:
                shap_values = _compute_shap_values(
                    self._explainer, self.pipeline, X_row
                )
                top_reasons = _shap_to_reasons(
                    shap_values,
                    self.feature_names,
                    X_row.flatten(),
                )
            except Exception as exc:
                logger.warning("SHAP computation failed: %s. Skipping reasons.", exc)
                top_reasons = ["SHAP explanation unavailable for this prediction"]
        else:
            top_reasons = self._fallback_reasons(customer, churn_probability)

        return PredictionResult(
            customer_id           = str(customer_id) if customer_id else None,
            churn_probability     = churn_probability,
            churn_label           = churn_label,
            risk_segment          = risk_segment,
            expected_revenue_loss = expected_revenue_loss,
            retention_strategy    = retention_strategy,
            top_reasons           = top_reasons,
            raw_shap_values       = shap_values,
            feature_names         = self.feature_names,
        )

    # -----------------------------------------------------------------------
    # Batch prediction
    # -----------------------------------------------------------------------

    def predict_batch(
        self, customers: List[Dict[str, Any]]
    ) -> List[PredictionResult]:
        """
        Score a list of customers and return one PredictionResult per row.

        Note: calls predict_single() in a loop. For very large batches
        (>10k rows), consider vectorising the preprocessing step.

        Parameters
        ----------
        customers : List of raw customer dicts.

        Returns
        -------
        List[PredictionResult] — same order as input.
        """
        results = []
        for i, customer in enumerate(customers):
            try:
                result = self.predict_single(customer)
                results.append(result)
            except Exception as exc:
                logger.error(
                    "predict_single failed for customer %d (%s): %s",
                    i,
                    customer.get("customer_id", "unknown"),
                    exc,
                )
                raise
        return results

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _align_features(self, row_enc: pd.DataFrame) -> np.ndarray:
        """
        Reindex *row_enc* to exactly match the training feature schema.

        Missing columns are filled with 0; extra columns are dropped.
        Returns a 2-D numpy array of shape (1, n_features).

        Raises
        ------
        RuntimeError : If feature_names is empty (model not loaded correctly).
        """
        if not self.feature_names:
            raise RuntimeError(
                "feature_names is empty. "
                "Ensure the model was loaded via ChurnPredictor.load()."
            )
        aligned = row_enc.reindex(columns=self.feature_names, fill_value=0)
        return aligned.values.astype(np.float64)

    def _fallback_reasons(
        self, customer: Dict[str, Any], churn_probability: float
    ) -> List[str]:
        """
        Rule-based fallback reasons when SHAP explainer is not available.

        Surfaces the three most impactful observable risk factors based on
        the raw feature values directly.

        Parameters
        ----------
        customer          : Raw input dict.
        churn_probability : Model's predicted probability.

        Returns
        -------
        List of up to 3 plain-English reason strings.
        """
        reasons: List[str] = []

        tenure          = int(customer.get("tenure", 0))
        monthly_charges = float(customer.get("monthly_charges", 0))
        support_calls   = int(customer.get("support_calls", 0))
        contract_type   = str(customer.get("contract_type", ""))
        usage_hours     = float(customer.get("usage_hours", 40))
        logins          = float(customer.get("number_of_logins", 8))

        if tenure < 12:
            reasons.append(
                f"Short account tenure ({tenure} months) increases churn risk significantly"
            )
        if monthly_charges > HIGH_CHARGES_THRESHOLD:
            reasons.append(
                f"High monthly charges (${monthly_charges:.0f}) increase churn risk moderately"
            )
        if support_calls >= HIGH_SUPPORT_THRESHOLD:
            reasons.append(
                f"High support call volume ({support_calls} calls) indicates service friction"
            )
        if "month" in contract_type.lower():
            reasons.append(
                "Month-to-month contract type increases likelihood of switching"
            )
        engagement = (logins / 50 + usage_hours / 200) / 2
        if engagement < LOW_ENGAGEMENT_THRESHOLD:
            reasons.append(
                f"Low engagement score ({engagement:.2f}) suggests reduced product attachment"
            )

        return reasons[:TOP_N_REASONS] if reasons else [
            f"Churn probability of {churn_probability:.1%} detected — "
            "load model with background data for detailed SHAP reasons"
        ]


# ---------------------------------------------------------------------------
# Convenience function (module-level)
# ---------------------------------------------------------------------------

def predict_single(
    customer:        Dict[str, Any],
    model_path:      Union[str, Path] = "models/best_model.pkl",
    metadata_path:   Union[str, Path] = "models/model_metadata.json",
    background_data: Optional[np.ndarray] = None,
) -> PredictionResult:
    """
    Module-level convenience wrapper — load model and score one customer
    in a single call. For repeated calls, prefer ChurnPredictor.load()
    once and reuse the instance.

    Parameters
    ----------
    customer        : Raw feature dict.
    model_path      : Path to best_model.pkl.
    metadata_path   : Path to model_metadata.json.
    background_data : Optional numpy background for SHAP (enables top_reasons).

    Returns
    -------
    PredictionResult
    """
    predictor = ChurnPredictor.load(
        model_path      = model_path,
        metadata_path   = metadata_path,
        background_data = background_data,
    )
    if background_data is not None:
        predictor.warm_up(background_data)
    return predictor.predict_single(customer)
