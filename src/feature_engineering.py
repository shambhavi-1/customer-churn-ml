"""
feature_engineering.py
=======================
Feature construction and validation for the Customer Churn Prediction System.

Purpose
-------
Accepts a raw or lightly-cleaned DataFrame and returns an enriched DataFrame
that is ready for model training or inference. All transformations are
deterministic and stateless — no fitting is required — making this module safe
to call on train, validation, and test splits identically.

Pipeline (in order)
-------------------
1. Normalise column names          (lowercase, strip whitespace, replace spaces)
2. Validate required columns       (raise FeatureEngineeringError if missing)
3. Validate value constraints      (tenure > 0, monthly_charges > 0)
4. Impute optional columns         (fill NaN / missing cols with sensible defaults)
5. Engineer new features           (CLV, engagement score, ratios, tenure group)

Engineered features
-------------------
customer_lifetime_value     : tenure × monthly_charges  — proxy for revenue at risk
engagement_score            : normalised mean of (number_of_logins, usage_hours)
support_interaction_ratio   : support_calls / (tenure + 1)  — avoids div-by-zero
charge_ratio                : monthly_charges / (total_charges + 1)
tenure_group                : ordinal bucket: "new" | "developing" | "established" | "loyal"

Usage
-----
    from src.feature_engineering import engineer_features

    df_out = engineer_features(df_raw)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Columns that MUST be present (post name-normalisation)
REQUIRED_COLUMNS: list[str] = [
    "tenure",
    "monthly_charges",
    "support_calls",
    "contract_type",
    "internet_service",
    "payment_method",
]

# Optional columns and their scalar fill defaults when absent / NaN
OPTIONAL_COLUMNS: dict[str, object] = {
    "total_charges":    None,   # derived from tenure × monthly_charges (see below)
    "number_of_logins": 8,
    "usage_hours":      40.0,
}

# Tenure-group bucket edges (months) and human-readable labels
TENURE_BINS:   list[int] = [0, 12, 36, 60, np.inf]
TENURE_LABELS: list[str] = ["new", "developing", "established", "loyal"]


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class FeatureEngineeringError(ValueError):
    """Raised when the input DataFrame fails schema or value validation."""


# ---------------------------------------------------------------------------
# Step 1 — Column name normalisation
# ---------------------------------------------------------------------------

def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of *df* with all column names:
      - stripped of leading/trailing whitespace
      - lowercased
      - internal spaces replaced with underscores

    Examples
    --------
    "Monthly Charges" → "monthly_charges"
    "  Tenure "       → "tenure"
    """
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )
    logger.debug("Columns after normalisation: %s", df.columns.tolist())
    return df


# ---------------------------------------------------------------------------
# Step 2 — Required column validation
# ---------------------------------------------------------------------------

def _validate_required_columns(df: pd.DataFrame) -> None:
    """
    Raise :class:`FeatureEngineeringError` listing every missing required column.

    Parameters
    ----------
    df : DataFrame with normalised column names.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise FeatureEngineeringError(
            f"Input DataFrame is missing {len(missing)} required column(s): "
            f"{missing}. "
            f"Present columns: {df.columns.tolist()}"
        )
    logger.debug("All required columns present.")


# ---------------------------------------------------------------------------
# Step 3 — Value constraint validation
# ---------------------------------------------------------------------------

def _validate_values(df: pd.DataFrame) -> None:
    """
    Raise :class:`FeatureEngineeringError` if any business-rule constraint is
    violated on non-null rows.

    Rules
    -----
    - tenure          must be >= 0  (negative tenure is nonsensical)
    - monthly_charges must be >  0  (a customer must have a positive charge)
    """
    errors: list[str] = []

    # tenure >= 0
    bad_tenure = df["tenure"].dropna()
    bad_tenure = bad_tenure[bad_tenure < 0]
    if not bad_tenure.empty:
        errors.append(
            f"'tenure' has {len(bad_tenure)} row(s) with negative values "
            f"(indices: {bad_tenure.index.tolist()[:10]}{'…' if len(bad_tenure) > 10 else ''})."
        )

    # monthly_charges > 0
    bad_charges = df["monthly_charges"].dropna()
    bad_charges = bad_charges[bad_charges <= 0]
    if not bad_charges.empty:
        errors.append(
            f"'monthly_charges' has {len(bad_charges)} row(s) with zero or "
            f"negative values "
            f"(indices: {bad_charges.index.tolist()[:10]}{'…' if len(bad_charges) > 10 else ''})."
        )

    if errors:
        raise FeatureEngineeringError(
            "Value constraint violation(s) detected:\n  " + "\n  ".join(errors)
        )

    logger.debug("Value constraints passed.")


# ---------------------------------------------------------------------------
# Step 4 — Optional column imputation
# ---------------------------------------------------------------------------

def _impute_optional_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure optional columns exist with sensible defaults.

    - total_charges  : if absent or NaN → tenure × monthly_charges
    - number_of_logins : if absent or NaN → 8  (conservative engagement proxy)
    - usage_hours      : if absent or NaN → 40.0
    """
    df = df.copy()

    # --- total_charges (derived, not a fixed scalar)
    if "total_charges" not in df.columns:
        logger.debug("'total_charges' column absent — deriving from tenure × monthly_charges.")
        df["total_charges"] = df["tenure"] * df["monthly_charges"]
    else:
        null_mask = df["total_charges"].isna()
        if null_mask.any():
            logger.debug(
                "Imputing %d NaN value(s) in 'total_charges' using tenure × monthly_charges.",
                null_mask.sum(),
            )
            df.loc[null_mask, "total_charges"] = (
                df.loc[null_mask, "tenure"] * df.loc[null_mask, "monthly_charges"]
            )

    # --- fixed-scalar optional columns
    for col, default in [("number_of_logins", 8), ("usage_hours", 40.0)]:
        if col not in df.columns:
            logger.debug("'%s' column absent — filling with default value %s.", col, default)
            df[col] = float(default)
        else:
            null_mask = df[col].isna()
            if null_mask.any():
                logger.debug(
                    "Imputing %d NaN value(s) in '%s' with default %s.",
                    null_mask.sum(), col, default,
                )
                df.loc[null_mask, col] = float(default)

    return df


# ---------------------------------------------------------------------------
# Step 5 — Feature engineering
# ---------------------------------------------------------------------------

def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append five derived features to *df* and return the augmented copy.

    New columns
    -----------
    customer_lifetime_value   float  tenure × monthly_charges
    engagement_score          float  mean of min-max-scaled logins and usage_hours
                                     → 0 = fully disengaged, 1 = maximally engaged
    support_interaction_ratio float  support_calls / (tenure + 1)
    charge_ratio              float  monthly_charges / (total_charges + 1)
    tenure_group              str    ordinal label derived from TENURE_BINS
    """
    df = df.copy()

    # 1. Customer Lifetime Value — revenue at risk if customer churns
    df["customer_lifetime_value"] = (
        df["tenure"] * df["monthly_charges"]
    ).round(2)

    # 2. Engagement score — normalised to [0, 1]
    #    We clip to avoid edge-case division by zero on constant columns.
    login_max = max(df["number_of_logins"].max(), 1)
    usage_max = max(df["usage_hours"].max(), 1)
    login_norm = df["number_of_logins"] / login_max
    usage_norm = df["usage_hours"]      / usage_max
    df["engagement_score"] = ((login_norm + usage_norm) / 2).round(4)

    # 3. Support interaction ratio — call pressure relative to account age
    df["support_interaction_ratio"] = (
        df["support_calls"] / (df["tenure"] + 1)
    ).round(4)

    # 4. Charge ratio — monthly burden relative to cumulative spend
    df["charge_ratio"] = (
        df["monthly_charges"] / (df["total_charges"] + 1)
    ).round(6)

    # 5. Tenure group — human-readable ordinal bucket
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=TENURE_BINS,
        labels=TENURE_LABELS,
        right=True,
        include_lowest=True,
    ).astype(str)   # keep as plain string for downstream encoders

    logger.debug(
        "Engineered features added: customer_lifetime_value, engagement_score, "
        "support_interaction_ratio, charge_ratio, tenure_group."
    )
    return df


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline on *df* and return the
    processed DataFrame.

    Steps
    -----
    1. Normalise column names
    2. Validate required columns  → raises FeatureEngineeringError if missing
    3. Validate value constraints → raises FeatureEngineeringError on violations
    4. Impute optional columns
    5. Add engineered features

    Parameters
    ----------
    df : Raw input DataFrame (may contain NaN in optional columns).

    Returns
    -------
    pd.DataFrame
        Original columns + 5 engineered features, with optional columns
        guaranteed to be present and non-null.

    Raises
    ------
    FeatureEngineeringError
        If required columns are missing or value constraints are violated.

    Examples
    --------
    >>> import pandas as pd
    >>> from src.feature_engineering import engineer_features
    >>> raw = pd.read_csv("data/churn_dataset.csv")
    >>> processed = engineer_features(raw)
    >>> processed.shape
    (5000, 16)
    """
    logger.info("Starting feature engineering pipeline (input shape: %s).", df.shape)

    df = _normalize_column_names(df)
    _validate_required_columns(df)
    _validate_values(df)
    df = _impute_optional_columns(df)
    df = _add_engineered_features(df)

    logger.info(
        "Feature engineering complete. Output shape: %s. "
        "New features: customer_lifetime_value, engagement_score, "
        "support_interaction_ratio, charge_ratio, tenure_group.",
        df.shape,
    )
    return df
