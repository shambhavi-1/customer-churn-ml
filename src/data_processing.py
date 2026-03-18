"""
data_processing.py
==================
Data loading, feature engineering, encoding, and train/test splitting.

Purpose
-------
The central pipeline stage that takes a raw CSV and produces four clean,
model-ready arrays: X_train, X_test, y_train, y_test.

Pipeline (in order)
-------------------
1. Load CSV from disk
2. Apply feature engineering  (via src.feature_engineering.engineer_features)
3. Drop non-modelling columns  (customer_id, raw columns superseded by features)
4. Encode categorical variables
       - Binary / low-cardinality : LabelEncoder
       - Multi-class nominal      : One-Hot Encoding (drop='first' to avoid multicollinearity)
5. Stratified train / test split  (default 80 / 20, preserves churn ratio)
6. Return X_train, X_test, y_train, y_test as numpy arrays + column name list

Design notes
------------
- No model training, no SMOTE — those live in later pipeline stages.
- Encoding is fit on the training split only, then applied to test to prevent
  data leakage.  The fitted encoders are stored on the DataProcessor instance
  so they can be reused during inference.

Usage
-----
    from src.data_processing import DataProcessor

    processor = DataProcessor()
    X_train, X_test, y_train, y_test = processor.run("data/churn_dataset.csv")

    # Or step-by-step:
    df       = processor.load("data/churn_dataset.csv")
    df_eng   = processor.apply_feature_engineering(df)
    df_enc   = processor.encode(df_eng, fit=True)
    splits   = processor.split(df_enc)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.feature_engineering import engineer_features, FeatureEngineeringError

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Column containing the target label
TARGET_COLUMN: str = "churn"

# Columns to drop before modelling — identifiers, not features
DROP_COLUMNS: list[str] = ["customer_id"]

# Categorical columns and their encoding strategy
#   "label" : Binary or ordinal — encoded as a single integer column
#   "onehot": Nominal multi-class — expanded into k-1 dummy columns
CATEGORICAL_ENCODING: dict[str, str] = {
    "contract_type":    "onehot",
    "internet_service": "onehot",
    "payment_method":   "onehot",
    "tenure_group":     "label",   # already ordinal: new < developing < established < loyal
}

# Ordinal order for tenure_group (new = 0 … loyal = 3)
TENURE_GROUP_ORDER: list[str] = ["new", "developing", "established", "loyal"]


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class DataProcessorConfig:
    """Runtime settings for DataProcessor."""

    data_path:     Path  = Path("data/churn_dataset.csv")
    target_column: str   = TARGET_COLUMN
    test_size:     float = 0.20
    random_state:  int   = 42
    drop_columns:  list[str] = field(default_factory=lambda: list(DROP_COLUMNS))


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DataProcessor:
    """
    Loads, engineers, encodes, and splits the churn dataset.

    Encoding is fit on the training partition and reused for test / inference
    to eliminate data leakage.

    Parameters
    ----------
    config : DataProcessorConfig  (optional — sensible defaults provided)
    """

    def __init__(self, config: Optional[DataProcessorConfig] = None) -> None:
        self.config = config or DataProcessorConfig()

        # Fitted state — populated during encode(fit=True)
        self._label_encoders: dict[str, LabelEncoder] = {}
        self._onehot_columns: dict[str, list[str]]    = {}   # col → dummy column names
        self._feature_names:  list[str]               = []
        self._is_fitted:      bool                    = False

    # -----------------------------------------------------------------------
    # Step 1 — Load
    # -----------------------------------------------------------------------

    def load(self, path: Optional[Path] = None) -> pd.DataFrame:
        """
        Read the raw CSV from *path* (falls back to config.data_path).

        Raises
        ------
        FileNotFoundError  if the CSV does not exist.
        ValueError         if the target column is absent.
        """
        csv_path = Path(path) if path is not None else self.config.data_path
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at '{csv_path}'. "
                "Run  python data/generate_data.py  first."
            )

        df = pd.read_csv(csv_path)
        logger.info("Loaded dataset: %s  shape=%s", csv_path, df.shape)

        if self.config.target_column not in df.columns:
            raise ValueError(
                f"Target column '{self.config.target_column}' not found in CSV. "
                f"Available columns: {df.columns.tolist()}"
            )

        return df

    # -----------------------------------------------------------------------
    # Step 2 — Apply feature engineering
    # -----------------------------------------------------------------------

    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Delegate to  src.feature_engineering.engineer_features  and log results.

        Returns the enriched DataFrame (original cols + 5 engineered features).

        Raises
        ------
        FeatureEngineeringError  propagated from the feature engineering module.
        """
        logger.info("Applying feature engineering …")
        df_eng = engineer_features(df)
        new_cols = [c for c in df_eng.columns if c not in df.columns]
        logger.info(
            "Feature engineering complete. Added %d column(s): %s",
            len(new_cols), new_cols,
        )
        return df_eng

    # -----------------------------------------------------------------------
    # Step 3 — Drop non-modelling columns
    # -----------------------------------------------------------------------

    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove identifier and other non-modelling columns."""
        to_drop = [c for c in self.config.drop_columns if c in df.columns]
        if to_drop:
            logger.info("Dropping columns: %s", to_drop)
            df = df.drop(columns=to_drop)
        return df

    # -----------------------------------------------------------------------
    # Step 4 — Encode categorical variables
    # -----------------------------------------------------------------------

    def encode(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode all categorical columns defined in CATEGORICAL_ENCODING.

        Parameters
        ----------
        df  : DataFrame after feature engineering and column dropping.
        fit : If True, fit encoders on *df* (training split).
              If False, apply already-fitted encoders (test / inference split).

        Encoding strategies
        -------------------
        "label"  — tenure_group  →  ordinal integer  (new=0 … loyal=3)
        "onehot" — contract_type, internet_service, payment_method
                   → pd.get_dummies with drop_first=True to avoid multicollinearity

        Returns
        -------
        DataFrame with all categorical columns replaced by numeric equivalents.
        """
        df = df.copy()

        for col, strategy in CATEGORICAL_ENCODING.items():
            if col not in df.columns:
                logger.warning("Categorical column '%s' not found — skipping.", col)
                continue

            if strategy == "label":
                df = self._encode_label(df, col, fit=fit)

            elif strategy == "onehot":
                df = self._encode_onehot(df, col, fit=fit)

        if fit:
            self._is_fitted = True
            logger.info("Encoder fitting complete.")

        return df

    def _encode_label(self, df: pd.DataFrame, col: str, fit: bool) -> pd.DataFrame:
        """
        Ordinal-encode *col* as integer.
        tenure_group uses a fixed ordered mapping; all others use LabelEncoder.
        """
        if col == "tenure_group":
            # Fixed ordinal mapping — same regardless of fit/transform
            order_map = {label: idx for idx, label in enumerate(TENURE_GROUP_ORDER)}
            df[col] = df[col].map(order_map).fillna(0).astype(int)
            logger.debug("Ordinal-encoded '%s'  map=%s", col, order_map)
        else:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self._label_encoders[col] = le
                logger.debug("LabelEncoder fitted on '%s'  classes=%s", col, le.classes_.tolist())
            else:
                le = self._label_encoders.get(col)
                if le is None:
                    raise RuntimeError(
                        f"LabelEncoder for '{col}' not found. "
                        "Call encode(fit=True) on training data first."
                    )
                # Handle unseen categories gracefully → map to most frequent class
                known = set(le.classes_)
                df[col] = df[col].astype(str).apply(
                    lambda v: v if v in known else le.classes_[0]
                )
                df[col] = le.transform(df[col])
                logger.debug("LabelEncoder applied to '%s'.", col)
        return df

    def _encode_onehot(self, df: pd.DataFrame, col: str, fit: bool) -> pd.DataFrame:
        """
        One-hot-encode *col*, dropping the first dummy to avoid multicollinearity.
        On fit=True the dummy column names are stored; on fit=False the stored
        columns are reindexed to guarantee identical schema.
        """
        dummies = pd.get_dummies(df[col].astype(str), prefix=col, drop_first=True, dtype=int)

        if fit:
            self._onehot_columns[col] = dummies.columns.tolist()
            logger.debug(
                "One-hot-encoded '%s'  → %d dummy column(s): %s",
                col, len(dummies.columns), dummies.columns.tolist(),
            )
        else:
            expected_cols = self._onehot_columns.get(col, [])
            if not expected_cols:
                raise RuntimeError(
                    f"One-hot schema for '{col}' not found. "
                    "Call encode(fit=True) on training data first."
                )
            # Reindex ensures test set has exactly the same columns as training
            dummies = dummies.reindex(columns=expected_cols, fill_value=0)
            logger.debug("One-hot schema applied to '%s'.", col)

        df = df.drop(columns=[col])
        df = pd.concat([df, dummies], axis=1)
        return df

    # -----------------------------------------------------------------------
    # Step 5 — Stratified train / test split
    # -----------------------------------------------------------------------

    def split(
        self,
        df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Separate features from target, then perform a stratified split.

        Parameters
        ----------
        df : Fully encoded DataFrame (must contain the target column).

        Returns
        -------
        X_train, X_test, y_train, y_test  — all as numpy arrays.
        """
        if self.config.target_column not in df.columns:
            raise ValueError(
                f"Target column '{self.config.target_column}' missing from "
                "DataFrame passed to split(). "
                "Ensure it was not accidentally dropped during encoding."
            )

        X = df.drop(columns=[self.config.target_column])
        y = df[self.config.target_column].values.astype(int)

        self._feature_names = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y,
            test_size    = self.config.test_size,
            stratify     = y,
            random_state = self.config.random_state,
        )

        logger.info(
            "Train/test split complete — "
            "train=%d  test=%d  features=%d  "
            "churn_rate_train=%.2f%%  churn_rate_test=%.2f%%",
            len(y_train), len(y_test), X_train.shape[1],
            y_train.mean() * 100, y_test.mean() * 100,
        )
        return X_train, X_test, y_train, y_test

    # -----------------------------------------------------------------------
    # Public convenience method — run full pipeline in one call
    # -----------------------------------------------------------------------

    def run(
        self,
        path: Optional[Path] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Execute the complete pipeline end-to-end and return the four splits.

        Steps
        -----
        load → feature_engineering → drop_columns → encode(fit=True) → split

        Parameters
        ----------
        path : CSV path override (uses config.data_path if not provided).

        Returns
        -------
        X_train, X_test, y_train, y_test  (numpy arrays)
        """
        df     = self.load(path)
        df_eng = self.apply_feature_engineering(df)
        df_eng = self._drop_columns(df_eng)
        df_enc = self.encode(df_eng, fit=True)
        return self.split(df_enc)

    # -----------------------------------------------------------------------
    # Helpers / accessors
    # -----------------------------------------------------------------------

    @property
    def feature_names(self) -> list[str]:
        """Ordered list of feature names after encoding (populated after split)."""
        if not self._feature_names:
            raise RuntimeError("Feature names not available yet — call run() first.")
        return self._feature_names

    def summary(
        self,
        X_train: np.ndarray,
        X_test:  np.ndarray,
        y_train: np.ndarray,
        y_test:  np.ndarray,
    ) -> None:
        """Print a concise summary of the processed splits to stdout."""
        print("\n" + "=" * 55)
        print("  DATA PROCESSING SUMMARY")
        print("=" * 55)
        print(f"  Features        : {X_train.shape[1]}")
        print(f"  Train samples   : {len(y_train):,}  "
              f"(churn={y_train.mean()*100:.1f}%)")
        print(f"  Test  samples   : {len(y_test):,}  "
              f"(churn={y_test.mean()*100:.1f}%)")
        print(f"  Train shape     : {X_train.shape}")
        print(f"  Test  shape     : {X_test.shape}")
        print("\n  Feature list:")
        for i, name in enumerate(self.feature_names, 1):
            print(f"    {i:>2}. {name}")
        print("=" * 55 + "\n")
