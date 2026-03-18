"""
train_model.py
==============
Model training, SMOTE comparison, evaluation, and artefact persistence.

Purpose
-------
Trains three classifiers (Logistic Regression, Random Forest, XGBoost) each
in two variants — WITH SMOTE and WITHOUT SMOTE — evaluates every variant on
the held-out test set, selects the best by ROC-AUC, and saves three artefacts:

    models/best_model.pkl
    models/model_metadata.json
    models/feature_importance.csv

Pipeline design
---------------
Every model is wrapped in an imbalanced-learn Pipeline so that SMOTE
oversampling is applied only to the training fold during cross-validation,
never to the validation fold — preventing data leakage.

    SMOTE variant    :  ImbPipeline([("smote", SMOTE()), ("scaler"*, ...), ("clf", <model>)])
    No-SMOTE variant :  ImbPipeline([("scaler"*, ...), ("clf", <model>)])
    (* scaler included only for Logistic Regression)

Metrics
-------
  ROC-AUC  |  Precision  |  Recall  |  F1-score

Dependencies
------------
    pip install xgboost imbalanced-learn scikit-learn numpy pandas

Usage
-----
    from src.train_model import ModelTrainer

    trainer = ModelTrainer(feature_names=feature_names)
    best    = trainer.train_all(X_train, y_train, X_test, y_test)
"""

import json
import logging
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS_DIR    = Path("models")
RANDOM_STATE  = 42
CV_FOLDS      = 5
MODEL_NAMES   = ["logistic_regression", "random_forest", "xgboost"]
SMOTE_VARIANTS = [False, True]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    """
    Holds one trained pipeline variant with its full evaluation record.

    Attributes
    ----------
    run_id              : Unique key, e.g. "xgboost_smote"
    model_name          : Base model name
    smote               : Whether SMOTE was applied in this variant
    pipeline            : Fitted ImbPipeline (ready for predict_proba)
    cv_roc_auc          : Mean ROC-AUC across CV folds (train data only)
    cv_roc_auc_std      : Std-dev of CV ROC-AUC scores
    test_metrics        : Dict with roc_auc, precision, recall, f1 on held-out test
    feature_importances : Ordered dict feature → importance score (or None)
    """

    run_id:              str
    model_name:          str
    smote:               bool
    pipeline:            Any
    cv_roc_auc:          float
    cv_roc_auc_std:      float
    test_metrics:        Dict[str, float]
    feature_importances: Optional[Dict[str, float]] = None


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _compute_metrics(pipeline: ImbPipeline, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Compute ROC-AUC, Precision, Recall, and F1 for a fitted pipeline.

    ROC-AUC uses raw predicted probabilities.
    Precision / Recall / F1 use a 0.5 decision threshold.
    """
    y_prob = pipeline.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "roc_auc":   round(float(roc_auc_score(y, y_prob)), 4),
        "precision": round(float(precision_score(y, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y, y_pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y, y_pred, zero_division=0)), 4),
    }


def _extract_feature_importances(
    pipeline: ImbPipeline,
    feature_names: List[str],
) -> Optional[Dict[str, float]]:
    """
    Extract feature importance scores from the classifier step of a pipeline.

    Supports:
      - Tree-based models  (feature_importances_  — Random Forest, XGBoost)
      - Linear models      (coef_                 — Logistic Regression)

    Returns ordered dict (feature → importance), sorted descending, or None
    if the vector length does not match feature_names.
    """
    clf = pipeline.steps[-1][1]   # classifier is always the final named step

    if hasattr(clf, "feature_importances_"):
        raw = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        raw = np.abs(clf.coef_[0])
    else:
        logger.warning(
            "Classifier %s has no feature importance attribute.",
            type(clf).__name__,
        )
        return None

    if len(raw) != len(feature_names):
        logger.warning(
            "Importance vector length (%d) != feature_names length (%d). "
            "Skipping feature_importance.csv.",
            len(raw), len(feature_names),
        )
        return None

    return dict(
        sorted(
            zip(feature_names, raw.tolist()),
            key=lambda kv: kv[1],
            reverse=True,
        )
    )


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

def build_pipeline(model_name: str, use_smote: bool, random_state: int) -> ImbPipeline:
    """
    Construct a fully specified imblearn Pipeline for the given model variant.

    All pipelines — with and without SMOTE — use ImbPipeline so that the
    API is identical for cross_validate, fit, and predict.

    Pipeline steps
    --------------
    logistic_regression + SMOTE    :  smote  → scaler → clf
    logistic_regression (no SMOTE) :  scaler → clf
    random_forest + SMOTE          :  smote  → clf
    random_forest (no SMOTE)       :  clf
    xgboost + SMOTE                :  smote  → clf
    xgboost (no SMOTE)             :  clf

    Parameters
    ----------
    model_name   : "logistic_regression" | "random_forest" | "xgboost"
    use_smote    : Prepend SMOTE step when True.
    random_state : Seed for SMOTE, classifiers, and internal randomness.

    Returns
    -------
    ImbPipeline — not yet fitted.

    Raises
    ------
    ValueError : If model_name is not one of the three supported names.
    """
    smote_step = SMOTE(
        sampling_strategy="minority",
        k_neighbors=5,
        random_state=random_state,
    )

    # ── Logistic Regression ────────────────────────────────────────────────
    if model_name == "logistic_regression":
        clf = LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=random_state,
        )
        if use_smote:
            steps = [
                ("smote",  smote_step),
                ("scaler", StandardScaler()),
                ("clf",    clf),
            ]
        else:
            steps = [
                ("scaler", StandardScaler()),
                ("clf",    clf),
            ]

    # ── Random Forest ──────────────────────────────────────────────────────
    elif model_name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        if use_smote:
            steps = [("smote", smote_step), ("clf", clf)]
        else:
            steps = [("clf", clf)]

    # ── XGBoost ────────────────────────────────────────────────────────────
    elif model_name == "xgboost":
        clf = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
        if use_smote:
            steps = [("smote", smote_step), ("clf", clf)]
        else:
            steps = [("clf", clf)]

    else:
        raise ValueError(
            f"Unsupported model_name='{model_name}'. "
            f"Choose from: {MODEL_NAMES}"
        )

    return ImbPipeline(steps=steps)


# ---------------------------------------------------------------------------
# Main trainer class
# ---------------------------------------------------------------------------

class ModelTrainer:
    """
    Trains Logistic Regression, Random Forest, and XGBoost — each with and
    without SMOTE (6 total variants) — then selects the best by test ROC-AUC.

    Evaluation
    ----------
    1. Stratified K-Fold CV on training set   → CV ROC-AUC ± std
    2. Final evaluation on held-out test set  → ROC-AUC, Precision, Recall, F1

    Artefacts saved
    ---------------
    models/best_model.pkl
    models/model_metadata.json
    models/feature_importance.csv

    Parameters
    ----------
    feature_names : Ordered list of feature column names (must match X columns).
    models_dir    : Directory where artefacts will be written.
    random_state  : Global random seed for full reproducibility.
    cv_folds      : Number of stratified CV folds (default 5).
    """

    def __init__(
        self,
        feature_names: List[str],
        models_dir:    Path = MODELS_DIR,
        random_state:  int  = RANDOM_STATE,
        cv_folds:      int  = CV_FOLDS,
    ) -> None:
        self.feature_names = feature_names
        self.models_dir    = Path(models_dir)
        self.random_state  = random_state
        self.cv_folds      = cv_folds
        self.results:      List[TrainingResult] = []

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def train_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test:  np.ndarray,
        y_test:  np.ndarray,
    ) -> TrainingResult:
        """
        Train all 6 model × SMOTE variants, print comparison table,
        save artefacts, and return the best TrainingResult.

        Variants trained (in order)
        ---------------------------
        logistic_regression_no_smote
        logistic_regression_smote
        random_forest_no_smote
        random_forest_smote
        xgboost_no_smote
        xgboost_smote

        Parameters
        ----------
        X_train, y_train : Training split — numpy arrays, already encoded.
        X_test,  y_test  : Held-out test split.

        Returns
        -------
        TrainingResult — highest test ROC-AUC variant, with artefacts saved.
        """
        self.results = []

        for model_name in MODEL_NAMES:
            for use_smote in SMOTE_VARIANTS:
                result = self._train_one(
                    model_name=model_name,
                    use_smote=use_smote,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                )
                self.results.append(result)

        self._print_comparison_table()

        best = max(self.results, key=lambda r: r.test_metrics["roc_auc"])
        logger.info(
            "★  Best model: %s  "
            "(CV ROC-AUC=%.4f ± %.4f  |  Test ROC-AUC=%.4f)",
            best.run_id, best.cv_roc_auc, best.cv_roc_auc_std,
            best.test_metrics["roc_auc"],
        )

        self._save_artefacts(best)
        return best

    # -----------------------------------------------------------------------
    # Private — train one variant
    # -----------------------------------------------------------------------

    def _train_one(
        self,
        model_name: str,
        use_smote:  bool,
        X_train:    np.ndarray,
        y_train:    np.ndarray,
        X_test:     np.ndarray,
        y_test:     np.ndarray,
    ) -> TrainingResult:
        """
        Train a single model × SMOTE variant end-to-end.

        Steps
        -----
        1. Build ImbPipeline via build_pipeline()
        2. Stratified K-Fold CV on training data (roc_auc scoring)
        3. Refit pipeline on the full training set
        4. Evaluate on held-out test set
        5. Extract feature importances

        Returns fully populated TrainingResult.
        """
        run_id = f"{model_name}_{'smote' if use_smote else 'no_smote'}"
        logger.info("▶ Training: %s", run_id)

        # 1. Build pipeline
        pipeline = build_pipeline(model_name, use_smote, self.random_state)

        # 2. Stratified K-Fold cross-validation on training data
        skf = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state,
        )
        cv_scores = cross_validate(
            estimator=pipeline,
            X=X_train,
            y=y_train,
            cv=skf,
            scoring="roc_auc",
            n_jobs=-1,
            return_train_score=False,
        )
        cv_roc_auc     = float(np.mean(cv_scores["test_score"]))
        cv_roc_auc_std = float(np.std(cv_scores["test_score"]))

        # 3. Refit on full training set
        pipeline.fit(X_train, y_train)

        # 4. Evaluate on held-out test set
        test_metrics = _compute_metrics(pipeline, X_test, y_test)

        # 5. Extract feature importances
        importances = _extract_feature_importances(pipeline, self.feature_names)

        logger.info(
            "  CV  ROC-AUC = %.4f ± %.4f  |  "
            "Test AUC=%.4f  Prec=%.4f  Rec=%.4f  F1=%.4f",
            cv_roc_auc, cv_roc_auc_std,
            test_metrics["roc_auc"],
            test_metrics["precision"],
            test_metrics["recall"],
            test_metrics["f1"],
        )

        return TrainingResult(
            run_id              = run_id,
            model_name          = model_name,
            smote               = use_smote,
            pipeline            = pipeline,
            cv_roc_auc          = cv_roc_auc,
            cv_roc_auc_std      = cv_roc_auc_std,
            test_metrics        = test_metrics,
            feature_importances = importances,
        )

    # -----------------------------------------------------------------------
    # Artefact persistence
    # -----------------------------------------------------------------------

    def _save_artefacts(self, best: TrainingResult) -> None:
        """
        Persist three artefacts for the best model:

        1. models/best_model.pkl          — pickled ImbPipeline
        2. models/model_metadata.json     — best model info + all 6-run comparison
        3. models/feature_importance.csv  — feature importance scores, sorted desc
        """
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # ── 1. best_model.pkl ──────────────────────────────────────────────
        model_path = self.models_dir / "best_model.pkl"
        with open(model_path, "wb") as fh:
            pickle.dump(best.pipeline, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved → %s", model_path)

        # ── 2. model_metadata.json ─────────────────────────────────────────
        all_runs_summary = [
            {
                "run_id":         r.run_id,
                "model_name":     r.model_name,
                "smote":          r.smote,
                "cv_roc_auc":     round(r.cv_roc_auc, 4),
                "cv_roc_auc_std": round(r.cv_roc_auc_std, 4),
                "test_roc_auc":   r.test_metrics["roc_auc"],
                "test_precision": r.test_metrics["precision"],
                "test_recall":    r.test_metrics["recall"],
                "test_f1":        r.test_metrics["f1"],
            }
            for r in self.results
        ]

        metadata = {
            "trained_at":   datetime.now(timezone.utc).isoformat(),
            "random_state": self.random_state,
            "cv_folds":     self.cv_folds,
            "best_model": {
                "run_id":         best.run_id,
                "model_name":     best.model_name,
                "smote":          best.smote,
                "cv_roc_auc":     round(best.cv_roc_auc, 4),
                "cv_roc_auc_std": round(best.cv_roc_auc_std, 4),
                "test_roc_auc":   best.test_metrics["roc_auc"],
                "test_precision": best.test_metrics["precision"],
                "test_recall":    best.test_metrics["recall"],
                "test_f1":        best.test_metrics["f1"],
            },
            "feature_names": self.feature_names,
            "all_runs":      all_runs_summary,
        }

        meta_path = self.models_dir / "model_metadata.json"
        with open(meta_path, "w") as fh:
            json.dump(metadata, fh, indent=2)
        logger.info("Saved → %s", meta_path)

        # ── 3. feature_importance.csv ──────────────────────────────────────
        if best.feature_importances:
            fi_df = (
                pd.DataFrame(
                    list(best.feature_importances.items()),
                    columns=["feature", "importance"],
                )
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )
            fi_path = self.models_dir / "feature_importance.csv"
            fi_df.to_csv(fi_path, index=False)
            logger.info("Saved → %s", fi_path)
        else:
            logger.warning(
                "No feature importances for '%s'. "
                "feature_importance.csv not written.", best.model_name,
            )

    # -----------------------------------------------------------------------
    # Reporting
    # -----------------------------------------------------------------------

    def _print_comparison_table(self) -> None:
        """
        Print a formatted side-by-side comparison of all 6 trained variants
        sorted by test ROC-AUC descending. Best row is marked with ★.
        """
        sorted_results = sorted(
            self.results,
            key=lambda r: r.test_metrics["roc_auc"],
            reverse=True,
        )
        best_run_id = sorted_results[0].run_id
        sep = "=" * 94

        print(f"\n{sep}")
        print(
            f"  {'#':<3} {'RUN ID':<38} {'SMOTE':<6} "
            f"{'CV AUC':>8} {'±STD':>6} {'TEST AUC':>9} "
            f"{'PREC':>7} {'REC':>7} {'F1':>7}"
        )
        print("─" * 94)

        for rank, r in enumerate(sorted_results, start=1):
            star        = "★" if r.run_id == best_run_id else " "
            smote_label = "Yes" if r.smote else "No"
            print(
                f"{star} {rank:<3} {r.run_id:<38} {smote_label:<6} "
                f"{r.cv_roc_auc:>8.4f} {r.cv_roc_auc_std:>6.4f} "
                f"{r.test_metrics['roc_auc']:>9.4f} "
                f"{r.test_metrics['precision']:>7.4f} "
                f"{r.test_metrics['recall']:>7.4f} "
                f"{r.test_metrics['f1']:>7.4f}"
            )

        print(f"{sep}\n")

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    def load_best_model(self, models_dir: Optional[Path] = None) -> ImbPipeline:
        """
        Load and return the best model pipeline from disk.

        Parameters
        ----------
        models_dir : Override instance models_dir if provided.

        Returns
        -------
        Fitted ImbPipeline.

        Raises
        ------
        FileNotFoundError : If best_model.pkl does not exist.
        """
        path = (Path(models_dir) if models_dir else self.models_dir) / "best_model.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"best_model.pkl not found at '{path}'. "
                "Run train_all() first."
            )
        with open(path, "rb") as fh:
            pipeline = pickle.load(fh)
        logger.info("Loaded best model from %s", path)
        return pipeline
