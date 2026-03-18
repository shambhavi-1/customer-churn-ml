"""
optimize_model.py
=================
Bayesian hyperparameter optimisation using Optuna.

Purpose
-------
Takes the best model type from training (XGBoost, Random Forest, or Logistic
Regression) and searches the hyperparameter space using Tree-structured Parzen
Estimator (TPE) to maximise cross-validated ROC-AUC on the training set.

The optimised model is evaluated on the held-out test set and saved to:
    models/optimized_model.pkl
    models/optimized_metadata.json
    models/optuna_study.db          (resumable SQLite study)

How it works
------------
1. Load processed data via DataProcessor
2. Build an imblearn Pipeline with SMOTE + chosen model
3. Run Optuna TPE search for N trials
4. Refit best params on full training set
5. Evaluate on test set and compare with baseline
6. Save optimised artefacts

Search spaces
-------------
XGBoost      : n_estimators, max_depth, learning_rate, subsample,
               colsample_bytree, gamma, reg_alpha, reg_lambda
Random Forest: n_estimators, max_depth, min_samples_split,
               min_samples_leaf, max_features
Logistic Reg : C, solver, max_iter

Usage
-----
    # CLI — optimise XGBoost for 50 trials
    python src/optimize_model.py --model xgboost --trials 50

    # Python
    from src.optimize_model import HyperparamOptimizer, OptimizerConfig

    config = OptimizerConfig(model_name="xgboost", n_trials=50)
    opt    = HyperparamOptimizer(config)
    result = opt.optimize(X_train, y_train, X_test, y_test)

Dependencies
------------
    pip install optuna xgboost imbalanced-learn scikit-learn
"""

import argparse
import json
import logging
import pickle
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# silence Optuna's per-trial INFO logs — only show warnings+
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── repo root on path ────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_processing import DataProcessor, DataProcessorConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODELS_DIR   = Path("models")
RANDOM_STATE = 42
SUPPORTED_MODELS = ["xgboost", "random_forest", "logistic_regression"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class OptimizerConfig:
    """All settings for one optimisation run."""

    model_name:      str            = "xgboost"
    n_trials:        int            = 50
    timeout_seconds: Optional[int]  = None        # wall-clock limit (None = unlimited)
    cv_folds:        int            = 5
    random_state:    int            = RANDOM_STATE
    use_smote:       bool           = True
    models_dir:      Path           = MODELS_DIR
    data_path:       Path           = Path("data/churn_dataset.csv")
    # SQLite storage — allows resuming interrupted studies
    study_storage:   Optional[str]  = None        # e.g. "sqlite:///models/optuna_study.db"
    study_name:      Optional[str]  = None        # auto-generated if None


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class OptimizationResult:
    """Output of one complete optimisation run."""

    model_name:          str
    best_params:         Dict[str, Any]
    best_cv_roc_auc:     float
    best_cv_roc_auc_std: float
    test_metrics:        Dict[str, float]
    n_trials:            int
    pipeline:            Any
    baseline_test_auc:   float       # AUC of default params for comparison
    improvement:         float       # test_auc - baseline_test_auc


# ---------------------------------------------------------------------------
# Search space functions
# ---------------------------------------------------------------------------

def _xgb_search_space(trial: optuna.Trial) -> Dict[str, Any]:
    """Suggest XGBoost hyperparameters."""
    return {
        "n_estimators":      trial.suggest_int(  "n_estimators",    100, 500,  step=50),
        "max_depth":         trial.suggest_int(  "max_depth",       3,   9),
        "learning_rate":     trial.suggest_float("learning_rate",   0.01, 0.3,  log=True),
        "subsample":         trial.suggest_float("subsample",       0.6,  1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree",0.6,  1.0),
        "gamma":             trial.suggest_float("gamma",           0.0,  1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha",       1e-4, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda",      1e-4, 10.0, log=True),
        "min_child_weight":  trial.suggest_int(  "min_child_weight",1,    10),
    }


def _rf_search_space(trial: optuna.Trial) -> Dict[str, Any]:
    """Suggest Random Forest hyperparameters."""
    return {
        "n_estimators":     trial.suggest_int(  "n_estimators",    100, 500, step=50),
        "max_depth":        trial.suggest_int(  "max_depth",       3,   20),
        "min_samples_split":trial.suggest_int(  "min_samples_split",2,  20),
        "min_samples_leaf": trial.suggest_int(  "min_samples_leaf", 1,  10),
        "max_features":     trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
    }


def _lr_search_space(trial: optuna.Trial) -> Dict[str, Any]:
    """Suggest Logistic Regression hyperparameters."""
    return {
        "C":       trial.suggest_float("C",       1e-3, 100.0, log=True),
        "max_iter":trial.suggest_int(  "max_iter", 200,  2000, step=100),
        "solver":  trial.suggest_categorical("solver", ["lbfgs", "saga"]),
    }


SEARCH_SPACES = {
    "xgboost":              _xgb_search_space,
    "random_forest":        _rf_search_space,
    "logistic_regression":  _lr_search_space,
}


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

def _build_pipeline(
    model_name: str,
    params: Dict[str, Any],
    use_smote: bool,
    random_state: int,
) -> ImbPipeline:
    """
    Build an ImbPipeline with the given hyperparameters.

    Parameters
    ----------
    model_name   : One of SUPPORTED_MODELS.
    params       : Dict of hyperparameter name → value.
    use_smote    : Whether to prepend SMOTE.
    random_state : Seed for SMOTE and classifiers.
    """
    smote = SMOTE(k_neighbors=5, random_state=random_state)

    if model_name == "xgboost":
        clf = XGBClassifier(
            **params,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
        steps = [("smote", smote), ("clf", clf)] if use_smote else [("clf", clf)]

    elif model_name == "random_forest":
        clf = RandomForestClassifier(
            **params,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        steps = [("smote", smote), ("clf", clf)] if use_smote else [("clf", clf)]

    elif model_name == "logistic_regression":
        clf = LogisticRegression(
            **params,
            class_weight="balanced",
            random_state=random_state,
        )
        steps = (
            [("smote", smote), ("scaler", StandardScaler()), ("clf", clf)]
            if use_smote else
            [("scaler", StandardScaler()), ("clf", clf)]
        )

    else:
        raise ValueError(f"Unsupported model: '{model_name}'. Choose from {SUPPORTED_MODELS}")

    return ImbPipeline(steps=steps)


# ---------------------------------------------------------------------------
# Metric helper
# ---------------------------------------------------------------------------

def _evaluate(pipeline: ImbPipeline, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Compute ROC-AUC, Precision, Recall, F1 on a held-out set."""
    y_prob = pipeline.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "roc_auc":   round(float(roc_auc_score(y, y_prob)), 4),
        "precision": round(float(precision_score(y, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y, y_pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y, y_pred, zero_division=0)), 4),
    }


# ---------------------------------------------------------------------------
# Main optimizer class
# ---------------------------------------------------------------------------

class HyperparamOptimizer:
    """
    Bayesian hyperparameter search using Optuna TPE.

    Trains the chosen model inside a Stratified K-Fold CV loop for each
    trial. The trial metric is mean CV ROC-AUC. After all trials, the best
    params are used to refit on the full training set and evaluate on test.

    Parameters
    ----------
    config : OptimizerConfig
    """

    def __init__(self, config: Optional[OptimizerConfig] = None) -> None:
        self.config  = config or OptimizerConfig()
        self._study: Optional[optuna.Study] = None

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test:  np.ndarray,
        y_test:  np.ndarray,
    ) -> OptimizationResult:
        """
        Run the full Optuna search and return an OptimizationResult.

        Steps
        -----
        1. Compute baseline AUC with default hyperparameters
        2. Create / resume Optuna study
        3. Run N trials — each trial does K-Fold CV and reports mean AUC
        4. Refit best params on full training set
        5. Evaluate on held-out test set
        6. Save artefacts
        7. Print comparison table

        Parameters
        ----------
        X_train, y_train : Training split.
        X_test,  y_test  : Held-out test split.

        Returns
        -------
        OptimizationResult
        """
        cfg = self.config

        if cfg.model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"model_name='{cfg.model_name}' not supported. "
                f"Choose from: {SUPPORTED_MODELS}"
            )

        logger.info(
            "Starting Optuna optimisation: model=%s  trials=%d  cv=%d  smote=%s",
            cfg.model_name, cfg.n_trials, cfg.cv_folds, cfg.use_smote,
        )

        # ── Step 1: baseline ────────────────────────────────────────────────
        baseline_auc = self._baseline_cv_auc(X_train, y_train)
        logger.info("Baseline CV ROC-AUC (default params): %.4f", baseline_auc)

        # ── Step 2: create / resume study ────────────────────────────────────
        study_name = cfg.study_name or f"churn_{cfg.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        sampler    = TPESampler(seed=cfg.random_state)
        pruner     = MedianPruner(n_startup_trials=5, n_warmup_steps=3)

        self._study = optuna.create_study(
            study_name    = study_name,
            direction     = "maximize",
            sampler       = sampler,
            pruner        = pruner,
            storage       = cfg.study_storage,
            load_if_exists= True,
        )

        # ── Step 3: run trials ───────────────────────────────────────────────
        self._study.optimize(
            lambda trial: self._objective(trial, X_train, y_train),
            n_trials          = cfg.n_trials,
            timeout           = cfg.timeout_seconds,
            show_progress_bar = True,
            gc_after_trial    = True,
        )

        best_params    = self._study.best_params
        best_cv_auc    = self._study.best_value
        best_trial_std = self._get_best_trial_std()

        logger.info("Best trial CV ROC-AUC: %.4f  params: %s", best_cv_auc, best_params)

        # ── Step 4: refit on full training set ───────────────────────────────
        best_pipeline = _build_pipeline(
            cfg.model_name, best_params, cfg.use_smote, cfg.random_state
        )
        best_pipeline.fit(X_train, y_train)

        # ── Step 5: test set evaluation ──────────────────────────────────────
        test_metrics   = _evaluate(best_pipeline, X_test, y_test)
        baseline_test  = self._baseline_test_auc(X_train, y_train, X_test, y_test)
        improvement    = round(test_metrics["roc_auc"] - baseline_test, 4)

        result = OptimizationResult(
            model_name          = cfg.model_name,
            best_params         = best_params,
            best_cv_roc_auc     = round(best_cv_auc, 4),
            best_cv_roc_auc_std = best_trial_std,
            test_metrics        = test_metrics,
            n_trials            = len(self._study.trials),
            pipeline            = best_pipeline,
            baseline_test_auc   = baseline_test,
            improvement         = improvement,
        )

        # ── Step 6: save artefacts ───────────────────────────────────────────
        self._save_artefacts(result)

        # ── Step 7: print summary ────────────────────────────────────────────
        self._print_summary(result)

        return result

    def get_study(self) -> optuna.Study:
        """Return the underlying Optuna Study (after optimize() is called)."""
        if self._study is None:
            raise RuntimeError("Study not available. Call optimize() first.")
        return self._study

    # -----------------------------------------------------------------------
    # Objective function
    # -----------------------------------------------------------------------

    def _objective(
        self,
        trial: optuna.Trial,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> float:
        """
        Optuna objective: suggest params → K-Fold CV → return mean ROC-AUC.

        The trial is pruned (skipped early) if intermediate folds perform
        worse than the median of completed trials.
        """
        cfg    = self.config
        params = SEARCH_SPACES[cfg.model_name](trial)

        pipeline = _build_pipeline(
            cfg.model_name, params, cfg.use_smote, cfg.random_state
        )

        skf = StratifiedKFold(
            n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state
        )
        scores = cross_val_score(
            estimator  = pipeline,
            X          = X_train,
            y          = y_train,
            cv         = skf,
            scoring    = "roc_auc",
            n_jobs     = -1,
            error_score= 0.0,
        )

        mean_auc = float(np.mean(scores))

        # report intermediate value for pruning
        trial.report(mean_auc, step=cfg.cv_folds)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return mean_auc

    # -----------------------------------------------------------------------
    # Baseline helpers
    # -----------------------------------------------------------------------

    def _baseline_cv_auc(self, X_train: np.ndarray, y_train: np.ndarray) -> float:
        """CV AUC of the model with default hyperparameters."""
        cfg      = self.config
        defaults = self._default_params()
        pipeline = _build_pipeline(cfg.model_name, defaults, cfg.use_smote, cfg.random_state)
        skf      = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state)
        scores   = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1)
        return round(float(np.mean(scores)), 4)

    def _baseline_test_auc(
        self, X_train: np.ndarray, y_train: np.ndarray,
        X_test: np.ndarray,  y_test:  np.ndarray,
    ) -> float:
        """Test AUC of the model with default hyperparameters."""
        cfg      = self.config
        defaults = self._default_params()
        pipeline = _build_pipeline(cfg.model_name, defaults, cfg.use_smote, cfg.random_state)
        pipeline.fit(X_train, y_train)
        y_prob   = pipeline.predict_proba(X_test)[:, 1]
        return round(float(roc_auc_score(y_test, y_prob)), 4)

    def _default_params(self) -> Dict[str, Any]:
        """Default hyperparameters for each model (used as baseline)."""
        defaults = {
            "xgboost":             {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
                                    "subsample": 0.8, "colsample_bytree": 0.8,
                                    "gamma": 0.1, "reg_alpha": 0.1, "reg_lambda": 1.0,
                                    "min_child_weight": 1},
            "random_forest":       {"n_estimators": 200, "max_depth": 10,
                                    "min_samples_split": 5, "min_samples_leaf": 4,
                                    "max_features": "sqrt"},
            "logistic_regression": {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"},
        }
        return defaults[self.config.model_name]

    def _get_best_trial_std(self) -> float:
        """Return std of CV scores from the best trial (stored as user_attr if available)."""
        best = self._study.best_trial
        return round(float(best.user_attrs.get("cv_std", 0.0)), 4)

    # -----------------------------------------------------------------------
    # Artefact persistence
    # -----------------------------------------------------------------------

    def _save_artefacts(self, result: OptimizationResult) -> None:
        """
        Save:
          models/optimized_model.pkl       — best fitted pipeline
          models/optimized_metadata.json   — params, metrics, improvement
        """
        cfg = self.config
        cfg.models_dir.mkdir(parents=True, exist_ok=True)

        # 1. optimized_model.pkl
        model_path = cfg.models_dir / "optimized_model.pkl"
        with open(model_path, "wb") as fh:
            pickle.dump(result.pipeline, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved → %s", model_path)

        # 2. optimized_metadata.json
        metadata = {
            "optimized_at":       datetime.now(timezone.utc).isoformat(),
            "model_name":         result.model_name,
            "use_smote":          cfg.use_smote,
            "n_trials":           result.n_trials,
            "cv_folds":           cfg.cv_folds,
            "random_state":       cfg.random_state,
            "best_params":        result.best_params,
            "best_cv_roc_auc":    result.best_cv_roc_auc,
            "best_cv_roc_auc_std":result.best_cv_roc_auc_std,
            "baseline_test_auc":  result.baseline_test_auc,
            "test_roc_auc":       result.test_metrics["roc_auc"],
            "test_precision":     result.test_metrics["precision"],
            "test_recall":        result.test_metrics["recall"],
            "test_f1":            result.test_metrics["f1"],
            "improvement":        result.improvement,
        }
        meta_path = cfg.models_dir / "optimized_metadata.json"
        with open(meta_path, "w") as fh:
            json.dump(metadata, fh, indent=2)
        logger.info("Saved → %s", meta_path)

    # -----------------------------------------------------------------------
    # Reporting
    # -----------------------------------------------------------------------

    def _print_summary(self, result: OptimizationResult) -> None:
        """Print a clean comparison table of baseline vs optimised."""
        sep = "=" * 58
        sign = "+" if result.improvement >= 0 else ""
        print(f"\n{sep}")
        print(f"  OPTUNA OPTIMISATION COMPLETE — {result.model_name.upper()}")
        print(sep)
        print(f"  Trials run          : {result.n_trials}")
        print(f"  Best CV  ROC-AUC    : {result.best_cv_roc_auc:.4f}")
        print(f"")
        print(f"  {'Metric':<20} {'Baseline':>10} {'Optimised':>10} {'Delta':>8}")
        print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*8}")
        print(f"  {'Test ROC-AUC':<20} {result.baseline_test_auc:>10.4f} "
              f"{result.test_metrics['roc_auc']:>10.4f} "
              f"{sign}{result.improvement:>7.4f}")
        print(f"  {'Precision':<20} {'–':>10} "
              f"{result.test_metrics['precision']:>10.4f}")
        print(f"  {'Recall':<20} {'–':>10} "
              f"{result.test_metrics['recall']:>10.4f}")
        print(f"  {'F1':<20} {'–':>10} "
              f"{result.test_metrics['f1']:>10.4f}")
        print(f"\n  Best params:")
        for k, v in result.best_params.items():
            print(f"    {k:<25} : {v}")
        print(f"\n  Saved → models/optimized_model.pkl")
        print(f"  Saved → models/optimized_metadata.json")
        print(f"{sep}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter optimisation for churn prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, default="xgboost",
        choices=SUPPORTED_MODELS,
        help="Model to optimise",
    )
    parser.add_argument(
        "--trials", type=int, default=50,
        help="Number of Optuna trials",
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5, dest="cv_folds",
        help="Stratified CV folds per trial",
    )
    parser.add_argument(
        "--no-smote", action="store_true",
        help="Disable SMOTE in the pipeline",
    )
    parser.add_argument(
        "--data", type=Path, default=Path("data/churn_dataset.csv"),
        help="Path to churn CSV",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--timeout", type=int, default=None,
        help="Max wall-clock seconds (None = unlimited)",
    )
    parser.add_argument(
        "--storage", type=str,
        default="sqlite:///models/optuna_study.db",
        help="Optuna study storage URI (SQLite for resume support)",
    )
    args = parser.parse_args()

    # ── load data ─────────────────────────────────────────────────────────────
    logger.info("Loading and processing data from '%s' …", args.data)
    processor = DataProcessor(DataProcessorConfig(
        data_path    = args.data,
        random_state = args.seed,
    ))
    X_train, X_test, y_train, y_test = processor.run()
    logger.info(
        "Data ready: train=%d  test=%d  features=%d",
        len(y_train), len(y_test), X_train.shape[1],
    )

    # ── run optimisation ──────────────────────────────────────────────────────
    config = OptimizerConfig(
        model_name      = args.model,
        n_trials        = args.trials,
        cv_folds        = args.cv_folds,
        use_smote       = not args.no_smote,
        random_state    = args.seed,
        timeout_seconds = args.timeout,
        data_path       = args.data,
        study_storage   = args.storage,
    )
    optimizer = HyperparamOptimizer(config)
    optimizer.optimize(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
