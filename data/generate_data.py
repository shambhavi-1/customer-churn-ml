"""
generate_data.py
================
Synthetic dataset generator for the Customer Churn Prediction System.

Purpose
-------
Generates a realistic, labelled telecom customer dataset where churn is
driven by a sigmoid function over business-logic risk factors — not random
assignment. Adds ~3% NaN to selected columns to simulate real-world messiness.

Logic-based churn drivers
--------------------------
  HIGH churn risk  : low tenure, high monthly charges, month-to-month
                     contract, high support calls, low engagement
  LOW  churn risk  : long tenure, yearly contract, high engagement

Usage
-----
    python data/generate_data.py               # 5000 rows, seed 42
    python data/generate_data.py --rows 10000 --seed 7
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_ROWS   = 5_000
SEED     = 42
OUT_PATH = Path("data/churn_dataset.csv")

CONTRACT_TYPES   = ["Month-to-Month", "One Year", "Two Year"]
INTERNET_SERVICES = ["DSL", "Fiber Optic", "No"]
PAYMENT_METHODS  = ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"]

# Columns to receive ~3 % NaN
NULLABLE_COLS = ["total_charges", "number_of_logins", "usage_hours"]
NAN_RATE = 0.03


# ---------------------------------------------------------------------------
# Feature generators
# ---------------------------------------------------------------------------

def generate_base_features(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate all raw feature columns for *n* customers.

    Returns a DataFrame with every column except churn.
    """
    tenure           = rng.integers(0, 73, size=n)                     # 0–72 months
    monthly_charges  = rng.uniform(300, 1200, size=n).round(2)
    total_charges    = (monthly_charges * tenure + rng.uniform(0, 500, size=n)).round(2)

    contract_type    = rng.choice(CONTRACT_TYPES,    size=n,
                                  p=[0.55, 0.30, 0.15])                # realistic skew
    internet_service = rng.choice(INTERNET_SERVICES, size=n,
                                  p=[0.40, 0.45, 0.15])
    payment_method   = rng.choice(PAYMENT_METHODS,   size=n)

    support_calls    = rng.integers(0, 11, size=n)                     # 0–10
    number_of_logins = rng.integers(0, 51, size=n).astype(float)       # 0–50
    usage_hours      = rng.uniform(0, 200, size=n).round(1)            # 0–200

    customer_id = [f"CUST-{i:05d}" for i in range(1, n + 1)]

    return pd.DataFrame({
        "customer_id":      customer_id,
        "tenure":           tenure,
        "monthly_charges":  monthly_charges,
        "total_charges":    total_charges,
        "contract_type":    contract_type,
        "internet_service": internet_service,
        "payment_method":   payment_method,
        "support_calls":    support_calls,
        "number_of_logins": number_of_logins,
        "usage_hours":      usage_hours,
    })


# ---------------------------------------------------------------------------
# Sigmoid-based churn label generator
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def compute_churn_probability(df: pd.DataFrame) -> np.ndarray:
    """
    Build a risk score from business rules, pass through sigmoid, and
    return per-customer churn probabilities in [0, 1].

    Risk score contributions
    ------------------------
    + tenure           : low tenure  → high risk   (negative slope)
    + monthly_charges  : high charges → high risk   (positive slope)
    + contract_type    : M2M = +1.5, One Year = 0, Two Year = −1.5
    + support_calls    : each call adds +0.25 risk
    + engagement       : low logins + low usage → high risk
    """
    n = len(df)

    # --- tenure: 0 months = +2.0, 72 months = −2.0 (linear)
    tenure_score = 2.0 - (df["tenure"] / 72) * 4.0

    # --- monthly charges: 300 = −1.0, 1200 = +1.5
    charge_score = (df["monthly_charges"] - 300) / (1200 - 300) * 2.5 - 1.0

    # --- contract type
    contract_map = {"Month-to-Month": 1.5, "One Year": 0.0, "Two Year": -1.5}
    contract_score = df["contract_type"].map(contract_map).values

    # --- support calls: 0 = 0, 10 = +2.5
    support_score = df["support_calls"] * 0.25

    # --- engagement: combine normalised logins + usage (low = high risk)
    login_norm = df["number_of_logins"].fillna(df["number_of_logins"].median()) / 50
    usage_norm = df["usage_hours"].fillna(df["usage_hours"].median()) / 200
    engagement_score = -(login_norm + usage_norm)   # low engagement → positive risk

    # --- raw risk score (bias −1.0 keeps baseline churn ~20–25 %)
    risk = (
        tenure_score
        + charge_score
        + contract_score
        + support_score
        + engagement_score
        - 2.2
    )

    return _sigmoid(risk)


def generate_churn_labels(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """
    Draw binary churn labels by sampling Bernoulli(p) per customer,
    where p comes from the sigmoid risk model + small Gaussian noise.
    """
    probs = compute_churn_probability(df)
    # Add a little noise so the boundary isn't perfectly sharp
    noisy_probs = np.clip(probs + rng.normal(0, 0.05, len(df)), 0, 1)
    return rng.binomial(1, noisy_probs).astype(int)


# ---------------------------------------------------------------------------
# NaN injection
# ---------------------------------------------------------------------------

def inject_nulls(df: pd.DataFrame, rng: np.random.Generator,
                 columns: list[str], rate: float) -> pd.DataFrame:
    """Randomly set *rate* fraction of values to NaN in each listed column."""
    df = df.copy()
    n = len(df)
    n_nulls = max(1, int(n * rate))
    for col in columns:
        null_idx = rng.choice(df.index, size=n_nulls, replace=False)
        df.loc[null_idx, col] = np.nan
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(rows: int = N_ROWS, seed: int = SEED, output: Path = OUT_PATH) -> None:
    """
    Orchestrate dataset generation and save to *output*.

    Parameters
    ----------
    rows   : Number of synthetic customer records.
    seed   : Random seed for full reproducibility.
    output : Destination CSV path.
    """
    rng = np.random.default_rng(seed)
    logger.info("Generating %d customer records (seed=%d) …", rows, seed)

    # 1. Base features
    df = generate_base_features(rows, rng)

    # 2. Logic-based churn labels
    df["churn"] = generate_churn_labels(df, rng)

    # 3. Inject realistic NaN values
    df = inject_nulls(df, rng, NULLABLE_COLS, NAN_RATE)

    # 4. Save
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    # 5. Summary — flush logging first so output order is correct on Windows
    import sys
    logging.shutdown()          # flush all log handlers
    sys.stdout.flush()

    churn_rate = df["churn"].mean() * 100
    print("\n" + "=" * 45, flush=True)
    print("  DATASET SUMMARY", flush=True)
    print("=" * 45, flush=True)
    print(f"  Shape        : {df.shape[0]:,} rows × {df.shape[1]} columns", flush=True)
    print(f"  Churn rate   : {churn_rate:.2f}%  "
          f"({df['churn'].sum():,} churned / {(df['churn'] == 0).sum():,} retained)",
          flush=True)
    print(f"  Saved        : {output}", flush=True)
    print("\n  NaN counts in nullable columns:", flush=True)
    for col in NULLABLE_COLS:
        n_nan = df[col].isna().sum()
        pct   = n_nan / rows * 100
        print(f"    {col:<22}: {n_nan:>3} NaN  ({pct:.1f}%)", flush=True)
    print("=" * 45 + "\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic telecom churn dataset")
    parser.add_argument("--rows",   type=int,      default=N_ROWS,
                        help="Number of customer records (default: 5000)")
    parser.add_argument("--seed",   type=int,      default=SEED,
                        help="Random seed (default: 42)")
    parser.add_argument("--output", type=Path,     default=OUT_PATH,
                        help="Output CSV path (default: data/churn_dataset.csv)")
    args = parser.parse_args()
    main(rows=args.rows, seed=args.seed, output=args.output)
