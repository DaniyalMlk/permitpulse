"""Split the permit filings and score the average-days baseline.

With no model, the product tells every filing the average DaysToApprove.
That average is computed on the training 70% only, then scored on the
held-out 30%. Every printed number comes from this script.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUTCOME = "DaysToApprove"
TEST_FRACTION = 0.30
SEED = 42
DEFAULT_DATA = Path(r"C:\Users\Expedition\Documents\PermitPulseData2.xlsx")


def load(path):
    df = pd.read_excel(path, sheet_name="Regression_Data")
    if OUTCOME not in df.columns:
        raise SystemExit(f"{path} has no {OUTCOME} column")
    missing = int(df[OUTCOME].isna().sum())
    if missing:
        raise SystemExit(f"{missing} rows are missing {OUTCOME}")
    return df


def split(df, test_fraction=TEST_FRACTION, seed=SEED):
    """Randomly set aside test_fraction of the rows. Under 30 rows, keep all for training."""
    n = len(df)
    if n < 30:
        return df.copy(), df.iloc[0:0].copy()
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_test = int(np.floor(n * test_fraction))
    test_pos = order[:n_test]
    train_pos = order[n_test:]
    return df.iloc[train_pos].copy(), df.iloc[test_pos].copy()


def score_constant(y, prediction):
    """Error of predicting one constant number for every row."""
    y = np.asarray(y, dtype=float)
    err = y - prediction
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    median_ae = float(np.median(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot else float("nan")
    return {
        "n": int(len(y)),
        "mae_days": mae,
        "rmse_days": rmse,
        "median_abs_error_days": median_ae,
        "r2": r2,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--test-fraction", type=float, default=TEST_FRACTION)
    args = parser.parse_args()

    df = load(args.data)
    train, test = split(df, args.test_fraction, args.seed)
    y_train = train[OUTCOME].to_numpy(dtype=float)
    y_test = test[OUTCOME].to_numpy(dtype=float)

    # The baseline the product uses with no model: the training average.
    baseline = float(np.mean(y_train))
    train_score = score_constant(y_train, baseline)
    test_score = score_constant(y_test, baseline)

    filed = df["Filing Date"]
    print(f"Source: {args.data}")
    print(f"Rows: {len(df):,}   one row per filing")
    print(f"Period: {filed.min():%Y-%m-%d} to {filed.max():%Y-%m-%d}")
    print(f"Split: seed {args.seed}, {1 - args.test_fraction:.0%} train / {args.test_fraction:.0%} test")
    print(f"Train rows: {len(train):,}    Test rows: {len(test):,}")
    print()
    print(f"Baseline prediction: {baseline:.4f} days")
    print(f"  training mean of {OUTCOME} (this is the number the baseline guesses)")
    print(f"  all-rows mean: {df[OUTCOME].mean():.4f} days")
    print(f"  test-set mean: {y_test.mean():.4f} days (not used for the guess)")
    print(f"  training median: {np.median(y_train):.1f} days")
    print()
    print("Efficacy on the test rows, guessing the training average for every filing:")
    print(f"  typical error (MAE): {test_score['mae_days']:.4f} days")
    print(f"  RMSE:                {test_score['rmse_days']:.4f} days")
    print(f"  median absolute error: {test_score['median_abs_error_days']:.4f} days")
    print(f"  R²:                  {test_score['r2']:.4f}")
    print()
    print("Same guess scored on the training rows (R² of the average is 0 by definition):")
    print(f"  typical error (MAE): {train_score['mae_days']:.4f} days")
    print(f"  RMSE:                {train_score['rmse_days']:.4f} days")
    print(f"  R²:                  {train_score['r2']:.4f}")
    return {"baseline_days": baseline, "train": train_score, "test": test_score}


if __name__ == "__main__":
    main()
