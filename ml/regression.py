"""Fit the v1 DaysToApprove regression on the 70/30 split.

Season-year dummies use meteorological seasons. Winter_Y is December of Y-1
plus January and February of Y (Winter_2025 = Dec 2024, Jan 2025, Feb 2025).
Approved Date and Signoff Date are left out: they are only known after approval.
"""

import argparse
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.linalg import qr

from baseline import DEFAULT_DATA, OUTCOME, SEED, TEST_FRACTION, load, score_constant, split

BOROUGH_COLUMNS = [
    "Manhattan_Borough",
    "Staten_Borough",
    "Brooklyn_Borough",
    "Queens_Borough",
    "Bronx_Borough",
]
BOROUGH_REFERENCE = "Manhattan_Borough"
REQUIRED = ["File_Standard", "WT_EarthWork", "WT_Foundation"]
SEASON_ORDER = {"Winter": 0, "Spring": 1, "Summer": 2, "Fall": 3}


def season_year(dates):
    month = dates.dt.month
    year = dates.dt.year
    season = pd.Series(index=dates.index, dtype="object")
    season.loc[month.isin([12, 1, 2])] = "Winter"
    season.loc[month.isin([3, 4, 5])] = "Spring"
    season.loc[month.isin([6, 7, 8])] = "Summer"
    season.loc[month.isin([9, 10, 11])] = "Fall"
    label_year = year + (month.eq(12)).astype(int)
    return season + "_" + label_year.astype(str)


def season_key(label):
    name, year = label.split("_")
    return (int(year), SEASON_ORDER[name])


def safe_name(label):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(label)).strip("_")


def prepare(df):
    out = df.copy()
    out["SeasonYear"] = season_year(out["Filing Date"])
    cost = out["Initial_Cost"].astype(float)
    area = out["Floor_Area"].astype(float)
    out["log_Initial_Cost"] = np.log1p(cost.clip(lower=0))
    out["log_Floor_Area"] = np.log1p(area.clip(lower=0))
    return out


def varying(frame, columns):
    kept = []
    dropped = []
    for column in columns:
        if frame[column].nunique(dropna=False) > 1:
            kept.append(column)
        else:
            dropped.append(column)
    return kept, dropped


def design(frame, plain, jobs, seasons):
    pieces = [frame[column].astype(float) for column in plain]
    for job in jobs:
        pieces.append((frame["Job Type"] == job).astype(float).rename(f"Job_{safe_name(job)}"))
    for season in seasons:
        pieces.append((frame["SeasonYear"] == season).astype(float).rename(season))
    X = pd.concat(pieces, axis=1)
    return sm.add_constant(X, has_constant="add")


def independent_columns(X, tol=1e-10):
    """Keep a full-rank set of columns. Identical flags (Shed and SidewalkSheds) otherwise split one coefficient in two."""
    values = X.to_numpy(dtype=float)
    norms = np.linalg.norm(values, axis=0)
    norms[norms == 0] = 1.0
    _q, r, piv = qr(values / norms, mode="economic", pivoting=True)
    diag = np.abs(np.diag(r))
    keep_n = int(np.sum(diag > tol * diag[0])) if diag.size and diag[0] else 0
    keep = [X.columns[i] for i in sorted(piv[:keep_n], key=lambda i: list(X.columns).index(X.columns[i]))]
    dropped = [str(X.columns[i]) for i in piv[keep_n:]]
    return X.loc[:, keep], dropped


def score_predictions(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = y - pred
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "mae_days": float(np.mean(np.abs(err))),
        "rmse_days": float(np.sqrt(np.mean(err ** 2))),
        "median_abs_error_days": float(np.median(np.abs(err))),
        "r2": float(1 - ss_res / ss_tot) if ss_tot else float("nan"),
    }


def fit_regression(data_path=DEFAULT_DATA, seed=SEED, test_fraction=TEST_FRACTION):
    """Fit v1 on the training split and score it against the average-days baseline."""
    df = prepare(load(data_path))
    train, test = split(df, test_fraction, seed)

    work_columns = [column for column in df.columns if column.startswith("WT_")]
    work_kept, work_dropped = varying(train, work_columns)
    for column in ("WT_EarthWork", "WT_Foundation"):
        if column not in work_kept:
            work_kept.append(column)
    work_kept = ["WT_EarthWork", "WT_Foundation"] + [
        column for column in sorted(work_kept) if column not in ("WT_EarthWork", "WT_Foundation")
    ]

    boroughs = [column for column in BOROUGH_COLUMNS if column != BOROUGH_REFERENCE]
    flags = [
        "File_Standard",
        "SubmittedWeekend",
        "Little E",
        "In Compliance with NYCECC",
        "Exempt from NYCECC",
        "Proposed Dwelling Units",
        "log_Initial_Cost",
        "log_Floor_Area",
    ]
    plain = work_kept + boroughs + flags
    rename = {
        "Little E": "Little_E",
        "In Compliance with NYCECC": "NYCECC_Compliant",
        "Exempt from NYCECC": "NYCECC_Exempt",
        "Proposed Dwelling Units": "Proposed_Dwelling_Units",
    }

    job_reference = train["Job Type"].mode().iloc[0]
    jobs = sorted(job for job in train["Job Type"].unique() if job != job_reference)
    season_reference = train["SeasonYear"].mode().iloc[0]
    seasons = sorted(
        (season for season in train["SeasonYear"].unique() if season != season_reference),
        key=season_key,
    )

    X_train = design(train, plain, jobs, seasons).rename(columns=rename)
    X_test = design(test, plain, jobs, seasons).rename(columns=rename)
    X_train, dropped_collinear = independent_columns(X_train)
    X_test = X_test.loc[:, X_train.columns]
    y_train = train[OUTCOME].astype(float)
    y_test = test[OUTCOME].astype(float)
    model = sm.OLS(y_train, X_train).fit()

    train_hat = model.predict(X_train)
    test_hat = model.predict(X_test)
    train_fit = score_predictions(y_train, train_hat)
    test_fit = score_predictions(y_test, test_hat)
    baseline = float(y_train.mean())
    baseline_test = score_constant(y_test, baseline)

    return {
        "data_path": data_path,
        "seed": seed,
        "df": df,
        "train": train,
        "test": test,
        "model": model,
        "work_dropped": work_dropped,
        "dropped_collinear": dropped_collinear,
        "job_reference": job_reference,
        "season_reference": season_reference,
        "train_fit": train_fit,
        "test_fit": test_fit,
        "baseline": baseline,
        "baseline_test": baseline_test,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=argparse.FileType("rb"), default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--test-fraction", type=float, default=TEST_FRACTION)
    args = parser.parse_args()
    data_path = DEFAULT_DATA if args.data is None else args.data.name
    if args.data is not None:
        args.data.close()

    fitted = fit_regression(data_path, args.seed, args.test_fraction)
    df = fitted["df"]
    model = fitted["model"]
    train = fitted["train"]
    test = fitted["test"]
    train_fit = fitted["train_fit"]
    test_fit = fitted["test_fit"]
    baseline = fitted["baseline"]
    baseline_test = fitted["baseline_test"]
    filed = df["Filing Date"]
    both_review = int(((df["File_Standard"] + df["File_Certif"]) == 1).sum())
    print(f"Source: {data_path}")
    print(f"Rows: {len(df):,}    Period: {filed.min():%Y-%m-%d} to {filed.max():%Y-%m-%d}")
    print(f"Split: seed {args.seed}, train {len(train):,} / test {len(test):,}")
    print(f"Outcome: {OUTCOME}")
    print("Left out: Approved Date, Signoff Date (known only after approval),")
    print("          File_Certif (the other filing-review flag), Borough text,")
    print(f"          work types with no variation in training: {', '.join(fitted['work_dropped']) or 'none'}")
    print(f"File_Standard + File_Certif covers every row: {both_review == len(df)} ({both_review:,} of {len(df):,})")
    print("Reference borough: Manhattan (all borough flags 0)")
    print(f"Reference job type: {fitted['job_reference']}")
    print(f"Reference season-year: {fitted['season_reference']}")
    print(f"Dropped as duplicates of another column: {', '.join(fitted['dropped_collinear']) or 'none'}")
    print("Job types:", ", ".join(f"{name} ({count:,})" for name, count in df["Job Type"].value_counts().items()))
    print("Winter_Y = December Y-1, January Y, February Y")
    print()
    print(f"Training R²:          {model.rsquared:.4f}")
    print(f"Training adjusted R²: {model.rsquared_adj:.4f}")
    print(f"Training MAE:         {train_fit['mae_days']:.4f} days")
    print(f"Training RMSE:        {train_fit['rmse_days']:.4f} days")
    print(f"Test R²:              {test_fit['r2']:.4f}")
    print(f"Test MAE:             {test_fit['mae_days']:.4f} days")
    print(f"Test RMSE:            {test_fit['rmse_days']:.4f} days")
    print(f"Baseline test MAE:    {baseline_test['mae_days']:.4f} days (always guess {baseline:.4f})")
    print(f"Baseline test R²:     {baseline_test['r2']:.4f}")
    print(f"Parameters (with intercept): {int(model.df_model) + 1}")
    print()
    print(model.summary().as_text())
    print()
    counts = df["SeasonYear"].value_counts()
    print("Season-year row counts (full data):")
    for label in sorted(counts.index, key=season_key):
        marker = "  reference" if label == fitted["season_reference"] else ""
        print(f"  {label:<16} {counts[label]:>8,}{marker}")


if __name__ == "__main__":
    main()
