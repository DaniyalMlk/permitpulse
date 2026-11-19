"""Write the v1 coefficient table to Excel.

Run from the repo root:

    python ml/export_coefficients.py
"""

from pathlib import Path

import pandas as pd

from regression import fit_regression

OUT_PATH = Path(__file__).resolve().parent / "regression_coefficients.xlsx"


def main():
    fitted = fit_regression()
    model = fitted["model"]
    ci = model.conf_int(alpha=0.05)
    coefficients = pd.DataFrame(
        {
            "Variable": model.params.index.astype(str),
            "Coefficient": model.params.to_numpy(dtype=float),
            "Std. Error": model.bse.to_numpy(dtype=float),
            "t": model.tvalues.to_numpy(dtype=float),
            "P>|t|": model.pvalues.to_numpy(dtype=float),
            "95% CI low": ci.iloc[:, 0].to_numpy(dtype=float),
            "95% CI high": ci.iloc[:, 1].to_numpy(dtype=float),
        }
    )
    summary = pd.DataFrame(
        [
            ("Outcome", "DaysToApprove"),
            ("Training rows", len(fitted["train"])),
            ("Test rows", len(fitted["test"])),
            ("Seed", fitted["seed"]),
            ("Training R²", model.rsquared),
            ("Training adjusted R²", model.rsquared_adj),
            ("Training MAE (days)", fitted["train_fit"]["mae_days"]),
            ("Test R²", fitted["test_fit"]["r2"]),
            ("Test MAE (days)", fitted["test_fit"]["mae_days"]),
            ("Test RMSE (days)", fitted["test_fit"]["rmse_days"]),
            ("Baseline prediction (training mean, days)", fitted["baseline"]),
            ("Baseline test MAE (days)", fitted["baseline_test"]["mae_days"]),
            ("Baseline test R²", fitted["baseline_test"]["r2"]),
            ("Reference borough", "Manhattan"),
            ("Reference job type", fitted["job_reference"]),
            ("Reference season-year", fitted["season_reference"]),
            ("Dropped duplicate column", ", ".join(fitted["dropped_collinear"]) or "none"),
            ("Winter definition", "Winter_Y = December Y-1, January Y, February Y"),
        ],
        columns=["Item", "Value"],
    )

    with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
        coefficients.to_excel(writer, sheet_name="Coefficients", index=False)
        summary.to_excel(writer, sheet_name="Fit summary", index=False)
        for sheet_name in ("Coefficients", "Fit summary"):
            sheet = writer.sheets[sheet_name]
            sheet.auto_filter.ref = sheet.dimensions
            sheet.freeze_panes = "A2"
            for column in sheet.columns:
                width = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column)
                sheet.column_dimensions[column[0].column_letter].width = min(max(width + 2, 12), 48)

    print(f"{len(coefficients)} coefficients")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
