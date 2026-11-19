"""Score one new filing with the saved v1 coefficients and write the decision.

The new filing is not a training row. Full demolition never varies in the
spreadsheet, so this is a 2-dwelling Brooklyn alteration, which is the
product input that story can actually use.
"""

import math
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
COEF_PATH = HERE / "regression_coefficients.xlsx"
OUT_PATH = HERE / "filing_run.xlsx"
IDLE_COST_PER_DAY = 1000  # assumption, labeled as such in the workbook

# One new filing the product would see. Alteration and Spring 2026 are the
# reference job type and a real season dummy. Unlisted flags are 0.
FILING = {
    "Project": "2-dwelling alteration in Brooklyn",
    "Borough": "Brooklyn",
    "Job type": "Alteration",
    "Filing date": "2026-04-15",
    "Season-year": "Spring_2026",
    "Review": "Standard plan examination",
    "Work types on": "General construction, structural",
    "Initial cost": 80000,
    "Floor area (sq ft)": 1200,
    "Proposed dwelling units": 2,
    "NYCECC": "Compliant",
    "Little E": "No",
    "Filed on a weekend": "No",
}

FEATURES = {
    "const": 1,
    "Brooklyn_Borough": 1,
    "File_Standard": 1,
    "WT_GeneralConstruction": 1,
    "WT_Structural": 1,
    "NYCECC_Compliant": 1,
    "Proposed_Dwelling_Units": 2,
    "log_Initial_Cost": math.log1p(80000),
    "log_Floor_Area": math.log1p(1200),
    "Spring_2026": 1,
}


def main():
    coefficients = pd.read_excel(COEF_PATH, sheet_name="Coefficients")
    summary = pd.read_excel(COEF_PATH, sheet_name="Fit summary")
    baseline = float(
        summary.loc[summary["Item"] == "Baseline prediction (training mean, days)", "Value"].iloc[0]
    )
    lookup = coefficients.set_index("Variable")
    missing = [name for name in FEATURES if name not in lookup.index]
    if missing:
        raise SystemExit(f"Coefficients file has no {missing}")

    rows = []
    for name, value in FEATURES.items():
        coefficient = float(lookup.loc[name, "Coefficient"])
        rows.append(
            {
                "Term": name,
                "Input": value,
                "Coefficient": coefficient,
                "Days from this term": value * coefficient,
            }
        )
    score = pd.DataFrame(rows)
    predicted = float(score["Days from this term"].sum())
    gap = predicted - baseline
    if gap > 0:
        decision = "Hold hiring and equipment delivery until the model's date"
        baseline_loss = gap * IDLE_COST_PER_DAY
    else:
        decision = "Hire and deliver equipment earlier than the 19-day average"
        baseline_loss = 0.0

    filing = pd.DataFrame({"Field": list(FILING), "Value": list(FILING.values())})
    decision_sheet = pd.DataFrame(
        [
            ("Baseline (training average)", round(baseline, 2)),
            ("Model prediction (days)", round(predicted, 2)),
            ("Model minus baseline (days)", round(gap, 2)),
            ("Decision", decision),
            ("Idle cost per day (assumption)", IDLE_COST_PER_DAY),
            ("Cost of staffing to the average, if the model date is right", round(baseline_loss, 2)),
            ("Cost of staffing to the model date, if the model date is right", 0),
        ],
        columns=["Item", "Value"],
    )

    with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
        filing.to_excel(writer, sheet_name="New filing", index=False)
        score.to_excel(writer, sheet_name="Score", index=False)
        decision_sheet.to_excel(writer, sheet_name="Decision", index=False)
        for sheet_name in ("New filing", "Score", "Decision"):
            sheet = writer.sheets[sheet_name]
            sheet.freeze_panes = "A2"
            for column in sheet.columns:
                width = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column)
                sheet.column_dimensions[column[0].column_letter].width = min(max(width + 2, 14), 78)

    print(f"Baseline {baseline:.2f} days")
    print(f"Prediction {predicted:.2f} days")
    print(f"Gap {gap:.2f} days")
    print(decision)
    print(f"Baseline idle cost ${baseline_loss:,.0f} at ${IDLE_COST_PER_DAY:,.0f}/day")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
