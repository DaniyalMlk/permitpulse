# PermitPulse — Frontend integration package

> **In this repo:** the website backend (`backend/main.py`) replaces `api.py` and `frontend_example.js`.
> After running `train.py`, put `permitpulse_histgb.joblib` in the top-level `models/` folder and the
> backend switches to this model automatically.

## What this delivers

- `train.py`: reproduces **HistGB / Teammate base** business-day P10/P50/P90 on the same 2021–2023 training cohort and features as the experiment; stores trained models **and preprocessing** in `models/permitpulse_histgb.joblib`.
- `predictor.py`: prediction function that reuses the exact feature-engineering code and fitted encoder.
- `api.py`: FastAPI `POST /predict` endpoint.
- `frontend_example.js`: example frontend request.
- `model_features.py`: teammate's base feature-generation logic, including deriving `filing_kind` from filing number and engineered cost/building/date features.

**Important:** Trained model binaries are NOT included. The original ~970k-row Parquet data was not attached, so training cannot be completed here. Your teammate must run `train.py` in their environment to create them. Do not deploy before verifying the trained bundle and API together.

## Setup and training

From the directory containing this package, ensure `data/dob_now_clean_columns.parquet` exists (copy or symlink from your original project). Python 3.11 or 3.12 recommended.

```bash
python -m pip install -r requirements.txt
python train.py --data data/dob_now_clean_columns.parquet --output models
uvicorn api:app --reload
```

Check `http://localhost:8000/health` then `http://localhost:8000/docs`.

## API request

`POST http://localhost:8000/predict`, `Content-Type: application/json`:

```json
{
  "filing_date": "2026-11-04",
  "filing_review_type": "Standard Plan Examination",
  "job_type": "Alteration",
  "borough": "Manhattan",
  "filing_kind": "I",
  "initial_cost": 250000,
  "total_construction_floor_area": 1000,
  "building_type": "Mixed Use",
  "proposed_no_of_stories": 5,
  "general_construction_work_type": "YES"
}
```

**`filing_kind` example above is illustrative; use actual historical codes from the source dataset, not invented labels.** In the original notebook, `filing_kind` is the alphabetic segment extracted from `job_filing_number` by the regex `-([A-Za-z]+)(\d+)$`, such as `I` from `...-I1`. The API also accepts `job_filing_number` to derive it. If your frontend has neither, the model will infer `Unknown`, and the deployed accuracy may be substantially worse. Add this field to the form or retrain and re-evaluate a frontend-only feature model.

Other optional fields include `owner_type`, `review_building_code`, `little_e`, `existing_stories`, `existing_height`, `proposed_height`, `existing_dwelling_units`, `proposed_dwelling_units`, `latitude`, `longitude`, `postcode`, `nta`, `commmunity_board`, `applicant_professional_title`, and any `*_work_type` YES/NO flag. These must be **known at filing time**. Missing values are supported but can reduce accuracy. Filings submitted with only the minimum required inputs were **not** separately evaluated.

Example response shape (numbers are illustrative, not computed):

```json
{
  "unit": "business_days",
  "p10": 2.0,
  "p50": 11.0,
  "p90": 80.0,
  "target_definition": "Business days from filing to approval; excludes filing day",
  "notes": []
}
```

Use P50 as the typical timeline and P10/P90 as the conditional range, but avoid claiming 80% coverage for a subgroup before checking calibration. Do not interpret these as **remaining days on an existing application**; the model predicts total duration beginning at filing.

## Known reported results (NOT newly reproduced in this package)

Results previously supplied from the matched teammate-base **2025 retrospectively observed approved-only holdout**: HistGB base Overall P50 MAE **13.291 business days**, Standard Plan P50 MAE **23.479 business days**; Standard Plan P90 pinball **9.577**, coverage **85.313%**, interval width **71.096 days**. A separately trained squared-error HistGB model reached Overall R² **0.320** but was not selected for the website's P50 display, as its MAE was higher. These are two different loss functions/models. Results may not generalize to pending or unseen submissions; held-out 2025 was already inspected multiple times.

See `models/metrics.json` after training for the locally reproduced results.

## Deployment cautions

- Business days exclude weekends and observed **New York State holidays** from the `holidays` library. That is only a proxy for actual NYC DOB closure dates. Do not promise specific approval dates without reconciling DOB holidays.
- Retrospective snapshots may reflect fields updated after filing; prediction-time data availability must be verified.
- The evaluation excludes unapproved/pending filings, likely understating the wait for some applications.
- Do not blindly expose a local `http://localhost:8000` endpoint to a hosted webpage. Deploy the API and configure CORS for your actual frontend origin. No unrestricted CORS is enabled in this package.
- Save model artifacts in deployment; never fit encoders against a single incoming request. Pin Python and scikit-learn versions for artifact compatibility.
- If selecting or tuning after reviewing 2025, it is no longer an untouched final holdout.
