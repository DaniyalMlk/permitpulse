"""Features matching the team's HistGB baseline experiment; freeze with the trained model."""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
BASE_CAT = ["borough", "job_type", "filing_review_type", "building_type"]
EXTRA_CAT = ["review_building_code", "little_e", "owner_type", "commmunity_board", "postcode", "nta", "applicant_professional_title", "filing_kind"]
NUM_RAW = ["initial_cost", "total_construction_floor_area", "existing_stories", "proposed_no_of_stories", "existing_height", "proposed_height", "existing_dwelling_units", "proposed_dwelling_units", "latitude", "longitude"]
EXTRA_FIELDS = ["job_description", "applicant_license", "work_on_floor", "bin", "in_compliance_with_nycecc", "exempt_from_nycecc"]
CATEGORICAL = BASE_CAT + EXTRA_CAT

def parse_date_column(series):
    raw = series.astype("string")
    result = pd.to_datetime(raw, format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    missing = result.isna() & raw.fillna("").str.strip().ne("")
    if missing.any():
        result.loc[missing] = pd.to_datetime(raw.loc[missing], format="mixed", errors="coerce")
    return result.astype("datetime64[ns]")

def load_all(path):
    fields = pq.ParquetFile(path).schema.names
    work_cols = [c for c in fields if c.endswith("_work_type")]
    needed = list(dict.fromkeys(BASE_CAT + [c for c in EXTRA_CAT if c != "filing_kind"]
        + NUM_RAW + work_cols + EXTRA_FIELDS
        + ["job_filing_number", "filing_date", "approved_date", "filing_status"]))
    raw = pd.read_parquet(path, columns=needed)
    for col in ["filing_date", "approved_date"]:
        raw[col] = parse_date_column(raw[col])
    import holidays
    year_min = int(raw.filing_date.dt.year.min()) if raw.filing_date.notna().any() else 2016
    year_max = int(max(raw.approved_date.dt.year.max(), raw.filing_date.dt.year.max())) if raw.approved_date.notna().any() else 2027
    ny_holidays = np.array(sorted(holidays.US(subdiv="NY", years=range(year_min, year_max + 2), observed=True).keys()), dtype="datetime64[D]")
    starts = raw.filing_date.to_numpy(dtype="datetime64[D]")
    ends = raw.approved_date.to_numpy(dtype="datetime64[D]")
    valid_dates = ~np.isnat(starts) & ~np.isnat(ends)
    durations = np.full(len(raw), np.nan, dtype=float)
    durations[valid_dates] = np.busday_count(starts[valid_dates] + np.timedelta64(1, "D"), ends[valid_dates] + np.timedelta64(1, "D"), holidays=ny_holidays)
    raw["approval_business_days"] = durations
    valid = raw.filing_date.dt.year.between(2021, 2026) & raw.approval_business_days.ge(0)
    valid &= raw.filing_status.ne("Filing Withdrawn")
    # Include 2026 Q1 for a final comparison after settings are fixed.
    valid &= raw.filing_date.lt("2026-04-01")
    return raw, raw.loc[valid].copy(), work_cols

def make_features(data, work_cols):
    x = pd.DataFrame(index=data.index)
    cats = BASE_CAT + EXTRA_CAT
    for c in [c for c in cats if c != "filing_kind"]:
        x[c] = data[c].astype("string").str.strip().replace("", pd.NA).fillna("Unknown").astype(str)
    suffix = data.job_filing_number.astype("string").str.extract(r"-([A-Za-z]+)(\d+)$")
    x["filing_kind"] = suffix[0].str.upper().fillna("Unknown").astype(str)
    if "filing_kind" in data.columns:
        provided = data["filing_kind"].astype("string").str.strip().str.upper()
        x["filing_kind"] = provided.where(provided.notna() & provided.ne(""), x["filing_kind"]).astype(str)
    x["filing_sequence"] = pd.to_numeric(suffix[1], errors="coerce")
    bounds = {"existing_stories": 200, "proposed_no_of_stories": 200,
              "existing_height": 2000, "proposed_height": 2000,
              "existing_dwelling_units": 10000, "proposed_dwelling_units": 10000}
    raw = {}
    for c in NUM_RAW:
        s = pd.to_numeric(data[c].astype("string").str.replace(r"[$,]", "", regex=True), errors="coerce").astype(float)
        if c not in ["latitude", "longitude"]:
            s = s.mask(s < 0)
        if c in bounds:
            s = s.mask(s > bounds[c])
        if c == "latitude":
            s = s.where(s.between(40, 42))
        if c == "longitude":
            s = s.where(s.between(-75, -73))
        raw[c] = s
        x[c] = s
        x[c + "_missing"] = s.isna().astype("float32")
    x["log_initial_cost"] = np.log1p(raw["initial_cost"])
    x["log_floor_area"] = np.log1p(raw["total_construction_floor_area"])
    x["cost_per_sqft"] = raw["initial_cost"] / raw["total_construction_floor_area"].where(raw["total_construction_floor_area"] > 0)
    for name, old, new in [("stories_change", "existing_stories", "proposed_no_of_stories"),
                           ("height_change", "existing_height", "proposed_height"),
                           ("units_change", "existing_dwelling_units", "proposed_dwelling_units")]:
        x[name] = raw[new] - raw[old]
    for c in work_cols:
        x[c] = data[c].astype("string").str.strip().str.upper().map({"YES": 1., "NO": 0.}).astype("float32")
    x["work_type_count"] = x[[c for c in work_cols if c != "sidewalk_shed_work_type"]].sum(axis=1, min_count=1)
    dates = data.filing_date
    x["filing_year"], x["filing_month"] = dates.dt.year, dates.dt.month
    x["filing_dayofweek"], x["filing_dayofyear"] = dates.dt.dayofweek, dates.dt.dayofyear
    x["filing_month_sin"] = np.sin(2 * np.pi * dates.dt.month / 12)
    x["filing_month_cos"] = np.cos(2 * np.pi * dates.dt.month / 12)
    num = [c for c in x if c not in cats]
    x[num] = x[num].replace([np.inf, -np.inf], np.nan).astype("float32")
    return x

def encode(train_x, val_x, cats):
    nums = [c for c in train_x if c not in cats]
    prep = ColumnTransformer([
        ("categorical", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                       max_categories=200, dtype=np.float32), cats),
        ("numeric", "passthrough", nums),
    ], verbose_feature_names_out=False)
    a = prep.fit_transform(train_x).astype("float32")
    b = prep.transform(val_x).astype("float32")
    return prep, a, b, [True] * len(cats) + [False] * len(nums)
