"""PermitPulse backend: serves the website and connects it to the Supabase database.

Forecasts come from the newest model available:
  1. The HistGB quantile models (ml/) if models/permitpulse_histgb.joblib exists
  2. Otherwise the linear regression stored in the database (model_coefficients)
"""
import os
import re
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import psycopg
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from psycopg.rows import dict_row
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parent.parent
LINEAR_MODEL = "weekday_regression_v1"
HISTGB_MODEL = "histgb_quantile_v1"
HISTGB_FILE = ROOT / "models" / "permitpulse_histgb.joblib"

WORK_TYPES = {
    "GeneralConstruction": "General construction",
    "Plumbing": "Plumbing",
    "MechanicSystems": "Mechanical systems",
    "Structural": "Structural",
    "Foundation": "Foundation",
    "EarthWork": "Earth work",
    "Sprinkler": "Sprinkler",
    "StandPipe": "Standpipe",
    "Shed": "Sidewalk shed",
    "Scaffold": "Scaffold",
    "Fence": "Fence",
    "Sign": "Sign",
    "CurbCut": "Curb cut",
    "Antenna": "Antenna",
    "Solar": "Solar",
    "GreenRoof": "Green roof",
    "ProtMechanicMethods": "Protection (mechanical methods)",
}

app = FastAPI(title="PermitPulse API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://daniyalmlk.github.io"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)


class ProjectIn(BaseModel):
    name: str = Field("", max_length=200)
    borough: Literal["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
    job_type: Literal["Alteration", "Alteration CO", "New Building", "ALT-CO", "Full Demolition"] = "Alteration"
    review_type: Literal["Professional Certification", "Standard Plan Examination"] = "Standard Plan Examination"
    work_types: list[str] = []
    estimated_cost: float = Field(0, ge=0, le=5_000_000_000)
    floor_area: float = Field(0, ge=0, le=50_000_000)
    dwelling_units: int = Field(0, ge=0, le=10_000)
    building_type: Literal["1 Family", "2 Family", "3 Family", "Other"] | None = None
    stories: int | None = Field(None, ge=1, le=200)
    planned_filing_date: date

    @field_validator("work_types")
    @classmethod
    def known_work_types(cls, value):
        unknown = [w for w in value if w not in WORK_TYPES]
        if unknown:
            raise ValueError("unknown work types: " + ", ".join(unknown))
        return [w for w in WORK_TYPES if w in value]


def connect():
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise HTTPException(503, "The server has no database connection (DATABASE_URL is not set).")
    try:
        # prepare_threshold=None keeps this working with Supabase's connection pooler
        return psycopg.connect(url, row_factory=dict_row, prepare_threshold=None, connect_timeout=10)
    except psycopg.OperationalError:
        raise HTTPException(503, "Could not reach the database. Try again in a minute.")


def add_business_days(start, n):
    try:
        import holidays
        closed = holidays.US(subdiv="NY", years=range(start.year, start.year + 4), observed=True)
    except ImportError:
        closed = {}
    day, left = start, int(round(n))
    while left > 0:
        day += timedelta(days=1)
        if day.weekday() < 5 and day not in closed:
            left -= 1
    return day


def to_date(start, days, unit):
    if unit == "business_days":
        return add_business_days(start, days)
    return start + timedelta(days=int(round(days)))


def term_label(term):
    if term == "Initial_Cost":
        return "Job cost"
    if term == "Floor_Area":
        return "Floor area"
    if term == "DwellingUnits":
        return "Dwelling units"
    if term == "SubmittedWeekend":
        return "Filed on a weekend"
    if term.endswith("_Borough"):
        name = term.split("_")[0]
        return "Borough: " + ("Staten Island" if name == "Staten" else name)
    if term.startswith("WT_"):
        return "Work type: " + WORK_TYPES.get(term[3:], term[3:])
    return term


def term_value(term, p):
    if term == "(Intercept)":
        return 1
    if term == "Initial_Cost":
        return p.estimated_cost
    if term == "Floor_Area":
        return p.floor_area
    if term == "DwellingUnits":
        return p.dwelling_units
    if term == "SubmittedWeekend":
        return 1 if p.planned_filing_date.isoweekday() >= 6 else 0
    if term.endswith("_Borough"):
        borough = "Staten" if p.borough == "Staten Island" else p.borough
        return 1 if term.split("_")[0] == borough else 0
    if term.startswith("WT_"):
        return 1 if term[3:] in p.work_types else 0
    return 0


def linear_forecast(conn, p):
    run = conn.execute(
        "select r_squared, n_obs, residual_std_error from model_runs where name = %s", [LINEAR_MODEL]
    ).fetchone()
    coefs = conn.execute(
        "select term, estimate from model_coefficients where model_name = %s", [LINEAR_MODEL]
    ).fetchall()
    if run is None or not coefs:
        raise HTTPException(503, "The regression model is missing from the database. Run db/seed.sql.")

    base, total, factors = 0.0, 0.0, []
    for row in coefs:
        days = row["estimate"] * term_value(row["term"], p)
        total += days
        if row["term"] == "(Intercept)":
            base = days
        elif abs(days) >= 0.05:
            factors.append({"label": term_label(row["term"]), "days": round(days, 1)})
    factors.sort(key=lambda f: -abs(f["days"]))
    other = total - base - sum(f["days"] for f in factors)
    if abs(other) >= 0.05:
        factors.append({"label": "Other small factors", "days": round(other, 1)})

    # 80% range: +/- 1.2816 residual standard errors around the prediction
    spread = 1.2816 * run["residual_std_error"]
    p50 = max(0.0, total)
    return {
        "model": LINEAR_MODEL,
        "model_type": "linear_regression",
        "unit": "calendar_days",
        "p50_days": round(p50, 1),
        "low_days": round(max(0.0, total - spread), 1),
        "high_days": round(total + spread, 1),
        "base_days": round(base, 1),
        "factors": factors,
        "model_info": {"r_squared": run["r_squared"], "n_obs": run["n_obs"]},
        "notes": ["Early model (R squared 0.04). A better model is being trained."],
    }


_histgb = None


def load_histgb():
    global _histgb
    if _histgb is None and HISTGB_FILE.exists():
        sys.path.insert(0, str(ROOT / "ml"))
        from predictor import PermitPulsePredictor
        _histgb = PermitPulsePredictor(str(HISTGB_FILE))
    return _histgb


# our work type codes -> column names in the DOB data (letters only, lower case)
WORK_COLUMN_NAMES = {
    "MechanicSystems": ["mechanicalsystems", "mechanicsystems"],
    "ProtMechanicMethods": ["protectionmechanicalmethods", "protmechanicmethods"],
    "Shed": ["shed", "sidewalkshed"],
    "Fence": ["fence", "constructionfence"],
}
DATASET_JOB_TYPES = {"ALT-CO": "ALT-CO - New Building with Existing Elements to Remain"}


def known_value(predictor, column, value):
    """Spell a form value the way the training data did (e.g. Manhattan -> MANHATTAN)."""
    try:
        encoder = predictor.bundle["preprocessor"].named_transformers_["categorical"]
        known = [str(k) for k in encoder.categories_[list(encoder.feature_names_in_).index(column)]]
    except (KeyError, AttributeError, ValueError):
        return value
    wanted = value.casefold()
    for k in known:
        if k.casefold() == wanted:
            return k
    for k in known:
        if k.casefold().startswith(wanted):
            return k
    return value


def histgb_forecast(conn, p, predictor):
    request = {
        "filing_date": p.planned_filing_date.isoformat(),
        "filing_review_type": known_value(predictor, "filing_review_type", p.review_type),
        "job_type": known_value(predictor, "job_type", DATASET_JOB_TYPES.get(p.job_type, p.job_type)),
        "borough": known_value(predictor, "borough", p.borough),
        "filing_kind": "I",  # a new job is an initial filing (job number ends in -I1)
        "initial_cost": p.estimated_cost,
        "total_construction_floor_area": p.floor_area,
        "proposed_dwelling_units": p.dwelling_units,
    }
    if p.building_type:
        request["building_type"] = known_value(predictor, "building_type", p.building_type)
    if p.stories:
        request["proposed_no_of_stories"] = p.stories
    chosen = {name for w in p.work_types for name in WORK_COLUMN_NAMES.get(w, [w.lower()])}
    for column in predictor.bundle["work_columns"]:
        key = re.sub(r"[^a-z]", "", column.lower().replace("work_type", ""))
        request[column] = "YES" if key in chosen else "NO"

    out = predictor.predict(request)
    conn.execute(
        "insert into model_runs (name, description, model_type) values (%s, %s, %s) on conflict (name) do nothing",
        [HISTGB_MODEL, "HistGradientBoosting quantile models (P10/P50/P90), business days, trained on 2021-2023 filings",
         "histgb_quantile"],
    )
    return {
        "model": HISTGB_MODEL,
        "model_type": "histgb_quantile",
        "unit": "business_days",
        "p50_days": out["p50"],
        "low_days": out["p10"],
        "high_days": out["p90"],
        "base_days": None,
        "factors": [],
        "model_info": {"trained_on": predictor.bundle.get("training_years")},
        "notes": out.get("notes", []),
    }


def forecast(conn, p):
    note = None
    if HISTGB_FILE.exists():
        try:
            predictor = load_histgb()
            result = histgb_forecast(conn, p, predictor)
        except Exception as error:  # a broken model file should not take the site down
            note = f"New model could not run ({type(error).__name__}); used the regression instead."
        else:
            result["notes"] = list(result["notes"])
            return with_dates(result, p.planned_filing_date)
    result = linear_forecast(conn, p)
    if note:
        result["notes"].append(note)
    return with_dates(result, p.planned_filing_date)


def with_dates(result, start):
    for key in ("p50", "low", "high"):
        result[key + "_date"] = to_date(start, result[key + "_days"], result["unit"]).isoformat()
    return result


@app.get("/api/health")
def health():
    status = {"server": "ok", "model": HISTGB_MODEL if HISTGB_FILE.exists() else LINEAR_MODEL}
    try:
        with connect() as conn:
            status["saved_projects"] = conn.execute("select count(*) as n from projects").fetchone()["n"]
            status["database"] = "connected"
    except HTTPException as error:
        status["database"] = error.detail
    return status


@app.post("/api/predict")
def predict(p: ProjectIn):
    with connect() as conn:
        return forecast(conn, p)


@app.get("/api/projects")
def list_projects():
    with connect() as conn:
        rows = conn.execute(
            """
            select p.id, p.name, p.borough, p.job_type, p.review_type, p.work_types, p.estimated_cost,
                   p.floor_area, p.dwelling_units, p.building_type, p.stories, p.planned_filing_date,
                   f.model_name, f.predicted_days, f.low_days, f.high_days, f.unit
            from projects p
            left join lateral (
                select * from forecasts where project_id = p.id order by created_at desc, id desc limit 1
            ) f on true
            order by p.created_at desc, p.id desc
            limit 100
            """
        ).fetchall()
    for row in rows:
        row["p50_date"] = None
        if row["predicted_days"] is not None:
            row["p50_date"] = to_date(row["planned_filing_date"], float(row["predicted_days"]), row["unit"]).isoformat()
    return rows


@app.post("/api/projects", status_code=201)
def save_project(p: ProjectIn):
    with connect() as conn:
        result = forecast(conn, p)
        project = conn.execute(
            """
            insert into projects (name, borough, job_type, review_type, work_types, estimated_cost, floor_area,
                                  dwelling_units, building_type, stories, planned_filing_date)
            values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            returning id
            """,
            [p.name.strip() or "Untitled project", p.borough, p.job_type, p.review_type, p.work_types,
             p.estimated_cost, p.floor_area, p.dwelling_units, p.building_type, p.stories, p.planned_filing_date],
        ).fetchone()
        conn.execute(
            """
            insert into forecasts (project_id, model_name, predicted_days, low_days, high_days, unit)
            values (%s, %s, %s, %s, %s, %s)
            """,
            [project["id"], result["model"], result["p50_days"], result["low_days"], result["high_days"], result["unit"]],
        )
    return {"id": project["id"], "forecast": result}


@app.delete("/api/projects/{project_id}", status_code=204)
def delete_project(project_id: int):
    with connect() as conn:
        deleted = conn.execute("delete from projects where id = %s", [project_id]).rowcount
    if deleted == 0:
        raise HTTPException(404, "Project not found")
    return Response(status_code=204)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(ROOT / "index.html")
