"""PermitPulse backend: serves the website and connects it to the Supabase database.

Forecasts come from the newest model available:
  1. The HistGB quantile models (ml/) if models/permitpulse_histgb.joblib exists
  2. Otherwise the linear regression stored in the database (model_coefficients)

Security:
  - Every query uses parameters, never string building.
  - Saved projects belong to the browser that saved them (a random key kept in that browser).
    Others cannot see or delete them. The three example projects are read-only.
  - Saves are rate limited and the table has a size cap, so nobody can fill the database.
  - Database use is limited to a few connections at a time.
  - The page is sent with strict security headers (CSP, no framing, HTTPS only).
  - No API docs, no cross-site access, no error details.
"""
import base64
import hashlib
import os
import re
import sys
import threading
import time
from collections import deque
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import psycopg
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from psycopg.rows import dict_row
from pydantic import BaseModel, ConfigDict, Field, field_validator

ROOT = Path(__file__).resolve().parent.parent
LINEAR_MODEL = "weekday_regression_v1"
HISTGB_MODEL = "histgb_quantile_v1"
HISTGB_FILE = ROOT / "models" / "permitpulse_histgb.joblib"

MAX_BODY_BYTES = 16_384          # a project is well under 1 KB
MAX_PROJECTS = 2_000             # whole table
MAX_PROJECTS_PER_BROWSER = 100
SAVES_PER_MINUTE = 30            # across everyone
DB_REQUESTS_PER_MINUTE = 300     # across everyone
DB_CONNECTIONS = 4               # at the same time
MODEL_CACHE_SECONDS = 600
HEALTH_CACHE_SECONDS = 15

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

# No public API docs: the only client is our own page
app = FastAPI(title="PermitPulse", docs_url=None, redoc_url=None, openapi_url=None)


# ---------- The page, its fonts and security headers ----------

PAGE = (ROOT / "index.html").read_text(encoding="utf-8")
FONTS = {name: (ROOT / "fonts" / name).read_bytes() for name in ("overpass.woff2", "public-sans.woff2")}


def script_hashes(html):
    # Only the page's own inline scripts may run (CSP hashes), nothing injected
    return " ".join(
        "'sha256-" + base64.b64encode(hashlib.sha256(body.encode("utf-8")).digest()).decode() + "'"
        for body in re.findall(r"<script>(.*?)</script>", html, re.S)
    )


CSP = (
    "default-src 'none'; "
    f"script-src 'self' {script_hashes(PAGE)}; "
    "style-src 'self' 'unsafe-inline'; "
    "font-src 'self'; img-src 'self' data:; connect-src 'self'; "
    "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
)
SECURITY_HEADERS = {
    "Content-Security-Policy": CSP,
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
}


@app.middleware("http")
async def guard(request: Request, call_next):
    if request.method in ("POST", "PUT", "PATCH"):
        length = request.headers.get("content-length")
        if length is None:
            return JSONResponse({"detail": "Request body needs a length."}, status_code=411)
        if not length.isdigit() or int(length) > MAX_BODY_BYTES:
            return JSONResponse({"detail": "Request body is too large."}, status_code=413)
    response = await call_next(request)
    for key, value in SECURITY_HEADERS.items():
        response.headers[key] = value
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.exception_handler(RequestValidationError)
async def bad_input(request: Request, exc: RequestValidationError):
    # Say which field is wrong without echoing back what was sent
    errors = [{"loc": [str(part) for part in e.get("loc", ())][-2:], "msg": str(e.get("msg", "Invalid value"))}
              for e in exc.errors()[:10]]
    return JSONResponse({"detail": errors}, status_code=422)


@app.get("/", include_in_schema=False)
def home():
    return HTMLResponse(PAGE, headers={"Cache-Control": "no-cache"})


@app.get("/fonts/{name}", include_in_schema=False)
def font(name: str):
    data = FONTS.get(name)
    if data is None:
        raise HTTPException(404, "Not found")
    return Response(data, media_type="font/woff2", headers={"Cache-Control": "public, max-age=31536000, immutable"})


# ---------- Input ----------

class ProjectIn(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    name: str = Field("", max_length=200)
    borough: Literal["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
    job_type: Literal["Alteration", "Alteration CO", "New Building", "ALT-CO", "Full Demolition"] = "Alteration"
    review_type: Literal["Professional Certification", "Standard Plan Examination"] = "Standard Plan Examination"
    work_types: list[str] = Field([], max_length=len(WORK_TYPES))
    estimated_cost: float = Field(0, ge=0, le=5_000_000_000)
    floor_area: float = Field(0, ge=0, le=50_000_000)
    dwelling_units: int = Field(0, ge=0, le=10_000)
    building_type: Literal["1 Family", "2 Family", "3 Family", "Other"] | None = None
    stories: int | None = Field(None, ge=1, le=200)
    planned_filing_date: date

    @field_validator("name")
    @classmethod
    def clean_name(cls, value):
        # drop control and invisible formatting characters (e.g. right-to-left overrides)
        return "".join(ch for ch in value if ch.isprintable()).strip()

    @field_validator("work_types")
    @classmethod
    def known_work_types(cls, value):
        unknown = [w for w in value if w not in WORK_TYPES]
        if unknown:
            raise ValueError("unknown work type")
        return [w for w in WORK_TYPES if w in value]

    @field_validator("planned_filing_date")
    @classmethod
    def sensible_date(cls, value):
        if not date(2020, 1, 1) <= value <= date(2035, 12, 31):
            raise ValueError("must be between 2020 and 2035")
        return value


def owner_hash(key, required):
    """The browser's random key -> the hash we store. The key itself is never stored."""
    if key is None and not required:
        return None
    if key is None or not re.fullmatch(r"[0-9a-f]{64}", key):
        raise HTTPException(400, "Missing or invalid browser key.")
    return hashlib.sha256(key.encode()).hexdigest()


# ---------- Limits ----------

class RateLimit:
    def __init__(self, count, seconds):
        self.count, self.seconds = count, seconds
        self.times = deque()
        self.lock = threading.Lock()

    def allow(self):
        now = time.monotonic()
        with self.lock:
            while self.times and now - self.times[0] > self.seconds:
                self.times.popleft()
            if len(self.times) >= self.count:
                return False
            self.times.append(now)
            return True


save_limit = RateLimit(SAVES_PER_MINUTE, 60)
db_limit = RateLimit(DB_REQUESTS_PER_MINUTE, 60)
db_slots = threading.BoundedSemaphore(DB_CONNECTIONS)


def connect():
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise HTTPException(503, "The server has no database connection.")
    options = {"row_factory": dict_row, "prepare_threshold": None, "connect_timeout": 10}
    if "sslmode" not in url:
        options["sslmode"] = "require"  # always encrypt the connection to Supabase
    try:
        # prepare_threshold=None keeps this working with Supabase's connection pooler
        return psycopg.connect(url, **options)
    except psycopg.OperationalError:
        raise HTTPException(503, "Could not reach the database. Try again in a minute.")


@contextmanager
def database():
    if not db_limit.allow():
        raise HTTPException(429, "Too many requests right now. Try again in a minute.")
    if not db_slots.acquire(timeout=10):
        raise HTTPException(503, "The server is busy. Try again in a moment.")
    try:
        with connect() as conn:
            yield conn
    finally:
        db_slots.release()


# ---------- Dates ----------

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


# ---------- Linear regression (coefficients from the database) ----------

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


_model_cache = {"at": 0.0, "model": None}


def linear_model():
    # The coefficients rarely change, so forecasts do not need a database trip each time
    if _model_cache["model"] and time.monotonic() - _model_cache["at"] < MODEL_CACHE_SECONDS:
        return _model_cache["model"]
    with database() as conn:
        run = conn.execute(
            "select r_squared, n_obs, residual_std_error from model_runs where name = %s", [LINEAR_MODEL]
        ).fetchone()
        coefs = conn.execute(
            "select term, estimate from model_coefficients where model_name = %s", [LINEAR_MODEL]
        ).fetchall()
    if run is None or not coefs:
        raise HTTPException(503, "The forecast model is not available right now.")
    _model_cache.update(at=time.monotonic(), model=(run, coefs))
    return run, coefs


def linear_forecast(p):
    run, coefs = linear_model()
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
        "high_days": round(max(p50, total + spread), 1),
        "base_days": round(base, 1),
        "factors": factors,
        "model_info": {"r_squared": run["r_squared"], "n_obs": run["n_obs"]},
        "notes": ["Early model (R squared 0.04). A better model is being trained."],
    }


# ---------- Quantile model (file in models/, code in ml/) ----------

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


def histgb_forecast(p, predictor):
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
        "notes": list(out.get("notes", [])),
    }


def forecast(p):
    note = None
    if HISTGB_FILE.exists():
        try:
            return with_dates(histgb_forecast(p, load_histgb()), p.planned_filing_date)
        except Exception as error:  # a broken model file should not take the site down
            note = f"New model could not run ({type(error).__name__}); used the regression instead."
    result = linear_forecast(p)
    if note:
        result["notes"].append(note)
    return with_dates(result, p.planned_filing_date)


def with_dates(result, start):
    for key in ("p50", "low", "high"):
        result[key + "_date"] = to_date(start, result[key + "_days"], result["unit"]).isoformat()
    return result


# ---------- API ----------

_health_cache = {"at": 0.0, "database": None}


@app.get("/api/health")
def health():
    status = {"server": "ok", "model": HISTGB_MODEL if HISTGB_FILE.exists() else LINEAR_MODEL}
    if time.monotonic() - _health_cache["at"] > HEALTH_CACHE_SECONDS:
        try:
            with database() as conn:
                conn.execute("select 1")
            _health_cache["database"] = "connected"
        except HTTPException:
            _health_cache["database"] = "unavailable"
        _health_cache["at"] = time.monotonic()
    status["database"] = _health_cache["database"]
    return status


@app.post("/api/predict")
def predict(p: ProjectIn):
    return forecast(p)


@app.get("/api/projects")
def list_projects(x_owner_key: str | None = Header(None)):
    owner = owner_hash(x_owner_key, required=False)
    with database() as conn:
        rows = conn.execute(
            """
            select p.id, p.name, p.borough, p.job_type, p.review_type, p.work_types, p.estimated_cost,
                   p.floor_area, p.dwelling_units, p.building_type, p.stories, p.planned_filing_date,
                   f.model_name, f.predicted_days, f.low_days, f.high_days, f.unit,
                   coalesce(p.owner_hash = %(owner)s, false) as mine
            from projects p
            left join lateral (
                select * from forecasts where project_id = p.id order by created_at desc, id desc limit 1
            ) f on true
            where p.owner_hash is null or p.owner_hash = %(owner)s
            order by p.created_at desc, p.id desc
            limit 100
            """,
            {"owner": owner},
        ).fetchall()
    for row in rows:
        row["p50_date"] = None
        if row["predicted_days"] is not None:
            row["p50_date"] = to_date(row["planned_filing_date"], float(row["predicted_days"]), row["unit"]).isoformat()
    return rows


@app.post("/api/projects", status_code=201)
def save_project(p: ProjectIn, x_owner_key: str | None = Header(None)):
    owner = owner_hash(x_owner_key, required=True)
    if not save_limit.allow():
        raise HTTPException(429, "Too many saves right now. Try again in a minute.")
    result = forecast(p)
    with database() as conn:
        counts = conn.execute(
            "select count(*) as total, count(*) filter (where owner_hash = %s) as mine from projects", [owner]
        ).fetchone()
        if counts["total"] >= MAX_PROJECTS:
            raise HTTPException(503, "Saving is paused because the database is full.")
        if counts["mine"] >= MAX_PROJECTS_PER_BROWSER:
            raise HTTPException(429, f"You can keep up to {MAX_PROJECTS_PER_BROWSER} projects. Delete some to save more.")
        if result["model"] == HISTGB_MODEL:
            conn.execute(
                "insert into model_runs (name, description, model_type) values (%s, %s, %s) on conflict (name) do nothing",
                [HISTGB_MODEL, "HistGradientBoosting quantile models (P10/P50/P90), business days, trained on 2021-2023 filings",
                 "histgb_quantile"],
            )
        project = conn.execute(
            """
            insert into projects (name, borough, job_type, review_type, work_types, estimated_cost, floor_area,
                                  dwelling_units, building_type, stories, planned_filing_date, owner_hash)
            values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            returning id
            """,
            [p.name or "Untitled project", p.borough, p.job_type, p.review_type, p.work_types, p.estimated_cost,
             p.floor_area, p.dwelling_units, p.building_type, p.stories, p.planned_filing_date, owner],
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
def delete_project(project_id: int, x_owner_key: str | None = Header(None)):
    owner = owner_hash(x_owner_key, required=True)
    with database() as conn:
        # only the browser that saved a project can delete it; the examples have no owner
        deleted = conn.execute(
            "delete from projects where id = %s and owner_hash = %s", [project_id, owner]
        ).rowcount
    if deleted == 0:
        raise HTTPException(404, "Project not found")
    return Response(status_code=204)
