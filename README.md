# PermitPulse

PermitPulse forecasts how long an NYC construction permit will take to reach DOB approval, based on project factors such as borough, job type, work type, cost and building characteristics. Each forecast comes with an honest error range.

TECH-UB 24 Projects in Programming & Data Science · Fall 2026 team project

**Live app (frontend + backend + database):** https://permitpulse-qroy.onrender.com
Free hosting: the first visit after 15 quiet minutes takes about a minute while the server wakes up.

https://daniyalmlk.github.io/permitpulse/ forwards to the live app.

## How it fits together

```
browser (index.html)  ->  FastAPI backend (backend/main.py, on Render)  ->  Supabase Postgres
  form inputs              POST /api/predict  runs the model              model_coefficients
                           POST /api/projects saves project + forecast   projects, forecasts
                           GET  /api/projects lists saved projects
```

The model right now is our first linear regression. Its coefficients live in the database
(`model_coefficients`) and the backend reads them for every forecast. The quantile model in `ml/`
(P10/P50/P90 in business days) takes over automatically once its trained file is in `models/`.

## Data

[NYC DOB NOW: Build – Job Application Filings](https://data.cityofnewyork.us/d/w9ak-ipjd) (NYC Open Data)

## Repo layout

- `index.html` – the website (served by the backend at `/`; on GitHub Pages it forwards to the live app)
- `fonts/` – the two fonts, served by our backend (SIL Open Font License)
- `backend/main.py` – FastAPI backend: forecasts, saved projects, health check
- `backend/requirements.txt` – backend packages
- `ml/` – quantile model: feature code, training script and predictor
- `db/schema.sql` – database tables (Supabase / Postgres)
- `db/seed.sql` – sample data: our regression model, 2,000 training filings, example users, projects and forecasts
- `db/002_backend.sql` – small update for the backend (forecast units)
- `db/003_security.sql` – private saved projects and no public access to the tables
- `db/check.sql` – confirms the database holds the current model and data
- `render.yaml` – Render deploy settings
- `hello.py` – team hello-world (each member added their name through a pull request)

## API

| Method | Path | What it does |
|---|---|---|
| GET | `/api/health` | server, model and database status |
| POST | `/api/predict` | forecast for a project (nothing saved) |
| GET | `/api/projects` | the 3 examples plus this browser's saved projects |
| POST | `/api/projects` | save a project and its forecast |
| DELETE | `/api/projects/{id}` | delete one of this browser's saved projects |

Saving and deleting need the `X-Owner-Key` header: a random 64-character key the page creates and keeps in the browser.

## Database

Supabase (free plan, Postgres). In the Supabase SQL Editor run `db/schema.sql`, `db/seed.sql`, `db/002_backend.sql`, then `db/003_security.sql`.
Never commit database passwords or API keys.

To check the database is current, run `db/check.sql`. It should return:

| coefficients | coefficient_sum | n_obs | r_squared | training_rows | actual_days_sum | predicted_days_sum | has_unit_column |
|---|---|---|---|---|---|---|---|
| 26 | 146.2937 | 263165 | 0.0441 | 2000 | 33937 | 38263.61 | 1 |

`saved_projects` is 3 for a fresh database and grows as people save projects.

## Security

- Every query uses parameters, so input can never change the SQL.
- Saved projects are private to the browser that saved them. Only a hash of the browser's key is stored. The examples cannot be deleted.
- The tables have row level security and no rights for Supabase's public roles, so the public API key cannot read or change anything. Only the backend, using the database password from Render's settings, can.
- The connection to Supabase is always encrypted.
- Saves are rate limited, each browser can keep up to 100 projects, the table is capped at 2,000, and only a few database connections run at once.
- The page is sent with a strict Content Security Policy (only its own scripts run), and it cannot be framed. It loads nothing from other sites.
- No API docs, no cross-site API access, and errors never echo input or internals.
- Packages are pinned to versions with no known vulnerabilities (checked with `pip-audit`).

## Run it locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
export DATABASE_URL="postgresql://..."   # Supabase > Connect > Session pooler
uvicorn backend.main:app --reload
```

Open http://localhost:8000

## Deploy (Render, free)

1. Render > New > Blueprint, pick this repo (it reads `render.yaml`).
2. Set `DATABASE_URL` to the Supabase **Session pooler** connection string (Render has no IPv6, so the direct connection does not work).
3. Deploy. The health check is `/api/health`.

When the quantile model file is added to `models/`, change the build command to
`pip install -r backend/requirements.txt -r ml/requirements.txt` and pin the scikit-learn version it was trained with.

## Team

Daniyal, Walter Kusuma, Allan You, Jiayi Kalem Xu, Justin Sun
