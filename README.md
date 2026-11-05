# PermitPulse

PermitPulse forecasts how long an NYC construction permit will take to reach DOB approval, based on project factors such as borough, job type, work type, cost and building characteristics. Each forecast comes with an honest error range.

TECH-UB 24 Projects in Programming & Data Science · Fall 2026 team project

**Live app (frontend + backend + database):** https://permitpulse.onrender.com
Free hosting: the first visit after 15 quiet minutes takes about a minute while the server wakes up.

Static copy of the page: https://daniyalmlk.github.io/permitpulse/ (it calls the same backend)

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

- `index.html` – the website (served by the backend at `/` and by GitHub Pages)
- `backend/main.py` – FastAPI backend: forecasts, saved projects, health check
- `backend/requirements.txt` – backend packages
- `ml/` – quantile model: feature code, training script and predictor
- `db/schema.sql` – database tables (Supabase / Postgres)
- `db/seed.sql` – sample data: our regression model, 2,000 training filings, example users, projects and forecasts
- `db/002_backend.sql` – small update for the backend (forecast units)
- `render.yaml` – Render deploy settings
- `hello.py` – team hello-world (each member added their name through a pull request)

## API

| Method | Path | What it does |
|---|---|---|
| GET | `/api/health` | server, model and database status |
| POST | `/api/predict` | forecast for a project (nothing saved) |
| GET | `/api/projects` | saved projects with their latest forecast |
| POST | `/api/projects` | save a project and its forecast |
| DELETE | `/api/projects/{id}` | delete a saved project |

Interactive docs: `/docs` on the backend URL.

## Database

Supabase (free plan, Postgres). In the Supabase SQL Editor run `db/schema.sql`, then `db/seed.sql`, then `db/002_backend.sql`.
Never commit database passwords or API keys.

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
