# PermitPulse

PermitPulse forecasts how long an NYC construction permit will take to reach DOB approval, based on project factors such as borough, job type, work type, cost and building characteristics. Each forecast comes with an honest error range, compared against the DOB average.

TECH-UB 24 Projects in Programming & Data Science · Fall 2026 team project

**Live prototype:** https://daniyalmlk.github.io/permitpulse/
Clickable MVP. Forecast numbers are placeholders until the model is trained.

## Data

[NYC DOB NOW: Build – Job Application Filings](https://data.cityofnewyork.us/d/w9ak-ipjd) (NYC Open Data)

## Repo layout

- `index.html` – clickable prototype, served by GitHub Pages
- `db/schema.sql` – database tables (Supabase / Postgres)
- `db/seed.sql` – sample data: our regression model, 2,000 training filings, example users, projects and forecasts
- `hello.py` – team hello-world (each member added their name through a pull request)

## Database

Supabase (free plan, Postgres). Run `db/schema.sql`, then `db/seed.sql`, in the Supabase SQL Editor. Never commit database passwords or API keys.

## Team

Daniyal, Walter Kusuma, Allan You, Jiayi Kalem Xu, Justin Sun
