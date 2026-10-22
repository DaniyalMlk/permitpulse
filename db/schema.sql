-- PermitPulse database skeleton (Supabase / PostgreSQL)
-- Forecasts how long an NYC DOB NOW permit filing takes to reach approval.
--
-- Tables
--   model_runs          one row per trained model, with fit statistics
--   model_coefficients  the regression terms of each model run
--   training_sample     a sample of DOB filings the model was trained on (actual vs. fitted days)
--   app_users           placeholder users (contractors, expediters, architects, developers)
--   projects            projects a user describes in the app (the model's inputs)
--   forecasts           model output for a project, with an 80% range
-- Function
--   predict_days(project_id, model_name)  applies a model's coefficients to a saved project
--
-- Row level security is on for every table with no public policies, so only the
-- backend (service role / direct database connection) can read or write.

create table if not exists public.model_runs (
  id                 bigint generated always as identity primary key,
  name               text not null unique,
  description        text,
  target             text not null,
  n_obs              integer not null check (n_obs > 0),
  r_squared          double precision,
  adj_r_squared      double precision,
  f_statistic        double precision,
  df_model           integer,
  df_residual        integer,
  residual_std_error double precision check (residual_std_error >= 0),
  created_at         timestamptz not null default now()
);

create table if not exists public.model_coefficients (
  model_run_id bigint not null references public.model_runs (id) on delete cascade,
  term         text not null,
  estimate     double precision not null,
  std_error    double precision,
  t_value      double precision,
  p_value      double precision check (p_value between 0 and 1),
  primary key (model_run_id, term)
);

create table if not exists public.training_sample (
  id            bigint generated always as identity primary key,
  model_run_id  bigint not null references public.model_runs (id) on delete cascade,
  initial_cost  numeric(14, 2) check (initial_cost >= 0),
  actual_days   integer not null check (actual_days >= 0),
  fitted_days   numeric(10, 2) not null,
  residual_days numeric(10, 2) generated always as (actual_days - fitted_days) stored
);
create index if not exists training_sample_model_idx on public.training_sample (model_run_id);

create table if not exists public.app_users (
  id         uuid primary key default gen_random_uuid(),
  full_name  text not null,
  email      text not null unique,
  role       text not null check (role in ('contractor', 'expediter', 'architect', 'developer', 'other')),
  created_at timestamptz not null default now()
);

create table if not exists public.projects (
  id                  uuid primary key default gen_random_uuid(),
  user_id             uuid not null references public.app_users (id) on delete cascade,
  name                text not null,
  borough             text not null check (borough in ('Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island')),
  job_type            text not null check (job_type in ('Alteration', 'Alteration CO', 'New Building', 'ALT-CO', 'Full Demolition')),
  filing_review_type  text not null check (filing_review_type in ('Professional Certification', 'Standard Plan Examination')),
  -- work type codes match the model terms (WT_<code>)
  work_types          text[] not null default '{}' check (work_types <@ array[
                        'Sprinkler', 'Plumbing', 'StandPipe', 'Antenna', 'Sign', 'CurbCut', 'Fence',
                        'Scaffold', 'Shed', 'EarthWork', 'Foundation', 'GeneralConstruction',
                        'MechanicSystems', 'ProtMechanicMethods', 'Structural', 'GreenRoof', 'Solar']::text[]),
  estimated_cost      numeric(14, 2) not null default 0 check (estimated_cost >= 0),
  floor_area          numeric(12, 2) not null default 0 check (floor_area >= 0),
  dwelling_units      integer not null default 0 check (dwelling_units >= 0),
  building_type       text check (building_type in ('1 Family', '2 Family', '3 Family', 'Other')),
  proposed_stories    integer check (proposed_stories > 0),
  planned_filing_date date not null,
  created_at          timestamptz not null default now()
);
create index if not exists projects_user_idx on public.projects (user_id);

create table if not exists public.forecasts (
  id             uuid primary key default gen_random_uuid(),
  project_id     uuid not null references public.projects (id) on delete cascade,
  model_run_id   bigint not null references public.model_runs (id),
  predicted_days numeric(10, 1) not null,
  low_days       numeric(10, 1) not null check (low_days >= 0),
  high_days      numeric(10, 1) not null,
  created_at     timestamptz not null default now(),
  check (low_days <= high_days)
);
create index if not exists forecasts_project_idx on public.forecasts (project_id);

-- Applies a model's coefficients to one saved project. Returns predicted days to approval.
create or replace function public.predict_days(p_project_id uuid, p_model_name text default 'weekday-ols-v0')
returns numeric
language sql
stable
set search_path = public
as $$
  select round(sum(c.estimate * case
           when c.term = '(Intercept)'      then 1
           when c.term = 'Initial_Cost'     then p.estimated_cost
           when c.term = 'Floor_Area'       then p.floor_area
           when c.term = 'DwellingUnits'    then p.dwelling_units
           when c.term = 'SubmittedWeekend' then (extract(isodow from p.planned_filing_date) >= 6)::int
           when c.term like '%\_Borough'    then (split_part(c.term, '_', 1) =
                                                  case p.borough when 'Staten Island' then 'Staten' else p.borough end)::int
           when c.term like 'WT\_%'         then (substr(c.term, 4) = any (p.work_types))::int
           else 0
         end)::numeric, 1)
  from public.projects p
  join public.model_runs m on m.name = p_model_name
  join public.model_coefficients c on c.model_run_id = m.id
  where p.id = p_project_id
$$;

alter table public.model_runs         enable row level security;
alter table public.model_coefficients enable row level security;
alter table public.training_sample    enable row level security;
alter table public.app_users          enable row level security;
alter table public.projects           enable row level security;
alter table public.forecasts          enable row level security;
