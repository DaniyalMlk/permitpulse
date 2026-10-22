-- PermitPulse database (Supabase / Postgres)
-- Run this first, then seed.sql

-- regression models we have trained
create table model_runs (
  name text primary key,
  description text,
  n_obs integer,
  r_squared double precision,
  adj_r_squared double precision,
  f_statistic double precision,
  residual_std_error double precision,
  created_at timestamptz default now()
);

-- one row per term in each model
create table model_coefficients (
  model_name text references model_runs(name) on delete cascade,
  term text,
  estimate double precision not null,
  std_error double precision,
  t_value double precision,
  p_value double precision,
  primary key (model_name, term)
);

-- sample of DOB filings used to train the model
create table training_sample (
  id serial primary key,
  model_name text references model_runs(name) on delete cascade,
  initial_cost numeric,
  actual_days integer not null,
  predicted_days numeric not null
);

-- people using the app
create table app_users (
  id serial primary key,
  full_name text not null,
  email text unique not null,
  role text check (role in ('contractor', 'expediter', 'architect', 'developer', 'other')),
  created_at timestamptz default now()
);

-- projects a user enters in the app (inputs for the model)
create table projects (
  id serial primary key,
  user_id integer references app_users(id) on delete cascade,
  name text not null,
  borough text not null check (borough in ('Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island')),
  job_type text,
  review_type text,
  work_types text[],
  estimated_cost numeric default 0,
  floor_area numeric default 0,
  dwelling_units integer default 0,
  building_type text,
  stories integer,
  planned_filing_date date not null,
  created_at timestamptz default now()
);

-- model output for a project
create table forecasts (
  id serial primary key,
  project_id integer references projects(id) on delete cascade,
  model_name text references model_runs(name),
  predicted_days numeric not null,
  low_days numeric,
  high_days numeric,
  created_at timestamptz default now()
);

-- turn on row level security so only our backend can read and write
alter table model_runs enable row level security;
alter table model_coefficients enable row level security;
alter table training_sample enable row level security;
alter table app_users enable row level security;
alter table projects enable row level security;
alter table forecasts enable row level security;
