-- Run once in the Supabase SQL Editor, after schema.sql and seed.sql
-- Forecasts can now come from models that count calendar days or business days

alter table forecasts add column if not exists unit text not null default 'calendar_days';
alter table model_runs add column if not exists model_type text not null default 'linear_regression';
