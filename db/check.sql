-- Quick check that the database holds the current model and training data.
-- Run in the Supabase SQL Editor and compare with the expected values in the README.

select
  (select count(*) from model_coefficients) as coefficients,
  (select round(sum(estimate)::numeric, 4) from model_coefficients) as coefficient_sum,
  (select n_obs from model_runs where name = 'weekday_regression_v1') as n_obs,
  (select round(r_squared::numeric, 4) from model_runs where name = 'weekday_regression_v1') as r_squared,
  (select count(*) from training_sample) as training_rows,
  (select sum(actual_days) from training_sample) as actual_days_sum,
  (select round(sum(predicted_days)::numeric, 2) from training_sample) as predicted_days_sum,
  (select count(*) from information_schema.columns where table_name = 'forecasts' and column_name = 'unit') as has_unit_column,
  (select count(*) from projects) as saved_projects;
