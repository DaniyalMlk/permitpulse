-- Run once in the Supabase SQL Editor, after 002_backend.sql

-- 1. Saved projects belong to the browser that saved them.
--    We store a hash of the browser's random key, never the key itself.
--    The three example projects have no owner, so nobody can delete them.
alter table projects add column if not exists owner_hash text;
create index if not exists projects_owner_hash_idx on projects (owner_hash);

-- 2. Nobody can use the tables through Supabase's public API (the anon/publishable key).
--    Row level security is already on with no policies; this removes the table rights too.
--    Only our backend, which connects with the database password, can read or write.
revoke all on all tables in schema public from anon, authenticated;
revoke all on all sequences in schema public from anon, authenticated;
revoke all on all functions in schema public from anon, authenticated;
alter default privileges in schema public revoke all on tables from anon, authenticated;
alter default privileges in schema public revoke all on sequences from anon, authenticated;
alter default privileges in schema public revoke all on functions from anon, authenticated;

-- 3. Remove test saves made before this update. They have no owner, so nobody could delete them later.
--    The three examples stay (they belong to the example users).
delete from projects where owner_hash is null and user_id is null;

-- 4. Check: every table should show rls_on = true, policies = 0, and false for both public columns
select c.relname as table_name,
       c.relrowsecurity as rls_on,
       (select count(*) from pg_policies p where p.schemaname = 'public' and p.tablename = c.relname) as policies,
       has_table_privilege('anon', c.oid, 'select') as public_can_read,
       has_table_privilege('anon', c.oid, 'insert, update, delete') as public_can_write
from pg_class c
join pg_namespace n on n.oid = c.relnamespace
where n.nspname = 'public' and c.relkind = 'r'
order by c.relname;
