-- ═══════════════════════════════════════════════════════════════
-- AgroModules — PostgreSQL Database Initialization
-- ═══════════════════════════════════════════════════════════════
-- Run this script once to create the database.
--
-- Option 1 — psql CLI:
--   psql -U postgres -f init_db.sql
--
-- Option 2 — pgAdmin:
--   Open Query Tool → paste this → Execute
--
-- Option 3 — Run init_db.bat (auto-detects psql)
-- ═══════════════════════════════════════════════════════════════

-- Create database (ignore error if already exists)
SELECT 'CREATE DATABASE agrodb'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'agrodb')\gexec

-- Connect to the new database
\c agrodb

-- The tables (users, disease_diagnoses) are auto-created
-- by SQLAlchemy on Auth service startup via create_tables().
-- This script only ensures the database itself exists.

-- Optional: grant privileges if using a non-postgres user
-- GRANT ALL PRIVILEGES ON DATABASE agrodb TO your_user;

