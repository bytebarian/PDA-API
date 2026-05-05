-- Enable the pgvector extension for this database on first startup.
-- This script is run automatically by the PostgreSQL Docker entrypoint
-- for every file found in /docker-entrypoint-initdb.d/.
CREATE EXTENSION IF NOT EXISTS vector;
