-- Cortex Knowledge Graph Schema
-- This file is the single source of truth for the database schema.
-- Never alter tables inline. Add migrations as versioned ALTER statements below.
-- Version: 1.0.0

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
PRAGMA synchronous = NORMAL;

-- ---------------------------------------------------------------------------
-- Nodes: memory units in the graph
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS nodes (
    id             TEXT    PRIMARY KEY,                          -- UUID4
    type           TEXT    NOT NULL                             -- observation | fact | convention | error
                           CHECK (type IN ('observation', 'fact', 'convention', 'error')),
    tier           INTEGER NOT NULL DEFAULT 1                   -- 1=ephemeral 2=semantic 3=procedural
                           CHECK (tier IN (1, 2, 3)),
    text           TEXT    NOT NULL,                            -- one sentence, no reasoning
    rationale      TEXT,                                        -- one sentence why, nullable
    embedding      BLOB,                                        -- serialized float32/int8/int2 numpy array
    precision_bits INTEGER NOT NULL DEFAULT 32                  -- 32 | 8 | 2
                           CHECK (precision_bits IN (32, 8, 2)),
    weight         REAL    NOT NULL DEFAULT 1.0,                -- access-weighted strength score
    project        TEXT    NOT NULL,                            -- absolute path to project root
    scope          TEXT    NOT NULL DEFAULT 'project'           -- project | module | session
                           CHECK (scope IN ('project', 'module', 'session')),
    source         TEXT    NOT NULL                             -- jsonl | ast | git | nlp
                           CHECK (source IN ('jsonl', 'ast', 'git', 'nlp')),
    last_accessed  INTEGER NOT NULL,                            -- unix timestamp
    created_at     INTEGER NOT NULL,                            -- unix timestamp
    session_count  INTEGER NOT NULL DEFAULT 1                   -- distinct sessions that accessed this
);

-- ---------------------------------------------------------------------------
-- Edges: weighted relationships between nodes
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS edges (
    source_id  TEXT    NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    target_id  TEXT    NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    strength   REAL    NOT NULL DEFAULT 1.0,                   -- co-retrieval count
    last_seen  INTEGER NOT NULL,                               -- unix timestamp
    PRIMARY KEY (source_id, target_id),
    CHECK (source_id != target_id)
);

-- ---------------------------------------------------------------------------
-- Sessions: lightweight log of extraction runs
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sessions (
    id              TEXT    PRIMARY KEY,                        -- UUID4
    project         TEXT    NOT NULL,
    started_at      INTEGER NOT NULL,
    ended_at        INTEGER,
    tokens_raw      INTEGER,                                   -- what would have been sent
    tokens_injected INTEGER,                                   -- what was actually injected
    nodes_written   INTEGER NOT NULL DEFAULT 0,
    nodes_evicted   INTEGER NOT NULL DEFAULT 0,
    nodes_promoted  INTEGER NOT NULL DEFAULT 0,                -- nodes promoted to a higher tier
    transcript_path TEXT                                       -- path to .cortex/sessions/<id>.jsonl
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_nodes_project      ON nodes(project);
CREATE INDEX IF NOT EXISTS idx_nodes_tier         ON nodes(tier);
CREATE INDEX IF NOT EXISTS idx_nodes_type         ON nodes(type);
CREATE INDEX IF NOT EXISTS idx_nodes_weight       ON nodes(weight);
CREATE INDEX IF NOT EXISTS idx_nodes_last_accessed ON nodes(last_accessed);
CREATE INDEX IF NOT EXISTS idx_nodes_project_tier ON nodes(project, tier);
CREATE INDEX IF NOT EXISTS idx_edges_source       ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target       ON edges(target_id);
CREATE INDEX IF NOT EXISTS idx_sessions_project   ON sessions(project);
CREATE INDEX IF NOT EXISTS idx_nodes_project_source ON nodes(project, source);
CREATE INDEX IF NOT EXISTS idx_nodes_project_scope  ON nodes(project, scope);

-- ---------------------------------------------------------------------------
-- Migrations (append only — never modify above)
-- ---------------------------------------------------------------------------
-- v1.0.0 → initial schema
-- v1.1.0 → add idx_nodes_project_source, idx_nodes_project_scope
