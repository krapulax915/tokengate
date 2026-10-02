PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS requests (
  id TEXT PRIMARY KEY,
  ts REAL NOT NULL,                    -- unix seconds
  api_key_id TEXT,
  task_id TEXT NOT NULL,
  task_type TEXT NOT NULL,
  requested_model TEXT,                -- "auto" or pinned id
  chosen_model TEXT,
  provider TEXT,
  policy TEXT,
  explore INTEGER DEFAULT 0,
  decision_reason TEXT,
  cache_hit INTEGER DEFAULT 0,
  shortcut TEXT,
  stream INTEGER DEFAULT 0,
  status INTEGER,
  prompt_tokens INTEGER,
  completion_tokens INTEGER,
  cost_usd REAL,
  baseline_cost_usd REAL,
  ttft_ms REAL,
  total_ms REAL,
  overhead_ms REAL,
  attempts INTEGER DEFAULT 1,          -- >1 if fallback or cascade
  error TEXT
);
CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(ts);
CREATE INDEX IF NOT EXISTS idx_requests_task ON requests(task_id);
CREATE INDEX IF NOT EXISTS idx_requests_type_model ON requests(task_type, chosen_model);

CREATE TABLE IF NOT EXISTS outcomes (
  task_id TEXT PRIMARY KEY,
  ts REAL NOT NULL,
  success INTEGER NOT NULL,            -- 0/1
  score REAL,                          -- optional 0..1
  source TEXT NOT NULL                 -- client | checker | shadow_judge
);

CREATE TABLE IF NOT EXISTS shadow_evals (
  id TEXT PRIMARY KEY,
  request_id TEXT NOT NULL,
  ts REAL NOT NULL,
  baseline_model TEXT,
  judge_model TEXT,
  verdict TEXT,                        -- equal_or_better | worse
  eval_cost_usd REAL
);

CREATE TABLE IF NOT EXISTS arm_snapshots (   -- periodic snapshots for the learning-curve chart
  ts REAL NOT NULL,
  task_type TEXT NOT NULL,
  model_id TEXT NOT NULL,
  n_labeled INTEGER,
  successes INTEGER,
  cost_sum_usd REAL,
  PRIMARY KEY (ts, task_type, model_id)
);
