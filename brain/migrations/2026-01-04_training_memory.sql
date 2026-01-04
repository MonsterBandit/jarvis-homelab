-- Alice v0 Training Memory (Scope B)

CREATE TABLE IF NOT EXISTS memory_users (
  user_id TEXT PRIMARY KEY,
  display_name TEXT NOT NULL,
  role TEXT NOT NULL CHECK(role IN ('admin','user')),
  access_level TEXT NOT NULL CHECK(access_level IN ('read_only','write')),
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS memory_concepts (
  concept_id INTEGER PRIMARY KEY AUTOINCREMENT,
  concept_key TEXT NOT NULL UNIQUE,
  category TEXT,
  default_unit TEXT,
  notes TEXT,
  confidence REAL NOT NULL DEFAULT 0.2,
  locked INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS memory_aliases (
  alias_id INTEGER PRIMARY KEY AUTOINCREMENT,
  concept_id INTEGER NOT NULL,
  user_id TEXT NOT NULL,
  preferred_name TEXT NOT NULL,
  pattern_notes TEXT,
  confidence REAL NOT NULL DEFAULT 0.2,
  locked INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(concept_id, user_id),
  FOREIGN KEY (concept_id) REFERENCES memory_concepts(concept_id) ON DELETE CASCADE,
  FOREIGN KEY (user_id) REFERENCES memory_users(user_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS memory_preferences (
  pref_id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id TEXT NOT NULL,
  pref_key TEXT NOT NULL,
  pref_value TEXT NOT NULL,
  scope TEXT NOT NULL DEFAULT 'phase_6_45',
  confidence REAL NOT NULL DEFAULT 0.2,
  locked INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(user_id, pref_key, scope),
  FOREIGN KEY (user_id) REFERENCES memory_users(user_id) ON DELETE CASCADE
);

INSERT OR IGNORE INTO memory_users (user_id, display_name, role, access_level)
VALUES
  ('admin_tim', 'Tim', 'admin', 'write'),
  ('user_kaitlyn', 'Kaitlyn', 'user', 'read_only');
