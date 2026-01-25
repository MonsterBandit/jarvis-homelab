-- Identity & Memory v1 — SQLite migration
-- Contract: additive-only, idempotent, no destructive changes.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    display_name TEXT,
    status TEXT NOT NULL CHECK (status IN ('provisional', 'reinforced', 'permanent')),
    created_at TEXT NOT NULL,
    expires_at TEXT,                -- NULL once permanent
    last_seen_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS user_lifecycle_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    details TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS memory_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    memory_key TEXT NOT NULL,
    memory_value TEXT NOT NULL,
    memory_tier INTEGER NOT NULL CHECK (memory_tier IN (1, 2)),
    confidence REAL NOT NULL DEFAULT 1.0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    expires_at TEXT,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    UNIQUE (user_id, memory_key)
);

CREATE TABLE IF NOT EXISTS memory_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id INTEGER,
    user_id TEXT NOT NULL,
    action TEXT NOT NULL,
    actor TEXT NOT NULL,          -- 'user' | 'admin' | 'system'
    created_at TEXT NOT NULL,
    FOREIGN KEY (memory_id) REFERENCES memory_items(id) ON DELETE SET NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS admin_inbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    user_id TEXT,
    payload TEXT,
    created_at TEXT NOT NULL,
    acknowledged INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_users_status ON users(status);
CREATE INDEX IF NOT EXISTS idx_memory_user ON memory_items(user_id);
CREATE INDEX IF NOT EXISTS idx_memory_expires ON memory_items(expires_at);
CREATE INDEX IF NOT EXISTS idx_admin_inbox_ack ON admin_inbox(acknowledged);

-- Reserved for v1.1 (not activated here):
-- device_identity_map(device_id -> user_id) stored as identity signal, not personal memory.
