CREATE TABLE IF NOT EXISTS searches (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    product     TEXT NOT NULL,
    aspects     TEXT,                    -- JSON list of aspects analyzed
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS reviews (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    search_id     INTEGER NOT NULL REFERENCES searches(id),
    source_type   TEXT NOT NULL CHECK (source_type IN ('expert', 'customer')),
    source        TEXT NOT NULL,          -- e.g. 'rtings.com', 'Amazon'
    title         TEXT,
    url           TEXT,
    text          TEXT NOT NULL,          -- text the model scored
    stars         REAL NOT NULL,          -- model's predicted rating (1-5)
    stated_score  REAL                    -- reviewer's own score out of 5, if they gave one
);

CREATE TABLE IF NOT EXISTS aspect_mentions (
    review_id   INTEGER NOT NULL REFERENCES reviews(id),
    aspect      TEXT NOT NULL,
    sentiment   TEXT NOT NULL CHECK (sentiment IN ('positive', 'negative', 'neutral')),
    quote       TEXT
);

CREATE TABLE IF NOT EXISTS offers (
    search_id   INTEGER NOT NULL REFERENCES searches(id),
    store       TEXT NOT NULL,
    title       TEXT,
    url         TEXT,
    price       REAL
);
