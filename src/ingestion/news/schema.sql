CREATE TABLE IF NOT EXISTS articles (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    date              TEXT NOT NULL,
    date_int          INTEGER NOT NULL,
    title             TEXT NOT NULL,
    url               TEXT NOT NULL,
    source            TEXT,
    dedup_cluster_id  INTEGER NOT NULL,
    is_canonical      INTEGER NOT NULL,
    lang              TEXT,
    gdelt_themes      TEXT,
    gdelt_locations   TEXT,
    gdelt_organizations TEXT,
    body              TEXT,
    body_status       TEXT
);

CREATE TABLE IF NOT EXISTS raw_news_scored_v2 (
    id                       INTEGER PRIMARY KEY,
    category                 TEXT,
    is_tradeable             INTEGER,
    skip_reason              TEXT,
    sentiment                INTEGER,
    impact                   INTEGER,
    certainty                INTEGER,
    weight                   INTEGER,
    final_signal             REAL,
    processed_at             TEXT,
    scoring_input_mode       TEXT,
    original_content_status  TEXT
);

CREATE INDEX IF NOT EXISTS idx_articles_cluster ON articles(dedup_cluster_id);
CREATE INDEX IF NOT EXISTS idx_articles_url ON articles(url);
CREATE INDEX IF NOT EXISTS idx_articles_date ON articles(date);
CREATE INDEX IF NOT EXISTS idx_articles_status ON articles(body_status);
CREATE INDEX IF NOT EXISTS idx_v2_category ON raw_news_scored_v2(category);
CREATE INDEX IF NOT EXISTS idx_v2_signal ON raw_news_scored_v2(final_signal);

CREATE VIEW IF NOT EXISTS v_articles_scored AS
SELECT
    a.id, a.date, a.date_int, a.title, a.url, a.source,
    a.dedup_cluster_id, a.is_canonical, a.lang,
    a.gdelt_themes, a.gdelt_locations, a.gdelt_organizations,
    a.body, a.body_status,
    s.category, s.is_tradeable, s.skip_reason,
    s.sentiment, s.impact, s.certainty,
    s.weight, s.final_signal,
    s.scoring_input_mode, s.original_content_status,
    s.processed_at
FROM articles a
LEFT JOIN articles canon
    ON canon.dedup_cluster_id = a.dedup_cluster_id AND canon.is_canonical = 1
LEFT JOIN raw_news_scored_v2 s
    ON s.id = canon.id;
