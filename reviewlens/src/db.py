"""SQLite storage: every search, review, aspect mention, and price offer goes here,
and all ratings are calculated with SQL (see sql/queries.sql)."""
import json
import re
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "reviews.db"


def _load_queries() -> dict:
    text = (ROOT / "sql" / "queries.sql").read_text()
    parts = re.split(r"^-- name: (\w+)\s*$", text, flags=re.M)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


QUERIES = _load_queries()


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.executescript((ROOT / "sql" / "schema.sql").read_text())
    return conn


def new_search(conn, product: str) -> int:
    cur = conn.execute("INSERT INTO searches (product) VALUES (?)", (product,))
    conn.commit()
    return cur.lastrowid


def set_aspects(conn, search_id: int, aspects: list[str]):
    conn.execute("UPDATE searches SET aspects = ? WHERE id = ?", (json.dumps(aspects), search_id))
    conn.commit()


def get_aspects(conn, search_id: int) -> list[str]:
    row = conn.execute("SELECT aspects FROM searches WHERE id = ?", (search_id,)).fetchone()
    return json.loads(row[0]) if row and row[0] else []


def add_review(conn, search_id, source_type, source, text, stars, title=None, url=None,
               stated_score=None) -> int:
    cur = conn.execute(
        "INSERT INTO reviews (search_id, source_type, source, title, url, text, stars, stated_score)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (search_id, source_type, source, title, url, text, float(stars), stated_score))
    return cur.lastrowid


def add_aspects(conn, review_id, mentions):
    conn.executemany(
        "INSERT INTO aspect_mentions (review_id, aspect, sentiment, quote) VALUES (?, ?, ?, ?)",
        [(review_id, m["aspect"], m["sentiment"], m.get("quote")) for m in mentions])


def add_offers(conn, search_id, offers):
    conn.executemany(
        "INSERT INTO offers (search_id, store, title, url, price) VALUES (?, ?, ?, ?, ?)",
        [(search_id, o["store"], o["title"], o["url"], o["price"]) for o in offers])


def copy_expert_data(conn, from_search: int, to_search: int):
    """Reuse cached expert reviews and offers in a new search (no new API calls)."""
    old = conn.execute("SELECT id, source, title, url, text, stars, stated_score FROM reviews "
                       "WHERE search_id = ? AND source_type = 'expert'", (from_search,)).fetchall()
    for rid, source, title, url, text, stars, stated in old:
        new_id = add_review(conn, to_search, "expert", source, text, stars, title, url, stated)
        conn.execute("INSERT INTO aspect_mentions SELECT ?, aspect, sentiment, quote "
                     "FROM aspect_mentions WHERE review_id = ?", (new_id, rid))
    conn.execute("INSERT INTO offers SELECT ?, store, title, url, price FROM offers "
                 "WHERE search_id = ?", (to_search, from_search))
    conn.commit()


def query(conn, name: str, **params) -> pd.DataFrame:
    return pd.read_sql_query(QUERIES[name], conn, params=params)
