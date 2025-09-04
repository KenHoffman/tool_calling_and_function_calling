# memory_store_sqlite.py
# Minimal "memory tool" backed by SQLite (FTS5 + JSON1), no external deps.

from __future__ import annotations
import sqlite3
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple

ISO_FMT = "%Y-%m-%d %H:%M:%S"  # naive UTC

def utc_now_str() -> str:
    return datetime.utcnow().strftime(ISO_FMT)

def add_days(dt: datetime, days: int) -> str:
    return (dt + timedelta(days=days)).strftime(ISO_FMT)

@dataclass
class Memory:
    id: int
    user_id: str
    text: str
    tags: Optional[List[str]]
    created_at: str
    expires_at: Optional[str]

class MemoryStore:
    """
    SQLite-backed memory store suitable for LLM tool-calling.
    Schema:
      - memories: canonical rows (soft delete via deleted_at)
      - memories_fts: FTS5 index mirroring memories.text (BM25 ranking)
    Requires SQLite compiled with FTS5 and JSON1 (common in modern Python builds).
    """

    def __init__(self, path: str = ":memory:"):
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    # ---------- Public "tool-like" API ----------

    def memory_add(
            self,
            user_id: str,
            text: str,
            tags: Optional[List[str]] = None,
            ttl_days: Optional[int] = None,
    ) -> int:
        """Add or refresh a memory. Returns memory id."""
        now = utc_now_str()
        expires = add_days(datetime.utcnow(), ttl_days) if ttl_days else None
        tags_json = json.dumps(tags) if tags else None

        # Upsert semantics: if same user_id + exact text exists (and not soft-deleted),
        # refresh timestamps/expiry and optionally replace tags if provided.
        with self.conn:
            existing = self.conn.execute(
                """
                SELECT id, tags FROM memories
                WHERE user_id = ? AND text = ? AND deleted_at IS NULL
                """,
                (user_id, text),
            ).fetchone()

            if existing:
                if tags_json is not None:
                    self.conn.execute(
                        """
                        UPDATE memories
                        SET tags = ?, created_at = ?, expires_at = ?
                        WHERE id = ?
                        """,
                        (tags_json, now, expires, existing["id"]),
                    )
                else:
                    self.conn.execute(
                        """
                        UPDATE memories
                        SET created_at = ?, expires_at = ?
                        WHERE id = ?
                        """,
                        (now, expires, existing["id"]),
                    )
                return int(existing["id"])
            else:
                cur = self.conn.execute(
                    """
                    INSERT INTO memories (user_id, text, tags, created_at, expires_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (user_id, text, tags_json, now, expires),
                )
                return int(cur.lastrowid)

    def memory_search(
            self,
            user_id: str,
            query: str,
            top_k: int = 5,
            tag_any: Optional[List[str]] = None,
            include_expired: bool = False,
    ) -> List[Memory]:
        """Ranked search (FTS5 BM25). Falls back to LIKE if FTS unavailable."""
        now = utc_now_str()
        conds, params = ["m.user_id = ?", "m.deleted_at IS NULL"], [user_id]
        if not include_expired:
            conds.append("(m.expires_at IS NULL OR m.expires_at > ?)")
            params.append(now)

        # Tag filter (any match) using JSON1
        tag_join = ""
        if tag_any:
            tag_placeholders = ",".join("?" * len(tag_any))
            tag_join = f"""
                AND EXISTS (
                    SELECT 1 FROM json_each(m.tags)
                    WHERE json_each.value IN ({tag_placeholders})
                )
            """
            params.extend(tag_any)

        # Try FTS5 first
        if self._fts_enabled():
            sql = f"""
                SELECT m.id, m.user_id, m.text, m.tags, m.created_at, m.expires_at,
                       bm25(memories_fts) AS score
                FROM memories_fts
                JOIN memories m ON m.id = memories_fts.rowid
                WHERE memories_fts MATCH ?
                  AND {' AND '.join(conds)} {tag_join}
                ORDER BY score
                LIMIT ?
            """
            rows = self.conn.execute(sql, (query, *params, top_k)).fetchall()
        else:
            # Fallback: LIKE search, newest first (no BM25)
            sql = f"""
                SELECT m.id, m.user_id, m.text, m.tags, m.created_at, m.expires_at
                FROM memories m
                WHERE {' AND '.join(conds)} {tag_join}
                  AND m.text LIKE ?
                ORDER BY m.created_at DESC
                LIMIT ?
            """
            rows = self.conn.execute(sql, (*params, f"%{query}%", top_k)).fetchall()

        return [self._row_to_memory(r) for r in rows]

    def memory_delete(self, user_id: str, ids: List[int]) -> int:
        """Soft-delete by id for a given user. Returns count updated."""
        if not ids:
            return 0
        now = utc_now_str()
        placeholders = ",".join("?" * len(ids))
        with self.conn:
            cur = self.conn.execute(
                f"""
                UPDATE memories
                SET deleted_at = ?
                WHERE user_id = ? AND id IN ({placeholders}) AND deleted_at IS NULL
                """,
                (now, user_id, *ids),
            )
            return cur.rowcount

    def purge(self) -> Tuple[int, int]:
        """
        Hard-delete expired or soft-deleted rows (keeps DB tidy).
        Returns (deleted_count, fts_affected_estimate).
        """
        now = utc_now_str()
        with self.conn:
            cur = self.conn.execute(
                """
                DELETE FROM memories
                WHERE deleted_at IS NOT NULL
                   OR (expires_at IS NOT NULL AND expires_at <= ?)
                """,
                (now,),
            )
            # FTS triggers will keep the index in sync.
            return cur.rowcount, cur.rowcount

    # ---------- Internal helpers ----------

    def _row_to_memory(self, r: sqlite3.Row) -> Memory:
        tags = json.loads(r["tags"]) if r["tags"] else None
        return Memory(
            id=int(r["id"]),
            user_id=r["user_id"],
            text=r["text"],
            tags=tags,
            created_at=r["created_at"],
            expires_at=r["expires_at"],
        )

    def _fts_enabled(self) -> bool:
        try:
            self.conn.execute("SELECT 1 FROM memories_fts LIMIT 1")
            return True
        except sqlite3.DatabaseError:
            return False

    def _init_db(self) -> None:
        with self.conn:
            # Defensive pragmas for durability/perf in typical apps
            self.conn.execute("PRAGMA foreign_keys = ON;")
            self.conn.execute("PRAGMA journal_mode = WAL;")
            self.conn.execute("PRAGMA synchronous = NORMAL;")

            # Core table
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                                                        id          INTEGER PRIMARY KEY,
                                                        user_id     TEXT NOT NULL,
                                                        text        TEXT NOT NULL,
                                                        tags        TEXT,                 -- JSON array (e.g., ["pet","name"])
                                                        created_at  TEXT NOT NULL,        -- UTC ISO (naive)
                                                        expires_at  TEXT,                 -- UTC ISO (naive) or NULL
                                                        deleted_at  TEXT                  -- NULL when active; non-NULL = soft-deleted
                );
                """
            )
            # Useful index for per-user queries and expiry checks
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_mem_user ON memories(user_id, created_at DESC);"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_mem_active ON memories(deleted_at, expires_at);"
            )

            # Try to set up FTS5 (if available)
            try:
                self.conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
                    USING fts5(
                        text,
                        content='memories',
                        content_rowid='id'
                    );
                    """
                )
                # Triggers to keep FTS in sync
                self.conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                      INSERT INTO memories_fts(rowid, text) VALUES (new.id, new.text);
                    END;
                    """
                )
                self.conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                      INSERT INTO memories_fts(memories_fts, rowid, text)
                      VALUES('delete', old.id, old.text);
                    END;
                    """
                )
                self.conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE OF text ON memories BEGIN
                                                                   INSERT INTO memories_fts(memories_fts, rowid, text)
                                                                   VALUES('delete', old.id, old.text);
                    INSERT INTO memories_fts(rowid, text) VALUES (new.id, new.text);
                    END;
                    """
                )
            except sqlite3.DatabaseError:
                # FTS5 not available; searches will fall back to LIKE.
                pass


# ---------------------------- Demo & Usage ----------------------------
if __name__ == "__main__":
    store = MemoryStore("memory.db")

    uid = "ken"

    # Add memories
    m1 = store.memory_add(uid, "User’s dog = Nova", tags=["pet", "name"])
    m2 = store.memory_add(uid, "Prefers character-driven first-contact audiobooks",
                          tags=["preference", "audiobooks"], ttl_days=365)
    m3 = store.memory_add(uid, "Lives in Colorado; likes local meetups",
                          tags=["location", "meetups"])

    # Search (FTS BM25 if available)
    print("\nSearch: 'dog name'")
    for mem in store.memory_search(uid, query='"dog" OR name', top_k=3):
        print(mem)

    print("\nSearch by tag (any of ['audiobooks','meetups'])")
    for mem in store.memory_search(uid, query="first-contact OR meetups", tag_any=["audiobooks", "meetups"]):
        print(mem)

    # Soft-delete one
    deleted = store.memory_delete(uid, [m1])
    print(f"\nSoft-deleted {deleted} row(s)")

    # Purge expired/soft-deleted (hard-delete)
    purged, _ = store.purge()
    print(f"Purge removed {purged} row(s)")