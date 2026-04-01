from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import secrets
import sqlite3
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

from server.settings import ServerSettings
from server.supabase_client import SupabaseClient
from server.supabase_storage import SupabaseConversationStore
from utils import ensure_dir


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _iso_window_start(window_seconds: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=window_seconds)).replace(microsecond=0).isoformat()


def _hash_api_key(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def _memory_key(category: str, summary: str) -> str:
    normalized = f"{category.strip().lower()}::{summary.strip().lower()}"
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


class ConversationStore:
    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)
        ensure_dir(self.database_path.parent)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL DEFAULT '',
                    title TEXT NOT NULL,
                    system_preset TEXT NOT NULL,
                    system_prompt TEXT NOT NULL,
                    settings_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS request_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    owner_id TEXT,
                    conversation_id TEXT,
                    api_key_id TEXT,
                    route TEXT NOT NULL,
                    status_code INTEGER NOT NULL,
                    latency_ms REAL NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS profiles (
                    owner_id TEXT PRIMARY KEY,
                    profile_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS memory_items (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    memory_key TEXT NOT NULL,
                    category TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    context_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_used_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    conversation_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    summary_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS api_keys (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    label TEXT NOT NULL,
                    key_prefix TEXT NOT NULL,
                    key_hash TEXT NOT NULL UNIQUE,
                    rate_limit_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    last_used_at TEXT,
                    revoked_at TEXT,
                    usage_count INTEGER NOT NULL DEFAULT 0
                );
                """
            )
            conversation_columns = {row["name"] for row in connection.execute("PRAGMA table_info(conversations)").fetchall()}
            if "owner_id" not in conversation_columns:
                connection.execute("ALTER TABLE conversations ADD COLUMN owner_id TEXT NOT NULL DEFAULT ''")
            request_log_columns = {row["name"] for row in connection.execute("PRAGMA table_info(request_logs)").fetchall()}
            if "owner_id" not in request_log_columns:
                connection.execute("ALTER TABLE request_logs ADD COLUMN owner_id TEXT")
            if "api_key_id" not in request_log_columns:
                connection.execute("ALTER TABLE request_logs ADD COLUMN api_key_id TEXT")
            connection.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_conversations_owner_updated ON conversations(owner_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, id ASC);
                CREATE INDEX IF NOT EXISTS idx_request_logs_key_created ON request_logs(api_key_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_memory_items_owner_used ON memory_items(owner_id, last_used_at DESC);
                CREATE INDEX IF NOT EXISTS idx_api_keys_owner_created ON api_keys(owner_id, created_at DESC);
                """
            )

    def _json_loads(self, raw: str | None) -> Dict[str, Any]:
        if not raw:
            return {}
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    def _conversation_summary_from_row(self, row: sqlite3.Row) -> Dict[str, Any]:
        preview = row["preview"] or ""
        return {
            "id": row["id"],
            "title": row["title"],
            "system_preset": row["system_preset"],
            "system_prompt": row["system_prompt"],
            "settings": self._json_loads(row["settings_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "message_count": int(row["message_count"] or 0),
            "preview": str(preview),
        }

    def create_conversation(
        self,
        owner_id: str,
        title: str = "New conversation",
        system_preset: str = "medbrief-medical",
        system_prompt: str = "",
        settings: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        conversation_id = uuid4().hex
        now = _utc_now()
        payload = {
            "id": conversation_id,
            "owner_id": owner_id,
            "title": title.strip() or "New conversation",
            "system_preset": system_preset.strip() or "medbrief-medical",
            "system_prompt": system_prompt.strip(),
            "settings_json": json.dumps(settings or {}),
            "created_at": now,
            "updated_at": now,
        }
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversations (id, owner_id, title, system_preset, system_prompt, settings_json, created_at, updated_at)
                VALUES (:id, :owner_id, :title, :system_preset, :system_prompt, :settings_json, :created_at, :updated_at)
                """,
                payload,
            )
        return self.get_conversation(owner_id, conversation_id)

    def list_conversations(self, owner_id: str) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    conversations.*,
                    COUNT(messages.id) AS message_count,
                    COALESCE(
                        MAX(CASE WHEN messages.role = 'assistant' THEN substr(messages.content, 1, 220) END),
                        MAX(CASE WHEN messages.role = 'user' THEN substr(messages.content, 1, 220) END),
                        ''
                    ) AS preview
                FROM conversations
                LEFT JOIN messages ON messages.conversation_id = conversations.id
                WHERE conversations.owner_id = ?
                GROUP BY conversations.id
                ORDER BY conversations.updated_at DESC
                """,
                (owner_id,),
            ).fetchall()
        return [self._conversation_summary_from_row(row) for row in rows]

    def get_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, role, content, metadata_json, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id ASC
                """,
                (conversation_id,),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "role": row["role"],
                "content": row["content"],
                "metadata": self._json_loads(row["metadata_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def get_conversation(self, owner_id: str, conversation_id: str) -> Dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    conversations.*,
                    COUNT(messages.id) AS message_count,
                    COALESCE(
                        MAX(CASE WHEN messages.role = 'assistant' THEN substr(messages.content, 1, 220) END),
                        MAX(CASE WHEN messages.role = 'user' THEN substr(messages.content, 1, 220) END),
                        ''
                    ) AS preview
                FROM conversations
                LEFT JOIN messages ON messages.conversation_id = conversations.id
                WHERE conversations.id = ? AND conversations.owner_id = ?
                GROUP BY conversations.id
                """,
                (conversation_id, owner_id),
            ).fetchone()
        if row is None:
            raise KeyError(conversation_id)
        conversation = self._conversation_summary_from_row(row)
        conversation["messages"] = self.get_messages(conversation_id)
        conversation["summary_state"] = self.get_conversation_summary(owner_id, conversation_id).get("summary", {})
        return conversation

    def update_conversation(
        self,
        owner_id: str,
        conversation_id: str,
        *,
        title: str | None = None,
        system_preset: str | None = None,
        system_prompt: str | None = None,
        settings: Dict[str, Any] | None = None,
        touch: bool = True,
    ) -> Dict[str, Any]:
        current = self.get_conversation(owner_id, conversation_id)
        updated_title = title if title is not None else current["title"]
        updated_system_preset = system_preset if system_preset is not None else current["system_preset"]
        updated_system_prompt = system_prompt if system_prompt is not None else current["system_prompt"]
        updated_settings = settings if settings is not None else current["settings"]
        updated_at = _utc_now() if touch else current["updated_at"]
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE conversations
                SET title = ?, system_preset = ?, system_prompt = ?, settings_json = ?, updated_at = ?
                WHERE id = ? AND owner_id = ?
                """,
                (
                    updated_title,
                    updated_system_preset,
                    updated_system_prompt,
                    json.dumps(updated_settings or {}),
                    updated_at,
                    conversation_id,
                    owner_id,
                ),
            )
        return self.get_conversation(owner_id, conversation_id)

    def delete_conversation(self, owner_id: str, conversation_id: str) -> None:
        self.get_conversation(owner_id, conversation_id)
        with self._connect() as connection:
            connection.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            connection.execute("DELETE FROM conversation_summaries WHERE conversation_id = ? AND owner_id = ?", (conversation_id, owner_id))
            deleted = connection.execute(
                "DELETE FROM conversations WHERE id = ? AND owner_id = ?",
                (conversation_id, owner_id),
            ).rowcount
        if deleted == 0:
            raise KeyError(conversation_id)

    def add_message(
        self,
        owner_id: str,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        self.get_conversation(owner_id, conversation_id)
        now = _utc_now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO messages (conversation_id, role, content, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (conversation_id, role, content, json.dumps(metadata or {}), now),
            )
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
            message_id = int(cursor.lastrowid)
        return {
            "id": message_id,
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "created_at": now,
        }

    def log_request(
        self,
        *,
        owner_id: str | None = None,
        route: str,
        status_code: int,
        latency_ms: float,
        conversation_id: str | None = None,
        api_key_id: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO request_logs (owner_id, conversation_id, api_key_id, route, status_code, latency_ms, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    owner_id,
                    conversation_id,
                    api_key_id,
                    route,
                    status_code,
                    float(latency_ms),
                    json.dumps(metadata or {}),
                    _utc_now(),
                ),
            )

    def get_profile(self, owner_id: str) -> Dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT owner_id, profile_json, created_at, updated_at
                FROM profiles
                WHERE owner_id = ?
                """,
                (owner_id,),
            ).fetchone()
        if row is None:
            now = _utc_now()
            return {
                "owner_id": owner_id,
                "profile": {},
                "created_at": now,
                "updated_at": now,
            }
        return {
            "owner_id": row["owner_id"],
            "profile": self._json_loads(row["profile_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def upsert_profile(self, owner_id: str, profile: Dict[str, Any]) -> Dict[str, Any]:
        current = self.get_profile(owner_id)
        merged = dict(current["profile"])
        merged.update(profile)
        now = _utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO profiles (owner_id, profile_json, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(owner_id) DO UPDATE SET
                    profile_json = excluded.profile_json,
                    updated_at = excluded.updated_at
                """,
                (owner_id, json.dumps(merged), current["created_at"], now),
            )
        return self.get_profile(owner_id)

    def list_memory_items(self, owner_id: str) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, owner_id, memory_key, category, summary, context_json, created_at, updated_at, last_used_at
                FROM memory_items
                WHERE owner_id = ?
                ORDER BY last_used_at DESC, updated_at DESC
                """,
                (owner_id,),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "owner_id": row["owner_id"],
                "memory_key": row["memory_key"],
                "category": row["category"],
                "summary": row["summary"],
                "context": self._json_loads(row["context_json"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "last_used_at": row["last_used_at"],
            }
            for row in rows
        ]

    def save_memory_items(self, owner_id: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cleaned_items = []
        for item in items:
            category = str(item.get("category", "")).strip().lower() or "context"
            summary = " ".join(str(item.get("summary", "")).split()).strip()
            if not summary:
                continue
            cleaned_items.append(
                {
                    "category": category,
                    "summary": summary,
                    "context": item.get("context") if isinstance(item.get("context"), dict) else {},
                }
            )
        if not cleaned_items:
            return self.list_memory_items(owner_id)

        existing = {item["memory_key"]: item for item in self.list_memory_items(owner_id)}
        now = _utc_now()
        with self._connect() as connection:
            for item in cleaned_items:
                memory_key = _memory_key(item["category"], item["summary"])
                current = existing.get(memory_key)
                if current:
                    merged_context = dict(current.get("context") or {})
                    merged_context.update(item["context"])
                    connection.execute(
                        """
                        UPDATE memory_items
                        SET summary = ?, context_json = ?, updated_at = ?, last_used_at = ?
                        WHERE id = ? AND owner_id = ?
                        """,
                        (
                            item["summary"],
                            json.dumps(merged_context),
                            now,
                            now,
                            current["id"],
                            owner_id,
                        ),
                    )
                else:
                    connection.execute(
                        """
                        INSERT INTO memory_items (id, owner_id, memory_key, category, summary, context_json, created_at, updated_at, last_used_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            uuid4().hex,
                            owner_id,
                            memory_key,
                            item["category"],
                            item["summary"],
                            json.dumps(item["context"]),
                            now,
                            now,
                            now,
                        ),
                    )
            rows = connection.execute(
                """
                SELECT id
                FROM memory_items
                WHERE owner_id = ?
                ORDER BY last_used_at DESC, updated_at DESC
                """,
                (owner_id,),
            ).fetchall()
            stale_ids = [row["id"] for row in rows[18:]]
            if stale_ids:
                connection.executemany("DELETE FROM memory_items WHERE id = ? AND owner_id = ?", [(memory_id, owner_id) for memory_id in stale_ids])
        return self.list_memory_items(owner_id)

    def delete_memory_item(self, owner_id: str, memory_id: str) -> None:
        with self._connect() as connection:
            deleted = connection.execute(
                "DELETE FROM memory_items WHERE id = ? AND owner_id = ?",
                (memory_id, owner_id),
            ).rowcount
        if deleted == 0:
            raise KeyError(memory_id)

    def clear_memory(self, owner_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM memory_items WHERE owner_id = ?", (owner_id,))

    def get_conversation_summary(self, owner_id: str, conversation_id: str) -> Dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT conversation_id, owner_id, summary_json, created_at, updated_at
                FROM conversation_summaries
                WHERE owner_id = ? AND conversation_id = ?
                """,
                (owner_id, conversation_id),
            ).fetchone()
        if row is None:
            now = _utc_now()
            return {
                "conversation_id": conversation_id,
                "owner_id": owner_id,
                "summary": {},
                "created_at": now,
                "updated_at": now,
            }
        return {
            "conversation_id": row["conversation_id"],
            "owner_id": row["owner_id"],
            "summary": self._json_loads(row["summary_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def upsert_conversation_summary(self, owner_id: str, conversation_id: str, summary: Dict[str, Any]) -> Dict[str, Any]:
        current = self.get_conversation_summary(owner_id, conversation_id)
        merged = dict(current["summary"])
        merged.update(summary)
        now = _utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversation_summaries (conversation_id, owner_id, summary_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    owner_id = excluded.owner_id,
                    summary_json = excluded.summary_json,
                    updated_at = excluded.updated_at
                """,
                (conversation_id, owner_id, json.dumps(merged), current["created_at"], now),
            )
        return self.get_conversation_summary(owner_id, conversation_id)

    def _api_key_row_to_public(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "label": row["label"],
            "key_prefix": row["key_prefix"],
            "created_at": row["created_at"],
            "last_used_at": row["last_used_at"],
            "revoked_at": row["revoked_at"],
            "usage_count": int(row["usage_count"] or 0),
            "rate_limits": self._json_loads(row["rate_limit_json"]),
        }

    def list_api_keys(self, owner_id: str) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, label, key_prefix, rate_limit_json, created_at, last_used_at, revoked_at, usage_count
                FROM api_keys
                WHERE owner_id = ?
                ORDER BY CASE WHEN revoked_at IS NULL THEN 0 ELSE 1 END ASC, created_at DESC
                """,
                (owner_id,),
            ).fetchall()
        return [self._api_key_row_to_public(row) for row in rows]

    def create_api_key(self, owner_id: str, label: str, rate_limits: Dict[str, Any]) -> Dict[str, Any]:
        now = _utc_now()
        api_key_id = uuid4().hex
        secret = f"mbai_{secrets.token_urlsafe(24)}"
        key_prefix = f"{secret[:14]}…{secret[-4:]}"
        payload = {
            "id": api_key_id,
            "owner_id": owner_id,
            "label": " ".join((label or "Project key").split())[:80] or "Project key",
            "key_prefix": key_prefix,
            "key_hash": _hash_api_key(secret),
            "rate_limit_json": json.dumps(rate_limits or {}),
            "created_at": now,
            "last_used_at": None,
            "revoked_at": None,
            "usage_count": 0,
        }
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO api_keys (id, owner_id, label, key_prefix, key_hash, rate_limit_json, created_at, last_used_at, revoked_at, usage_count)
                VALUES (:id, :owner_id, :label, :key_prefix, :key_hash, :rate_limit_json, :created_at, :last_used_at, :revoked_at, :usage_count)
                """,
                payload,
            )
            row = connection.execute(
                """
                SELECT id, label, key_prefix, rate_limit_json, created_at, last_used_at, revoked_at, usage_count
                FROM api_keys
                WHERE id = ?
                """,
                (api_key_id,),
            ).fetchone()
        return {
            "secret": secret,
            "record": self._api_key_row_to_public(row),
        }

    def get_api_key_by_secret(self, secret: str) -> Dict[str, Any] | None:
        secret = secret.strip()
        if not secret:
            return None
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, owner_id, label, key_prefix, key_hash, rate_limit_json, created_at, last_used_at, revoked_at, usage_count
                FROM api_keys
                WHERE key_hash = ?
                """,
                (_hash_api_key(secret),),
            ).fetchone()
        if row is None or row["revoked_at"]:
            return None
        return {
            "id": row["id"],
            "owner_id": row["owner_id"],
            "label": row["label"],
            "key_prefix": row["key_prefix"],
            "key_hash": row["key_hash"],
            "created_at": row["created_at"],
            "last_used_at": row["last_used_at"],
            "revoked_at": row["revoked_at"],
            "usage_count": int(row["usage_count"] or 0),
            "rate_limits": self._json_loads(row["rate_limit_json"]),
        }

    def touch_api_key(self, api_key_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE api_keys
                SET last_used_at = ?, usage_count = usage_count + 1
                WHERE id = ?
                """,
                (_utc_now(), api_key_id),
            )

    def revoke_api_key(self, owner_id: str, api_key_id: str) -> None:
        with self._connect() as connection:
            deleted = connection.execute(
                """
                UPDATE api_keys
                SET revoked_at = ?
                WHERE id = ? AND owner_id = ? AND revoked_at IS NULL
                """,
                (_utc_now(), api_key_id, owner_id),
            ).rowcount
        if deleted == 0:
            raise KeyError(api_key_id)

    def count_recent_api_key_requests(self, api_key_id: str, window_seconds: int) -> int:
        since = _iso_window_start(window_seconds)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM request_logs
                WHERE api_key_id = ? AND created_at >= ?
                """,
                (api_key_id, since),
            ).fetchone()
        return int(row["count"] or 0) if row else 0

    def ensure_owner_record(self, owner_id: str, email: str = "") -> None:
        current = self.get_profile(owner_id)
        profile = dict(current.get("profile") or {})
        if email and not profile.get("email"):
            profile["email"] = email
        self.upsert_profile(owner_id, profile)


def build_conversation_store(settings: ServerSettings) -> ConversationStore | SupabaseConversationStore:
    if settings.supabase_configured:
        return SupabaseConversationStore(
            SupabaseClient(
                url=settings.supabase_url,
                anon_key=settings.supabase_anon_key,
                service_role_key=settings.supabase_service_role_key,
                timeout_seconds=max(10.0, settings.generation_timeout_seconds),
            )
        )
    return ConversationStore(settings.database_path)
