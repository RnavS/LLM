from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import secrets
from typing import Any, Dict, List
from uuid import uuid4

from server.supabase_client import SupabaseClient


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _iso_window_start(window_seconds: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=window_seconds)).replace(microsecond=0).isoformat()


def _hash_api_key(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def _memory_key(category: str, summary: str) -> str:
    normalized = f"{category.strip().lower()}::{summary.strip().lower()}"
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


class SupabaseConversationStore:
    def __init__(self, client: SupabaseClient):
        self.client = client

    def _json_loads(self, raw: str | None) -> Dict[str, Any]:
        if not raw:
            return {}
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    def ensure_owner_record(self, owner_id: str, email: str = "") -> None:
        current = self.get_profile(owner_id)
        profile = dict(current.get("profile") or {})
        if email and not profile.get("email"):
            profile["email"] = email
        self.client.insert_rows(
            "profiles",
            [
                {
                    "owner_id": owner_id,
                    "profile_json": json.dumps(profile),
                    "created_at": current["created_at"],
                    "updated_at": _utc_now(),
                }
            ],
            upsert=True,
            on_conflict=["owner_id"],
        )

    def _conversation_summary_from_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": str(row.get("id", "")).strip(),
            "title": str(row.get("title", "")).strip() or "New conversation",
            "system_preset": str(row.get("system_preset", "")).strip(),
            "system_prompt": str(row.get("system_prompt", "")).strip(),
            "settings": self._json_loads(str(row.get("settings_json", "") or "")),
            "created_at": str(row.get("created_at", "")).strip(),
            "updated_at": str(row.get("updated_at", "")).strip(),
            "message_count": int(row.get("message_count", 0) or 0),
            "preview": str(row.get("preview", "") or ""),
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
        self.client.insert_rows(
            "conversations",
            [
                {
                    "id": conversation_id,
                    "owner_id": owner_id,
                    "title": title.strip() or "New conversation",
                    "system_preset": system_preset.strip() or "medbrief-medical",
                    "system_prompt": system_prompt.strip(),
                    "settings_json": json.dumps(settings or {}),
                    "message_count": 0,
                    "preview": "",
                    "created_at": now,
                    "updated_at": now,
                }
            ],
        )
        return self.get_conversation(owner_id, conversation_id)

    def list_conversations(self, owner_id: str) -> List[Dict[str, Any]]:
        response = self.client.select_rows(
            "conversations",
            filters={"owner_id": f"eq.{owner_id}"},
            order="updated_at.desc",
        )
        return [self._conversation_summary_from_row(row) for row in list(response.body or [])]

    def get_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        response = self.client.select_rows(
            "messages",
            filters={"conversation_id": f"eq.{conversation_id}"},
            order="created_at.asc",
        )
        return [
            {
                "id": row.get("id"),
                "role": row.get("role"),
                "content": row.get("content"),
                "metadata": self._json_loads(str(row.get("metadata_json", "") or "")),
                "created_at": row.get("created_at"),
            }
            for row in list(response.body or [])
        ]

    def get_conversation(self, owner_id: str, conversation_id: str) -> Dict[str, Any]:
        response = self.client.select_rows(
            "conversations",
            filters={"id": f"eq.{conversation_id}", "owner_id": f"eq.{owner_id}"},
            single=True,
        )
        row = dict(response.body or {})
        if not row:
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
        payload = {
            "title": title if title is not None else current["title"],
            "system_preset": system_preset if system_preset is not None else current["system_preset"],
            "system_prompt": system_prompt if system_prompt is not None else current["system_prompt"],
            "settings_json": json.dumps(settings if settings is not None else current["settings"]),
            "updated_at": _utc_now() if touch else current["updated_at"],
        }
        rows = self.client.update_rows(
            "conversations",
            filters={"id": f"eq.{conversation_id}", "owner_id": f"eq.{owner_id}"},
            payload=payload,
        )
        if not rows:
            raise KeyError(conversation_id)
        return self.get_conversation(owner_id, conversation_id)

    def delete_conversation(self, owner_id: str, conversation_id: str) -> None:
        self.get_conversation(owner_id, conversation_id)
        self.client.delete_rows("messages", filters={"conversation_id": f"eq.{conversation_id}"})
        self.client.delete_rows(
            "conversation_summaries",
            filters={"conversation_id": f"eq.{conversation_id}", "owner_id": f"eq.{owner_id}"},
        )
        self.client.delete_rows("request_logs", filters={"conversation_id": f"eq.{conversation_id}"})
        deleted = self.client.delete_rows(
            "conversations",
            filters={"id": f"eq.{conversation_id}", "owner_id": f"eq.{owner_id}"},
        )
        if not deleted:
            raise KeyError(conversation_id)

    def add_message(
        self,
        owner_id: str,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        conversation = self.get_conversation(owner_id, conversation_id)
        now = _utc_now()
        message_id = uuid4().hex
        self.client.insert_rows(
            "messages",
            [
                {
                    "id": message_id,
                    "conversation_id": conversation_id,
                    "owner_id": owner_id,
                    "role": role,
                    "content": content,
                    "metadata_json": json.dumps(metadata or {}),
                    "created_at": now,
                }
            ],
        )
        preview = conversation["preview"]
        if role == "assistant" or not preview:
            preview = content[:220]
        self.client.update_rows(
            "conversations",
            filters={"id": f"eq.{conversation_id}", "owner_id": f"eq.{owner_id}"},
            payload={
                "updated_at": now,
                "message_count": int(conversation.get("message_count", 0) or 0) + 1,
                "preview": preview,
            },
        )
        return {
            "id": message_id,
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "created_at": now,
        }

    def replace_conversation_messages(
        self,
        owner_id: str,
        conversation_id: str,
        messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        conversation = self.get_conversation(owner_id, conversation_id)
        normalized_messages: List[Dict[str, Any]] = []
        preview = ""
        now = _utc_now()
        for item in messages:
            role = str(item.get("role", "")).strip().lower()
            content = str(item.get("content", "")).strip()
            if role not in {"system", "user", "assistant"} or not content:
                continue
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            normalized_messages.append(
                {
                    "id": uuid4().hex,
                    "conversation_id": conversation_id,
                    "owner_id": owner_id,
                    "role": role,
                    "content": content,
                    "metadata_json": json.dumps(metadata),
                    "created_at": now,
                }
            )
            if role == "assistant" or not preview:
                preview = content[:220]

        self.client.delete_rows("messages", filters={"conversation_id": f"eq.{conversation_id}"})
        if normalized_messages:
            self.client.insert_rows("messages", normalized_messages)
        self.client.update_rows(
            "conversations",
            filters={"id": f"eq.{conversation_id}", "owner_id": f"eq.{owner_id}"},
            payload={
                "updated_at": now,
                "message_count": len(normalized_messages),
                "preview": preview or conversation.get("preview", ""),
            },
        )
        return self.get_conversation(owner_id, conversation_id)

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
        self.client.insert_rows(
            "request_logs",
            [
                {
                    "id": uuid4().hex,
                    "owner_id": owner_id or "",
                    "conversation_id": conversation_id,
                    "api_key_id": api_key_id,
                    "route": route,
                    "status_code": int(status_code),
                    "latency_ms": float(latency_ms),
                    "metadata_json": json.dumps(metadata or {}),
                    "created_at": _utc_now(),
                }
            ],
        )

    def get_profile(self, owner_id: str) -> Dict[str, Any]:
        response = self.client.select_rows(
            "profiles",
            filters={"owner_id": f"eq.{owner_id}"},
            single=True,
        )
        row = dict(response.body or {})
        if not row:
            now = _utc_now()
            return {
                "owner_id": owner_id,
                "profile": {},
                "created_at": now,
                "updated_at": now,
            }
        return {
            "owner_id": row.get("owner_id"),
            "profile": self._json_loads(str(row.get("profile_json", "") or "")),
            "created_at": row.get("created_at"),
            "updated_at": row.get("updated_at"),
        }

    def upsert_profile(self, owner_id: str, profile: Dict[str, Any]) -> Dict[str, Any]:
        current = self.get_profile(owner_id)
        merged = dict(current["profile"])
        merged.update(profile)
        self.client.insert_rows(
            "profiles",
            [
                {
                    "owner_id": owner_id,
                    "profile_json": json.dumps(merged),
                    "created_at": current["created_at"],
                    "updated_at": _utc_now(),
                }
            ],
            upsert=True,
            on_conflict=["owner_id"],
        )
        return self.get_profile(owner_id)

    def list_memory_items(self, owner_id: str) -> List[Dict[str, Any]]:
        response = self.client.select_rows(
            "memory_items",
            filters={"owner_id": f"eq.{owner_id}"},
            order="last_used_at.desc,updated_at.desc",
        )
        return [
            {
                "id": row.get("id"),
                "owner_id": row.get("owner_id"),
                "memory_key": row.get("memory_key"),
                "category": row.get("category"),
                "summary": row.get("summary"),
                "context": self._json_loads(str(row.get("context_json", "") or "")),
                "created_at": row.get("created_at"),
                "updated_at": row.get("updated_at"),
                "last_used_at": row.get("last_used_at"),
            }
            for row in list(response.body or [])
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
        for item in cleaned_items:
            memory_key = _memory_key(item["category"], item["summary"])
            current = existing.get(memory_key)
            if current:
                merged_context = dict(current.get("context") or {})
                merged_context.update(item["context"])
                self.client.update_rows(
                    "memory_items",
                    filters={"id": f"eq.{current['id']}", "owner_id": f"eq.{owner_id}"},
                    payload={
                        "summary": item["summary"],
                        "context_json": json.dumps(merged_context),
                        "updated_at": now,
                        "last_used_at": now,
                    },
                )
            else:
                self.client.insert_rows(
                    "memory_items",
                    [
                        {
                            "id": uuid4().hex,
                            "owner_id": owner_id,
                            "memory_key": memory_key,
                            "category": item["category"],
                            "summary": item["summary"],
                            "context_json": json.dumps(item["context"]),
                            "created_at": now,
                            "updated_at": now,
                            "last_used_at": now,
                        }
                    ],
                )
        rows = self.list_memory_items(owner_id)
        stale = rows[18:]
        for item in stale:
            self.client.delete_rows(
                "memory_items",
                filters={"id": f"eq.{item['id']}", "owner_id": f"eq.{owner_id}"},
            )
        return self.list_memory_items(owner_id)

    def delete_memory_item(self, owner_id: str, memory_id: str) -> None:
        deleted = self.client.delete_rows(
            "memory_items",
            filters={"id": f"eq.{memory_id}", "owner_id": f"eq.{owner_id}"},
        )
        if not deleted:
            raise KeyError(memory_id)

    def clear_memory(self, owner_id: str) -> None:
        self.client.delete_rows("memory_items", filters={"owner_id": f"eq.{owner_id}"})

    def get_conversation_summary(self, owner_id: str, conversation_id: str) -> Dict[str, Any]:
        response = self.client.select_rows(
            "conversation_summaries",
            filters={"owner_id": f"eq.{owner_id}", "conversation_id": f"eq.{conversation_id}"},
            single=True,
        )
        row = dict(response.body or {})
        if not row:
            now = _utc_now()
            return {
                "conversation_id": conversation_id,
                "owner_id": owner_id,
                "summary": {},
                "created_at": now,
                "updated_at": now,
            }
        return {
            "conversation_id": row.get("conversation_id"),
            "owner_id": row.get("owner_id"),
            "summary": self._json_loads(str(row.get("summary_json", "") or "")),
            "created_at": row.get("created_at"),
            "updated_at": row.get("updated_at"),
        }

    def upsert_conversation_summary(self, owner_id: str, conversation_id: str, summary: Dict[str, Any]) -> Dict[str, Any]:
        current = self.get_conversation_summary(owner_id, conversation_id)
        merged = dict(current["summary"])
        merged.update(summary)
        self.client.insert_rows(
            "conversation_summaries",
            [
                {
                    "conversation_id": conversation_id,
                    "owner_id": owner_id,
                    "summary_json": json.dumps(merged),
                    "created_at": current["created_at"],
                    "updated_at": _utc_now(),
                }
            ],
            upsert=True,
            on_conflict=["conversation_id"],
        )
        return self.get_conversation_summary(owner_id, conversation_id)

    def _api_key_row_to_public(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": row.get("id"),
            "label": row.get("label"),
            "key_prefix": row.get("key_prefix"),
            "created_at": row.get("created_at"),
            "last_used_at": row.get("last_used_at"),
            "revoked_at": row.get("revoked_at"),
            "usage_count": int(row.get("usage_count", 0) or 0),
            "rate_limits": self._json_loads(str(row.get("rate_limit_json", "") or "")),
        }

    def list_api_keys(self, owner_id: str) -> List[Dict[str, Any]]:
        response = self.client.select_rows(
            "api_keys",
            filters={"owner_id": f"eq.{owner_id}"},
            order="created_at.desc",
        )
        rows = [self._api_key_row_to_public(row) for row in list(response.body or [])]
        rows.sort(key=lambda item: (item.get("revoked_at") is not None, item.get("created_at") or ""), reverse=False)
        return rows

    def create_api_key(self, owner_id: str, label: str, rate_limits: Dict[str, Any]) -> Dict[str, Any]:
        now = _utc_now()
        api_key_id = uuid4().hex
        secret = f"mbai_{secrets.token_urlsafe(24)}"
        key_prefix = f"{secret[:14]}…{secret[-4:]}"
        self.client.update_rows(
            "api_keys",
            filters={"owner_id": f"eq.{owner_id}", "revoked_at": "is.null"},
            payload={"revoked_at": now},
        )
        row = {
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
        self.client.insert_rows("api_keys", [row])
        return {
            "secret": secret,
            "record": self._api_key_row_to_public(row),
        }

    def get_api_key_by_secret(self, secret: str) -> Dict[str, Any] | None:
        secret = secret.strip()
        if not secret:
            return None
        response = self.client.select_rows(
            "api_keys",
            filters={"key_hash": f"eq.{_hash_api_key(secret)}", "revoked_at": "is.null"},
            single=True,
        )
        row = dict(response.body or {})
        if not row:
            return None
        return {
            "id": row.get("id"),
            "owner_id": row.get("owner_id"),
            "label": row.get("label"),
            "key_prefix": row.get("key_prefix"),
            "key_hash": row.get("key_hash"),
            "created_at": row.get("created_at"),
            "last_used_at": row.get("last_used_at"),
            "revoked_at": row.get("revoked_at"),
            "usage_count": int(row.get("usage_count", 0) or 0),
            "rate_limits": self._json_loads(str(row.get("rate_limit_json", "") or "")),
        }

    def touch_api_key(self, api_key_id: str) -> None:
        current = self.client.select_rows(
            "api_keys",
            filters={"id": f"eq.{api_key_id}"},
            single=True,
        )
        row = dict(current.body or {})
        if not row:
            return
        self.client.update_rows(
            "api_keys",
            filters={"id": f"eq.{api_key_id}"},
            payload={
                "last_used_at": _utc_now(),
                "usage_count": int(row.get("usage_count", 0) or 0) + 1,
            },
        )

    def revoke_api_key(self, owner_id: str, api_key_id: str) -> None:
        current = self.client.select_rows(
            "api_keys",
            filters={"id": f"eq.{api_key_id}", "owner_id": f"eq.{owner_id}", "revoked_at": "is.null"},
            single=True,
        )
        if not dict(current.body or {}):
            raise KeyError(api_key_id)
        updated = self.client.update_rows(
            "api_keys",
            filters={"id": f"eq.{api_key_id}", "owner_id": f"eq.{owner_id}", "revoked_at": "is.null"},
            payload={"revoked_at": _utc_now()},
        )
        if not updated:
            raise KeyError(api_key_id)

    def count_recent_api_key_requests(self, api_key_id: str, window_seconds: int) -> int:
        since = _iso_window_start(window_seconds)
        return self.client.count_rows(
            "request_logs",
            filters={"api_key_id": f"eq.{api_key_id}", "created_at": f"gte.{since}"},
        )
