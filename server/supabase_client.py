from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Dict, Iterable, List, Mapping
from urllib import error, parse, request


@dataclass
class SupabaseResponse:
    body: Any
    headers: Mapping[str, str]
    status_code: int


class SupabaseClient:
    def __init__(
        self,
        *,
        url: str,
        anon_key: str,
        service_role_key: str,
        timeout_seconds: float = 20.0,
    ):
        self.url = url.rstrip("/")
        self.anon_key = anon_key.strip()
        self.service_role_key = service_role_key.strip()
        self.timeout_seconds = timeout_seconds

    @property
    def configured(self) -> bool:
        return bool(self.url and self.anon_key and self.service_role_key)

    def sign_up(self, *, email: str, password: str) -> Dict[str, Any]:
        response = self._request_json(
            "POST",
            "/auth/v1/signup",
            body={"email": email, "password": password},
            api_key=self.anon_key,
        )
        return dict(response.body or {})

    def sign_in_with_password(self, *, email: str, password: str) -> Dict[str, Any]:
        response = self._request_json(
            "POST",
            "/auth/v1/token",
            query={"grant_type": "password"},
            body={"email": email, "password": password},
            api_key=self.anon_key,
        )
        return dict(response.body or {})

    def select_rows(
        self,
        table: str,
        *,
        filters: Mapping[str, str] | None = None,
        select: str = "*",
        order: str | None = None,
        limit: int | None = None,
        single: bool = False,
        use_service_role: bool = True,
        prefer_count: bool = False,
    ) -> SupabaseResponse:
        query: Dict[str, str] = {"select": select}
        if filters:
            query.update(filters)
        if order:
            query["order"] = order
        if limit is not None:
            query["limit"] = str(limit)
        headers: Dict[str, str] = {}
        if single:
            headers["Accept"] = "application/vnd.pgrst.object+json"
        if prefer_count:
            headers["Prefer"] = "count=exact"
        return self._request_json(
            "GET",
            f"/rest/v1/{table}",
            query=query,
            headers=headers,
            api_key=self.service_role_key if use_service_role else self.anon_key,
        )

    def insert_rows(
        self,
        table: str,
        rows: List[Dict[str, Any]],
        *,
        upsert: bool = False,
        on_conflict: Iterable[str] | None = None,
    ) -> List[Dict[str, Any]]:
        prefer_parts = ["return=representation"]
        if upsert:
            prefer_parts.insert(0, "resolution=merge-duplicates")
        query: Dict[str, str] = {}
        if on_conflict:
            query["on_conflict"] = ",".join(on_conflict)
        response = self._request_json(
            "POST",
            f"/rest/v1/{table}",
            query=query,
            body=rows,
            headers={"Prefer": ",".join(prefer_parts)},
            api_key=self.service_role_key,
        )
        return list(response.body or [])

    def update_rows(
        self,
        table: str,
        *,
        filters: Mapping[str, str],
        payload: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        response = self._request_json(
            "PATCH",
            f"/rest/v1/{table}",
            query=dict(filters),
            body=payload,
            headers={"Prefer": "return=representation"},
            api_key=self.service_role_key,
        )
        return list(response.body or [])

    def delete_rows(
        self,
        table: str,
        *,
        filters: Mapping[str, str],
    ) -> List[Dict[str, Any]]:
        response = self._request_json(
            "DELETE",
            f"/rest/v1/{table}",
            query=dict(filters),
            headers={"Prefer": "return=representation"},
            api_key=self.service_role_key,
        )
        return list(response.body or [])

    def count_rows(
        self,
        table: str,
        *,
        filters: Mapping[str, str] | None = None,
    ) -> int:
        response = self.select_rows(
            table,
            filters=filters,
            select="id",
            limit=1,
            prefer_count=True,
            use_service_role=True,
        )
        content_range = str(response.headers.get("content-range", "")).strip()
        if "/" in content_range:
            try:
                return int(content_range.split("/", 1)[1])
            except ValueError:
                pass
        body = response.body if isinstance(response.body, list) else []
        return len(body)

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, str] | None = None,
        body: Any = None,
        headers: Mapping[str, str] | None = None,
        api_key: str,
    ) -> SupabaseResponse:
        if not self.configured:
            raise ConnectionError("Supabase is not configured.")

        query_string = f"?{parse.urlencode(query or {}, doseq=True)}" if query else ""
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request_headers = {
            "apikey": api_key,
            "Authorization": f"Bearer {api_key}",
        }
        if body is not None:
            request_headers["Content-Type"] = "application/json"
        if headers:
            request_headers.update(dict(headers))
        req = request.Request(
            f"{self.url}{path}{query_string}",
            data=data,
            method=method.upper(),
            headers=request_headers,
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                raw_body = response.read().decode("utf-8")
                parsed_body = self._decode_body(raw_body)
                return SupabaseResponse(
                    body=parsed_body,
                    headers={key.lower(): value for key, value in response.headers.items()},
                    status_code=response.status,
                )
        except error.HTTPError as exc:
            raw_body = exc.read().decode("utf-8", errors="ignore")
            if exc.code == 406:
                return SupabaseResponse(body={}, headers={}, status_code=406)
            parsed_body = self._decode_body(raw_body)
            message = parsed_body.get("message") if isinstance(parsed_body, dict) else ""
            hint = parsed_body.get("hint") if isinstance(parsed_body, dict) else ""
            detail = " ".join(part for part in (str(message).strip(), str(hint).strip()) if part).strip()
            raise ConnectionError(detail or raw_body or f"Supabase returned HTTP {exc.code}.") from exc
        except error.URLError as exc:
            raise ConnectionError("Could not reach Supabase.") from exc

    @staticmethod
    def _decode_body(raw_body: str) -> Any:
        if not raw_body.strip():
            return {}
        try:
            return json.loads(raw_body)
        except json.JSONDecodeError:
            return raw_body
