from __future__ import annotations

from base64 import urlsafe_b64decode, urlsafe_b64encode
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import secrets
from typing import Any, Dict, Tuple

from fastapi import Header, HTTPException, Request, Response, status


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _hash_signature(secret: str, payload_segment: str) -> str:
    return hmac.new(secret.encode("utf-8"), payload_segment.encode("utf-8"), hashlib.sha256).hexdigest()


def _encode_payload(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return urlsafe_b64encode(raw).decode("utf-8").rstrip("=")


def _decode_payload(encoded: str) -> Dict[str, Any] | None:
    try:
        padding = "=" * ((4 - len(encoded) % 4) % 4)
        raw = urlsafe_b64decode((encoded + padding).encode("utf-8")).decode("utf-8")
        payload = json.loads(raw)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def build_app_session_token(*, user_id: str, email: str, session_secret: str, max_age_days: int) -> str:
    issued_at = _utc_now()
    expires_at = issued_at + timedelta(days=max(1, int(max_age_days or 1)))
    payload = {
        "user_id": str(user_id).strip(),
        "email": str(email).strip(),
        "iat": int(issued_at.timestamp()),
        "exp": int(expires_at.timestamp()),
    }
    encoded_payload = _encode_payload(payload)
    signature = _hash_signature(session_secret, encoded_payload)
    return f"{encoded_payload}.{signature}"


def parse_app_session_token(token: str, session_secret: str) -> Dict[str, Any] | None:
    encoded_payload, separator, signature = token.strip().partition(".")
    if not encoded_payload or not separator or not signature:
        return None
    expected_signature = _hash_signature(session_secret, encoded_payload)
    if not secrets.compare_digest(signature, expected_signature):
        return None
    payload = _decode_payload(encoded_payload)
    if not payload:
        return None
    try:
        expires_at = int(payload.get("exp", 0) or 0)
    except (TypeError, ValueError):
        return None
    if expires_at <= int(_utc_now().timestamp()):
        return None
    user_id = str(payload.get("user_id", "")).strip()
    if not user_id:
        return None
    payload["user_id"] = user_id
    payload["email"] = str(payload.get("email", "")).strip()
    return payload


def apply_app_session(response: Response, settings, *, user_id: str, email: str) -> None:
    token = build_app_session_token(
        user_id=user_id,
        email=email,
        session_secret=settings.session_secret,
        max_age_days=settings.session_cookie_max_age_days,
    )
    response.set_cookie(
        key=settings.session_cookie_name,
        value=token,
        max_age=settings.session_cookie_max_age_days * 24 * 60 * 60,
        httponly=True,
        samesite="lax",
        secure=bool(settings.session_cookie_secure),
    )


def clear_app_session(response: Response, settings) -> None:
    response.delete_cookie(
        key=settings.session_cookie_name,
        httponly=True,
        samesite="lax",
        secure=bool(settings.session_cookie_secure),
    )


def load_request_session(request: Request) -> Tuple[Dict[str, Any] | None, bool]:
    settings = request.app.state.settings
    raw_cookie = request.cookies.get(settings.session_cookie_name, "").strip()
    if not raw_cookie:
        return None, False
    session = parse_app_session_token(raw_cookie, settings.session_secret)
    return session, session is None


def _bearer_token(authorization: str | None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        return ""
    return authorization.split(" ", 1)[1].strip()


def _legacy_env_key_match(request: Request, token: str) -> dict[str, str] | None:
    expected_key = str(request.app.state.settings.api_key or "").strip()
    if expected_key and secrets.compare_digest(token, expected_key):
        return {
            "id": "legacy-env-key",
            "owner_id": "",
            "label": "Legacy env key",
            "key_prefix": f"{token[:8]}…{token[-4:]}" if len(token) > 12 else token,
        }
    return None


def require_api_key(
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, str]:
    token = _bearer_token(authorization)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )

    legacy_key = _legacy_env_key_match(request, token)
    if legacy_key is not None:
        return legacy_key

    store = request.app.state.store
    limiter = request.app.state.limiter
    settings = request.app.state.settings
    api_key = store.get_api_key_by_secret(token)
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )

    api_key_id = str(api_key["id"])
    minute_limit = int(api_key.get("rate_limits", {}).get("minute", settings.generated_key_rate_limit_minute))
    hour_limit = int(api_key.get("rate_limits", {}).get("hour", settings.generated_key_rate_limit_hour))
    day_limit = int(api_key.get("rate_limits", {}).get("day", settings.generated_key_rate_limit_day))

    if minute_limit > 0 and not limiter.allow("api-key-minute", api_key_id, minute_limit, 60):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="This API key is temporarily rate-limited. Please wait a moment and try again.",
        )
    if hour_limit > 0 and store.count_recent_api_key_requests(api_key_id, 60 * 60) >= hour_limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="This API key has reached its hourly limit.",
        )
    if day_limit > 0 and store.count_recent_api_key_requests(api_key_id, 24 * 60 * 60) >= day_limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="This API key has reached its daily limit.",
        )

    store.touch_api_key(api_key_id)
    return {
        "id": api_key_id,
        "owner_id": str(api_key.get("owner_id", "")),
        "label": str(api_key.get("label", "")),
        "key_prefix": str(api_key.get("key_prefix", "")),
    }


def get_request_session(request: Request) -> Dict[str, Any]:
    session = getattr(request.state, "auth_session", None)
    return dict(session or {})


def get_request_owner_id(request: Request) -> str:
    session = get_request_session(request)
    return str(session.get("user_id", "")).strip()


def require_app_session(request: Request) -> str:
    owner_id = get_request_owner_id(request)
    if owner_id:
        return owner_id
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required.",
    )
