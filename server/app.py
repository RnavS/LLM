from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import re
import time
from typing import Any, AsyncIterator, Dict
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse

from presets import classify_query_mode
from server.auth import apply_app_session, clear_app_session, get_request_owner_id, get_request_session, load_request_session, require_api_key, require_app_session
from server.rate_limit import RateLimiter
from server.schemas import ApiKeyCreateRequest, AuthLoginRequest, AuthSignupRequest, ChatCompletionRequest, ConversationCreateRequest, ConversationMessageRequest, UserProfileRequest
from server.service import LocalAssistantService
from server.supabase_client import SupabaseClient
from server.settings import ServerSettings, load_server_settings
from server.storage import build_conversation_store


def _json_sse(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _load_state(app: FastAPI) -> tuple[ServerSettings, LocalAssistantService, Any, RateLimiter]:
    settings = app.state.settings
    service = app.state.service
    store = app.state.store
    limiter = app.state.limiter
    return settings, service, store, limiter


def _conversation_settings_payload(payload_settings: Any) -> Dict[str, Any]:
    if payload_settings is None:
        return {}
    if hasattr(payload_settings, "to_runtime_kwargs"):
        return payload_settings.to_runtime_kwargs()
    if isinstance(payload_settings, dict):
        return dict(payload_settings)
    return {}


def _check_message_rate_limit(request: Request) -> None:
    settings, _, _, limiter = _load_state(request.app)
    owner_id = get_request_owner_id(request)
    if limiter.allow("messages", owner_id, settings.message_rate_limit, settings.message_rate_window_seconds):
        return
    raise HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail="Too many messages in a short period. Please wait a moment and try again.",
    )


def _client_ip(request: Request) -> str:
    forwarded_for = str(request.headers.get("x-forwarded-for", "")).strip()
    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip() or "unknown"
    if request.client and request.client.host:
        return str(request.client.host)
    return "unknown"


def _check_auth_rate_limit(request: Request, action: str, limit: int, window_seconds: int = 60) -> None:
    _, _, _, limiter = _load_state(request.app)
    key = f"{action}:{_client_ip(request)}"
    if limiter.allow("auth", key, limit, window_seconds):
        return
    raise HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail="Too many authentication attempts. Please wait a moment and try again.",
    )


def _validate_auth_payload(email: str, password: str) -> tuple[str, str]:
    cleaned_email = str(email or "").strip().lower()
    cleaned_password = str(password or "")
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", cleaned_email):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Please enter a valid email address.")
    if len(cleaned_password) < 8:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Password must be at least 8 characters long.")
    return cleaned_email, cleaned_password


def _require_supabase_auth_client(app: FastAPI) -> SupabaseClient:
    client = getattr(app.state, "supabase_client", None)
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase auth is not configured.",
        )
    return client


def _auth_session_payload(request: Request) -> Dict[str, Any]:
    session = get_request_session(request)
    if not session:
        return {
            "authenticated": False,
            "user": None,
            "mode": "signed-out",
        }
    return {
        "authenticated": True,
        "mode": str(session.get("mode") or ("supabase" if request.app.state.settings.supabase_configured else "local-dev")),
        "user": {
            "id": session.get("user_id", ""),
            "email": session.get("email", ""),
        },
    }


def _conversation_with_personalization(store: ConversationStore, owner_id: str, conversation: Dict[str, Any]) -> Dict[str, Any]:
    enriched = dict(conversation)
    enriched["memory_items"] = store.list_memory_items(owner_id)
    enriched["summary_state"] = store.get_conversation_summary(owner_id, conversation["id"]).get("summary", {})
    return enriched


def _merge_owner_personalization(
    store: ConversationStore,
    owner_id: str,
    generation_settings: Dict[str, Any],
) -> Dict[str, Any]:
    merged = dict(generation_settings)
    profile = dict(store.get_profile(owner_id).get("profile", {}))
    for key, value in profile.items():
        if value is None or value == "":
            continue
        merged.setdefault(key, value)
    merged.setdefault("memory_items", store.list_memory_items(owner_id))
    merged.setdefault("conversation_summary", {})
    return merged


def _persist_stateless_personalization(
    service: LocalAssistantService,
    store: ConversationStore,
    owner_id: str,
    messages: list[Dict[str, Any]],
    generation_settings: Dict[str, Any],
    result: Dict[str, Any],
) -> None:
    latest_user_text = next(
        (
            str(message.get("content", "")).strip()
            for message in reversed(messages)
            if str(message.get("role", "")).strip() == "user" and str(message.get("content", "")).strip()
        ),
        "",
    )
    if not latest_user_text:
        return
    pseudo_conversation = {
        "messages": list(messages) + [{"role": "assistant", "content": str(result.get("reply", "")).strip()}],
        "settings": generation_settings,
        "memory_items": store.list_memory_items(owner_id),
        "summary_state": {},
    }
    personalization = service.build_personalization_state(pseudo_conversation, latest_user_text, generation_settings)
    store.save_memory_items(owner_id, personalization.get("memory_items", []))


def _metadata_flag(metadata: Dict[str, Any], *keys: str) -> bool:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
    return False


def _normalize_completion_messages(messages: list[Dict[str, Any]], assistant_reply: str) -> list[Dict[str, Any]]:
    normalized: list[Dict[str, Any]] = []
    for item in messages:
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if role not in {"system", "user", "assistant"} or not content:
            continue
        normalized.append(
            {
                "role": role,
                "content": content,
                "metadata": item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
            }
        )
    reply = str(assistant_reply or "").strip()
    if reply:
        normalized.append({"role": "assistant", "content": reply, "metadata": {}})
    return normalized


def _chat_completion_persistence_options(payload: ChatCompletionRequest) -> Dict[str, Any]:
    metadata = payload.metadata if isinstance(payload.metadata, dict) else {}
    conversation_id = str(metadata.get("conversation_id", "")).strip()
    title = str(metadata.get("conversation_title") or metadata.get("title") or "").strip()
    persist = _metadata_flag(metadata, "persist", "persist_conversation", "store", "store_conversation") or bool(conversation_id)
    return {
        "persist": persist,
        "conversation_id": conversation_id,
        "title": title,
    }


def _persist_completion_conversation(
    service: LocalAssistantService,
    store: Any,
    owner_id: str,
    messages: list[Dict[str, Any]],
    generation_settings: Dict[str, Any],
    result: Dict[str, Any],
    persistence: Dict[str, Any],
) -> Dict[str, Any] | None:
    if not persistence.get("persist"):
        return None

    conversation_id = str(persistence.get("conversation_id", "")).strip()
    conversation_title = str(persistence.get("title", "")).strip()
    last_user_text = next(
        (
            str(message.get("content", "")).strip()
            for message in reversed(messages)
            if str(message.get("role", "")).strip().lower() == "user" and str(message.get("content", "")).strip()
        ),
        "",
    )

    if conversation_id:
        try:
            conversation = store.get_conversation(owner_id, conversation_id)
        except KeyError:
            conversation = store.create_conversation(
                owner_id,
                title=conversation_title or (last_user_text[:60] or "API conversation"),
                system_preset=str(generation_settings.get("system_preset") or "medbrief-medical"),
                system_prompt=str(generation_settings.get("system_prompt") or ""),
                settings=generation_settings,
            )
            conversation_id = str(conversation["id"])
    else:
        conversation = store.create_conversation(
            owner_id,
            title=conversation_title or (last_user_text[:60] or "API conversation"),
            system_preset=str(generation_settings.get("system_preset") or "medbrief-medical"),
            system_prompt=str(generation_settings.get("system_prompt") or ""),
            settings=generation_settings,
        )
        conversation_id = str(conversation["id"])

    if conversation_title:
        store.update_conversation(owner_id, conversation_id, title=conversation_title, settings=generation_settings)
    else:
        existing_title = str(conversation.get("title", "")).strip()
        if existing_title in {"", "New chat", "New conversation", "API conversation"} and last_user_text:
            store.update_conversation(owner_id, conversation_id, title=last_user_text[:60], settings=generation_settings)
        else:
            store.update_conversation(owner_id, conversation_id, settings=generation_settings)

    stored_conversation = store.replace_conversation_messages(
        owner_id,
        conversation_id,
        _normalize_completion_messages(messages, str(result.get("reply", ""))),
    )
    personalization = service.build_personalization_state(stored_conversation, last_user_text, generation_settings)
    store.save_memory_items(owner_id, personalization.get("memory_items", []))
    store.upsert_conversation_summary(owner_id, conversation_id, personalization.get("conversation_summary", {}))
    return store.get_conversation(owner_id, conversation_id)


def _phase_sequence(content: str, settings_payload: Dict[str, Any]) -> list[Dict[str, str]]:
    mode = classify_query_mode(
        content,
        primary_use=str(settings_payload.get("primary_use", "balanced")),
        site_context=str(settings_payload.get("site_context", "")),
    )
    web_enabled = bool(settings_payload.get("web_search_enabled"))
    if mode == "psychology":
        return [
            {"phase": "listening", "label": "Listening carefully"},
            {"phase": "understanding", "label": "Understanding the pattern"},
            {"phase": "writing", "label": "Writing your response"},
        ]
    if mode == "portfolio":
        return [
            {"phase": "understanding", "label": "Understanding the project question"},
            {"phase": "retrieving", "label": "Retrieving local project context"},
            {"phase": "writing", "label": "Writing your answer"},
        ]
    if mode == "crisis":
        return [
            {"phase": "safety", "label": "Prioritizing immediate safety"},
            {"phase": "writing", "label": "Writing urgent guidance"},
        ]
    if mode == "medical":
        return [
            {"phase": "understanding", "label": "Understanding the medical question"},
            {"phase": "retrieving", "label": "Searching trusted sources" if web_enabled else "Retrieving medical knowledge"},
            {"phase": "reviewing", "label": "Comparing findings" if web_enabled else "Reviewing evidence"},
            {"phase": "writing", "label": "Writing your answer"},
        ]
    return [
        {"phase": "understanding", "label": "Understanding your request"},
        {"phase": "retrieving", "label": "Checking sources" if web_enabled else "Checking local context"},
        {"phase": "writing", "label": "Writing your answer"},
    ]


def _frontend_missing_page(settings: ServerSettings) -> HTMLResponse:
    content = f"""
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <title>{settings.product_name}</title>
        <style>
          body {{
            background: #0f172a;
            color: #e2e8f0;
            font-family: Arial, sans-serif;
            margin: 0;
            min-height: 100vh;
            display: grid;
            place-items: center;
            padding: 32px;
          }}
          main {{
            max-width: 720px;
            background: rgba(15, 23, 42, 0.92);
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-radius: 20px;
            padding: 28px;
            box-shadow: 0 20px 60px rgba(15, 23, 42, 0.45);
          }}
          code {{
            background: rgba(148, 163, 184, 0.15);
            padding: 2px 6px;
            border-radius: 6px;
          }}
        </style>
      </head>
      <body>
        <main>
          <h1>{settings.product_name} frontend build not found</h1>
          <p>The API server is running, but the built React app is missing.</p>
          <p>Build the frontend or run the launcher script, then refresh:</p>
          <pre><code>npm install
npm run build</code></pre>
          <p>Expected build folder: <code>{settings.frontend_dist}</code></p>
        </main>
      </body>
    </html>
    """
    return HTMLResponse(content=content, status_code=200)


def _frontend_file_response(path, cache_control: str) -> FileResponse:
    response = FileResponse(path)
    response.headers["Cache-Control"] = cache_control
    response.headers["Pragma"] = "no-cache" if "no-store" in cache_control else "public"
    return response


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = load_server_settings()
    app.state.settings = settings
    app.state.service = LocalAssistantService(settings)
    app.state.supabase_client = (
        SupabaseClient(
            url=settings.supabase_url,
            anon_key=settings.supabase_anon_key,
            service_role_key=settings.supabase_service_role_key,
            timeout_seconds=max(10.0, settings.generation_timeout_seconds),
        )
        if settings.supabase_configured
        else None
    )
    app.state.store = build_conversation_store(settings)
    app.state.limiter = RateLimiter()
    yield


app = FastAPI(title="MedBrief AI", version="1.0.0", lifespan=lifespan)

_bootstrap_settings = load_server_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_bootstrap_settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def medbrief_session_middleware(request: Request, call_next):
    session, should_clear_cookie = load_request_session(request)
    if session is None and not request.app.state.settings.supabase_configured:
        session = {
            "user_id": "local-dev-user",
            "email": "local@medbrief.dev",
            "mode": "local-dev",
        }
        should_clear_cookie = False
    request.state.auth_session = session or {}
    request.state.owner_id = str((session or {}).get("user_id", "")).strip()
    response = await call_next(request)
    set_cookie_header = str(response.headers.get("set-cookie", ""))
    if should_clear_cookie and f"{request.app.state.settings.session_cookie_name}=" not in set_cookie_header:
        clear_app_session(response, request.app.state.settings)
    return response


@app.get("/", include_in_schema=False)
async def serve_frontend_root(request: Request) -> Response:
    settings, _, _, _ = _load_state(request.app)
    index_file = settings.frontend_dist / "index.html"
    if index_file.exists():
        return _frontend_file_response(index_file, "no-store, no-cache, must-revalidate")
    return _frontend_missing_page(settings)


@app.get("/api/auth/session")
async def auth_session(request: Request) -> Dict[str, Any]:
    return _auth_session_payload(request)


@app.post("/api/auth/signup")
async def auth_signup(request: Request, payload: AuthSignupRequest) -> Response:
    _check_auth_rate_limit(request, "signup", limit=6)
    email, password = _validate_auth_payload(payload.email, payload.password)
    supabase_client = _require_supabase_auth_client(request.app)
    try:
        supabase_client.sign_up(email=email, password=password)
        auth_payload = supabase_client.sign_in_with_password(email=email, password=password)
    except ConnectionError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc) or "Unable to create your account.") from exc
    user = dict(auth_payload.get("user") or {})
    user_id = str(user.get("id", "")).strip()
    if not user_id:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Authentication provider did not return a user id.")
    _, _, store, _ = _load_state(request.app)
    store.ensure_owner_record(user_id, email=email)
    response = JSONResponse(
        {
            "authenticated": True,
            "mode": "supabase",
            "user": {
                "id": user_id,
                "email": email,
            },
        }
    )
    apply_app_session(response, request.app.state.settings, user_id=user_id, email=email)
    return response


@app.post("/api/auth/login")
async def auth_login(request: Request, payload: AuthLoginRequest) -> Response:
    _check_auth_rate_limit(request, "login", limit=10)
    email, password = _validate_auth_payload(payload.email, payload.password)
    supabase_client = _require_supabase_auth_client(request.app)
    try:
        auth_payload = supabase_client.sign_in_with_password(email=email, password=password)
    except ConnectionError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc) or "Unable to sign in.") from exc
    user = dict(auth_payload.get("user") or {})
    user_id = str(user.get("id", "")).strip()
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Sign-in failed.")
    _, _, store, _ = _load_state(request.app)
    store.ensure_owner_record(user_id, email=email)
    response = JSONResponse(
        {
            "authenticated": True,
            "mode": "supabase",
            "user": {
                "id": user_id,
                "email": email,
            },
        }
    )
    apply_app_session(response, request.app.state.settings, user_id=user_id, email=email)
    return response


@app.post("/api/auth/logout")
async def auth_logout(request: Request) -> Response:
    response = JSONResponse({"authenticated": False, "user": None, "mode": "signed-out"})
    clear_app_session(response, request.app.state.settings)
    return response


@app.get("/api/health", dependencies=[Depends(require_app_session)])
async def api_health(request: Request) -> Dict[str, Any]:
    _, service, _, _ = _load_state(request.app)
    return service.health_payload()


@app.get("/api/config", dependencies=[Depends(require_app_session)])
async def api_config(request: Request) -> Dict[str, Any]:
    _, service, _, _ = _load_state(request.app)
    return service.app_config_payload()


@app.get("/api/profile", dependencies=[Depends(require_app_session)])
async def get_profile(request: Request) -> Dict[str, Any]:
    _, _, store, _ = _load_state(request.app)
    return store.get_profile(get_request_owner_id(request))


@app.put("/api/profile", dependencies=[Depends(require_app_session)])
async def update_profile(request: Request, payload: UserProfileRequest) -> Dict[str, Any]:
    _, _, store, _ = _load_state(request.app)
    profile_update = payload.model_dump(exclude_none=True)
    return store.upsert_profile(get_request_owner_id(request), profile_update)


@app.get("/api/profile/memory", dependencies=[Depends(require_app_session)])
async def list_profile_memory(request: Request) -> Dict[str, Any]:
    _, _, store, _ = _load_state(request.app)
    return {"data": store.list_memory_items(get_request_owner_id(request))}


@app.delete("/api/profile/memory", dependencies=[Depends(require_app_session)])
async def clear_profile_memory(request: Request) -> Dict[str, str]:
    _, _, store, _ = _load_state(request.app)
    store.clear_memory(get_request_owner_id(request))
    return {"status": "cleared"}


@app.delete("/api/profile/memory/{memory_id}", dependencies=[Depends(require_app_session)])
async def delete_profile_memory_item(request: Request, memory_id: str) -> Dict[str, str]:
    _, _, store, _ = _load_state(request.app)
    try:
        store.delete_memory_item(get_request_owner_id(request), memory_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory item not found.") from exc
    return {"status": "deleted"}


@app.get("/api/keys", dependencies=[Depends(require_app_session)])
async def list_generated_keys(request: Request) -> Dict[str, Any]:
    settings, _, store, _ = _load_state(request.app)
    if not settings.api_key_self_serve_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return {"data": store.list_api_keys(get_request_owner_id(request))}


@app.post("/api/keys", dependencies=[Depends(require_app_session)])
async def create_generated_key(request: Request, payload: ApiKeyCreateRequest) -> Dict[str, Any]:
    settings, _, store, _ = _load_state(request.app)
    if not settings.api_key_self_serve_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    created = store.create_api_key(
        get_request_owner_id(request),
        payload.label or "Project key",
        {
            "minute": settings.generated_key_rate_limit_minute,
            "hour": settings.generated_key_rate_limit_hour,
            "day": settings.generated_key_rate_limit_day,
        },
    )
    return {
        "api_key": created["secret"],
        "record": created["record"],
    }


@app.delete("/api/keys/{key_id}", dependencies=[Depends(require_app_session)])
async def revoke_generated_key(request: Request, key_id: str) -> Dict[str, str]:
    settings, _, store, _ = _load_state(request.app)
    if not settings.api_key_self_serve_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    try:
        store.revoke_api_key(get_request_owner_id(request), key_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="API key not found.") from exc
    return {"status": "revoked"}


@app.get("/api/conversations", dependencies=[Depends(require_app_session)])
async def list_conversations(request: Request) -> Dict[str, Any]:
    _, _, store, _ = _load_state(request.app)
    return {"data": store.list_conversations(get_request_owner_id(request))}


@app.post("/api/conversations", dependencies=[Depends(require_app_session)])
async def create_conversation(request: Request, payload: ConversationCreateRequest) -> Dict[str, Any]:
    _, service, store, _ = _load_state(request.app)
    owner_id = get_request_owner_id(request)
    profile = store.get_profile(owner_id)["profile"]
    defaults = service.default_conversation_payload(title=payload.title or "New conversation")
    settings_payload = defaults["settings"]
    for key, value in profile.items():
        settings_payload.setdefault(key, value)
    settings_payload.update(_conversation_settings_payload(payload.settings))
    conversation = store.create_conversation(
        owner_id,
        title=payload.title or defaults["title"] or "New conversation",
        system_preset=payload.system_preset or settings_payload.get("system_preset") or defaults["system_preset"],
        system_prompt=payload.system_prompt or settings_payload.get("system_prompt") or defaults["system_prompt"],
        settings=settings_payload,
    )
    return conversation


@app.get("/api/conversations/{conversation_id}", dependencies=[Depends(require_app_session)])
async def get_conversation(request: Request, conversation_id: str) -> Dict[str, Any]:
    _, _, store, _ = _load_state(request.app)
    try:
        return store.get_conversation(get_request_owner_id(request), conversation_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.") from exc


@app.delete("/api/conversations/{conversation_id}", dependencies=[Depends(require_app_session)])
async def delete_conversation(request: Request, conversation_id: str) -> Dict[str, str]:
    _, _, store, _ = _load_state(request.app)
    try:
        store.delete_conversation(get_request_owner_id(request), conversation_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.") from exc
    return {"status": "deleted"}


@app.post("/api/conversations/{conversation_id}/messages", dependencies=[Depends(require_app_session)])
async def create_conversation_message(
    request: Request,
    conversation_id: str,
    payload: ConversationMessageRequest,
) -> Response:
    settings, service, store, _ = _load_state(request.app)
    owner_id = get_request_owner_id(request)
    _check_message_rate_limit(request)
    started_at = time.perf_counter()

    try:
        conversation = store.get_conversation(owner_id, conversation_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.") from exc

    content = payload.content.strip()
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Message content cannot be empty.")

    incoming_settings = _conversation_settings_payload(payload.settings)
    if incoming_settings:
        conversation = store.update_conversation(
            owner_id,
            conversation_id,
            system_preset=incoming_settings.get("system_preset") or conversation["system_preset"],
            system_prompt=incoming_settings.get("system_prompt") or conversation["system_prompt"],
            settings=service.resolve_conversation_settings(conversation, incoming_settings),
        )

    if conversation["title"] in {"New chat", "New conversation"} and not conversation["messages"]:
        conversation = store.update_conversation(
            owner_id,
            conversation_id,
            title=service.title_for_first_message(content),
        )

    user_message = store.add_message(owner_id, conversation_id, "user", content)
    conversation = _conversation_with_personalization(store, owner_id, store.get_conversation(owner_id, conversation_id))
    effective_settings_for_phases = service.resolve_conversation_settings(conversation, incoming_settings)
    phase_sequence = _phase_sequence(content, effective_settings_for_phases)

    async def run_generation() -> Dict[str, Any]:
        return await asyncio.wait_for(
            asyncio.to_thread(service.generate_for_conversation, conversation, content, incoming_settings),
            timeout=settings.generation_timeout_seconds,
        )

    async def run_timeout_fallback() -> Dict[str, Any]:
        return await asyncio.wait_for(
            asyncio.to_thread(service.generate_timeout_fallback, conversation, content, incoming_settings),
            timeout=max(6.0, min(settings.generation_timeout_seconds, 10.0)),
        )

    async def run_last_resort_fallback() -> Dict[str, Any]:
        return await asyncio.to_thread(service.generate_last_resort_fallback, conversation, content, incoming_settings)

    async def finalize_success(result: Dict[str, Any]) -> Dict[str, Any]:
        assistant_message = store.add_message(
            owner_id,
            conversation_id,
            "assistant",
            str(result["reply"]),
            metadata={
                "sources": result.get("used_context", []),
                "mode": result.get("mode"),
                "model_id": result.get("model_id"),
                "model_backend": result.get("model_backend"),
            },
        )
        personalization = service.build_personalization_state(
            _conversation_with_personalization(store, owner_id, store.get_conversation(owner_id, conversation_id)),
            content,
            incoming_settings,
        )
        store.save_memory_items(owner_id, personalization.get("memory_items", []))
        store.upsert_conversation_summary(owner_id, conversation_id, personalization.get("conversation_summary", {}))
        updated_conversation = store.get_conversation(owner_id, conversation_id)
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        store.log_request(
            owner_id=owner_id,
            route="/api/conversations/{conversation_id}/messages",
            status_code=200,
            latency_ms=latency_ms,
            conversation_id=conversation_id,
            metadata={"stream": payload.stream, "mode": result.get("mode"), "model_id": result.get("model_id", service.model_id)},
        )
        return {
            "conversation": updated_conversation,
            "message": assistant_message,
            "reply": result["reply"],
            "sources": result.get("used_context", []),
            "usage": {
                "prompt_tokens": int(result.get("prompt_tokens", 0)),
                "completion_tokens": int(result.get("completion_tokens", 0)),
                "total_tokens": int(result.get("total_tokens", 0)),
            },
        }

    if not payload.stream:
        try:
            result = await run_generation()
        except asyncio.TimeoutError as exc:
            try:
                fallback_result = await run_timeout_fallback()
            except Exception:
                store.log_request(
                    owner_id=owner_id,
                    route="/api/conversations/{conversation_id}/messages",
                    status_code=504,
                    latency_ms=(time.perf_counter() - started_at) * 1000.0,
                    conversation_id=conversation_id,
                    metadata={"stream": payload.stream},
                )
                return JSONResponse(await finalize_success(await run_last_resort_fallback()))
            return JSONResponse(await finalize_success(fallback_result))
        return JSONResponse(await finalize_success(result))

    async def event_stream() -> AsyncIterator[str]:
        yield _json_sse(
            {
                "type": "message_start",
                "conversation_id": conversation_id,
                "message": user_message,
            }
        )
        for phase in phase_sequence[:2]:
            yield _json_sse({"type": "phase", "phase": phase["phase"], "label": phase["label"]})
            await asyncio.sleep(0)
        try:
            result = await run_generation()
        except asyncio.TimeoutError:
            try:
                yield _json_sse({"type": "phase", "phase": "fallback", "label": "Switching to a faster grounded answer"})
                result = await run_timeout_fallback()
            except Exception:
                store.log_request(
                    owner_id=owner_id,
                    route="/api/conversations/{conversation_id}/messages",
                    status_code=504,
                    latency_ms=(time.perf_counter() - started_at) * 1000.0,
                    conversation_id=conversation_id,
                    metadata={"stream": payload.stream},
                )
                result = await run_last_resort_fallback()
        for phase in phase_sequence[2:]:
            yield _json_sse({"type": "phase", "phase": phase["phase"], "label": phase["label"]})
            await asyncio.sleep(0)
        chunk = ""
        for word in str(result["reply"]).split():
            candidate = word if not chunk else f"{chunk} {word}"
            if len(candidate) <= 28:
                chunk = candidate
                continue
            yield _json_sse({"type": "delta", "delta": chunk + " "})
            chunk = word
        if chunk:
            yield _json_sse({"type": "delta", "delta": chunk})
        response_payload = await finalize_success(result)
        yield _json_sse({"type": "done", **response_payload})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/v1/models")
async def list_models(request: Request, api_key: Dict[str, str] = Depends(require_api_key)) -> Dict[str, Any]:
    _, service, store, _ = _load_state(request.app)
    payload = service.list_models_payload()
    store.log_request(
        owner_id=api_key.get("owner_id") or get_request_owner_id(request),
        route="/v1/models",
        status_code=200,
        latency_ms=0.0,
        api_key_id=api_key.get("id"),
        metadata={"model_count": len(payload.get("data", []))},
    )
    return payload


@app.get("/v1/conversations")
async def list_api_conversations(request: Request, api_key: Dict[str, str] = Depends(require_api_key)) -> Dict[str, Any]:
    _, _, store, _ = _load_state(request.app)
    owner_id = str(api_key.get("owner_id", "")).strip()
    return {"data": store.list_conversations(owner_id)}


@app.post("/v1/conversations")
async def create_api_conversation(
    request: Request,
    payload: ConversationCreateRequest,
    api_key: Dict[str, str] = Depends(require_api_key),
) -> Dict[str, Any]:
    _, service, store, _ = _load_state(request.app)
    owner_id = str(api_key.get("owner_id", "")).strip()
    profile = store.get_profile(owner_id)["profile"]
    defaults = service.default_conversation_payload(title=payload.title or "New conversation")
    settings_payload = defaults["settings"]
    for key, value in profile.items():
        settings_payload.setdefault(key, value)
    settings_payload.update(_conversation_settings_payload(payload.settings))
    return store.create_conversation(
        owner_id,
        title=payload.title or defaults["title"] or "New conversation",
        system_preset=payload.system_preset or settings_payload.get("system_preset") or defaults["system_preset"],
        system_prompt=payload.system_prompt or settings_payload.get("system_prompt") or defaults["system_prompt"],
        settings=settings_payload,
    )


@app.get("/v1/conversations/{conversation_id}")
async def get_api_conversation(
    request: Request,
    conversation_id: str,
    api_key: Dict[str, str] = Depends(require_api_key),
) -> Dict[str, Any]:
    _, _, store, _ = _load_state(request.app)
    try:
        return store.get_conversation(str(api_key.get("owner_id", "")).strip(), conversation_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.") from exc


@app.delete("/v1/conversations/{conversation_id}")
async def delete_api_conversation(
    request: Request,
    conversation_id: str,
    api_key: Dict[str, str] = Depends(require_api_key),
) -> Dict[str, str]:
    _, _, store, _ = _load_state(request.app)
    try:
        store.delete_conversation(str(api_key.get("owner_id", "")).strip(), conversation_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.") from exc
    return {"status": "deleted"}


@app.get("/v1/profile/memory")
async def list_api_memory(request: Request, api_key: Dict[str, str] = Depends(require_api_key)) -> Dict[str, Any]:
    _, _, store, _ = _load_state(request.app)
    return {"data": store.list_memory_items(str(api_key.get("owner_id", "")).strip())}


@app.post("/api/chat/completions", dependencies=[Depends(require_app_session)])
async def app_chat_completions(request: Request, payload: ChatCompletionRequest) -> Response:
    settings, service, store, _ = _load_state(request.app)
    owner_id = get_request_owner_id(request)
    started_at = time.perf_counter()
    completion_id = f"chatcmpl-{uuid4().hex}"
    created = int(datetime.now(timezone.utc).timestamp())
    generation_settings = _merge_owner_personalization(
        store,
        owner_id,
        payload.to_generation_settings().to_runtime_kwargs(),
    )
    messages = [message.model_dump() for message in payload.messages]
    persistence = _chat_completion_persistence_options(payload)

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(service.generate_from_messages, messages, generation_settings),
            timeout=settings.generation_timeout_seconds,
        )
    except asyncio.TimeoutError:
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(service.generate_timeout_fallback_for_messages, messages, generation_settings),
                timeout=max(6.0, min(settings.generation_timeout_seconds, 10.0)),
            )
        except Exception:
            result = await asyncio.to_thread(service.generate_last_resort_fallback_for_messages, messages, generation_settings)

    latency_ms = (time.perf_counter() - started_at) * 1000.0
    stored_conversation = _persist_completion_conversation(
        service,
        store,
        owner_id,
        messages,
        generation_settings,
        result,
        persistence,
    )
    if stored_conversation is None:
        _persist_stateless_personalization(service, store, owner_id, messages, generation_settings, result)
    store.log_request(
        owner_id=owner_id,
        route="/api/chat/completions",
        status_code=200,
        latency_ms=latency_ms,
        metadata={
            "stream": payload.stream,
            "model_id": result.get("model_id", service.model_id),
            "conversation_id": (stored_conversation or {}).get("id", ""),
        },
    )

    if not payload.stream:
        response_payload = service.build_openai_response(result, completion_id=completion_id, created=created)
        if stored_conversation is not None:
            response_payload["medbrief"] = {
                "conversation_id": stored_conversation["id"],
                "stored": True,
            }
        response = JSONResponse(response_payload)
        if stored_conversation is not None:
            response.headers["X-MedBrief-Conversation-Id"] = str(stored_conversation["id"])
        return response

    async def event_stream() -> AsyncIterator[str]:
        for chunk in service.openai_stream_chunks(result, completion_id=completion_id, created=created):
            yield _json_sse(chunk)
        yield "data: [DONE]\n\n"

    response = StreamingResponse(event_stream(), media_type="text/event-stream")
    if stored_conversation is not None:
        response.headers["X-MedBrief-Conversation-Id"] = str(stored_conversation["id"])
    return response


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    payload: ChatCompletionRequest,
    api_key: Dict[str, str] = Depends(require_api_key),
) -> Response:
    settings, service, store, _ = _load_state(request.app)
    owner_id = api_key.get("owner_id") or get_request_owner_id(request)
    started_at = time.perf_counter()
    completion_id = f"chatcmpl-{uuid4().hex}"
    created = int(datetime.now(timezone.utc).timestamp())
    generation_settings = _merge_owner_personalization(
        store,
        owner_id,
        payload.to_generation_settings().to_runtime_kwargs(),
    )
    messages = [message.model_dump() for message in payload.messages]
    persistence = _chat_completion_persistence_options(payload)
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(service.generate_from_messages, messages, generation_settings),
            timeout=settings.generation_timeout_seconds,
        )
    except asyncio.TimeoutError:
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(service.generate_timeout_fallback_for_messages, messages, generation_settings),
                timeout=max(6.0, min(settings.generation_timeout_seconds, 10.0)),
            )
        except Exception:
            result = await asyncio.to_thread(service.generate_last_resort_fallback_for_messages, messages, generation_settings)

    latency_ms = (time.perf_counter() - started_at) * 1000.0
    stored_conversation = _persist_completion_conversation(
        service,
        store,
        owner_id,
        messages,
        generation_settings,
        result,
        persistence,
    )
    if stored_conversation is None:
        _persist_stateless_personalization(service, store, owner_id, messages, generation_settings, result)
    store.log_request(
        owner_id=owner_id,
        route="/v1/chat/completions",
        status_code=200,
        latency_ms=latency_ms,
        api_key_id=api_key.get("id"),
        metadata={
            "stream": payload.stream,
            "model_id": result.get("model_id", service.model_id),
            "conversation_id": (stored_conversation or {}).get("id", ""),
        },
    )

    if not payload.stream:
        response_payload = service.build_openai_response(result, completion_id=completion_id, created=created)
        if stored_conversation is not None:
            response_payload["medbrief"] = {
                "conversation_id": stored_conversation["id"],
                "stored": True,
            }
        response = JSONResponse(response_payload)
        if stored_conversation is not None:
            response.headers["X-MedBrief-Conversation-Id"] = str(stored_conversation["id"])
        return response

    async def event_stream() -> AsyncIterator[str]:
        for chunk in service.openai_stream_chunks(result, completion_id=completion_id, created=created):
            yield _json_sse(chunk)
        yield "data: [DONE]\n\n"

    response = StreamingResponse(event_stream(), media_type="text/event-stream")
    if stored_conversation is not None:
        response.headers["X-MedBrief-Conversation-Id"] = str(stored_conversation["id"])
    return response


@app.get("/{full_path:path}", include_in_schema=False)
async def serve_frontend_asset(request: Request, full_path: str) -> Response:
    if full_path.startswith("api/") or full_path.startswith("v1/"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    settings, _, _, _ = _load_state(request.app)
    if not settings.frontend_dist_exists:
        return _frontend_missing_page(settings)

    if full_path:
        candidate = (settings.frontend_dist / full_path).resolve()
        try:
            candidate.relative_to(settings.frontend_dist)
        except ValueError:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from None
        if candidate.exists() and candidate.is_file():
            cache_control = "public, max-age=31536000, immutable" if "assets" in candidate.parts else "no-store, no-cache, must-revalidate"
            return _frontend_file_response(candidate, cache_control)

    index_file = settings.frontend_dist / "index.html"
    if index_file.exists():
        return _frontend_file_response(index_file, "no-store, no-cache, must-revalidate")
    return _frontend_missing_page(settings)
