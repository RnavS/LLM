from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import List


def _parse_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'").strip('"')
    return values


def _get_env(key: str, default: str, env_file_values: dict[str, str]) -> str:
    return os.environ.get(key, env_file_values.get(key, default))


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_optional_bool(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _parse_int(value: str, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_float(value: str, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_origins(value: str) -> List[str]:
    origins = [origin.strip() for origin in value.split(",") if origin.strip()]
    return origins or [
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ]


def _default_checkpoint(root_dir: Path) -> str:
    candidates = [
        root_dir / "checkpoints" / "advanced_local",
        root_dir / "checkpoints" / "smoke_local",
        root_dir / "checkpoints" / "local_chatbot",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return "checkpoints/advanced_local"


def _is_serverless_runtime() -> bool:
    return bool(
        os.environ.get("VERCEL")
        or os.environ.get("AWS_LAMBDA_FUNCTION_NAME")
        or os.environ.get("AWS_EXECUTION_ENV")
    )


def _resolve_writable_path(
    root_dir: Path,
    raw_value: str,
    default_value: str,
    runtime_root: Path,
) -> Path:
    candidate = Path(raw_value or default_value)
    if candidate.is_absolute():
        return candidate.resolve()
    if _is_serverless_runtime():
        return (runtime_root / candidate).resolve()
    return (root_dir / candidate).resolve()


def _is_loopback_host(value: str) -> bool:
    lowered = value.strip().lower()
    return lowered in {"127.0.0.1", "localhost", "::1"}


def _should_secure_cookies(origins: List[str], host: str, explicit_value: str) -> bool:
    parsed = _parse_optional_bool(explicit_value)
    if parsed is not None:
        return parsed
    if _is_loopback_host(host):
        return False
    if not origins:
        return False
    return not all("localhost" in origin or "127.0.0.1" in origin for origin in origins)


@dataclass
class ServerSettings:
    root_dir: Path
    product_name: str
    host: str
    port: int
    api_key: str
    checkpoint: str
    tokenizer_model: str
    device: str
    system_preset: str
    system_prompt: str
    knowledge_index_path: str
    retrieval_top_k: int
    disable_retrieval: bool
    response_mode: str
    model_backend: str
    provider_base_url: str
    provider_api_key: str
    provider_model: str
    provider_timeout_seconds: float
    ollama_base_url: str
    ollama_model: str
    ollama_embedding_model: str
    ollama_timeout_seconds: float
    ollama_keep_alive: str
    semantic_retrieval_enabled: bool
    semantic_rerank_limit: int
    web_search_enabled: bool
    web_search_max_results: int
    web_timeout_seconds: float
    generation_timeout_seconds: float
    session_cookie_name: str
    session_cookie_max_age_days: int
    session_secret: str
    session_cookie_secure: bool
    message_rate_limit: int
    message_rate_window_seconds: int
    api_key_self_serve_enabled: bool
    generated_key_rate_limit_minute: int
    generated_key_rate_limit_hour: int
    generated_key_rate_limit_day: int
    max_completion_tokens: int
    web_cache_dir: Path
    database_path: Path
    frontend_dist: Path
    cors_allow_origins: List[str]
    supabase_url: str
    supabase_anon_key: str
    supabase_service_role_key: str

    @property
    def frontend_dist_exists(self) -> bool:
        return self.frontend_dist.exists()

    @property
    def hosted_provider_configured(self) -> bool:
        return bool(self.provider_base_url.strip() and self.provider_api_key.strip() and self.provider_model.strip())

    @property
    def supabase_configured(self) -> bool:
        return bool(
            self.supabase_url.strip()
            and self.supabase_anon_key.strip()
            and self.supabase_service_role_key.strip()
        )


def load_server_settings(root_dir: str | Path | None = None) -> ServerSettings:
    resolved_root = Path(root_dir or Path(__file__).resolve().parent.parent).resolve()
    env_file_values = _parse_dotenv(resolved_root / ".env")
    runtime_root = Path(
        _get_env(
            "LLM_RUNTIME_DIR",
            "/tmp/medbrief-runtime" if _is_serverless_runtime() else str(resolved_root),
            env_file_values,
        )
    ).resolve()
    host = _get_env("LLM_HOST", "127.0.0.1", env_file_values)
    cors_allow_origins = _parse_origins(
        _get_env(
            "LLM_ALLOW_ORIGINS",
            "http://127.0.0.1:5173,http://localhost:5173,http://127.0.0.1:8000,http://localhost:8000",
            env_file_values,
        )
    )
    return ServerSettings(
        root_dir=resolved_root,
        product_name=_get_env("LLM_PRODUCT_NAME", "MedBrief AI", env_file_values),
        host=host,
        port=_parse_int(_get_env("LLM_PORT", "8000", env_file_values), 8000),
        api_key=_get_env("LLM_API_KEY", "", env_file_values),
        checkpoint=_get_env("LLM_CHECKPOINT", _default_checkpoint(resolved_root), env_file_values),
        tokenizer_model=_get_env("LLM_TOKENIZER_MODEL", "", env_file_values),
        device=_get_env("LLM_DEVICE", "auto", env_file_values),
        system_preset=_get_env("LLM_SYSTEM_PRESET", "medbrief-medical", env_file_values),
        system_prompt=_get_env("LLM_SYSTEM_PROMPT", "", env_file_values),
        knowledge_index_path=_get_env(
            "LLM_KNOWLEDGE_INDEX",
            "data/index/knowledge_index.pkl",
            env_file_values,
        ),
        retrieval_top_k=_parse_int(_get_env("LLM_RETRIEVAL_TOP_K", "4", env_file_values), 4),
        disable_retrieval=_parse_bool(_get_env("LLM_DISABLE_RETRIEVAL", "false", env_file_values)),
        response_mode=_get_env("LLM_RESPONSE_MODE", "assistant", env_file_values),
        model_backend=_get_env("LLM_MODEL_BACKEND", "auto", env_file_values),
        provider_base_url=_get_env("LLM_PROVIDER_BASE_URL", "", env_file_values),
        provider_api_key=_get_env("LLM_PROVIDER_API_KEY", "", env_file_values),
        provider_model=_get_env("LLM_PROVIDER_MODEL", "", env_file_values),
        provider_timeout_seconds=_parse_float(_get_env("LLM_PROVIDER_TIMEOUT_SECONDS", "60.0", env_file_values), 60.0),
        ollama_base_url=_get_env("LLM_OLLAMA_BASE_URL", "http://127.0.0.1:11434", env_file_values),
        ollama_model=_get_env("LLM_OLLAMA_MODEL", "llama3.1:8b", env_file_values),
        ollama_embedding_model=_get_env("LLM_OLLAMA_EMBEDDING_MODEL", "nomic-embed-text", env_file_values),
        ollama_timeout_seconds=_parse_float(_get_env("LLM_OLLAMA_TIMEOUT_SECONDS", "120.0", env_file_values), 120.0),
        ollama_keep_alive=_get_env("LLM_OLLAMA_KEEP_ALIVE", "10m", env_file_values),
        semantic_retrieval_enabled=_parse_bool(_get_env("LLM_SEMANTIC_RETRIEVAL_ENABLED", "true", env_file_values)),
        semantic_rerank_limit=_parse_int(_get_env("LLM_SEMANTIC_RERANK_LIMIT", "8", env_file_values), 8),
        web_search_enabled=_parse_bool(_get_env("LLM_WEB_SEARCH_ENABLED", "true", env_file_values)),
        web_search_max_results=_parse_int(_get_env("LLM_WEB_SEARCH_MAX_RESULTS", "3", env_file_values), 3),
        web_timeout_seconds=_parse_float(_get_env("LLM_WEB_TIMEOUT_SECONDS", "2.5", env_file_values), 2.5),
        generation_timeout_seconds=_parse_float(_get_env("LLM_GENERATION_TIMEOUT_SECONDS", "18.0", env_file_values), 18.0),
        session_cookie_name=_get_env("LLM_SESSION_COOKIE_NAME", "medbrief_session", env_file_values),
        session_cookie_max_age_days=_parse_int(_get_env("LLM_SESSION_COOKIE_MAX_AGE_DAYS", "30", env_file_values), 30),
        session_secret=_get_env("LLM_SESSION_SECRET", "local-dev-session-secret", env_file_values),
        session_cookie_secure=_should_secure_cookies(
            cors_allow_origins,
            host,
            _get_env("LLM_SESSION_COOKIE_SECURE", "auto", env_file_values),
        ),
        message_rate_limit=_parse_int(_get_env("LLM_MESSAGE_RATE_LIMIT", "20", env_file_values), 20),
        message_rate_window_seconds=_parse_int(_get_env("LLM_MESSAGE_RATE_WINDOW_SECONDS", "60", env_file_values), 60),
        api_key_self_serve_enabled=_parse_bool(_get_env("LLM_API_KEY_SELF_SERVE_ENABLED", "true", env_file_values)),
        generated_key_rate_limit_minute=_parse_int(_get_env("LLM_GENERATED_KEY_RATE_LIMIT_MINUTE", "6", env_file_values), 6),
        generated_key_rate_limit_hour=_parse_int(_get_env("LLM_GENERATED_KEY_RATE_LIMIT_HOUR", "60", env_file_values), 60),
        generated_key_rate_limit_day=_parse_int(_get_env("LLM_GENERATED_KEY_RATE_LIMIT_DAY", "250", env_file_values), 250),
        max_completion_tokens=_parse_int(_get_env("LLM_MAX_COMPLETION_TOKENS", "400", env_file_values), 400),
        web_cache_dir=_resolve_writable_path(
            resolved_root,
            _get_env("LLM_WEB_CACHE_DIR", "data/web_cache", env_file_values),
            "data/web_cache",
            runtime_root,
        ),
        database_path=_resolve_writable_path(
            resolved_root,
            _get_env("LLM_DATABASE_PATH", "data/app/medbrief.sqlite3", env_file_values),
            "data/app/medbrief.sqlite3",
            runtime_root,
        ),
        frontend_dist=(resolved_root / _get_env("LLM_FRONTEND_DIST", "frontend_static", env_file_values)).resolve(),
        cors_allow_origins=cors_allow_origins,
        supabase_url=_get_env("SUPABASE_URL", "", env_file_values),
        supabase_anon_key=_get_env("SUPABASE_ANON_KEY", "", env_file_values),
        supabase_service_role_key=_get_env("SUPABASE_SERVICE_ROLE_KEY", "", env_file_values),
    )
