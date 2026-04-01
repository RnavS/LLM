from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class GenerationSettings(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    model_backend: str | None = None
    model: str | None = None
    answer_style: str | None = None
    reader_level: str | None = None
    tone_preference: str | None = None
    primary_use: str | None = None
    display_name: str | None = None
    profile_note: str | None = None
    site_context: str | None = None
    system_preset: str | None = None
    system_prompt: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    max_new_tokens: int | None = None
    max_tokens: int | None = Field(default=None, alias="max_tokens")
    retrieval_top_k: int | None = None
    disable_retrieval: bool = False
    response_mode: str | None = None
    web_search_enabled: bool | None = None
    web_search_max_results: int | None = None

    def to_runtime_kwargs(self) -> Dict[str, Any]:
        max_new_tokens = self.max_new_tokens if self.max_new_tokens is not None else self.max_tokens
        payload: Dict[str, Any] = {
            "model_backend": self.model_backend,
            "model": self.model,
            "answer_style": self.answer_style,
            "reader_level": self.reader_level,
            "tone_preference": self.tone_preference,
            "primary_use": self.primary_use,
            "display_name": self.display_name,
            "profile_note": self.profile_note,
            "site_context": self.site_context,
            "system_preset": self.system_preset or "",
            "system_prompt": self.system_prompt or "",
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "max_new_tokens": max_new_tokens,
            "retrieval_top_k": self.retrieval_top_k,
            "disable_retrieval": bool(self.disable_retrieval),
            "response_mode": self.response_mode,
            "web_search_enabled": self.web_search_enabled,
            "web_search_max_results": self.web_search_max_results,
        }
        return {key: value for key, value in payload.items() if value is not None and value != ""}


class ConversationCreateRequest(BaseModel):
    title: str | None = None
    system_preset: str | None = None
    system_prompt: str | None = None
    settings: GenerationSettings | None = None


class ConversationMessageRequest(BaseModel):
    content: str
    stream: bool = True
    settings: GenerationSettings | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    model: str | None = None
    messages: List[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    max_tokens: int | None = None
    metadata: Dict[str, Any] | None = None
    model_backend: str | None = None
    answer_style: str | None = None
    reader_level: str | None = None
    tone_preference: str | None = None
    primary_use: str | None = None
    display_name: str | None = None
    profile_note: str | None = None
    site_context: str | None = None
    system_preset: str | None = None
    system_prompt: str | None = None
    retrieval_top_k: int | None = None
    disable_retrieval: bool = False
    response_mode: str | None = None
    web_search_enabled: bool | None = None
    web_search_max_results: int | None = None

    def to_generation_settings(self) -> GenerationSettings:
        return GenerationSettings(
            model_backend=self.model_backend,
            model=self.model,
            answer_style=self.answer_style,
            reader_level=self.reader_level,
            tone_preference=self.tone_preference,
            primary_use=self.primary_use,
            display_name=self.display_name,
            profile_note=self.profile_note,
            site_context=self.site_context,
            system_preset=self.system_preset,
            system_prompt=self.system_prompt,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            max_tokens=self.max_tokens,
            retrieval_top_k=self.retrieval_top_k,
            disable_retrieval=self.disable_retrieval,
            response_mode=self.response_mode,
            web_search_enabled=self.web_search_enabled,
            web_search_max_results=self.web_search_max_results,
        )


class UserProfileRequest(BaseModel):
    display_name: str | None = None
    answer_style: str | None = None
    reader_level: str | None = None
    tone_preference: str | None = None
    primary_use: str | None = None
    profile_note: str | None = None
    site_context: str | None = None


class ApiKeyCreateRequest(BaseModel):
    label: str | None = None


class AuthSignupRequest(BaseModel):
    email: str
    password: str


class AuthLoginRequest(BaseModel):
    email: str
    password: str
