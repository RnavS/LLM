from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, Iterator, List, Sequence

import torch

from config import GenerationConfig, ModelConfig
from dataset import ConversationFormatter, build_chat_input_ids, extract_assistant_reply
from model import GPTLanguageModel
from presets import (
    DEFAULT_SYSTEM_PRESET,
    SYSTEM_PROMPTS,
    asks_for_dosing,
    build_query_safety_instruction,
    get_generation_defaults,
    get_system_prompt,
    has_red_flag_symptoms,
)
from retrieval import KnowledgeIndex, STOPWORDS
from tokenizer import ChatTokenizer
from utils import get_device, load_checkpoint


@dataclass
class AssistantRuntime:
    model: GPTLanguageModel
    tokenizer: ChatTokenizer
    model_config: ModelConfig
    generation_config: GenerationConfig
    formatter: ConversationFormatter
    device: torch.device
    checkpoint_path: Path
    knowledge_index: KnowledgeIndex | None = None


@dataclass
class RuntimeRequestConfig:
    system_preset: str
    system_prompt: str
    max_new_tokens: int
    temperature: float
    top_k: int | None
    top_p: float | None
    repetition_penalty: float
    retrieval_top_k: int
    disable_retrieval: bool


def get_supported_presets() -> List[str]:
    return sorted(SYSTEM_PROMPTS)


def _looks_low_quality(reply: str) -> bool:
    cleaned = reply.strip()
    if len(cleaned) < 24:
        return True
    if "\ufffd" in cleaned or "</" in cleaned:
        return True
    words = re.findall(r"[a-zA-Z0-9']+", cleaned.lower())
    if len(words) < 6:
        return True
    unique_ratio = len(set(words)) / max(1, len(words))
    if unique_ratio < 0.35:
        return True
    alpha_chars = sum(character.isalpha() for character in cleaned)
    if alpha_chars / max(1, len(cleaned)) < 0.55:
        return True
    return False


def _content_terms(text: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[a-zA-Z0-9']+", text.lower())
        if term not in STOPWORDS and len(term) > 2
    }


def _is_direct_information_query(text: str) -> bool:
    lowered = text.lower().strip()
    return any(
        lowered.startswith(prefix)
        for prefix in (
            "what is",
            "what are",
            "how does",
            "how do",
            "explain",
            "define",
            "tell me about",
            "why does",
            "why do",
        )
    )


def _should_prefer_retrieval_fallback(
    user_text: str,
    reply: str,
    used_context: Sequence[dict[str, str | float]],
) -> bool:
    if not used_context:
        return False
    if _looks_low_quality(reply):
        return True

    query_terms = _content_terms(user_text)
    reply_terms = _content_terms(reply)
    context_terms = _content_terms(" ".join(str(chunk.get("text", "")) for chunk in used_context))

    shared_query_terms = query_terms.intersection(reply_terms)
    shared_context_terms = context_terms.intersection(reply_terms)

    if asks_for_dosing(user_text) or has_red_flag_symptoms(user_text):
        return len(shared_query_terms) == 0

    if _is_direct_information_query(user_text) and len(shared_query_terms) == 0:
        return True

    return len(shared_query_terms) == 0 and len(shared_context_terms) < 4


def _build_retrieval_fallback(user_text: str, used_context: Sequence[dict[str, str | float]]) -> str:
    sentences = []
    query_terms = {
        term
        for term in re.findall(r"[a-zA-Z0-9']+", user_text.lower())
        if term not in STOPWORDS
    }
    wants_dosing = asks_for_dosing(user_text)
    red_flag = has_red_flag_symptoms(user_text)
    for chunk in used_context:
        for sentence in re.split(r"(?<=[.!?])\s+", str(chunk.get("text", ""))):
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence_terms = set(re.findall(r"[a-zA-Z0-9']+", sentence.lower()))
            score = len(query_terms.intersection(sentence_terms))
            lowered_sentence = sentence.lower()
            if wants_dosing and any(term in lowered_sentence for term in ["dose", "dosing", "medication", "prescription", "drug"]):
                score += 4
            if red_flag and any(term in lowered_sentence for term in ["emergency", "urgent", "immediate", "seek care"]):
                score += 4
            sentences.append((score, sentence))

    sentences.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
    chosen = []
    seen = set()
    for _score, sentence in sentences:
        normalized = sentence.lower()
        if normalized in seen:
            continue
        chosen.append(sentence)
        seen.add(normalized)
        if len(chosen) == 2:
            break

    if not chosen and used_context:
        chosen.append(str(used_context[0].get("text", "")).strip().split(". ")[0].strip())

    prefix = ""
    if red_flag:
        prefix = "This could be urgent. Seek emergency care or immediate clinician evaluation. "
    elif wants_dosing:
        prefix = "I can't provide authoritative medication dosing. "
    return (prefix + " ".join(chosen)).strip()


def _chunk_text_for_streaming(text: str, max_chars: int = 28) -> Iterator[str]:
    words = text.split()
    if not words:
        return

    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            yield current + " "
        current = word
    if current:
        yield current


def _resolve_request_config(
    runtime: AssistantRuntime,
    system_preset: str = "",
    system_prompt: str = "",
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
    retrieval_top_k: int | None = None,
    disable_retrieval: bool = False,
) -> RuntimeRequestConfig:
    resolved_preset = system_preset or runtime.generation_config.system_preset or DEFAULT_SYSTEM_PRESET
    preset_defaults = get_generation_defaults(resolved_preset)
    resolved_system_prompt = get_system_prompt(resolved_preset, override=system_prompt or runtime.generation_config.system_prompt)
    return RuntimeRequestConfig(
        system_preset=resolved_preset,
        system_prompt=resolved_system_prompt,
        max_new_tokens=max_new_tokens if max_new_tokens is not None else runtime.generation_config.max_new_tokens,
        temperature=temperature if temperature is not None else float(preset_defaults["temperature"]),
        top_k=top_k if top_k is not None else int(preset_defaults["top_k"]),
        top_p=top_p if top_p is not None else float(preset_defaults["top_p"]),
        repetition_penalty=(
            repetition_penalty
            if repetition_penalty is not None
            else float(preset_defaults["repetition_penalty"])
        ),
        retrieval_top_k=(
            retrieval_top_k if retrieval_top_k is not None else runtime.generation_config.retrieval_top_k
        ),
        disable_retrieval=disable_retrieval,
    )


def load_runtime(
    checkpoint: str,
    tokenizer_model: str = "",
    device_name: str = "auto",
    system_preset: str = "",
    system_prompt: str = "",
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
    knowledge_index_path: str = "",
    retrieval_top_k: int | None = None,
    disable_retrieval: bool = False,
) -> AssistantRuntime:
    device = get_device(device_name)
    checkpoint_payload, checkpoint_path = load_checkpoint(checkpoint, device)

    model_config = ModelConfig.from_dict(checkpoint_payload["model_config"])
    model = GPTLanguageModel(model_config).to(device)
    model.load_state_dict(checkpoint_payload["model_state"])
    model.eval()

    tokenizer_path = Path(tokenizer_model or checkpoint_payload["tokenizer_model"])
    tokenizer = ChatTokenizer(tokenizer_path)

    training_config = checkpoint_payload.get("training_config", {})
    resolved_preset = system_preset or training_config.get("system_preset") or DEFAULT_SYSTEM_PRESET
    preset_defaults = get_generation_defaults(resolved_preset)
    resolved_system_prompt = get_system_prompt(resolved_preset, override=system_prompt)

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens if max_new_tokens is not None else int(preset_defaults["max_new_tokens"]),
        temperature=temperature if temperature is not None else float(preset_defaults["temperature"]),
        top_k=top_k if top_k is not None else int(preset_defaults["top_k"]),
        top_p=top_p if top_p is not None else float(preset_defaults["top_p"]),
        repetition_penalty=(
            repetition_penalty
            if repetition_penalty is not None
            else float(preset_defaults["repetition_penalty"])
        ),
        system_preset=resolved_preset,
        system_prompt=resolved_system_prompt,
        knowledge_index_path=knowledge_index_path or training_config.get("knowledge_index_path", "data/index/knowledge_index.pkl"),
        retrieval_top_k=retrieval_top_k if retrieval_top_k is not None else int(training_config.get("retrieval_top_k", 4)),
    )

    formatter = ConversationFormatter(default_system_prompt=generation_config.system_prompt)
    knowledge_index = None
    if not disable_retrieval:
        candidate_index_path = Path(generation_config.knowledge_index_path)
        if candidate_index_path.exists():
            knowledge_index = KnowledgeIndex.load(candidate_index_path)

    return AssistantRuntime(
        model=model,
        tokenizer=tokenizer,
        model_config=model_config,
        generation_config=generation_config,
        formatter=formatter,
        device=device,
        checkpoint_path=checkpoint_path,
        knowledge_index=knowledge_index,
    )


def complete_messages(
    runtime: AssistantRuntime,
    messages: Sequence[Dict[str, str]],
    system_preset: str = "",
    system_prompt: str = "",
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
    retrieval_top_k: int | None = None,
    disable_retrieval: bool = False,
) -> Dict[str, Any]:
    request_config = _resolve_request_config(
        runtime,
        system_preset=system_preset,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        retrieval_top_k=retrieval_top_k,
        disable_retrieval=disable_retrieval,
    )

    cleaned_messages = [
        {"role": str(message.get("role", "")).strip().lower(), "content": str(message.get("content", ""))}
        for message in messages
        if str(message.get("role", "")).strip().lower() in {"system", "user", "assistant"}
        and str(message.get("content", "")).strip()
    ]
    if not cleaned_messages:
        raise ValueError("At least one non-empty message is required.")

    user_messages = [message for message in cleaned_messages if message["role"] == "user"]
    if not user_messages:
        raise ValueError("A user message is required to generate a reply.")
    last_user_text = user_messages[-1]["content"]

    extra_system_instruction = build_query_safety_instruction(last_user_text)
    retrieved_context: List[Dict[str, str | float]] = []
    if (
        runtime.knowledge_index is not None
        and not request_config.disable_retrieval
        and request_config.retrieval_top_k > 0
    ):
        retrieved_context = runtime.knowledge_index.retrieve(
            last_user_text,
            top_k=request_config.retrieval_top_k,
        )

    prompt_ids, kept_messages, used_context = build_chat_input_ids(
        tokenizer=runtime.tokenizer,
        formatter=runtime.formatter,
        system_prompt=request_config.system_prompt,
        messages=cleaned_messages,
        block_size=runtime.model_config.block_size,
        max_new_tokens=request_config.max_new_tokens,
        retrieved_context=retrieved_context,
        extra_system_instruction=extra_system_instruction,
    )

    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=runtime.device)
    with torch.no_grad():
        generated = runtime.model.generate(
            prompt_tensor,
            max_new_tokens=request_config.max_new_tokens,
            temperature=request_config.temperature,
            top_k=request_config.top_k,
            top_p=request_config.top_p,
            repetition_penalty=request_config.repetition_penalty,
            stop_token_ids=runtime.tokenizer.default_stop_ids,
        )

    reply = extract_assistant_reply(
        runtime.tokenizer,
        generated[0].tolist(),
        prompt_length=len(prompt_ids),
        stop_ids=runtime.tokenizer.default_stop_ids,
    )
    fallback_context = list(used_context) if used_context else list(retrieved_context)
    if asks_for_dosing(last_user_text):
        reply = (
            "I can't provide authoritative medication dosing. "
            "Safe dosing depends on the specific drug, the reason for taking it, age, body size, other medicines, and medical history. "
            "Please check with a pharmacist or clinician for the exact dose."
        )
    elif has_red_flag_symptoms(last_user_text) and _looks_low_quality(reply) and not fallback_context:
        reply = "This could be urgent. Seek emergency care or immediate clinician evaluation right away."
    if fallback_context and _should_prefer_retrieval_fallback(last_user_text, reply, fallback_context):
        reply = _build_retrieval_fallback(last_user_text, fallback_context)

    updated_messages = list(kept_messages)
    updated_messages.append({"role": "assistant", "content": reply})

    completion_tokens = runtime.tokenizer.encode(reply) if reply else []
    return {
        "reply": reply,
        "messages": updated_messages,
        "history": updated_messages,
        "used_context": fallback_context,
        "extra_system_instruction": extra_system_instruction,
        "request_config": request_config,
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": len(completion_tokens),
        "total_tokens": len(prompt_ids) + len(completion_tokens),
    }


def generate_reply(
    runtime: AssistantRuntime,
    user_text: str,
    history: Sequence[dict[str, str]] | None = None,
    system_preset: str = "",
    system_prompt: str = "",
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
    retrieval_top_k: int | None = None,
    disable_retrieval: bool = False,
) -> Dict[str, Any]:
    history = list(history or [])
    history.append({"role": "user", "content": user_text})
    return complete_messages(
        runtime,
        history,
        system_preset=system_preset,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        retrieval_top_k=retrieval_top_k,
        disable_retrieval=disable_retrieval,
    )


def stream_reply_chunks(
    runtime: AssistantRuntime,
    user_text: str,
    history: Sequence[dict[str, str]] | None = None,
    system_preset: str = "",
    system_prompt: str = "",
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
    retrieval_top_k: int | None = None,
    disable_retrieval: bool = False,
) -> Iterator[Dict[str, Any]]:
    result = generate_reply(
        runtime,
        user_text,
        history=history,
        system_preset=system_preset,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        retrieval_top_k=retrieval_top_k,
        disable_retrieval=disable_retrieval,
    )
    yield {"type": "start"}
    for chunk in _chunk_text_for_streaming(str(result["reply"])):
        yield {"type": "delta", "delta": chunk}
    yield {"type": "done", "result": result}
