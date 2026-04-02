from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from grounding import build_grounded_reply, extract_focus_phrase
from presets import DEFAULT_SYSTEM_PRESET, SYSTEM_PROMPTS, asks_for_dosing, build_query_safety_instruction, classify_query_mode, get_generation_defaults, get_system_prompt, has_red_flag_symptoms, is_crisis_query, wants_advice
from retrieval import KnowledgeIndex, STOPWORDS
from server.hosted_provider_client import HostedProviderClient
from server.ollama_client import OllamaClient
from server.settings import ServerSettings
from utils import resolve_checkpoint_path
from web_research import WebResearchClient

try:
    from runtime import AssistantRuntime, _build_retrieval_fallback, _should_prefer_retrieval_fallback, complete_messages, load_runtime
    _RUNTIME_IMPORT_ERROR: Exception | None = None
except Exception as exc:
    AssistantRuntime = Any
    _RUNTIME_IMPORT_ERROR = exc

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

        if len(shared_query_terms) == 0 and len(shared_context_terms) == 0:
            return True

        return False

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
                if score <= 0:
                    continue
                sentences.append((score, sentence))
        sentences.sort(key=lambda item: item[0], reverse=True)
        selected = []
        seen = set()
        for _, sentence in sentences:
            if sentence in seen:
                continue
            seen.add(sentence)
            selected.append(sentence)
            if len(selected) == 2:
                break
        return " ".join(selected).strip()

    def load_runtime(*args, **kwargs):
        raise RuntimeError(
            "Local checkpoint runtime dependencies are not installed. Install requirements-local.txt to use the local model backend."
        ) from _RUNTIME_IMPORT_ERROR

    def complete_messages(*args, **kwargs):
        raise RuntimeError(
            "Local checkpoint runtime dependencies are not installed. Install requirements-local.txt to use the local model backend."
        ) from _RUNTIME_IMPORT_ERROR


FOLLOW_UP_ONLY_TERMS = {
    "again",
    "answer",
    "briefly",
    "clarify",
    "context",
    "deeper",
    "detail",
    "details",
    "elaborate",
    "explain",
    "expanded",
    "further",
    "more",
    "reanswer",
    "rephrase",
    "reword",
    "rewrite",
    "shorter",
    "simpler",
    "sources",
    "that",
    "this",
}

FOLLOW_UP_TOPIC_PATTERNS = {
    "symptoms": ("symptom", "symptoms", "sign", "signs"),
    "causes": ("cause", "causes", "why"),
    "treatment": ("treat", "treatment", "therapy", "manage", "management"),
    "diagnosis": ("diagnosis", "diagnose", "test", "testing"),
    "prevention": ("prevent", "prevention", "avoid"),
    "complications": ("complication", "complications", "risk", "risks"),
    "prognosis": ("prognosis", "outlook", "survival", "recovery"),
    "types": ("type", "types", "kind", "kinds", "forms"),
}
FOLLOW_UP_TOPIC_TERMS = {
    term
    for label, terms in FOLLOW_UP_TOPIC_PATTERNS.items()
    for term in (label, *terms)
}
FAST_GROUNDED_PREFIXES = (
    "explain",
    "how does",
    "how do",
    "what is",
    "what are",
    "what does",
    "what happens if",
    "why does",
    "why do",
    "define",
    "tell me about",
    "symptoms of",
    "causes of",
    "signs of",
)
SYNTHESIS_TERMS = {
    "analyze",
    "analysis",
    "compare",
    "comparison",
    "difference",
    "differentiate",
    "summarize",
    "summary",
    "versus",
    "vs",
}
ANCHOR_NOISE_TERMS = {
    "common",
    "current",
    "today",
    "latest",
    "recent",
    "effect",
    "effects",
    "impact",
    "impacts",
}
BODY_TARGET_TERMS = {
    "brain",
    "chest",
    "colon",
    "eyes",
    "eye",
    "gut",
    "heart",
    "kidney",
    "kidneys",
    "liver",
    "lung",
    "lungs",
    "pancreas",
    "skin",
    "stomach",
}


def _derive_model_id(checkpoint_path: str | Path) -> str:
    path = Path(checkpoint_path)
    if path.is_dir():
        return path.name
    if path.parent.name:
        return path.parent.name
    return path.stem


def _normalize_messages(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for message in messages:
        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", "")).strip()
        if role not in {"system", "user", "assistant"} or not content:
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def _auto_title_from_text(text: str, fallback: str = "New chat") -> str:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return fallback
    return cleaned[:60] + ("..." if len(cleaned) > 60 else "")


def _estimate_text_tokens(text: str) -> int:
    words = text.split()
    return max(1, int(len(words) * 1.3)) if words else 0


def _profile_instruction(
    answer_style: str,
    reader_level: str,
    tone_preference: str = "",
    primary_use: str = "",
    display_name: str = "",
    profile_note: str = "",
) -> str:
    instructions: List[str] = []
    if reader_level == "advanced":
        instructions.append("Use precise language when it helps, but remain readable.")
    else:
        instructions.append("Prefer clear, human language over jargon.")
    if answer_style == "concise":
        instructions.append("Keep the answer short and focused.")
    elif answer_style == "detailed":
        instructions.append("Include a little more explanation, but stay organized.")
    else:
        instructions.append("Keep the answer balanced: concise first, then a short explanation.")
    if tone_preference == "direct":
        instructions.append("Be direct, grounded, and efficient.")
    elif tone_preference == "practical":
        instructions.append("Favor concrete next steps and structured advice.")
    else:
        instructions.append("Stay calm, warm, and polished.")
    if primary_use == "support":
        instructions.append("The user often values emotional clarity and psychologically informed support.")
    elif primary_use == "portfolio":
        instructions.append("The user often values polished explanation of products, ideas, and mission.")
    elif primary_use == "healthcare":
        instructions.append("The user often values careful healthcare explanation.")
    if display_name.strip():
        instructions.append(f"If it feels natural, you may occasionally address the user as {display_name.strip()}.")
    if profile_note.strip():
        instructions.append(f"User context to keep in mind: {profile_note.strip()}")
    return " ".join(instructions)


def _mode_instruction(mode: str, raw_user_text: str) -> str:
    if mode == "crisis":
        return (
            "Respond with immediate safety-first guidance. Be supportive, serious, and direct. "
            "Prioritize real-world help right now and do not drift into normal conversation."
        )
    if mode == "psychology":
        instruction = (
            "Use supportive psychology mode. Start with emotional acknowledgment, reflect the core issue clearly, "
            "notice patterns or loops only when relevant, offer one useful reframe or next step, and ask at most one low-pressure question. "
            "Interpret counsel, support, plan, and follow-up requests in emotional conversations as requests for personal guidance, not legal, political, or historical topics."
        )
        if wants_advice(raw_user_text):
            instruction += " The user is asking for advice, so give practical advice instead of staying purely reflective."
        return instruction
    if mode == "medical":
        return (
            "Use healthcare information mode. Answer directly, rely on grounded sources, separate facts from possibilities, "
            "stay non-diagnostic and non-prescribing, and recommend professional care when appropriate."
        )
    if mode == "portfolio":
        return (
            "Use portfolio representative mode. Explain projects, mission, product value, and design choices clearly and credibly. "
            "Sound polished, intelligent, and concise."
        )
    instruction = "Use general high-quality assistant mode. Answer clearly, directly, and with strong structure."
    if wants_advice(raw_user_text):
        instruction += " Give concrete advice when the user is asking for it."
    return instruction


def _context_instruction(mode: str) -> str:
    if mode == "psychology":
        return (
            "Use the grounded support context only as background. Keep the response human, emotionally intelligent, and specific to the user's situation."
        )
    if mode == "portfolio":
        return (
            "Use the grounded product and portfolio context as the factual basis. Do not invent project details that are not in the context."
        )
    if mode == "medical":
        return (
            "Use the grounded healthcare context as the primary factual basis. If the context is incomplete or conflicting, say so plainly. Do not diagnose or prescribe."
        )
    return (
        "Use the grounded context as the factual basis whenever it is relevant. If it is weak or incomplete, say so instead of guessing."
    )


def _conversation_memory(conversation: Sequence[Dict[str, str]]) -> str:
    if len(conversation) <= 8:
        return ""
    earlier_users = [message["content"].strip() for message in conversation[:-8] if message["role"] == "user" and message["content"].strip()]
    if not earlier_users:
        return ""
    snippets = []
    seen = set()
    for content in reversed(earlier_users):
        snippet = " ".join(content.split())
        if len(snippet) > 90:
            snippet = snippet[:90].rstrip() + "..."
        lowered = snippet.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        snippets.append(snippet)
        if len(snippets) == 3:
            break
    if not snippets:
        return ""
    return "Earlier user context: " + " | ".join(reversed(snippets))


def _compress_conversation(conversation: Sequence[Dict[str, str]]) -> tuple[str, List[Dict[str, str]]]:
    return _conversation_memory(conversation), list(conversation[-8:])


def _normalize_grounding_query(query: str) -> str:
    cleaned = " ".join(query.strip().split())
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip().lower()


def _needs_live_research(query: str, mode: str, local_context_count: int) -> bool:
    if is_crisis_query(query):
        return False
    if mode in {"psychology", "portfolio", "crisis"}:
        return False
    lowered = query.lower()
    if any(token in lowered for token in ("latest", "current", "today", "recent", "new research", "news")):
        return True
    if mode == "medical":
        return local_context_count < 2 or any(
            token in lowered
            for token in (
                "food",
                "drink",
                "candy",
                "soda",
                "snack",
                "supplement",
                "vitamin",
                "ingredient",
                "medication",
                "skittles",
            )
        )
    return local_context_count == 0


def _memory_snapshot(memory_items: Sequence[Dict[str, Any]], conversation_summary: Dict[str, Any] | None = None) -> str:
    lines: List[str] = []
    summary = dict(conversation_summary or {})
    active_topic = " ".join(str(summary.get("active_topic", "")).split()).strip()
    user_need = " ".join(str(summary.get("user_need", "")).split()).strip()
    open_loop = " ".join(str(summary.get("open_loop", "")).split()).strip()
    if active_topic:
        lines.append(f"Active topic: {active_topic}")
    if user_need:
        lines.append(f"Current user need: {user_need}")
    if open_loop:
        lines.append(f"Open thread: {open_loop}")
    for item in list(memory_items)[:6]:
        category = str(item.get("category", "")).strip().replace("_", " ")
        summary_text = " ".join(str(item.get("summary", "")).split()).strip()
        if not summary_text:
            continue
        if category:
            lines.append(f"{category.title()}: {summary_text}")
        else:
            lines.append(summary_text)
    if not lines:
        return ""
    return "Saved personalization memory:\n" + "\n".join(f"- {line}" for line in lines)


def _is_smalltalk_query(text: str) -> bool:
    normalized = re.sub(r"[^a-z0-9\s]", " ", text.strip().lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return False
    if normalized in {
        "hi",
        "hello",
        "hey",
        "yo",
        "sup",
        "good morning",
        "good afternoon",
        "good evening",
        "how are you",
        "are you there",
        "can you hear me",
        "can you help me",
        "hello can you hear me",
        "test",
        "testing",
        "can you respond",
        "can you reply",
    }:
        return True
    words = normalized.split()
    if len(words) <= 4 and words[0] in {"hi", "hello", "hey"}:
        return True
    return False


def _smalltalk_response(raw_user_text: str, resolved_settings: Dict[str, Any]) -> str:
    display_name = " ".join(str(resolved_settings.get("display_name", "")).split()).strip()
    prefix = f"{display_name}, " if display_name else ""
    lowered = raw_user_text.strip().lower()
    if "how are you" in lowered:
        return f"{prefix}I'm here and ready to help. You can ask for advice, emotional support, healthcare research, project explanations, or a direct answer to a general question."
    if any(phrase in lowered for phrase in ("can you hear me", "are you there", "can you respond", "can you reply", "test", "testing")):
        return f"{prefix}Yes, I'm here. Tell me what you want help with, and I'll respond directly."
    return f"{prefix}Hi. What would you like help with?"


def _plan_response_shape(mode: str, raw_user_text: str, answer_style: str) -> str:
    compact = answer_style == "concise"
    expanded = answer_style == "detailed"
    if mode == "crisis":
        return (
            "Reply in a safety-first format: one clear acknowledgment, immediate action steps, and one short follow-up question only if it helps with safety."
        )
    if mode == "psychology":
        if wants_advice(raw_user_text):
            return (
                "Reply like a strong modern assistant: brief acknowledgment, direct advice, two or three concrete next steps, and at most one gentle follow-up question."
            )
        if compact:
            return "Reply in three short parts: acknowledgment, insight, and one useful next step."
        if expanded:
            return "Reply with a calm acknowledgment, a clear pattern insight, a practical next step, and one thoughtful follow-up question."
        return "Reply with a brief acknowledgment, one strong insight, one practical next step, and one optional follow-up question."
    if mode == "medical":
        if compact:
            return "Start with the direct answer, then add one short explanatory paragraph grounded in the best evidence."
        if expanded:
            return "Start with the direct answer, then give a concise explanation, then add source-backed details and uncertainty if needed."
        return "Start with the direct answer, follow with a short explanation, and keep the rest tightly grounded in the retrieved sources."
    if mode == "portfolio":
        return "Lead with what it is and why it matters, then explain the value clearly and credibly in natural language."
    if compact:
        return "Lead with the answer, then add only the most useful detail."
    if expanded:
        return "Lead with the answer, then add a short structured explanation with the most useful detail or example."
    return "Lead with the answer, then add a concise explanation and the most useful supporting detail."


def _build_conversation_state(
    messages: Sequence[Dict[str, str]],
    mode: str,
    raw_user_text: str,
) -> Dict[str, str]:
    recent_users = [message["content"].strip() for message in messages if message["role"] == "user" and message["content"].strip()]
    active_topic = extract_focus_phrase(raw_user_text or (recent_users[-1] if recent_users else ""))
    if not active_topic:
        active_topic = " ".join(raw_user_text.split())[:140]
    if mode == "psychology":
        user_need = "direct advice" if wants_advice(raw_user_text) else "reflection and support"
    elif mode == "medical":
        user_need = "careful medical explanation"
    elif mode == "portfolio":
        user_need = "project and mission explanation"
    elif mode == "crisis":
        user_need = "immediate safety support"
    else:
        user_need = "clear general explanation"
    open_loop = ""
    if recent_users:
        recent = recent_users[-1].strip()
        if recent.endswith("?"):
            open_loop = "The user is actively asking for an answer."
        elif mode == "psychology":
            open_loop = "The user may need emotionally aware follow-through."
    return {
        "active_topic": active_topic,
        "user_need": user_need,
        "open_loop": open_loop,
        "mode": mode,
    }


def _extract_memory_items(
    messages: Sequence[Dict[str, str]],
    raw_user_text: str,
    resolved_settings: Dict[str, Any],
) -> List[Dict[str, Any]]:
    memory_items: List[Dict[str, Any]] = []
    display_name = str(resolved_settings.get("display_name", "")).strip()
    if display_name:
        memory_items.append({"category": "identity", "summary": f"Preferred name: {display_name}"})

    site_context = str(resolved_settings.get("site_context", "")).strip()
    if site_context:
        memory_items.append({"category": "context", "summary": f"Current context: {site_context}"})

    profile_note = " ".join(str(resolved_settings.get("profile_note", "")).split()).strip()
    if profile_note:
        memory_items.append({"category": "preference", "summary": profile_note[:180]})

    lowered = raw_user_text.lower()
    if wants_advice(raw_user_text):
        memory_items.append({"category": "support_preference", "summary": "Often wants direct practical advice."})
    if re.search(r"\bcall me ([a-z][a-z0-9 _-]{1,30})", lowered):
        match = re.search(r"\bcall me ([a-z][a-z0-9 _-]{1,30})", lowered)
        if match:
            memory_items.append({"category": "identity", "summary": f"Preferred name: {match.group(1).strip().title()}"})
    if re.search(r"\b(i am|i'm|im) working on\b", lowered) or re.search(r"\b(i am|i'm|im) building\b", lowered):
        snippet = " ".join(raw_user_text.split())[:180]
        memory_items.append({"category": "project", "summary": snippet})
    if re.search(r"\bmy goal is\b", lowered) or re.search(r"\bi want to\b", lowered):
        snippet = " ".join(raw_user_text.split())[:180]
        memory_items.append({"category": "goal", "summary": snippet})
    if any(token in lowered for token in ("overwhelmed", "burnout", "burned out", "anxious", "anxiety", "avoid", "procrast")):
        snippet = " ".join(raw_user_text.split())[:180]
        memory_items.append({"category": "support_context", "summary": snippet})

    deduped: List[Dict[str, Any]] = []
    seen = set()
    for item in memory_items:
        key = (item["category"], item["summary"].lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped[:8]


def _crisis_response(raw_user_text: str) -> str:
    lowered = raw_user_text.lower()
    if any(token in lowered for token in ("hurt someone", "kill someone")):
        return (
            "This sounds urgent.\n\n"
            "Please move away from anything you could use to hurt someone, contact emergency services right now, and get immediate in-person support from someone nearby.\n\n"
            "If you can, tell me whether you are alone right now."
        )
    if is_crisis_query(raw_user_text):
        return (
            "I'm really sorry you're going through this, and I'm glad you said it out loud.\n\n"
            "If there is any chance you might act on this or you do not feel safe, call or text 988 right now if you're in the U.S., "
            "or contact local emergency services / go to the nearest emergency room.\n\n"
            "If you can, reach out to one trusted person right now and tell them you need them to stay with you or help you get immediate support.\n\n"
            "Are you in immediate danger right now?"
        )
    return (
        "This could be urgent.\n\n"
        "Please seek emergency medical care or immediate clinician evaluation right away, especially if you feel unsafe, severely unwell, or unable to manage this on your own."
    )


def _crisis_followup_response(raw_user_text: str) -> str:
    lowered = raw_user_text.lower().strip()
    if "friend" in lowered or "friends" in lowered or "family" in lowered or "partner" in lowered:
        return (
            "Yes, involve them now if there is anyone you trust.\n\n"
            "Send a direct message like: \"I'm not safe being alone right now and I need you to stay with me or help me get support.\" "
            "Then call or text 988 if you're in the U.S., or your local emergency number if you might act on this.\n\n"
            "If you want, I can help you word that message in one sentence."
        )
    if lowered in {"yes", "yeah", "yep", "i am", "im", "i'm"}:
        return (
            "Thank you for telling me.\n\n"
            "Please call or text 988 right now if you're in the U.S., or call local emergency services now. "
            "If you can, put this chat down for a moment and contact one trusted person immediately so you are not dealing with this alone.\n\n"
            "Are you alone right now?"
        )
    if lowered in {"no", "not right now", "i don't think so", "i dont think so"}:
        return (
            "Thank you for answering.\n\n"
            "Even if you do not think you'll act on it right this second, I still want to take this seriously. "
            "Please text or call 988 if you're in the U.S., or contact a trusted person now and let them know you're struggling.\n\n"
            "Can you reach someone right now?"
        )
    if "alone" in lowered:
        return (
            "Thank you for telling me.\n\n"
            "Please contact one person right now and tell them you need them with you, then call or text 988 if you're in the U.S. or your local emergency number if you might act on this.\n\n"
            "If you want, send the message now: \"I'm not safe being alone and I need you with me right now.\""
        )
    return (
        "I'm staying with you on the safety issue here.\n\n"
        "Please contact 988 if you're in the U.S. or your local emergency number if you're at risk of acting on this, and reach out to one trusted person right now so you're not alone with it."
    )


def _supportive_fallback(
    raw_user_text: str,
    resolved_settings: Dict[str, Any] | None = None,
    memory_items: Sequence[Dict[str, Any]] | None = None,
    conversation_summary: Dict[str, Any] | None = None,
) -> str:
    settings = dict(resolved_settings or {})
    lowered = raw_user_text.lower()
    name = str(settings.get("display_name", "")).strip()
    prefix = f"{name}, " if name else ""
    memory_text = _memory_snapshot(memory_items or [], conversation_summary)
    summary_text = " ".join(
        " ".join(str(value).split())
        for value in (conversation_summary or {}).values()
        if str(value).strip()
    ).lower()
    goal_hint = ""
    if "project" in memory_text.lower() or "goal" in memory_text.lower():
        goal_hint = " Keep the advice realistic for what they are actively working on."
    if "friend" in lowered or "friends" in lowered or "family" in lowered:
        return (
            f"{prefix}If this is about depression or feeling unsafe, the most useful move is not to protect everyone else from it. "
            "Pick one person who is steady, tell them plainly that you're struggling, and ask for one specific thing: a call, company, a ride, or help making the next safe step."
        )
    if lowered.strip() in {"a plan", "plan", "a concrete plan", "concrete plan"} or (
        "plan" in lowered and len(re.findall(r"[a-z0-9']+", lowered)) <= 5
    ):
        if any(term in f"{lowered} {summary_text} {memory_text.lower()}" for term in ("depress", "lost desire", "hopeless", "empty", "can t try", "can't try")):
            return (
                f"{prefix}Here is a practical plan for the next few hours: first, do one body-level reset such as water, food, a shower, or stepping outside for five minutes. "
                "Second, choose one tiny task that proves the day is not completely shut down, like replying to one message or clearing one surface. "
                "Third, tell one trusted person that you're having a rough day instead of carrying it alone."
            )
        return (
            f"{prefix}Here is a simple plan: name the main problem in one sentence, choose the smallest step that would reduce pressure today, and do only that step before deciding what comes next. "
            "If you want, give me the situation and I will turn it into a tighter plan."
        )
    if any(
        phrase in lowered
        for phrase in (
            "help me with my depression",
            "help me with depression",
            "lost all desire",
            "no desire to try",
            "dont want to try anymore",
            "don't want to try anymore",
            "can't try anymore",
            "cant try anymore",
            "feel hopeless",
            "feeling hopeless",
            "feel empty",
            "feeling empty",
        )
    ):
        return (
            f"{prefix}That sounds heavier than ordinary stress. When you lose desire to try, everything starts to feel far away and effort can feel pointless. "
            "Do not ask yourself to fix your whole life tonight. Focus on one stabilizing step, one low-pressure task, and one human contact. "
            "If you want, I can help you build that plan around what is hardest right now."
        )
    if "depress" in lowered and not lowered.startswith(("what is depression", "define depression", "what's depression")):
        return (
            f"{prefix}Depression can make everything feel slowed down, flat, or impossible to care about, even when part of you still wants things to be different. "
            "A useful first move is to lower the demand completely and focus on stabilization: one basic care step, one very small task, and one point of contact with another person. "
            "If you want, I can help you make that concrete for tonight."
        )
    if "counsel me" in lowered or "support me" in lowered:
        return (
            f"{prefix}I can do that. Tell me the part that feels heaviest right now, and I will respond clearly and practically. "
            "If you already know what you want, say whether you need perspective, a plan, or help naming what you are feeling."
        )
    if ("avoid" in lowered or "avoiding" in lowered or "putting off" in lowered or "procrast" in lowered) and ("guilt" in lowered or "guilty" in lowered):
        return (
            f"{prefix}It sounds like you may be stuck in a loop where avoidance brings short-term relief, then guilt makes the work feel even heavier. "
            f"A practical move is to shrink the task until it feels almost too easy, do only the first five to ten minutes, and stop treating the whole project like one giant decision.{goal_hint}"
        )
    if any(token in lowered for token in ("overwhelm", "overwhelmed")) and any(
        token in lowered for token in ("avoid", "avoiding", "procrast", "work", "task", "behind", "deadline")
    ):
        return (
            f"{prefix}This sounds like overload turning into avoidance, and then avoidance making the work feel even larger. "
            "The move here is not to motivate yourself into a huge catch-up session. Pick the one task that would reduce the most pressure, define the smallest visible version of it, and work on only that for fifteen minutes. "
            "If you want, I can help you turn the thing you're avoiding into a concrete first step."
        )
    if "overwhelm" in lowered or "overwhelmed" in lowered:
        return (
            f"{prefix}That sounds like a lot to carry at once. When overwhelm piles up, it can make everything feel heavier than it is. "
            "A useful next step is to separate what needs attention today from what is just pressing on you mentally. Would it help to sort that out together?"
        )
    if "anx" in lowered or "panic" in lowered or "worry" in lowered:
        return (
            f"{prefix}That sounds stressful and mentally noisy. Anxiety can make the next step feel bigger and less clear than it really is. "
            "Try naming the specific fear, then choose one small action that would make the situation 10 percent more manageable."
        )
    if "burnout" in lowered or "exhaust" in lowered or "drained" in lowered:
        return (
            f"{prefix}You sound worn down, not just stuck. When exhaustion and pressure stack together, even simple tasks can start feeling impossible. "
            "It may help to lower the bar for the next step and focus on one thing that protects your energy today."
        )
    return (
        f"{prefix}It sounds like you need something more useful than a vague pep talk. I can help you think it through clearly and keep it practical. "
        "Tell me the part you want most right now: perspective, a plan, or help sorting the situation out."
    )


def _portfolio_fallback(raw_user_text: str, contexts: Sequence[Dict[str, str | float]]) -> str:
    lowered = raw_user_text.lower()
    if "recall" in lowered:
        return (
            "Recall is a mental wellness product focused on emotional reflection, pattern recognition, supportive conversation, and helping users move from confusion to clarity. "
            "It matters because it combines psychological usefulness, safety, and product polish without pretending to replace therapy, psychiatry, or emergency care."
        )
    if "medbrief" in lowered:
        return (
            "MedBrief AI is a premium assistant designed to blend psychology-aware support, careful healthcare information, polished project explanation, and strong general help. "
            "Its value is that it adapts to the user’s intent while staying calm, trustworthy, and grounded."
        )
    if any(phrase in lowered for phrase in ("what kind of projects", "what projects", "featured in this portfolio", "in this portfolio", "portfolio feature")):
        return (
            "This portfolio centers on digital health and healthcare innovation work. The featured projects combine psychology-aware support, careful healthcare information, "
            "AI-assisted user experiences, and polished product design, with MedBrief AI and Recall representing the strongest examples of that direction."
        )
    if contexts:
        text = " ".join(str(contexts[0].get("text", "")).split())
        if len(text) > 320:
            text = text[:320].rstrip() + "..."
        return text or "This project is designed to be credible, polished, and useful in a healthcare and psychology-aware environment."
    return "This project is designed to be credible, polished, and useful in a healthcare and psychology-aware environment."


def _portfolio_sources(raw_user_text: str, contexts: Sequence[Dict[str, str | float]]) -> List[Dict[str, str | float]]:
    lowered = raw_user_text.lower()
    if "recall" in lowered:
        matching = [item for item in contexts if "recall" in " ".join(str(item.get(key, "")).lower() for key in ("title", "source", "text"))]
        if matching:
            return matching[:2]
        return []
    if "medbrief" in lowered:
        matching = [item for item in contexts if "medbrief" in " ".join(str(item.get(key, "")).lower() for key in ("title", "source", "text"))]
        if matching:
            return matching[:2]
        return []
    return list(contexts[:2])


def _needs_last_resort_reply(reply: str) -> bool:
    lowered = reply.strip().lower()
    return (
        not lowered
        or "i cannot verify enough source material" in lowered
        or "i ran into a response problem" in lowered
        or "please ask again" in lowered
    )


def _context_sentence_summary(contexts: Sequence[Dict[str, str | float]], max_sentences: int = 2) -> str:
    sentences: List[str] = []
    seen: set[str] = set()
    for item in contexts:
        text = " ".join(str(item.get("text", "")).split())
        if not text:
            continue
        for candidate in re.split(r"(?<=[.!?])\s+", text):
            cleaned = " ".join(candidate.split()).strip()
            if len(cleaned) < 40:
                continue
            lowered = cleaned.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            sentences.append(cleaned)
            if len(sentences) >= max_sentences:
                return " ".join(sentences)
    return " ".join(sentences)


def _medical_last_resort_reply(raw_user_text: str, contexts: Sequence[Dict[str, str | float]]) -> str:
    lowered = raw_user_text.lower()
    if "lupus" in lowered and "nephritis" not in lowered:
        return (
            "Lupus, usually referring to systemic lupus erythematosus (SLE), is a chronic autoimmune disease in which the immune system attacks healthy tissues in different parts of the body. "
            "It can affect the skin, joints, kidneys, blood cells, heart, lungs, and brain."
        )
    context_summary = _context_sentence_summary(contexts)
    if context_summary:
        return context_summary
    if any(token in lowered for token in ("gushers", "skittles", "candy", "fruit snack", "fruit snacks")):
        if any(token in lowered for token in ("intestine", "intestines", "gut", "bowel", "bowels", "digestive", "stomach", "colon")):
            return (
                "Occasional sugary candy such as Gushers or Skittles is not known for a unique direct injury to the intestines in an otherwise healthy person. "
                "The more typical effect of eating a lot at once is digestive upset, such as stomach discomfort, bloating, nausea, or diarrhea. "
                "The bigger concern is frequent high added-sugar intake over time rather than isolated intestinal damage."
            )
        if "liver" in lowered:
            return (
                "Occasional candy such as Skittles or Gushers is not known for a unique direct toxic effect on the liver in an otherwise healthy person. "
                "The bigger issue is repeated high added-sugar intake over time, which can contribute to weight gain, insulin resistance, high triglycerides, and fatty liver disease."
            )
        if any(token in lowered for token in ("kidney", "kidneys", "renal")):
            return (
                "Sugary candy does not have a unique kidney-specific effect from a small occasional serving in an otherwise healthy person. "
                "The larger concern is long-term high added-sugar intake because it can worsen diabetes, blood pressure, and metabolic health, which are major risk factors for kidney disease."
            )
    return (
        "I can give a careful general answer, but I do not have enough strong evidence here to be more specific. "
        "In general, the effect depends on the substance, the amount, how often it is used, and the person's overall health context."
    )


def _medical_reply_needs_repair(raw_user_text: str, reply: str) -> bool:
    lowered_query = raw_user_text.strip().lower()
    lowered_reply = reply.strip().lower()
    if not lowered_reply:
        return True
    if lowered_query.startswith(("what is ", "what's ", "define ", "definition of ")) and lowered_reply.startswith("examples include"):
        return True
    if "lupus" in lowered_query and "nephritis" not in lowered_query and "lupus nephritis" in lowered_reply:
        return True
    return False


def _general_last_resort_reply(raw_user_text: str, contexts: Sequence[Dict[str, str | float]]) -> str:
    context_summary = _context_sentence_summary(contexts)
    if context_summary:
        return context_summary
    lowered = raw_user_text.lower()
    if "portfolio" in lowered or "project" in lowered or "website" in lowered:
        return _portfolio_fallback(raw_user_text, contexts)
    return (
        "Here is the most careful direct answer I can give right now: I do not have enough strong evidence to make a precise claim on that exact phrasing, "
        "so the safest answer is to keep it general rather than guess."
    )


def _psychology_reply_off_topic(raw_user_text: str, reply: str) -> bool:
    lowered_query = raw_user_text.lower()
    lowered_reply = reply.lower()
    if any(
        phrase in lowered_reply
        for phrase in (
            "special counsel",
            "robert mueller",
            "donald trump",
            "russian president",
            "criminal investigation",
        )
    ):
        return True
    if not any(
        phrase in lowered_query
        for phrase in (
            "counsel",
            "depress",
            "anx",
            "overwhelm",
            "burnout",
            "lonely",
            "guilt",
            "sad",
            "hopeless",
            "desire",
            "lost",
            "avoid",
            "procrast",
            "stuck",
            "panic",
            "emotion",
            "feeling",
        )
    ):
        return False
    return not any(
        phrase in lowered_reply
        for phrase in (
            "feel",
            "feeling",
            "support",
            "step",
            "plan",
            "stress",
            "pressure",
            "overwhelm",
            "anx",
            "depress",
            "friend",
            "family",
            "work",
            "rest",
            "sleep",
            "guilt",
            "loop",
            "pattern",
        )
    )


def _portfolio_reply_off_topic(reply: str) -> bool:
    lowered = reply.lower()
    if any(term in lowered for term in ("portfolio", "project", "product", "medbrief", "recall", "healthcare", "assistant", "innovation", "tool")):
        return False
    return any(term in lowered for term in ("nephritis", "kidney", "dosage", "suicide", "988", "donald trump", "special counsel"))


def _should_use_fast_grounded_answer(raw_user_text: str, effective_query: str, contexts: Sequence[Dict[str, str | float]]) -> bool:
    if not contexts:
        return False
    lowered = raw_user_text.strip().lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", lowered)
    words = [word for word in normalized.split() if word]
    if not words:
        return False
    if _looks_ambiguous_followup(lowered):
        return True
    if lowered.startswith(("what about ", "how about ", "and ", "also ")):
        return True
    if any(word in SYNTHESIS_TERMS for word in words):
        return False
    if lowered.startswith(FAST_GROUNDED_PREFIXES):
        return True
    if len(words) <= 4:
        return True
    effective_words = [word for word in re.findall(r"[a-z0-9']+", effective_query.lower()) if word]
    return len(effective_words) <= 4


def _anchor_terms(query: str) -> set[str]:
    terms = {
        term
        for term in re.findall(r"[a-zA-Z0-9']+", query.lower())
        if len(term) > 2 and term not in STOPWORDS and term not in ANCHOR_NOISE_TERMS
    }
    return {term for term in terms if term not in FOLLOW_UP_TOPIC_TERMS}


def _context_matches_anchor(item: Dict[str, str | float], anchor_terms: set[str]) -> bool:
    if not anchor_terms:
        return True
    haystack = " ".join(
        str(item.get(key, "")).lower()
        for key in ("text", "title", "source", "web_domain", "domain")
    )
    matched = {term for term in anchor_terms if term in haystack}
    if len(anchor_terms) <= 2:
        return len(matched) == len(anchor_terms)
    return len(matched) / max(1, len(anchor_terms)) >= 0.6


def _anchor_overlap_ratio(item: Dict[str, str | float], anchor_terms: set[str]) -> float:
    if not anchor_terms:
        return 1.0
    haystack = " ".join(
        str(item.get(key, "")).lower()
        for key in ("text", "title", "source", "web_domain", "domain", "url")
    )
    matched = {term for term in anchor_terms if term in haystack}
    return len(matched) / max(1, len(anchor_terms))


def _exact_topic_match(item: Dict[str, str | float], anchor_terms: set[str]) -> bool:
    if not anchor_terms:
        return False
    haystack = " ".join(
        str(item.get(key, "")).lower()
        for key in ("title", "source", "url", "web_domain", "domain")
    )
    return all(term in haystack for term in anchor_terms)


def _context_mentions_any(item: Dict[str, str | float], terms: set[str]) -> bool:
    if not terms:
        return True
    haystack = " ".join(
        str(item.get(key, "")).lower()
        for key in ("text", "title", "source", "web_domain", "domain", "url")
    )
    return any(term in haystack for term in terms)


def _last_message(messages: Sequence[Dict[str, str]], role: str, before_index: int) -> str:
    for index in range(before_index - 1, -1, -1):
        message = messages[index]
        if message["role"] == role and message["content"].strip():
            return message["content"].strip()
    return ""


def _last_substantive_user_message(messages: Sequence[Dict[str, str]], before_index: int) -> str:
    fallback = ""
    for index in range(before_index - 1, -1, -1):
        message = messages[index]
        if message["role"] != "user":
            continue
        content = message["content"].strip()
        if not content:
            continue
        if not fallback:
            fallback = content
        if not _looks_ambiguous_followup(content):
            return content
    return fallback


def _looks_ambiguous_followup(text: str) -> bool:
    lowered = text.strip().lower()
    if not lowered:
        return True
    normalized = re.sub(r"[^a-z0-9\s]", " ", lowered)
    words = [word for word in normalized.split() if word]
    if not words:
        return True
    if len(words) <= 3 and all(word in FOLLOW_UP_ONLY_TERMS for word in words):
        return True
    if lowered in {
        "again",
        "answer again",
        "explain again",
        "reanswer",
        "rephrase",
        "reword",
        "rewrite",
        "try again",
    }:
        return True
    return False


def _is_short_followup(text: str) -> bool:
    words = [word for word in re.findall(r"[a-z0-9']+", text.lower()) if word]
    return 0 < len(words) <= 8


def _looks_like_crisis_followup(text: str) -> bool:
    lowered = re.sub(r"\s+", " ", text.strip().lower())
    if not lowered:
        return False
    if lowered in {
        "yes",
        "yeah",
        "yep",
        "no",
        "not right now",
        "i am",
        "im",
        "i'm",
        "alone",
        "not alone",
        "unsafe",
        "i feel unsafe",
        "help",
        "please help",
        "what do i do",
        "what should i do",
        "can you stay with me",
        "stay with me",
    }:
        return True
    return any(
        phrase in lowered
        for phrase in (
            "am i safe",
            "i am alone",
            "i'm alone",
            "not safe",
            "need help now",
            "stay with me",
            "reach someone",
            "call someone",
            "what do i do now",
        )
    )


def _recent_mode_hint(
    normalized_messages: Sequence[Dict[str, str]],
    resolved_settings: Dict[str, Any],
) -> str:
    summary_mode = str((resolved_settings.get("conversation_summary") or {}).get("mode", "")).strip().lower()
    if summary_mode in {"crisis", "psychology", "medical", "portfolio"}:
        return summary_mode
    primary_use = str(resolved_settings.get("primary_use", "balanced"))
    site_context = str(resolved_settings.get("site_context", ""))
    for message in reversed(normalized_messages[:-1]):
        if message["role"] != "user":
            continue
        content = message["content"].strip()
        if not content:
            continue
        mode = classify_query_mode(content, primary_use=primary_use, site_context=site_context)
        if mode != "general":
            return mode
    return "general"


def _resolve_mode(
    normalized_messages: Sequence[Dict[str, str]],
    raw_user_text: str,
    resolved_settings: Dict[str, Any],
) -> str:
    lowered = raw_user_text.strip().lower()
    base_mode = classify_query_mode(
        raw_user_text,
        primary_use=str(resolved_settings.get("primary_use", "balanced")),
        site_context=str(resolved_settings.get("site_context", "")),
    )
    recent_mode = _recent_mode_hint(normalized_messages, resolved_settings)
    if base_mode == "crisis":
        return base_mode
    if recent_mode == "crisis" and (_looks_like_crisis_followup(raw_user_text) or _looks_ambiguous_followup(raw_user_text) or lowered.startswith(("what about ", "how about ", "and ", "also ", "then ", "now "))):
        return "crisis"
    if base_mode != "general":
        return base_mode
    if not _is_short_followup(raw_user_text):
        return base_mode
    if recent_mode in {"psychology", "medical", "portfolio"} and (
        _looks_ambiguous_followup(raw_user_text)
        or lowered.startswith(("what about ", "how about ", "and ", "also ", "then ", "now "))
        or len(re.findall(r"[a-z0-9']+", lowered)) <= 4
    ):
        return recent_mode
    if recent_mode == "psychology" and any(
        token in lowered
        for token in (
            "plan",
            "friends",
            "family",
            "partner",
            "relationship",
            "school",
            "work",
            "job",
            "today",
            "tonight",
            "tomorrow",
            "advice",
            "help",
        )
    ):
        return "psychology"
    return base_mode


def _resolved_grounding_query(messages: Sequence[Dict[str, str]]) -> str:
    user_indexes = [index for index, message in enumerate(messages) if message["role"] == "user"]
    if not user_indexes:
        return ""
    last_user_index = user_indexes[-1]
    last_user_text = messages[last_user_index]["content"].strip()
    if not last_user_text:
        return ""

    previous_user_text = _last_message(messages, "user", last_user_index)
    anchor_source_text = _last_substantive_user_message(messages, last_user_index)
    previous_assistant_text = _last_message(messages, "assistant", last_user_index)
    lowered = last_user_text.lower()
    anchor = extract_focus_phrase(anchor_source_text) if anchor_source_text else ""
    followup_topic_hint = _looks_ambiguous_followup(last_user_text) or lowered.startswith(
        ("what about ", "how about ", "and ", "also ", "now ", "then ")
    )

    if not previous_user_text:
        return last_user_text

    for label, tokens in FOLLOW_UP_TOPIC_PATTERNS.items():
        if followup_topic_hint and any(token in lowered for token in tokens) and anchor:
            return f"{anchor} {label}"

    if _looks_ambiguous_followup(last_user_text):
        return anchor_source_text or previous_user_text

    if any(phrase in lowered for phrase in ("sources", "where did you get", "cite", "citation")):
        return anchor_source_text or previous_user_text

    if lowered.startswith(("what about ", "how about ", "and ", "also ", "now ", "then ")):
        return f"{anchor_source_text or previous_user_text}. Follow-up: {last_user_text}"

    reference_terms = set(re.findall(r"[a-z0-9']+", lowered))
    if reference_terms.intersection({"that", "this", "it", "those", "them"}) and len(last_user_text.split()) <= 10:
        anchor = anchor or anchor_source_text or previous_user_text
        if previous_assistant_text:
            return f"{anchor}. Follow-up request: {last_user_text}. Prior answer: {previous_assistant_text[:240]}"
        return f"{anchor}. Follow-up request: {last_user_text}"

    return last_user_text


class LocalAssistantService:
    def __init__(self, settings: ServerSettings):
        self.settings = settings
        self.checkpoint_model_id = _derive_model_id(settings.checkpoint)
        self.runtime: AssistantRuntime | None = None
        self.knowledge_index = self._load_knowledge_index()
        self.web_research = WebResearchClient(
            cache_dir=settings.web_cache_dir,
            timeout_seconds=settings.web_timeout_seconds,
        )
        self.ollama_client = OllamaClient(
            base_url=settings.ollama_base_url,
            timeout_seconds=settings.ollama_timeout_seconds,
            keep_alive=settings.ollama_keep_alive,
        )
        self.hosted_provider_client = HostedProviderClient(
            base_url=settings.provider_base_url,
            api_key=settings.provider_api_key,
            timeout_seconds=settings.provider_timeout_seconds,
        )
        self._ollama_models = self._load_ollama_models()
        self._embedding_cache: Dict[str, np.ndarray] = {}

    @property
    def model_id(self) -> str:
        backend = self._select_model_backend(self.settings.model_backend)
        if backend == "hosted_api":
            return self._resolve_hosted_model(self.settings.provider_model)
        if backend == "ollama":
            return self._resolve_ollama_model(self.settings.ollama_model)
        return self.checkpoint_model_id

    def _load_ollama_models(self) -> List[str]:
        try:
            return self.ollama_client.list_models()
        except Exception:
            return []

    def _get_ollama_models(self, refresh: bool = False) -> List[str]:
        if refresh or not self._ollama_models:
            self._ollama_models = self._load_ollama_models()
        return list(self._ollama_models)

    def _ollama_available(self) -> bool:
        return bool(self._get_ollama_models())

    def _hosted_provider_available(self) -> bool:
        return self.settings.hosted_provider_configured and self.hosted_provider_client.configured

    def _supports_chat_backend(self, backend: str) -> bool:
        return backend in {"ollama", "hosted_api"}

    def _select_model_backend(self, requested_backend: str | None) -> str:
        candidate = str(requested_backend or self.settings.model_backend or "auto").strip().lower()
        if candidate == "hosted_api":
            if self._hosted_provider_available():
                return "hosted_api"
            return "ollama" if self._ollama_available() else "custom"
        if candidate == "ollama":
            return "ollama" if self._ollama_available() else "custom"
        if candidate == "custom":
            return "custom"
        if candidate == "auto":
            if self._hosted_provider_available():
                return "hosted_api"
            return "ollama" if self._ollama_available() else "custom"
        if self._hosted_provider_available():
            return "hosted_api"
        return "ollama" if self._ollama_available() else "custom"

    def _resolve_hosted_model(self, requested_model: str | None = None) -> str:
        return str(requested_model or self.settings.provider_model or "").strip()

    def _default_chat_model(self, requested_backend: str | None = None) -> str:
        backend = self._select_model_backend(requested_backend)
        if backend == "hosted_api":
            return self._resolve_hosted_model()
        if backend == "ollama":
            return self._resolve_ollama_model(self.settings.ollama_model)
        return ""

    def _resolve_ollama_model(self, requested_model: str | None = None) -> str:
        preferred = str(requested_model or self.settings.ollama_model or "").strip()
        models = self._get_ollama_models(refresh=not bool(self._ollama_models))
        if preferred and preferred in models:
            return preferred
        for candidate in ("qwen2.5:14b", "qwen2.5:7b", "llama3.1:8b", "mistral:7b", "qwen2.5-coder:14b"):
            if candidate in models:
                return candidate
        return preferred or (models[0] if models else "")

    def _resolve_embedding_model(self) -> str:
        preferred = str(self.settings.ollama_embedding_model or "").strip()
        models = self._get_ollama_models(refresh=not bool(self._ollama_models))
        if preferred and preferred in models:
            return preferred
        for candidate in ("nomic-embed-text", "mxbai-embed-large", "snowflake-arctic-embed2"):
            if candidate in models:
                return candidate
        return ""

    def _embed_texts(self, texts: Sequence[str]) -> List[np.ndarray]:
        if not self.settings.semantic_retrieval_enabled or not self._ollama_available():
            return []
        model_name = self._resolve_embedding_model()
        if not model_name:
            return []
        cleaned = [" ".join(str(text).split()).strip() for text in texts]
        if not any(cleaned):
            return []

        vectors: List[np.ndarray] = []
        missing_indexes: List[int] = []
        missing_payload: List[str] = []
        for index, text in enumerate(cleaned):
            cache_key = f"{model_name}::{text[:1200]}"
            cached = self._embedding_cache.get(cache_key)
            if cached is not None:
                vectors.append(cached)
                continue
            vectors.append(np.array([], dtype=float))
            missing_indexes.append(index)
            missing_payload.append(text[:1200])
        if missing_payload:
            try:
                embedded = self.ollama_client.embed(model=model_name, inputs=missing_payload)
            except Exception:
                return []
            if len(embedded) != len(missing_payload):
                return []
            for index, vector in zip(missing_indexes, embedded):
                array = np.array(vector, dtype=float)
                cache_key = f"{model_name}::{cleaned[index][:1200]}"
                self._embedding_cache[cache_key] = array
                vectors[index] = array
        if any(vector.size == 0 for vector in vectors):
            return []
        return vectors

    def _semantic_rerank(self, query: str, contexts: Sequence[Dict[str, str | float]]) -> List[Dict[str, str | float]]:
        candidate_contexts = list(contexts)
        if len(candidate_contexts) <= 1:
            return candidate_contexts
        limited = candidate_contexts[: max(2, int(self.settings.semantic_rerank_limit))]
        texts = [query]
        texts.extend(
            " ".join(
                part
                for part in (
                    str(item.get("title", "")).strip(),
                    " ".join(str(item.get("text", "")).split())[:1400],
                )
                if part
            )
            for item in limited
        )
        vectors = self._embed_texts(texts)
        if len(vectors) != len(texts):
            return candidate_contexts
        query_vector = vectors[0]
        query_norm = float(np.linalg.norm(query_vector))
        if query_norm == 0.0:
            return candidate_contexts
        reranked: List[tuple[float, Dict[str, str | float]]] = []
        for item, vector in zip(limited, vectors[1:]):
            vector_norm = float(np.linalg.norm(vector))
            if vector_norm == 0.0:
                semantic_score = 0.0
            else:
                semantic_score = float(np.dot(query_vector, vector) / (query_norm * vector_norm))
            lexical_score = float(item.get("score", 0.0) or 0.0)
            trust_score = float(item.get("trust_tier", 0.0) or 0.0)
            combined = semantic_score * 4.4 + lexical_score + trust_score / 6.0
            reranked.append((combined, item))
        reranked.sort(key=lambda pair: pair[0], reverse=True)
        ordered = [item for _, item in reranked]
        if len(candidate_contexts) > len(limited):
            ordered.extend(candidate_contexts[len(limited):])
        return ordered

    def _load_knowledge_index(self) -> KnowledgeIndex | None:
        candidate = Path(self.settings.knowledge_index_path)
        if candidate.exists():
            return KnowledgeIndex.load(candidate)
        return None

    def _ensure_runtime(self) -> AssistantRuntime:
        if self.runtime is None:
            self.runtime = load_runtime(
                checkpoint=self.settings.checkpoint,
                tokenizer_model=self.settings.tokenizer_model,
                device_name=self.settings.device,
                system_preset=self.settings.system_preset,
                system_prompt=self.settings.system_prompt,
                knowledge_index_path=self.settings.knowledge_index_path,
                retrieval_top_k=self.settings.retrieval_top_k,
                disable_retrieval=self.settings.disable_retrieval,
            )
        return self.runtime

    def _default_system_prompt(self, system_preset: str, system_prompt: str = "") -> str:
        return get_system_prompt(system_preset or self.settings.system_preset, override=system_prompt or self.settings.system_prompt)

    def _resolve_settings(self, settings_overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
        overrides = dict(settings_overrides or {})
        system_preset = str(overrides.get("system_preset") or self.settings.system_preset or DEFAULT_SYSTEM_PRESET)
        generation_defaults = get_generation_defaults(system_preset)
        requested_backend = str(overrides.get("model_backend") or self.settings.model_backend or "auto").lower()
        requested_max_new_tokens = int(overrides.get("max_new_tokens", generation_defaults["max_new_tokens"]))
        clamped_max_new_tokens = max(32, min(requested_max_new_tokens, self.settings.max_completion_tokens))
        return {
            "model_backend": requested_backend,
            "model": str(overrides.get("model") or self._default_chat_model(requested_backend)).strip(),
            "answer_style": str(overrides.get("answer_style") or "balanced").lower(),
            "reader_level": str(overrides.get("reader_level") or "everyday").lower(),
            "tone_preference": str(overrides.get("tone_preference") or "calm").lower(),
            "primary_use": str(overrides.get("primary_use") or "balanced").lower(),
            "display_name": str(overrides.get("display_name") or "").strip(),
            "profile_note": str(overrides.get("profile_note") or "").strip(),
            "site_context": str(overrides.get("site_context") or "").strip(),
            "system_preset": system_preset,
            "system_prompt": self._default_system_prompt(system_preset, str(overrides.get("system_prompt") or "")),
            "temperature": max(0.0, min(float(overrides.get("temperature", generation_defaults["temperature"])), 2.0)),
            "top_p": max(0.0, min(float(overrides.get("top_p", generation_defaults["top_p"])), 1.0)),
            "top_k": int(overrides.get("top_k", generation_defaults["top_k"])),
            "repetition_penalty": float(overrides.get("repetition_penalty", generation_defaults["repetition_penalty"])),
            "max_new_tokens": clamped_max_new_tokens,
            "retrieval_top_k": int(overrides.get("retrieval_top_k", self.settings.retrieval_top_k)),
            "disable_retrieval": bool(overrides.get("disable_retrieval", self.settings.disable_retrieval)),
            "response_mode": str(overrides.get("response_mode", self.settings.response_mode or "assistant")).lower(),
            "web_search_enabled": bool(overrides.get("web_search_enabled", self.settings.web_search_enabled)),
            "web_search_max_results": int(overrides.get("web_search_max_results", self.settings.web_search_max_results)),
            "memory_items": list(overrides.get("memory_items") or []),
            "conversation_summary": dict(overrides.get("conversation_summary") or {}),
        }

    def _collect_grounding_context(self, query: str, resolved_settings: Dict[str, Any], mode: str = "general") -> List[Dict[str, str | float]]:
        if is_crisis_query(query):
            return []
        combined: List[tuple[float, Dict[str, str | float]]] = []
        local_items: List[Dict[str, str | float]] = []
        if self.knowledge_index is not None and not resolved_settings["disable_retrieval"]:
            for item in self.knowledge_index.retrieve(query, top_k=resolved_settings["retrieval_top_k"]):
                enriched = dict(item)
                enriched.setdefault("title", Path(str(item.get("source", ""))).stem.replace("_", " "))
                enriched.setdefault("url", str(item.get("source", "")))
                enriched.setdefault("trust_tier", 7.2)
                enriched.setdefault("source_type", "local")
                local_items.append(enriched)
                combined.append((float(item.get("score", 0.0)) + 0.4, enriched))
        anchor_terms = _anchor_terms(query)
        has_strong_local_match = any(_context_matches_anchor(item, anchor_terms) for item in local_items)
        should_use_web = resolved_settings["web_search_enabled"] and (
            _needs_live_research(query, mode, len(local_items))
            or ((mode in {"medical", "general"}) and bool(anchor_terms) and not has_strong_local_match)
        )
        if should_use_web and not has_strong_local_match:
            try:
                live_items = self.web_research.retrieve(
                    query,
                    max_results=max(2, int(resolved_settings["web_search_max_results"] or 2)),
                )
            except Exception:
                live_items = []
            for item in live_items:
                combined.append((float(item.get("score", 0.0)) + 0.9, item))

        combined.sort(key=lambda pair: pair[0], reverse=True)
        deduped: List[Dict[str, str | float]] = []
        seen_sources = set()
        for _, item in combined:
            source = str(item.get("source", "")).rstrip("/")
            if source in seen_sources:
                continue
            seen_sources.add(source)
            deduped.append(item)
            if len(deduped) == max(resolved_settings["retrieval_top_k"] + 1, resolved_settings["web_search_max_results"] + 1, 6):
                break
        body_targets = {term for term in anchor_terms if term in BODY_TARGET_TERMS}
        anchor_filtered = [item for item in deduped if _context_matches_anchor(item, anchor_terms)]
        if mode == "medical" and anchor_terms and len(anchor_terms) >= 2 and not anchor_filtered:
            relaxed_anchor_filtered = [
                item
                for item in deduped
                if _context_mentions_any(item, body_targets)
                and (
                    _anchor_overlap_ratio(item, anchor_terms) >= 0.5
                    or (
                        _anchor_overlap_ratio(item, anchor_terms) >= 0.34
                        and float(item.get("trust_tier", 0.0) or 0.0) >= 5.5
                    )
                )
            ]
            if relaxed_anchor_filtered:
                anchor_filtered = relaxed_anchor_filtered
            elif any(
                token in query.lower()
                for token in ("food", "drink", "candy", "soda", "snack", "supplement", "vitamin", "ingredient", "skittles")
            ):
                trusted_candidates = [
                    item
                    for item in deduped
                    if float(item.get("trust_tier", 0.0) or 0.0) >= 5.5
                    and _context_mentions_any(item, body_targets)
                ]
                if trusted_candidates:
                    anchor_filtered = trusted_candidates
                else:
                    return []
            else:
                return []
        candidate_pool = anchor_filtered if anchor_filtered else deduped
        if self.settings.semantic_retrieval_enabled:
            candidate_pool = self._semantic_rerank(query, candidate_pool)
        if anchor_terms:
            candidate_pool = sorted(
                candidate_pool,
                key=lambda item: (
                    0 if _exact_topic_match(item, anchor_terms) else 1,
                    -float(item.get("trust_tier", 0.0) or 0.0),
                    -float(item.get("score", 0.0) or 0.0),
                ),
            )
        return candidate_pool

    def _format_context_block(self, contexts: Sequence[Dict[str, str | float]]) -> str:
        blocks: List[str] = []
        for index, item in enumerate(contexts, start=1):
            title = str(item.get("title", "")).strip()
            source = str(item.get("source", "")).strip()
            text = " ".join(str(item.get("text", "")).split())
            if len(text) > 1200:
                text = text[:1200].rstrip() + "..."
            heading = title or source or f"Source {index}"
            blocks.append(f"[{index}] {heading}\n{text}")
        return "\n\n".join(blocks)

    def _build_ollama_messages(
        self,
        normalized_messages: Sequence[Dict[str, str]],
        resolved_settings: Dict[str, Any],
        contexts: Sequence[Dict[str, str | float]] | None = None,
    ) -> List[Dict[str, str]]:
        contexts = list(contexts or [])
        extra_systems = [
            message["content"].strip()
            for message in normalized_messages
            if message["role"] == "system" and message["content"].strip()
        ]
        conversation = [message for message in normalized_messages if message["role"] != "system"]
        memory_note, conversation = _compress_conversation(conversation)
        last_user_text = next(
            (message["content"].strip() for message in reversed(conversation) if message["role"] == "user" and message["content"].strip()),
            "",
        )
        mode = _resolve_mode(conversation, last_user_text, resolved_settings)
        memory_items = list(resolved_settings.get("memory_items") or [])
        conversation_summary = dict(resolved_settings.get("conversation_summary") or {})
        memory_snapshot = _memory_snapshot(memory_items, conversation_summary)
        system_parts = [resolved_settings["system_prompt"]]
        system_parts.extend(extra_systems)
        system_parts.append(
            _profile_instruction(
                resolved_settings["answer_style"],
                resolved_settings["reader_level"],
                tone_preference=str(resolved_settings.get("tone_preference", "")),
                primary_use=str(resolved_settings.get("primary_use", "")),
                display_name=str(resolved_settings.get("display_name", "")),
                profile_note=str(resolved_settings.get("profile_note", "")),
            )
        )
        system_parts.append(_mode_instruction(mode, last_user_text))
        system_parts.append(_plan_response_shape(mode, last_user_text, resolved_settings["answer_style"]))
        if resolved_settings.get("site_context"):
            system_parts.append(f"Current page or product context: {resolved_settings['site_context']}")
        if memory_note:
            system_parts.append(memory_note)
        if memory_snapshot:
            system_parts.append(memory_snapshot)
        if contexts:
            system_parts.append(_context_instruction(mode))
            system_parts.append(f"<context>\n{self._format_context_block(contexts)}\n</context>")
        else:
            system_parts.append(
                "Answer directly, clearly, and honestly. If you are not reasonably sure, say that you are unsure instead of inventing information."
            )
        extra_instruction = build_query_safety_instruction(last_user_text)
        if extra_instruction:
            system_parts.append(extra_instruction)
        system_message = "\n\n".join(part for part in system_parts if part and part.strip())
        ollama_messages = [{"role": "system", "content": system_message}]
        ollama_messages.extend(conversation)
        return ollama_messages

    def _grounded_response(
        self,
        normalized_messages: Sequence[Dict[str, str]],
        resolved_settings: Dict[str, Any],
        *,
        effective_query: str = "",
        contexts: Sequence[Dict[str, str | float]] | None = None,
        mode: str = "general",
        allow_model_fallback: bool = True,
    ) -> Dict[str, Any]:
        base_query = effective_query or _normalize_grounding_query(_resolved_grounding_query(normalized_messages))
        final_query = self.web_research.normalize_query(base_query) if base_query else ""
        last_user_text = next(
            (message["content"].strip() for message in reversed(normalized_messages) if message["role"] == "user" and message["content"].strip()),
            "",
        )
        if is_crisis_query(final_query):
            reply = _crisis_response(final_query)
            updated_messages = list(normalized_messages) + [{"role": "assistant", "content": reply}]
            prompt_tokens = _estimate_text_tokens(" ".join(message["content"] for message in normalized_messages))
            completion_tokens = _estimate_text_tokens(reply)
            return {
                "reply": reply,
                "messages": updated_messages,
                "history": updated_messages,
                "used_context": [],
                "extra_system_instruction": "",
                "request_config": resolved_settings,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "confidence": 0.99,
                "mode": "crisis",
                "effective_query": final_query,
                "model_id": self.model_id,
                "model_backend": "grounded",
            }
        if contexts is not None:
            final_contexts = list(contexts)
        elif asks_for_dosing(final_query) or has_red_flag_symptoms(final_query):
            final_contexts = []
        else:
            final_contexts = self._collect_grounding_context(final_query, resolved_settings, mode=mode)
        chat_backend = self._select_model_backend(resolved_settings.get("model_backend"))
        if (
            not final_contexts
            and allow_model_fallback
            and self._supports_chat_backend(chat_backend)
            and mode in {"medical", "general", "portfolio"}
            and not asks_for_dosing(final_query)
            and not has_red_flag_symptoms(final_query)
        ):
            synthesis_settings = dict(resolved_settings)
            synthesis_settings["temperature"] = min(float(synthesis_settings.get("temperature", 0.2) or 0.2), 0.18)
            synthesis_settings["top_p"] = min(float(synthesis_settings.get("top_p", 0.8) or 0.8), 0.75)
            synthesis_settings["top_k"] = min(int(synthesis_settings.get("top_k", 20) or 20), 18)
            synthesis_settings["max_new_tokens"] = min(int(synthesis_settings.get("max_new_tokens", 160) or 160), 140)
            return self._chat_backend_response(
                chat_backend,
                normalized_messages,
                synthesis_settings,
                contexts=[],
                effective_query=final_query,
                mode=mode,
                allow_grounded_fallback=False,
            )
        grounded = build_grounded_reply(
            final_query,
            final_contexts,
            answer_style=resolved_settings["answer_style"],
            reader_level=resolved_settings["reader_level"],
        )
        reply = str(grounded["reply"])
        if mode == "medical" and _medical_reply_needs_repair(final_query or last_user_text, reply):
            reply = _medical_last_resort_reply(final_query or last_user_text, final_contexts)
        if _needs_last_resort_reply(reply):
            if mode == "portfolio":
                reply = _portfolio_fallback(final_query or base_query, final_contexts)
            elif mode == "medical":
                reply = _medical_last_resort_reply(final_query or base_query, final_contexts)
            else:
                reply = _general_last_resort_reply(final_query or base_query, final_contexts)
        updated_messages = list(normalized_messages) + [{"role": "assistant", "content": reply}]
        prompt_tokens = _estimate_text_tokens(" ".join(message["content"] for message in normalized_messages))
        completion_tokens = _estimate_text_tokens(reply)
        return {
            "reply": reply,
            "messages": updated_messages,
            "history": updated_messages,
            "used_context": list(grounded.get("used_context", final_contexts)),
            "extra_system_instruction": "",
            "request_config": resolved_settings,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "confidence": float(grounded.get("confidence", 0.0)),
            "mode": mode if mode != "general" else str(grounded.get("mode", "grounded")),
            "effective_query": final_query,
            "model_id": self.model_id,
            "model_backend": "grounded",
        }

    def _ollama_response(
        self,
        normalized_messages: Sequence[Dict[str, str]],
        resolved_settings: Dict[str, Any],
        *,
        contexts: Sequence[Dict[str, str | float]] | None = None,
        effective_query: str = "",
        mode: str = "general",
        allow_grounded_fallback: bool = True,
    ) -> Dict[str, Any]:
        model_name = self._resolve_ollama_model(resolved_settings.get("model"))
        if not model_name:
            return self._grounded_response(
                normalized_messages,
                resolved_settings,
                mode=mode,
                allow_model_fallback=False,
            )

        last_user_text = next(
            (message["content"].strip() for message in reversed(normalized_messages) if message["role"] == "user" and message["content"].strip()),
            "",
        )
        ollama_messages = self._build_ollama_messages(normalized_messages, resolved_settings, contexts=contexts)
        try:
            ollama_result = self.ollama_client.chat(
                model=model_name,
                messages=ollama_messages,
                temperature=float(resolved_settings["temperature"]),
                top_p=float(resolved_settings["top_p"]),
                top_k=int(resolved_settings["top_k"]),
                repetition_penalty=float(resolved_settings["repetition_penalty"]),
                max_new_tokens=int(resolved_settings["max_new_tokens"]),
            )
        except Exception:
            if allow_grounded_fallback:
                return self._grounded_response(
                    normalized_messages,
                    resolved_settings,
                    effective_query=effective_query,
                    contexts=contexts,
                    mode=mode,
                    allow_model_fallback=False,
                )
            fallback_query = effective_query or last_user_text
            if mode == "portfolio":
                fallback_reply = _portfolio_fallback(fallback_query, contexts or [])
            else:
                fallback_reply = (
                    "I’m not getting strong source coverage for that exact phrasing, but the safest general answer is that this depends on the amount, frequency, and the person’s broader health context. "
                    "If you want, ask it a little more specifically and I’ll answer it directly."
                )
            updated_messages = list(normalized_messages) + [{"role": "assistant", "content": fallback_reply}]
            prompt_tokens = _estimate_text_tokens(" ".join(message["content"] for message in normalized_messages))
            completion_tokens = _estimate_text_tokens(fallback_reply)
            return {
                "reply": fallback_reply,
                "messages": updated_messages,
                "history": updated_messages,
                "used_context": list(contexts or []),
                "extra_system_instruction": "",
                "request_config": resolved_settings,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "confidence": 0.18,
                "mode": mode,
                "effective_query": fallback_query,
                "model_id": self.model_id,
                "model_backend": "fallback",
            }
        reply = ollama_result.reply.strip()
        if not reply and mode == "psychology":
            reply = _supportive_fallback(
                last_user_text,
                resolved_settings,
                resolved_settings.get("memory_items") or [],
                resolved_settings.get("conversation_summary") or {},
            )
        fallback_context = list(contexts or [])
        if fallback_context and _should_prefer_retrieval_fallback(last_user_text, reply, fallback_context):
            reply = _build_retrieval_fallback(last_user_text, fallback_context)
        if not reply and fallback_context:
            grounded = build_grounded_reply(
                effective_query or last_user_text,
                fallback_context,
                answer_style=resolved_settings["answer_style"],
                reader_level=resolved_settings["reader_level"],
            )
            reply = str(grounded["reply"])
        if not reply:
            reply = "I ran into a response problem and don't want to fake an answer. Please try again, rephrase it, or ask me to answer more directly."
        updated_messages = list(normalized_messages) + [{"role": "assistant", "content": reply}]
        prompt_tokens = int(ollama_result.prompt_tokens or _estimate_text_tokens(" ".join(message["content"] for message in ollama_messages)))
        completion_tokens = int(ollama_result.completion_tokens or _estimate_text_tokens(reply))
        return {
            "reply": reply,
            "messages": updated_messages,
            "history": updated_messages,
            "used_context": fallback_context,
            "extra_system_instruction": "",
            "request_config": resolved_settings,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "confidence": 1.0 if fallback_context else 0.6,
            "mode": mode if mode != "general" else ("assistant" if fallback_context else "model"),
            "effective_query": effective_query or last_user_text,
            "model_id": model_name,
            "model_backend": "ollama",
        }

    def _hosted_response(
        self,
        normalized_messages: Sequence[Dict[str, str]],
        resolved_settings: Dict[str, Any],
        *,
        contexts: Sequence[Dict[str, str | float]] | None = None,
        effective_query: str = "",
        mode: str = "general",
        allow_grounded_fallback: bool = True,
    ) -> Dict[str, Any]:
        model_name = self._resolve_hosted_model(resolved_settings.get("model"))
        if not model_name:
            return self._grounded_response(
                normalized_messages,
                resolved_settings,
                mode=mode,
                allow_model_fallback=False,
            )

        last_user_text = next(
            (message["content"].strip() for message in reversed(normalized_messages) if message["role"] == "user" and message["content"].strip()),
            "",
        )
        hosted_messages = self._build_ollama_messages(normalized_messages, resolved_settings, contexts=contexts)
        try:
            hosted_result = self.hosted_provider_client.chat(
                model=model_name,
                messages=hosted_messages,
                temperature=float(resolved_settings["temperature"]),
                top_p=float(resolved_settings["top_p"]),
                max_new_tokens=int(resolved_settings["max_new_tokens"]),
            )
        except Exception:
            if allow_grounded_fallback:
                return self._grounded_response(
                    normalized_messages,
                    resolved_settings,
                    effective_query=effective_query,
                    contexts=contexts,
                    mode=mode,
                    allow_model_fallback=False,
                )
            fallback_query = effective_query or last_user_text
            fallback_reply = _portfolio_fallback(fallback_query, contexts or []) if mode == "portfolio" else (
                "I’m not getting a stable upstream response right now. Ask again in a more specific way and I’ll try a tighter grounded answer."
            )
            updated_messages = list(normalized_messages) + [{"role": "assistant", "content": fallback_reply}]
            prompt_tokens = _estimate_text_tokens(" ".join(message["content"] for message in normalized_messages))
            completion_tokens = _estimate_text_tokens(fallback_reply)
            return {
                "reply": fallback_reply,
                "messages": updated_messages,
                "history": updated_messages,
                "used_context": list(contexts or []),
                "extra_system_instruction": "",
                "request_config": resolved_settings,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "confidence": 0.18,
                "mode": mode,
                "effective_query": fallback_query,
                "model_id": model_name,
                "model_backend": "fallback",
            }
        reply = hosted_result.reply.strip()
        if not reply and mode == "psychology":
            reply = _supportive_fallback(
                last_user_text,
                resolved_settings,
                resolved_settings.get("memory_items") or [],
                resolved_settings.get("conversation_summary") or {},
            )
        fallback_context = list(contexts or [])
        if fallback_context and _should_prefer_retrieval_fallback(last_user_text, reply, fallback_context):
            reply = _build_retrieval_fallback(last_user_text, fallback_context)
        if not reply and fallback_context:
            grounded = build_grounded_reply(
                effective_query or last_user_text,
                fallback_context,
                answer_style=resolved_settings["answer_style"],
                reader_level=resolved_settings["reader_level"],
            )
            reply = str(grounded["reply"])
        if not reply:
            reply = "I ran into a response problem and don't want to fake an answer. Please try again, rephrase it, or ask me to answer more directly."
        updated_messages = list(normalized_messages) + [{"role": "assistant", "content": reply}]
        prompt_tokens = int(hosted_result.prompt_tokens or _estimate_text_tokens(" ".join(message["content"] for message in hosted_messages)))
        completion_tokens = int(hosted_result.completion_tokens or _estimate_text_tokens(reply))
        return {
            "reply": reply,
            "messages": updated_messages,
            "history": updated_messages,
            "used_context": fallback_context,
            "extra_system_instruction": "",
            "request_config": resolved_settings,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "confidence": 1.0 if fallback_context else 0.6,
            "mode": mode if mode != "general" else ("assistant" if fallback_context else "model"),
            "effective_query": effective_query or last_user_text,
            "model_id": model_name,
            "model_backend": "hosted_api",
        }

    def _chat_backend_response(
        self,
        backend: str,
        normalized_messages: Sequence[Dict[str, str]],
        resolved_settings: Dict[str, Any],
        *,
        contexts: Sequence[Dict[str, str | float]] | None = None,
        effective_query: str = "",
        mode: str = "general",
        allow_grounded_fallback: bool = True,
    ) -> Dict[str, Any]:
        if backend == "hosted_api":
            return self._hosted_response(
                normalized_messages,
                resolved_settings,
                contexts=contexts,
                effective_query=effective_query,
                mode=mode,
                allow_grounded_fallback=allow_grounded_fallback,
            )
        return self._ollama_response(
            normalized_messages,
            resolved_settings,
            contexts=contexts,
            effective_query=effective_query,
            mode=mode,
            allow_grounded_fallback=allow_grounded_fallback,
        )

    def _assistant_response(
        self,
        normalized_messages: Sequence[Dict[str, str]],
        resolved_settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        backend = self._select_model_backend(resolved_settings.get("model_backend"))
        effective_query = self.web_research.normalize_query(_normalize_grounding_query(_resolved_grounding_query(normalized_messages)))
        last_user_text = next(
            (message["content"].strip() for message in reversed(normalized_messages) if message["role"] == "user" and message["content"].strip()),
            "",
        )
        mode = _resolve_mode(normalized_messages, last_user_text, resolved_settings)
        if mode == "general" and _is_smalltalk_query(last_user_text):
            reply = _smalltalk_response(last_user_text, resolved_settings)
            prompt_tokens = _estimate_text_tokens(" ".join(message["content"] for message in normalized_messages))
            completion_tokens = _estimate_text_tokens(reply)
            updated_messages = list(normalized_messages) + [{"role": "assistant", "content": reply}]
            return {
                "reply": reply,
                "messages": updated_messages,
                "history": updated_messages,
                "used_context": [],
                "extra_system_instruction": "",
                "request_config": resolved_settings,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "confidence": 0.94,
                "mode": "general",
                "effective_query": effective_query,
                "model_id": self.model_id,
                "model_backend": "grounded",
            }
        if mode == "crisis" or asks_for_dosing(last_user_text) or has_red_flag_symptoms(last_user_text):
            reply = _crisis_response(last_user_text) if is_crisis_query(last_user_text) else ""
            if mode == "crisis" and not reply:
                reply = _crisis_followup_response(last_user_text)
            if reply:
                prompt_tokens = _estimate_text_tokens(" ".join(message["content"] for message in normalized_messages))
                completion_tokens = _estimate_text_tokens(reply)
                updated_messages = list(normalized_messages) + [{"role": "assistant", "content": reply}]
                return {
                    "reply": reply,
                    "messages": updated_messages,
                    "history": updated_messages,
                    "used_context": [],
                    "extra_system_instruction": "",
                    "request_config": resolved_settings,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "confidence": 0.99,
                    "mode": "crisis",
                    "effective_query": effective_query,
                    "model_id": self.model_id,
                    "model_backend": "grounded",
                }
            return self._grounded_response(
                normalized_messages,
                resolved_settings,
                effective_query=effective_query,
                contexts=[],
                mode=mode,
            )
        contexts = self._collect_grounding_context(effective_query, resolved_settings, mode=mode)
        memory_items = list(resolved_settings.get("memory_items") or [])
        conversation_summary = dict(resolved_settings.get("conversation_summary") or {})
        if mode == "psychology":
            support_contexts = [
                item for item in contexts
                if str(item.get("domain", "")).strip().lower() == "psychology"
                or "psychology" in str(item.get("source", "")).lower()
            ][:2]
            if not self._supports_chat_backend(backend):
                reply = _supportive_fallback(last_user_text, resolved_settings, memory_items, conversation_summary)
                prompt_tokens = _estimate_text_tokens(" ".join(message["content"] for message in normalized_messages))
                completion_tokens = _estimate_text_tokens(reply)
                updated_messages = list(normalized_messages) + [{"role": "assistant", "content": reply}]
                return {
                    "reply": reply,
                    "messages": updated_messages,
                    "history": updated_messages,
                    "used_context": support_contexts,
                    "extra_system_instruction": "",
                    "request_config": resolved_settings,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "confidence": 0.72,
                    "mode": "psychology",
                    "effective_query": effective_query,
                    "model_id": self.model_id,
                    "model_backend": "grounded",
                }
            support_settings = dict(resolved_settings)
            support_settings["temperature"] = min(float(support_settings.get("temperature", 0.2) or 0.2), 0.45)
            support_settings["top_p"] = min(float(support_settings.get("top_p", 0.8) or 0.8), 0.9)
            support_settings["top_k"] = max(int(support_settings.get("top_k", 20) or 20), 24)
            support_settings["max_new_tokens"] = max(int(support_settings.get("max_new_tokens", 160) or 160), 180)
            result = self._chat_backend_response(
                backend,
                normalized_messages,
                support_settings,
                contexts=support_contexts,
                effective_query=effective_query,
                mode=mode,
            )
            if _psychology_reply_off_topic(last_user_text, str(result.get("reply", ""))):
                reply = _supportive_fallback(last_user_text, resolved_settings, memory_items, conversation_summary)
                prompt_tokens = _estimate_text_tokens(" ".join(message["content"] for message in normalized_messages))
                completion_tokens = _estimate_text_tokens(reply)
                updated_messages = list(normalized_messages) + [{"role": "assistant", "content": reply}]
                return {
                    "reply": reply,
                    "messages": updated_messages,
                    "history": updated_messages,
                    "used_context": support_contexts,
                    "extra_system_instruction": "",
                    "request_config": resolved_settings,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "confidence": 0.68,
                    "mode": "psychology",
                    "effective_query": effective_query,
                    "model_id": self.model_id,
                    "model_backend": "grounded",
                }
            return result
        if mode == "portfolio":
            used_context = _portfolio_sources(last_user_text, contexts)
            if self._supports_chat_backend(backend):
                result = self._chat_backend_response(
                    backend,
                    normalized_messages,
                    resolved_settings,
                    contexts=used_context or contexts[:2],
                    effective_query=effective_query,
                    mode=mode,
                )
                if not _portfolio_reply_off_topic(str(result.get("reply", ""))):
                    return result
            reply = _portfolio_fallback(last_user_text, contexts)
            prompt_tokens = _estimate_text_tokens(" ".join(message["content"] for message in normalized_messages))
            completion_tokens = _estimate_text_tokens(reply)
            updated_messages = list(normalized_messages) + [{"role": "assistant", "content": reply}]
            return {
                "reply": reply,
                "messages": updated_messages,
                "history": updated_messages,
                "used_context": used_context,
                "extra_system_instruction": "",
                "request_config": resolved_settings,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "confidence": 0.88,
                "mode": "portfolio",
                "effective_query": effective_query,
                "model_id": self.model_id,
                "model_backend": "grounded",
            }
        if mode == "medical":
            return self._grounded_response(
                normalized_messages,
                resolved_settings,
                effective_query=effective_query,
                contexts=contexts,
                mode=mode,
            )
        if mode == "general" and (_should_use_fast_grounded_answer(last_user_text, effective_query, contexts) or _needs_live_research(effective_query, mode, len(contexts))):
            return self._grounded_response(
                normalized_messages,
                resolved_settings,
                effective_query=effective_query,
                contexts=contexts,
                mode=mode,
            )
        if not self._supports_chat_backend(backend):
            return self._grounded_response(
                normalized_messages,
                resolved_settings,
                effective_query=effective_query,
                contexts=contexts,
                mode=mode,
            )
        return self._chat_backend_response(
            backend,
            normalized_messages,
            resolved_settings,
            contexts=contexts,
            effective_query=effective_query,
            mode=mode,
        )

    def _model_response(
        self,
        normalized_messages: Sequence[Dict[str, str]],
        resolved_settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        backend = self._select_model_backend(resolved_settings.get("model_backend"))
        last_user_text = next(
            (message["content"].strip() for message in reversed(normalized_messages) if message["role"] == "user" and message["content"].strip()),
            "",
        )
        mode = _resolve_mode(normalized_messages, last_user_text, resolved_settings)
        if self._supports_chat_backend(backend):
            return self._chat_backend_response(backend, normalized_messages, resolved_settings, mode=mode)
        runtime = self._ensure_runtime()
        result = complete_messages(
            runtime,
            normalized_messages,
            system_preset=resolved_settings["system_preset"],
            system_prompt=resolved_settings["system_prompt"],
            max_new_tokens=resolved_settings["max_new_tokens"],
            temperature=resolved_settings["temperature"],
            top_k=resolved_settings["top_k"],
            top_p=resolved_settings["top_p"],
            repetition_penalty=resolved_settings["repetition_penalty"],
            retrieval_top_k=resolved_settings["retrieval_top_k"],
            disable_retrieval=resolved_settings["disable_retrieval"],
        )
        result["model_id"] = self.checkpoint_model_id
        result["model_backend"] = "custom"
        result["mode"] = mode
        return result

    def health_payload(self) -> Dict[str, Any]:
        active_backend = self._select_model_backend(self.settings.model_backend)
        available_models = []
        if self._hosted_provider_available():
            available_models.append(self._resolve_hosted_model(self.settings.provider_model))
        available_models.extend(self._get_ollama_models())
        return {
            "status": "ok",
            "product_name": self.settings.product_name,
            "model_id": self.model_id,
            "model_backend": active_backend,
            "checkpoint_path": str(resolve_checkpoint_path(self.settings.checkpoint)),
            "device": str(self.settings.device),
            "model_loaded": self.runtime is not None or self._supports_chat_backend(active_backend),
            "hosted_provider_configured": self._hosted_provider_available(),
            "ollama_available": self._ollama_available(),
            "available_models": [model_name for model_name in available_models if model_name],
            "knowledge_index_loaded": self.knowledge_index is not None,
            "web_search_enabled": self.settings.web_search_enabled,
            "local_only": not self.settings.web_search_enabled,
            "response_mode": self.settings.response_mode,
            "semantic_retrieval_enabled": self.settings.semantic_retrieval_enabled,
            "api_key_self_serve_enabled": self.settings.api_key_self_serve_enabled,
            "generated_key_rate_limits": {
                "minute": self.settings.generated_key_rate_limit_minute,
                "hour": self.settings.generated_key_rate_limit_hour,
                "day": self.settings.generated_key_rate_limit_day,
            },
            "max_completion_tokens": self.settings.max_completion_tokens,
        }

    def app_config_payload(self) -> Dict[str, Any]:
        defaults = get_generation_defaults(self.settings.system_preset or DEFAULT_SYSTEM_PRESET)
        active_backend = self._select_model_backend(self.settings.model_backend)
        available_models = []
        if self._hosted_provider_available():
            available_models.append(self._resolve_hosted_model(self.settings.provider_model))
        available_models.extend(self._get_ollama_models())
        return {
            "product_name": self.settings.product_name,
            "model_id": self.model_id,
            "active_model": self.model_id,
            "model_backend": active_backend,
            "checkpoint_path": str(resolve_checkpoint_path(self.settings.checkpoint)),
            "device": self.settings.device,
            "default_system_preset": self.settings.system_preset or DEFAULT_SYSTEM_PRESET,
            "default_model_backend": self.settings.model_backend or "auto",
            "available_model_backends": ["auto", "hosted_api", "ollama", "custom"],
            "available_models": [model_name for model_name in available_models if model_name],
            "hosted_provider_configured": self._hosted_provider_available(),
            "ollama_available": self._ollama_available(),
            "system_presets": sorted(SYSTEM_PROMPTS),
            "default_generation": {
                "model_backend": self.settings.model_backend,
                "model": self._default_chat_model(self.settings.model_backend),
                "answer_style": "balanced",
                "reader_level": "everyday",
                "tone_preference": "calm",
                "primary_use": "balanced",
                "display_name": "",
                "profile_note": "",
                "site_context": "",
                "temperature": float(defaults["temperature"]),
                "top_p": float(defaults["top_p"]),
                "top_k": int(defaults["top_k"]),
                "repetition_penalty": float(defaults["repetition_penalty"]),
                "max_new_tokens": int(defaults["max_new_tokens"]),
                "retrieval_top_k": self.settings.retrieval_top_k,
                "disable_retrieval": self.settings.disable_retrieval,
                "system_prompt": self._default_system_prompt(self.settings.system_preset),
                "response_mode": self.settings.response_mode,
                "web_search_enabled": self.settings.web_search_enabled,
                "web_search_max_results": self.settings.web_search_max_results,
            },
            "knowledge_index_loaded": self.knowledge_index is not None,
            "web_search_enabled": self.settings.web_search_enabled,
            "local_only": not self.settings.web_search_enabled,
            "model_loaded": self.runtime is not None or self._supports_chat_backend(active_backend),
            "memory_enabled": True,
            "semantic_retrieval_enabled": self.settings.semantic_retrieval_enabled,
            "api_key_self_serve_enabled": self.settings.api_key_self_serve_enabled,
            "generated_key_rate_limits": {
                "minute": self.settings.generated_key_rate_limit_minute,
                "hour": self.settings.generated_key_rate_limit_hour,
                "day": self.settings.generated_key_rate_limit_day,
            },
            "max_completion_tokens": self.settings.max_completion_tokens,
        }

    def list_models_payload(self) -> Dict[str, Any]:
        data = []
        if self._hosted_provider_available():
            data.append(
                {
                    "id": self._resolve_hosted_model(self.settings.provider_model),
                    "object": "model",
                    "owned_by": "hosted-provider",
                }
            )
        for model_name in self._get_ollama_models(refresh=True):
            data.append(
                {
                    "id": model_name,
                    "object": "model",
                    "owned_by": "ollama",
                }
            )
        data.append(
            {
                "id": self.checkpoint_model_id,
                "object": "model",
                "owned_by": "local-checkpoint",
            }
        )
        return {
            "object": "list",
            "data": data,
        }

    def generate_from_messages(
        self,
        messages: Sequence[Dict[str, Any]],
        settings_overrides: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        normalized_messages = _normalize_messages(messages)
        if not normalized_messages:
            raise ValueError("No valid chat messages were provided.")

        resolved_settings = self._resolve_settings(settings_overrides)
        if resolved_settings["response_mode"] == "model":
            result = self._model_response(normalized_messages, resolved_settings)
        elif resolved_settings["response_mode"] == "assistant":
            result = self._assistant_response(normalized_messages, resolved_settings)
        else:
            result = self._grounded_response(normalized_messages, resolved_settings)
        if not str(result.get("reply", "")).strip():
            fallback = "I hit a response problem and do not want to pretend otherwise. Please ask again and I will try a simpler direct answer."
            result = dict(result)
            result["reply"] = fallback
            result["completion_tokens"] = _estimate_text_tokens(fallback)
            result["total_tokens"] = int(result.get("prompt_tokens", 0)) + int(result["completion_tokens"])
        return result

    def resolve_conversation_settings(
        self,
        conversation: Dict[str, Any],
        incoming_settings: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        base_settings = dict(conversation.get("settings") or {})
        if incoming_settings:
            base_settings.update({key: value for key, value in incoming_settings.items() if value is not None})
        if not base_settings.get("system_preset"):
            base_settings["system_preset"] = conversation.get("system_preset") or self.settings.system_preset
        if conversation.get("system_prompt") and not base_settings.get("system_prompt"):
            base_settings["system_prompt"] = conversation["system_prompt"]
        if not base_settings.get("model_backend"):
            base_settings["model_backend"] = self.settings.model_backend
        if not base_settings.get("model"):
            base_settings["model"] = self._default_chat_model(base_settings.get("model_backend"))
        if not base_settings.get("answer_style"):
            base_settings["answer_style"] = "balanced"
        if not base_settings.get("reader_level"):
            base_settings["reader_level"] = "everyday"
        if not base_settings.get("tone_preference"):
            base_settings["tone_preference"] = "calm"
        if not base_settings.get("primary_use"):
            base_settings["primary_use"] = "balanced"
        if "display_name" not in base_settings:
            base_settings["display_name"] = ""
        if "profile_note" not in base_settings:
            base_settings["profile_note"] = ""
        if "site_context" not in base_settings:
            base_settings["site_context"] = ""
        if not base_settings.get("response_mode"):
            base_settings["response_mode"] = self.settings.response_mode
        if base_settings.get("web_search_enabled") is False and not int(base_settings.get("web_search_max_results") or 0):
            base_settings["web_search_enabled"] = self.settings.web_search_enabled
        if "web_search_enabled" not in base_settings:
            base_settings["web_search_enabled"] = self.settings.web_search_enabled
        if not base_settings.get("web_search_max_results"):
            base_settings["web_search_max_results"] = self.settings.web_search_max_results
        if "memory_items" not in base_settings:
            base_settings["memory_items"] = list(conversation.get("memory_items") or [])
        if "conversation_summary" not in base_settings:
            base_settings["conversation_summary"] = dict(conversation.get("summary_state") or {})
        return base_settings

    def generate_for_conversation(
        self,
        conversation: Dict[str, Any],
        content: str,
        incoming_settings: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        history_messages = [
            {"role": message["role"], "content": message["content"]}
            for message in conversation.get("messages", [])
        ]
        resolved_settings = self.resolve_conversation_settings(conversation, incoming_settings)
        if history_messages and history_messages[-1]["role"] == "user" and history_messages[-1]["content"].strip() == content.strip():
            return self.generate_from_messages(history_messages, resolved_settings)
        return self.generate_from_messages(
            history_messages + [{"role": "user", "content": content}],
            resolved_settings,
        )

    def generate_timeout_fallback_for_messages(
        self,
        messages: Sequence[Dict[str, Any]],
        settings_overrides: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        fallback_settings = dict(settings_overrides or {})
        fallback_settings["response_mode"] = "grounded"
        fallback_settings["model_backend"] = "custom"
        fallback_settings["disable_retrieval"] = False
        fallback_settings["web_search_enabled"] = False
        fallback_settings["web_search_max_results"] = 0
        fallback_settings["retrieval_top_k"] = max(4, int(fallback_settings.get("retrieval_top_k", 4) or 4))
        return self.generate_last_resort_fallback_for_messages(messages, fallback_settings)

    def generate_timeout_fallback(
        self,
        conversation: Dict[str, Any],
        content: str,
        incoming_settings: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        history_messages = [
            {"role": message["role"], "content": message["content"]}
            for message in conversation.get("messages", [])
        ]
        resolved_settings = self.resolve_conversation_settings(conversation, incoming_settings)
        if history_messages and history_messages[-1]["role"] == "user" and history_messages[-1]["content"].strip() == content.strip():
            return self.generate_timeout_fallback_for_messages(history_messages, resolved_settings)
        return self.generate_timeout_fallback_for_messages(
            history_messages + [{"role": "user", "content": content}],
            resolved_settings,
        )

    def generate_last_resort_fallback_for_messages(
        self,
        messages: Sequence[Dict[str, Any]],
        settings_overrides: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        normalized_messages = _normalize_messages(messages)
        if not normalized_messages:
            reply = "I need a user message before I can answer."
            completion_tokens = _estimate_text_tokens(reply)
            return {
                "reply": reply,
                "messages": [{"role": "assistant", "content": reply}],
                "history": [{"role": "assistant", "content": reply}],
                "used_context": [],
                "extra_system_instruction": "",
                "request_config": dict(settings_overrides or {}),
                "prompt_tokens": 0,
                "completion_tokens": completion_tokens,
                "total_tokens": completion_tokens,
                "confidence": 0.2,
                "mode": "fallback",
                "effective_query": "",
                "model_id": self.model_id,
                "model_backend": "fallback",
            }
        resolved_settings = self._resolve_settings(settings_overrides)
        rescue_settings = dict(resolved_settings)
        rescue_settings["response_mode"] = "grounded"
        rescue_settings["model_backend"] = "custom"
        rescue_settings["disable_retrieval"] = False
        rescue_settings["web_search_enabled"] = False
        rescue_settings["web_search_max_results"] = 0
        rescue_settings["retrieval_top_k"] = max(4, int(rescue_settings.get("retrieval_top_k", 4) or 4))
        latest_user_text = next(
            (message["content"].strip() for message in reversed(normalized_messages) if message["role"] == "user" and message["content"].strip()),
            "",
        )
        effective_query = self.web_research.normalize_query(_normalize_grounding_query(_resolved_grounding_query(normalized_messages)))
        mode = _resolve_mode(normalized_messages, latest_user_text, rescue_settings)
        contexts: List[Dict[str, str | float]] = []
        if mode in {"medical", "general", "portfolio"}:
            contexts = self._collect_grounding_context(effective_query, rescue_settings, mode=mode)
        if mode == "crisis":
            reply = _crisis_response(latest_user_text) or _crisis_followup_response(latest_user_text)
            used_context = []
            confidence = 0.99
        elif mode == "psychology":
            reply = _supportive_fallback(
                latest_user_text,
                rescue_settings,
                rescue_settings.get("memory_items") or [],
                rescue_settings.get("conversation_summary") or {},
            )
            used_context = contexts
            confidence = 0.72
        elif mode == "portfolio":
            reply = _portfolio_fallback(latest_user_text or effective_query, contexts)
            used_context = _portfolio_sources(latest_user_text or effective_query, contexts)
            confidence = 0.84 if used_context else 0.66
        elif mode == "medical":
            grounded = build_grounded_reply(
                effective_query or latest_user_text,
                contexts,
                answer_style=rescue_settings["answer_style"],
                reader_level=rescue_settings["reader_level"],
            )
            reply = str(grounded.get("reply", "")).strip()
            if _needs_last_resort_reply(reply):
                reply = _medical_last_resort_reply(latest_user_text or effective_query, contexts)
            used_context = list(grounded.get("used_context", contexts))
            confidence = float(grounded.get("confidence", 0.0)) if used_context else 0.45
        else:
            grounded = build_grounded_reply(
                effective_query or latest_user_text,
                contexts,
                answer_style=rescue_settings["answer_style"],
                reader_level=rescue_settings["reader_level"],
            )
            reply = str(grounded.get("reply", "")).strip()
            if _needs_last_resort_reply(reply):
                reply = _general_last_resort_reply(latest_user_text or effective_query, contexts)
            used_context = list(grounded.get("used_context", contexts))
            confidence = float(grounded.get("confidence", 0.0)) if used_context else 0.35
        if not str(reply).strip():
            reply = _general_last_resort_reply(latest_user_text or effective_query, contexts)
            used_context = list(contexts[:2])
            confidence = 0.25
        updated_messages = list(normalized_messages) + [{"role": "assistant", "content": reply}]
        prompt_tokens = _estimate_text_tokens(" ".join(message["content"] for message in normalized_messages))
        completion_tokens = _estimate_text_tokens(reply)
        return {
            "reply": reply,
            "messages": updated_messages,
            "history": updated_messages,
            "used_context": used_context,
            "extra_system_instruction": "",
            "request_config": rescue_settings,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "confidence": confidence,
            "mode": mode,
            "effective_query": effective_query or latest_user_text,
            "model_id": self.model_id,
            "model_backend": "fallback",
        }

    def generate_last_resort_fallback(
        self,
        conversation: Dict[str, Any],
        content: str,
        incoming_settings: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        history_messages = [
            {"role": message["role"], "content": message["content"]}
            for message in conversation.get("messages", [])
        ]
        resolved_settings = self.resolve_conversation_settings(conversation, incoming_settings)
        if history_messages and history_messages[-1]["role"] == "user" and history_messages[-1]["content"].strip() == content.strip():
            return self.generate_last_resort_fallback_for_messages(history_messages, resolved_settings)
        return self.generate_last_resort_fallback_for_messages(
            history_messages + [{"role": "user", "content": content}],
            resolved_settings,
        )

    def build_personalization_state(
        self,
        conversation: Dict[str, Any],
        latest_user_text: str,
        incoming_settings: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        history_messages = [
            {"role": message["role"], "content": message["content"]}
            for message in conversation.get("messages", [])
            if str(message.get("content", "")).strip()
        ]
        resolved_settings = self.resolve_conversation_settings(conversation, incoming_settings)
        mode = _resolve_mode(history_messages, latest_user_text, resolved_settings)
        return {
            "memory_items": _extract_memory_items(history_messages, latest_user_text, resolved_settings),
            "conversation_summary": _build_conversation_state(history_messages, mode, latest_user_text),
            "mode": mode,
        }

    def build_openai_response(
        self,
        result: Dict[str, Any],
        *,
        completion_id: str,
        created: int,
    ) -> Dict[str, Any]:
        model_name = str(result.get("model_id") or self.model_id)
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result["reply"],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": int(result.get("prompt_tokens", 0)),
                "completion_tokens": int(result.get("completion_tokens", 0)),
                "total_tokens": int(result.get("total_tokens", 0)),
            },
            "system_fingerprint": "local-grounded-runtime",
        }

    def openai_stream_chunks(
        self,
        result: Dict[str, Any],
        *,
        completion_id: str,
        created: int,
    ) -> Iterable[Dict[str, Any]]:
        model_name = str(result.get("model_id") or self.model_id)
        yield {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        reply = str(result["reply"])
        chunk = ""
        for word in reply.split():
            candidate = word if not chunk else f"{chunk} {word}"
            if len(candidate) <= 28:
                chunk = candidate
                continue
            yield {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {"content": chunk + " "}, "finish_reason": None}],
            }
            chunk = word
        if chunk:
            yield {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
            }
        yield {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }

    def default_conversation_payload(self, title: str | None = None) -> Dict[str, Any]:
        defaults = get_generation_defaults(self.settings.system_preset or DEFAULT_SYSTEM_PRESET)
        return {
            "title": title or "New conversation",
            "system_preset": self.settings.system_preset or DEFAULT_SYSTEM_PRESET,
            "system_prompt": self._default_system_prompt(self.settings.system_preset),
            "settings": {
                "model_backend": self.settings.model_backend,
                "model": self._default_chat_model(self.settings.model_backend),
                "answer_style": "balanced",
                "reader_level": "everyday",
                "tone_preference": "calm",
                "primary_use": "balanced",
                "display_name": "",
                "profile_note": "",
                "site_context": "",
                "temperature": float(defaults["temperature"]),
                "top_p": float(defaults["top_p"]),
                "top_k": int(defaults["top_k"]),
                "repetition_penalty": float(defaults["repetition_penalty"]),
                "max_new_tokens": int(defaults["max_new_tokens"]),
                "retrieval_top_k": self.settings.retrieval_top_k,
                "disable_retrieval": self.settings.disable_retrieval,
                "system_preset": self.settings.system_preset,
                "system_prompt": self._default_system_prompt(self.settings.system_preset),
                "response_mode": self.settings.response_mode,
                "web_search_enabled": self.settings.web_search_enabled,
                "web_search_max_results": self.settings.web_search_max_results,
            },
        }

    def title_for_first_message(self, content: str) -> str:
        return _auto_title_from_text(content)
