from __future__ import annotations

import re
from typing import Dict, List, Sequence
from urllib.parse import urlparse

from presets import asks_for_diagnosis, asks_for_dosing, has_red_flag_symptoms, is_medical_query
from retrieval import STOPWORDS, tokenize_retrieval_text


SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")
MEDICAL_REDIRECT = (
    "I can't provide authoritative medication dosing. "
    "Please use a pharmacist, clinician, or the official medication label for exact dosing."
)
QUERY_ASPECT_HINTS = {
    "symptoms": {"symptom", "symptoms", "sign", "signs", "pain", "swelling", "fever", "urine", "urination", "rash", "cough", "runny", "nose", "eyes", "spots"},
    "causes": {"cause", "causes", "because", "trigger", "triggered", "infection", "autoimmune"},
    "treatment": {"treat", "treatment", "therapy", "medication", "manage", "management"},
    "diagnosis": {"diagnosis", "diagnose", "test", "testing", "biopsy", "urine", "blood"},
    "prevention": {"prevent", "prevention", "avoid", "reduce"},
    "complications": {"complication", "complications", "failure", "damage", "serious"},
    "prognosis": {"outlook", "prognosis", "recover", "recovery", "chronic"},
    "types": {"type", "types", "form", "forms", "kind", "kinds"},
}
GROUNDING_NOISE_TERMS = {
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


def extract_focus_phrase(query: str) -> str:
    lowered = query.strip().lower()
    patterns = (
        r"^(?:what is|what are|define|explain|tell me about)\s+(.+?)(?:\?|$)",
        r"^(?:how does|how do|why does|why do)\s+(.+?)(?:\?|$)",
    )
    for pattern in patterns:
        match = re.match(pattern, lowered)
        if match:
            focus = match.group(1).strip(" .?")
            focus = re.sub(
                r"\b(work|works|working|happen|happens|happening|occur|occurs|occurring|function|functions|functioning|operate|operates|operating)\b$",
                "",
                focus,
            ).strip(" .?")
            return focus
    return re.sub(
        r"\b(work|works|working|happen|happens|happening|occur|occurs|occurring|function|functions|functioning|operate|operates|operating)\b$",
        "",
        lowered.strip(" .?"),
    ).strip(" .?")


def clean_sentence(text: str) -> str:
    cleaned = re.sub(r"\[[^\]]+\]", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -")
    if not cleaned:
        return ""
    if ":" in cleaned and len(cleaned.rsplit(":", 1)[-1].split()) < 3:
        return ""
    if cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned[0].upper() + cleaned[1:]


def split_sentences(text: str) -> List[str]:
    sentences = []
    for sentence in SENTENCE_PATTERN.split(text):
        cleaned = clean_sentence(sentence)
        if len(cleaned) < 35:
            continue
        if sum(character.isalpha() for character in cleaned) < 20:
            continue
        sentences.append(cleaned)
    return sentences


def _content_terms(text: str) -> set[str]:
    return {
        term
        for term in tokenize_retrieval_text(text)
        if term not in STOPWORDS and len(term) > 2
    }


def _source_host(source: str) -> str:
    host = urlparse(source).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host or source


def _dedupe_sources(sources: Sequence[Dict[str, str | float]]) -> List[Dict[str, str | float]]:
    deduped: List[Dict[str, str | float]] = []
    seen = set()
    for source in sources:
        key = str(source.get("source", "")).rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(dict(source))
    return deduped


def _sentence_score(
    query: str,
    focus: str,
    sentence: str,
    source: Dict[str, str | float],
    index: int,
) -> float:
    query_terms = _content_terms(query)
    sentence_terms = _content_terms(sentence)
    score = float(len(query_terms.intersection(sentence_terms)))
    lowered_sentence = sentence.lower()
    lowered_focus = focus.lower()
    if lowered_focus and lowered_focus in lowered_sentence:
        score += 4.0
    if lowered_focus and any(pattern in lowered_sentence for pattern in (f"{lowered_focus} is", f"{lowered_focus} means", f"{lowered_focus} refers to")):
        score += 7.0
    if index == 0:
        score += 0.8
    if source.get("domain") == "medical":
        score += 0.6
    trust_tier = float(source.get("trust_tier", 0.0) or 0.0)
    if trust_tier > 0:
        score += min(2.8, trust_tier / 3.0)
    web_domain = str(source.get("web_domain", ""))
    if web_domain.endswith(("medlineplus.gov", "mayoclinic.org", "cdc.gov", "nih.gov", "nhs.uk", "who.int", "merckmanuals.com", "msdmanuals.com", "clevelandclinic.org")):
        score += 2.2
    if web_domain == "wikipedia.org":
        score += 1.4
    if 55 <= len(sentence) <= 240:
        score += 0.5
    if lowered_sentence.startswith("this article ") or lowered_sentence.startswith("in this article ") or "learn about" in lowered_sentence:
        score -= 3.0
    lowered_query = query.lower()
    for label, hints in QUERY_ASPECT_HINTS.items():
        if label not in lowered_query:
            continue
        matched_hints = {hint for hint in hints if hint in lowered_sentence}
        if matched_hints:
            score += 4.0 + 1.5 * float(len(matched_hints))
            if "," in sentence and len(matched_hints) >= 2:
                score += 1.5
        else:
            score -= 3.0
    return score


def _supports_urgent_guidance(query: str) -> bool:
    lowered = query.lower()
    return any(
        token in lowered
        for token in ("serious", "emergency", "urgent", "doctor", "care", "hospital", "trouble breathing", "chest pain", "fainting", "confusion")
    )


def build_grounded_reply(
    query: str,
    contexts: Sequence[Dict[str, str | float]],
    answer_style: str = "balanced",
    reader_level: str = "everyday",
) -> Dict[str, object]:
    if asks_for_dosing(query):
        return {
            "reply": MEDICAL_REDIRECT,
            "used_context": list(contexts),
            "confidence": 0.98,
            "mode": "grounded",
        }

    if not contexts:
        focus = extract_focus_phrase(query) or query.strip()
        fallback = (
            f"I cannot verify enough source material to answer {focus} precisely right now, so the safest direct answer is that it depends on the exact context, amount, and overall health situation."
            if focus
            else "I cannot verify enough source material to answer that precisely right now, so the safest direct answer is to keep the answer general rather than guess."
        )
        if has_red_flag_symptoms(query):
            fallback = "This could be urgent. Seek emergency care or immediate clinician evaluation right away."
        return {
            "reply": fallback,
            "used_context": [],
            "confidence": 0.0,
            "mode": "grounded",
        }

    focus = extract_focus_phrase(query)
    aspect_terms = {hint for hints in QUERY_ASPECT_HINTS.values() for hint in hints}
    focus_terms = {
        term
        for term in _content_terms(focus or query)
        if term not in GROUNDING_NOISE_TERMS and term not in aspect_terms and term not in QUERY_ASPECT_HINTS
    }
    preferred_contexts = [
        source
        for source in contexts
        if focus_terms
        and focus_terms.issubset(
            _content_terms(" ".join(str(source.get(key, "")) for key in ("title", "source", "url", "web_domain")))
        )
    ]
    working_contexts = list(preferred_contexts or contexts)
    candidates: List[tuple[float, str, Dict[str, str | float]]] = []
    for source in working_contexts:
        for index, sentence in enumerate(split_sentences(str(source.get("text", "")))):
            score = _sentence_score(query, focus, sentence, source, index)
            candidates.append((score, sentence, source))

    candidates.sort(key=lambda item: item[0], reverse=True)
    active_aspects = [label for label in QUERY_ASPECT_HINTS if label in query.lower()]
    if active_aspects:
        prioritized_candidates = [
            item
            for item in candidates
            if any(hint in item[1].lower() for label in active_aspects for hint in QUERY_ASPECT_HINTS[label])
        ]
        if prioritized_candidates:
            remaining_candidates = [item for item in candidates if item not in prioritized_candidates]
            candidates = prioritized_candidates + remaining_candidates
    chosen_sentences: List[str] = []
    chosen_sources: List[Dict[str, str | float]] = []
    seen_terms: List[set[str]] = []
    for score, sentence, source in candidates:
        if score <= 0:
            continue
        sentence_terms = _content_terms(sentence)
        if any(len(sentence_terms.intersection(existing)) >= max(3, len(sentence_terms) // 2) for existing in seen_terms):
            continue
        chosen_sentences.append(sentence)
        chosen_sources.append(source)
        seen_terms.append(sentence_terms)
        if len(chosen_sentences) == 2:
            break

    if not chosen_sentences:
        chosen_sentences = split_sentences(str(working_contexts[0].get("text", "")))[:2]
        chosen_sources = [working_contexts[0]]

    if active_aspects:
        aspect_filtered = [
            sentence
            for sentence in chosen_sentences
            if any(hint in sentence.lower() for label in active_aspects for hint in QUERY_ASPECT_HINTS[label])
        ]
        if aspect_filtered:
            chosen_sentences = aspect_filtered[:1]

    if answer_style == "concise":
        chosen_sentences = chosen_sentences[:1]
    elif answer_style == "detailed":
        chosen_sentences = chosen_sentences[:3]
    else:
        chosen_sentences = chosen_sentences[:2]

    reply = " ".join(chosen_sentences).strip()
    if asks_for_diagnosis(query):
        diagnosis_line = "I can’t tell from chat whether you have this. A clinician would need your history, symptoms, and exam to evaluate it properly."
        reply = f"{reply} {diagnosis_line}".strip()
    if has_red_flag_symptoms(query):
        reply = "This could be urgent. Seek emergency care or immediate clinician evaluation. " + reply
    elif is_medical_query(query) and _supports_urgent_guidance(query):
        reply = f"{reply} Seek urgent medical care if symptoms are severe, rapidly worsening, or involve trouble breathing, chest pain, fainting, or confusion."
    elif reader_level == "everyday" and "glomeruli" in reply.lower():
        reply = reply.replace("glomeruli", "kidney filters (glomeruli)")

    confidence = min(0.99, 0.45 + 0.12 * len(chosen_sentences))
    deduped_sources = _dedupe_sources(chosen_sources or list(working_contexts))
    return {
        "reply": reply,
        "used_context": deduped_sources,
        "confidence": confidence,
        "mode": "grounded",
        "source_hosts": sorted({_source_host(str(item.get("source", ""))) for item in deduped_sources}),
    }
