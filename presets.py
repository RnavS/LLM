from __future__ import annotations

import re
from typing import Dict


DEFAULT_SYSTEM_PRESET = "medbrief-medical"


SYSTEM_PROMPTS: Dict[str, str] = {
    "default": (
        "You are a helpful assistant running fully offline on the user's machine. "
        "Answer clearly and honestly."
    ),
    "medbrief-medical": (
        "You are MedBrief AI, a premium assistant for emotionally intelligent support, careful healthcare information, polished project explanation, and strong general help. "
        "Be direct, coherent, natural, and useful. Adapt to the user's intent without announcing modes. "
        "For support or advice, acknowledge the situation briefly, then give thoughtful, concrete guidance that feels personal rather than generic. "
        "For medical questions, ground the answer in retrieved sources, explain clearly, stay non-diagnostic and non-prescribing, and be explicit about uncertainty when needed. "
        "For project, product, or portfolio questions, sound articulate, credible, and polished. "
        "For general questions, answer like a strong modern assistant: direct answer first, short explanation second, then the most useful detail. "
        "Do not hallucinate facts, do not invent memories or credentials, do not overstate certainty, and do not sound robotic, canned, or filler-heavy. "
        "If something is urgent or unsafe, prioritize safety immediately."
    ),
    "factual-medical-lite": (
        "You are a precise, intellectually serious local assistant. "
        "Prefer factual, educational answers over creative improvisation. "
        "Answer directly first, then add a short explanation. "
        "If the supplied context is strong, ground your answer in it. "
        "If the context is weak or absent, say what you are uncertain about instead of pretending to know. "
        "For medical questions, provide educational information only, not diagnosis, dosing, or treatment authority. "
        "Mention urgent or emergency care when the user's symptoms sound high-risk."
    ),
}


PRESET_GENERATION_DEFAULTS: Dict[str, Dict[str, float | int]] = {
    "default": {
        "temperature": 0.9,
        "top_k": 50,
        "top_p": 0.95,
        "repetition_penalty": 1.1,
        "max_new_tokens": 160,
    },
    "medbrief-medical": {
        "temperature": 0.2,
        "top_k": 20,
        "top_p": 0.8,
        "repetition_penalty": 1.03,
        "max_new_tokens": 160,
    },
    "factual-medical-lite": {
        "temperature": 0.35,
        "top_k": 30,
        "top_p": 0.85,
        "repetition_penalty": 1.05,
        "max_new_tokens": 96,
    },
}


PRESET_SAMPLE_PROMPTS: Dict[str, str] = {
    "default": "Introduce yourself in one short paragraph.",
    "medbrief-medical": "What is hypertension?",
    "factual-medical-lite": "What is hypertension?",
}


MODEL_SIZE_PRESETS: Dict[str, Dict[str, int | float]] = {
    "small": {
        "block_size": 256,
        "n_embd": 256,
        "n_head": 4,
        "n_layer": 4,
        "dropout": 0.1,
    },
    "base": {
        "block_size": 384,
        "n_embd": 384,
        "n_head": 6,
        "n_layer": 6,
        "dropout": 0.1,
    },
    "large": {
        "block_size": 512,
        "n_embd": 512,
        "n_head": 8,
        "n_layer": 8,
        "dropout": 0.1,
    },
    "xlarge": {
        "block_size": 768,
        "n_embd": 768,
        "n_head": 12,
        "n_layer": 12,
        "dropout": 0.1,
    },
}


MEDICAL_KEYWORDS = [
    "autoimmune",
    "doctor",
    "medical",
    "medicine",
    "health",
    "kidney",
    "kidneys",
    "renal",
    "urine",
    "urinary",
    "liver",
    "intestine",
    "intestines",
    "intestinal",
    "gut",
    "bowel",
    "bowels",
    "colon",
    "digestive",
    "digestion",
    "stomach",
    "glucose",
    "blood sugar",
    "nutrition",
    "diet",
    "food",
    "drink",
    "candy",
    "sugar",
    "soda",
    "supplement",
    "vitamin",
    "ingredient",
    "side effect",
    "side effects",
    "long term",
    "symptom",
    "diagnosis",
    "dose",
    "dosage",
    "blood pressure",
    "heart rate",
    "pain",
    "infection",
    "fever",
    "cough",
    "nausea",
    "lupus",
    "stroke",
    "chest pain",
    "shortness of breath",
    "hypertension",
    "diabetes",
    "asthma",
    "pregnancy",
]


PSYCHOLOGY_KEYWORDS = [
    "anxious",
    "anxiety",
    "avoidance",
    "burnout",
    "counsel",
    "counseling",
    "counsellor",
    "counselor",
    "depressed",
    "depression",
    "empty",
    "emotion",
    "emotional",
    "exhausted",
    "feeling",
    "guilt",
    "guilty",
    "lonely",
    "loneliness",
    "mental health",
    "motivation",
    "overwhelmed",
    "panic",
    "pattern",
    "perfectionism",
    "procrastination",
    "rumination",
    "ruminating",
    "self criticism",
    "self-criticism",
    "shame",
    "shutdown",
    "stressed",
    "stress",
    "stuck",
    "therapy",
    "thought loop",
    "thoughts",
    "trauma",
    "worry",
]


PORTFOLIO_KEYWORDS = [
    "about this website",
    "about your work",
    "feature",
    "healthcare innovation",
    "how this works",
    "medbrief",
    "mission",
    "portfolio",
    "product",
    "project",
    "recall",
    "roadmap",
    "startup",
    "this app",
    "this website",
    "tool",
    "vision",
    "what did you build",
    "what is this app",
    "what is this project",
]


CRISIS_PATTERNS = [
    r"\bi want to die\b",
    r"\bi wanna die\b",
    r"\bi should die\b",
    r"\bi don't want to live\b",
    r"\bi do not want to live\b",
    r"\bi feel like dying\b",
    r"\bi am feeling like dying\b",
    r"\bi'm feeling like dying\b",
    r"\bi feel like i want to die\b",
    r"\bi want to disappear\b",
    r"\bi don't want to be here\b",
    r"\bi do not want to be here\b",
    r"\bi want it to end\b",
    r"\bkill myself\b",
    r"\bhurt myself\b",
    r"\bsuicidal\b",
    r"\bend my life\b",
    r"\bthinking about suicide\b",
    r"\bconsidering suicide\b",
    r"\bsuicide seems (nice|good|better)\b",
    r"\bsuicide sounds (nice|good|better)\b",
    r"^\s*suicide\s*$",
    r"^\s*suicidal\s*$",
    r"\bcan't stay safe\b",
    r"\bcan'?t stay safe\b",
    r"\bnot safe with myself\b",
    r"\bhurt someone\b",
    r"\bkill someone\b",
    r"\boverdose\b",
    r"\bsuicidal thoughts?\b",
    r"\bending my life\b",
]


ADVICE_PATTERNS = [
    r"\bwhat should i do\b",
    r"\bany advice\b",
    r"\bhelp me decide\b",
    r"\bhow do i handle\b",
    r"\bwhat would help\b",
]


RED_FLAG_PATTERNS = {
    "cardiorespiratory": [
        r"\bchest pain\b",
        r"\bshortness of breath\b",
        r"\btrouble breathing\b",
        r"\bcan'?t breathe\b",
        r"\bsevere chest pressure\b",
    ],
    "stroke": [
        r"\bone side of (my|the) face\b",
        r"\bslurred speech\b",
        r"\bcan'?t move (my|one) arm\b",
        r"\bstroke symptoms?\b",
        r"\bface drooping\b",
    ],
    "bleeding": [
        r"\bsevere bleeding\b",
        r"\bwon'?t stop bleeding\b",
        r"\bcoughing up blood\b",
        r"\bvomiting blood\b",
    ],
    "mental_health": [
        r"\bsuicidal\b",
        r"\bwant to die\b",
        r"\bkill myself\b",
        r"\bhurt myself\b",
    ],
}


DOSING_PATTERNS = [
    r"\bdose\b",
    r"\bdosage\b",
    r"\bhow many mg\b",
    r"\bhow many milligrams\b",
    r"\bmilligrams?\b",
    r"\bhow much should i take\b",
    r"\bcan i take\b",
    r"\bprescribe\b",
]


DIAGNOSIS_PATTERNS = [
    r"\bdo i have\b",
    r"\bwhat do i have\b",
    r"\bam i having\b",
    r"\bis this cancer\b",
    r"\bis this serious\b",
    r"\bdiagnose me\b",
    r"\bwhat is wrong with me\b",
]


def get_system_prompt(preset: str = DEFAULT_SYSTEM_PRESET, override: str = "") -> str:
    if override.strip():
        return override.strip()
    return SYSTEM_PROMPTS.get(preset, SYSTEM_PROMPTS[DEFAULT_SYSTEM_PRESET])


def get_generation_defaults(preset: str = DEFAULT_SYSTEM_PRESET) -> Dict[str, float | int]:
    return dict(PRESET_GENERATION_DEFAULTS.get(preset, PRESET_GENERATION_DEFAULTS[DEFAULT_SYSTEM_PRESET]))


def get_sample_prompt(preset: str = DEFAULT_SYSTEM_PRESET) -> str:
    return PRESET_SAMPLE_PROMPTS.get(preset, PRESET_SAMPLE_PROMPTS[DEFAULT_SYSTEM_PRESET])


def get_model_preset(name: str = "base") -> Dict[str, int | float]:
    return dict(MODEL_SIZE_PRESETS.get(name, MODEL_SIZE_PRESETS["base"]))


def is_medical_query(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in MEDICAL_KEYWORDS)


def is_psychology_query(text: str) -> bool:
    lowered = text.lower()
    if any(keyword in lowered for keyword in PSYCHOLOGY_KEYWORDS):
        return True
    if any(
        phrase in lowered
        for phrase in (
            "counsel me",
            "support me",
            "help me emotionally",
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
            "emotionally exhausted",
        )
    ):
        return True
    return lowered.startswith(("i feel", "i'm feeling", "i am feeling", "i've been feeling", "i have been feeling"))


def is_portfolio_query(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in PORTFOLIO_KEYWORDS)


def is_crisis_query(text: str) -> bool:
    normalized = _normalize_intent_text(text)
    if has_red_flag_symptoms(normalized):
        return True
    if any(re.search(pattern, normalized) for pattern in CRISIS_PATTERNS):
        return True

    tokens = normalized.split()
    token_set = set(tokens)
    first_person = token_set.intersection({"i", "im", "i'm", "ive", "i've", "me", "my", "myself"})

    def fuzzy_token_match(candidates: tuple[str, ...], threshold: float = 0.82) -> bool:
        for token in tokens:
            if token in candidates:
                return True
            for candidate in candidates:
                if len(token) >= 3 and len(candidate) >= 3:
                    if re.sub(r"(.)\1+", r"\1", token) == candidate:
                        return True
                    if _token_similarity(token, candidate) >= threshold:
                        return True
        return False

    die_signal = fuzzy_token_match(("die", "dying", "dead", "disappear"))
    self_harm_signal = fuzzy_token_match(("kill", "hurt", "harm", "overdose", "suicide", "suicidal"))
    desire_signal = fuzzy_token_match(("want", "wanna", "need", "should", "going", "gonna", "wish"))
    unsafe_signal = "unsafe" in token_set or ("not" in token_set and "safe" in token_set)
    life_negation = ("not" in token_set and {"live", "alive", "here"}.intersection(token_set)) or "disappear" in token_set

    if first_person and (
        (die_signal and desire_signal)
        or (self_harm_signal and {"myself", "me", "self"}.intersection(token_set))
        or (self_harm_signal and desire_signal)
        or unsafe_signal
        or life_negation
    ):
        return True

    if "suicide" in token_set and any(term in token_set for term in ("nice", "good", "better", "tempting", "appealing")):
        return True
    return False


def wants_advice(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in ADVICE_PATTERNS)


def has_red_flag_symptoms(text: str) -> bool:
    lowered = text.lower()
    for patterns in RED_FLAG_PATTERNS.values():
        for pattern in patterns:
            if re.search(pattern, lowered):
                return True
    return False


def asks_for_dosing(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in DOSING_PATTERNS)


def asks_for_diagnosis(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in DIAGNOSIS_PATTERNS)


def classify_query_mode(text: str, primary_use: str = "balanced", site_context: str = "") -> str:
    lowered_primary = primary_use.strip().lower()
    lowered_context = site_context.strip().lower()
    if is_crisis_query(text):
        return "crisis"
    if is_portfolio_query(text) or any(token in lowered_context for token in ("portfolio", "project", "website")):
        return "portfolio"
    if is_medical_query(text):
        return "medical"
    if is_psychology_query(text) or any(token in lowered_context for token in ("recall", "support", "wellness", "mental")):
        return "psychology"
    if lowered_primary == "support":
        return "psychology"
    if lowered_primary == "portfolio":
        return "portfolio"
    if lowered_primary == "healthcare":
        return "medical"
    return "general"


def build_query_safety_instruction(text: str) -> str:
    instructions = []
    if is_crisis_query(text):
        instructions.append(
            "This prompt may involve a crisis or emergency. Prioritize immediate safety guidance and do not drift into casual conversation."
        )
    elif is_medical_query(text):
        instructions.append(
            "This is a medical-education question. "
            "Stay educational, avoid diagnostic certainty, and keep the answer measured."
        )
    elif is_psychology_query(text):
        instructions.append(
            "This is a psychology or emotional-support question. Be warm, clear, and psychologically informed without pretending to be a licensed therapist."
        )
    elif is_portfolio_query(text):
        instructions.append(
            "This is a project or portfolio question. Explain clearly, sound polished, and focus on value, mission, and design quality."
        )
    if asks_for_diagnosis(text):
        instructions.append(
            "Do not diagnose the user. Explain possibilities carefully and recommend professional evaluation when appropriate."
        )
    if asks_for_dosing(text):
        instructions.append(
            "Do not provide medication dosing, prescribing, or authoritative treatment instructions."
        )
    return " ".join(instructions).strip()


def _normalize_intent_text(text: str) -> str:
    lowered = text.lower().replace("’", "'")
    lowered = re.sub(r"[^a-z0-9'\s]", " ", lowered)
    lowered = re.sub(r"\bim\b", "i am", lowered)
    lowered = re.sub(r"\bi m\b", "i am ", lowered)
    lowered = re.sub(r"\bwanna\b", "want", lowered)
    lowered = re.sub(r"\bgonna\b", "going", lowered)
    lowered = re.sub(r"([a-z])\1{2,}", r"\1", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def _token_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    if abs(len(left) - len(right)) > 2:
        return 0.0
    matches = sum(1 for a, b in zip(left, right) if a == b)
    longest = max(len(left), len(right), 1)
    prefix_bonus = 0.2 if left[:2] == right[:2] else 0.0
    return min(1.0, (matches / longest) + prefix_bonus)
