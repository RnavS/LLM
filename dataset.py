from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

try:
    import torch
except ImportError:  # pragma: no cover - makes data prep usable before training deps are installed
    torch = None

from presets import DEFAULT_SYSTEM_PRESET, get_system_prompt
from utils import ensure_dir, load_text, save_json, save_text


DEFAULT_SYSTEM_PROMPT = get_system_prompt(DEFAULT_SYSTEM_PRESET)
ROLE_TOKENS = {
    "system": ("<system>", "</system>"),
    "user": ("<user>", "</user>"),
    "assistant": ("<assistant>", "</assistant>"),
}


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_long_block(text: str, max_chars: int) -> List[str]:
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        candidate = sentence if not current else f"{current} {sentence}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(sentence) <= max_chars:
            current = sentence
        else:
            for offset in range(0, len(sentence), max_chars):
                chunks.append(sentence[offset : offset + max_chars])
            current = ""
    if current:
        chunks.append(current)
    return chunks


def chunk_raw_text(text: str, max_chars: int = 1200) -> List[str]:
    normalized = normalize_text(text)
    blocks = [block.strip() for block in re.split(r"\n\s*\n", normalized) if block.strip()]
    if not blocks:
        blocks = [normalized]

    chunks: List[str] = []
    current = ""
    for block in blocks:
        candidate = block if not current else f"{current}\n\n{block}"
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        if len(block) <= max_chars:
            current = block
        else:
            chunks.extend(split_long_block(block, max_chars))
    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk.strip()]


def collect_supported_files(path: str | Path, suffixes: Sequence[str]) -> List[Path]:
    directory = Path(path)
    if not directory.exists():
        return []
    return [file_path for file_path in sorted(directory.rglob("*")) if file_path.is_file() and file_path.suffix.lower() in suffixes]


def directory_has_supported_files(path: str | Path, suffixes: Sequence[str]) -> bool:
    return bool(collect_supported_files(path, suffixes))


class ConversationFormatter:
    def __init__(self, default_system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        self.default_system_prompt = default_system_prompt

    def format_message(self, role: str, content: str) -> str:
        role = role.lower().strip()
        if role not in ROLE_TOKENS:
            raise ValueError(f"Unsupported role: {role}")
        open_token, close_token = ROLE_TOKENS[role]
        content = normalize_text(content)
        return f"{open_token}\n{content}\n{close_token}"

    def format_context(self, retrieved_context: Sequence[Dict[str, str | float]] | None = None) -> str:
        if not retrieved_context:
            return ""
        parts = ["<context>"]
        for index, chunk in enumerate(retrieved_context, start=1):
            source = Path(str(chunk.get("source", "local"))).name
            domain = str(chunk.get("domain", "general"))
            score = chunk.get("score", "")
            text = normalize_text(str(chunk.get("text", "")))
            parts.append(f"[Source {index} | {domain} | {source} | score={score}]")
            parts.append(text)
        parts.append("</context>")
        return "\n".join(parts)

    def format_conversation(
        self,
        messages: Sequence[Dict[str, str]],
        system_prompt: str | None = None,
        include_assistant_prompt: bool = False,
        add_conversation_end: bool = True,
        retrieved_context: Sequence[Dict[str, str | float]] | None = None,
        extra_system_instruction: str = "",
    ) -> str:
        parts: List[str] = []
        effective_system_prompt = (system_prompt or self.default_system_prompt).strip()
        if extra_system_instruction.strip():
            effective_system_prompt = f"{effective_system_prompt}\n\n{extra_system_instruction.strip()}".strip()

        has_system_message = any(message.get("role", "").lower() == "system" for message in messages)
        if effective_system_prompt and not has_system_message:
            parts.append(self.format_message("system", effective_system_prompt))

        context_block = self.format_context(retrieved_context)
        if context_block:
            parts.append(context_block)

        for message in messages:
            role = message.get("role", "").lower().strip()
            content = message.get("content", "")
            if not content or not content.strip():
                continue
            parts.append(self.format_message(role, content))

        if include_assistant_prompt:
            parts.append("<assistant>\n")

        if add_conversation_end:
            parts.append("<conversation_end>")

        return "\n".join(parts).strip()


def load_chat_records(input_path: str | Path) -> List[Dict[str, object]]:
    path = Path(input_path)
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        raw_records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif suffix == ".json":
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and "messages" in loaded:
            raw_records = [loaded]
        elif isinstance(loaded, list):
            raw_records = loaded
        else:
            raise ValueError("JSON chat data must be a list of records or a single record with 'messages'.")
    else:
        raise ValueError("Chat datasets must use .json or .jsonl format.")

    conversations: List[Dict[str, object]] = []
    for record in raw_records:
        system_prompt = record.get("system") if isinstance(record, dict) else None
        domain = record.get("domain", "general") if isinstance(record, dict) else "general"
        if isinstance(record, dict) and "messages" in record:
            messages = record["messages"]
        elif isinstance(record, dict) and "prompt" in record and "response" in record:
            messages = [
                {"role": "user", "content": record["prompt"]},
                {"role": "assistant", "content": record["response"]},
            ]
        elif isinstance(record, list):
            messages = record
        else:
            raise ValueError("Each chat record must contain 'messages' or a 'prompt'/'response' pair.")

        cleaned_messages: List[Dict[str, str]] = []
        for message in messages:
            role = message.get("role", "").lower().strip()
            content = message.get("content", "")
            if role not in ROLE_TOKENS or not str(content).strip():
                continue
            cleaned_messages.append({"role": role, "content": str(content)})

        if cleaned_messages:
            conversations.append(
                {
                    "system_prompt": system_prompt,
                    "messages": cleaned_messages,
                    "domain": str(domain or "general").lower(),
                }
            )

    return conversations


def prepare_corpus(
    input_path: str | Path,
    output_path: str | Path,
    mode: str = "auto",
    default_system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    raw_chunk_chars: int = 1200,
) -> Dict[str, object]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    formatter = ConversationFormatter(default_system_prompt=default_system_prompt)

    if mode == "auto":
        mode = "chat" if input_path.suffix.lower() in {".json", ".jsonl"} else "raw"

    if mode == "raw":
        chunks = chunk_raw_text(load_text(input_path), max_chars=raw_chunk_chars)
        samples = [
            formatter.format_conversation(
                [{"role": "assistant", "content": chunk}],
                system_prompt=default_system_prompt,
            )
            for chunk in chunks
        ]
    elif mode == "chat":
        conversations = load_chat_records(input_path)
        samples = [
            formatter.format_conversation(
                conversation["messages"],  # type: ignore[index]
                system_prompt=conversation.get("system_prompt") or default_system_prompt,
            )
            for conversation in conversations
        ]
    else:
        raise ValueError("mode must be one of: auto, raw, chat")

    prepared_text = "\n\n".join(sample for sample in samples if sample.strip()).strip() + "\n"
    save_text(output_path, prepared_text)

    metadata = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "mode": mode,
        "num_samples": len(samples),
        "default_system_prompt": default_system_prompt,
        "raw_chunk_chars": raw_chunk_chars,
    }
    save_json(output_path.with_suffix(output_path.suffix + ".meta.json"), metadata)
    return metadata


def _load_raw_domain_samples(
    knowledge_dir: str | Path,
    formatter: ConversationFormatter,
    system_prompt: str,
    raw_chunk_chars: int,
) -> List[str]:
    samples: List[str] = []
    for file_path in collect_supported_files(knowledge_dir, suffixes=(".txt", ".md")):
        chunks = chunk_raw_text(load_text(file_path), max_chars=raw_chunk_chars)
        samples.extend(
            formatter.format_conversation(
                [{"role": "assistant", "content": chunk}],
                system_prompt=system_prompt,
            )
            for chunk in chunks
        )
    return samples


def _load_seed_chat_samples(
    seed_chat_dir: str | Path,
    formatter: ConversationFormatter,
    system_prompt: str,
) -> Dict[str, List[str]]:
    domain_samples: Dict[str, List[str]] = {"general": [], "medical": []}
    for file_path in collect_supported_files(seed_chat_dir, suffixes=(".jsonl", ".json")):
        for conversation in load_chat_records(file_path):
            domain = str(conversation.get("domain", "general")).lower()
            target_domain = "medical" if domain == "medical" else "general"
            domain_samples[target_domain].append(
                formatter.format_conversation(
                    conversation["messages"],  # type: ignore[index]
                    system_prompt=conversation.get("system_prompt") or system_prompt,
                )
            )
    return domain_samples


def prepare_blended_corpus(
    output_path: str | Path,
    default_system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    general_knowledge_dir: str | Path = "data/knowledge/general",
    medical_knowledge_dir: str | Path = "data/knowledge/medical",
    seed_chat_dir: str | Path = "data/chat_seed",
    fiction_input: str | Path = "",
    general_weight: int = 7,
    medical_weight: int = 2,
    fiction_weight: int = 1,
    raw_chunk_chars: int = 1200,
) -> Dict[str, object]:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    formatter = ConversationFormatter(default_system_prompt=default_system_prompt)

    general_samples = _load_raw_domain_samples(
        general_knowledge_dir,
        formatter=formatter,
        system_prompt=default_system_prompt,
        raw_chunk_chars=raw_chunk_chars,
    )
    medical_samples = _load_raw_domain_samples(
        medical_knowledge_dir,
        formatter=formatter,
        system_prompt=default_system_prompt,
        raw_chunk_chars=raw_chunk_chars,
    )
    seed_samples = _load_seed_chat_samples(seed_chat_dir, formatter=formatter, system_prompt=default_system_prompt)

    fiction_samples: List[str] = []
    fiction_path = Path(fiction_input) if str(fiction_input).strip() else None
    if fiction_path and fiction_path.exists():
        fiction_samples = [
            formatter.format_conversation(
                [{"role": "assistant", "content": chunk}],
                system_prompt=default_system_prompt,
            )
            for chunk in chunk_raw_text(load_text(fiction_path), max_chars=raw_chunk_chars)
        ]

    def rebalance_samples(samples: List[str], target_count: int) -> List[str]:
        if not samples or target_count <= 0:
            return []
        if target_count == len(samples):
            return list(samples)
        if target_count < len(samples):
            step = len(samples) / float(target_count)
            return [samples[min(int(index * step), len(samples) - 1)] for index in range(target_count)]
        rebalanced = []
        for index in range(target_count):
            rebalanced.append(samples[index % len(samples)])
        return rebalanced

    general_bucket = list(general_samples) + list(seed_samples["general"])
    medical_bucket = list(medical_samples) + list(seed_samples["medical"])
    fiction_bucket = list(fiction_samples)

    weighted_buckets = {
        "general": (general_bucket, max(0, general_weight)),
        "medical": (medical_bucket, max(0, medical_weight)),
        "fiction": (fiction_bucket, max(0, fiction_weight)),
    }

    ratios = {
        name: weight / float(sum(weight for _, weight in weighted_buckets.values()) or 1)
        for name, (_, weight) in weighted_buckets.items()
        if weight > 0
    }
    preferred_names = [name for name in ("general", "medical") if weighted_buckets[name][0] and ratios.get(name, 0.0) > 0]
    names_for_total = preferred_names or [name for name, (samples, _) in weighted_buckets.items() if samples and ratios.get(name, 0.0) > 0]

    required_totals = []
    for name in names_for_total:
        samples, _ = weighted_buckets[name]
        ratio = ratios.get(name, 0.0)
        if samples and ratio > 0:
            required_totals.append(int(math.ceil(len(samples) / ratio)))
    target_total = max(required_totals) if required_totals else 0

    weighted_sections: List[str] = []
    target_counts = {}
    for name, (samples, _) in weighted_buckets.items():
        ratio = ratios.get(name, 0.0)
        target_count = int(round(target_total * ratio)) if samples and ratio > 0 else 0
        if samples and target_count == 0:
            target_count = 1
        target_counts[name] = target_count
        weighted_sections.extend(rebalance_samples(samples, target_count))

    if not weighted_sections:
        raise ValueError("No blended training data was found. Add local knowledge docs, seed chats, or fiction input.")

    prepared_text = "\n\n".join(sample for sample in weighted_sections if sample.strip()).strip() + "\n"
    save_text(output_path, prepared_text)

    metadata = {
        "mode": "blend",
        "output_path": str(output_path),
        "default_system_prompt": default_system_prompt,
        "weights": {
            "general": general_weight,
            "medical": medical_weight,
            "fiction": fiction_weight,
        },
        "counts": {
            "general_raw": len(general_samples),
            "general_seed": len(seed_samples["general"]),
            "medical_raw": len(medical_samples),
            "medical_seed": len(seed_samples["medical"]),
            "fiction_raw": len(fiction_samples),
            "general_final": target_counts["general"],
            "medical_final": target_counts["medical"],
            "fiction_final": target_counts["fiction"],
            "total_weighted": len(weighted_sections),
        },
        "sources": {
            "general_knowledge_dir": str(general_knowledge_dir),
            "medical_knowledge_dir": str(medical_knowledge_dir),
            "seed_chat_dir": str(seed_chat_dir),
            "fiction_input": str(fiction_path) if fiction_path else "",
        },
    }
    save_json(output_path.with_suffix(output_path.suffix + ".meta.json"), metadata)
    return metadata


@dataclass
class PackedTokenDataset:
    token_ids: List[int]
    block_size: int
    val_ratio: float = 0.1

    def __post_init__(self) -> None:
        if torch is None:
            raise ImportError("torch is required to build training batches. Install requirements.txt first.")
        if len(self.token_ids) < self.block_size + 2:
            raise ValueError(f"Need at least {self.block_size + 2} tokens, got {len(self.token_ids)}.")

        split_index = int(len(self.token_ids) * (1.0 - self.val_ratio))
        split_index = min(max(split_index, self.block_size + 1), len(self.token_ids) - self.block_size - 1)

        self.train_data = torch.tensor(self.token_ids[:split_index], dtype=torch.long)
        self.val_data = torch.tensor(self.token_ids[split_index:], dtype=torch.long)

        if len(self.val_data) < self.block_size + 1:
            self.val_data = self.train_data[-(self.block_size + 1) :].clone()

    @property
    def num_train_tokens(self) -> int:
        return int(self.train_data.numel())

    @property
    def num_val_tokens(self) -> int:
        return int(self.val_data.numel())

    def get_batch(self, split: str, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if torch is None:
            raise ImportError("torch is required to sample training batches. Install requirements.txt first.")
        source = self.train_data if split == "train" else self.val_data
        max_offset = len(source) - self.block_size - 1
        offsets = torch.randint(0, max_offset + 1, (batch_size,))
        x = torch.stack([source[offset : offset + self.block_size] for offset in offsets])
        y = torch.stack([source[offset + 1 : offset + self.block_size + 1] for offset in offsets])
        return x.to(device), y.to(device)


def build_chat_input_ids(
    tokenizer,
    formatter: ConversationFormatter,
    system_prompt: str,
    messages: Sequence[Dict[str, str]],
    block_size: int,
    max_new_tokens: int,
    retrieved_context: Sequence[Dict[str, str | float]] | None = None,
    extra_system_instruction: str = "",
) -> tuple[List[int], List[Dict[str, str]], List[Dict[str, str | float]]]:
    reserve_tokens = min(max_new_tokens, max(32, block_size // 3))
    max_prompt_tokens = max(32, block_size - reserve_tokens)
    working_messages = [dict(message) for message in messages]
    working_context = [dict(chunk) for chunk in (retrieved_context or [])]

    while True:
        prompt_text = formatter.format_conversation(
            working_messages,
            system_prompt=system_prompt,
            include_assistant_prompt=True,
            add_conversation_end=False,
            retrieved_context=working_context,
            extra_system_instruction=extra_system_instruction,
        )
        prompt_ids = tokenizer.encode(prompt_text)
        if len(prompt_ids) <= max_prompt_tokens:
            return prompt_ids, working_messages, working_context
        if len(working_messages) > 1:
            working_messages = working_messages[1:]
            continue
        if working_context:
            working_context = working_context[:-1]
            continue
        if not working_messages:
            return prompt_ids[-max_prompt_tokens:], working_messages, working_context
        working_messages = working_messages[1:]


def extract_assistant_reply(
    tokenizer,
    full_sequence: Sequence[int],
    prompt_length: int,
    stop_ids: Iterable[int],
) -> str:
    stop_ids = set(stop_ids)
    reply_ids: List[int] = []
    for token_id in full_sequence[prompt_length:]:
        if token_id in stop_ids:
            break
        reply_ids.append(token_id)
    return tokenizer.decode(reply_ids).strip()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare datasets for local training.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Prepare one raw or chat dataset.")
    prepare_parser.add_argument("--input", required=True, help="Input .txt, .json, or .jsonl dataset path.")
    prepare_parser.add_argument("--output", default="data/prepared_corpus.txt", help="Prepared text output path.")
    prepare_parser.add_argument("--mode", default="auto", choices=["auto", "raw", "chat"])
    prepare_parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    prepare_parser.add_argument("--raw-chunk-chars", type=int, default=1200)

    blend_parser = subparsers.add_parser("blend", help="Prepare a weighted blended corpus.")
    blend_parser.add_argument("--output", default="data/prepared_corpus.txt")
    blend_parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    blend_parser.add_argument("--general-knowledge-dir", default="data/knowledge/general")
    blend_parser.add_argument("--medical-knowledge-dir", default="data/knowledge/medical")
    blend_parser.add_argument("--seed-chat-dir", default="data/chat_seed")
    blend_parser.add_argument("--fiction-input", default="")
    blend_parser.add_argument("--general-weight", type=int, default=7)
    blend_parser.add_argument("--medical-weight", type=int, default=2)
    blend_parser.add_argument("--fiction-weight", type=int, default=1)
    blend_parser.add_argument("--raw-chunk-chars", type=int, default=1200)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.command == "prepare":
        metadata = prepare_corpus(
            input_path=args.input,
            output_path=args.output,
            mode=args.mode,
            default_system_prompt=args.system_prompt,
            raw_chunk_chars=args.raw_chunk_chars,
        )
    else:
        metadata = prepare_blended_corpus(
            output_path=args.output,
            default_system_prompt=args.system_prompt,
            general_knowledge_dir=args.general_knowledge_dir,
            medical_knowledge_dir=args.medical_knowledge_dir,
            seed_chat_dir=args.seed_chat_dir,
            fiction_input=args.fiction_input,
            general_weight=args.general_weight,
            medical_weight=args.medical_weight,
            fiction_weight=args.fiction_weight,
            raw_chunk_chars=args.raw_chunk_chars,
        )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
