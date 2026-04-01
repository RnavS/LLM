from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Type, TypeVar


T = TypeVar("T", bound="SerializableConfig")


class SerializableConfig:
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        valid_fields = {field.name for field in fields(cls)}
        filtered = {key: value for key, value in data.items() if key in valid_fields}
        return cls(**filtered)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls: Type[T], path: str | Path) -> T:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


@dataclass
class ModelConfig(SerializableConfig):
    vocab_size: int = 0
    block_size: int = 256
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    dropout: float = 0.1
    bias: bool = True


@dataclass
class TrainingConfig(SerializableConfig):
    input_path: str = "cleaned_data.txt"
    fiction_input: str = ""
    dataset_mode: str = "auto"
    prepared_path: str = "data/prepared_corpus.txt"
    tokenizer_prefix: str = "data/tokenizer/local_chatbot"
    output_dir: str = "checkpoints/local_chatbot"
    model_preset: str = "base"
    system_preset: str = "factual-medical-lite"
    general_knowledge_dir: str = "data/knowledge/general"
    medical_knowledge_dir: str = "data/knowledge/medical"
    seed_chat_dir: str = "data/chat_seed"
    general_weight: int = 7
    medical_weight: int = 2
    fiction_weight: int = 1
    knowledge_index_path: str = "data/index/knowledge_index.pkl"
    retrieval_top_k: int = 4
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    max_steps: int = 5000
    epochs: int = 0
    warmup_steps: int = 200
    eval_interval: int = 200
    eval_batches: int = 20
    log_interval: int = 20
    sample_interval: int = 200
    save_interval: int = 500
    val_ratio: float = 0.1
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "auto"
    use_amp: bool = True
    compile_model: bool = False
    vocab_size: int = 8000
    tokenizer_model_type: str = "bpe"
    tokenizer_character_coverage: float = 1.0
    retrain_tokenizer: bool = False
    sample_prompt: str = "What is hypertension?"
    sample_system_prompt: str = (
        "You are a precise, intellectually serious local assistant focused on factual, educational answers."
    )
    sample_max_new_tokens: int = 80
    raw_chunk_chars: int = 1200


@dataclass
class GenerationConfig(SerializableConfig):
    max_new_tokens: int = 160
    temperature: float = 0.35
    top_k: int = 30
    top_p: float = 0.85
    repetition_penalty: float = 1.05
    system_preset: str = "factual-medical-lite"
    system_prompt: str = (
        "You are a precise, intellectually serious local assistant focused on factual, educational answers."
    )
    knowledge_index_path: str = "data/index/knowledge_index.pkl"
    retrieval_top_k: int = 4
    response_mode: str = "grounded"
    web_search_enabled: bool = True
    web_search_max_results: int = 4
