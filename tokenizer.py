from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

from utils import ensure_dir, save_json

try:
    import sentencepiece as spm
except ImportError:  # pragma: no cover - handled at runtime with a friendly message
    spm = None


SPECIAL_TOKENS = [
    "<system>",
    "</system>",
    "<context>",
    "</context>",
    "<user>",
    "</user>",
    "<assistant>",
    "</assistant>",
    "<conversation_end>",
]


def require_sentencepiece() -> None:
    if spm is None:
        raise ImportError(
            "sentencepiece is required for tokenizer training/loading. "
            "Install dependencies from requirements.txt before running this script."
        )


class ChatTokenizer:
    def __init__(self, model_file: str | Path):
        require_sentencepiece()
        self.model_file = str(model_file)
        self.processor = spm.SentencePieceProcessor(model_file=self.model_file)
        self.special_token_ids: Dict[str, int] = {
            token: self.processor.piece_to_id(token) for token in SPECIAL_TOKENS
        }

    @property
    def vocab_size(self) -> int:
        return self.processor.get_piece_size()

    @property
    def pad_id(self) -> int:
        return self.processor.pad_id()

    @property
    def bos_id(self) -> int:
        return self.processor.bos_id()

    @property
    def eos_id(self) -> int:
        return self.processor.eos_id()

    @property
    def unk_id(self) -> int:
        return self.processor.unk_id()

    @property
    def assistant_start_id(self) -> int:
        return self.token_to_id("<assistant>")

    @property
    def assistant_end_id(self) -> int:
        return self.token_to_id("</assistant>")

    @property
    def conversation_end_id(self) -> int:
        return self.token_to_id("<conversation_end>")

    @property
    def default_stop_ids(self) -> List[int]:
        return [self.assistant_end_id, self.conversation_end_id, self.eos_id]

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        return list(self.processor.encode(text, out_type=int, add_bos=add_bos, add_eos=add_eos))

    def decode(self, token_ids: Iterable[int]) -> str:
        return self.processor.decode(list(token_ids))

    def token_to_id(self, token: str) -> int:
        return self.processor.piece_to_id(token)

    def id_to_token(self, token_id: int) -> str:
        return self.processor.id_to_piece(token_id)

    def has_token(self, token: str) -> bool:
        token_id = self.token_to_id(token)
        return token_id >= 0 and self.id_to_token(token_id) == token


def train_tokenizer(
    input_path: str | Path,
    model_prefix: str | Path,
    vocab_size: int = 8000,
    model_type: str = "bpe",
    character_coverage: float = 1.0,
) -> Path:
    require_sentencepiece()
    input_path = Path(input_path)
    model_prefix = Path(model_prefix)
    ensure_dir(model_prefix.parent)

    spm.SentencePieceTrainer.train(
        input=str(input_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        byte_fallback=True,
        normalization_rule_name="nfkc_cf",
        split_digits=True,
        max_sentence_length=16384,
        train_extremely_large_corpus=True,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        user_defined_symbols=SPECIAL_TOKENS,
    )

    metadata = {
        "input_path": str(input_path),
        "model_prefix": str(model_prefix),
        "model_file": str(model_prefix.with_suffix(".model")),
        "vocab_file": str(model_prefix.with_suffix(".vocab")),
        "vocab_size": vocab_size,
        "model_type": model_type,
        "character_coverage": character_coverage,
        "special_tokens": SPECIAL_TOKENS,
    }
    save_json(model_prefix.parent / f"{model_prefix.name}_tokenizer_config.json", metadata)
    return model_prefix.with_suffix(".model")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or inspect the local tokenizer.")
    parser.add_argument("--input", required=True, help="Prepared training corpus text file.")
    parser.add_argument(
        "--model-prefix",
        default="data/tokenizer/local_chatbot",
        help="Output prefix for the SentencePiece model and vocab files.",
    )
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--model-type", default="bpe", choices=["bpe", "unigram", "char", "word"])
    parser.add_argument("--character-coverage", type=float, default=1.0)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    model_file = train_tokenizer(
        input_path=args.input,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
    )
    print(f"Tokenizer saved to: {model_file}")


if __name__ == "__main__":
    main()
