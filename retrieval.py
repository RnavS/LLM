from __future__ import annotations

import argparse
import math
import pickle
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from presets import is_medical_query
from utils import configure_console_output, ensure_dir, load_text


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9']+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "many",
    "my",
    "of",
    "on",
    "or",
    "should",
    "the",
    "to",
    "what",
    "when",
    "with",
}


def tokenize_retrieval_text(text: str) -> List[str]:
    return [token for token in TOKEN_PATTERN.findall(text.lower()) if token not in STOPWORDS]


def chunk_text_by_words(text: str, chunk_words: int = 350, overlap_words: int = 50) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    step = max(1, chunk_words - overlap_words)
    while start < len(words):
        chunk = words[start : start + chunk_words]
        if not chunk:
            break
        chunks.append(" ".join(chunk).strip())
        start += step
    return chunks


@dataclass
class KnowledgeChunk:
    text: str
    source_path: str
    domain: str
    tfidf_vector: Dict[str, float]
    norm: float

    def as_prompt_dict(self, score: float) -> Dict[str, str | float]:
        return {
            "text": self.text,
            "source": self.source_path,
            "domain": self.domain,
            "score": round(score, 4),
        }

    def to_dict(self) -> Dict[str, object]:
        return {
            "text": self.text,
            "source_path": self.source_path,
            "domain": self.domain,
            "tfidf_vector": self.tfidf_vector,
            "norm": self.norm,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "KnowledgeChunk":
        return cls(
            text=str(payload["text"]),
            source_path=str(payload["source_path"]),
            domain=str(payload["domain"]),
            tfidf_vector=dict(payload["tfidf_vector"]),
            norm=float(payload["norm"]),
        )


class KnowledgeIndex:
    def __init__(self, chunks: Sequence[KnowledgeChunk], idf: Dict[str, float]):
        self.chunks = list(chunks)
        self.idf = dict(idf)

    def _vectorize(self, tokens: Sequence[str]) -> tuple[Dict[str, float], float]:
        counts = Counter(tokens)
        if not counts:
            return {}, 0.0

        total = sum(counts.values())
        vector: Dict[str, float] = {}
        for token, count in counts.items():
            idf = self.idf.get(token)
            if idf is None:
                continue
            vector[token] = (count / total) * idf
        norm = math.sqrt(sum(weight * weight for weight in vector.values()))
        return vector, norm

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict[str, str | float]]:
        query_tokens = tokenize_retrieval_text(query)
        query_vector, query_norm = self._vectorize(query_tokens)
        if not query_vector or query_norm == 0:
            return []

        query_is_medical = is_medical_query(query)
        scored_chunks: List[tuple[float, KnowledgeChunk]] = []
        for chunk in self.chunks:
            overlap = set(query_vector).intersection(chunk.tfidf_vector)
            if not overlap:
                continue
            score = sum(query_vector[token] * chunk.tfidf_vector[token] for token in overlap)
            score /= max(query_norm * chunk.norm, 1e-8)
            if query_is_medical and chunk.domain == "medical":
                score *= 1.15
            scored_chunks.append((score, chunk))

        scored_chunks.sort(key=lambda item: item[0], reverse=True)
        return [chunk.as_prompt_dict(score) for score, chunk in scored_chunks[:top_k] if score > 0]

    def save(self, output_path: str | Path) -> Path:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        with output_path.open("wb") as handle:
            pickle.dump(
                {
                    "chunks": [chunk.to_dict() for chunk in self.chunks],
                    "idf": self.idf,
                },
                handle,
            )
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "KnowledgeIndex":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        chunks = [KnowledgeChunk.from_dict(chunk_payload) for chunk_payload in payload["chunks"]]
        return cls(chunks=chunks, idf=payload["idf"])


def build_index_from_directories(
    knowledge_dirs: Sequence[str | Path],
    chunk_words: int = 350,
    overlap_words: int = 50,
) -> KnowledgeIndex:
    documents: List[tuple[str, str, str]] = []
    for directory in knowledge_dirs:
        path = Path(directory)
        if not path.exists():
            continue
        for file_path in sorted(path.rglob("*")):
            if not file_path.is_file() or file_path.suffix.lower() not in {".txt", ".md"}:
                continue
            domain = file_path.parent.name.lower()
            text = load_text(file_path).strip()
            for chunk in chunk_text_by_words(text, chunk_words=chunk_words, overlap_words=overlap_words):
                documents.append((chunk, str(file_path), domain))

    if not documents:
        raise ValueError("No knowledge documents found to index.")

    tokenized_docs = [tokenize_retrieval_text(text) for text, _, _ in documents]
    document_frequency: Counter[str] = Counter()
    for tokens in tokenized_docs:
        document_frequency.update(set(tokens))

    total_docs = len(tokenized_docs)
    idf = {
        token: math.log((1 + total_docs) / (1 + frequency)) + 1.0
        for token, frequency in document_frequency.items()
    }

    chunks: List[KnowledgeChunk] = []
    for (text, source_path, domain), tokens in zip(documents, tokenized_docs):
        counts = Counter(tokens)
        total_tokens = max(1, sum(counts.values()))
        vector = {
            token: (count / total_tokens) * idf[token]
            for token, count in counts.items()
            if token in idf
        }
        norm = math.sqrt(sum(weight * weight for weight in vector.values()))
        chunks.append(
            KnowledgeChunk(
                text=text,
                source_path=source_path,
                domain=domain,
                tfidf_vector=vector,
                norm=norm or 1e-8,
            )
        )

    return KnowledgeIndex(chunks=chunks, idf=idf)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build or query a fully local knowledge index.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build a TF-IDF index from local documents.")
    build_parser.add_argument("--knowledge-dir", action="append", default=[], help="Knowledge root directory.")
    build_parser.add_argument("--output", default="data/index/knowledge_index.pkl")
    build_parser.add_argument("--chunk-words", type=int, default=350)
    build_parser.add_argument("--overlap-words", type=int, default=50)

    query_parser = subparsers.add_parser("query", help="Query an existing knowledge index.")
    query_parser.add_argument("--index", default="data/index/knowledge_index.pkl")
    query_parser.add_argument("--prompt", required=True)
    query_parser.add_argument("--top-k", type=int, default=4)
    return parser


def main() -> None:
    configure_console_output()
    args = build_arg_parser().parse_args()
    if args.command == "build":
        index = build_index_from_directories(
            knowledge_dirs=args.knowledge_dir or ["data/knowledge"],
            chunk_words=args.chunk_words,
            overlap_words=args.overlap_words,
        )
        output_path = index.save(args.output)
        print(f"Knowledge index saved to: {output_path}")
        print(f"Indexed chunks: {len(index.chunks)}")
        return

    if args.command == "query":
        index = KnowledgeIndex.load(args.index)
        results = index.retrieve(args.prompt, top_k=args.top_k)
        for result in results:
            print(f"[{result['domain']}] {result['source']} score={result['score']}")
            print(result["text"])
            print()


if __name__ == "__main__":
    main()
