from __future__ import annotations

import argparse
from pathlib import Path

from presets import DEFAULT_SYSTEM_PRESET, SYSTEM_PROMPTS
from runtime import generate_reply, load_runtime
from utils import configure_console_output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-turn generation using the local chatbot checkpoint.")
    parser.add_argument("--checkpoint", default="checkpoints/local_chatbot")
    parser.add_argument("--tokenizer-model", default="")
    parser.add_argument("--prompt", default="", help="User message to send to the model.")
    parser.add_argument("--system-preset", default=DEFAULT_SYSTEM_PRESET, choices=sorted(SYSTEM_PROMPTS))
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--knowledge-index", default="data/index/knowledge_index.pkl")
    parser.add_argument("--retrieval-top-k", type=int, default=4)
    parser.add_argument("--disable-retrieval", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    configure_console_output()
    args = build_arg_parser().parse_args()
    if not args.prompt:
        args.prompt = input("User prompt: ").strip()

    runtime = load_runtime(
        checkpoint=args.checkpoint,
        tokenizer_model=args.tokenizer_model,
        device_name=args.device,
        system_preset=args.system_preset,
        system_prompt=args.system_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        knowledge_index_path=args.knowledge_index,
        retrieval_top_k=args.retrieval_top_k,
        disable_retrieval=args.disable_retrieval,
    )

    result = generate_reply(runtime, args.prompt, history=[])
    print(f"Loaded checkpoint: {runtime.checkpoint_path}")
    if result["used_context"]:
        source_names = ", ".join(sorted({Path(str(chunk['source'])).name for chunk in result["used_context"]}))
        print(f"Sources: {source_names}")
    print(f"Reply:\n{result['reply']}")


if __name__ == "__main__":
    main()
