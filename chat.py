from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from presets import DEFAULT_SYSTEM_PRESET, SYSTEM_PROMPTS
from runtime import generate_reply, load_runtime
from utils import configure_console_output, ensure_dir, save_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive terminal chat for the local model.")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/local_chatbot",
        help="Checkpoint file or checkpoint directory containing latest.pt.",
    )
    parser.add_argument("--tokenizer-model", default="", help="Override tokenizer model path.")
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


def save_session(path: str | Path, system_prompt: str, history: List[dict[str, str]]) -> Path:
    path = Path(path)
    ensure_dir(path.parent if path.parent != Path(".") else ".")
    save_json(path, {"system_prompt": system_prompt, "history": history})
    return path


def load_session(path: str | Path) -> tuple[str, List[dict[str, str]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    system_prompt = payload.get("system_prompt", "")
    history = payload.get("history", [])
    return system_prompt, history


def print_help() -> None:
    print("Commands:")
    print("  /reset               Clear the current conversation history")
    print("  /save [path]         Save the current session to JSON")
    print("  /load <path>         Load a previously saved session")
    print("  /system [prompt]     Show or replace the system prompt")
    print("  /help                Show commands")
    print("  /exit                Quit")


def main() -> None:
    configure_console_output()
    args = build_arg_parser().parse_args()
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

    history: List[dict[str, str]] = []
    system_prompt = runtime.generation_config.system_prompt

    print(f"Loaded checkpoint: {runtime.checkpoint_path}")
    print(f"Using system preset: {runtime.generation_config.system_preset}")
    print(f"Retrieval enabled: {'yes' if runtime.knowledge_index is not None else 'no'}")
    print_help()

    while True:
        try:
            user_input = input("\nYou> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat.")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            command, _, remainder = user_input.partition(" ")
            remainder = remainder.strip()

            if command == "/exit":
                print("Exiting chat.")
                break
            if command == "/reset":
                history.clear()
                print("Conversation cleared.")
                continue
            if command == "/save":
                target = remainder or "chat_session.json"
                saved_path = save_session(target, system_prompt, history)
                print(f"Session saved to {saved_path}")
                continue
            if command == "/load":
                if not remainder:
                    print("Usage: /load path/to/session.json")
                    continue
                system_prompt, history = load_session(remainder)
                runtime.generation_config.system_prompt = system_prompt
                runtime.formatter.default_system_prompt = system_prompt
                print(f"Session loaded from {remainder}")
                continue
            if command == "/system":
                if remainder:
                    system_prompt = remainder
                    runtime.generation_config.system_prompt = system_prompt
                    runtime.formatter.default_system_prompt = system_prompt
                    print("System prompt updated.")
                else:
                    print(f"Current system prompt:\n{system_prompt}")
                continue
            if command == "/help":
                print_help()
                continue

            print("Unknown command. Type /help to see available commands.")
            continue

        result = generate_reply(runtime, user_input, history=history)
        history = result["history"]  # type: ignore[assignment]
        reply = str(result["reply"])
        used_context = result["used_context"]

        print(f"Assistant> {reply or '[empty reply]'}")
        if used_context:
            source_names = ", ".join(sorted({Path(str(chunk['source'])).name for chunk in used_context}))
            print(f"Sources> {source_names}")


if __name__ == "__main__":
    main()
