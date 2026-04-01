from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch

from config import GenerationConfig, ModelConfig, TrainingConfig
from dataset import (
    ConversationFormatter,
    PackedTokenDataset,
    build_chat_input_ids,
    directory_has_supported_files,
    extract_assistant_reply,
    prepare_blended_corpus,
    prepare_corpus,
)
from model import GPTLanguageModel
from presets import (
    DEFAULT_SYSTEM_PRESET,
    MODEL_SIZE_PRESETS,
    SYSTEM_PROMPTS,
    get_generation_defaults,
    get_model_preset,
    get_sample_prompt,
    get_system_prompt,
)
from tokenizer import ChatTokenizer, train_tokenizer
from utils import (
    append_jsonl,
    count_parameters,
    configure_console_output,
    elapsed_since,
    ensure_dir,
    get_device,
    load_checkpoint,
    load_text,
    lr_with_warmup_and_cosine_decay,
    save_checkpoint,
    set_seed,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a fully local GPT-style factual chatbot.")
    parser.add_argument("--input", default="cleaned_data.txt", help="Fallback raw or chat dataset path.")
    parser.add_argument("--fiction-input", default="", help="Optional legacy fiction/style file to keep at low weight.")
    parser.add_argument("--dataset-mode", default="auto", choices=["auto", "raw", "chat"])
    parser.add_argument("--prepared-path", default="data/prepared_corpus.txt")

    parser.add_argument("--system-preset", default=DEFAULT_SYSTEM_PRESET, choices=sorted(SYSTEM_PROMPTS))
    parser.add_argument("--system-prompt", default="", help="Optional full override for the preset system prompt.")
    parser.add_argument("--sample-prompt", default="", help="Prompt used for periodic sample generations during training.")
    parser.add_argument("--model-preset", default="base", choices=sorted(MODEL_SIZE_PRESETS))

    parser.add_argument("--general-knowledge-dir", default="data/knowledge/general")
    parser.add_argument("--medical-knowledge-dir", default="data/knowledge/medical")
    parser.add_argument("--seed-chat-dir", default="data/chat_seed")
    parser.add_argument("--general-weight", type=int, default=7)
    parser.add_argument("--medical-weight", type=int, default=2)
    parser.add_argument("--fiction-weight", type=int, default=1)
    parser.add_argument("--raw-chunk-chars", type=int, default=1200)

    parser.add_argument("--tokenizer-prefix", default="data/tokenizer/local_chatbot")
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--tokenizer-model-type", default="bpe", choices=["bpe", "unigram", "char", "word"])
    parser.add_argument("--tokenizer-character-coverage", type=float, default=1.0)
    parser.add_argument("--retrain-tokenizer", action="store_true")

    parser.add_argument("--knowledge-index-path", default="data/index/knowledge_index.pkl")
    parser.add_argument("--retrieval-top-k", type=int, default=4)

    parser.add_argument("--block-size", type=int, default=0)
    parser.add_argument("--n-embd", type=int, default=0)
    parser.add_argument("--n-head", type=int, default=0)
    parser.add_argument("--n-layer", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=-1.0)
    parser.add_argument("--no-bias", action="store_true", help="Disable linear layer bias terms.")

    parser.add_argument("--output-dir", default="checkpoints/local_chatbot")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--sample-interval", type=int, default=200)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision even on CUDA.")
    parser.add_argument("--compile-model", action="store_true", help="Use torch.compile when available.")
    parser.add_argument("--resume", default="", help="Checkpoint file or directory containing latest.pt")
    parser.add_argument("--sample-max-new-tokens", type=int, default=80)
    return parser


def estimate_loss(
    model: GPTLanguageModel,
    dataset: PackedTokenDataset,
    batch_size: int,
    eval_batches: int,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float]:
    model.eval()
    losses: dict[str, float] = {}
    with torch.no_grad():
        for split in ("train", "val"):
            split_losses = []
            for _ in range(eval_batches):
                x, y = dataset.get_batch(split, batch_size, device)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    _, loss = model(x, y)
                split_losses.append(float(loss.item()))
            losses[split] = sum(split_losses) / len(split_losses)
    model.train()
    return losses


def save_training_checkpoint(
    model: GPTLanguageModel,
    optimizer: torch.optim.Optimizer,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    tokenizer_model_path: Path,
    output_dir: Path,
    step: int,
    best_val_loss: float,
    latest_train_loss: float,
) -> None:
    payload = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": model_config.to_dict(),
        "training_config": training_config.to_dict(),
        "tokenizer_model": str(tokenizer_model_path),
        "best_val_loss": best_val_loss,
        "latest_train_loss": latest_train_loss,
    }
    save_checkpoint(output_dir / "latest.pt", payload)
    save_checkpoint(output_dir / f"step_{step:07d}.pt", payload)


def should_use_blended_corpus(args: argparse.Namespace, fiction_input: str) -> bool:
    return any(
        [
            directory_has_supported_files(args.general_knowledge_dir, (".txt", ".md")),
            directory_has_supported_files(args.medical_knowledge_dir, (".txt", ".md")),
            directory_has_supported_files(args.seed_chat_dir, (".jsonl", ".json")),
            bool(fiction_input and Path(fiction_input).exists()),
        ]
    )


def main() -> None:
    configure_console_output()
    args = build_arg_parser().parse_args()
    device = get_device(args.device)
    set_seed(args.seed)

    model_preset = get_model_preset(args.model_preset)
    block_size = int(args.block_size or model_preset["block_size"])
    n_embd = int(args.n_embd or model_preset["n_embd"])
    n_head = int(args.n_head or model_preset["n_head"])
    n_layer = int(args.n_layer or model_preset["n_layer"])
    dropout = float(model_preset["dropout"] if args.dropout < 0 else args.dropout)

    resolved_system_prompt = get_system_prompt(args.system_preset, override=args.system_prompt)
    preset_generation_defaults = get_generation_defaults(args.system_preset)
    sample_prompt = args.sample_prompt or get_sample_prompt(args.system_preset)
    fiction_input = args.fiction_input or args.input

    training_config = TrainingConfig(
        input_path=args.input,
        fiction_input=fiction_input,
        dataset_mode=args.dataset_mode,
        prepared_path=args.prepared_path,
        tokenizer_prefix=args.tokenizer_prefix,
        output_dir=args.output_dir,
        model_preset=args.model_preset,
        system_preset=args.system_preset,
        general_knowledge_dir=args.general_knowledge_dir,
        medical_knowledge_dir=args.medical_knowledge_dir,
        seed_chat_dir=args.seed_chat_dir,
        general_weight=args.general_weight,
        medical_weight=args.medical_weight,
        fiction_weight=args.fiction_weight,
        knowledge_index_path=args.knowledge_index_path,
        retrieval_top_k=args.retrieval_top_k,
        batch_size=args.batch_size,
        gradient_accumulation_steps=max(1, args.gradient_accumulation_steps),
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        log_interval=args.log_interval,
        sample_interval=args.sample_interval,
        save_interval=args.save_interval,
        val_ratio=args.val_ratio,
        grad_clip=args.grad_clip,
        seed=args.seed,
        device=str(device),
        use_amp=not args.no_amp,
        compile_model=args.compile_model,
        vocab_size=args.vocab_size,
        tokenizer_model_type=args.tokenizer_model_type,
        tokenizer_character_coverage=args.tokenizer_character_coverage,
        retrain_tokenizer=args.retrain_tokenizer,
        sample_prompt=sample_prompt,
        sample_system_prompt=resolved_system_prompt,
        sample_max_new_tokens=args.sample_max_new_tokens,
        raw_chunk_chars=args.raw_chunk_chars,
    )

    if should_use_blended_corpus(args, fiction_input):
        prepared_metadata = prepare_blended_corpus(
            output_path=training_config.prepared_path,
            default_system_prompt=resolved_system_prompt,
            general_knowledge_dir=training_config.general_knowledge_dir,
            medical_knowledge_dir=training_config.medical_knowledge_dir,
            seed_chat_dir=training_config.seed_chat_dir,
            fiction_input=training_config.fiction_input,
            general_weight=training_config.general_weight,
            medical_weight=training_config.medical_weight,
            fiction_weight=training_config.fiction_weight,
            raw_chunk_chars=training_config.raw_chunk_chars,
        )
    else:
        prepared_metadata = prepare_corpus(
            input_path=training_config.input_path,
            output_path=training_config.prepared_path,
            mode=training_config.dataset_mode,
            default_system_prompt=resolved_system_prompt,
            raw_chunk_chars=training_config.raw_chunk_chars,
        )

    tokenizer_prefix = Path(training_config.tokenizer_prefix)
    tokenizer_model_path = tokenizer_prefix.with_suffix(".model")
    if training_config.retrain_tokenizer or not tokenizer_model_path.exists():
        tokenizer_model_path = train_tokenizer(
            input_path=training_config.prepared_path,
            model_prefix=training_config.tokenizer_prefix,
            vocab_size=training_config.vocab_size,
            model_type=training_config.tokenizer_model_type,
            character_coverage=training_config.tokenizer_character_coverage,
        )

    tokenizer = ChatTokenizer(tokenizer_model_path)
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout,
        bias=not args.no_bias,
    )

    model = GPTLanguageModel(model_config).to(device)
    if training_config.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        betas=(0.9, 0.95),
    )

    output_dir = ensure_dir(training_config.output_dir)
    metrics_path = output_dir / "metrics.jsonl"
    model_config.save_json(output_dir / "model_config.json")
    training_config.save_json(output_dir / "training_config.json")

    start_step = 0
    best_val_loss = math.inf
    if args.resume:
        checkpoint, checkpoint_path = load_checkpoint(args.resume, device)
        restored_model_config = ModelConfig.from_dict(checkpoint["model_config"])
        if restored_model_config.to_dict() != model_config.to_dict():
            print("Resuming with checkpoint model configuration instead of CLI overrides.")
            model_config = restored_model_config
            model = GPTLanguageModel(model_config).to(device)
            if training_config.compile_model and hasattr(torch, "compile"):
                model = torch.compile(model)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=training_config.learning_rate,
                weight_decay=training_config.weight_decay,
                betas=(0.9, 0.95),
            )
            model_config.save_json(output_dir / "model_config.json")

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_step = int(checkpoint["step"]) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", math.inf))
        tokenizer_model_path = Path(checkpoint.get("tokenizer_model", tokenizer_model_path))
        tokenizer = ChatTokenizer(tokenizer_model_path)
        print(f"Resumed from checkpoint: {checkpoint_path}")

    token_ids = tokenizer.encode(load_text(training_config.prepared_path))
    dataset = PackedTokenDataset(token_ids, block_size=model_config.block_size, val_ratio=training_config.val_ratio)

    effective_batch_tokens = (
        training_config.batch_size
        * training_config.gradient_accumulation_steps
        * model_config.block_size
    )
    steps_per_epoch = max(1, dataset.num_train_tokens // max(1, effective_batch_tokens))
    if training_config.epochs > 0:
        training_config.max_steps = max(training_config.max_steps, steps_per_epoch * training_config.epochs)

    formatter = ConversationFormatter(default_system_prompt=resolved_system_prompt)
    generation_config = GenerationConfig(
        max_new_tokens=training_config.sample_max_new_tokens,
        temperature=float(preset_generation_defaults["temperature"]),
        top_k=int(preset_generation_defaults["top_k"]),
        top_p=float(preset_generation_defaults["top_p"]),
        repetition_penalty=float(preset_generation_defaults["repetition_penalty"]),
        system_preset=training_config.system_preset,
        system_prompt=resolved_system_prompt,
        knowledge_index_path=training_config.knowledge_index_path,
        retrieval_top_k=training_config.retrieval_top_k,
    )
    training_config.save_json(output_dir / "training_config.json")

    use_amp = training_config.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"Device: {device}")
    print(f"Prepared dataset mode: {prepared_metadata['mode']}")
    if "counts" in prepared_metadata:
        print(f"Prepared counts: {prepared_metadata['counts']}")
    else:
        print(f"Prepared samples: {prepared_metadata['num_samples']}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Train tokens: {dataset.num_train_tokens} | Val tokens: {dataset.num_val_tokens}")
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Model preset: {training_config.model_preset}")
    print(f"Gradient accumulation: {training_config.gradient_accumulation_steps}")
    print(f"AMP enabled: {use_amp}")
    print(f"Training for {training_config.max_steps} steps")

    training_start = time.time()
    latest_train_loss = math.inf

    for step in range(start_step, training_config.max_steps):
        lr = lr_with_warmup_and_cosine_decay(
            step=step,
            max_steps=training_config.max_steps,
            max_lr=training_config.learning_rate,
            min_lr=training_config.min_learning_rate,
            warmup_steps=training_config.warmup_steps,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0

        for _micro_step in range(training_config.gradient_accumulation_steps):
            x, y = dataset.get_batch("train", training_config.batch_size, device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                _, loss = model(x, y)
                loss = loss / training_config.gradient_accumulation_steps
            accumulated_loss += float(loss.item())
            scaler.scale(loss).backward()

        latest_train_loss = accumulated_loss

        if training_config.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step % training_config.log_interval == 0:
            print(
                f"step {step:>6} | train loss {latest_train_loss:.4f} | lr {lr:.6f} | elapsed {elapsed_since(training_start)}"
            )
            append_jsonl(
                metrics_path,
                {
                    "type": "train",
                    "step": step,
                    "loss": latest_train_loss,
                    "learning_rate": lr,
                },
            )

        if step % training_config.eval_interval == 0 or step == training_config.max_steps - 1:
            losses = estimate_loss(
                model=model,
                dataset=dataset,
                batch_size=training_config.batch_size,
                eval_batches=training_config.eval_batches,
                device=device,
                use_amp=use_amp,
            )
            best_val_loss = min(best_val_loss, losses["val"])
            print(
                f"eval step {step:>6} | train {losses['train']:.4f} | val {losses['val']:.4f} | best val {best_val_loss:.4f}"
            )
            append_jsonl(
                metrics_path,
                {
                    "type": "eval",
                    "step": step,
                    "train_loss": losses["train"],
                    "val_loss": losses["val"],
                    "best_val_loss": best_val_loss,
                },
            )

        if step % training_config.sample_interval == 0:
            sample_prompt_ids, _, _ = build_chat_input_ids(
                tokenizer=tokenizer,
                formatter=formatter,
                system_prompt=generation_config.system_prompt,
                messages=[{"role": "user", "content": training_config.sample_prompt}],
                block_size=model_config.block_size,
                max_new_tokens=generation_config.max_new_tokens,
            )
            prompt_tensor = torch.tensor([sample_prompt_ids], dtype=torch.long, device=device)
            model.eval()
            with torch.no_grad():
                sampled = model.generate(
                    prompt_tensor,
                    max_new_tokens=generation_config.max_new_tokens,
                    temperature=generation_config.temperature,
                    top_k=generation_config.top_k,
                    top_p=generation_config.top_p,
                    repetition_penalty=generation_config.repetition_penalty,
                    stop_token_ids=tokenizer.default_stop_ids,
                )
            model.train()
            sample_reply = extract_assistant_reply(
                tokenizer,
                sampled[0].tolist(),
                prompt_length=len(sample_prompt_ids),
                stop_ids=tokenizer.default_stop_ids,
            )
            print(f"sample step {step:>6} | user: {training_config.sample_prompt}")
            print(f"assistant: {sample_reply or '[empty reply]'}")
            append_jsonl(
                metrics_path,
                {
                    "type": "sample",
                    "step": step,
                    "prompt": training_config.sample_prompt,
                    "reply": sample_reply,
                },
            )

        should_save = step % training_config.save_interval == 0 or step == training_config.max_steps - 1
        if should_save:
            save_training_checkpoint(
                model=model,
                optimizer=optimizer,
                model_config=model_config,
                training_config=training_config,
                tokenizer_model_path=tokenizer_model_path,
                output_dir=output_dir,
                step=step,
                best_val_loss=best_val_loss,
                latest_train_loss=latest_train_loss,
            )
            print(f"checkpoint saved at step {step}: {output_dir / 'latest.pt'}")

    print(f"Training complete in {elapsed_since(training_start)}")
    print(f"Final checkpoint: {output_dir / 'latest.pt'}")


if __name__ == "__main__":
    main()
