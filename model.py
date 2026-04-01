from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


def top_k_top_p_filter(logits: torch.Tensor, top_k: int | None, top_p: float | None) -> torch.Tensor:
    filtered = logits.clone()

    if top_k is not None and top_k > 0:
        top_k = min(top_k, filtered.size(-1))
        threshold = torch.topk(filtered, k=top_k, dim=-1).values[..., -1, None]
        filtered = torch.where(filtered < threshold, torch.full_like(filtered, float("-inf")), filtered)

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        remove_mask = cumulative_probs > top_p
        remove_mask[..., 1:] = remove_mask[..., :-1].clone()
        remove_mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))

        filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

    return filtered


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head.")

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_dim = config.n_embd // config.n_head

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.block_size, config.block_size)).view(
            1, 1, config.block_size, config.block_size
        )
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, channels = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        k = k.view(batch_size, sequence_length, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(batch_size, sequence_length, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, sequence_length, self.n_head, self.head_dim).transpose(1, 2)

        attention = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        attention = attention.masked_fill(self.causal_mask[:, :, :sequence_length, :sequence_length] == 0, float("-inf"))
        attention = F.softmax(attention, dim=-1)
        attention = self.attn_dropout(attention)

        output = attention @ v
        output = output.transpose(1, 2).contiguous().view(batch_size, sequence_length, channels)
        return self.resid_dropout(self.c_proj(output))


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden_size = 4 * config.n_embd
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, hidden_size, bias=config.bias),
            nn.GELU(),
            nn.Linear(hidden_size, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
                "ln_f": nn.LayerNorm(config.n_embd),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer["wte"].weight = self.lm_head.weight

        self.apply(self._init_weights)
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, sequence_length = idx.shape
        if sequence_length > self.config.block_size:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds block size {self.config.block_size}."
            )

        positions = torch.arange(0, sequence_length, device=idx.device, dtype=torch.long)
        token_embeddings = self.transformer["wte"](idx)
        position_embeddings = self.transformer["wpe"](positions)[None, :, :]
        x = self.transformer["drop"](token_embeddings + position_embeddings)

        for block in self.transformer["h"]:
            x = block(x)
        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        stop_token_ids: Sequence[int] | None = None,
    ) -> torch.Tensor:
        generated = idx
        stop_token_ids = set(stop_token_ids or [])

        for _ in range(max_new_tokens):
            idx_cond = generated[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            next_token_logits = logits[:, -1, :]

            if repetition_penalty and repetition_penalty != 1.0:
                for row_index in range(generated.size(0)):
                    seen_tokens = torch.unique(generated[row_index])
                    negative_mask = next_token_logits[row_index, seen_tokens] < 0
                    next_token_logits[row_index, seen_tokens[negative_mask]] *= repetition_penalty
                    next_token_logits[row_index, seen_tokens[~negative_mask]] /= repetition_penalty

            if temperature <= 0:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                scaled_logits = next_token_logits / max(temperature, 1e-5)
                filtered_logits = top_k_top_p_filter(scaled_logits, top_k=top_k, top_p=top_p)
                probabilities = F.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)

            generated = torch.cat((generated, next_token), dim=1)

            if stop_token_ids and all(token.item() in stop_token_ids for token in next_token.view(-1)):
                break

        return generated

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
