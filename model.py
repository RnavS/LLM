import torch
import torch.nn as nn
import torch.nn.functional as F

block_size = 16
embedding_dim = 512
vocab_size = None 

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        assert embedding_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.qkv_proj = nn.Linear(embedding_dim, embedding_dim * 3)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.register_buffer("causal_mask", torch.tril(torch.ones(block_size, block_size)).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv_proj(x).view(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, T, C))

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_hidden_dim):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embedding_dim, num_heads)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embedding_dim)
        )
        self.ln2 = nn.LayerNorm(embedding_dim)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, block_size, num_heads = 4, num_layers = 4):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding_table = nn.Embedding(block_size, embedding_dim)
        self.blocks = nn.Sequential(*[
            TransformerBlock(embedding_dim, num_heads, embedding_dim * 4)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.position_embedding_table.num_embeddings, \
            f"Input sequence too long: got {T}, max {self.position_embedding_table.num_embeddings}"
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx
