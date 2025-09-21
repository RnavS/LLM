from model import TinyTransformer, block_size, embedding_dim
from utils import load_data, make_vocab, encode, decode, restore_text
import torch
import pickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open("vocab.pkl", "rb") as f:
    stringtoindex, indextostring = pickle.load(f)

vocab_size = len(stringtoindex)
model = TinyTransformer(vocab_size, embedding_dim, block_size).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

@torch.inference_mode()
def generate(model, prompt, max_tokens=150, temperature=0.8, top_k=40, top_p=0.9, repetition_penalty=1.2):
    context = torch.tensor(encode(prompt, stringtoindex), dtype=torch.long).unsqueeze(0).to(device)
    generated = context

    for _ in range(max_tokens):
        idx_condensed = generated[:, -block_size:]
        logits = model(idx_condensed)
        logits = logits[:, -1, :] / temperature

        for token_id in set(generated[0].tolist()):
            logits[0, token_id] /= repetition_penalty

        probs = torch.softmax(logits, dim=-1)

        if top_k:
            top_probs, top_idx = torch.topk(probs, k=top_k)
            probs = torch.zeros_like(probs).scatter(1, top_idx, top_probs)
            probs /= probs.sum(dim=-1, keepdim=True)

        if top_p and top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cumulative_probs > top_p

            if torch.any(cutoff):
                cutoff_idx = cutoff.nonzero()[0][0] + 1
                sorted_probs = sorted_probs[:, :cutoff_idx]
                sorted_idx = sorted_idx[:, :cutoff_idx]
                probs = torch.zeros_like(probs).scatter(1, sorted_idx, sorted_probs)
                probs /= probs.sum(dim=-1, keepdim=True)

        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)

    raw_output = decode(generated[0].tolist(), indextostring)
    return restore_text(raw_output.split())

prompt = input("Enter a prompt: ")
output = generate(model, prompt, max_tokens=150, temperature=0.6, top_k=40, top_p=0.9, repetition_penalty=1.1)

print("\n--- Generated Output ---\n")
print(output)

with open("output.txt", "w") as f:
    f.write(f"Prompt: {prompt}\n\nGenerated:\n{output}\n")
