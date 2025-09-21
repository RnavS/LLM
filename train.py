from utils import load_data, make_vocab, count_char_freq, preview_vocab, encode, decode, clean_text, restore_text, advanced_clean_text
from model import TinyTransformer, block_size, embedding_dim
import os
import torch
import pickle

start_step = 0
checkpoint_path = None
raw_text = load_data("cleaned_data.txt")
text = advanced_clean_text(raw_text)
stringtoindex, indextostring = make_vocab(text)
preview_vocab(stringtoindex)

encoded_text = encode(text, stringtoindex)
print("First 20 tokens:", encoded_text[:20])
print("Decoded preview:", decode(encoded_text[:20], indextostring))

vocab_size = len(stringtoindex)
model = TinyTransformer(vocab_size, embedding_dim, block_size)
data = torch.tensor(encoded_text, dtype=torch.long)

def get_batch(data, batch_size=4, block_size=block_size):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y

# model.load_state_dict(torch.load(checkpoint_path))
print(f"Resumed from checkpoint: {checkpoint_path}")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

for step in range(start_step, 300010):
    xb, yb = get_batch(data)
    logits = model(xb)
    B, T, V = logits.shape
    loss = loss_fn(logits.view(B * T, V), yb.view(B * T))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f"Step {step} | Loss: {loss.item():.4f}")
    
    if step % 10000 == 0 and step != 0:
        checkpoint_path = f"model_step{step}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")


with open("vocab.pkl", "wb") as f:
    pickle.dump((stringtoindex, indextostring), f)

with open("last_checkpoint_step.txt", "w") as f:
    f.write(str(step))

torch.save(model.state_dict(), "model.pth")
