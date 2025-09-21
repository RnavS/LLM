import re
from collections import Counter

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def clean_text(text):
    text = text.lower()
    text = text.replace('\n', ' <newline> ')
    text = re.sub(r'\.', ' <period> <eos> ', text)
    text = re.sub(r'\!', ' <exclamation> <eos> ', text)
    text = re.sub(r'\?', ' <question> <eos> ', text)
    text = re.sub(r',', ' <comma> ', text)
    text = re.sub(r'["“”]', ' <quote> ', text)
    text = re.sub(r'[-–—]', ' <dash> ', text)
    return text

def restore_text(tokens):
    text = ' '.join(tokens)
    text = re.sub(r'\s*<period>\s*', '.', text)
    text = re.sub(r'\s*<comma>\s*', ',', text)
    text = re.sub(r'\s*<question>\s*', '?', text)
    text = re.sub(r'\s*<exclamation>\s*', '!', text)
    text = re.sub(r'\s*<quote>\s*', '"', text)
    text = re.sub(r'\s*<dash>\s*', '-', text)
    text = re.sub(r'\s*<newline>\s*', '\n', text)
    text = re.sub(r'\s*<eos>\s*', ' <eos> ', text)
    sentences = []
    
    for segment in text.split('<eos>'):
        segment = segment.strip()
        if segment:
            segment = segment[0].upper() + segment[1:] if len(segment) > 1 else segment.upper()
            sentences.append(segment)
    return ' '.join(sentences)


def advanced_clean_text(text):
    text = text.replace('“', '"').replace('”', '"').replace('’', "'").replace('‘', "'")
    text = text.replace("—", "-").replace("–", "-")
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # keep only ASCII
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\s*([.?!,;:])\s*', r'\1 ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    if text and text[-1] not in ".!?":
        text += '.'

    return text

def make_vocab(text):
    words = text.split()
    unique_words = sorted(set(words))
    stringtoindex = {word: i for i, word in enumerate(unique_words)}
    indextostring = {i: word for word, i in stringtoindex.items()}
    return stringtoindex, indextostring

def preview_vocab(token_map, count=10):
    print("Previewing first", count, "tokens:")
    for token, idx in list(token_map.items())[:count]:
        print(f"'{token}': {idx}")

def preprocess(text, mode="advanced"):
    if mode == "basic":
        return clean_text(text)
    return advanced_clean_text(text)

def count_char_freq(text):
    words = text.split()
    frequent = Counter(words)
    return frequent.most_common(10)

def encode(text, stringtoindex):
    return [stringtoindex[word] for word in text.split() if word in stringtoindex]

def decode(indices, indextostring):
    return ' '.join([indextostring[i] for i in indices])
