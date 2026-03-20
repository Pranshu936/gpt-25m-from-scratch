import config

# load text
with open(config.data_path, "r", encoding="utf-8") as f:
    text = f.read()

# -------------------------
# BUILD VOCABULARY
# -------------------------

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# -------------------------
# ENCODE / DECODE
# -------------------------

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

# -------------------------
# TEST
# -------------------------

print("Vocabulary built!")
print("Vocab size:", vocab_size)

sample = "hello"
encoded = encode(sample)
decoded = decode(encoded)

print("\nSample test:")
print("Text:", sample)
print("Encoded:", encoded)
print("Decoded:", decoded)
