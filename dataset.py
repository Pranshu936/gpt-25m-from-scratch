import torch
import config
from tokenizer import encode

# -------------------------
# LOAD TEXT
# -------------------------

with open(config.data_path, "r", encoding="utf-8") as f:
    text = f.read()

# -------------------------
# ENCODE FULL DATASET
# -------------------------

data = torch.tensor(encode(text), dtype=torch.long)

# -------------------------
# TRAIN / VAL SPLIT
# -------------------------

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# -------------------------
# GET BATCH FUNCTION
# -------------------------

def get_batch(split):
    data = train_data if split == 'train' else val_data
    
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    
    x = torch.stack([data[i:i+config.block_size] for i in ix])
    y = torch.stack([data[i+1:i+config.block_size+1] for i in ix])
    
    return x.to(config.device), y.to(config.device)

# -------------------------
# TEST
# -------------------------

x, y = get_batch("train")

print("Batch created!")
print("x shape:", x.shape)
print("y shape:", y.shape)

print("\nSample x[0]:", x[0])
print("Sample y[0]:", y[0])
