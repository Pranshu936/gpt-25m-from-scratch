
# =========================
# CONFIG FILE (SMART VERSION - FIXED BLOCK_SIZE)
# =========================

import torch
import random

# -------------------------
# SEED (for reproducibility)
# -------------------------
seed = 42
torch.manual_seed(seed)
random.seed(seed)

# -------------------------
# DEVICE
# -------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -------------------------
# DATA
# -------------------------
batch_size = 8
block_size = 8  # Adjusted to be smaller than the smallest data split (val_data length 10)

# -------------------------
# MODEL (25M TARGET)
# -------------------------
n_layer = 8
n_head = 6
n_embd = 384

# -------------------------
# DERIVED VALUES (IMPORTANT)
# -------------------------
head_dim = n_embd // n_head

# -------------------------
# TRAINING
# -------------------------
learning_rate = 3e-4
max_steps = 5000

# -------------------------
# EVAL
# -------------------------
eval_interval = 200
eval_iters = 50

# -------------------------
# DROPOUT
# -------------------------
dropout = 0.1

# -------------------------
# PATHS
# -------------------------
data_path = "data/input.txt"
checkpoint_path = "checkpoints/model.pt"

# -------------------------
# SANITY CHECKS (VERY IMPORTANT)
# -------------------------
assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
