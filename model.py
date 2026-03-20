import torch
import torch.nn as nn
import config
from tokenizer import vocab_size

class GPT(nn.Module):
    def __init__(self):
        super().__init__()

        # -------------------------
        # TOKEN EMBEDDING
        # -------------------------
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=config.n_embd
        )

        # -------------------------
        # POSITION EMBEDDING
        # -------------------------
        self.position_embedding = nn.Embedding(
            num_embeddings=config.block_size,
            embedding_dim=config.n_embd
        )

    def forward(self, idx):
        B, T = idx.shape

        # token embeddings
        tok_emb = self.token_embedding(idx)  # (B, T, n_embd)

        # position embeddings
        pos = torch.arange(T, device=config.device)
        pos_emb = self.position_embedding(pos)  # (T, n_embd)

        # combine
        x = tok_emb + pos_emb

        return x
