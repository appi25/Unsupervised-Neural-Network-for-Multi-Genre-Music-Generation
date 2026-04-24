"""Task 3: Music Transformer for long-horizon autoregressive generation.

Architecture (from project specification):
    p(X) = prod_t p(x_t | x_<t)
    L_TR = -sum_t log p_theta(x_t | x_<t)
    Perplexity = exp(L_TR / T)
"""
import torch
import torch.nn as nn


class MusicTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6,
                 dim_feedforward=1024, max_seq_len=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        sz = x.size(1)
        mask = torch.triu(
            torch.ones(sz, sz) * float('-inf'), diagonal=1).to(x.device)
        x = self.embedding(x) + self.pos_encoding[:, :sz, :]
        output = self.transformer(x, mask=mask)
        return self.fc_out(output)
