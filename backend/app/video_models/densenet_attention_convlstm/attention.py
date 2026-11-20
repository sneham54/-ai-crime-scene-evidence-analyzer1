import torch
from torch import nn


class TemporalMultiHeadSelfAttention(nn.Module):
    """Self-attention over temporal feature tokens (B,T,C), preserving shape."""
    def __init__(self, channels, heads=4, dropout=0.1):
        super().__init__()
        if channels % heads:
            raise ValueError("channels must be divisible by heads")
        self.norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.output_norm = nn.LayerNorm(channels)

    def forward(self, sequence):
        if sequence.ndim != 3:
            raise ValueError("attention input must have shape (batch, time, channels)")
        normalized = self.norm(sequence)
        attended, weights = self.attention(normalized, normalized, normalized, need_weights=False)
        return self.output_norm(sequence + attended), weights
