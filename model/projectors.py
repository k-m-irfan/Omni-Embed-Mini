"""
Projectors for Omni-Embed.

ConvDownProjector: Conv1d temporal downsampling + 2-layer SwiGLU (used by
the audio path to collapse encoder frames to a fixed token budget).
"""

import torch.nn as nn
import torch.nn.functional as F


class ConvDownProjector(nn.Module):
    """Conv1d temporal downsampling + 2-layer SwiGLU projection.

    Reduces token count via strided convolution, then projects through
    two gated linear units with residual connection.

    Architecture:
        Conv1d(stride) -> LayerNorm -> SwiGLU -> LayerNorm -> SwiGLU (residual) -> LayerNorm
    """

    def __init__(self, input_dim, output_dim, input_tokens, target_tokens, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = output_dim * 3
        self.stride = max(1, input_tokens // target_tokens)
        self.conv = nn.Conv1d(
            input_dim, input_dim,
            kernel_size=self.stride * 2 + 1,
            stride=self.stride,
            padding=self.stride,
        )
        self.norm = nn.LayerNorm(input_dim)
        self.gate_up1 = nn.Linear(input_dim, hidden_dim * 2, bias=False)
        self.down1 = nn.Linear(hidden_dim, output_dim, bias=False)
        self.mid_norm = nn.LayerNorm(output_dim)
        self.gate_up2 = nn.Linear(output_dim, hidden_dim * 2, bias=False)
        self.down2 = nn.Linear(hidden_dim, output_dim, bias=False)
        self.out_norm = nn.LayerNorm(output_dim)
        self.target_tokens = target_tokens

    def forward(self, x):
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm(x)
        if x.shape[1] > self.target_tokens:
            x = x[:, :self.target_tokens, :]
        elif x.shape[1] < self.target_tokens:
            x = F.pad(x, (0, 0, 0, self.target_tokens - x.shape[1]))
        gv = self.gate_up1(x)
        gate, value = gv.chunk(2, dim=-1)
        x = self.down1(F.silu(gate) * value)
        x = self.mid_norm(x)
        gv = self.gate_up2(x)
        gate, value = gv.chunk(2, dim=-1)
        x = x + self.down2(F.silu(gate) * value)
        x = self.out_norm(x)
        return x
