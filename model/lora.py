"""LoRA (Low-Rank Adaptation) for encoder and backbone fine-tuning."""

import torch.nn as nn


class LoRALinear(nn.Module):
    """Drop-in nn.Linear replacement with low-rank adaptation.

    Exposes weight/bias/in_features/out_features from the original Linear
    so that modules like nn.MultiheadAttention that access out_proj.weight
    directly continue to work.
    """

    def __init__(self, original, r=16, alpha=32, dropout=0.05):
        super().__init__()
        self.original = original
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.r, self.alpha = r, alpha
        self.scaling = alpha / r
        # Inherit DEVICE from the wrapped linear so the adapters land on
        # the same GPU as the parent module (injection often happens
        # mid-training, after accelerate.prepare has moved the model to
        # GPU — a fresh nn.Linear would default to CPU and crash on the
        # first forward). Keep dtype at float32 for numerical stability
        # during optimizer updates; autocast will cast forward to bf16.
        device = original.weight.device
        self.lora_A = nn.Linear(original.in_features, r, bias=False, device=device)
        self.lora_B = nn.Linear(r, original.out_features, bias=False, device=device)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

    @property
    def weight(self):
        return self.original.weight

    @property
    def bias(self):
        return self.original.bias

    def forward(self, x):
        return self.original(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


def inject_lora(module, target_suffixes, r=16, alpha=32, dropout=0.05):
    """Replace nn.Linear layers whose full dotted name ends with a target suffix."""
    for name, child in list(module.named_modules()):
        if isinstance(child, nn.Linear) and any(name.endswith(t) for t in target_suffixes):
            parts = name.split(".")
            parent = module
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], LoRALinear(child, r, alpha, dropout))
