"""SigLIP pairwise sigmoid contrastive loss.

Each (anchor, candidate) pair is scored independently via sigmoid — no
softmax normalization. Positive pairs are pushed toward +1, negatives
toward 0. Every pair contributes equally to the gradient, unlike InfoNCE
where the hardest negative dominates.

Reference: Zhai et al., "Sigmoid Loss for Language Image Pre-Training", 2023.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class SigLIPLoss(nn.Module):
    """SigLIP contrastive loss with learnable temperature and bias.

    Args:
        temperature_init: float — initial temperature (divisor, default 0.07)
        bias_init: float — initial logit bias (default -10.0)
    """

    def __init__(self, temperature_init=0.07, bias_init=-10.0):
        super().__init__()
        self.log_temperature = nn.Parameter(
            torch.tensor(temperature_init).log()
        )
        self.bias = nn.Parameter(torch.tensor(bias_init))

    @property
    def temperature(self):
        """Current temperature value (clamped)."""
        return self.log_temperature.exp().clamp(0.01, 0.3)

    def forward(self, anchor_emb, positive_emb, hard_negatives=None):
        """Compute SigLIP loss.

        Args:
            anchor_emb: (B, D) — L2-normalized anchor embeddings
            positive_emb: (B, D) — L2-normalized positive embeddings
            hard_negatives: optional (B, K, D) — L2-normalized hard negative embeddings

        Returns:
            loss: scalar tensor
        """
        temp = self.temperature
        B = anchor_emb.shape[0]

        # Force matmul operands to a common dtype. Under bf16 autocast,
        # F.normalize upstream can promote one of (anchor, positive) to
        # float32 while the cache-sourced teacher stays bf16 — the @
        # kernel rejects mixed dtypes.
        if positive_emb.dtype != anchor_emb.dtype:
            positive_emb = positive_emb.to(anchor_emb.dtype)

        # Pairwise similarity: (B, B)
        logits = anchor_emb @ positive_emb.T / temp + self.bias

        # Labels: +1 for diagonal (matched pairs), -1 for off-diagonal (negatives)
        labels = 2 * torch.eye(B, device=logits.device, dtype=logits.dtype) - 1

        # SigLIP: -log σ(label * logit) for each pair
        loss = -F.logsigmoid(labels * logits)

        # Hard negatives: all are negative pairs
        if hard_negatives is not None and hard_negatives.shape[1] > 0:
            if hard_negatives.dtype != anchor_emb.dtype:
                hard_negatives = hard_negatives.to(anchor_emb.dtype)
            hn_logits = torch.bmm(
                anchor_emb.unsqueeze(1),
                hard_negatives.transpose(1, 2),
            ).squeeze(1) / temp + self.bias  # (B, K)
            loss = torch.cat([loss, -F.logsigmoid(-hn_logits)], dim=1)

        return loss.mean()


def gather_embeddings(emb):
    """All-gather embeddings across DDP processes.

    Args:
        emb: (B, D) — local embeddings

    Returns:
        (world_size * B, D) — gathered embeddings from all processes
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return emb

    world_size = dist.get_world_size()
    gathered = [torch.zeros_like(emb) for _ in range(world_size)]
    dist.all_gather(gathered, emb.contiguous())

    # Replace own rank's entry with the original (preserves gradient)
    rank = dist.get_rank()
    gathered[rank] = emb

    return torch.cat(gathered, dim=0)
