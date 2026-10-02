"""Matryoshka Representation Learning (MRL) loss.

Slices the backbone's L2-normalized, hidden_size-dim pooled vector at each
coarse MRL dimension and runs SigLIP independently at every slice. Smaller
dimensions get higher weight (1/sqrt(d)) since they're harder to compress to.

Optionally adds a cosine self-distillation term that pulls the student
(media path) toward the teacher (frozen-backbone caption path) at every MRL
dimension — same weighting scheme.

There is no projection head: MRL_DIMS are capped at the backbone's native
hidden_size (1024 for the 0.9B / Qwen3-Embedding-0.6B, 2048 for the
2.3B / Qwen3-VL-Embedding-2B).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .siglip import SigLIPLoss, gather_embeddings


# Coarse MRL dim grid. Max dim == backbone hidden_size.
MRL_DIMS = [128, 256, 512, 1024]


class MRLLoss(nn.Module):
    """Matryoshka contrastive loss with optional self-distillation.

    Args:
        dims: list of int — MRL dimensions to evaluate at
        temperature_init: float — SigLIP starting temperature
        gather_distributed: bool — all_gather across ranks before loss
        distill_weight: float — weight for cosine distillation (0 disables)
    """

    def __init__(
        self,
        dims=None,
        temperature_init=0.07,
        gather_distributed=True,
        distill_weight=0.0,
    ):
        super().__init__()
        self.dims = dims or MRL_DIMS
        self.gather_distributed = gather_distributed
        self.distill_weight = distill_weight

        # Shared SigLIP (shared temperature + bias across dims)
        self.siglip = SigLIPLoss(temperature_init)

        if distill_weight > 0:
            from .distillation import DistillationLoss
            self.distillation = DistillationLoss(self.dims)
        else:
            self.distillation = None

        # 1/sqrt(d) weighting, normalized to sum to 1
        raw = {d: 1.0 / math.sqrt(d) for d in self.dims}
        total = sum(raw.values())
        self.dim_weights = {d: w / total for d, w in raw.items()}

    def forward(self, anchor_full, positive_full, hard_negatives=None,
                distill_target=None):
        """Compute MRL contrastive loss + optional distillation.

        Args:
            anchor_full: (B, max_dim) — student (media) embeddings
            positive_full: (B, max_dim) — teacher (cascaded caption) embeddings
            hard_negatives: optional (B, K, max_dim)
            distill_target: optional (B, max_dim) — detached teacher target

        Returns:
            total_loss: scalar
            per_dim_losses: dict keyed by dim (+ "distill") for logging
        """
        if self.gather_distributed:
            anchor_full = gather_embeddings(anchor_full)
            positive_full = gather_embeddings(positive_full)
            if hard_negatives is not None:
                hard_negatives = gather_embeddings(
                    hard_negatives.reshape(-1, hard_negatives.shape[-1])
                ).reshape(-1, hard_negatives.shape[1], hard_negatives.shape[2])

        total_loss = torch.tensor(0.0, device=anchor_full.device)
        per_dim_losses = {}

        for d in self.dims:
            a = F.normalize(anchor_full[:, :d], p=2, dim=-1)
            p = F.normalize(positive_full[:, :d], p=2, dim=-1)

            hn = None
            if hard_negatives is not None and hard_negatives.shape[-1] >= d:
                hn = F.normalize(hard_negatives[:, :, :d], p=2, dim=-1)

            loss_d = self.siglip(a, p, hn)
            total_loss = total_loss + self.dim_weights[d] * loss_d
            per_dim_losses[d] = loss_d.item()

        if self.distillation is not None and distill_target is not None:
            distill_loss, _ = self.distillation(anchor_full, distill_target)
            total_loss = total_loss + self.distill_weight * distill_loss
            per_dim_losses["distill"] = distill_loss.item()

        return total_loss, per_dim_losses
