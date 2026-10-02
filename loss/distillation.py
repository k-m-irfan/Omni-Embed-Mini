"""Self-distillation loss from frozen text backbone.

The frozen backbone (Qwen3-Embedding / Qwen3-VL-Embedding) is already a
strong text embedding model.
The text-only path produces high-quality embeddings that serve as soft
targets for the media path.

Loss: cosine distillation at each MRL dimension — forces media embeddings
to match the text embedding at every granularity level.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """MRL-aware cosine distillation from text embeddings.

    For each MRL dimension, computes:
        loss_d = 1 - cosine_sim(media_emb[:d], text_emb[:d])

    Weighted by 1/sqrt(d) like the SigLIP MRL loss.

    Args:
        dims: list of int — MRL dimensions to distill at
    """

    def __init__(self, dims):
        super().__init__()
        self.dims = dims
        # Same weighting scheme as MRL loss
        import math
        raw = {d: 1.0 / math.sqrt(d) for d in dims}
        total = sum(raw.values())
        self.weights = {d: w / total for d, w in raw.items()}

    def forward(self, media_emb, text_target):
        """Compute distillation loss.

        Args:
            media_emb: (B, max_dim) — media embedding (gradients flow through this)
            text_target: (B, max_dim) — frozen text embedding (detached, no gradients)

        Returns:
            loss: scalar
            per_dim: dict {dim: float} for logging
        """
        total_loss = torch.tensor(0.0, device=media_emb.device)
        per_dim = {}

        for d in self.dims:
            m = F.normalize(media_emb[:, :d], p=2, dim=-1)
            t = F.normalize(text_target[:, :d], p=2, dim=-1)
            if t.dtype != m.dtype:
                t = t.to(m.dtype)
            cosine_sim = (m * t).sum(dim=-1)  # (B,)
            loss_d = (1.0 - cosine_sim).mean()
            total_loss = total_loss + self.weights[d] * loss_d
            per_dim[d] = loss_d.item()

        return total_loss, per_dim
