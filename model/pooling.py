"""EOS token pooling for embedding extraction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EOSPooling(nn.Module):
    """Extract the hidden state at the last non-padding token (EOS position).

    For embedding models, the EOS token captures the summary representation
    of the full input sequence via causal attention (the last token attends
    to every earlier token).

    The last actual token is determined by attention_mask, ensuring we never
    accidentally pool from a padding position.
    """

    def forward(self, last_hidden_state, attention_mask):
        """
        Args:
            last_hidden_state: (B, S, D) — backbone output
            attention_mask: (B, S) — 1 for real tokens, 0 for padding

        Returns:
            (B, D) — L2-normalized embedding at last real token
        """
        # Right-pad → col 0 is all 1s (real tokens start at 0).
        # Left-pad  → col 0 may have zeros; real tokens end at col -1.
        if bool(attention_mask[:, 0].all()):
            seq_lens = attention_mask.sum(dim=1, dtype=torch.long) - 1  # (B,)
            batch_idx = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
            eos_hidden = last_hidden_state[batch_idx, seq_lens]  # (B, D)
        else:
            eos_hidden = last_hidden_state[:, -1]  # (B, D)
        return F.normalize(eos_hidden, p=2, dim=-1)
