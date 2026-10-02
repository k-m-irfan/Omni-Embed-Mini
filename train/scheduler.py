"""Learning rate schedulers for Omni-Embed training."""

import math
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.0):
    """Cosine learning rate schedule with linear warmup.

    Args:
        optimizer: torch optimizer
        warmup_steps: int — linear warmup steps
        total_steps: int — total training steps
        min_lr_ratio: float — minimum LR as fraction of peak LR

    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)

    return LambdaLR(optimizer, lr_lambda)
