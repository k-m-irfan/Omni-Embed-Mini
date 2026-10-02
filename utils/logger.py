"""Training logger for Omni-Embed."""

import time
from collections import defaultdict


def _format_eta(seconds):
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    hours = seconds / 3600
    if hours < 24:
        return f"{hours:.1f}h"
    return f"{hours / 24:.1f}d"


class TrainingLogger:
    """Logs training metrics with progress, ETA, per-modality loss, and optional wandb.

    Tracks running average loss per modality so you can see which modalities
    are learning and which are stuck.
    """

    def __init__(self, accelerator, log_every=10, use_wandb=False,
                 total_steps=0, phase_switch_step=0):
        self.accelerator = accelerator
        self.log_every = log_every
        self.use_wandb = use_wandb
        self.total_steps = total_steps
        self.phase_switch_step = phase_switch_step
        # _train_start is lazy: set on the first real training step so the
        # teacher-cache build / other one-time startup work doesn't leak
        # into the ETA rate computation.
        self._train_start = None
        self._train_start_step = 0
        self._step_start = time.time()
        self._loss_accum = 0.0
        self._count = 0
        # Per-modality EMA loss. Kept across log windows so every wandb
        # step emits a value for EVERY modality we've seen, not just the
        # one in this batch. Otherwise each modality's series is sparse
        # (one point per batch of that modality) and wandb UI hides it.
        self._mod_ema = {}
        self._mod_ema_alpha = 0.1  # new-sample weight
        self._mod_last = {}        # most recent raw batch loss per modality
        # Window counters (for the per-modality string printed to stdout)
        self._mod_loss = defaultdict(float)
        self._mod_count = defaultdict(int)

    def log_step(self, step, loss, per_dim_losses, lr, temperature, modality, phase,
                 cache_hit_rate=None):
        """Log a training step."""
        # Lazy-start the training clock on the first real step we see.
        if self._train_start is None:
            self._train_start = time.time()
            self._train_start_step = step

        self._loss_accum += loss
        self._count += 1
        self._mod_loss[modality] += loss
        self._mod_count[modality] += 1
        # Update per-modality EMA (seeded on first observation)
        prev = self._mod_ema.get(modality)
        if prev is None:
            self._mod_ema[modality] = loss
        else:
            a = self._mod_ema_alpha
            self._mod_ema[modality] = (1 - a) * prev + a * loss
        self._mod_last[modality] = loss

        if step % self.log_every != 0 or not self.accelerator.is_main_process:
            return

        elapsed = time.time() - self._step_start
        total_elapsed = time.time() - self._train_start
        avg_loss = self._loss_accum / max(self._count, 1)
        steps_per_sec = self._count / max(elapsed, 0.01)

        # Progress and ETA — use the time/steps elapsed since training
        # actually started (not wall-clock since Logger construction).
        pct = 100 * step / max(self.total_steps, 1)
        steps_since_start = max(1, step - self._train_start_step)
        if total_elapsed > 0:
            eta_sec = (self.total_steps - step) * (total_elapsed / steps_since_start)
            eta_str = _format_eta(eta_sec)
        else:
            eta_str = "?"

        # Dim losses — show subset to keep readable
        dim_keys = sorted(k for k in per_dim_losses if isinstance(k, int))
        if len(dim_keys) > 8:
            show_dims = [dim_keys[0], dim_keys[len(dim_keys)//2], dim_keys[-1]]
            show_dims = sorted(set(show_dims + [d for d in dim_keys if d >= 128]))
        else:
            show_dims = dim_keys
        dim_str = " ".join(f"d{d}={per_dim_losses[d]:.3f}" for d in show_dims)

        # Distillation term
        extra_str = ""
        if "distill" in per_dim_losses:
            extra_str += f" distill={per_dim_losses['distill']:.4f}"

        # Cache hit rate
        cache_str = ""
        if cache_hit_rate is not None and cache_hit_rate > 0:
            cache_str = f" hn={cache_hit_rate:.0%}"

        # Per-modality average losses
        mod_str = " ".join(
            f"{m}={self._mod_loss[m]/max(self._mod_count[m],1):.3f}"
            for m in sorted(self._mod_loss.keys())
        )

        print(
            f"[{step:>6d}/{self.total_steps} {pct:4.1f}% ETA:{eta_str}] "
            f"loss={avg_loss:.4f} | {dim_str}{extra_str} | "
            f"lr={lr:.2e} temp={temperature:.4f} | "
            f"mod={modality} P{phase}{cache_str} | "
            f"{steps_per_sec:.1f} it/s"
        )
        if mod_str:
            print(f"  modality losses: {mod_str}")

        # Wandb
        if self.use_wandb:
            import wandb
            log_dict = {
                "loss": avg_loss,
                "lr": lr,
                "temperature": temperature,
                "phase": phase,
                "steps_per_sec": steps_per_sec,
                "progress_pct": pct,
            }
            for d, l in per_dim_losses.items():
                log_dict[f"loss_dim/{d}"] = l
            # Per-modality loss to wandb — log EVERY modality seen so far,
            # using the EMA-smoothed value. Every step thus carries a value
            # for every known modality, so wandb draws a dense curve per
            # modality even when a batch touches only one of them.
            for m, v in self._mod_ema.items():
                log_dict[f"loss_mod/{m}"] = v
            if cache_hit_rate is not None:
                log_dict["mining/cache_hit_rate"] = cache_hit_rate
            wandb.log(log_dict, step=step)

        # Reset accumulators
        self._loss_accum = 0.0
        self._count = 0
        self._mod_loss.clear()
        self._mod_count.clear()
        self._step_start = time.time()
