"""Checkpoint management for Omni-Embed training."""

import os
import json
import glob
import shutil

import torch


def _atomic_torch_save(obj, path):
    """Write a torch object durably: full write to a .tmp, fsync, then atomic
    os.replace onto the final path. A crash mid-write truncates only the .tmp;
    the real file is only ever swapped in from a fully-flushed temp, which
    prevents 'PytorchStreamReader failed ... central directory' corruption if
    the process dies mid-save. Assumes tmp and path share a directory (same
    filesystem)."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_json_dump(obj, path, **kw):
    """Atomic-rename counterpart for JSON state files."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, **kw)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class CheckpointManager:
    """Manages training checkpoints with best-N retention.

    Checkpoint structure:
        output_dir/
        ├── latest/
        │   ├── trainable_weights.pt
        │   └── optim_state.pt
        ├── best_step_1000/
        │   └── trainable_weights.pt
        └── training_state.json

    Args:
        output_dir: str — checkpoint directory
        keep_best: int — number of best checkpoints to retain
    """

    def __init__(self, output_dir, keep_best=3, num_replicas=None):
        self.output_dir = output_dir
        self.keep_best = keep_best
        # World size this run trains at; persisted into training_state.json so a
        # resume can refuse a mismatched relaunch (which would silently rescale
        # steps_per_epoch / phase_switch / LR schedule).
        self.num_replicas = num_replicas
        self.best_checkpoints = []  # list of (loss, step, path)
        os.makedirs(output_dir, exist_ok=True)

    def save(self, model, optimizer, scheduler, step, epoch, loss, best_loss,
             tag=None, miner_cache=None):
        """Save a checkpoint.

        Args:
            model: unwrapped OmniEmbedModel
            optimizer: optimizer
            scheduler: LR scheduler
            step: int — global step
            epoch: int — current epoch
            loss: float — current loss
            best_loss: float — best loss so far
            tag: str — optional tag for the checkpoint dir name
            miner_cache: optional NegativeCache — hard negative cache to persist
        """
        # Save latest
        latest_dir = os.path.join(self.output_dir, "latest")
        os.makedirs(latest_dir, exist_ok=True)

        trainable = model.get_trainable_state_dict()
        # Write optim FIRST (atomically), then weights: resume gates on
        # weights_path existing, so weights must be the last thing to land.
        _atomic_torch_save({
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }, os.path.join(latest_dir, "optim_state.pt"))
        _atomic_torch_save(trainable, os.path.join(latest_dir, "trainable_weights.pt"))

        # Save miner cache (so resume doesn't lose mining progress)
        if miner_cache is not None:
            cache_data = miner_cache._data.copy()
            _atomic_json_dump(cache_data, os.path.join(latest_dir, "miner_cache.json"))

        # Save training state
        state = {
            "global_step": step,
            "epoch": epoch,
            "num_replicas": self.num_replicas,
            "loss": loss,
            "best_loss": best_loss,
            "phase": model._phase,
            "best_checkpoints": [(l, s, p) for l, s, p in self.best_checkpoints],
        }
        _atomic_json_dump(state, os.path.join(self.output_dir, "training_state.json"), indent=2)

        # Save best checkpoint
        ckpt_name = tag or f"best_step_{step}"
        ckpt_dir = os.path.join(self.output_dir, ckpt_name)

        # Check if this is a new best
        if loss < best_loss or len(self.best_checkpoints) < self.keep_best:
            os.makedirs(ckpt_dir, exist_ok=True)
            _atomic_torch_save(trainable, os.path.join(ckpt_dir, "trainable_weights.pt"))
            self.best_checkpoints.append((loss, step, ckpt_dir))
            self.best_checkpoints.sort(key=lambda x: x[0])

            # Remove excess checkpoints
            while len(self.best_checkpoints) > self.keep_best:
                _, _, path = self.best_checkpoints.pop()
                if os.path.exists(path) and path != latest_dir:
                    shutil.rmtree(path)

        print(f"  Checkpoint saved: step={step}, loss={loss:.4f}")

    def save_epoch_snapshot(self, model, epoch, step):
        """Persist a per-epoch weight snapshot that is NEVER pruned.

        Separate from the keep_best pool, which retains only the lowest-loss
        checkpoints. This writes
        `<output_dir>/epoch_{epoch}/trainable_weights.pt` unconditionally so
        every epoch is retained for later export+eval. Same on-disk shape as a
        best_step_* dir (trainable weights only), so export_hf.py can consume it
        directly. Cheap: trainable params only (projectors + LoRA), no optimizer.
        """
        snap_dir = os.path.join(self.output_dir, f"epoch_{epoch}")
        os.makedirs(snap_dir, exist_ok=True)
        _atomic_torch_save(model.get_trainable_state_dict(),
                           os.path.join(snap_dir, "trainable_weights.pt"))
        print(f"  Per-epoch snapshot saved: epoch_{epoch} (step={step})")

    def load(self, checkpoint_dir, model, optimizer, scheduler,
             accelerator=None, miner_cache=None):
        """Load a checkpoint.

        Args:
            checkpoint_dir: str — path to checkpoint
            model: OmniEmbedModel (may be wrapped by accelerator)
            optimizer: optimizer
            scheduler: scheduler
            accelerator: optional accelerator
            miner_cache: optional NegativeCache — restore mining progress

        Returns:
            dict with training state, or None if not found
        """
        weights_path = os.path.join(checkpoint_dir, "trainable_weights.pt")
        optim_path = os.path.join(checkpoint_dir, "optim_state.pt")
        cache_path = os.path.join(checkpoint_dir, "miner_cache.json")
        state_path = os.path.join(os.path.dirname(checkpoint_dir), "training_state.json")

        if not os.path.exists(weights_path):
            print(f"No checkpoint found at {weights_path}")
            return None

        # Load weights
        unwrapped = accelerator.unwrap_model(model) if accelerator else model
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        unwrapped.load_state_dict(state, strict=False)
        print(f"  Loaded weights from {weights_path}")

        # Load optimizer state — no try/except. If the optimizer/scheduler
        # state in the checkpoint is incompatible with the current run's
        # config (e.g., LR schedule, optimizer params changed), we want to
        # crash here, not silently restart from random optimizer state.
        # optim_loaded=False signals the caller that the LR scheduler was NOT
        # restored (optim_state.pt absent, e.g. deleted or never written) so
        # it can deterministically rebuild the
        # LR position from global_step. See train.py resume path.
        optim_loaded = False
        if os.path.exists(optim_path):
            optim_state = torch.load(optim_path, map_location="cpu", weights_only=True)
            optimizer.load_state_dict(optim_state["optimizer"])
            scheduler.load_state_dict(optim_state["scheduler"])
            optim_loaded = True
            print(f"  Loaded optimizer + scheduler state")
        else:
            print(f"  optim_state.pt ABSENT at {optim_path} — optimizer/scheduler "
                  f"NOT restored; caller will rebuild the LR schedule from global_step")

        # Load miner cache
        if miner_cache is not None and os.path.exists(cache_path):
            with open(cache_path) as f:
                cache_data = json.load(f)
            for k, v in cache_data.items():
                miner_cache.put(int(k), v)
            print(f"  Loaded miner cache: {len(cache_data)} entries")

        # Load training state
        if os.path.exists(state_path):
            with open(state_path) as f:
                training_state = json.load(f)
            self.best_checkpoints = [
                (l, s, p) for l, s, p in training_state.get("best_checkpoints", [])
            ]
            training_state["_optim_loaded"] = optim_loaded
            return training_state

        return {"global_step": 0, "epoch": 0, "best_loss": float("inf"),
                "_optim_loaded": optim_loaded}
