"""Resolve a model argument (HF Hub id or local export dir) and load it."""
from __future__ import annotations

from pathlib import Path


def resolve_model(model: str) -> tuple[str, str]:
    """Return (local_dir, short_name) for an exported Omni-Embed-Mini model.

    `model` is either a local directory produced by scripts/export_hf.py or a
    Hugging Face Hub repo id such as MBZUAI/Omni-Embed-Mini-0.9B (downloaded
    into the HF cache on first use).
    """
    p = Path(model).expanduser()
    if p.is_dir():
        local = p.resolve()
    else:
        from huggingface_hub import snapshot_download
        local = Path(snapshot_download(model))
    if not (local / "config.json").exists() or not (local / "modeling_omni_embed.py").exists():
        raise FileNotFoundError(
            f"{model!r} is not an Omni-Embed-Mini export "
            "(expected config.json + modeling_omni_embed.py)")
    name = model.rstrip("/").rsplit("/", 1)[-1]
    return str(local), name


def load_wrapper(model: str, device: str = "cuda:0", batch_size: int = 8):
    from omni_embed_wrapper import OmniEmbedMTEBWrapper
    local, name = resolve_model(model)
    return OmniEmbedMTEBWrapper(local, device=device, batch_size=batch_size, name=name)


def cache_name(model: str) -> str:
    """Directory name under which results for `model` are stored."""
    return f"OmniEmbed[{model.rstrip('/').rsplit('/', 1)[-1]}]"
