"""Export a trained Omni-Embed checkpoint to a fully HuggingFace-compatible
model directory.

The exported directory is self-contained and loads with AutoModel +
AutoProcessor, regardless of whether the training run used `vision_mode:
custom` (0.6B text backbone + our own Qwen3.5 vision tower) or `native`
(Qwen3-VL-Embedding-* with built-in vision) or `none` (audio-only).

Inference is identical across all three — same three lines of code, only the
path changes:

    from transformers import AutoModel, AutoProcessor
    model = AutoModel.from_pretrained("exported/omni-embed-mini-0.9b",
                                      trust_remote_code=True).eval().cuda()
    proc  = AutoProcessor.from_pretrained("exported/omni-embed-mini-0.9b",
                                          trust_remote_code=True)

    # Text
    inputs = proc(text="a dog barking", return_tensors="pt").to("cuda")
    emb = model.encode(**inputs)

    # Audio
    emb = model.encode(**proc(audio=wav, text="what is this?",
                              return_tensors="pt").to("cuda"))

    # Image / Video — works for both custom and native vision exports
    emb = model.encode(**proc(images=pil, return_tensors="pt").to("cuda"))
    emb = model.encode(**proc(videos=[frames], return_tensors="pt").to("cuda"))

    emb_256 = model.encode(..., truncate_dim=256)  # Matryoshka

Usage:
    python scripts/export_hf.py \
        --checkpoint checkpoints/omni-embed-mini-0.9b-tm-mm/epoch_4 \
        --config    configs/omni_embed_mini_0.9b.yaml \
        --output    exported/omni-embed-mini-0.9b
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Checkpoint dir containing trainable_weights.pt")
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Auto-append the checkpoint step (e.g. `best_step_5040` → `-5040`) to the
    # output dir name, unless the output already contains a digit suffix. This
    # ensures the exported model dir always identifies which training step
    # produced it, which downstream tooling uses to label table rows.
    import re
    ckpt_basename = os.path.basename(args.checkpoint.rstrip("/"))
    step_match = re.search(r"(\d+)$", ckpt_basename)
    out_basename = os.path.basename(args.output.rstrip("/"))
    if step_match and not re.search(r"-\d+$", out_basename):
        args.output = f"{args.output.rstrip('/')}-{step_match.group(1)}"

    os.makedirs(args.output, exist_ok=True)
    print(f"Exporting to {args.output}")

    # ── Build the training model and load trainable weights ──
    from model.omni_embed import OmniEmbedModel
    model = OmniEmbedModel(config)
    vision_mode = model.vision_mode
    # Baked from data.video_as_images: when true, video frames go through the
    # per-frame video projector (no VideoSpatialPooler); otherwise the pooled
    # path is used.
    video_as_images = bool(config.get("data", {}).get("video_as_images", False))
    print(f"  video_as_images = {video_as_images}")

    weights_path = os.path.join(args.checkpoint, "trainable_weights.pt")
    state = torch.load(weights_path, map_location="cpu", weights_only=True)

    # Text-backbone LoRA ablation: the freshly built model has no backbone LoRA,
    # so a plain strict=False load would silently drop the trained backbone-LoRA
    # tensors. Inject the LoRA before load so the adapters land, then merge them
    # into the base weights so save_pretrained yields a standard backbone.
    _has_bb_lora = any(("backbone" in k and "lora_" in k) for k in state)
    if _has_bb_lora:
        model.activate_text_backbone_lora()
        print("  [txtlora] injected text-backbone LoRA before load")

    # Phase-2 encoder LoRA (whisper/dasheng + custom Qwen3.5 vision): same
    # issue — the freshly built model has no encoder-LoRA modules, so a plain
    # strict=False load would silently drop them and export stock encoders.
    # Inject via activate_phase2 (mirrors train.py's phase-2 setup) so the
    # adapters land, then merge below.
    _has_enc_lora = any(
        (("whisper_encoder" in k or "dasheng_encoder" in k or "vision_encoder" in k)
         and "lora_" in k) for k in state
    )
    if _has_enc_lora:
        model.activate_phase2(config.get("lora", {}))
        print("  [enclora] injected Phase-2 encoder LoRA before load")

    model.load_state_dict(state, strict=False)

    if _has_bb_lora:
        from model.lora import LoRALinear
        n_merged = 0
        for mod in model.backbone.modules():
            for cname, child in list(mod.named_children()):
                if isinstance(child, LoRALinear):
                    base = child.original
                    dW = (child.lora_B.weight.data.float()
                          @ child.lora_A.weight.data.float()) * child.scaling
                    base.weight.data = (base.weight.data.float() + dW).to(base.weight.dtype)
                    setattr(mod, cname, base)
                    n_merged += 1
        print(f"  [txtlora] merged {n_merged} backbone LoRA adapters into base weights")

    if _has_enc_lora:
        from model.lora import LoRALinear
        # Merge Phase-2 encoder LoRA into base encoder weights so the .pt dumps
        # below carry the adaptation (no LoRA at inference — plain merged Linears).
        enc_roots = [("whisper", model.whisper_encoder),
                     ("dasheng", model.dasheng_encoder)]
        if vision_mode == "custom" and model.vision_encoder is not None:
            enc_roots.append(("vision", model.vision_encoder))
        for ename, eroot in enc_roots:
            if eroot is None:
                continue
            n_emerged = 0
            for mod in eroot.modules():
                for cname, child in list(mod.named_children()):
                    if isinstance(child, LoRALinear):
                        base = child.original
                        dW = (child.lora_B.weight.data.float()
                              @ child.lora_A.weight.data.float()) * child.scaling
                        base.weight.data = (base.weight.data.float() + dW).to(base.weight.dtype)
                        setattr(mod, cname, base)
                        n_emerged += 1
            print(f"  [enclora] merged {n_emerged} {ename} LoRA adapters into base weights")

    print(f"  vision_mode = {vision_mode}")
    print(f"  Loaded {len(state)} trainable tensors")

    # ── Dump backbone + tokenizer ──
    model.backbone.save_pretrained(os.path.join(args.output, "backbone"))
    model.tokenizer.save_pretrained(args.output)

    # ── Dump audio encoders (always present) ──
    torch.save(model.whisper_encoder.model.state_dict(),
               os.path.join(args.output, "whisper_encoder.pt"))
    if model.dasheng_encoder.model is not None:
        torch.save(model.dasheng_encoder.model.state_dict(),
                   os.path.join(args.output, "dasheng_encoder.pt"))

    # ── Dump custom vision encoder ONLY in vision_mode=custom ──
    if vision_mode == "custom" and model.vision_encoder is not None:
        torch.save(model.vision_encoder.visual.state_dict(),
                   os.path.join(args.output, "vision_encoder.pt"))

    # ── Dump projectors + LoRA adapters ──
    # Keys depend on vision_mode (vision projectors exist only in custom).
    projector_keys = ["whisper_down_proj.", "dasheng_down_proj."]
    if vision_mode == "custom":
        projector_keys += ["image_projector.", "video_pooler.", "video_projector."]

    projector_state = {}
    for name, param in model.named_parameters():
        if any(name.startswith(k) for k in projector_keys) or "lora_" in name:
            projector_state[name] = param.data.contiguous().cpu()
    torch.save(projector_state, os.path.join(args.output, "projector_weights.pt"))
    print(f"  Saved {len(projector_state)} projector/LoRA tensors")

    # ── Config + processor_config ──
    hf_config = {
        "model_type": "omni_embed",
        "architectures": ["OmniEmbedForEmbedding"],
        "vision_mode": vision_mode,
        "backbone_model": config["model"]["backbone"],
        "hidden_size": int(model.hidden_size),
        "whisper_model": config["audio"]["whisper"],
        "dasheng_model": config["audio"]["dasheng"],
        "whisper_dim": int(model.whisper_encoder.output_dim),
        "dasheng_dim": int(model.dasheng_encoder.output_dim),
        "tokens_per_encoder": int(config["audio"]["tokens_per_encoder"]),
        "num_video_tokens": int(config["vision"].get("num_video_tokens", 196)),
        "video_as_images": video_as_images,
        "mrl_dims": config["model"].get("mrl_dims_coarse", [128, 256, 512, 1024]),
        "auto_map": {
            "AutoConfig": "configuration_omni_embed.OmniEmbedConfig",
            "AutoModel": "modeling_omni_embed.OmniEmbedForEmbedding",
            "AutoProcessor": "processing_omni_embed.OmniEmbedProcessor",
        },
    }
    if vision_mode == "custom":
        hf_config["vision_model"] = config["vision"]["encoder"]
        hf_config["vision_dim"] = int(model.vision_encoder.output_dim)
    with open(os.path.join(args.output, "config.json"), "w") as f:
        json.dump(hf_config, f, indent=2)

    processor_cfg = {
        "processor_class": "OmniEmbedProcessor",
        "auto_map": {"AutoProcessor": "processing_omni_embed.OmniEmbedProcessor"},
        "vision_mode": vision_mode,
        "backbone_model": config["model"]["backbone"],
        "whisper_model": config["audio"]["whisper"],
        "tokens_per_encoder": int(config["audio"]["tokens_per_encoder"]),
        "num_video_tokens": int(config["vision"].get("num_video_tokens", 196)),
        "video_as_images": video_as_images,
    }
    if vision_mode == "custom":
        processor_cfg["vision_model"] = config["vision"]["encoder"]
        # Custom (0.9B) trains still images (incl. visual_doc) through a fixed
        # Resize((image_size,image_size)) in data/dataset.py. Bake that same
        # square resize into the exported processor so eval AND inference feed
        # still images in-distribution — native-res doc pages are far OOD (huge
        # grids the small Qwen3.5 encoder never saw) and collapse ViDoRe. 0 disables.
        processor_cfg["still_image_size"] = int(config["vision"].get("image_size", 224))
    with open(os.path.join(args.output, "processor_config.json"), "w") as f:
        json.dump(processor_cfg, f, indent=2)

    _write_configuration(args.output)
    _write_modeling(args.output)
    _write_processing(args.output)

    print(f"\nDone. Load with:")
    print(f'  model = AutoModel.from_pretrained("{args.output}", trust_remote_code=True)')
    print(f'  proc  = AutoProcessor.from_pretrained("{args.output}", trust_remote_code=True)')


# ── Remote code: configuration_omni_embed.py ──────────────────────────────

def _write_configuration(output_dir):
    code = '''"""OmniEmbed HuggingFace configuration."""
from transformers import PretrainedConfig


class OmniEmbedConfig(PretrainedConfig):
    model_type = "omni_embed"

    def __init__(
        self,
        vision_mode="custom",
        backbone_model="Qwen/Qwen3-Embedding-0.6B",
        hidden_size=1024,
        whisper_model="openai/whisper-small",
        whisper_dim=768,
        dasheng_model="mispeech/dasheng-base",
        dasheng_dim=768,
        tokens_per_encoder=128,
        vision_model=None,
        vision_dim=1024,
        num_video_tokens=196,
        video_as_images=False,
        mrl_dims=None,
        **kwargs,
    ):
        self.vision_mode = vision_mode
        self.backbone_model = backbone_model
        self.hidden_size = hidden_size
        self.whisper_model = whisper_model
        self.whisper_dim = whisper_dim
        self.dasheng_model = dasheng_model
        self.dasheng_dim = dasheng_dim
        self.tokens_per_encoder = tokens_per_encoder
        self.vision_model = vision_model
        self.vision_dim = vision_dim
        self.num_video_tokens = num_video_tokens
        self.video_as_images = video_as_images
        self.mrl_dims = mrl_dims or [128, 256, 512, 1024]
        super().__init__(**kwargs)
'''
    with open(os.path.join(output_dir, "configuration_omni_embed.py"), "w") as f:
        f.write(code)


# Remote-code files written into the export. They load with AutoModel /
# AutoProcessor (trust_remote_code=True) and match evaluations/omni_embed_wrapper.py.

# ── Remote code: modeling_omni_embed.py ───────────────────────────────────

def _write_modeling(output_dir):
    code = r'''"""OmniEmbed — multimodal embedding model, HF remote-code build.

The forward path here mirrors model/omni_embed.py from the training repo,
so inference matches the training-time forward on the same inputs. Three vision strategies selected by
config.vision_mode:

  - custom : Qwen3.5-0.8B visual tower + projectors (mirrors training)
             • forward_and_split splits merged tokens by grid_thw
             • images: per-item DimProjector
             • videos: pad+mask → VideoSpatialPooler → DimProjector
             • injection via one-hot matmul (differentiable; forward-identical
               to in-place substitution)
  - native : backbone (e.g. Qwen3-VL-Embedding-2B) owns vision; we pass
             pixel_values + grid_thw through so MRoPE + visual substitution
             run natively, and only matmul-inject audio
  - none   : no vision (audio-only builds)

The processor emits HF-canonical arg names (pixel_values,
pixel_values_videos, image_grid_thw, video_grid_thw, whisper_features,
dasheng_audio). Forward accepts those directly.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, PreTrainedModel, WhisperConfig, WhisperModel

from dataclasses import dataclass
from typing import Optional
from transformers.utils import ModelOutput

from .configuration_omni_embed import OmniEmbedConfig


@dataclass
class OmniEmbedOutput(ModelOutput):
    """Output of OmniEmbedForEmbedding.forward().

    pooler_output: (B, D) L2-normalised embedding -- the retrieval vector.
    last_hidden_state: (B, T, D) raw backbone states.
    """
    pooler_output: Optional["torch.FloatTensor"] = None
    last_hidden_state: Optional["torch.FloatTensor"] = None


WHISPER_MAX_MEL = 3000

# Vanilla Qwen3-Embedding-0.6B text-eval format. Mirror of model/omni_embed.py
# constants so the exported artifact is self-contained.
QWEN3_QUERY_TEMPLATE = "Instruct: {instruction}\nQuery: {query}"
DEFAULT_TASK_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


# ── Sub-encoders (self-contained) ────────────────────────────────────────

class WhisperSpeechEncoder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self._name = model_name
        self.model = WhisperModel(WhisperConfig.from_pretrained(model_name)).encoder
        self.output_dim = self.model.config.d_model

    def load_pretrained(self):
        w = WhisperModel.from_pretrained(self._name)
        self.model.load_state_dict(w.encoder.state_dict())

    def forward(self, mel):
        mel = mel.to(dtype=next(self.model.parameters()).dtype)
        T = mel.shape[-1]
        if T < WHISPER_MAX_MEL:
            mel = F.pad(mel, (0, WHISPER_MAX_MEL - T))
        elif T > WHISPER_MAX_MEL:
            mel = mel[:, :, :WHISPER_MAX_MEL]
        return self.model(input_features=mel).last_hidden_state


class DashengEncoder(nn.Module):
    # Must mirror model/encoders.py:DashengEncoder — both prefixed
    # ("mispeech/dasheng-*") and bare ("dasheng-*") names are supported
    # in training, and the exported model needs the same mapping so
    # configs using either form load identically.
    _DIMS = {
        "mispeech/dasheng-base": 768, "dasheng-base": 768,
    }
    _CTORS = {
        "mispeech/dasheng-base": "dasheng_base", "dasheng-base": "dasheng_base",
    }

    def __init__(self, model_name):
        super().__init__()
        self._name = model_name
        self.output_dim = self._DIMS.get(model_name, 768)
        self.model = None

    def load_pretrained(self, weights_path=None):
        import dasheng
        self.model = getattr(dasheng, self._CTORS[self._name])()
        if weights_path and os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.output_dim = self.model.embed_dim
        # Pin the mel front_end + init_bn to CPU so STFT runs on CPU
        # (dodges ROCm HIPFFT bugs); the rest of dasheng runs on GPU.
        self.model.front_end.to("cpu")
        self.model.init_bn.to("cpu")
        self._cpu_pinned = True

    def forward(self, raw_audio):
        # ROCm cuFFT crashes on dasheng's GPU STFT for short clips
        # (HIPFFT_PARSE_ERROR / HIPFFT_INTERNAL_ERROR). Run only the mel
        # front_end on CPU (torchaudio STFT works there); the heavy
        # transformer body (forward_spectrogram) stays on the original
        # device. front_end+init_bn are pinned to CPU at load time, so
        # there is no per-call .to() overhead.
        dev = raw_audio.device
        with torch.amp.autocast("cuda", enabled=False):
            m = self.model.float()
            spec = m.forward_to_spec(raw_audio.float().cpu())
            return m.forward_spectrogram(spec.to(dev))


class ConvDownProjector(nn.Module):
    def __init__(self, input_dim, output_dim, input_tokens, target_tokens):
        super().__init__()
        hidden = output_dim * 3
        self.stride = max(1, input_tokens // target_tokens)
        self.conv = nn.Conv1d(input_dim, input_dim,
                              kernel_size=self.stride * 2 + 1,
                              stride=self.stride, padding=self.stride)
        self.norm = nn.LayerNorm(input_dim)
        self.gate_up1 = nn.Linear(input_dim, hidden * 2, bias=False)
        self.down1 = nn.Linear(hidden, output_dim, bias=False)
        self.mid_norm = nn.LayerNorm(output_dim)
        self.gate_up2 = nn.Linear(output_dim, hidden * 2, bias=False)
        self.down2 = nn.Linear(hidden, output_dim, bias=False)
        self.out_norm = nn.LayerNorm(output_dim)
        self.target_tokens = target_tokens

    def forward(self, x):
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm(x)
        if x.shape[1] > self.target_tokens:
            x = x[:, :self.target_tokens]
        elif x.shape[1] < self.target_tokens:
            x = F.pad(x, (0, 0, 0, self.target_tokens - x.shape[1]))
        g, v = self.gate_up1(x).chunk(2, dim=-1)
        x = self.mid_norm(self.down1(F.silu(g) * v))
        g, v = self.gate_up2(x).chunk(2, dim=-1)
        return self.out_norm(x + self.down2(F.silu(g) * v))


class DimProjector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.proj(x)


class VideoSpatialPooler(nn.Module):
    """Mirrors model/vision.py:VideoSpatialPooler."""
    def __init__(self, input_dim, num_output_tokens, num_heads=16):
        super().__init__()
        self.num_output_tokens = num_output_tokens
        self.input_dim = input_dim
        self.query = nn.Parameter(torch.randn(1, num_output_tokens, input_dim) * 0.02)
        self.attn1 = nn.MultiheadAttention(input_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(input_dim)
        self.attn2 = nn.MultiheadAttention(input_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(input_dim)

    def forward(self, tokens, attention_mask=None):
        B = tokens.shape[0]
        q = self.query.expand(B, -1, -1)
        kpm = (attention_mask == 0) if attention_mask is not None else None
        o, _ = self.attn1(q, tokens, tokens, key_padding_mask=kpm)
        o = self.norm1(q + o)
        o2, _ = self.attn2(o, tokens, tokens, key_padding_mask=kpm)
        return self.norm2(o + o2)


class Qwen35VisionEncoderWrapper(nn.Module):
    """Mirrors model/vision.py:Qwen35VisionEncoder semantics.

    Wraps a loaded `visual` module (extracted from Qwen3.5-0.8B) and
    exposes `forward_and_split` so per-item tokens can be split by
    `grid_thw` exactly as in training.
    """
    def __init__(self):
        super().__init__()
        self.visual = None
        self.spatial_merge_size = 2

    def set_visual(self, visual, spatial_merge_size):
        self.visual = visual
        self.spatial_merge_size = spatial_merge_size

    def forward(self, pixel_values, grid_thw):
        dtype = self.visual.merger.linear_fc2.weight.dtype
        out = self.visual(pixel_values.to(dtype), grid_thw=grid_thw)
        return out.pooler_output

    def forward_and_split(self, pixel_values, grid_thw):
        merged = self.forward(pixel_values, grid_thw)
        sms = self.spatial_merge_size
        tokens_per_item = (
            grid_thw[:, 0] * (grid_thw[:, 1] // sms) * (grid_thw[:, 2] // sms)
        )
        return merged.split(tokens_per_item.tolist(), dim=0)


# ── Main model ───────────────────────────────────────────────────────────

class OmniEmbedForEmbedding(PreTrainedModel):
    """Inference-time mirror of model/omni_embed.py:OmniEmbedModel.

    Module layout and forward semantics are identical to the training model,
    so `model.load_state_dict(projector_weights, strict=False)` restores
    every trainable parameter into an exactly-matching graph.
    """
    config_class = OmniEmbedConfig
    base_model_prefix = "omni_embed"

    def __init__(self, config: OmniEmbedConfig):
        super().__init__(config)

        self.vision_mode = getattr(config, "vision_mode", "custom")
        self.backbone = AutoModel.from_pretrained(
            config.backbone_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        )
        bb_cfg = self.backbone.config
        self.hidden_size = getattr(bb_cfg, "hidden_size", None)
        if self.hidden_size is None and hasattr(bb_cfg, "text_config"):
            self.hidden_size = bb_cfg.text_config.hidden_size

        # Audio (always added) — names match training: whisper_down_proj / dasheng_down_proj
        self.whisper_encoder = WhisperSpeechEncoder(config.whisper_model)
        self.dasheng_encoder = DashengEncoder(config.dasheng_model)
        tpe = config.tokens_per_encoder
        self.whisper_down_proj = ConvDownProjector(config.whisper_dim,
                                                   self.hidden_size, 1500, tpe)
        self.dasheng_down_proj = ConvDownProjector(config.dasheng_dim,
                                                   self.hidden_size, 750, tpe)
        self.tokens_per_encoder = tpe
        self.num_audio_tokens_per_chunk = tpe * 2

        # Custom vision stack — module names MUST match training so
        # projector_weights.pt keys line up (image_projector, video_pooler,
        # video_projector).
        self.vision_encoder = None
        self.image_projector = None
        self.video_pooler = None
        self.video_projector = None
        if self.vision_mode == "custom":
            self.vision_model_name = config.vision_model
            self.vision_dim = config.vision_dim
            self.vision_encoder = Qwen35VisionEncoderWrapper()
            self.image_projector = DimProjector(config.vision_dim, self.hidden_size)
            self.video_pooler = VideoSpatialPooler(
                config.vision_dim, config.num_video_tokens, num_heads=16,
            )
            self.video_projector = DimProjector(config.vision_dim, self.hidden_size)

        self.num_video_tokens = config.num_video_tokens
        # video_as_images: route video frames through the per-frame
        # video_projector (no VideoSpatialPooler), grouped per video via
        # video_num_frames. Mirrors model/omni_embed.py::_encode_video.
        self.video_as_images = getattr(config, "video_as_images", False)
        self._pad_ids = {}
        self._loaded = False
        # Set by the wrapper after construction (AutoProcessor.tokenizer),
        # so the native text path can run without needing a processor.
        self.tokenizer = None

        # Native-mode audio injection hook (mirrors training model).
        # VL backbones require canonical input_ids for 3D MRoPE; we can't
        # pre-build inputs_embeds. Instead, hook embed_tokens and splice
        # audio features at <|audio_pad|> rows mid-forward.
        self._pending_audio_feats = None
        self._pending_audio_input_ids = None
        if self.vision_mode == "native":
            self.backbone.get_input_embeddings().register_forward_hook(
                self._audio_inject_hook
            )

    def _audio_inject_hook(self, module, args, output):
        if self._pending_audio_feats is None or self._pending_audio_input_ids is None:
            return output
        input_ids = self._pending_audio_input_ids
        out = output.clone()
        mask = (input_ids == self._pad_ids.get("audio"))
        for b in range(out.shape[0]):
            pos = mask[b].nonzero(as_tuple=True)[0]
            feats = self._pending_audio_feats[b]
            n = min(pos.numel(), feats.shape[0])
            if n > 0:
                out[b, pos[:n]] = feats[:n].to(out.dtype)
        return out

    # ── Deferred weight loading ──
    def _load_weights(self, model_dir):
        if self._loaded:
            return
        dev = next(self.backbone.parameters()).device

        # Stock Qwen3-Embedding-0.6B has vocab_size=151669, but the OmniEmbed
        # processor emits tokens up to 151671 (audio_pad). The exporter saves
        # a vocab-resized 151676 backbone under <model_dir>/backbone/; reload
        # it so embedding lookups succeed for media tokens.
        bbp = os.path.join(model_dir, "backbone")
        if os.path.isdir(bbp) and os.path.exists(os.path.join(bbp, "config.json")):
            self.backbone = AutoModel.from_pretrained(
                bbp, torch_dtype=torch.bfloat16, trust_remote_code=True,
            ).to(dev)

        p = os.path.join(model_dir, "whisper_encoder.pt")
        if os.path.exists(p):
            self.whisper_encoder.model.load_state_dict(torch.load(p, map_location="cpu"))
        else:
            self.whisper_encoder.load_pretrained()
        self.whisper_encoder.to(dev).eval()

        p = os.path.join(model_dir, "dasheng_encoder.pt")
        self.dasheng_encoder.load_pretrained(weights_path=p if os.path.exists(p) else None)
        self.dasheng_encoder.to(dev).eval()
        # Re-pin front_end + init_bn to CPU after the device move (the
        # CPU pinning happens inside load_pretrained but .to(dev) above
        # would otherwise drag them back to GPU).
        if getattr(self.dasheng_encoder, "_cpu_pinned", False):
            self.dasheng_encoder.model.front_end.to("cpu")
            self.dasheng_encoder.model.init_bn.to("cpu")

        if self.vision_mode == "custom":
            from transformers import AutoModelForImageTextToText
            full = AutoModelForImageTextToText.from_pretrained(
                self.vision_model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
            )
            visual = full.model.visual
            vc = full.config.vision_config
            sms = getattr(vc, "spatial_merge_size",
                          getattr(vc, "merge_size", 2))
            del full
            p = os.path.join(model_dir, "vision_encoder.pt")
            if os.path.exists(p):
                visual.load_state_dict(torch.load(p, map_location="cpu"))
            self.vision_encoder.set_visual(visual.to(dev).eval(), sms)

        p = os.path.join(model_dir, "projector_weights.pt")
        if os.path.exists(p):
            self.load_state_dict(torch.load(p, map_location="cpu"), strict=False)

        # Re-register the audio-inject hook on the (possibly replaced) backbone.
        # __init__ registered it on the initial backbone; the vocab-resized
        # reload above replaced self.backbone, orphaning that hook. Without
        # this, vision_mode=="native" silently ignores audio (whisper+dasheng
        # features are computed but never spliced into inputs_embeds).
        if self.vision_mode == "native":
            self.backbone.get_input_embeddings().register_forward_hook(
                self._audio_inject_hook
            )

        mods = [self.whisper_encoder, self.dasheng_encoder, self.backbone]
        if self.vision_encoder is not None:
            mods.append(self.vision_encoder)
        for m in mods:
            for param in m.parameters():
                param.requires_grad = False

        self._loaded = True


    # ── Loading ───────────────────────────────────────────────────────────
    REQUIRED_FILES = ("whisper_encoder.pt", "dasheng_encoder.pt",
                      "projector_weights.pt")

    @classmethod
    def _resolve_dir(cls, name_or_path, **hub_kwargs):
        """Return a LOCAL directory for `name_or_path`, downloading if needed."""
        if os.path.isdir(str(name_or_path)):
            return str(name_or_path)
        from huggingface_hub import snapshot_download
        return snapshot_download(str(name_or_path), **hub_kwargs)

    @classmethod
    def _verify_weights_present(cls, model_dir, config):
        """Fail LOUDLY if the weight files are missing.

        Every load inside _load_weights is guarded by os.path.exists and falls
        back to STOCK pretrained encoders / randomly-initialised projectors when
        a file is absent. Check up front so a partial path fails loudly.
        """
        missing = [f for f in cls.REQUIRED_FILES
                   if not os.path.exists(os.path.join(model_dir, f))]
        if getattr(config, "vision_mode", "custom") == "custom" \
                and not os.path.exists(os.path.join(model_dir, "vision_encoder.pt")):
            missing.append("vision_encoder.pt")
        if not os.path.isdir(os.path.join(model_dir, "backbone")):
            missing.append("backbone/")
        if missing:
            raise OSError(
                f"OmniEmbed: {model_dir!r} is missing {missing}. Loading would "
                "silently fall back to stock encoders and untrained projectors. "
                "Pass a directory produced by snapshot_download() (or a full "
                "local export), not a partial copy."
            )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args,
                        torch_dtype=None, dtype=None, config=None,
                        trust_remote_code=None, **kwargs):
        """Load an exported Omni-Embed checkpoint.

        Accepts a hub id or a local directory. The export keeps its backbone in
        `backbone/` and its encoders/projectors in `.pt` files rather than a
        top-level safetensors, so this does not call super().from_pretrained().
        """
        from transformers import AutoConfig
        hub_keys = ("revision", "cache_dir", "token", "local_files_only",
                    "proxies", "resume_download", "force_download")
        hub_kwargs = {k: kwargs.pop(k) for k in hub_keys if k in kwargs}
        model_dir = cls._resolve_dir(pretrained_model_name_or_path, **hub_kwargs)

        if config is None:
            config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        config._name_or_path = model_dir       # deferred loader reads from here
        cls._verify_weights_present(model_dir, config)

        model = cls(config)
        # transformers >=5 strips torch_dtype/dtype from kwargs and records it on
        # the config instead, so accept it from either place. Missing this leaves
        # the projectors in fp32 while the backbone loads as bf16 ->
        # "mat1 and mat2 must have the same dtype" on the first image forward.
        dt = torch_dtype if torch_dtype is not None else dtype
        if dt is None:
            dt = getattr(config, "dtype", None) or getattr(config, "torch_dtype", None)
        if dt is not None and dt != "auto":
            if isinstance(dt, str):
                dt = getattr(torch, dt)
            # Cast the WHOLE model. Casting only the projectors leaves the audio
            # encoders in fp32 and shifts audio embeddings.
            model = model.to(dt)
        # Auto-wire the tokenizer and the media placeholder ids. Without pad ids
        # the model computes media features and then silently DROPS them --
        # embeddings come back looking like text-only, with no error. Callers
        # of the plain nn.Module API can still override via set_pad_ids().
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
            model.tokenizer = tok
            model.set_pad_ids(
                audio_pad=tok.convert_tokens_to_ids("<|audio_pad|>"),
                image_pad=tok.convert_tokens_to_ids("<|image_pad|>"),
                video_pad=tok.convert_tokens_to_ids("<|video_pad|>"),
            )
        except Exception as e:                       # never mask a real problem
            raise RuntimeError(
                f"OmniEmbed: could not wire media placeholder ids from {model_dir!r} "
                f"({e}). Media inputs would be silently ignored."
            ) from e
        model.eval()
        # NOTE: weights are materialised on the first forward (see _load_weights),
        # by which point the caller's .cuda()/.to() has already run.
        return model

    def _apply(self, *a, **kw):
        """Keep Dasheng's front_end/init_bn pinned to CPU across device moves."""
        out = super()._apply(*a, **kw)
        enc = getattr(self, "dasheng_encoder", None)
        if enc is not None and getattr(enc, "_cpu_pinned", False):
            try:
                enc.model.front_end.to("cpu")
                enc.model.init_bn.to("cpu")
            except Exception:
                pass
        return out

    def set_pad_ids(self, audio_pad, image_pad=None, video_pad=None):
        self._pad_ids = {"audio": audio_pad, "image": image_pad, "video": video_pad}

    # ── Encoding helpers ──
    def _encode_audio(self, mel, raw_audio):
        B, mel_T = mel.shape[0], mel.shape[-1]
        spc = int(30.0 * 16000)
        nc = max(1, (mel_T + WHISPER_MAX_MEL - 1) // WHISPER_MAX_MEL)
        chunks = []
        for c in range(nc):
            ms, me = c * WHISPER_MAX_MEL, min((c + 1) * WHISPER_MAX_MEL, mel_T)
            mc = mel[:, :, ms:me]
            if mc.shape[-1] < WHISPER_MAX_MEL:
                mc = F.pad(mc, (0, WHISPER_MAX_MEL - mc.shape[-1]))
            wh = self.whisper_encoder(mc)

            aus, aue = c * spc, min((c + 1) * spc, raw_audio.shape[-1])
            ac = raw_audio[:, :, aus:aue]
            # ROCm cuFFT raises HIPFFT_PARSE_ERROR on the 8000-sample STFT
            # path used by short clips.
            # Pad to 16000 (1 sec) — the length dasheng's standalone wrapper
            # uses, known to plan cleanly on HIP.
            if ac.shape[-1] < 16000:
                ac = F.pad(ac, (0, 16000 - ac.shape[-1]))
            dh = self.dasheng_encoder(ac.squeeze(1))

            dt = next(self.whisper_down_proj.parameters()).dtype
            wp = self.whisper_down_proj(wh.to(dt))
            dp = self.dasheng_down_proj(dh.to(dt))
            B_, T_, D_ = wp.shape
            chunks.append(torch.stack([wp, dp], dim=2).reshape(B_, T_ * 2, D_))
        return torch.cat(chunks, dim=1)

    # ── Per-modality encoders (custom mode) — mirror training ─────────

    def _encode_image_custom(self, pixel_values, grid_thw):
        """Mirror of training _encode_image: forward_and_split → per-item DimProjector."""
        features = self.vision_encoder.forward_and_split(pixel_values, grid_thw)
        return [self.image_projector(f) for f in features]

    def _encode_video_custom(self, pixel_values, grid_thw, video_num_frames=None):
        """Mirror of training _encode_video.

        video_as_images (video_num_frames given): each frame's tokens go
        through the per-frame video_projector (no pooler), concatenated per
        video. Returns list[(N_i, hidden)] — one tensor per video.

        Pooled (video_num_frames None): pad+mask → VideoSpatialPooler →
        DimProjector, returns (B, num_video_tokens, hidden).
        """
        features = self.vision_encoder.forward_and_split(pixel_values, grid_thw)

        # ── video_as_images: per-frame projector, grouped per video (no pooler) ──
        if video_num_frames is not None:
            out, off = [], 0
            for f in video_num_frames:
                f = int(f)
                toks = torch.cat(
                    [self.video_projector(features[off + j]) for j in range(f)],
                    dim=0,
                )
                out.append(toks)
                off += f
            return out

        # ── Pooled path (VideoSpatialPooler) ──
        B = len(features)
        max_tokens = max(f.shape[0] for f in features)
        dim = features[0].shape[1]
        dev, dtype = features[0].device, features[0].dtype
        batched = torch.zeros(B, max_tokens, dim, device=dev, dtype=dtype)
        attn_mask = torch.zeros(B, max_tokens, device=dev, dtype=torch.long)
        for i, f in enumerate(features):
            batched[i, :f.shape[0]] = f
            attn_mask[i, :f.shape[0]] = 1
        pooled = self.video_pooler(batched, attention_mask=attn_mask)
        return self.video_projector(pooled)

    # ── Gradient-free token replacement (forward-identical to training matmul) ──

    def _build_inputs_embeds(self, input_ids,
                             audio_feats=None,
                             image_feats_list=None,
                             video_feats=None):
        """Mirror of training _build_inputs_embeds for inference.

        Uses indexed assignment (no grad needed) but produces tensors
        byte-identical to the one-hot-matmul training path.
        """
        B, S = input_ids.shape
        text_embeds = self.backbone.get_input_embeddings()(input_ids)
        dtype = text_embeds.dtype
        out = text_embeds.clone()

        if audio_feats is not None and self._pad_ids.get("audio") is not None:
            mask = (input_ids == self._pad_ids["audio"])
            for b in range(B):
                pos = mask[b].nonzero(as_tuple=True)[0]
                n = min(pos.numel(), audio_feats[b].shape[0])
                if n > 0:
                    out[b, pos[:n]] = audio_feats[b][:n].to(dtype)

        if image_feats_list is not None and self._pad_ids.get("image") is not None:
            mask = (input_ids == self._pad_ids["image"])
            for b in range(B):
                pos = mask[b].nonzero(as_tuple=True)[0]
                n = min(pos.numel(), image_feats_list[b].shape[0])
                if n > 0:
                    out[b, pos[:n]] = image_feats_list[b][:n].to(dtype)

        if video_feats is not None and self._pad_ids.get("video") is not None:
            mask = (input_ids == self._pad_ids["video"])
            # video_feats has one entry per video, not per batch row: rows
            # without video are skipped, so vidx (not b) indexes the list. The
            # pooled path is a (B, num_video_tokens, H) tensor.
            vidx = 0
            for b in range(B):
                pos = mask[b].nonzero(as_tuple=True)[0]
                if pos.numel() == 0:
                    continue
                if isinstance(video_feats, list):
                    feat_b = video_feats[vidx]
                    n = min(pos.numel(), feat_b.shape[0])
                    if n > 0:
                        out[b, pos[:n]] = feat_b[:n].to(dtype)
                else:
                    n = min(pos.numel(), video_feats.shape[1])
                    if n > 0:
                        out[b, pos[:n]] = video_feats[vidx, :n].to(dtype)
                vidx += 1

        return out

    # ── Forward ──
    def _forward_hidden(
        self,
        input_ids,
        attention_mask=None,
        whisper_features=None,
        dasheng_audio=None,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        video_num_frames=None,
        **kwargs,
    ):
        if not self._loaded:
            self._load_weights(getattr(self.config, "_name_or_path", "."))

        if self.vision_mode == "native":
            # VL backbones require canonical input_ids for 3D M-RoPE on the
            # first forward (rope_deltas is None). Forward hook on
            # embed_tokens splices audio at <|audio_pad|> rows mid-forward.
            audio_feats = None
            if whisper_features is not None and dasheng_audio is not None:
                af = self._encode_audio(whisper_features, dasheng_audio)
                audio_feats = [af[i] for i in range(af.shape[0])]
            self._pending_audio_feats = audio_feats
            self._pending_audio_input_ids = input_ids

            img_id = self._pad_ids.get("image")
            vid_id = self._pad_ids.get("video")
            mm = torch.zeros_like(input_ids)
            if img_id is not None:
                mm[input_ids == img_id] = 1
            if vid_id is not None:
                mm[input_ids == vid_id] = 2

            bb_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                mm_token_type_ids=mm,
                use_cache=False,
            )
            if pixel_values is not None:
                bb_kwargs["pixel_values"] = pixel_values
                bb_kwargs["image_grid_thw"] = image_grid_thw
            if pixel_values_videos is not None:
                bb_kwargs["pixel_values_videos"] = pixel_values_videos
                bb_kwargs["video_grid_thw"] = video_grid_thw

            try:
                if hasattr(self.backbone, "rope_deltas"):
                    self.backbone.rope_deltas = None
                out = self.backbone(**bb_kwargs)
            finally:
                self._pending_audio_feats = None
                self._pending_audio_input_ids = None
            return out.last_hidden_state

        # ── custom / none: pre-build inputs_embeds (no MRoPE concerns) ──
        audio_feats = None
        if whisper_features is not None and dasheng_audio is not None:
            af = self._encode_audio(whisper_features, dasheng_audio)
            audio_feats = [af[i] for i in range(af.shape[0])]

        image_feats_list = None
        video_feats = None
        if self.vision_mode == "custom":
            if pixel_values is not None and image_grid_thw is not None:
                image_feats_list = self._encode_image_custom(pixel_values, image_grid_thw)
            if pixel_values_videos is not None and video_grid_thw is not None:
                # Pass per-video frame counts so frames route through the
                # per-frame video_projector (video_num_frames emitted by the
                # processor when video_as_images). None → pooled path.
                _vnf = video_num_frames
                if _vnf is None and self.video_as_images:
                    # Single un-batched video: all grid rows belong to one video.
                    _vnf = [video_grid_thw.shape[0]]
                if _vnf is not None and torch.is_tensor(_vnf):
                    _vnf = _vnf.tolist()
                video_feats = self._encode_video_custom(
                    pixel_values_videos, video_grid_thw, _vnf,
                )

        embeds = self._build_inputs_embeds(
            input_ids,
            audio_feats=audio_feats,
            image_feats_list=image_feats_list,
            video_feats=video_feats,
        )
        out = self.backbone(
            inputs_embeds=embeds, attention_mask=attention_mask, use_cache=False,
        )
        return out.last_hidden_state


    def _pool(self, lhs, attention_mask=None, truncate_dim=None):
        """Last-real-token pooling + L2 norm.

        Shared by forward() and encode() so the two cannot drift apart.
        """
        am = attention_mask
        if am is not None and not bool(am[:, 0].all()):
            # Left-padded: last real token is at position -1
            emb = lhs[:, -1]
        elif am is not None:
            # Right-padded: last real token per row is at sum(mask)-1
            sl = am.sum(dim=1) - 1
            bi = torch.arange(lhs.shape[0], device=lhs.device)
            emb = lhs[bi, sl]
        else:
            emb = lhs[:, -1]
        emb = F.normalize(emb, p=2, dim=-1)
        if truncate_dim and truncate_dim < emb.shape[-1]:
            emb = F.normalize(emb[:, :truncate_dim], p=2, dim=-1)
        return emb

    def forward(self, input_ids=None, attention_mask=None, truncate_dim=None,
                return_dict=True, **kwargs):
        """HF-idiomatic forward: returns an object exposing `.pooler_output`.

        `pooler_output` is the L2-normalised embedding -- identical to what
        encode() returns. `last_hidden_state` is also exposed. encode() calls
        _forward_hidden directly.
        """
        # Pure-text inputs must follow encode_text() semantics (backbone embeds
        # in directly). _forward_hidden's native branch instead passes input_ids
        # so the VL backbone can build 3D M-RoPE, which for text-only input
        # yields a slightly different hidden state for text-only input.
        _media = ("whisper_features", "dasheng_audio", "pixel_values",
                  "pixel_values_videos")
        if not any(kwargs.get(k) is not None for k in _media):
            if not self._loaded:
                self._load_weights(getattr(self.config, "_name_or_path", "."))
            _emb = self.backbone.get_input_embeddings()(input_ids)
            lhs = self.backbone(inputs_embeds=_emb,
                                attention_mask=attention_mask,
                                use_cache=False).last_hidden_state
        else:
            lhs = self._forward_hidden(input_ids=input_ids,
                                       attention_mask=attention_mask, **kwargs)
        pooled = self._pool(lhs, attention_mask, truncate_dim)
        if not return_dict:
            return (pooled, lhs)
        return OmniEmbedOutput(last_hidden_state=lhs, pooler_output=pooled)

    @torch.no_grad()
    def encode(self, truncate_dim=None, **kwargs):
        lhs = self._forward_hidden(**kwargs)
        am = kwargs.get("attention_mask")
        if am is not None and not bool(am[:, 0].all()):
            # Left-padded: last real token is at position -1
            emb = lhs[:, -1]
        elif am is not None:
            # Right-padded: last real token per row is at sum(mask)-1
            sl = am.sum(dim=1) - 1
            bi = torch.arange(lhs.shape[0], device=lhs.device)
            emb = lhs[bi, sl]
        else:
            emb = lhs[:, -1]
        emb = F.normalize(emb, p=2, dim=-1)
        if truncate_dim and truncate_dim < emb.shape[-1]:
            emb = F.normalize(emb[:, :truncate_dim], p=2, dim=-1)
        return emb

    # ── Vanilla Qwen3-Embedding text path ─────────────────────────────

    @staticmethod
    def format_text_native(text, *, is_query=False, instruction=None):
        """Apply vanilla Qwen3-Embedding formatting to a single string."""
        if is_query:
            instr = instruction or DEFAULT_TASK_INSTRUCTION
            return QWEN3_QUERY_TEMPLATE.format(instruction=instr, query=text)
        return text

    def tokenize_text_native(self, texts, *, is_query=False, instruction=None,
                             max_length=1024, device=None):
        """Tokenize for the vanilla Qwen3-Embedding text path.

        - no chat wrap, no modality prefix
        - add_special_tokens=False
        - explicit <|endoftext|> appended to every row
        - left-padded
        """
        if self.tokenizer is None:
            raise RuntimeError(
                "OmniEmbedForEmbedding.tokenize_text_native requires "
                "`self.tokenizer` to be set (typically by the wrapper "
                "assigning `model.tokenizer = processor.tokenizer`)."
            )
        formatted = [
            self.format_text_native(t, is_query=is_query, instruction=instruction)
            for t in texts
        ]
        tok = self.tokenizer
        prev_side = tok.padding_side
        tok.padding_side = "left"
        try:
            enc = tok(
                formatted, return_tensors="pt", padding=True,
                truncation=True, max_length=max(1, max_length - 1),
                add_special_tokens=False,
            )
        finally:
            tok.padding_side = prev_side
        # Qwen3-Embedding native recipe terminates every row with <|endoftext|>
        # (id 151643), NOT <|im_end|> (151645 = tok.eos_token_id). Using the
        # chat terminator shifts EOS-pooling and badly degrades retrieval.
        # No silent fallback.
        eos = tok.convert_tokens_to_ids("<|endoftext|>")
        if eos is None or eos == tok.unk_token_id:
            raise RuntimeError(
                "tokenizer has no '<|endoftext|>' token — required for "
                "the Qwen3-Embedding native retrieval recipe."
            )
        B = enc["input_ids"].shape[0]
        eos_col = torch.full((B, 1), eos, dtype=enc["input_ids"].dtype)
        mask_col = torch.ones((B, 1), dtype=enc["attention_mask"].dtype)
        enc["input_ids"] = torch.cat([enc["input_ids"], eos_col], dim=1)
        enc["attention_mask"] = torch.cat([enc["attention_mask"], mask_col], dim=1)
        if device is not None:
            enc = {k: v.to(device) for k, v in enc.items()}
        return dict(enc)

    @torch.no_grad()
    def encode_text(self, input_ids, attention_mask):
        """Text-only forward: backbone embeds + causal forward + EOS pool.

        Identical semantics to model/omni_embed.py::encode_text. Uses the
        padding-side-agnostic pooling from encode().
        """
        if not self._loaded:
            self._load_weights(getattr(self.config, "_name_or_path", "."))
        inputs_embeds = self.backbone.get_input_embeddings()(input_ids)
        out = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )
        lhs = out.last_hidden_state
        if attention_mask is not None and not bool(attention_mask[:, 0].all()):
            emb = lhs[:, -1]
        elif attention_mask is not None:
            sl = attention_mask.sum(dim=1) - 1
            bi = torch.arange(lhs.shape[0], device=lhs.device)
            emb = lhs[bi, sl]
        else:
            emb = lhs[:, -1]
        return F.normalize(emb, p=2, dim=-1)
'''
    with open(os.path.join(output_dir, "modeling_omni_embed.py"), "w") as f:
        f.write(code)


# ── Remote code: processing_omni_embed.py ─────────────────────────────────

def _write_processing(output_dir):
    code = r'''"""OmniEmbed processor — wraps tokenizer + Whisper FE + image processor.

Mode-aware: in vision_mode=native it pulls the backbone's own image
processor (e.g. Qwen3-VL-Embedding-2B's processor); in custom it pulls the
companion Qwen3.5 vision processor; in none it skips images/videos entirely.

Placeholder templates follow the mode:
  - custom : <|image_start|> ... <|image_end|>, <|video_start|> ... <|video_end|>
  - native : <|vision_start|> ... <|vision_end|> (Qwen-VL convention)
"""
import json
import os

import numpy as np
import torch
from transformers import AutoTokenizer, WhisperFeatureExtractor
from transformers.processing_utils import ProcessorMixin


WHISPER_MAX_MEL = 3000


class OmniEmbedProcessor(ProcessorMixin):
    attributes = []
    tokenizer_class = "AutoTokenizer"

    def __init__(self, tokenizer=None, whisper_fe=None, image_processor=None,
                 vision_mode="custom", tokens_per_encoder=128,
                 num_video_tokens=196, video_as_images=False,
                 still_image_size=0, **kwargs):
        self.tokenizer = tokenizer
        self.whisper_fe = whisper_fe
        self.image_processor = image_processor
        self.vision_mode = vision_mode
        self.tokens_per_encoder = tokens_per_encoder
        self.num_video_tokens = num_video_tokens
        # video_as_images: emit per-frame (image-like) video-pad counts + a
        # video_num_frames tensor instead of a fixed num_video_tokens block.
        self.video_as_images = video_as_images
        # Custom (0.9B): square-resize still images to this size before the
        # Qwen3.5 image processor, matching train-time Resize (data/dataset.py).
        # 0/None disables (native mode keeps dynamic full-resolution grids).
        self.still_image_size = int(still_image_size or 0)

        self.audio_pad_id = tokenizer.convert_tokens_to_ids("<|audio_pad|>")
        self.image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.video_pad_id = tokenizer.convert_tokens_to_ids("<|video_pad|>")

        # Placeholder wrappers depend on mode
        if vision_mode == "native":
            self.img_open, self.img_close = "<|vision_start|>", "<|vision_end|>"
            self.vid_open, self.vid_close = "<|vision_start|>", "<|vision_end|>"
        else:
            self.img_open, self.img_close = "<|image_start|>", "<|image_end|>"
            self.vid_open, self.vid_close = "<|video_start|>", "<|video_end|>"

    # Hub-side kwargs that must reach BOTH the tokenizer and the
    # processor_config.json fetch, so `revision="v1.0"` pins every piece.
    _HUB_KWARGS = ("revision", "token", "cache_dir", "local_files_only",
                   "force_download", "proxies", "subfolder")

    @classmethod
    def _resolve_processor_config(cls, name_or_path, **hub_kwargs):
        """Return a LOCAL path to processor_config.json, downloading if needed.

        A bare repo id is not a directory, so the file is fetched from the Hub;
        otherwise every setting would silently fall back to its default. Only
        this one small file is fetched, so building the processor never pulls
        weights.
        """
        p = os.path.join(str(name_or_path), "processor_config.json")
        if os.path.exists(p):
            return p
        if os.path.isdir(str(name_or_path)):
            return None            # genuine local export that lacks the file
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError
        try:
            return hf_hub_download(str(name_or_path), "processor_config.json",
                                   **hub_kwargs)
        except EntryNotFoundError:
            return None            # repo reachable, file genuinely absent

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        hub_kwargs = {k: kwargs[k] for k in cls._HUB_KWARGS if k in kwargs}
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=True, **hub_kwargs,
        )
        pc_path = cls._resolve_processor_config(
            pretrained_model_name_or_path, **hub_kwargs)
        if pc_path is not None:
            with open(pc_path) as f:
                pc = json.load(f)
        else:
            pc = {}

        vision_mode = pc.get("vision_mode", "custom")
        whisper_name = pc.get("whisper_model", "openai/whisper-small")
        whisper_fe = WhisperFeatureExtractor.from_pretrained(whisper_name)

        image_processor = None
        if vision_mode != "none":
            from transformers import AutoImageProcessor
            # custom → pc["vision_model"]; native → pc["backbone_model"]
            ip_src = (pc.get("vision_model") if vision_mode == "custom"
                      else pc.get("backbone_model"))
            if ip_src:
                image_processor = AutoImageProcessor.from_pretrained(
                    ip_src, trust_remote_code=True,
                )
        if vision_mode != "none" and image_processor is None:
            raise OSError(
                f"OmniEmbed: no image processor could be built for "
                f"{pretrained_model_name_or_path!r} (vision_mode={vision_mode!r}). "
                "processor_config.json is missing or has no vision_model / "
                "backbone_model entry; image and video would fail at call time."
            )

        return cls(
            tokenizer=tokenizer,
            whisper_fe=whisper_fe,
            image_processor=image_processor,
            vision_mode=vision_mode,
            tokens_per_encoder=pc.get("tokens_per_encoder", 128),
            num_video_tokens=pc.get("num_video_tokens", 196),
            video_as_images=pc.get("video_as_images", False),
            still_image_size=pc.get("still_image_size", 0),
        )

    def save_pretrained(self, save_directory, **kwargs):
        self.tokenizer.save_pretrained(save_directory)

    def __call__(self, text=None, audio=None, images=None, videos=None,
                 return_tensors="pt", is_query=False, instruction=None,
                 **kwargs):
        # Text-only inputs use the vanilla Qwen3-Embedding native format
        # (no chat wrap, add_special_tokens=False, explicit <|endoftext|>,
        # left-padded). This keeps text embeddings in the same subspace
        # as stock Qwen3-Embedding-0.6B.
        # Exception: with native vision_mode and OMNI_NATIVE_FULL_CHAT=1,
        # text-only inputs go through the chat-wrapped path (the template the
        # Qwen3-VL-Embedding backbone was contrastively pretrained on).
        import os as _os_top
        _native_chat = (self.vision_mode == "native"
                        and _os_top.environ.get("OMNI_NATIVE_FULL_CHAT", "0") == "1")
        if audio is None and images is None and videos is None and not _native_chat:
            assert text is not None, "processor called with no inputs"
            if is_query:
                instr = instruction or (
                    "Given a web search query, retrieve relevant "
                    "passages that answer the query"
                )
                formatted = (f"Instruct: {instr}\nQuery: {text}"
                             if isinstance(text, str)
                             else [f"Instruct: {instr}\nQuery: {t}" for t in text])
            else:
                formatted = text
            tok = self.tokenizer
            prev_side = tok.padding_side
            tok.padding_side = "left"
            try:
                enc = tok(formatted, return_tensors=return_tensors,
                          padding=True, add_special_tokens=False)
            finally:
                tok.padding_side = prev_side
            # Default terminator is tok.eos_token_id (the evaluation pipeline
            # relies on it). Callers that need the Qwen3-Embedding native
            # retrieval recipe (trailing <|endoftext|>, matching
            # modeling_omni_embed.tokenize_text_native) pass `_eos_id`;
            # apply_chat_template does this.
            eos = tok.eos_token_id
            _eos_override = kwargs.get("_eos_id")
            if _eos_override is not None:
                eos = _eos_override
            B = enc["input_ids"].shape[0]
            eos_col = torch.full((B, 1), eos, dtype=enc["input_ids"].dtype)
            mask_col = torch.ones((B, 1), dtype=enc["attention_mask"].dtype)
            enc["input_ids"] = torch.cat([enc["input_ids"], eos_col], dim=1)
            enc["attention_mask"] = torch.cat([enc["attention_mask"], mask_col], dim=1)
            return enc

        # Media present — keep chat-wrapped so EOS pools at <|im_end|>.
        prompt, media_tensors = self._compose_prompt(text, audio, images, videos)
        # Native full-chat: right-pad to match Qwen3-VL-Embedding's native ST
        # (Qwen3VLProcessor.from_pretrained(..., padding_side='right')); pool
        # lands at the trailing `\n` after `assistant`.
        _native_chat = (self.vision_mode == "native"
                        and os.environ.get("OMNI_NATIVE_FULL_CHAT", "0") == "1")
        prev_side = self.tokenizer.padding_side
        if _native_chat:
            self.tokenizer.padding_side = "right"
        try:
            encoded = self.tokenizer(prompt, return_tensors=return_tensors, padding=True)
        finally:
            self.tokenizer.padding_side = prev_side
        encoded.update(media_tensors)
        return encoded


    # ── OpenAI-style message API ─────────────────────────────────────────
    @staticmethod
    def _load_audio(a, target_sr=16000):
        import numpy as _np
        if isinstance(a, _np.ndarray):
            return a.astype(_np.float32, copy=False).reshape(-1)
        if hasattr(a, "detach"):                       # torch.Tensor
            return a.detach().cpu().numpy().astype(_np.float32).reshape(-1)
        if isinstance(a, dict) and "array" in a:
            return OmniEmbedProcessor._load_audio(a["array"], target_sr)
        if isinstance(a, str):
            try:
                import soundfile as _sf
                wav, sr = _sf.read(a, dtype="float32", always_2d=False)
            except Exception:
                import librosa as _lb
                wav, sr = _lb.load(a, sr=target_sr, mono=True)
            wav = _np.asarray(wav, dtype=_np.float32)
            if wav.ndim > 1:
                wav = wav.mean(axis=-1)                # to mono
            if sr != target_sr:
                import librosa as _lb
                wav = _lb.resample(wav, orig_sr=sr, target_sr=target_sr)
            return wav.astype(_np.float32).reshape(-1)
        raise TypeError(f"unsupported audio input: {type(a)}")

    @staticmethod
    def _load_image(im):
        from PIL import Image as _Image
        if isinstance(im, str):
            if im.startswith(("http://", "https://")):
                import io, urllib.request
                with urllib.request.urlopen(im) as r:
                    return _Image.open(io.BytesIO(r.read())).convert("RGB")
            return _Image.open(im).convert("RGB")
        if isinstance(im, _Image.Image):
            return im.convert("RGB")
        import numpy as _np
        if isinstance(im, _np.ndarray):
            return _Image.fromarray(im).convert("RGB")
        raise TypeError(f"unsupported image input: {type(im)}")

    @classmethod
    def _load_video(cls, v, num_frames=None):
        """Accepts a list of frames (paths/PIL/arrays) or a video file path."""
        if isinstance(v, (list, tuple)):
            return [cls._load_image(f) for f in v]
        if isinstance(v, str):
            try:
                import av
            except ImportError as e:
                raise ImportError(
                    "Decoding a video file needs PyAV (`pip install av`). "
                    "Alternatively pass an explicit list of frames."
                ) from e
            from PIL import Image as _Image
            with av.open(v) as container:
                frames = [f.to_ndarray(format="rgb24")
                          for f in container.decode(video=0)]
            if not frames:
                raise ValueError(f"no decodable video frames in {v!r}")
            n = num_frames or min(len(frames), 8)
            idx = [int(round(i * (len(frames) - 1) / max(1, n - 1)))
                   for i in range(n)] if n > 1 else [0]
            return [_Image.fromarray(frames[i]) for i in idx]
        raise TypeError(f"unsupported video input: {type(v)}")

    def _parse_messages(self, messages):
        """OpenAI-style messages -> (text, audio, images, videos)."""
        texts, audio, images, videos = [], None, [], None
        if isinstance(messages, dict):
            messages = [messages]
        for msg in messages:
            content = msg.get("content") if isinstance(msg, dict) else msg
            if isinstance(content, str):
                texts.append(content)
                continue
            for part in (content or []):
                if isinstance(part, str):
                    texts.append(part)
                    continue
                ptype = part.get("type")
                if ptype == "text":
                    texts.append(part.get("text", ""))
                elif ptype == "audio":
                    if audio is not None:
                        raise ValueError("at most one audio item per call")
                    audio = self._load_audio(part.get("audio"))
                elif ptype in ("image", "image_url", "doc", "document"):
                    src = part.get(ptype) or part.get("image") or part.get("url")
                    images.append(self._load_image(src))
                elif ptype == "video":
                    if videos is not None:
                        raise ValueError("at most one video item per call")
                    videos = self._load_video(part.get("video"))
                else:
                    raise ValueError(f"unknown content part type: {ptype!r}")
        text = " ".join(t for t in texts if t) or None
        return text, audio, (images or None), videos

    def apply_chat_template(self, messages, role="passage", tokenize=True,
                            return_tensors="pt", instruction=None, **kwargs):
        """Encode OpenAI-style multimodal messages.

        role: "query" applies the retrieval instruction template; "passage"
              (aliases: "document", "corpus") is the corpus side.

        NOTE ON TEXT RECIPES. For a PURE-TEXT input this uses the native
        Qwen3-Embedding recipe, which is what reproduces the reported MTEB
        numbers. If your corpus contains media (text->audio, text->image,
        text->page retrieval), the query text MUST be encoded in the same
        subspace as the documents: pass text_recipe="chat" so both sides match.
        Mixing recipes across query and corpus collapses cosine to noise.
        """
        role_l = str(role).lower()
        if role_l in ("query", "q"):
            is_query = True
        elif role_l in ("passage", "document", "doc", "corpus", "p"):
            is_query = False
        else:
            raise ValueError(f"role must be 'query' or 'passage', got {role!r}")
        text, audio, images, videos = self._parse_messages(messages)
        if text is None and audio is None and images is None and videos is None:
            raise ValueError("no content found in messages")

        recipe = kwargs.pop("text_recipe", "native")
        is_pure_text = (audio is None and images is None and videos is None)
        if not tokenize:
            prompt, _ = self._compose_prompt(text, audio, images, videos)
            return prompt
        if is_pure_text and recipe == "chat":
            # Chat-wrapped text: same geometry as media documents.
            wrapped = "<|im_start|>user\n" + (text or "") + "\n<|im_end|>"
            return self.tokenizer(wrapped, return_tensors=return_tensors,
                                  padding=True)
        if is_pure_text:
            # The Qwen3-Embedding native retrieval recipe terminates with
            # <|endoftext|>, NOT <|im_end|> (== eos_token_id). Passed
            # explicitly; __call__'s default terminator is unchanged.
            _eos = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
            if _eos is None or _eos == self.tokenizer.unk_token_id:
                raise RuntimeError(
                    "tokenizer has no '<|endoftext|>' token - required for the "
                    "Qwen3-Embedding native retrieval recipe."
                )
            kwargs.setdefault("_eos_id", _eos)
        return self(text=text, audio=audio, images=images, videos=videos,
                    return_tensors=return_tensors, is_query=is_query,
                    instruction=instruction, **kwargs)

    def _compose_prompt(self, text, audio, images, videos):
        media_tensors = {}
        segments = []

        if audio is not None:
            audio_np = np.asarray(audio, dtype=np.float32)
            mel = self.whisper_fe(audio_np, sampling_rate=16000,
                                  return_tensors="pt")["input_features"]
            dasheng = torch.from_numpy(audio_np).view(1, 1, -1)

            mel_T = mel.shape[-1]
            n_chunks = max(1, (mel_T + WHISPER_MAX_MEL - 1) // WHISPER_MAX_MEL)
            n_tokens = n_chunks * self.tokens_per_encoder * 2
            segments.append(
                "<|audio_start|>" + ("<|audio_pad|>" * n_tokens) + "<|audio_end|>"
            )
            media_tensors["whisper_features"] = mel
            media_tensors["dasheng_audio"] = dasheng

        if images is not None and self.vision_mode != "none":
            imgs = images if isinstance(images, (list, tuple)) else [images]
            # Native vision_mode (Qwen3-VL backbone): replicate native ST's
            # qwen_vl_utils.process_vision_info pre-resize so pixel grid
            # matches the backbone's pretraining (1.84M-pixel cap, not the
            # processor's 1.31M default).
            if self.vision_mode == "native":
                from qwen_vl_utils.vision_process import fetch_image as _fetch
                imgs = [_fetch({"image": im, "min_pixels": 4 * 32 * 32,
                                "max_pixels": 1800 * 32 * 32},
                               image_patch_size=16)
                        for im in imgs]
                proc_out = self.image_processor(
                    images=[[im] for im in imgs], return_tensors="pt",
                    do_resize=False,
                )
            else:
                # Custom (0.9B): match train-time Resize((S,S)) on still images
                # so doc pages / images are in-distribution for the small Qwen3.5
                # encoder (native-resolution pages are out of distribution).
                # Bilinear =
                # torchvision Resize default used in data/dataset.py.
                _s = int(getattr(self, "still_image_size", 0) or 0)
                if _s:
                    from PIL import Image as _PILImage
                    imgs = [im.convert("RGB").resize((_s, _s), _PILImage.BILINEAR)
                            for im in imgs]
                proc_out = self.image_processor(
                    images=[[im] for im in imgs], return_tensors="pt",
                )
            pv = proc_out["pixel_values"]
            thw = proc_out["image_grid_thw"]
            merge = getattr(self.image_processor, "merge_size", 2)
            for i in range(thw.shape[0]):
                t, h, w = thw[i].tolist()
                # Native (Qwen3-VL) needs total_patches // merge**2 to match
                # backbone's strict tokens==features check on large images. Custom uses (h//merge)*(w//merge) — same
                # truncation as Qwen35VisionEncoderWrapper.forward_and_split.
                if self.vision_mode == "native":
                    n = (t * h * w) // (merge * merge)
                else:
                    n = t * (h // merge) * (w // merge)
                segments.append(
                    self.img_open + ("<|image_pad|>" * n) + self.img_close
                )
            # Key names chosen so they route to the correct forward arg:
            #   native mode  → backbone expects `pixel_values` + `image_grid_thw`
            #   custom mode  → our modeling also reads `pixel_values` + `image_grid_thw`
            media_tensors["pixel_values"] = pv
            media_tensors["image_grid_thw"] = thw

        if videos is not None and self.vision_mode != "none":
            vids = videos if (isinstance(videos, (list, tuple)) and len(videos) and
                              isinstance(videos[0], (list, tuple))) else [videos]
            proc_out = self.image_processor(images=vids, return_tensors="pt")
            media_tensors["pixel_values_videos"] = proc_out["pixel_values"]
            media_tensors["video_grid_thw"] = proc_out["image_grid_thw"]
            # Token count — native mode should use backbone's own rule
            # (grid-based); custom uses our fixed num_video_tokens, UNLESS
            # video_as_images: then each frame emits image-like
            # pad tokens summed per video + a video_num_frames tensor so the
            # model routes frames through the per-frame video_projector.
            if self.vision_mode == "custom" and self.video_as_images:
                merge = getattr(self.image_processor, "merge_size", 2)
                grid = proc_out["image_grid_thw"]
                per_row = [t * (h // merge) * (w // merge)
                           for t, h, w in grid.tolist()]
                off = 0
                num_frames = []
                for v in vids:
                    f = len(v)
                    n = sum(per_row[off:off + f])
                    segments.append(
                        self.vid_open + ("<|video_pad|>" * n) + self.vid_close
                    )
                    num_frames.append(f)
                    off += f
                media_tensors["video_num_frames"] = torch.tensor(
                    num_frames, dtype=torch.long
                )
            elif self.vision_mode == "custom":
                for _ in vids:
                    segments.append(
                        self.vid_open
                        + ("<|video_pad|>" * self.num_video_tokens)
                        + self.vid_close
                    )
            else:
                merge = getattr(self.image_processor, "merge_size", 2)
                for i in range(proc_out["image_grid_thw"].shape[0]):
                    t, h, w = proc_out["image_grid_thw"][i].tolist()
                    # Same conditional as image branch above.
                    if self.vision_mode == "native":
                        n = (t * h * w) // (merge * merge)
                    else:
                        n = t * (h // merge) * (w // merge)
                    segments.append(
                        self.vid_open + ("<|video_pad|>" * n) + self.vid_close
                    )

        if text:
            segments.append(text)

        body = "\n".join(s for s in segments if s)
        # OMNI_NATIVE_FULL_CHAT=1 reproduces Qwen3-VL-Embedding's standalone
        # ST chat template (system + user + assistant generation header) so
        # the frozen Qwen3-VL backbone pools at the same position it was
        # contrastively pretrained on. Off by default — the training-time
        # format was the simpler user-only template (data/collator.py).
        import os as _os
        if (self.vision_mode == "native"
                and _os.environ.get("OMNI_NATIVE_FULL_CHAT", "0") == "1"):
            # Byte-identical to Qwen3-VL-Embedding's native ST chat template
            # (rendered by `apply_chat_template(..., add_generation_prompt=True)`).
            # No trailing <|endoftext|> — native pools at the trailing `\n`
            # after `assistant` (last attended token under right-padding).
            prompt = (
                "<|im_start|>system\nRepresent the user's input.<|im_end|>\n"
                f"<|im_start|>user\n{body}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        else:
            prompt = f"<|im_start|>user\n{body}\n<|im_end|>"
        return prompt, media_tensors
'''
    with open(os.path.join(output_dir, "processing_omni_embed.py"), "w") as f:
        f.write(code)


if __name__ == "__main__":
    main()
