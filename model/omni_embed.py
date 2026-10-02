"""
OmniEmbedModel — Multimodal embedding model.

Wraps Qwen3-Embedding (or any causal-LM embedding backbone) and adds media
encoders whose outputs are injected into the text token sequence. The
backbone's native causal forward pass and hidden_size-sized EOS pooling are
preserved end-to-end — no projection head — so the text-only path through
this model is byte-identical to running the backbone standalone.

Flow:
  - Whisper + Dasheng audio encoders → ConvDownProjector interleave → audio tokens
  - Qwen3.5 vision encoder → merger → (optional) VideoSpatialPooler → visual tokens
  - Media tokens replace placeholders in the text sequence (one-hot matmul)
  - Backbone runs its full causal forward over the fused sequence
  - EOS pooling → L2-normalized embedding (hidden_size-dim)

Training strategy (phased):
  Phase 1 (first 20%): Only media projectors trainable; backbone + encoders frozen
  Phase 2 (remaining 80%): + LoRA on media encoders (backbone stays frozen
    to preserve pretrained text capability — the teacher in self-distillation)
"""

import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from transformers import AutoModel, AutoTokenizer

from .encoders import WhisperSpeechEncoderSeq, DashengEncoder, WHISPER_MAX_MEL
from .projectors import ConvDownProjector
from .vision import Qwen35VisionEncoder, VideoSpatialPooler, DimProjector
from .pooling import EOSPooling
from .lora import LoRALinear, inject_lora


# Vanilla Qwen3-Embedding-0.6B eval format. The reference loader tokenizes
# with add_special_tokens=False, appends <|endoftext|> explicitly, and
# left-pads batches. Queries get an "Instruct: …\nQuery: …" prefix; passages
# are raw. Teacher-cache and MTEB text paths use this format so embeddings
# land in the same subspace as vanilla Qwen3-Embedding-0.6B.
QWEN3_QUERY_TEMPLATE = "Instruct: {instruction}\nQuery: {query}"
DEFAULT_TASK_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)

AUDIO_START_TOKEN = "<|audio_start|>"
AUDIO_END_TOKEN = "<|audio_end|>"
AUDIO_PAD_TOKEN = "<|audio_pad|>"
# Custom vision tokens — added only in vision_mode == "custom".
# In "native" mode we rely on the backbone tokenizer's own vision tokens
# (Qwen-VL convention: <|vision_start|> / <|image_pad|> / <|vision_end|>).
CUSTOM_IMAGE_START_TOKEN = "<|image_start|>"
CUSTOM_IMAGE_END_TOKEN = "<|image_end|>"
CUSTOM_IMAGE_PAD_TOKEN = "<|image_pad|>"
CUSTOM_VIDEO_START_TOKEN = "<|video_start|>"
CUSTOM_VIDEO_END_TOKEN = "<|video_end|>"
CUSTOM_VIDEO_PAD_TOKEN = "<|video_pad|>"

AUDIO_SPECIAL_TOKENS = [
    AUDIO_START_TOKEN, AUDIO_END_TOKEN, AUDIO_PAD_TOKEN,
]
CUSTOM_VISION_SPECIAL_TOKENS = [
    CUSTOM_IMAGE_START_TOKEN, CUSTOM_IMAGE_END_TOKEN, CUSTOM_IMAGE_PAD_TOKEN,
    CUSTOM_VIDEO_START_TOKEN, CUSTOM_VIDEO_END_TOKEN, CUSTOM_VIDEO_PAD_TOKEN,
]


class OmniEmbedModel(nn.Module):
    """Multimodal embedding model with backbone-agnostic design.

    Args:
        config: dict with keys:
            model.backbone: str — HuggingFace model name (e.g. "Qwen/Qwen3-Embedding-0.6B")
            vision.encoder: str — vision encoder model name (default "Qwen/Qwen3.5-0.8B")
            vision.num_video_tokens: int — video token budget after pooling (default 196)
            audio.whisper: str — Whisper model name
            audio.dasheng: str — Dasheng model name
            audio.tokens_per_encoder: int — tokens per encoder per 30s chunk (default 128)
            lora: dict — LoRA config (r, alpha, dropout)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        model_cfg = config.get("model", {})
        vision_cfg = config.get("vision", {})
        audio_cfg = config.get("audio", {})

        # vision_mode controls the vision stack:
        #   custom — add our own Qwen3.5-0.8B vision tower + projectors
        #            (for text-only backbones like Qwen3-Embedding-0.6B)
        #   native — use the backbone's built-in visual module
        #            (for VL backbones like Qwen/Qwen3-VL-Embedding-2B)
        #   none   — audio-only, no vision at all
        self.vision_mode = model_cfg.get("vision_mode", "custom")
        if self.vision_mode not in ("custom", "native", "none"):
            raise ValueError(f"vision_mode must be custom|native|none, got {self.vision_mode!r}")

        # ── Backbone (any embedding model) ──
        backbone_name = model_cfg.get("backbone", "Qwen/Qwen3-Embedding-0.6B")
        self.backbone = AutoModel.from_pretrained(
            backbone_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
        )
        # Hidden size lives at the top level on text-only embedders
        # (Qwen3-Embedding-0.6B) but under config.text_config on VL configs
        # (Qwen3-VL-Embedding-*). Handle both.
        bb_cfg = self.backbone.config
        self.hidden_size = getattr(bb_cfg, "hidden_size", None)
        if self.hidden_size is None and hasattr(bb_cfg, "text_config"):
            self.hidden_size = bb_cfg.text_config.hidden_size
        # Keep native causal attention — Qwen3-Embedding was trained causal.
        # EOS pooling only needs the last token, which sees the full sequence
        # via causal attention.
        # Backbone gradient checkpointing is off by default: the backbone is
        # frozen, so there is nothing to recompute. The text-backbone ablation
        # enables it (see activate_text_backbone_lora).

        # ── Tokenizer + special tokens ──
        self.tokenizer = AutoTokenizer.from_pretrained(
            backbone_name, trust_remote_code=True,
        )
        # Register audio tokens (always); register custom vision tokens only
        # when we're adding our own vision stack. In native mode the backbone
        # tokenizer already carries vision tokens.
        tokens_to_add = list(AUDIO_SPECIAL_TOKENS)
        if self.vision_mode == "custom":
            tokens_to_add += CUSTOM_VISION_SPECIAL_TOKENS
        n_added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": tokens_to_add}
        )
        if n_added > 0:
            self.backbone.resize_token_embeddings(len(self.tokenizer))

        # Cache audio pad id (always present)
        self._audio_pad_id = self.tokenizer.convert_tokens_to_ids(AUDIO_PAD_TOKEN)

        # Cache vision pad ids based on mode
        self._image_pad_id = None
        self._video_pad_id = None
        if self.vision_mode == "custom":
            self._image_pad_id = self.tokenizer.convert_tokens_to_ids(CUSTOM_IMAGE_PAD_TOKEN)
            self._video_pad_id = self.tokenizer.convert_tokens_to_ids(CUSTOM_VIDEO_PAD_TOKEN)
        elif self.vision_mode == "native":
            # Qwen-VL convention. If your VL backbone uses different tokens,
            # override via config.model.native_image_pad_token etc.
            img_tok = model_cfg.get("native_image_pad_token", "<|image_pad|>")
            vid_tok = model_cfg.get("native_video_pad_token", "<|video_pad|>")
            self._image_pad_id = self.tokenizer.convert_tokens_to_ids(img_tok)
            self._video_pad_id = self.tokenizer.convert_tokens_to_ids(vid_tok)
            # Qwen-VL tokenizers have unk_token_id=None, so test for truthy id.
            unk = self.tokenizer.unk_token_id
            def _missing(tid):
                return tid is None or (unk is not None and tid == unk)
            if _missing(self._image_pad_id):
                raise ValueError(
                    f"vision_mode=native but token {img_tok!r} not found in backbone "
                    f"tokenizer vocab. Override model.native_image_pad_token in config."
                )
            if _missing(self._video_pad_id):
                raise ValueError(
                    f"vision_mode=native but token {vid_tok!r} not found in backbone "
                    f"tokenizer vocab. Override model.native_video_pad_token in config."
                )

        # ── Vision stack ──
        self.vision_encoder = None
        self.image_projector = None
        self.video_pooler = None
        self.video_projector = None
        self.num_video_tokens = vision_cfg.get("num_video_tokens", 196)

        if self.vision_mode == "custom":
            vision_model = vision_cfg.get("encoder", "Qwen/Qwen3.5-0.8B")
            self.vision_encoder = Qwen35VisionEncoder(vision_model)
            self.vision_encoder.load_pretrained_weights()
            vision_out_dim = self.vision_encoder.output_dim  # 1024

            self.image_projector = DimProjector(vision_out_dim, self.hidden_size)
            self.video_pooler = VideoSpatialPooler(
                input_dim=vision_out_dim,
                num_output_tokens=self.num_video_tokens,
                num_heads=16,
            )
            self.video_projector = DimProjector(vision_out_dim, self.hidden_size)
        elif self.vision_mode == "native":
            # Backbone provides its own visual module — no projectors, no pooler.
            # Expect self.backbone.visual to exist (Qwen-VL-Embedding family).
            if not hasattr(self.backbone, "visual"):
                raise ValueError(
                    f"vision_mode=native but backbone {backbone_name!r} has no "
                    f"`visual` attribute. Expected a VL backbone like "
                    f"Qwen/Qwen3-VL-Embedding-2B."
                )

        # ── Audio encoders ──
        whisper_name = audio_cfg.get("whisper", "openai/whisper-small")
        dasheng_name = audio_cfg.get("dasheng", "mispeech/dasheng-base")

        self.whisper_encoder = WhisperSpeechEncoderSeq(whisper_name)
        self.whisper_encoder.load_pretrained_weights()

        self.dasheng_encoder = DashengEncoder(dasheng_name)
        self.dasheng_encoder.load_pretrained_weights()

        whisper_dim = self.whisper_encoder.output_dim
        dasheng_dim = self.dasheng_encoder.output_dim

        # Audio interleave projectors
        tokens_per_encoder = audio_cfg.get("tokens_per_encoder", 128)
        self.interleave_tokens = tokens_per_encoder
        self.num_audio_tokens_per_chunk = tokens_per_encoder * 2

        self.whisper_down_proj = ConvDownProjector(
            input_dim=whisper_dim,
            output_dim=self.hidden_size,
            input_tokens=1500,
            target_tokens=tokens_per_encoder,
        )
        self.dasheng_down_proj = ConvDownProjector(
            input_dim=dasheng_dim,
            output_dim=self.hidden_size,
            input_tokens=750,
            target_tokens=tokens_per_encoder,
        )

        # ── Embedding head ──
        # No projection head: the backbone's EOS-pooled, L2-normalized vector
        # IS the embedding. This keeps the text path identical to the stock
        # backbone and lets self-distillation target a matching representation.
        # MRL slices are taken directly on the backbone's hidden_size-dim
        # pooled vector — max(mrl_dims) must therefore equal hidden_size.
        # Each backbone ships its own config with an appropriate dim grid.
        self.pooling = EOSPooling()
        self.embedding_dim = self.hidden_size
        mrl_dims = list(model_cfg.get("mrl_dims_coarse", []))
        if mrl_dims and max(mrl_dims) != self.hidden_size:
            raise ValueError(
                f"model.mrl_dims_coarse max ({max(mrl_dims)}) must equal "
                f"backbone hidden_size ({self.hidden_size}). Update the "
                f"config for this backbone (see configs/omni_embed_mini_*.yaml "
                f"for examples)."
            )

        # ── Phase tracking ──
        # _has_encoder_lora: gates the encoder GRAD-CONTEXT (grads flow through
        #   encoders only when True) — set for lora and full-FT, NOT for frozen.
        # _phase2_active: gates the miner start + phase transition — set for ALL
        #   phase-2 modes (lora/full/frozen) so mining still runs when encoders
        #   are frozen.
        self._has_encoder_lora = False
        self._phase2_active = False
        self._encoder_full_ft = False
        self._phase = 1
        # Text-backbone LoRA ablation: when True, LoRA is injected into the
        # text backbone at phase 2. Normally the backbone is never adapted —
        # it is the frozen self-distillation teacher. This ablation measures
        # the text forgetting that adapting it induces
        # (see activate_text_backbone_lora).
        self._has_text_backbone_lora = False
        # Full text-backbone ablation: unfreeze the entire text backbone
        # instead of LoRA-adapting it (see activate_text_backbone_full).
        self._has_text_backbone_full = False

        # ── Freeze everything, then unfreeze projectors ──
        self._freeze_all()
        self._unfreeze_projectors()

        # ── Native-mode audio injection hook ──
        # VL backbones (e.g. Qwen3-VL-Embedding) enforce exclusive (input_ids,
        # inputs_embeds) AND require input_ids to compute 3-D M-RoPE on first
        # forward. So we can't pre-build inputs_embeds externally. Instead,
        # register a forward hook on the backbone's embed_tokens module: the
        # backbone calls embed_tokens(input_ids) internally, our hook fires
        # on that output and splices audio features at <|audio_pad|> rows.
        # Vision substitution + MRoPE then run natively on the spliced embeds.
        # threading.local so the miner thread (which calls encode_text on
        # plain text with no audio_pad rows) sees its own slot — always None,
        # so the hook is a no-op on its forwards even while the main thread
        # has audio state pending. A shared attribute would let the miner
        # see the main thread's stale audio_input_ids and IndexError when
        # the batch dims differ.
        self._pending = threading.local()
        if self.vision_mode == "native":
            self.backbone.get_input_embeddings().register_forward_hook(
                self._audio_inject_hook
            )

    def _audio_inject_hook(self, module, args, output):
        """Forward hook on embed_tokens that injects audio at audio_pad rows."""
        feats = getattr(self._pending, "audio_feats", None)
        input_ids = getattr(self._pending, "audio_input_ids", None)
        if feats is None or input_ids is None:
            return output
        out = output.clone()
        mask = (input_ids == self._audio_pad_id)
        for b in range(out.shape[0]):
            pos = mask[b].nonzero(as_tuple=True)[0]
            f_b = feats[b]
            n = min(pos.numel(), f_b.shape[0])
            if n > 0:
                out[b, pos[:n]] = f_b[:n].to(out.dtype)
        return out

    # ── Freeze/Unfreeze utilities ──────────────────────────────

    def _freeze_all(self):
        """Freeze all parameters."""
        for p in self.parameters():
            p.requires_grad = False

    def _unfreeze_projectors(self):
        """Unfreeze media projectors and poolers (Phase 1 trainable set).

        The backbone and media encoders stay frozen. No MRL head exists —
        the backbone's pooled hidden state is the embedding. Vision
        projectors exist only in vision_mode == custom.
        """
        modules = [self.whisper_down_proj, self.dasheng_down_proj]
        if self.vision_mode == "custom":
            modules += [self.image_projector, self.video_pooler, self.video_projector]
        for module in modules:
            for p in module.parameters():
                p.requires_grad = True

    def activate_phase2(self, lora_config=None, freeze_vision=None):
        """Inject LoRA adapters into the media encoders for Phase 2.

        Called at 20% of training. Projectors remain trainable. The backbone
        is NEVER adapted — keeping it frozen preserves Qwen3-Embedding's
        pretrained text capability, which is also the teacher signal in
        self-distillation.

        Args:
            lora_config: dict with keys r, alpha, dropout (defaults: 16, 32, 0.05)
            freeze_vision: if True, DON'T inject LoRA into the custom Qwen3.5
                vision tower (keep it frozen in phase 2); audio encoders still
                get LoRA. Prevents phase-2 vision-encoder drift (the
                projectors are fit against the frozen encoder in phase 1).
                When None, auto-read from config training.freeze_vision_in_phase2
                so export/eval reproduce the training-time choice with no extra
                plumbing. No-op in native mode (vision is the frozen backbone).
        """
        if lora_config is None:
            lora_config = self.config.get("lora", {})
        if freeze_vision is None:
            freeze_vision = self.config.get("training", {}).get(
                "freeze_vision_in_phase2", False)
        r = lora_config.get("r", 16)
        alpha = lora_config.get("alpha", 32)
        dropout = lora_config.get("dropout", 0.05)

        # LoRA on custom Qwen3.5 vision encoder (only in vision_mode=custom).
        # In native mode the backbone's visual module stays entirely frozen;
        # adapting it could drift its output away from what the backbone
        # internally expects. freeze_vision applies the same "keep frozen"
        # choice to the custom tower (audio LoRA below is unaffected).
        if self.vision_mode == "custom" and not freeze_vision:
            inject_lora(
                self.vision_encoder.visual,
                ["attn.qkv", "attn.proj"],
                r=r, alpha=alpha, dropout=dropout,
            )
            # Gradient checkpointing on `self.visual` lowers phase-2 video
            # memory. It is safe because the miner's `forward_lock` is
            # acquired by main-thread forward+backward AND by the miner per
            # encode-batch, so the miner cannot run between main's forward
            # and backward (recompute sees the same state).
            self.vision_encoder.enable_gradient_checkpointing()
        self._has_encoder_lora = True

        # LoRA on Whisper attention
        inject_lora(
            self.whisper_encoder.model,
            ["k_proj", "v_proj", "q_proj", "out_proj"],
            r=r, alpha=alpha, dropout=dropout,
        )

        # LoRA on Dasheng attention
        inject_lora(
            self.dasheng_encoder.model,
            ["attn.qkv", "attn.proj"],
            r=r, alpha=alpha, dropout=dropout,
        )

        self._phase = 2
        self._phase2_active = True

    def activate_phase2_full(self):
        """Phase-2 encoder ABLATION: full fine-tune (no LoRA).

        Unfreezes the media-encoder weights directly instead of injecting
        adapters. Projectors remain trainable; the backbone stays frozen
        (it is the self-distillation teacher and must not drift). Enables the
        grad-context (`_has_encoder_lora=True`) so gradients flow through the
        encoders, and gradient checkpointing on the custom vision tower to fit
        phase-2 video in HBM (mirrors the LoRA path).
        """
        # Move the encoders onto the model's compute device BEFORE unfreezing.
        # If a param is still on CPU when it enters the optimizer, Adam creates
        # its moment buffers (exp_avg/exp_avg_sq) on CPU while gradients arrive
        # on GPU, and optimizer.step() dies with "tensors on cuda:N and cpu".
        # Derive the device from a param already on GPU (projectors/backbone are
        # there after accelerate.prepare()).
        device = next((p.device for p in self.parameters()
                       if p.device.type == "cuda"), None)
        modules = [self.whisper_encoder, self.dasheng_encoder]
        if self.vision_mode == "custom" and self.vision_encoder is not None:
            modules.append(self.vision_encoder)
            self.vision_encoder.enable_gradient_checkpointing()
        for module in modules:
            if device is not None:
                module.to(device)
            for p in module.parameters():
                p.requires_grad = True
        self._encoder_full_ft = True
        self._has_encoder_lora = True   # gate: grads flow through encoders
        self._phase = 2
        self._phase2_active = True

    def activate_phase2_frozen(self):
        """Phase-2 encoder ABLATION: encoders stay frozen (no adaptation).

        No LoRA, no unfreezing — only the projectors keep training (as in
        Phase 1). `_has_encoder_lora` stays False so the encoder forward stays
        under `no_grad` (no wasted activation memory). `_phase2_active` is set
        so the hard-negative miner still starts.
        """
        self._has_encoder_lora = False  # encoders remain under no_grad
        self._phase = 2
        self._phase2_active = True

    def get_encoder_full_params(self):
        """Return media-encoder weights made trainable by activate_phase2_full."""
        params = []
        modules = [self.whisper_encoder, self.dasheng_encoder]
        if self.vision_mode == "custom" and self.vision_encoder is not None:
            modules.append(self.vision_encoder)
        for module in modules:
            for p in module.parameters():
                if p.requires_grad:
                    params.append(p)
        return params

    def get_phase2_lora_params(self):
        """Return LoRA parameters for adding to optimizer at phase transition."""
        lora_params = []
        for name, p in self.named_parameters():
            if "lora_" in name and p.requires_grad:
                lora_params.append(p)
        return lora_params

    def get_encoder_lora_params(self):
        """Return encoder LoRA params (whisper, dasheng, and custom vision if any)."""
        params = []
        modules = [self.whisper_encoder, self.dasheng_encoder]
        if self.vision_mode == "custom" and self.vision_encoder is not None:
            modules.append(self.vision_encoder)
        for module in modules:
            for name, p in module.named_parameters():
                if "lora_" in name and p.requires_grad:
                    params.append(p)
        return params

    def activate_text_backbone_lora(self, lora_config=None):
        """Forgetting ablation: inject LoRA into the (normally frozen) text backbone.

        Normally the backbone is never adapted (see activate_phase2's docstring).
        This method deliberately adapts it to demonstrate that touching the text
        path induces forgetting — the text/MTEB score is expected to DEGRADE vs
        the frozen main recipe. Safe to combine with any encoder_adapt mode; it
        only adds params on the backbone.

        The self-distillation teacher is a precomputed cache of the frozen
        backbone (and lora_B is zero-init, so injection is a no-op at t=0), so
        the teacher stays fixed while only the student backbone adapts. The main
        training forward runs the backbone WITHOUT no_grad already, so injecting
        trainable LoRA here is sufficient for gradients to flow — no grad-context
        change is needed (unlike the media encoders).

        Returns the newly-added backbone LoRA params (for optimizer + DDP
        broadcast), mirroring activate_phase2/get_encoder_lora_params.
        """
        if lora_config is None:
            lora_config = self.config.get("lora", {})
        r = lora_config.get("r", 16)
        alpha = lora_config.get("alpha", 32)
        dropout = lora_config.get("dropout", 0.0)
        # Qwen3 attention projections. inject_lora matches by dotted-name
        # suffix, scoped to self.backbone, so media-encoder q/k/v/o are not
        # touched (those live under whisper_encoder/dasheng_encoder/vision).
        inject_lora(
            self.backbone,
            ["q_proj", "k_proj", "v_proj", "o_proj"],
            r=r, alpha=alpha, dropout=dropout,
        )
        # inject_lora creates adapters in float32 (lora.py's default, for
        # optimizer-update precision under autocast). The main training forward
        # runs under bf16 autocast so that is fine there — BUT the hard-negative
        # miner encodes captions through this same backbone from a BACKGROUND
        # thread with NO autocast active (encode_text is @torch.no_grad and
        # unwrapped), so a float32 adapter hits F.linear(bf16_input,
        # float32_weight) → "mat1 and mat2 have the same dtype: BFloat16 !=
        # float". Cast the backbone adapters to the backbone's own dtype so the
        # forward is dtype-consistent with AND without autocast. (Media-encoder
        # LoRA avoids this only because its miner encode path runs under
        # autocast; the text-mining path does not.)
        bb_dtype = self.backbone.get_input_embeddings().weight.dtype
        new_params = self.get_text_backbone_lora_params()
        for p in new_params:
            p.data = p.data.to(bb_dtype)
        self._has_text_backbone_lora = True
        # Backbone gradient checkpointing: with the backbone now TRAINABLE (LoRA),
        # the backward would store activations for the entire backbone at seq_len
        # up to 4096, which can OOM on the 2.3B backbone. Recompute-in-backward
        # instead. Needs use_reentrant=False + enable_input_require_grads so the
        # checkpointed graph connects through the frozen input embeddings to the
        # LoRA params. Skipped under the miner's no_grad forward.
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            try:
                self.backbone.enable_input_require_grads()
                self.backbone.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
            except Exception as e:
                print(f"[text_backbone_lora] gradient checkpointing not enabled: {e}")
        return new_params

    def get_text_backbone_lora_params(self):
        """Return text-backbone LoRA params. Empty unless activated."""
        return [p for name, p in self.backbone.named_parameters()
                if "lora_" in name and p.requires_grad]

    def activate_text_backbone_full(self):
        """Forgetting ablation: unfreeze the entire text backbone (full fine-tune).

        Sibling of activate_text_backbone_lora — instead of injecting low-rank
        adapters, this trains every backbone weight directly.

        No dtype cast is needed (unlike the LoRA variant): the backbone weights are already
        the backbone's native dtype (bf16), so the miner's non-autocast forward
        stays dtype-consistent. The self-distillation teacher is a precomputed
        cache of the ORIGINAL frozen backbone, so it is unaffected — only the
        student backbone drifts. The backbone stays in eval() (Qwen3 has no
        dropout/BN, so eval vs train does not change the forward); gradients
        flow to the weights regardless of module mode.

        Returns the newly-trainable backbone params (for optimizer + DDP
        broadcast), mirroring activate_text_backbone_lora.
        """
        for p in self.backbone.parameters():
            p.requires_grad = True
        self._has_text_backbone_full = True
        return self.get_text_backbone_full_params()

    def get_text_backbone_full_params(self):
        """Return all trainable backbone params (full ablation). Empty unless activated."""
        return [p for name, p in self.backbone.named_parameters()
                if p.requires_grad]

    def train(self, mode=True):
        """Override train() to keep frozen modules in eval mode."""
        super().train(mode)
        # Backbone always stays in eval mode (Qwen3 has no dropout/BN).
        self.backbone.eval()
        # Encoders in eval by default (their LoRA submodules train). Exception:
        # the full-FT ablation trains the encoder weights directly, so the
        # encoders must follow the requested train/eval mode.
        enc_mode = mode if getattr(self, "_encoder_full_ft", False) else False
        self.whisper_encoder.train(enc_mode)
        self.dasheng_encoder.train(enc_mode)
        if self.vision_encoder is not None:
            self.vision_encoder.train(enc_mode)
        # Re-enable training for LoRA layers
        for module in self.modules():
            if isinstance(module, LoRALinear):
                module.lora_A.train(mode)
                module.lora_B.train(mode)
                module.dropout.train(mode)
        return self

    def print_param_summary(self):
        """Print parameter count summary."""
        total = 0
        trainable = 0
        components = {}
        for name, p in self.named_parameters():
            n = p.numel()
            total += n
            component = name.split(".")[0]
            if component not in components:
                components[component] = {"total": 0, "trainable": 0}
            components[component]["total"] += n
            if p.requires_grad:
                trainable += n
                components[component]["trainable"] += n

        print(f"\n{'Component':<30} {'Total':>12} {'Trainable':>12} {'%':>6}")
        print("-" * 64)
        for comp, counts in sorted(components.items()):
            pct = 100 * counts["trainable"] / max(counts["total"], 1)
            print(f"{comp:<30} {counts['total']:>12,} {counts['trainable']:>12,} {pct:>5.1f}%")
        print("-" * 64)
        pct = 100 * trainable / max(total, 1)
        print(f"{'TOTAL':<30} {total:>12,} {trainable:>12,} {pct:>5.1f}%")

    # ── Audio encoding (interleave) ────────────────────────────

    def _encode_audio(self, input_features, dasheng_audio):
        """Interleave Whisper + Dasheng → audio tokens.

        Args:
            input_features: (B, n_mels, T_mel) — Whisper mel features
            dasheng_audio: (B, 1, T_samples) — raw 16kHz audio

        Returns:
            list of (T_i, hidden_size) tensors — one per sample
        """
        B = input_features.shape[0]
        mel_T = input_features.shape[-1]
        samples_per_chunk = int(30.0 * 16000)
        n_chunks = max(1, (mel_T + WHISPER_MAX_MEL - 1) // WHISPER_MAX_MEL)
        ctx = nullcontext() if self._has_encoder_lora else torch.no_grad()

        all_chunk_outputs = []
        for c in range(n_chunks):
            # Whisper chunk
            mel_start = c * WHISPER_MAX_MEL
            mel_end = min((c + 1) * WHISPER_MAX_MEL, mel_T)
            mel_chunk = input_features[:, :, mel_start:mel_end]
            if mel_chunk.shape[-1] < WHISPER_MAX_MEL:
                mel_chunk = F.pad(mel_chunk, (0, WHISPER_MAX_MEL - mel_chunk.shape[-1]))

            with ctx:
                w_hidden = self.whisper_encoder(mel_chunk)

            # Dasheng chunk
            audio_start = c * samples_per_chunk
            audio_end = min((c + 1) * samples_per_chunk, dasheng_audio.shape[-1])
            audio_chunk = dasheng_audio[:, :, audio_start:audio_end]
            # ROCm cuFFT raises HIPFFT_PARSE_ERROR on 8000-sample STFT for
            # short clips. 16000 (1 sec) matches
            # dasheng's standalone wrapper and plans cleanly on HIP.
            min_samples = 16000
            if audio_chunk.shape[-1] < min_samples:
                audio_chunk = F.pad(audio_chunk, (0, min_samples - audio_chunk.shape[-1]))
            with ctx:
                d_hidden = self.dasheng_encoder(audio_chunk.squeeze(1))

            # Project
            w_proj = self.whisper_down_proj(w_hidden)
            d_proj = self.dasheng_down_proj(d_hidden)

            # Interleave: w1,d1,w2,d2,...
            B_c, T, D = w_proj.shape
            interleaved = torch.stack([w_proj, d_proj], dim=2)
            interleaved = interleaved.reshape(B_c, T * 2, D)

            all_chunk_outputs.append(interleaved)

        out = torch.cat(all_chunk_outputs, dim=1)
        return [out[i] for i in range(B)]

    # ── Vision encoding ────────────────────────────────────────

    def _encode_image(self, image_pixel_values, image_grid_thw):
        """Qwen3.5 vision → per-image tokens → project.

        Accepts pre-processed patches from Qwen3.5's native image processor.
        Token count per image is dynamic (depends on resolution/aspect ratio).

        Args:
            image_pixel_values: (total_patches, patch_dim) — from native processor
            image_grid_thw: (B, 3) — [t, h, w] per image

        Returns:
            list of (num_tokens_i, hidden_size) — one tensor per image
        """
        ctx = nullcontext() if self._has_encoder_lora else torch.no_grad()
        with ctx:
            features = self.vision_encoder.forward_and_split(
                image_pixel_values, image_grid_thw
            )
        return [self.image_projector(f) for f in features]

    def _encode_video(self, video_pixel_values, video_grid_thw, video_num_frames=None):
        """Qwen3.5 vision → video tokens.

        Two modes:
          • video_num_frames given (video_as_images): each frame's tokens go
            through the per-frame `video_projector` (no pooler), concatenated
            per video. Mirrors `_encode_image`.
            Returns list[(N_i, hidden)] — one variable-length tensor per video.
          • pooled (video_num_frames None): pool per-frame features to a fixed
            num_video_tokens via VideoSpatialPooler.
            Returns (B, num_video_tokens, hidden).

        Args:
            video_pixel_values: (total_patches, patch_dim) — from native processor
            video_grid_thw: (N_frames, 3) — one [1,h,w] row per frame
            video_num_frames: optional list[int] — frames per video (video_as_images)
        """
        ctx = nullcontext() if self._has_encoder_lora else torch.no_grad()
        with ctx:
            features = self.vision_encoder.forward_and_split(
                video_pixel_values, video_grid_thw
            )

        # ── video_as_images: per-frame projector, grouped per video (no pooler) ──
        if video_num_frames is not None:
            out, off = [], 0
            for f in video_num_frames:
                f = int(f)
                toks = torch.cat(
                    [self.video_projector(features[off + j]) for j in range(f)],
                    dim=0,
                )  # (F * tokens_per_frame, hidden)
                out.append(toks)
                off += f
            return out

        # ── Pooled path (VideoSpatialPooler) ──
        # Pad variable-length features to batch tensor
        B = len(features)
        max_tokens = max(f.shape[0] for f in features)
        dim = features[0].shape[1]
        device = features[0].device
        dtype = features[0].dtype

        batched = torch.zeros(B, max_tokens, dim, device=device, dtype=dtype)
        attn_mask = torch.zeros(B, max_tokens, device=device, dtype=torch.long)
        for i, feat in enumerate(features):
            batched[i, :feat.shape[0]] = feat
            attn_mask[i, :feat.shape[0]] = 1

        # Spatial pooling: variable tokens → fixed num_video_tokens
        pooled = self.video_pooler(batched, attention_mask=attn_mask)
        return self.video_projector(pooled)

    # ── Gradient-preserving token replacement ────────────────────

    def _build_inputs_embeds(
        self, input_ids, input_features=None, dasheng_audio=None,
        image_pixel_values=None, image_grid_thw=None,
        video_pixel_values=None, video_grid_thw=None,
        video_num_frames=None,
    ):
        """Embed tokens and replace media placeholders with encoded features.

        Uses one-hot matmul to place projector outputs at media token
        positions, preserving the gradient graph through the projectors.
        In-place indexed assignment (tensor[i, pos] = val) severs gradients;
        matmul (one_hot @ features) is a standard differentiable op.
        """
        B = input_ids.shape[0]
        with torch.no_grad():
            text_embeds = self.backbone.get_input_embeddings()(input_ids).detach()

        S, D = text_embeds.shape[1], text_embeds.shape[2]
        device = text_embeds.device
        dtype = text_embeds.dtype

        # Collect (positions, features) per sample — no in-place ops
        sample_media = [[] for _ in range(B)]

        # Audio
        if input_features is not None and dasheng_audio is not None:
            audio_projected = self._encode_audio(input_features, dasheng_audio)
            token_mask = input_ids == self._audio_pad_id
            for i in range(B):
                positions = token_mask[i].nonzero(as_tuple=True)[0]
                n = min(len(positions), audio_projected[i].shape[0])
                if n > 0:
                    sample_media[i].append((positions[:n], audio_projected[i][:n]))

        # Image / Video — only inject via matmul in vision_mode=custom.
        # In native mode these tensors are forwarded to the backbone below
        # so its built-in visual module + MRoPE position logic stay intact.
        if self.vision_mode == "custom":
            if image_pixel_values is not None and image_grid_thw is not None:
                image_features = self._encode_image(image_pixel_values, image_grid_thw)
                token_mask = input_ids == self._image_pad_id
                for i in range(B):
                    positions = token_mask[i].nonzero(as_tuple=True)[0]
                    n = min(len(positions), image_features[i].shape[0])
                    if n > 0:
                        sample_media[i].append((positions[:n], image_features[i][:n]))

            if video_pixel_values is not None and video_grid_thw is not None:
                vid_proj = self._encode_video(
                    video_pixel_values, video_grid_thw, video_num_frames
                )
                token_mask = input_ids == self._video_pad_id
                # vid_proj has one entry per video, not per batch row: rows
                # without video are skipped, so vidx (not i) indexes the list.
                vidx = 0
                for i in range(B):
                    positions = token_mask[i].nonzero(as_tuple=True)[0]
                    if len(positions) == 0:
                        continue
                    if isinstance(vid_proj, list):
                        feat_i = vid_proj[vidx]
                        n = min(len(positions), feat_i.shape[0])
                        if n > 0:
                            sample_media[i].append((positions[:n], feat_i[:n]))
                    else:
                        # pooled (num_videos, num_video_tokens, hidden)
                        n = min(len(positions), vid_proj.shape[1])
                        if n > 0:
                            sample_media[i].append((positions[:n], vid_proj[vidx, :n]))
                    vidx += 1

        # Build inputs_embeds per sample using one-hot matmul (differentiable)
        embeds_list = []
        for i in range(B):
            if not sample_media[i]:
                embeds_list.append(text_embeds[i])
                continue

            all_pos = torch.cat([p for p, _ in sample_media[i]])
            all_feat = torch.cat([f for _, f in sample_media[i]]).to(dtype)
            n_media = all_feat.shape[0]

            # One-hot matmul: guaranteed differentiable w.r.t. all_feat
            one_hot = torch.zeros(S, n_media, device=device, dtype=dtype)
            one_hot[all_pos, torch.arange(n_media, device=device)] = 1.0
            media_seq = one_hot @ all_feat  # (S, D)

            # Binary mask for position selection (constant, not in gradient graph)
            mask = one_hot.sum(dim=1, keepdim=True)  # (S, 1) — 1 at media pos

            embeds_list.append(text_embeds[i] * (1.0 - mask) + media_seq * mask)

        return torch.stack(embeds_list)

    # ── Forward ────────────────────────────────────────────────

    def forward(
        self,
        input_ids,
        attention_mask,
        input_features=None,
        dasheng_audio=None,
        image_pixel_values=None,
        image_grid_thw=None,
        video_pixel_values=None,
        video_grid_thw=None,
        video_num_frames=None,
        **kwargs,
    ):
        """Forward pass: encode multimodal input → embedding.

        Args:
            input_ids: (B, S) — tokenized text with media placeholders
            attention_mask: (B, S) — 1 for real tokens, 0 for padding
            input_features: (B, n_mels, T_mel) — Whisper mel (optional)
            dasheng_audio: (B, 1, T_samples) — raw audio (optional)
            image_pixel_values: (total_patches, patch_dim) — from Qwen3.5 processor (optional)
            image_grid_thw: (B, 3) — grid dims per image (optional)
            video_pixel_values: (total_patches, patch_dim) — from Qwen3.5 processor (optional)
            video_grid_thw: (N, 3) — grid dims per video, or one [1,h,w] row
                per frame when video_num_frames is given (optional)
            video_num_frames: list[int] — frames per video (video_as_images, optional)

        Returns:
            dict with:
                embedding: (B, hidden_size) — L2-normalized backbone embedding
                pooled: (B, hidden_size) — alias of embedding (kept for API compat)
        """
        if self.vision_mode == "native":
            # Compute audio features and stash for the embed-hook to splice in.
            audio_feats = None
            if input_features is not None and dasheng_audio is not None:
                audio_feats = self._encode_audio(input_features, dasheng_audio)
            self._pending.audio_feats = audio_feats
            self._pending.audio_input_ids = input_ids

            # mm_token_type_ids: 0=text, 1=image, 2=video — required for M-RoPE.
            mm_token_type_ids = torch.zeros_like(input_ids)
            mm_token_type_ids[input_ids == self._image_pad_id] = 1
            mm_token_type_ids[input_ids == self._video_pad_id] = 2

            bb_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                mm_token_type_ids=mm_token_type_ids,
                use_cache=False,
            )
            if image_pixel_values is not None:
                bb_kwargs["pixel_values"] = image_pixel_values
                bb_kwargs["image_grid_thw"] = image_grid_thw
            if video_pixel_values is not None:
                bb_kwargs["pixel_values_videos"] = video_pixel_values
                bb_kwargs["video_grid_thw"] = video_grid_thw

            try:
                # Reset rope_deltas so M-RoPE is recomputed for this batch.
                if hasattr(self.backbone, "rope_deltas"):
                    self.backbone.rope_deltas = None
                outputs = self.backbone(**bb_kwargs)
            finally:
                self._pending.audio_feats = None
                self._pending.audio_input_ids = None
        else:
            # custom / none: pre-build inputs_embeds externally (causal text
            # backbones like Qwen3-Embedding-0.6B; no MRoPE to worry about).
            inputs_embeds = self._build_inputs_embeds(
                input_ids, input_features, dasheng_audio,
                image_pixel_values, image_grid_thw,
                video_pixel_values, video_grid_thw,
                video_num_frames=video_num_frames,
            )
            outputs = self.backbone(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
            )
        last_hidden_state = outputs.last_hidden_state

        # EOS pooling (last actual token, not padding) — already L2-normalized.
        embedding = self.pooling(last_hidden_state, attention_mask)

        return {
            "embedding": embedding,
            "pooled": embedding,
        }

    @torch.no_grad()
    def encode(self, truncate_dim=None, **kwargs):
        """Inference-time encoding with optional MRL truncation.

        Args:
            truncate_dim: int — truncate embedding to this dim (Matryoshka slice)
            **kwargs: same as forward()

        Returns:
            (B, truncate_dim or hidden_size) — L2-normalized embedding
        """
        result = self.forward(**kwargs)
        emb = result["embedding"]
        if truncate_dim is not None and truncate_dim < emb.shape[-1]:
            emb = F.normalize(emb[:, :truncate_dim], p=2, dim=-1)
        return emb

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

        - no chat wrapping, no modality prefix
        - add_special_tokens=False
        - explicit <|endoftext|> appended to every row
        - left-padded (so pool position is always col -1)
        """
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
        # Qwen3-Embedding's native retrieval recipe terminates every row with
        # the literal <|endoftext|> token (id 151643), NOT the chat
        # <|im_end|> that tokenizer.eos_token_id resolves to (id 151645).
        # Using the wrong terminator shifts EOS-pooling to a different
        # embedding subspace and severely degrades retrieval. Fail loud if
        # <|endoftext|> isn't in vocab.
        eos = tok.convert_tokens_to_ids("<|endoftext|>")
        if eos is None or eos == tok.unk_token_id:
            raise RuntimeError(
                "tokenizer has no '<|endoftext|>' token — required for the "
                "Qwen3-Embedding native retrieval recipe."
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
        """Fast text-only encoding (no media processing).

        Identical to running the stock Qwen3-Embedding backbone: same causal
        forward, same EOS pooling, same L2-normalized hidden_size-dim output.

        Args:
            input_ids: (B, S)
            attention_mask: (B, S)

        Returns:
            (B, hidden_size) — L2-normalized text embedding
        """
        inputs_embeds = self.backbone.get_input_embeddings()(input_ids)
        outputs = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return self.pooling(outputs.last_hidden_state, attention_mask)

    def get_trainable_state_dict(self):
        """Return state dict with only trainable parameters."""
        return {
            name: param.data.cpu()
            for name, param in self.named_parameters()
            if param.requires_grad
        }
