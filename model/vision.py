"""Qwen3.5 vision encoder wrapper and video spatial pooling.

Qwen35VisionEncoder: Extracts Qwen3.5-0.8B's native vision tower (100.6M params).
  - 12 ViT blocks + spatial merger (2x2 merge, projects 768 -> 1024)
  - Accepts pre-processed patches from Qwen3.5's native image processor
  - Output: merged tokens (reduced_seq_len, 1024)

Uses Qwen3.5's native processor (Qwen2VLImageProcessor) for image/video
patch conversion. This handles:
  - smart_resize (aspect-ratio-preserving, factor-aligned)
  - Temporal frame grouping (temporal_patch_size=2)
  - Correct patch flattening to encoder input format

VideoSpatialPooler: Cross-attention to compress video tokens to a fixed count.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import BaseEncoder


class Qwen35VisionEncoder(BaseEncoder):
    """Standalone wrapper around Qwen3.5-0.8B's native vision tower.

    Extracts the visual module (100.6M params) from the full VLM and
    the image processor for correct patch conversion.

    The processor handles:
      - smart_resize: aspect-ratio-preserving resize to factor-aligned dims
      - Patch extraction: (B, T, C, H, W) -> (seq_len, patch_dim)
      - grid_thw computation

    Output (after merger): (merged_seq_len, 1024)
      Image (224x224):  49 tokens
      Image (448x256): ~112 tokens (dynamic, depends on aspect ratio)
      Video N frames:  variable tokens
    """

    def __init__(self, model_name="Qwen/Qwen3.5-0.8B"):
        super().__init__()
        self._model_name = model_name
        self.output_dim = 1024
        self.raw_dim = 768
        self.patch_size = 16
        self.temporal_patch_size = 2
        self.spatial_merge_size = 2
        self.visual = None
        self.image_processor = None
        # Toggled by OmniEmbedModel.activate_phase2 once LoRA gets injected
        # into self.visual's attention — recomputing the visual forward on
        # the backward pass trades extra compute for lower peak activation
        # memory. Pre-phase-2 the whole forward runs
        # under torch.no_grad() so checkpointing would be a pure no-op.
        self._grad_ckpt = False

    def load_pretrained_weights(self):
        """Load Qwen3.5-0.8B, extract vision tower and image processor."""
        if self.visual is not None:
            return
        from transformers import AutoModelForImageTextToText, AutoProcessor

        # Extract vision tower
        full_model = AutoModelForImageTextToText.from_pretrained(
            self._model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
        )
        self.visual = full_model.model.visual

        vc = full_model.config.vision_config
        self.patch_size = getattr(vc, 'patch_size', 16)
        self.temporal_patch_size = getattr(vc, 'temporal_patch_size', 2)
        self.spatial_merge_size = getattr(vc, 'spatial_merge_size',
                                          getattr(vc, 'merge_size', 2))
        self.output_dim = getattr(vc, 'out_hidden_size', 1024)
        self.raw_dim = getattr(vc, 'hidden_size', 768)
        del full_model
        torch.cuda.empty_cache()

        # Extract image processor
        processor = AutoProcessor.from_pretrained(
            self._model_name, trust_remote_code=True,
        )
        self.image_processor = processor.image_processor

    def _ensure_loaded(self):
        if self.visual is None:
            self.load_pretrained_weights()

    # ── Native processor wrappers ────────────────────────────────

    def process_images(self, pil_images):
        """Process PIL images through Qwen3.5's native processor.

        Args:
            pil_images: list of PIL.Image (can be different sizes)

        Returns:
            pixel_values: (total_patches, patch_dim) — ready for encoder
            image_grid_thw: (num_images, 3) — [t, h, w] per image
        """
        self._ensure_loaded()
        # Processor expects list of images, each as [frames...] for video
        # For single images, wrap each in a list of 1 frame
        inputs = self.image_processor(
            images=[[img] for img in pil_images],
            return_tensors="pt",
        )
        return inputs["pixel_values"], inputs["image_grid_thw"]

    def process_video_frames(self, frame_lists):
        """Process video frame lists through Qwen3.5's native processor.

        Args:
            frame_lists: list of [list of PIL.Image] — one frame list per video

        Returns:
            pixel_values: (total_patches, patch_dim)
            image_grid_thw: (num_videos, 3) — [t, h, w] per video
        """
        self._ensure_loaded()
        inputs = self.image_processor(
            images=frame_lists,
            return_tensors="pt",
        )
        return inputs["pixel_values"], inputs["image_grid_thw"]

    # ── Forward methods ──────────────────────────────────────────

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing on the visual tower (phase 2 only)."""
        self._grad_ckpt = True

    def forward(self, pixel_values, grid_thw):
        """Forward pass with pre-processed inputs.

        Args:
            pixel_values: (total_patches, patch_dim) — from native processor
            grid_thw: (num_items, 3) — [t, h, w] per image/video

        Returns:
            merged: (total_merged_tokens, output_dim)

        Processes items one at a time through self.visual rather than all
        concatenated. Qwen3-VL's attention cost scales with the total token
        count per call; chunking caps the per-call peak at ~1/B of a batched
        forward (for B items). Output is numerically identical because the
        vision transformer has no cross-item dependencies when grid_thw
        already partitions positions per item.
        """
        self._ensure_loaded()
        dtype = self.visual.merger.linear_fc2.weight.dtype
        px = pixel_values.to(dtype)

        # Patches per item, matching Qwen3-VL's patchify layout.
        patches_per_item = (
            grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
        ).tolist()

        use_ckpt = self._grad_ckpt and torch.is_grad_enabled()
        if use_ckpt:
            import torch.utils.checkpoint as _ckpt

        outputs = []
        offset = 0
        for i, n in enumerate(patches_per_item):
            item_px = px[offset:offset + n]
            item_grid = grid_thw[i:i + 1]
            if use_ckpt:
                item_out = _ckpt.checkpoint(
                    lambda p, g: self.visual(p, grid_thw=g).pooler_output,
                    item_px, item_grid, use_reentrant=False,
                )
            else:
                item_out = self.visual(item_px, grid_thw=item_grid).pooler_output
            outputs.append(item_out)
            offset += n

        return torch.cat(outputs, dim=0)

    def forward_and_split(self, pixel_values, grid_thw):
        """Forward pass, split output per image/video.

        Args:
            pixel_values: (total_patches, patch_dim)
            grid_thw: (N, 3)

        Returns:
            list of (num_tokens_i, output_dim) tensors — one per item
        """
        merged = self.forward(pixel_values, grid_thw)

        # Compute merged tokens per item
        sms = self.spatial_merge_size
        tokens_per_item = (
            grid_thw[:, 0] *
            (grid_thw[:, 1] // sms) *
            (grid_thw[:, 2] // sms)
        )
        return merged.split(tokens_per_item.tolist(), dim=0)


class VideoSpatialPooler(nn.Module):
    """Cross-attention pooling to compress variable-length video tokens.

    After the Qwen3.5 vision encoder + merger, a video may have variable
    token counts depending on resolution and frame count. This module
    compresses them to a fixed count using learnable queries.

    Architecture (2-layer cross-attention with residual):
        query x K/V (all video tokens)
        -> Layer 1: cross-attn + residual + LayerNorm
        -> Layer 2: cross-attn + residual + LayerNorm
        -> (B, num_output_tokens, dim)
    """

    def __init__(self, input_dim=1024, num_output_tokens=196, num_heads=16):
        super().__init__()
        self.num_output_tokens = num_output_tokens
        self.input_dim = input_dim

        self.query = nn.Parameter(
            torch.randn(1, num_output_tokens, input_dim) * 0.02
        )
        self.attn1 = nn.MultiheadAttention(input_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(input_dim)
        self.attn2 = nn.MultiheadAttention(input_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(input_dim)

    def forward(self, video_tokens, attention_mask=None):
        """
        Args:
            video_tokens: (B, T, D) — variable-length merged tokens (padded)
            attention_mask: (B, T) — 1 for real tokens, 0 for padding (optional)

        Returns:
            (B, num_output_tokens, D) — fixed-size compressed representation
        """
        B, T, D = video_tokens.shape
        q = self.query.expand(B, -1, -1)

        # Convert attention_mask to key_padding_mask for MHA (True = ignore)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)

        # Layer 1
        out, _ = self.attn1(q, video_tokens, video_tokens,
                            key_padding_mask=key_padding_mask)
        out = self.norm1(q + out)
        # Layer 2
        out2, _ = self.attn2(out, video_tokens, video_tokens,
                             key_padding_mask=key_padding_mask)
        out = self.norm2(out + out2)

        return out


class DimProjector(nn.Module):
    """Simple linear projection for dimension matching.

    Always uses a trainable Linear layer (even when dims match) so that
    gradients flow through the projector during alignment training.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.proj(x)
