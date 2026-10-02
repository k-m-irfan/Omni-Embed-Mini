"""Contrastive collator for Omni-Embed training.

Handles two pair types:
  - Caption pairs: anchor=media, positive=caption_text
  - Chat pairs: anchor=media+user_question, positive=assistant_answer

Uses the vision processor (Qwen3.5 in custom mode, Qwen3-VL in native mode)
for image/video patch conversion. Image token counts are dynamic (vary by
resolution/aspect ratio). Video tokens are either pooled to a fixed count
(VideoSpatialPooler) or, with video_as_images, emitted per frame.
"""

import torch
import torch.nn.functional as F
from PIL import Image


# Prompt templates
EMBED_MEDIA_TEMPLATE = "<|im_start|>user\n{media_placeholder}\n<|im_end|>"
EMBED_MEDIA_WITH_TEXT_TEMPLATE = "<|im_start|>user\n{media_placeholder}\n{text}\n<|im_end|>"
EMBED_TEXT_TEMPLATE = "<|im_start|>user\n{text}\n<|im_end|>"

WHISPER_MAX_MEL = 3000


class ContrastiveCollator:
    """Collate contrastive pairs for embedding training.

    Args:
        tokenizer: HuggingFace tokenizer
        vision_processor: vision image processor (for image/video patch conversion)
        max_seq_length: int — max sequence length
        tokens_per_encoder: int — audio tokens per encoder per chunk
        num_video_tokens: int — fixed video token count (after pooler)
        neg_cache: optional NegativeCache — for hard negative lookup
        dataset: optional dataset reference — for loading hard negative captions
    """

    def __init__(
        self,
        tokenizer,
        vision_processor=None,
        max_seq_length=2048,
        tokens_per_encoder=128,
        num_video_tokens=196,
        neg_cache=None,
        dataset=None,
        vision_mode="custom",
        video_as_images=False,
    ):
        self.tokenizer = tokenizer
        self.vision_processor = vision_processor
        self.max_seq_length = max_seq_length
        self.tokens_per_encoder = tokens_per_encoder
        self.num_video_tokens = num_video_tokens
        self.neg_cache = neg_cache
        self.dataset = dataset
        self.vision_mode = vision_mode
        # video_as_images: route video frames through the per-frame image
        # projector (no VideoSpatialPooler) — emit variable <|video_pad|> counts
        # (= total frame tokens per video) + `video_num_frames`. Custom mode
        # only; native mode forwards video to the backbone.
        self.video_as_images = video_as_images

        # Placeholder tokens switch based on vision_mode. In "native" we use
        # the Qwen-VL convention: <|vision_start|> ... <|vision_end|> with
        # <|image_pad|> / <|video_pad|> padding, all already in the backbone
        # tokenizer. In "custom" we use our own <|image_start|> / <|image_end|>
        # plus matching pads.
        if vision_mode == "native":
            self.img_start, self.img_end, self.img_pad = (
                "<|vision_start|>", "<|vision_end|>", "<|image_pad|>",
            )
            self.vid_start, self.vid_end, self.vid_pad = (
                "<|vision_start|>", "<|vision_end|>", "<|video_pad|>",
            )
        else:  # custom or none
            self.img_start, self.img_end, self.img_pad = (
                "<|image_start|>", "<|image_end|>", "<|image_pad|>",
            )
            self.vid_start, self.vid_end, self.vid_pad = (
                "<|video_start|>", "<|video_end|>", "<|video_pad|>",
            )

    def __call__(self, batch):
        modality = batch[0]["modality"]

        if modality == "omni":
            anchor = self._build_omni_anchor(batch)
        elif modality in ("speech", "audio"):
            anchor = self._build_audio_anchor(batch)
        elif modality in ("image", "visual_doc"):
            anchor = self._build_image_anchor(batch)
        elif modality == "video":
            anchor = self._build_video_anchor(batch)
        else:
            anchor = self._build_text_only_anchor(batch)

        positive = self._build_text_positive(batch)

        result = {}
        for k, v in anchor.items():
            result[f"anchor_{k}"] = v
        for k, v in positive.items():
            result[f"positive_{k}"] = v

        result["modality"] = modality
        result["pair_type"] = batch[0].get("pair_type", "caption")
        result["sample_ids"] = torch.tensor([s["sample_id"] for s in batch])

        if self.neg_cache is not None and self.dataset is not None:
            hn = self._build_hard_negatives(batch)
            if hn is not None:
                for k, v in hn.items():
                    result[f"hardneg_{k}"] = v

        return result

    # ── helpers ───────────────────────────────────────────────────

    def _media_prompt(self, placeholder, anchor_text=""):
        if anchor_text:
            return EMBED_MEDIA_WITH_TEXT_TEMPLATE.format(
                media_placeholder=placeholder, text=anchor_text,
            )
        return EMBED_MEDIA_TEMPLATE.format(media_placeholder=placeholder)

    def _compute_image_token_count(self, pil_images):
        """Compute per-image merged token counts using the vision processor.

        Runs the processor to get grid_thw, then calculates merged tokens:
            tokens = t * (h // merge_size) * (w // merge_size)

        Returns list of int token counts and the processed tensors.
        """
        if self.vision_processor is None:
            # Fallback: fixed 49 tokens per image
            return [49] * len(pil_images), None, None

        inputs = self.vision_processor(
            images=[[img] for img in pil_images],
            return_tensors="pt",
        )
        pixel_values = inputs["pixel_values"]
        grid_thw = inputs["image_grid_thw"]

        merge_size = getattr(self.vision_processor, 'merge_size', 2)
        token_counts = []
        for i in range(grid_thw.shape[0]):
            t, h, w = grid_thw[i].tolist()
            tokens = t * (h // merge_size) * (w // merge_size)
            token_counts.append(tokens)

        return token_counts, pixel_values, grid_thw

    def _compute_video_token_count(self, frame_lists):
        """Process video frames through the vision processor.

        Returns fixed num_video_tokens (pooler compresses variable → fixed).
        Also returns processed tensors.
        """
        if self.vision_processor is None:
            return self.num_video_tokens, None, None

        inputs = self.vision_processor(
            images=frame_lists,
            return_tensors="pt",
        )
        return self.num_video_tokens, inputs["pixel_values"], inputs["image_grid_thw"]

    def _video_imagelike(self, frame_lists):
        """Process video frames as images and return per-video pad
        counts (= sum of frame token counts) + frames-per-video.

        Returns (pixel_values, grid_thw, pad_counts:list[int], num_frames:list[int]).
        Each grid row is one frame ([1,h,w]); tokens/frame = (h//m)*(w//m).
        """
        inputs = self.vision_processor(images=frame_lists, return_tensors="pt")
        grid = inputs["image_grid_thw"]
        m = getattr(self.vision_processor, "merge_size", 2)
        per_row = (grid[:, 0] * (grid[:, 1] // m) * (grid[:, 2] // m)).tolist()
        num_frames = [len(fl) for fl in frame_lists]
        pad_counts, off = [], 0
        for f in num_frames:
            pad_counts.append(int(sum(per_row[off:off + f])))
            off += f
        return inputs["pixel_values"], grid, pad_counts, num_frames

    # ── Anchor builders ──────────────────────────────────────────

    def _build_audio_anchor(self, batch):
        B = len(batch)

        mel_lens = []
        for s in batch:
            mel_T = s["mel_features"].shape[-1]
            n_chunks = max(1, (mel_T + WHISPER_MAX_MEL - 1) // WHISPER_MAX_MEL)
            mel_lens.append(n_chunks * self.tokens_per_encoder * 2)

        texts = []
        for i, s in enumerate(batch):
            placeholder = "<|audio_start|>" + "<|audio_pad|>" * mel_lens[i] + "<|audio_end|>"
            texts.append(self._media_prompt(placeholder, s.get("anchor_text", "")))

        encoded = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_seq_length, return_tensors="pt",
        )

        max_mel_len = max(s["mel_features"].shape[-1] for s in batch)
        mel_features = torch.zeros(B, batch[0]["mel_features"].shape[0], max_mel_len)
        for i, s in enumerate(batch):
            t = s["mel_features"].shape[-1]
            mel_features[i, :, :t] = s["mel_features"]

        max_audio_len = max(s["dasheng_audio"].shape[-1] for s in batch)
        dasheng_audio = torch.zeros(B, 1, max_audio_len)
        for i, s in enumerate(batch):
            t = s["dasheng_audio"].shape[-1]
            dasheng_audio[i, :, :t] = s["dasheng_audio"]

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "input_features": mel_features,
            "dasheng_audio": dasheng_audio,
        }

    def _build_image_anchor(self, batch):
        """Build anchor for image/visual_doc using the vision processor.

        Each image may produce a different number of tokens depending on its
        resolution. The prompt uses that many <|image_pad|> placeholders.
        """
        # Collect PIL images
        pil_images = []
        for s in batch:
            pv = s["pixel_values"]  # (3, H, W) tensor from dataset
            # Convert back to PIL for processor (processor expects PIL)
            img = _tensor_to_pil(pv)
            pil_images.append(img)

        token_counts, pixel_values, grid_thw = self._compute_image_token_count(pil_images)

        # Build prompts with per-image token counts
        texts = []
        for i, s in enumerate(batch):
            n = token_counts[i]
            placeholder = self.img_start + self.img_pad * n + self.img_end
            texts.append(self._media_prompt(placeholder, s.get("anchor_text", "")))

        encoded = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_seq_length, return_tensors="pt",
        )

        result = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
        if pixel_values is not None:
            result["image_pixel_values"] = pixel_values
            result["image_grid_thw"] = grid_thw

        return result

    def _build_video_anchor(self, batch):
        """Build anchor for video using the vision processor.

        Frames from smart_sample_frames are converted to PIL and processed
        through Qwen3.5's native processor. The pooler compresses to
        num_video_tokens regardless of input token count.
        """
        frame_lists = []
        for s in batch:
            frames_tensor = s["frames"]  # (F, 3, H, W)
            pil_frames = [_tensor_to_pil(frames_tensor[j]) for j in range(frames_tensor.shape[0])]
            frame_lists.append(pil_frames)

        num_frames = None
        if self.video_as_images and self.vision_processor is not None:
            # video_as_images: variable per-video pad count = total frame tokens.
            pixel_values, grid_thw, pad_counts, num_frames = \
                self._video_imagelike(frame_lists)
            texts = [
                self._media_prompt(
                    self.vid_start + self.vid_pad * pad_counts[i] + self.vid_end,
                    s.get("anchor_text", ""),
                )
                for i, s in enumerate(batch)
            ]
        else:
            n_tokens = self.num_video_tokens  # fixed after pooler
            _, pixel_values, grid_thw = self._compute_video_token_count(frame_lists)
            texts = [
                self._media_prompt(
                    self.vid_start + self.vid_pad * n_tokens + self.vid_end,
                    s.get("anchor_text", ""),
                )
                for s in batch
            ]

        encoded = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_seq_length, return_tensors="pt",
        )

        result = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
        if pixel_values is not None:
            result["video_pixel_values"] = pixel_values
            result["video_grid_thw"] = grid_thw
        if num_frames is not None:
            result["video_num_frames"] = torch.tensor(num_frames, dtype=torch.long)

        return result

    def _build_omni_anchor(self, batch):
        """Build anchor for omni samples with mixed media.

        Each sample may have any combination of audio, image, and video.
        The prompt includes placeholder sections for each present media type,
        ordered as: [audio] [image] [video] [question_text]

        Samples without a certain media get zero-padded tensors so the batch
        can be stacked. The model's token replacement will match placeholder
        count to feature count per sample.
        """
        B = len(batch)

        # ── Determine per-sample media presence ──
        has_audio = [s.get("has_audio", False) for s in batch]
        has_image = [s.get("has_image", False) for s in batch]
        has_video = [s.get("has_video", False) for s in batch]
        any_audio = any(has_audio)
        any_image = any(has_image)
        any_video = any(has_video)

        # ── Build per-sample prompts with appropriate placeholders ──
        # Audio token counts (variable per sample)
        audio_token_counts = []
        for s in batch:
            if s.get("has_audio") and "mel_features" in s:
                mel_T = s["mel_features"].shape[-1]
                n_chunks = max(1, (mel_T + WHISPER_MAX_MEL - 1) // WHISPER_MAX_MEL)
                audio_token_counts.append(n_chunks * self.tokens_per_encoder * 2)
            else:
                audio_token_counts.append(0)

        # Image token counts (from processor, or 0 if no image)
        image_token_counts = []
        pil_images = []
        for s in batch:
            if s.get("has_image") and "pixel_values" in s:
                pil_images.append(_tensor_to_pil(s["pixel_values"]))
            else:
                pil_images.append(None)

        if any_image and self.vision_processor is not None:
            valid_imgs = [img for img in pil_images if img is not None]
            if valid_imgs:
                proc_out = self.vision_processor(
                    images=[[img] for img in valid_imgs],
                    return_tensors="pt",
                )
                merge_size = getattr(self.vision_processor, 'merge_size', 2)
                valid_idx = 0
                for i in range(B):
                    if pil_images[i] is not None:
                        t, h, w = proc_out["image_grid_thw"][valid_idx].tolist()
                        image_token_counts.append(t * (h // merge_size) * (w // merge_size))
                        valid_idx += 1
                    else:
                        image_token_counts.append(0)
            else:
                image_token_counts = [0] * B
        else:
            image_token_counts = [0] * B

        # Video token counts: fixed num_video_tokens (pooler) by default. With
        # video_as_images: per-video pad count = total frame tokens; compute
        # up front so the prompt matches, and stash tensors for reuse below.
        video_pad_per_sample = [self.num_video_tokens] * B  # pooler default
        _vid_cache = None
        if self.video_as_images and any_video and self.vision_processor is not None:
            _frame_lists, _vid_idx = [], []
            for i, s in enumerate(batch):
                if has_video[i] and "frames" in s:
                    ft = s["frames"]
                    _frame_lists.append([_tensor_to_pil(ft[j]) for j in range(ft.shape[0])])
                    _vid_idx.append(i)
            if _frame_lists:
                pv, grid, pad_counts, num_frames = self._video_imagelike(_frame_lists)
                for k, i in enumerate(_vid_idx):
                    video_pad_per_sample[i] = pad_counts[k]
                _vid_cache = (pv, grid, num_frames)

        # Build text prompts
        texts = []
        for i, s in enumerate(batch):
            parts = []
            # Audio section
            if audio_token_counts[i] > 0:
                parts.append(
                    "<|audio_start|>" + "<|audio_pad|>" * audio_token_counts[i] + "<|audio_end|>"
                )
            # Image section
            if image_token_counts[i] > 0:
                parts.append(
                    self.img_start + self.img_pad * image_token_counts[i] + self.img_end
                )
            # Video section
            if has_video[i]:
                parts.append(
                    self.vid_start + self.vid_pad * video_pad_per_sample[i] + self.vid_end
                )

            media_str = "\n".join(parts)
            anchor_text = s.get("anchor_text", "")
            if anchor_text:
                texts.append(f"<|im_start|>user\n{media_str}\n{anchor_text}\n<|im_end|>")
            else:
                texts.append(f"<|im_start|>user\n{media_str}\n<|im_end|>")

        encoded = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_seq_length, return_tensors="pt",
        )

        result = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

        # ── Audio features (zero-padded for samples without audio) ──
        if any_audio:
            max_mel_len = max(
                (s["mel_features"].shape[-1] if s.get("has_audio") and "mel_features" in s else 0)
                for s in batch
            )
            if max_mel_len > 0:
                # Infer mel_dim from the first audio sample (the mel bin count
                # depends on the Whisper variant) rather than hardcoding it.
                mel_dim = next(
                    s["mel_features"].shape[0]
                    for s in batch
                    if s.get("has_audio") and "mel_features" in s
                )
                mel_features = torch.zeros(B, mel_dim, max_mel_len)
                max_audio_len = max(
                    (s["dasheng_audio"].shape[-1] if s.get("has_audio") and "dasheng_audio" in s else 0)
                    for s in batch
                )
                dasheng_audio = torch.zeros(B, 1, max(max_audio_len, 1))
                for i, s in enumerate(batch):
                    if s.get("has_audio") and "mel_features" in s:
                        t = s["mel_features"].shape[-1]
                        mel_features[i, :, :t] = s["mel_features"]
                        t2 = s["dasheng_audio"].shape[-1]
                        dasheng_audio[i, :, :t2] = s["dasheng_audio"]
                result["input_features"] = mel_features
                result["dasheng_audio"] = dasheng_audio

        # ── Image features ──
        if any_image and self.vision_processor is not None:
            valid_imgs = [img for img in pil_images if img is not None]
            if valid_imgs:
                proc_out = self.vision_processor(
                    images=[[img] for img in valid_imgs],
                    return_tensors="pt",
                )
                result["image_pixel_values"] = proc_out["pixel_values"]
                result["image_grid_thw"] = proc_out["image_grid_thw"]

        # ── Video features ──
        if _vid_cache is not None:
            # video_as_images: reuse the up-front processing; emit frames-per-video.
            pv, grid, num_frames = _vid_cache
            result["video_pixel_values"] = pv
            result["video_grid_thw"] = grid
            result["video_num_frames"] = torch.tensor(num_frames, dtype=torch.long)
        elif any_video:
            frame_lists = []
            for i, s in enumerate(batch):
                if has_video[i] and "frames" in s:
                    frames_tensor = s["frames"]
                    pil_frames = [_tensor_to_pil(frames_tensor[j])
                                  for j in range(frames_tensor.shape[0])]
                    frame_lists.append(pil_frames)

            if frame_lists and self.vision_processor is not None:
                proc_out = self.vision_processor(
                    images=frame_lists,
                    return_tensors="pt",
                )
                result["video_pixel_values"] = proc_out["pixel_values"]
                result["video_grid_thw"] = proc_out["image_grid_thw"]

        return result

    def _build_text_only_anchor(self, batch):
        texts = []
        for s in batch:
            t = s.get("anchor_text") or s.get("positive_text", "")
            texts.append(EMBED_TEXT_TEMPLATE.format(text=t))

        encoded = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_seq_length, return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    # ── Teacher cache text (vanilla Qwen3 format, no wrap/prefix) ──

    @staticmethod
    def teacher_raw_text(sample):
        """Exact caption string the teacher will embed.

        No chat wrapper, no "{label}:" modality prefix. The teacher cache
        tokenizes this via `model.tokenize_text_native(...)` so its vector
        is byte-identical to the frozen backbone's native text embedding of
        the caption — the subspace the student must learn to hit for
        cross-modal retrieval against plain text embeddings to work.
        """
        return sample["positive_text"]

    # ── Positive builder (cascaded text target) ────────────────────

    _MODALITY_LABELS = {
        "speech": "Audio context",
        "audio": "Audio context",
        "image": "Image context",
        "visual_doc": "Document context",
        "video": "Video context",
    }

    def _build_text_positive(self, batch):
        """Build cascaded text targets for distillation.

        Replaces actual media with text descriptions (captions/transcripts),
        mimicking a cascaded system. For chat pairs, appends the question
        so the text target captures both media content and query intent.
        """
        texts = []
        for s in batch:
            positive_text = s["positive_text"]
            pair_type = s.get("pair_type", "caption")
            modality = s.get("modality", "")
            anchor_text = s.get("anchor_text", "")

            # Build modality-prefixed description
            if modality == "omni":
                label = self._omni_label(s)
            else:
                label = self._MODALITY_LABELS.get(modality, "Content context")

            description = f"{label}: {positive_text}"

            # For chat pairs, append the question/query
            if pair_type == "chat" and anchor_text:
                enriched = f"{description}\n{anchor_text}"
            else:
                enriched = description

            texts.append(EMBED_TEXT_TEMPLATE.format(text=enriched))

        encoded = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_seq_length, return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    @staticmethod
    def _omni_label(sample):
        """Build modality label for omni samples from per-sample flags."""
        parts = []
        if sample.get("has_image"):
            parts.append("Image")
        if sample.get("has_video"):
            parts.append("Video")
        if sample.get("has_audio"):
            parts.append("audio")
        if not parts:
            return "Content context"
        return " and ".join(parts) + " context"

    # ── Hard negative builder ────────────────────────────────────

    def _build_hard_negatives(self, batch):
        B = len(batch)
        max_negs = 5
        all_neg_texts = []
        has_any = False

        for s in batch:
            sid = s["sample_id"]
            neg_ids = self.neg_cache.get(sid, n=max_negs)
            if neg_ids:
                texts = []
                for nid in neg_ids:
                    if nid < len(self.dataset.captions):
                        texts.append(self.dataset.captions[nid])
                    else:
                        texts.append("")
                while len(texts) < max_negs:
                    texts.append(texts[-1] if texts else "")
                all_neg_texts.append(texts[:max_negs])
                has_any = True
            else:
                all_neg_texts.append([""] * max_negs)

        if not has_any:
            return None

        flat_texts = []
        for neg_list in all_neg_texts:
            for t in neg_list:
                flat_texts.append(EMBED_TEXT_TEMPLATE.format(text=t))

        encoded = self.tokenizer(
            flat_texts, padding=True, truncation=True,
            max_length=self.max_seq_length, return_tensors="pt",
        )

        S = encoded["input_ids"].shape[1]
        return {
            "input_ids": encoded["input_ids"].reshape(B, max_negs, S),
            "attention_mask": encoded["attention_mask"].reshape(B, max_negs, S),
        }


def _tensor_to_pil(tensor):
    """Convert a normalized (3, H, W) tensor back to PIL Image.

    Assumes normalize(mean=0.5, std=0.5) was applied: pixel = x * 0.5 + 0.5
    """
    x = tensor.float().clamp(-1, 1) * 0.5 + 0.5  # [0, 1]
    x = (x * 255).byte()
    return Image.fromarray(x.permute(1, 2, 0).numpy())
