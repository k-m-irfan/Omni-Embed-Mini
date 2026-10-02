"""Contrastive dataset wrapping MBZUAI/Omni-Sets for embedding training.

Efficient design:
  - Stores 1 descriptor per split (not per sample) — init is O(num_splits)
  - __getitem__ resolves split via bisect on cumulative offsets — O(log n)
  - Captions extracted lazily via HF .map() on media-dropped dataset — vectorized
  - Prints gated by LOCAL_RANK to avoid 8x duplicate output in DDP
  - Omni split handled properly: mixed media (audio+video+image) in single samples
"""

import os
import io as _io
import bisect
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, Audio


# ── Audio preprocessing ──────────────────────────────────────────

def _preprocess_audio(audio_array, whisper_fe, max_audio_duration=300.0):
    """Extract Whisper mel features and raw audio for Dasheng."""
    max_samples = int(max_audio_duration * 16000)
    if len(audio_array) > max_samples:
        audio_array = audio_array[:max_samples]

    n_fft = 400
    if len(audio_array) < n_fft:
        audio_array = np.pad(audio_array, (0, n_fft - len(audio_array)))

    hop = 160
    mel_len = (len(audio_array) + hop - 1) // hop
    mel = whisper_fe(
        audio_array, sampling_rate=16000, return_tensors="np"
    )["input_features"][0]

    dasheng = np.zeros((1, len(audio_array)), dtype=np.float32)
    dasheng[0, :len(audio_array)] = audio_array

    return {
        "mel_features": torch.from_numpy(mel),
        "mel_len": mel_len,
        "dasheng_audio": torch.from_numpy(dasheng),
    }


def _get_audio_array(audio_item, target_sr=16000):
    """Extract audio array from HuggingFace audio feature.

    Raises loudly on decode failure; do not add silent fallbacks.
    """
    if hasattr(audio_item, "get_all_samples"):
        samples = audio_item.get_all_samples()
        arr = samples.data.numpy().astype(np.float32).flatten()
        sr = samples.sample_rate
    elif isinstance(audio_item, dict):
        if "array" in audio_item:
            arr = np.array(audio_item["array"], dtype=np.float32)
            sr = audio_item.get("sampling_rate", target_sr)
        elif audio_item.get("bytes") or audio_item.get("path"):
            # Raw bytes/path from Audio(decode=False) — decode via soundfile,
            # already resampled to target_sr inside _decode_audio_bytes.
            arr = _decode_audio_bytes(audio_item, target_sr=target_sr)
            sr = target_sr
        else:
            raise ValueError(
                f"audio_item dict has neither 'array' nor 'bytes'/'path' "
                f"(keys={list(audio_item.keys())})"
            )
    else:
        raise TypeError(f"unexpected audio_item type: {type(audio_item).__name__}")

    if sr != target_sr:
        import librosa
        arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr)

    if arr.ndim > 1:
        arr = arr.mean(axis=0)

    if len(arr) == 0:
        raise ValueError("decoded audio array is empty")
    return arr


def _decode_audio_bytes(audio_data, target_sr=16000):
    """Decode audio from raw bytes or file path using soundfile.

    Raises on failure — do not swallow decode errors silently.
    """
    import soundfile as sf
    path = audio_data.get("path") if isinstance(audio_data, dict) else None
    raw_bytes = audio_data.get("bytes") if isinstance(audio_data, dict) else None

    if raw_bytes is not None:
        array, sr = sf.read(_io.BytesIO(raw_bytes), dtype="float32")
    elif path is not None:
        array, sr = sf.read(path, dtype="float32")
    else:
        raise ValueError("audio_data has neither 'bytes' nor 'path'")

    if array.ndim > 1:
        array = array.mean(axis=1)
    array = array.astype(np.float32)
    if sr != target_sr:
        import librosa
        array = librosa.resample(array, orig_sr=sr, target_sr=target_sr)
    if len(array) == 0:
        raise ValueError("decoded audio is empty")
    return array


def _extract_audio_from_video(video_src, target_sr=16000):
    """Extract audio track from video bytes or path using pyav.

    Returns None only when the video has no audio stream (legitimate —
    some videos are silent). All decode errors propagate.
    """
    import av

    if isinstance(video_src, bytes) and video_src:
        container = av.open(_io.BytesIO(video_src))
    elif isinstance(video_src, str) and os.path.exists(video_src):
        container = av.open(video_src)
    else:
        raise ValueError(
            f"video_src must be non-empty bytes or existing path, got "
            f"{type(video_src).__name__}"
        )

    if not container.streams.audio:
        container.close()
        return None

    audio_stream = container.streams.audio[0]
    audio_stream.codec_context.skip_frame = "DEFAULT"

    frames = []
    for frame in container.decode(audio=0):
        arr = frame.to_ndarray()
        if arr.ndim > 1:
            arr = arr.mean(axis=0)
        frames.append(arr.astype(np.float32))
    container.close()

    if not frames:
        raise ValueError("video had an audio stream but no frames decoded")

    audio = np.concatenate(frames)
    sr = audio_stream.rate

    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    if len(audio) == 0:
        raise ValueError("extracted audio is empty")
    return audio


# ── Message parsing ──────────────────────────────────────────────

_MEDIA_TAGS = ["<audio>", "<video>", "<image>", "</audio>", "</video>", "</image>"]
_TRANSCRIPTION_RE = re.compile(r"<transcription>(.*?)</transcription>", re.DOTALL)


def _strip_media_tags(text):
    for tag in _MEDIA_TAGS:
        text = text.replace(tag, "")
    return text.strip()


def _extract_caption(messages, split_name=None):
    """Extract the caption text used for the teacher prompt.

    For the `speech` split, Omni-Sets wraps the actual spoken content in
    <transcription>...</transcription> tags, surrounded by voice-metadata
    prose (pitch, accent, recording quality, etc.). We want the teacher
    to embed the semantic content (what was said), not the acoustic
    description — so we return the inside of the transcription tag when
    present. For all other splits, return the full assistant content.
    """
    parts = []
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if split_name == "speech":
                matches = _TRANSCRIPTION_RE.findall(content)
                if matches:
                    content = " ".join(m.strip() for m in matches)
            content = _strip_media_tags(content)
            if content:
                parts.append(content)
    return " ".join(parts) if parts else ""


def _extract_qa_pair(messages):
    question = ""
    answer = ""
    for msg in messages:
        role = msg.get("role", "")
        content = _strip_media_tags(msg.get("content", ""))
        if role == "user" and content and not question:
            question = content
        elif role == "assistant" and content and not answer:
            answer = content
    return question, answer


def _is_main_process():
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    return rank == 0


_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}


# ── Caption extraction for HF .map() ────────────────────────────

def _extract_caption_batch(batch, split_name=None, use_original_caption=False):
    captions = []
    originals = batch.get("original_text") if use_original_caption else None
    categories = batch.get("category", ["caption"] * len(batch["messages"]))
    for i, (msgs, cat) in enumerate(zip(batch["messages"], categories)):
        if cat == "chat":
            _, answer = _extract_qa_pair(msgs)
            captions.append(answer)
        elif originals is not None:
            captions.append((originals[i] or "").strip())
        else:
            captions.append(_extract_caption(msgs, split_name=split_name))
    return {"_caption": captions}


# ── Dataset ──────────────────────────────────────────────────────

class ContrastiveOmniDataset(Dataset):
    """Efficient contrastive dataset over MBZUAI/Omni-Sets.

    Handles all splits including omni (interleaved multi-modal samples).
    Omni samples can contain any combination of audio + image + video.

    Stores one descriptor per split. __getitem__ resolves via bisect.
    """

    def __init__(
        self,
        split_names=None,
        whisper_fe=None,
        image_size=224,
        max_video_frames=196,
        max_audio_duration=300.0,
        data_pct=1.0,
        use_original_caption=False,
        max_duration_s=None,
    ):
        super().__init__()
        self.use_original_caption = use_original_caption
        if split_names is None:
            raise ValueError("split_names must be provided (from config)")
        self.whisper_fe = whisper_fe
        self.image_size = image_size
        self.max_video_frames = max_video_frames
        self.max_audio_duration = max_audio_duration
        # Per-split duration caps (seconds) from the data.max_duration_s config.
        # Applied via the `duration_s` column before stride-sampling so the
        # filter survives data_pct and the teacher-cache's stable key map.
        self.max_duration_s = dict(max_duration_s or {})
        self._is_main = _is_main_process()

        from torchvision import transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.splits = []
        self.cumulative = [0]
        self._text_datasets = {}
        self._captions = None

        for split_name in split_names:
            ds = load_dataset(
                "MBZUAI/Omni-Sets", split_name, split="train",
                trust_remote_code=True,
            )
            if split_name in ("speech", "audio") and "media" in ds.column_names:
                # decode=False → datasets returns raw {bytes, path} dict.
                # datasets>=4.0 swapped the default audio decoder to
                # torchcodec, which is not ROCm-compatible; we decode
                # manually via soundfile in _get_audio_array.
                ds = ds.cast_column(
                    "media", Audio(sampling_rate=16000, decode=False)
                )
            elif split_name == "video" and "media" in ds.column_names:
                # datasets>=4.0 auto-decodes Video via torchcodec (not ROCm-
                # compatible). decode=False → raw {bytes, path} dict, and
                # load_video_frames decodes via pyav.
                from datasets import Video
                ds = ds.cast_column("media", Video(decode=False))
            elif split_name == "omni" and "media" in ds.column_names:
                # Omni media is mixed (video/image/audio) — don't auto-decode
                from datasets import Video
                ds = ds.cast_column("media", Video(decode=False))

            # ── Caption-only filter ──
            # Keep only samples whose assistant response is a dense
            # description of the media (category == "caption"). Chat/QA
            # pairs would give the teacher a narrow answer instead of a
            # full description, breaking the cascaded-teacher framing.
            # Omni has topic labels instead of caption/chat, so the filter
            # drops it.
            if "category" in ds.column_names:
                n_before = len(ds)
                keep_idx = [i for i, c in enumerate(ds["category"])
                            if c == "caption"]
                ds = ds.select(keep_idx)
                if self._is_main:
                    print(f"  [{split_name}] caption filter: "
                          f"{len(ds)}/{n_before} kept ({100*len(ds)/max(n_before,1):.1f}%)")
                source_indices = keep_idx
            else:
                source_indices = list(range(len(ds)))

            # ── Duration outlier filter (split-specific cap in seconds) ──
            # A handful of very long videos inflate Qwen3-VL's
            # patch-token tensor and cause OOM even at small batch sizes.
            # We drop anything over the configured threshold using the
            # `duration_s` column — no media decode needed.
            split_cap = self.max_duration_s.get(split_name)
            if split_cap is not None:
                if "duration_s" not in ds.column_names:
                    raise RuntimeError(
                        f"[{split_name}] max_duration_s={split_cap} configured "
                        "but dataset has no 'duration_s' column; remove the "
                        "cap or check the split's schema"
                    )
                n_before = len(ds)
                durations = ds["duration_s"]
                dur_keep_idx = [i for i, d in enumerate(durations)
                                if d is not None and d <= split_cap]
                ds = ds.select(dur_keep_idx)
                source_indices = [source_indices[i] for i in dur_keep_idx]
                if self._is_main:
                    print(f"  [{split_name}] duration<={split_cap}s filter: "
                          f"{len(ds)}/{n_before} kept "
                          f"({100*len(ds)/max(n_before,1):.1f}%)")

            if len(ds) == 0:
                if self._is_main:
                    print(f"  [{split_name}] no caption samples — skipping split")
                continue

            n_full = len(ds)
            if data_pct < 1.0:
                stride = max(1, int(round(1.0 / data_pct)))
                local_stride_idx = list(range(0, n_full, stride))
                ds = ds.select(local_stride_idx)
                original_indices = [source_indices[i] for i in local_stride_idx]
            else:
                original_indices = list(source_indices)

            heavy_cols = [c for c in ds.column_names if c in ("media", "question_audio")]
            self._text_datasets[split_name] = (
                ds.remove_columns(heavy_cols) if heavy_cols else ds
            )

            count = len(ds)
            self.splits.append((split_name, ds, count, original_indices))
            self.cumulative.append(self.cumulative[-1] + count)

            if self._is_main:
                print(f"  Loaded {split_name}: {count} samples")

        self.total_samples = self.cumulative[-1]
        if self._is_main:
            print(f"  Total: {self.total_samples} samples across {len(self.splits)} splits")

    def __len__(self):
        return self.total_samples

    def _resolve_index(self, idx):
        split_idx = bisect.bisect_right(self.cumulative, idx) - 1
        local_idx = idx - self.cumulative[split_idx]
        return split_idx, local_idx

    def get_split_and_local(self, idx):
        split_idx, local_idx = self._resolve_index(idx)
        # Return (name, ds, count); callers don't need the original indices.
        split_name, ds, count, _orig = self.splits[split_idx]
        return (split_name, ds, count), local_idx

    def stable_key(self, global_idx):
        """Return a cache key that is stable across data_pct values.

        Stride sampling ensures that the sample at (split, local_idx) in a
        20% run has a unique original_local_idx in the full split; the 20%
        subset is always a subset of 100% under the same stride logic.
        """
        split_idx, local_idx = self._resolve_index(global_idx)
        split_name, _ds, _count, original_indices = self.splits[split_idx]
        return f"{split_name}:{original_indices[local_idx]}"

    # ── Lazy captions ────────────────────────────────────────────

    @property
    def captions(self):
        if self._captions is not None:
            return self._captions

        if self._is_main:
            print("[Dataset] Extracting captions for mining index (batched, text-only)...")

        self._captions = []
        for split_name, ds, count, _orig in self.splits:
            text_ds = self._text_datasets[split_name]
            mapped = text_ds.map(
                _extract_caption_batch,
                batched=True,
                batch_size=1000,
                num_proc=4,
                remove_columns=text_ds.column_names,
                fn_kwargs={"split_name": split_name,
                           "use_original_caption": self.use_original_caption},
                desc=f"  Captions [{split_name}]" if self._is_main else None,
            )
            self._captions.extend(mapped["_caption"])

        if self._is_main:
            print(f"[Dataset] Extracted {len(self._captions)} captions")
        return self._captions

    # ── Lightweight text-only sample access (for teacher cache) ────

    def get_text_sample(self, global_idx):
        """Return a media-free sample dict suitable for text-only encoding.

        Used by the teacher-embedding precompute: pulls only the fields the
        positive-prompt builder needs (modality, pair_type, anchor_text,
        positive_text), without decoding audio/image/video. For omni
        samples, has_* flags default to False since we don't inspect media
        here — the teacher prompt falls back to the generic "Content
        context" label for omni, which is an intentional minor divergence
        from the training-time per-sample modality labeling.
        """
        split_idx, local_idx = self._resolve_index(global_idx)
        split_name, _ds, _count, _orig = self.splits[split_idx]
        text_ds = self._text_datasets[split_name]
        item = text_ds[local_idx]

        messages = item.get("messages", [])
        category = item.get("category", "caption")
        if category == "chat":
            q, a = _extract_qa_pair(messages)
            anchor_text, positive_text = q, a
        elif self.use_original_caption:
            anchor_text = ""
            positive_text = (item.get("original_text") or "").strip()
        else:
            anchor_text, positive_text = "", _extract_caption(messages, split_name=split_name)

        out = {
            "modality": split_name,
            "pair_type": category,
            "anchor_text": anchor_text,
            "positive_text": positive_text,
            "sample_id": global_idx,
        }
        if split_name == "omni":
            out.update(has_audio=False, has_image=False, has_video=False)
        return out

    # ── Sample access ────────────────────────────────────────────

    def __getitem__(self, idx):
        # Warn when a single sample fetch takes >10s (slow storage / huge media).
        import time as _t
        _t0 = _t.time()
        _rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
        split_idx, local_idx = self._resolve_index(idx)
        split_name, ds, count, _orig = self.splits[split_idx]

        _ts_row = _t.time()
        item = ds[local_idx]
        _tr = _t.time() - _ts_row
        if _tr > 10.0:
            print(f"[slow-sample rank{_rank}] ds[{local_idx}] row-read "
                  f"{_tr:.1f}s split={split_name} gid={idx}", flush=True)

        _ts_proc = _t.time()
        if split_name == "omni":
            out = self._process_omni_sample(item, idx)
        else:
            out = self._process_sample(item, split_name, idx)
        _tp = _t.time() - _ts_proc
        _ttot = _t.time() - _t0
        if _ttot > 10.0 or _tp > 10.0:
            print(f"[slow-sample rank{_rank}] sample gid={idx} split={split_name} "
                  f"local_idx={local_idx} row={_tr:.1f}s proc={_tp:.1f}s tot={_ttot:.1f}s",
                  flush=True)
        return out

    # ── Standard modality processing ─────────────────────────────

    def _process_sample(self, item, split_name, global_idx):
        messages = item.get("messages", [])
        category = item.get("category", "caption")

        if category == "chat":
            question, answer = _extract_qa_pair(messages)
            anchor_text = question
            positive_text = answer
        elif self.use_original_caption:
            anchor_text = ""
            positive_text = (item.get("original_text") or "").strip()
            if not positive_text:
                # Some samples (notably in visual_doc) have an empty
                # original_text; fall back to the dense caption for those so
                # the sample set is identical with and without
                # --original-caption.
                if not getattr(self, "_warned_empty_orig", False):
                    print(f"[dataset] use_original_caption: empty original_text "
                          f"(e.g. {split_name} gid={global_idx}) → falling back to "
                          f"dense caption for such samples (warned once).", flush=True)
                    self._warned_empty_orig = True
                positive_text = _extract_caption(messages, split_name=split_name)
        else:
            anchor_text = ""
            positive_text = _extract_caption(messages, split_name=split_name)

        result = {
            "modality": split_name,
            "pair_type": category,
            "anchor_text": anchor_text,
            "positive_text": positive_text,
            "sample_id": global_idx,
        }

        if split_name in ("speech", "audio"):
            audio_array = _get_audio_array(item.get("media"))
            if self.whisper_fe is None:
                raise RuntimeError(
                    f"[{split_name} gid={global_idx}] whisper_fe is None — "
                    "audio preprocessing requires a WhisperFeatureExtractor"
                )
            result.update(
                _preprocess_audio(audio_array, self.whisper_fe, self.max_audio_duration)
            )

        elif split_name in ("image", "visual_doc"):
            media = item.get("media")
            from PIL import Image
            if not isinstance(media, Image.Image):
                raise TypeError(
                    f"[{split_name} gid={global_idx}] expected PIL.Image, "
                    f"got {type(media).__name__}"
                )
            result["pixel_values"] = self.image_transform(media.convert("RGB"))

        elif split_name == "video":
            media = item.get("media")
            if media is None:
                raise ValueError(
                    f"[{split_name} gid={global_idx}] item['media'] is None"
                )
            from data.video_loader import load_video_frames
            frames, frame_indices = load_video_frames(
                media, self.max_video_frames, self.image_size,
            )
            result["frames"] = frames
            result["frame_indices"] = frame_indices

        return result

    # ── Omni (interleaved multi-modal) processing ────────────────

    def _process_omni_sample(self, item, global_idx):
        """Process an omni sample with mixed media (audio + image + video).

        A single omni sample can contain any combination of:
          - Video (with audio extracted from video track)
          - Image
          - Standalone audio
          - Question audio (spoken question, separate column)

        Media type is detected from file extension.
        All present media types are loaded and flagged.
        """
        messages = item.get("messages", [])
        question, answer = _extract_qa_pair(messages)

        media = item.get("media") or {}
        media_bytes = media.get("bytes") if isinstance(media, dict) else None
        media_path = media.get("path", "") if isinstance(media, dict) else ""
        ext = os.path.splitext(media_path)[1].lower()

        result = {
            "modality": "omni",
            "pair_type": "chat",
            "anchor_text": question,
            "positive_text": answer,
            "sample_id": global_idx,
            "has_audio": False,
            "has_image": False,
            "has_video": False,
        }

        # ── Main media (detected by extension) ──
        if ext in _VIDEO_EXTS:
            video_src = media_bytes or media_path
            # Video frames
            from data.video_loader import load_video_frames
            frames, frame_indices = load_video_frames(
                video_src, self.max_video_frames, self.image_size,
            )
            result["frames"] = frames
            result["frame_indices"] = frame_indices
            result["has_video"] = True

            # Extract audio from video track (None = silent video, legitimate)
            audio_array = _extract_audio_from_video(video_src)
            if audio_array is not None:
                if self.whisper_fe is None:
                    raise RuntimeError(
                        f"[omni gid={global_idx}] whisper_fe is None but "
                        "video has an audio track"
                    )
                result.update(
                    _preprocess_audio(audio_array, self.whisper_fe, self.max_audio_duration)
                )
                result["has_audio"] = True

        elif ext in _IMAGE_EXTS:
            from PIL import Image
            if not media_bytes:
                raise ValueError(
                    f"[omni gid={global_idx}] image ext but media has no bytes"
                )
            img = Image.open(_io.BytesIO(media_bytes)).convert("RGB")
            result["pixel_values"] = self.image_transform(img)
            result["has_image"] = True

        elif ext in _AUDIO_EXTS:
            audio_array = _decode_audio_bytes(media)
            if self.whisper_fe is None:
                raise RuntimeError(
                    f"[omni gid={global_idx}] whisper_fe is None but "
                    "sample has audio media"
                )
            result.update(
                _preprocess_audio(audio_array, self.whisper_fe, self.max_audio_duration)
            )
            result["has_audio"] = True

        else:
            raise ValueError(
                f"[omni gid={global_idx}] unrecognized media extension "
                f"{ext!r} (path={media_path!r})"
            )

        # ── Question audio (spoken question, separate column) ──
        if item.get("has_speech_question") and item.get("question_audio"):
            q_audio = _decode_audio_bytes(item["question_audio"])
            if self.whisper_fe is None:
                raise RuntimeError(
                    f"[omni gid={global_idx}] whisper_fe is None but "
                    "sample has a question_audio column"
                )
            q_feats = _preprocess_audio(q_audio, self.whisper_fe, self.max_audio_duration)
            if result["has_audio"]:
                result["mel_features"] = torch.cat(
                    [q_feats["mel_features"], result["mel_features"]], dim=-1
                )
                result["mel_len"] = q_feats["mel_len"] + result["mel_len"]
                result["dasheng_audio"] = torch.cat(
                    [q_feats["dasheng_audio"], result["dasheng_audio"]], dim=-1
                )
            else:
                result.update(q_feats)
                result["has_audio"] = True

        if not (result["has_audio"] or result["has_image"] or result["has_video"]):
            raise RuntimeError(
                f"[omni gid={global_idx}] no media loaded (ext={ext!r}, "
                f"path={media_path!r})"
            )

        return result

# ── Batch sampler ────────────────────────────────────────────────

class ModalityBatchSampler:
    """DDP-aware round-robin batch sampler: one batch per modality per cycle.

    Modalities: audio (speech + audio), image, video, visual_doc (+ omni if present).
    Omni is its OWN modality bucket (samples have mixed media — can't split).

    Each cycle emits one batch per modality. Smaller modalities are recycled.
    DDP sharding is handled internally (not by accelerate) to guarantee
    every rank sees every modality.
    """

    def __init__(self, dataset, batch_size, shuffle=True, seed=42,
                 batch_size_overrides=None, num_replicas=1, rank=0):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.batch_size_overrides = batch_size_overrides or {}
        self.num_replicas = num_replicas
        self.rank = rank

        # Build modality groups — omni stays as its own bucket
        self.modality_indices = {}
        for split_idx, (split_name, _ds, count, _orig) in enumerate(dataset.splits):
            mod = "audio" if split_name in ("speech", "audio") else split_name
            if mod not in self.modality_indices:
                self.modality_indices[mod] = []
            start = dataset.cumulative[split_idx]
            self.modality_indices[mod].extend(range(start, start + count))

        self.modalities = sorted(self.modality_indices.keys())
        self._epoch = 0

        if _is_main_process():
            for mod in self.modalities:
                bs = self.batch_size_overrides.get(mod, self.batch_size)
                n_batches = (len(self.modality_indices[mod]) + bs - 1) // bs
                print(f"  Sampler: {mod} = {len(self.modality_indices[mod])} samples, "
                      f"bs={bs}, {n_batches} batches/epoch")
            print(f"  DDP: {num_replicas} replicas, rank {rank}")

    def _batches_per_modality(self):
        """Return {modality: list[batch_indices]} for THIS rank.

        Two invariants hold by construction:

        1) Rank synchronization (DDP correctness). Every rank has the same
           per-modality batch count, so the round-robin in __iter__ yields
           the same modality at the same iteration index on every rank.
           Different modalities emit different tensor shapes and collective
           sequences in the forward; any per-step modality mismatch across
           ranks deadlocks NCCL.

        2) Cycle completeness ("smaller modalities are recycled"). After
           per-rank sharding, each modality is padded up to the per-rank
           count of the LARGEST modality by cyclic wrap, so every cycle
           emits exactly one batch per modality for the whole epoch.

        Padding for both invariants reuses the rank's OWN slice of
        all_batches where possible; if a modality had fewer global batches
        than num_replicas so this rank received zero, we fall back to the
        global list to keep the modality represented.
        """
        rng = np.random.RandomState(self.seed + self._epoch)
        per_mod = {}
        n_batches_full = {}
        for mod in self.modalities:
            idx = np.array(self.modality_indices[mod])
            if self.shuffle:
                rng.shuffle(idx)
            bs = self.batch_size_overrides.get(mod, self.batch_size)

            # Split this modality's indices into fixed-size batches.
            all_batches = [idx[i:i + bs].tolist()
                           for i in range(0, len(idx), bs)]
            n_batches_full[mod] = len(all_batches)
            if not all_batches:
                per_mod[mod] = []
                continue

            # This rank's slice, strided by num_replicas.
            my_batches = all_batches[self.rank::self.num_replicas]

            # Step 1 — rank synchronization: every rank has exactly
            # ceil(len(all_batches) / num_replicas) batches for this mod,
            # padded via cyclic wrap of all_batches (so ranks that began
            # empty still get real data).
            rank_target = (len(all_batches) + self.num_replicas - 1) // self.num_replicas
            if len(my_batches) < rank_target:
                for k in range(rank_target - len(my_batches)):
                    my_batches.append(all_batches[k % len(all_batches)])
            per_mod[mod] = my_batches

        # Step 2 — cycle completeness: recycle smaller modalities up to
        # the per-rank max so every round emits one batch per modality.
        max_per_rank = max((len(v) for v in per_mod.values()), default=0)
        for mod in list(per_mod.keys()):
            lst = per_mod[mod]
            if not lst or len(lst) == max_per_rank:
                continue
            for k in range(max_per_rank - len(lst)):
                lst.append(lst[k % len(lst)])
        return per_mod

    def __iter__(self):
        per_mod = self._batches_per_modality()
        # After both padding passes every rank has an identical per-
        # modality count, identical across modalities, identical across
        # ranks — round-robin is deterministic and rank-synchronous.
        max_per_mod = max((len(v) for v in per_mod.values()), default=0)
        for i in range(max_per_mod):
            for mod in self.modalities:
                if per_mod[mod]:  # skip modalities that had zero global batches
                    yield per_mod[mod][i]
        self._epoch += 1

    def __len__(self):
        """Batches this rank yields per epoch (identical across ranks)."""
        per_rank_counts = []
        for mod in self.modalities:
            bs = self.batch_size_overrides.get(mod, self.batch_size)
            n_batches_full = (len(self.modality_indices[mod]) + bs - 1) // bs
            if n_batches_full == 0:
                continue
            per_rank_counts.append(
                (n_batches_full + self.num_replicas - 1) // self.num_replicas
            )
        if not per_rank_counts:
            return 0
        max_per_rank = max(per_rank_counts)
        # Every non-empty modality is padded to max_per_rank, then
        # round-robin yields one batch per modality per cycle.
        n_nonempty_modalities = len(per_rank_counts)
        return max_per_rank * n_nonempty_modalities
