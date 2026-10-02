"""Monkey-patch datasets to use soundfile/PyAV instead of torchcodec for audio decoding.

torchcodec is not supported on AMD/ROCm. This patch replaces the Audio.decode_example
method to use soundfile for WAV/FLAC and PyAV for other formats.

Import this BEFORE any datasets audio loading:
    import patch_audio  # noqa: F401
"""

import io
import numpy as np


class _AudioDecoderShim:
    """Drop-in replacement for torchcodec.decoders.AudioDecoder.

    datasets>=4 returns AudioDecoder objects from Audio features.
    mteb accesses them via .get_all_samples() or via the dict protocol
    with "array" and "sampling_rate" keys.
    """

    def __init__(self, source, stream_index=None, sample_rate=None, num_channels=None):
        self._sample_rate = sample_rate or 16000
        self._num_channels = num_channels
        self._array = None
        self._source_sr = None
        self._hf_encoded = {"path": None, "bytes": None}

        class _Meta:
            path = None
        self.metadata = _Meta()

        self._load(source)

    def _load(self, source):
        import soundfile as sf

        if isinstance(source, (str, bytes)) and isinstance(source, str):
            # File path
            arr, sr = sf.read(source, dtype="float32")
        elif isinstance(source, bytes):
            arr, sr = sf.read(io.BytesIO(source), dtype="float32")
        elif hasattr(source, "read"):
            arr, sr = sf.read(source, dtype="float32")
        else:
            raise ValueError(f"Cannot decode audio from {type(source)}")

        if arr.ndim > 1:
            arr = arr.mean(axis=1)

        self._array = arr.astype(np.float32)
        self._source_sr = sr

        # Resample if needed. librosa is a hard requirement for the ROCm path:
        # without it, CLAP/Whisper/Dasheng processors that expect a specific
        # sampling rate would silently get wrong-rate audio and produce
        # garbage embeddings. Fail loud if librosa is missing.
        if self._sample_rate and sr != self._sample_rate:
            import librosa
            self._array = librosa.resample(
                self._array, orig_sr=sr, target_sr=self._sample_rate,
            )

    def get_all_samples(self):
        """Mimic torchcodec AudioDecoder.get_all_samples()."""
        import torch

        class _Samples:
            def __init__(self, data, sr):
                self.data = data
                self.sample_rate = sr
                self.pts_seconds = 0.0
                self.duration_seconds = len(data[0]) / sr if len(data.shape) > 1 else len(data) / sr

        arr_tensor = torch.from_numpy(self._array).unsqueeze(0)  # (1, T)
        sr = self._sample_rate or self._source_sr
        return _Samples(arr_tensor, sr)

    def __getitem__(self, key):
        """Support dict-like access: audio["array"], audio["sampling_rate"].
        Also supports audio["audio"] as self-reference for mteb collator compat.
        """
        if key == "array":
            if not isinstance(self._array, np.ndarray):
                self._array = np.array(self._array, dtype=np.float32)
            return self._array
        elif key == "sampling_rate":
            return self._sample_rate or self._source_sr
        elif key == "audio":
            # Self-reference: mteb collator does audio["audio"] to unwrap nested dicts
            return self
        elif key == "path":
            return self._hf_encoded.get("path")
        elif key == "bytes":
            return self._hf_encoded.get("bytes")
        raise KeyError(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key):
        return key in ("array", "sampling_rate", "audio", "path", "bytes")

    def keys(self):
        return ["array", "sampling_rate", "audio", "path", "bytes"]


def _patched_decode_example(self, value, token_per_repo_id=None):
    """Replacement for Audio.decode_example that uses soundfile instead of torchcodec."""
    if not self.decode:
        raise RuntimeError("Decoding is disabled for this feature.")

    path = value.get("path")
    audio_bytes = value.get("bytes")

    if path is None and audio_bytes is None:
        raise ValueError(f"Audio sample needs 'path' or 'bytes', got {value}")

    if audio_bytes is not None:
        audio = _AudioDecoderShim(
            audio_bytes,
            stream_index=getattr(self, 'stream_index', None),
            sample_rate=self.sampling_rate,
            num_channels=getattr(self, 'num_channels', None),
        )
    elif path is not None:
        from datasets.utils.file_utils import is_local_path
        if is_local_path(path):
            audio = _AudioDecoderShim(
                path,
                stream_index=getattr(self, 'stream_index', None),
                sample_rate=self.sampling_rate,
                num_channels=getattr(self, 'num_channels', None),
            )
        else:
            # Remote file — download first
            from datasets.download import DownloadConfig
            from datasets.utils.file_utils import xopen
            from datasets.table import string_to_dict
            from datasets import config

            token_per_repo_id = token_per_repo_id or {}
            source_url = path.split("::")[-1]
            pattern = (
                config.HUB_DATASETS_URL
                if source_url.startswith(config.HF_ENDPOINT)
                else config.HUB_DATASETS_HFFS_URL
            )
            source_url_fields = string_to_dict(source_url, pattern)
            token = (
                token_per_repo_id.get(source_url_fields["repo_id"])
                if source_url_fields is not None
                else None
            )
            download_config = DownloadConfig(token=token)
            f = xopen(path, "rb", download_config=download_config)
            audio = _AudioDecoderShim(
                f,
                stream_index=getattr(self, 'stream_index', None),
                sample_rate=self.sampling_rate,
                num_channels=getattr(self, 'num_channels', None),
            )

    audio._hf_encoded = {"path": path, "bytes": audio_bytes}
    audio.metadata.path = path
    return audio


# Apply the monkey-patch
from datasets.features.audio import Audio
Audio.decode_example = _patched_decode_example
print("[patch_audio] Patched datasets Audio to use soundfile instead of torchcodec")
