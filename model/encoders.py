"""
Audio encoder wrappers for Omni-Embed.

WhisperSpeechEncoderSeq: Full sequence (B, T_w, d) with >30s chunking.
DashengEncoder: Frame-level features (B, T_d, dim) via dasheng pip package.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperModel, WhisperConfig


WHISPER_MAX_MEL = 3000      # 30s at 100 mel frames/sec
WHISPER_OUT_FRAMES = 1500   # Whisper 2x downsample per chunk


class BaseEncoder(nn.Module):
    """Base class with shared freeze utilities."""

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_last_n(self, n):
        if hasattr(self, 'model') and hasattr(self.model, 'layers'):
            for layer in self.model.layers[-n:]:
                for p in layer.parameters():
                    p.requires_grad = True
            if hasattr(self.model, 'layer_norm'):
                for p in self.model.layer_norm.parameters():
                    p.requires_grad = True

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


class WhisperSpeechEncoderSeq(BaseEncoder):
    """
    Wraps a Whisper encoder (default whisper-small).
    Returns full sequence: (B, T_w, d_model), with >30s chunking support.
    """

    def __init__(self, model_name="openai/whisper-small"):
        super().__init__()
        self._model_name = model_name
        wconfig = WhisperConfig.from_pretrained(model_name)
        whisper = WhisperModel(wconfig)
        self.model = whisper.encoder
        self.output_dim = wconfig.d_model

    def load_pretrained_weights(self):
        whisper = WhisperModel.from_pretrained(self._model_name)
        self.model.load_state_dict(whisper.encoder.state_dict())
        self.output_dim = whisper.config.d_model

    def forward(self, input_features):
        """
        Args:
            input_features: (B, n_mels, T) — mel spectrogram features
        Returns:
            (B, T_w, d_model) — with chunking for T > 3000
        """
        target_dtype = next(self.model.parameters()).dtype
        input_features = input_features.to(dtype=target_dtype)

        T = input_features.shape[-1]
        if T <= WHISPER_MAX_MEL:
            if T < WHISPER_MAX_MEL:
                input_features = F.pad(input_features, (0, WHISPER_MAX_MEL - T))
            return self.model(input_features=input_features).last_hidden_state

        chunks = []
        for start in range(0, T, WHISPER_MAX_MEL):
            chunk = input_features[:, :, start:start + WHISPER_MAX_MEL]
            if chunk.shape[-1] < WHISPER_MAX_MEL:
                chunk = F.pad(chunk, (0, WHISPER_MAX_MEL - chunk.shape[-1]))
            chunks.append(self.model(input_features=chunk).last_hidden_state)
        return torch.cat(chunks, dim=1)


class DashengEncoder(BaseEncoder):
    """
    Wraps Dasheng audio encoder via dasheng pip package.
    Returns frame-level features: (B, T_d, dim).
    """

    DASHENG_MODELS = {
        "mispeech/dasheng-base": "dasheng_base",
        "dasheng-base": "dasheng_base",
    }

    DASHENG_DIMS = {
        "mispeech/dasheng-base": 768, "dasheng-base": 768,
    }

    def __init__(self, model_name="mispeech/dasheng-base"):
        super().__init__()
        self._model_name = model_name
        self.output_dim = self.DASHENG_DIMS.get(model_name, 768)
        # Eager load — no silent fallback. If the dasheng load fails
        # (network, missing file, dependency), crash here so the user
        # knows the audio path is broken before training starts.
        self.model = self._load_dasheng(model_name)
        self.output_dim = self.model.embed_dim
        self._pin_front_end_to_cpu()

    def load_pretrained_weights(self):
        if self.model is None:
            self.model = self._load_dasheng(self._model_name)
            self.output_dim = self.model.embed_dim
            self._pin_front_end_to_cpu()

    def _pin_front_end_to_cpu(self):
        # ROCm cuFFT crashes on dasheng's GPU STFT for short/edge-case
        # audio shapes (HIPFFT_PARSE_ERROR / HIPFFT_INTERNAL_ERROR).
        # Pin only the mel front_end + init_bn to CPU; the heavy
        # transformer body (forward_spectrogram) stays on whatever
        # device the rest of dasheng lives on. The front_end is tiny
        # (a window buffer + mel filterbank) so the CPU STFT cost is
        # well under 1ms per batch — negligible during training.
        self.model.front_end.to("cpu")
        self.model.init_bn.to("cpu")
        self._cpu_pinned = True

    @staticmethod
    def _load_dasheng(name):
        import dasheng
        loader_name = DashengEncoder.DASHENG_MODELS.get(name)
        if loader_name is None:
            raise ValueError(
                f"Unknown dasheng model: {name!r}. Supported: {list(DashengEncoder.DASHENG_MODELS.keys())}"
            )
        return getattr(dasheng, loader_name)()

    def forward(self, raw_audio):
        """
        Args:
            raw_audio: (B, T_samples) — raw 16kHz mono waveform
        Returns:
            (B, T_d, dim)
        """
        if self.model is None:
            self.load_pretrained_weights()
            self.model = self.model.to(raw_audio.device)
        dev = raw_audio.device
        # OmniEmbedModel.to(device) at training-init time would otherwise
        # drag the mel front_end back to GPU and re-trigger the HIPFFT
        # crash on edge-case shapes. Re-pin every call — it is a no-op
        # if already on CPU and only costs a one-time .to() per training
        # step's first audio batch.
        front_buf = next(self.model.front_end.buffers(), None)
        if front_buf is None or front_buf.device.type != "cpu":
            self.model.front_end.to("cpu")
            self.model.init_bn.to("cpu")
        with torch.amp.autocast("cuda", enabled=False):
            m = self.model.float()
            spec = m.forward_to_spec(raw_audio.float().cpu())
            return m.forward_spectrogram(spec.to(dev))
