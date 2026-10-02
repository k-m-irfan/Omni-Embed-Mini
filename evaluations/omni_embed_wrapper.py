"""Universal mteb encoder wrapper for an exported OmniEmbed model.

Works across all mteb-hosted benchmarks — text (MTEB-v2 BEIR), audio (MAEB),
image (ViDoRe, MMEB-v2), video (MMEB-v2) — by routing each batch through
the model's own AutoProcessor and then calling OmniEmbedForEmbedding.encode().

Text-only batches take a fast path that bypasses the processor:
`model.tokenize_text_native(...)` + `model.encode_text(...)` produce
embeddings in the backbone's native text subspace (passages raw,
queries `Instruct: {instr}\\nQuery: …`, explicit <|endoftext|>, left-
padded). Mixed / media batches still go through the processor's chat
template per item, with media features injected at placeholder rows.

Batches from mteb arrive as dicts whose keys depend on the task:
  - text retrieval:   {"text": [...]}  OR  {"body": [...], "title": [...]}
  - audio retrieval:  {"audio": [AudioArray, …], optional "text": [...]}
  - image retrieval:  {"image": [PIL, …],         optional "text": [...]}
  - video retrieval:  {"video": [[frames], …],    optional "text": [...]}

We pass one item at a time to the processor (it does not batch mixed
inputs), stack embeddings, and return (N, D) L2-normalised float32.
"""
from __future__ import annotations

import numpy as np
import torch

MAX_TOKENS = 1024  # hard cap to keep VRAM bounded; Qwen3 supports 32k but we rarely need it.

# Default Qwen3-Embedding retrieval instruction (matches their mteb loader).
DEFAULT_QUERY_INSTR = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


class OmniEmbedMTEBWrapper:
    def __init__(self, model_path: str, device: str = "cuda:0", batch_size: int = 8,
                 name: str | None = None):
        import importlib.util
        import json as _json
        import sys as _sys
        from transformers import AutoProcessor
        from mteb.models.model_meta import ModelMeta

        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size

        # Load the custom modeling + configuration from the exported dir
        # without going through AutoModel.from_pretrained (which insists on a
        # top-level model.safetensors / pytorch_model.bin). The exported layout
        # keeps backbone weights in `backbone/` and projector/LoRA weights in
        # `projector_weights.pt`; both are wired up by the model's deferred
        # `_load_weights(model_dir)` on first forward.
        #
        # modeling_omni_embed.py uses relative imports, so we register a
        # synthetic package and attach both submodules under it.
        pkg_name = "_omni_embed_exported"
        pkg_spec = importlib.util.spec_from_loader(pkg_name, loader=None, is_package=True)
        pkg = importlib.util.module_from_spec(pkg_spec)
        pkg.__path__ = [model_path]
        _sys.modules[pkg_name] = pkg

        def _load_sub(sub):
            spec = importlib.util.spec_from_file_location(
                f"{pkg_name}.{sub}", f"{model_path}/{sub}.py",
            )
            mod = importlib.util.module_from_spec(spec)
            _sys.modules[f"{pkg_name}.{sub}"] = mod
            spec.loader.exec_module(mod)
            return mod

        cfg_mod = _load_sub("configuration_omni_embed")
        mdl_mod = _load_sub("modeling_omni_embed")

        with open(f"{model_path}/config.json") as f:
            cfg_dict = _json.load(f)
        config = cfg_mod.OmniEmbedConfig(**cfg_dict)
        config._name_or_path = model_path  # _load_weights reads from here

        self.model = mdl_mod.OmniEmbedForEmbedding(config).eval().to(device)
        # Image-projector weights load in fp32 by default but the vision
        # encoder pipes bf16 features, which crashes on the first image
        # forward (mat1=BFloat16 vs mat2=Float). Cast everything to bf16 —
        # matches training-time dtype and is safe for the text path
        # (Qwen3 backbone was trained in bf16).
        self.model = self.model.to(dtype=torch.bfloat16)
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True,
        )
        # Give the model direct access to the tokenizer so the native
        # Qwen3-Embedding text path (model.tokenize_text_native +
        # model.encode_text) can run without going through the processor.
        # padding_side is flipped locally inside tokenize_text_native and
        # restored afterwards — no global mutation.
        self.text_tokenizer = self.processor.tokenizer
        self.model.tokenizer = self.text_tokenizer
        # Forward pad-ids into the model so build_inputs_embeds can locate
        # media placeholder rows. The processor already stored these.
        self.model.set_pad_ids(
            audio_pad=self.processor.audio_pad_id,
            image_pad=self.processor.image_pad_id,
            video_pad=self.processor.video_pad_id,
        )

        # Tag the cache name when OMNI_*_RECIPE is set to anything other than
        # 'native', so flipping it doesn't clobber prior cache entries.
        import os as _os
        q_recipe = _os.environ.get("OMNI_QUERY_RECIPE", "native").lower()
        c_recipe = _os.environ.get("OMNI_CORPUS_RECIPE", "native").lower()
        tag = ""
        if (q_recipe, c_recipe) != ("native", "native"):
            tag = f"_q-{q_recipe}_c-{c_recipe}"
        meta = ModelMeta.create_empty()
        name = name or model_path.rstrip('/').rsplit('/', 1)[-1]
        meta.name = f"OmniEmbed[{name}{tag}]"
        meta.modalities = ["text", "image", "audio", "video"]
        self.mteb_model_meta = meta

    # ── Per-item processor call → flat kwarg dict on device ──
    def _process_one(self, text=None, audio=None, image=None, video=None,
                     instruction=None, is_query=False):
        kwargs = {"return_tensors": "pt", "is_query": is_query}
        if instruction is not None:
            kwargs["instruction"] = instruction
        if text is not None:
            kwargs["text"] = text
        if audio is not None:
            kwargs["audio"] = audio
        if image is not None:
            kwargs["images"] = image
        if video is not None:
            kwargs["videos"] = video
        enc = self.processor(**kwargs)
        # Truncate text tokens to MAX_TOKENS to bound VRAM on long passages —
        # only for pure-text rows. With media present, blind clipping chops
        # off image/video/audio_pad tokens while pixel_values/whisper_features
        # stay full, breaking the backbone's tokens==features check.
        has_media = any(k in enc for k in
                        ("pixel_values", "pixel_values_videos", "whisper_features"))
        if not has_media and enc["input_ids"].shape[-1] > MAX_TOKENS:
            enc["input_ids"] = enc["input_ids"][:, :MAX_TOKENS]
            enc["attention_mask"] = enc["attention_mask"][:, :MAX_TOKENS]
        return {k: (v.to(self.device) if torch.is_tensor(v) else v)
                for k, v in enc.items()}

    def _encode_one(self, instruction=None, is_query=False, **media) -> np.ndarray:
        kwargs = self._process_one(**media, instruction=instruction, is_query=is_query)
        with torch.no_grad():
            emb = self.model.encode(**kwargs)
        return emb.float().cpu().numpy()

    def _encode_text_batch(self, texts, *, is_query, instruction) -> np.ndarray:
        """Batched native Qwen3-Embedding text path.

        Bypasses the processor's chat template entirely — inputs land in
        the backbone's native text subspace.
        """
        enc = self.model.tokenize_text_native(
            texts, is_query=is_query, instruction=instruction,
            max_length=MAX_TOKENS, device=self.device,
        )
        with torch.no_grad():
            emb = self.model.encode_text(enc["input_ids"], enc["attention_mask"])
        return emb.float().cpu().numpy()

    def _encode_chat_text_batch(self, texts, *, instruction=None) -> np.ndarray:
        """Batched chat-wrapped text path (matches the training-time positive format)."""
        # Inlined from data/collator.EMBED_TEXT_TEMPLATE to avoid pulling the
        # training package onto the eval worker's sys.path.
        # OMNI_NATIVE_FULL_CHAT=1 reproduces Qwen3-VL-Embedding's standalone
        # ST template for native-mode checkpoints (2.3B) so query and doc
        # share the backbone's pretraining pooling position.
        import os as _os
        if (getattr(self.model, "vision_mode", "custom") == "native"
                and _os.environ.get("OMNI_NATIVE_FULL_CHAT", "0") == "1"):
            # Byte-identical to Qwen3-VL-Embedding's native ST chat template
            # (rendered by `apply_chat_template(..., add_generation_prompt=True)`).
            # No trailing <|endoftext|> — native pools at the `\n` after
            # `assistant` (last attended token under right-padding).
            sys_text = (instruction or "Represent the user's input.").strip()
            if sys_text and sys_text[-1] not in ".!?":
                sys_text += "."
            template = (
                f"<|im_start|>system\n{sys_text}<|im_end|>\n"
                "<|im_start|>user\n{text}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            _padding_side = "right"
        else:
            template = "<|im_start|>user\n{text}\n<|im_end|>"
            _padding_side = self.model.tokenizer.padding_side
        wrapped = [template.format(text=t) for t in texts]
        tok = self.model.tokenizer
        _prev_side = tok.padding_side
        tok.padding_side = _padding_side
        try:
            enc = tok(
                wrapped, padding=True, truncation=True,
                max_length=MAX_TOKENS, return_tensors="pt",
            )
        finally:
            tok.padding_side = _prev_side
        input_ids = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)
        with torch.no_grad():
            emb = self.model.encode_text(input_ids, attn)
        return emb.float().cpu().numpy()

    # ── Extract N items from a batch dict, per modality ──
    @staticmethod
    def _rows(batch):
        # Determine per-item media from the flexible mteb batch shape
        # (column-oriented: dict of lists).
        n = 0
        for k in ("text", "body", "audio", "image", "video"):
            if k in batch and batch[k]:
                n = max(n, len(batch[k]))
        texts = batch.get("text") or [None] * n
        bodies = batch.get("body") or [None] * n
        titles = batch.get("title") or [None] * n
        audios = batch.get("audio") or [None] * n
        images = batch.get("image") or [None] * n
        videos = batch.get("video") or [None] * n
        # mteb's AudioCollator yields several shapes: np.ndarray, dict
        # {"array": ndarray, "sampling_rate": int}, our patch_audio shim
        # (with .keys() + __getitem__("array")), or a torch.Tensor. Our
        # processor expects a 1-D np.ndarray (float32), so normalise here.
        def _to_np(a):
            if a is None:
                return None
            if isinstance(a, np.ndarray):
                return a.astype(np.float32, copy=False).reshape(-1)
            if hasattr(a, "numpy"):  # torch.Tensor
                return a.detach().cpu().numpy().astype(np.float32).reshape(-1)
            if isinstance(a, dict) and "array" in a:
                return _to_np(a["array"])
            if hasattr(a, "__getitem__"):
                try:
                    return _to_np(a["array"])
                except (KeyError, TypeError):
                    pass
            return np.asarray(a, dtype=np.float32).reshape(-1)
        audios = [_to_np(a) for a in audios]
        out = []
        for i in range(n):
            txt = None
            if i < len(bodies) and bodies[i] is not None:
                t = titles[i] if i < len(titles) and titles[i] is not None else None
                txt = f"{t} {bodies[i]}".strip() if t else bodies[i]
            elif i < len(texts) and texts[i] is not None:
                txt = texts[i]
            out.append({
                "text":  txt,
                "audio": audios[i] if i < len(audios) else None,
                "image": images[i] if i < len(images) else None,
                "video": videos[i] if i < len(videos) else None,
            })
        return out

    # ── mteb EncoderProtocol ──
    @staticmethod
    def _resolve_instruction(task_metadata, prompt_type):
        """Pull the right instruction string out of mteb task_metadata.

        mteb stores task-specific retrieval instructions on task_metadata.prompt
        as either a str or a {query, passage} dict. It is passed through so
        the processor can render `Instruct: {instr}\\nQuery: …`.
        """
        if task_metadata is None:
            return None
        # 1) Explicit prompt on task metadata (model-side prompts_dict aren't
        #    wired here — we're not a SentenceTransformer-style loader).
        p = getattr(task_metadata, "prompt", None)
        if isinstance(p, dict):
            from mteb.types import PromptType
            if prompt_type == PromptType.query:
                return p.get("query") or next(iter(p.values()), None)
            if prompt_type == PromptType.document:
                return p.get("passage") or p.get("document")
            return next(iter(p.values()), None)
        if p:
            return p
        # 2) Fall back to the task's own abstask_prompt (mteb's default).
        #    For ArguAna this yields "Given a claim, find documents that
        #    refute the claim" — matches what the Qwen3-Embedding builtin
        #    loader resolves. No silent fallback: if mteb can't resolve the
        #    task, surface the error so the caller fixes the task list.
        from mteb.get_tasks import get_task
        ap = getattr(get_task(task_name=task_metadata.name), "abstask_prompt", None)
        return ap

    def encode(self, inputs, *, task_metadata=None, hf_split="", hf_subset="",
               prompt_type=None, **kwargs) -> np.ndarray:
        from mteb.types import PromptType
        is_query = prompt_type == PromptType.query
        instruction = self._resolve_instruction(task_metadata, prompt_type)

        # ── Per-task recipe selection (native backbone only) ──────────────
        # Qwen3-VL-Embedding was contrastively pretrained on the full ST chat
        # template (system+user+assistant, pool @ trailing "\n"). Our training
        # used the simpler user-only template for every modality, so:
        #   • native image/video/visdoc tasks → use the full chat template
        #     (nothing we trained lives on that path; query text and doc image
        #     must share the template or cross-modal cosine collapses).
        #   • audio/speech tasks → keep the user-only template the
        #     whisper/dasheng projectors were trained on.
        #   • pure-text tasks → keep the default recipe.
        # 0.9B (vision_mode=custom) ignores the flag (gated on vision_mode).
        import os as _os_fc
        _mods_fc = [str(m).lower()
                    for m in (getattr(task_metadata, "modalities", None) or [])]
        _native_media = (
            getattr(self.model, "vision_mode", "custom") == "native"
            and any(m in _mods_fc for m in ("image", "video"))
            and "audio" not in _mods_fc
        )
        _os_fc.environ["OMNI_NATIVE_FULL_CHAT"] = "1" if _native_media else "0"

        parts: list[np.ndarray] = []
        for batch in inputs:
            rows = self._rows(batch)
            rows = [r for r in rows
                    if not all(r[k] is None for k in ("text", "audio", "image", "video"))]

            # ── Visual-document protocol ──────────────────────────────────
            # ViDoRe corpus rows carry both a page image and its OCR text. By
            # default both are encoded (image + text), the protocol used for
            # the reported ViDoRe numbers. Set OMNI_VISDOC_IMAGE_ONLY=1 to
            # encode the page image only.
            _v = _os_fc.environ.get("OMNI_VISDOC_IMAGE_ONLY", "0").strip().lower()
            if _v in ("1", "true", "yes", "on"):
                _imgonly = True
            elif _v in ("0", "false", "no", "off"):
                _imgonly = False
            else:
                raise ValueError(
                    f"OMNI_VISDOC_IMAGE_ONLY={_v!r} is not a recognised boolean. "
                    "Use 1/0 (or true/false). Refusing to guess: a silently "
                    "mis-parsed value would switch the ViDoRe protocol and "
                    "make results non-comparable to the reported numbers."
                )
            if _imgonly and prompt_type == PromptType.document:
                for r in rows:
                    if r["image"] is not None:
                        r["text"] = None
            if not rows:
                continue

            # Fast path: every row in this batch is text-only → tokenize and
            # forward as a single batch. Switchable recipes via env vars
            # OMNI_QUERY_RECIPE / OMNI_CORPUS_RECIPE = native | chat
            # (default: native for custom vision_mode, chat for native):
            #   • native = the backbone's Qwen3-Embedding query/passage format.
            #   • chat   = the training-time positive format, which matches
            #     media inputs encoded through the processor.
            import os as _os
            # Native vision_mode (2.3B) uses the Qwen3-VL-Embedding backbone,
            # which was contrastively pretrained on the chat template with
            # last-token pooling — defaults must be chat to stay in the
            # backbone's pretraining subspace. Custom vision_mode keeps the
            # Qwen3-Embedding-0.6B native recipe defaults.
            _is_native = getattr(self.model, "vision_mode", "custom") == "native"
            _default_recipe = "chat" if _is_native else "native"
            q_recipe = _os.environ.get("OMNI_QUERY_RECIPE", _default_recipe).lower()
            c_recipe = _os.environ.get("OMNI_CORPUS_RECIPE", _default_recipe).lower()
            # Media-mixed tasks (ViDoRe, MMEB, MAEB audio): the corpus side
            # encodes images/audio/video through the processor's chat
            # template and pools at <|im_end|>. The model's projector was
            # only trained against that pooling subspace. Forcing the
            # query side onto the native Qwen3-Embedding path (which
            # pools at <|endoftext|>) lands queries in a different
            # subspace from the docs → cosine ~ random → ndcg ≈ 0.
            # Override both sides to chat recipe whenever the task's
            # corpus contains a non-text modality.
            mods = list(getattr(task_metadata, "modalities", None) or [])
            if any(m in mods for m in ("image", "audio", "video")):
                q_recipe = c_recipe = "chat"
                if not getattr(self, "_logged_chat_override", False):
                    tname = getattr(task_metadata, "name", "?")
                    print(f"[wrapper] media-mixed task '{tname}' "
                          f"(modalities={mods}) → forcing chat recipe", flush=True)
                    self._logged_chat_override = True
            if all(r["audio"] is None and r["image"] is None and r["video"] is None
                   and r["text"] is not None for r in rows):
                texts = [r["text"] for r in rows]
                recipe = q_recipe if is_query else c_recipe
                if recipe == "native":
                    parts.append(self._encode_text_batch(
                        texts, is_query=is_query, instruction=instruction,
                    ))
                else:
                    parts.append(self._encode_chat_text_batch(
                        texts, instruction=instruction,
                    ))
                continue

            # Mixed / media path: per-item through the processor.
            for item in rows:
                parts.append(self._encode_one(
                    instruction=instruction, is_query=is_query, **item,
                ))
        return np.concatenate(parts, axis=0) if parts else np.zeros((0, 1), dtype=np.float32)

    def similarity(self, e1, e2):
        if isinstance(e1, torch.Tensor):
            return torch.nn.functional.cosine_similarity(
                e1.unsqueeze(1), e2.unsqueeze(0), dim=2,
            )
        return np.asarray(e1, dtype=np.float32) @ np.asarray(e2, dtype=np.float32).T

    def similarity_pairwise(self, e1, e2):
        if isinstance(e1, torch.Tensor):
            return torch.nn.functional.cosine_similarity(e1, e2, dim=1)
        e1 = np.asarray(e1, dtype=np.float32)
        e2 = np.asarray(e2, dtype=np.float32)
        return np.sum(e1 * e2, axis=1)
