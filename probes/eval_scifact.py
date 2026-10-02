"""SciFact text + speech evaluation for training-time monitoring.

Evaluates the current training model on SciFact retrieval using both
text queries and speech queries. If alignment is working, text and
speech NDCG@10 should converge.

Usage from training loop:
    from probes.eval_scifact import evaluate_scifact
    results = evaluate_scifact(model, tokenizer, whisper_fe, device)
"""

import json
import os
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

# Optional spoken SciFact queries: <repo>/cache/probes/speech_cache/SciFact/
# {manifest.json, <qid>.wav}. Without them only the text score is reported.
_REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEECH_CACHE_DIR = os.path.join(_REPO_DIR, "cache", "probes", "speech_cache")

WHISPER_MAX_MEL = 3000
# Media anchor template (student path) — stays chat-wrapped because the
# media placeholders need a stable <|im_end|> for EOS pooling regardless
# of per-sample token count.
EMBED_AUDIO_TEMPLATE = "<|im_start|>user\n{audio_placeholder}\n<|im_end|>"

# SciFact task description passed to model.tokenize_text_native when
# is_query=True — it expands into "Instruct: {X}\nQuery: {q}<|endoftext|>".
SCIFACT_TASK_INSTRUCTION = (
    "Given a scientific claim, retrieve documents that support or refute "
    "the claim"
)


# ── Data loading ──────────────────────────────────────────────

def load_task_data(task_name="SciFact"):
    """Load queries, corpus, relevant_docs from MTEB task."""
    import mteb

    tasks = mteb.get_tasks(tasks=[task_name])
    if not tasks:
        raise ValueError(f"Task '{task_name}' not found")

    task = tasks[0]
    task.load_data()

    queries, corpus, relevant_docs = {}, {}, {}

    for subset_name, subset_data in task.dataset.items():
        for split in ["test", "dev", "validation"]:
            if split in subset_data:
                split_data = subset_data[split]
                if "queries" in split_data:
                    for item in split_data["queries"]:
                        qid = str(item.get("id", item.get("_id", "")))
                        text = item.get("text", "")
                        if qid and text:
                            queries[qid] = text
                if "corpus" in split_data:
                    for item in split_data["corpus"]:
                        cid = str(item.get("id", item.get("_id", "")))
                        corpus[cid] = {
                            "title": item.get("title", ""),
                            "text": item.get("text", ""),
                        }
                if "relevant_docs" in split_data:
                    relevant_docs = split_data["relevant_docs"]
                if queries:
                    break
        if queries:
            break

    return queries, corpus, relevant_docs


# ── Metrics ───────────────────────────────────────────────────

def compute_retrieval_scores(query_embs, corpus_embs, query_ids, corpus_ids,
                             relevant_docs, k_values=[1, 3, 5, 10, 100]):
    """Compute NDCG, MAP, Recall at k."""
    corpus_id_to_idx = {cid: idx for idx, cid in enumerate(corpus_ids)}
    sims = query_embs @ corpus_embs.T

    ndcg_scores = {k: [] for k in k_values}
    recall_scores = {k: [] for k in k_values}
    map_scores = []

    for i, qid in enumerate(query_ids):
        rels = relevant_docs.get(qid, {})
        if not rels:
            continue

        scores = sims[i].copy()
        if qid in corpus_id_to_idx:
            scores[corpus_id_to_idx[qid]] = -np.inf
        top_indices = np.argsort(-scores)

        for k in k_values:
            topk_ids = [corpus_ids[idx] for idx in top_indices[:k]]
            dcg = sum(rels.get(did, 0) / np.log2(rank + 2)
                      for rank, did in enumerate(topk_ids))
            ideal_scores = sorted(rels.values(), reverse=True)[:k]
            idcg = sum(s / np.log2(rank + 2) for rank, s in enumerate(ideal_scores))
            ndcg_scores[k].append(dcg / idcg if idcg > 0 else 0)

            n_relevant = len([d for d, s in rels.items() if s > 0])
            n_found = len([d for d in topk_ids if rels.get(d, 0) > 0])
            recall_scores[k].append(n_found / max(1, n_relevant))

        relevant_set = {d for d, s in rels.items() if s > 0}
        ap, n_rel_found = 0, 0
        for rank, idx in enumerate(top_indices[:1000]):
            did = corpus_ids[idx]
            if did in relevant_set:
                n_rel_found += 1
                ap += n_rel_found / (rank + 1)
        map_scores.append(ap / max(1, len(relevant_set)))

    metrics = {}
    for k in k_values:
        metrics[f"ndcg@{k}"] = np.mean(ndcg_scores[k]) if ndcg_scores[k] else 0
        metrics[f"recall@{k}"] = np.mean(recall_scores[k]) if recall_scores[k] else 0
    metrics["map"] = np.mean(map_scores) if map_scores else 0
    return metrics


# ── Encoding helpers ──────────────────────────────────────────

@torch.no_grad()
def _encode_texts(model, texts, device, *, is_query, instruction=None,
                  batch_size=64, truncate_dim=None, max_length=512):
    """Encode text strings through the vanilla Qwen3-Embedding text path.

    is_query=True  → tokenize_text_native wraps in "Instruct: …\\nQuery: …"
    is_query=False → passages go in raw
    Both paths append <|endoftext|>, left-pad, add_special_tokens=False.
    """
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        enc = model.tokenize_text_native(
            batch_texts, is_query=is_query, instruction=instruction,
            max_length=max_length, device=device,
        )
        emb = model.encode_text(enc["input_ids"], enc["attention_mask"])
        if truncate_dim:
            emb = F.normalize(emb[:, :truncate_dim], p=2, dim=-1)
        all_embs.append(emb.cpu().float().numpy())
    return np.concatenate(all_embs, axis=0)


@torch.no_grad()
def _encode_speech(model, tokenizer, whisper_fe, audio_paths, device,
                   tokens_per_encoder=128, batch_size=16, truncate_dim=None):
    """Encode speech files through the model's full audio path."""
    all_embs = []

    for i in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[i:i + batch_size]
        B = len(batch_paths)

        # Load and preprocess audio
        mel_list, dasheng_list, token_counts = [], [], []
        for path in batch_paths:
            audio, sr = sf.read(path)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

            if len(audio) < 400:
                audio = np.pad(audio, (0, 400 - len(audio)))

            mel = whisper_fe(
                audio, sampling_rate=16000, return_tensors="pt",
            )["input_features"][0]  # (n_mels, T_mel)
            mel_list.append(mel)

            dasheng = torch.zeros(1, len(audio), dtype=torch.float32)
            dasheng[0, :len(audio)] = torch.from_numpy(audio)
            dasheng_list.append(dasheng)

            mel_T = mel.shape[-1]
            n_chunks = max(1, (mel_T + WHISPER_MAX_MEL - 1) // WHISPER_MAX_MEL)
            token_counts.append(n_chunks * tokens_per_encoder * 2)

        # Build prompts with audio placeholders
        texts = []
        for tc in token_counts:
            placeholder = "<|audio_start|>" + "<|audio_pad|>" * tc + "<|audio_end|>"
            texts.append(EMBED_AUDIO_TEMPLATE.format(audio_placeholder=placeholder))

        encoded = tokenizer(
            texts, padding=True, truncation=True,
            max_length=2048, return_tensors="pt",
        )

        # Pad mel features to same length
        max_mel = max(m.shape[-1] for m in mel_list)
        mel_batch = torch.zeros(B, mel_list[0].shape[0], max_mel)
        for j, mel in enumerate(mel_list):
            mel_batch[j, :, :mel.shape[-1]] = mel

        max_audio = max(d.shape[-1] for d in dasheng_list)
        dasheng_batch = torch.zeros(B, 1, max_audio)
        for j, d in enumerate(dasheng_list):
            dasheng_batch[j, :, :d.shape[-1]] = d

        # Forward through model
        out = model(
            input_ids=encoded["input_ids"].to(device),
            attention_mask=encoded["attention_mask"].to(device),
            input_features=mel_batch.to(device),
            dasheng_audio=dasheng_batch.to(device),
        )
        emb = out["embedding"]
        if truncate_dim:
            emb = F.normalize(emb[:, :truncate_dim], p=2, dim=-1)
        all_embs.append(emb.cpu().float().numpy())

    return np.concatenate(all_embs, axis=0)


# ── Main evaluation function ─────────────────────────────────

_cached_task_data = None


def evaluate_scifact(model, tokenizer, whisper_fe, device,
                     truncate_dim=None, tokens_per_encoder=128):
    """Evaluate model on SciFact text + speech retrieval.

    Args:
        model: OmniEmbedModel (unwrapped, on device)
        tokenizer: model's tokenizer
        whisper_fe: WhisperFeatureExtractor
        device: torch device
        truncate_dim: optional MRL truncation dim
        tokens_per_encoder: audio tokens per encoder per chunk

    Returns:
        dict with text_ndcg10, speech_ndcg10, gap, and full metrics
    """
    global _cached_task_data
    model.eval()
    t_start = time.time()

    # Load data (cached after first call)
    if _cached_task_data is None:
        _cached_task_data = load_task_data("SciFact")
    queries, corpus, relevant_docs = _cached_task_data

    # Encode corpus (passages: raw text, no instruction, no chat wrap)
    corpus_ids = list(corpus.keys())
    corpus_texts = []
    for cid in corpus_ids:
        doc = corpus[cid]
        title = doc.get("title", "")
        text = doc.get("text", "")
        corpus_texts.append(
            f"{title} {text}".strip() if title else text
        )

    # Corpus: native Qwen3-Embedding passage format (raw text +
    # <|endoftext|> terminator).
    corpus_embs = _encode_texts(
        model, corpus_texts, device,
        is_query=False, truncate_dim=truncate_dim,
    )

    # Encode text queries — tokenize_text_native wraps with Qwen3's
    # "Instruct: {task}\nQuery: {q}" template when is_query=True.
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    text_query_embs = _encode_texts(
        model, query_texts, device,
        is_query=True, instruction=SCIFACT_TASK_INSTRUCTION,
        truncate_dim=truncate_dim,
    )

    # Text metrics
    text_scores = compute_retrieval_scores(
        text_query_embs, corpus_embs, query_ids, corpus_ids, relevant_docs,
    )

    # Encode speech queries
    speech_scores = None
    manifest_path = os.path.join(SPEECH_CACHE_DIR, "SciFact", "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)

        task_dir = os.path.join(SPEECH_CACHE_DIR, "SciFact")
        speech_paths, speech_query_ids = [], []
        for qid in query_ids:
            entry = manifest.get(qid, {})
            wav = entry.get("wav")
            if wav:
                full_path = os.path.join(task_dir, wav)
                if os.path.exists(full_path):
                    speech_paths.append(full_path)
                    speech_query_ids.append(qid)

        if speech_paths:
            speech_query_embs = _encode_speech(
                model, tokenizer, whisper_fe, speech_paths, device,
                tokens_per_encoder=tokens_per_encoder,
                truncate_dim=truncate_dim,
            )
            speech_scores = compute_retrieval_scores(
                speech_query_embs, corpus_embs, speech_query_ids,
                corpus_ids, relevant_docs,
            )

    elapsed = time.time() - t_start
    model.train()

    text_ndcg10 = text_scores["ndcg@10"]
    speech_ndcg10 = speech_scores["ndcg@10"] if speech_scores else 0
    gap = text_ndcg10 - speech_ndcg10

    return {
        "text_ndcg10": text_ndcg10,
        "speech_ndcg10": speech_ndcg10,
        "gap": gap,
        "text": text_scores,
        "speech": speech_scores,
        "elapsed_s": elapsed,
    }
