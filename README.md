<div align="center">

### Accepted at **EMNLP 2026 Findings 🏅**
# Omni-Embed-Mini: Binding Modalities Without Forgetting via Dense Distillation

#### [Mohammed Irfan Kurpath](https://scholar.google.com/citations?user=GJp0keYAAAAJ&hl=en), [Jaseel Muhammad Kaithakkodan](https://scholar.google.com/citations?user=-sqbA5oAAAAJ&hl=en), [Sahal Shaji Mullappilly](https://scholar.google.com/citations?user=LJWxVpUAAAAJ&hl=en), [Ivan Laptev](https://scholar.google.com/citations?user=-9ifK0cAAAAJ&hl=en), and [Hisham Cholakkal](https://scholar.google.com/citations?hl=en&user=bZ3YBRcAAAAJ)

#### **Mohamed Bin Zayed University of Artificial Intelligence (MBZUAI), UAE**
[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://omniembed.cvmbzuai.com)
[![Paper](https://img.shields.io/badge/arXiv-Paper-B31B1B.svg)](https://arxiv.org/abs/2610.02148)
[![Model 0.9B](https://img.shields.io/badge/🤗%20Model-Omni--Embed--Mini--0.9B-F9D371)](https://huggingface.co/MBZUAI/Omni-Embed-Mini-0.9B)
[![Model 2.3B](https://img.shields.io/badge/🤗%20Model-Omni--Embed--Mini--2.3B-F9D371)](https://huggingface.co/MBZUAI/Omni-Embed-Mini-2.3B)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-Omni--Sets-F9D371)](https://huggingface.co/datasets/MBZUAI/Omni-Sets)

</div>

Omni-Embed-Mini maps text, speech, audio, images, video and visually-rich
documents into one shared embedding space **without updating any text-side
parameter**. Each media sample is paired with a dense caption, and the
training target is the frozen backbone's own embedding of that caption, so
lightweight projectors plus LoRA adapters on the media encoders are enough to
align every modality.

## Installation

```bash
git clone https://github.com/k-m-irfan/Omni-Embed-Mini.git
cd Omni-Embed-Mini

conda create -n omni-embed-mini python=3.11 -y
conda activate omni-embed-mini

# PyTorch 2.7.1 for your accelerator
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/rocm6.3   # AMD
# pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126 # NVIDIA

pip install -r requirements.txt
```

One environment covers both training and evaluation.

## Inference

Both released models load with `transformers` and take OpenAI-style multimodal
messages (text, audio, image, video, document pages, or any combination):

```python
import torch
from transformers import AutoModel, AutoProcessor

REPO = "MBZUAI/Omni-Embed-Mini-0.9B"     # or "MBZUAI/Omni-Embed-Mini-2.3B"

model = AutoModel.from_pretrained(
    REPO, trust_remote_code=True, dtype=torch.bfloat16,
).cuda().eval()
processor = AutoProcessor.from_pretrained(REPO, trust_remote_code=True)

# Document side: any media, optionally with text.
messages = [{"role": "user", "content": [
    {"type": "audio", "audio": "path/to/clip.wav"},
    {"type": "text",  "text":  "rain on a tin roof at night"},
]}]
inputs = processor.apply_chat_template(
    messages, role="passage", tokenize=True, return_tensors="pt",
).to("cuda")
with torch.no_grad():
    doc = model(**inputs).pooler_output          # L2-normalised embedding

# Query side. text_recipe="chat" puts a text query in the same subspace as
# media documents; omit it for pure text-to-text retrieval.
query = processor.apply_chat_template(
    [{"role": "user", "content": "rain on a tin roof at night"}],
    role="query", tokenize=True, return_tensors="pt", text_recipe="chat",
).to("cuda")
with torch.no_grad():
    q = model(**query).pooler_output

print("cosine:", float(q @ doc.T))
```

Pass `truncate_dim=N` to the model call for Matryoshka embeddings
(128 / 256 / 512 / 1024, plus 2048 for the 2.3B). Other media use
`{"type": "image", "image": ...}` and `{"type": "video", "video": ...}`.

## Training

Training data is [MBZUAI/Omni-Sets](https://huggingface.co/datasets/MBZUAI/Omni-Sets)
(downloaded automatically on first run). The released models were trained on
8 x 64 GB GPUs:

```bash
bash scripts/train.sh configs/omni_embed_mini_0.9b.yaml --text_mining --media_mining
bash scripts/train.sh configs/omni_embed_mini_2.3b.yaml --text_mining --media_mining
```

- Checkpoints go to `checkpoints/omni-embed-mini-<size>-tm-mm/` (`latest/`, `best_step_*`, and one `epoch_N/` per epoch). Re-running the same command resumes from `latest/`; add `--fresh` to start over.
- `NUM_GPUS=4 bash scripts/train.sh ...` changes the GPU count.
- Any config value can be overridden, e.g. `--override training.epochs=5`.
- The released weights are the epoch-4 (0.9B) and epoch-8 (2.3B) snapshots.

Export a checkpoint to a Hugging Face model folder:

```bash
python scripts/export_hf.py \
    --checkpoint checkpoints/omni-embed-mini-0.9b-tm-mm/epoch_4 \
    --config     configs/omni_embed_mini_0.9b.yaml \
    --output     exported/omni-embed-mini-0.9b
```

## Evaluation

```bash
# all four benchmarks (MTEB-v2, MAEB, MMEB-V2, ViDoRe-V3), one task per GPU
bash evaluations/eval.sh --model MBZUAI/Omni-Embed-Mini-0.9B

# one benchmark, or specific tasks; a local export folder also works
bash evaluations/eval.sh --model MBZUAI/Omni-Embed-Mini-2.3B --bench maeb
bash evaluations/eval.sh --model exported/omni-embed-mini-0.9b-4 --tasks SciFact,VOC2007
```

MMEB-V2 images and video frames (~25 GB) must be downloaded once:

```bash
python evaluations/mmeb/prepare_data.py     # set MMEB_DATA_DIR to change the location
```

Results are cached in `evaluations/results/` and summarised per task and per
modality at the end of each run (`python evaluations/summarize.py --model <model>`
reprints them). Set `OMNI_TRUNCATE_DIM=256` (or 128/512/...) to evaluate a
Matryoshka-truncated embedding.

| Benchmark | Modality | Tasks | Metric |
|---|---|---|---|
| MTEB-v2 (BEIR subset) | text | 8 | nDCG@10 |
| MAEB (English subset) | speech / audio | 12 / 10 | task main score |
| MMEB-V2 | image / video | 10 / 6 | hit@1 |
| ViDoRe-V3 (English) | documents | 7 | nDCG@10 |

## Citation

```bibtex
@inproceedings{kurpath-etal-2026-omni-embed-mini,
    title = "{O}mni-{E}mbed-{M}ini: Binding Modalities Without Forgetting via Dense Distillation",
    author = "Kurpath, Mohammed Irfan and Kaithakkodan, Jaseel Muhammad and Mullappilly, Sahal Shaji and Laptev, Ivan and Cholakkal, Hisham",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2026",
    year = "2026",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/2610.02148"
}
```
