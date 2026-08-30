<div align="center">

# Omni-Embed-Mini

### Binding Modalities Without Forgetting via Dense Distillation

**Code and weights coming soon.**

</div>

---

Omni-Embed-Mini is a **0.9B-parameter omni-modal retrieval model** that maps text,
speech, audio, images, video and visually-rich documents into a single shared
cosine embedding space, *without updating a single text-side parameter*.

The key insight is that the teacher signal requires no separate model: each media
sample is paired with a dense cascaded caption, and the teacher target is simply
the frozen backbone's own embedding of that caption. Because teacher and student
share the same backbone weights, they inhabit byte-identical geometry, so
lightweight projectors plus phased LoRA adapters on the modality encoders suffice
for alignment.