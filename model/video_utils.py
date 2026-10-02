"""
Activity-based smart video frame selection using pyav.

Algorithm:
    1. Open video (file path or BytesIO), determine frame range
    2. Fast path: if segment <= max_frames, use all frames
    3. Decode all frames at low-res grayscale for activity scoring
    4. Compute per-transition diffs: |gray[i+1] - gray[i]|.mean() for each i
    5. Bucket diff scores into max_frames equal-width buckets
    6. From each bucket, select the frame with the highest diff (most activity)
    7. Load those frames at full resolution, apply transforms
    8. Return (frames_tensor, real_frame_indices)
"""

import io

import av
import numpy as np
import torch
from torchvision import transforms


_VIDEO_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def _build_transform(image_size=224):
    if image_size == 224:
        return _VIDEO_TRANSFORM
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def _open_container(video_src):
    if isinstance(video_src, bytes):
        return av.open(io.BytesIO(video_src))
    if isinstance(video_src, io.BytesIO):
        video_src.seek(0)
        return av.open(video_src)
    return av.open(video_src)


def _decode_all_frames(container, start_frame=0, end_frame=None):
    stream = container.streams.video[0]
    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i < start_frame:
            continue
        if end_frame is not None and i > end_frame:
            break
        frames.append((i, frame.to_ndarray(format="rgb24")))
    return frames


def smart_sample_frames(
    video_src,
    max_frames=96,
    image_size=224,
    start_sec=None,
    end_sec=None,
):
    container = _open_container(video_src)
    stream = container.streams.video[0]
    fps = float(stream.average_rate) if stream.average_rate else 30.0
    total_frames = stream.frames
    if total_frames == 0:
        if stream.duration and stream.time_base:
            total_frames = int(float(stream.duration * stream.time_base) * fps)
        else:
            total_frames = 0

    if start_sec is not None and end_sec is not None:
        start_frame = max(0, int(start_sec * fps))
        end_frame = int(end_sec * fps)
        if total_frames > 0:
            end_frame = min(total_frames - 1, end_frame)
        if end_frame <= start_frame:
            end_frame = start_frame + int(fps * 2)
    else:
        start_frame = 0
        end_frame = total_frames - 1 if total_frames > 0 else None

    decoded = _decode_all_frames(container, start_frame, end_frame)
    container.close()

    if not decoded:
        raise ValueError("smart_sample_frames: no frames decoded from video")

    segment_len = len(decoded)
    all_real_indices = [idx for idx, _ in decoded]
    all_frames_np = [frame for _, frame in decoded]

    if segment_len <= max_frames:
        selected_local = list(range(segment_len))
        while len(selected_local) < max_frames:
            selected_local.append(selected_local[-1])
        selected_local = selected_local[:max_frames]
        return _transform_frames(
            [all_frames_np[i] for i in selected_local],
            [all_real_indices[i] for i in selected_local],
            max_frames, image_size,
        )

    low = np.stack([
        np.array(
            transforms.functional.resize(
                transforms.functional.to_pil_image(f), [16, 16],
            )
        ) for f in all_frames_np
    ])

    gray = (
        0.299 * low[..., 0].astype(np.float32)
        + 0.587 * low[..., 1].astype(np.float32)
        + 0.114 * low[..., 2].astype(np.float32)
    )

    diffs = np.abs(gray[1:] - gray[:-1]).mean(axis=(1, 2))
    diffs = np.concatenate([[0.0], diffs])

    N = segment_len
    bucket_edges = np.linspace(0, N, max_frames + 1, dtype=int)
    selected_local = []

    for b in range(max_frames):
        bstart = bucket_edges[b]
        bend = bucket_edges[b + 1]
        if bstart >= bend:
            selected_local.append(selected_local[-1] if selected_local else 0)
        else:
            best = bstart + np.argmax(diffs[bstart:bend])
            selected_local.append(int(best))

    return _transform_frames(
        [all_frames_np[i] for i in selected_local],
        [all_real_indices[i] for i in selected_local],
        max_frames, image_size,
    )


def _transform_frames(frames_np, real_indices, max_frames, image_size):
    transform = _build_transform(image_size)
    processed = [transform(f) for f in frames_np]

    while len(processed) < max_frames:
        processed.append(processed[-1])
    processed = processed[:max_frames]

    frames_tensor = torch.stack(processed)
    index_tensor = torch.tensor(real_indices[:max_frames], dtype=torch.long)

    return frames_tensor, index_tensor
