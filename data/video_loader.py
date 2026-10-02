"""Video frame loading for the dataset."""

import os


def load_video_frames(media, max_frames=196, image_size=224):
    """Load video frames from Omni-Sets media field.

    The media field may be a dict with 'path' and/or 'bytes'.

    Args:
        media: dict with video data
        max_frames: int — max frames to extract
        image_size: int — output spatial resolution

    Returns:
        frames: (max_frames, 3, image_size, image_size)
        frame_indices: (max_frames,)
    """
    from model.video_utils import smart_sample_frames

    video_src = None

    if isinstance(media, dict):
        path = media.get("path")
        if path and os.path.exists(path):
            video_src = path
        elif media.get("bytes"):
            video_src = media["bytes"]
    elif isinstance(media, (str, bytes)):
        video_src = media

    if video_src is None:
        raise ValueError(
            f"load_video_frames: could not resolve video_src from media "
            f"(type={type(media).__name__}, "
            f"keys={list(media.keys()) if isinstance(media, dict) else 'n/a'})"
        )

    return smart_sample_frames(video_src, max_frames, image_size)
