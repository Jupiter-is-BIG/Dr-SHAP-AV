#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frame-level visual distortions applied to lip-video clips, analogous to the
acoustic noise injection in transforms.AddNoise. distortion_vid() is the
building block used by the VideoDistortion transform during dataloading/
evaluation (see transforms.py).
"""
import os
import random
import tempfile

import torch

from .av_dataset import load_video
from .distortions import (
    block_wise,
    color_contrast,
    gaussian_blur,
    gaussian_noise_color,
    jpeg_compression,
    video_compression,
)

# VC (video compression) re-encodes the whole clip via ffmpeg rather than
# transforming individual frames, so it is handled separately below. It is
# reachable only by calling distortion_vid() directly with vid_in_path set
# (e.g. from an ad-hoc script) — VideoDistortion/transforms.py never selects
# it, since a per-sample ffmpeg subprocess is too slow inside a DataLoader.
#
# CS (color saturation) is intentionally not included: see the note in
# distortions.py — it's nullified by the grayscale conversion downstream.
DISTORTION_TYPES = ["CC", "BW", "GNC", "GB", "JPEG", "VC"]
FRAME_DISTORTION_TYPES = ["CC", "BW", "GNC", "GB", "JPEG"]

_PARAM_DICT = {
    "CC": [0.52, 0.42, 0.32, 0.22, 0.12],      # smaller, worse (factor of contrast change)
    "BW": [64, 128, 256, 512, 1024],                  # larger, worse (num of null blocks)
    "GNC": [0.008, 0.016, 0.032, 0.064, 0.128],     # larger, worse (variance of Gaussian noise)
    "GB": [7, 11, 19, 31, 51],                       # larger, worse (kernel size for sd for Gaussian blur)
    "JPEG": [2, 5, 8, 11, 14],                     # larger, worse (image reduce factor for downsample compression)
    "VC": [35, 40, 45, 50, 55],                     # larger CRF, worse
}

_FUNC_DICT = {
    "CC": color_contrast,
    "BW": block_wise,
    "GNC": gaussian_noise_color,
    "GB": gaussian_blur,
    "JPEG": jpeg_compression,
}

_LOG_NAMES = {
    "CC": "color contrast change",
    "BW": "local block-wise",
    "GNC": "white Gaussian noise in color components",
    "GB": "Gaussian blur",
    "JPEG": "JPEG compression",
    "VC": "video compression (H.264 CRF)",
}


def get_distortion_parameter(dist_type, level):
    # level starts from 1, list starts from 0.
    return _PARAM_DICT[dist_type][level - 1]


def get_distortion_function(dist_type):
    return _FUNC_DICT[dist_type]


def convert_tensor_to_cv2_format(vid_tensor):
    """T x C x H x W (RGB) tensor -> list of H x W x C (BGR) numpy arrays."""
    frame_list = []
    for frame_tensor in vid_tensor:
        frame = frame_tensor.permute(1, 2, 0)
        frame_np = frame.cpu().numpy()
        frame_np = frame_np[..., ::-1]
        frame_list.append(frame_np)
    return frame_list


def convert_cv2_to_tensor_format(frame_list):
    """list of H x W x C (BGR) numpy arrays -> T x C x H x W (RGB) tensor."""
    tensor_list = []
    for frame in frame_list:
        frame = frame[..., ::-1]
        tensor_frame = torch.from_numpy(frame.copy()).permute(2, 0, 1).float()
        tensor_list.append(tensor_frame)
    return torch.stack(tensor_list)


def cut_or_pad_frames(vid_tensor, num_frames):
    """Ffmpeg re-encoding can drop/duplicate a frame or two; realign length."""
    if vid_tensor.size(0) == num_frames:
        return vid_tensor
    if vid_tensor.size(0) > num_frames:
        return vid_tensor[:num_frames]
    pad = num_frames - vid_tensor.size(0)
    last_frame = vid_tensor[-1:].expand(pad, *vid_tensor.shape[1:])
    return torch.cat([vid_tensor, last_frame], dim=0)


def _apply_video_compression(vid_in_tensor, vid_in_path, vid_out_path, crf):
    """
    VC re-encodes the whole clip via ffmpeg, so it needs a real file on disk.
    Falls back to a temp file when no explicit vid_out_path is given.
    """
    if vid_in_path is None:
        raise ValueError(
            "dist_type='VC' requires vid_in_path: video compression re-encodes "
            "the source file via ffmpeg and cannot run on an in-memory tensor alone."
        )

    cleanup_path = None
    out_path = vid_out_path
    if out_path is None:
        fd, out_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        cleanup_path = out_path

    try:
        video_compression(vid_in_path, out_path, crf)
        output_tensor = load_video(out_path)
        return cut_or_pad_frames(output_tensor, vid_in_tensor.shape[0])
    finally:
        if cleanup_path is not None and os.path.exists(cleanup_path):
            os.remove(cleanup_path)


def distortion_vid(vid_in_tensor, vid_in_path=None, dist_type="random", dist_level="random"):
    """
    Apply a visual distortion to a T x C x H x W (RGB) video tensor and
    return the distorted tensor. `vid_in_path` is only required when
    `dist_type == "VC"`; every other distortion type operates purely in
    memory, frame by frame.
    """
    if dist_type == "random":
        dist_type = random.choice(FRAME_DISTORTION_TYPES)

    dist_level = random.randint(1, 5) if dist_level == "random" else int(dist_level)
    dist_param = get_distortion_parameter(dist_type, dist_level)
    print(f"Apply level-{dist_level} {_LOG_NAMES[dist_type]} distortion...")

    if dist_type == "VC":
        return _apply_video_compression(vid_in_tensor, vid_in_path, None, dist_param)

    dist_function = get_distortion_function(dist_type)
    input_list = convert_tensor_to_cv2_format(vid_in_tensor)
    output_list = [dist_function(frame, dist_param) for frame in input_list]
    output_tensor = convert_cv2_to_tensor_format(output_list)
    assert vid_in_tensor.shape == output_tensor.shape

    return output_tensor
