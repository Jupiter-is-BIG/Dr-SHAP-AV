#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibrate a visual distortion's severity parameter to hit a target LPIPS
(AlexNet) distance from the clean reference frames. LPIPS compares deep
features rather than raw pixel error, so unlike a pixel-wise metric (PSNR)
it isn't blind to *where*/*how* a distortion places its error - e.g. a few
severely-occluded blocks (BW) and uniform blur (GB) can look very different
even at "equal" average squared error.

LPIPS is measured after center-cropping both the clean and distorted frames
to `--crop-size` (default 88, matching VideoTransform's CenterCrop) - the
distortion itself is applied to the full raw frame first, exactly like the
real pipeline (transforms.VideoDistortion runs before CenterCrop). Skipping
this crop step understates spatially localized distortions like BW: blocks
placed outside the eventual crop region never reach the model, but would
still count toward the metric.

Takes a labels CSV in the same format the dataloader uses
(dataset_name,rel_path,input_length,_,text - see datamodule/av_dataset.py's
load_list), calibrates the parameter independently per video, and reports
the mean/sd of the parameter (and of the achieved LPIPS, for a sanity check)
across the whole file.

Requires the `lpips` package: pip install lpips
(downloads pretrained AlexNet weights on first use)

Usage:
    python Find_Distortion_Param_For_LPIPS.py --csv-path lrs3_test.csv --root-dir /data/lrs3 \
        --type BW --target-lpips 0.15
    python Find_Distortion_Param_For_LPIPS.py --csv-path lrs3_test.csv --root-dir /data/lrs3 \
        --type GB --target-lpips 0.15 --limit 50
"""
import argparse
import os
import random
import tempfile

import numpy as np
import torch

try:
    import lpips
except ImportError as e:
    raise ImportError("This script requires the `lpips` package: pip install lpips") from e

from datamodule.av_dataset import load_video
from datamodule.distortions import video_compression
from datamodule.video_distortion import (
    DISTORTION_TYPES,
    FRAME_DISTORTION_TYPES,
    convert_tensor_to_cv2_format,
    get_distortion_function,
    get_distortion_parameter,
)

# direction=+1: larger param -> worse. direction=-1: smaller param -> worse.
# hi=None means the upper bound depends on frame size (resolved at runtime).
SEARCH_SPACE = {
    "CC":   dict(direction=-1, lo=1e-3, hi=1.0,  integer=False, odd=False),
    "GNC":  dict(direction=+1, lo=1e-4, hi=1.0,  integer=False, odd=False),
    "BW":   dict(direction=+1, lo=0,    hi=2000, integer=True,  odd=False),
    "GB":   dict(direction=+1, lo=1,    hi=None, integer=True,  odd=True),
    "JPEG": dict(direction=+1, lo=1,    hi=None, integer=True,  odd=False),
    "VC":   dict(direction=+1, lo=0,    hi=51,   integer=True,  odd=False),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find the distortion parameter that hits a target LPIPS-AlexNet distance, "
                    "averaged over every video in a labels CSV."
    )
    parser.add_argument("--csv-path", type=str, required=True,
                         help="Labels CSV, same format as the dataloader "
                              "(dataset_name,rel_path,input_length,_,text).")
    parser.add_argument("--root-dir", type=str, required=True,
                         help="Root dir the CSV's rel_path is relative to (root_dir/dataset_name/rel_path).")
    parser.add_argument("--type", type=str, required=True, choices=DISTORTION_TYPES)
    parser.add_argument("--target-lpips", type=float, required=True,
                         help="Target LPIPS-AlexNet distance (0 = identical, larger = more different).")
    parser.add_argument("--num-frames", type=int, default=10,
                         help="Frames (evenly sampled) to average LPIPS over, per video.")
    parser.add_argument("--crop-size", type=int, default=88,
                         help="Center-crop size, matching VideoTransform's CenterCrop (default 88).")
    parser.add_argument("--iterations", type=int, default=20, help="Bisection iterations, per video.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N rows of the CSV (for a quick pilot run).")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_hi(dist_type, frame_shape):
    hi = SEARCH_SPACE[dist_type]["hi"]
    if hi is not None:
        return hi
    min_dim = min(frame_shape[-2], frame_shape[-1])
    if dist_type == "GB":
        odd_max = min_dim - 1 if (min_dim - 1) % 2 == 1 else min_dim - 2
        return max(odd_max, 1)
    if dist_type == "JPEG":
        return max(min_dim, 1)
    raise ValueError(dist_type)


def snap(dist_type, value, frame_shape):
    cfg = SEARCH_SPACE[dist_type]
    lo, hi = cfg["lo"], resolve_hi(dist_type, frame_shape)
    value = max(lo, min(hi, value))
    if cfg["integer"]:
        value = int(round(value))
        if cfg["odd"] and value % 2 == 0:
            value += 1
        value = max(int(lo), min(int(hi), value))
    return value


def center_crop(frame, size):
    """frame: H x W x C numpy array. Crops the center `size` x `size` region."""
    h, w = frame.shape[:2]
    size = min(size, h, w)
    top, left = (h - size) // 2, (w - size) // 2
    return frame[top:top + size, left:left + size]


def bisection_search(dist_type, evaluate, target, iterations, frame_shape):
    """Find the distortion parameter whose evaluate(param) (an LPIPS distance,
    higher = more severe) is closest to `target`."""
    cfg = SEARCH_SPACE[dist_type]
    lo, hi = cfg["lo"], resolve_hi(dist_type, frame_shape)
    clean_end, worst_end = (lo, hi) if cfg["direction"] == +1 else (hi, lo)

    best = {"param": None, "value": None, "gap": float("inf")}

    def record(param, val):
        gap = abs(val - target)
        if gap < best["gap"]:
            best.update(param=param, value=val, gap=gap)

    val_clean = evaluate(snap(dist_type, clean_end, frame_shape))
    val_worst = evaluate(snap(dist_type, worst_end, frame_shape))
    record(snap(dist_type, clean_end, frame_shape), val_clean)
    record(snap(dist_type, worst_end, frame_shape), val_worst)

    if not (val_clean <= target <= val_worst):
        print(
            f"WARNING: target LPIPS {target:.4f} is outside the achievable range "
            f"[{val_clean:.4f}, {val_worst:.4f}] for {dist_type} on this clip; "
            f"returning the closest achievable value instead."
        )

    a, b = clean_end, worst_end  # a: "clean" side, b: "worst" side
    for _ in range(iterations):
        mid = (a + b) / 2.0
        mid_snapped = snap(dist_type, mid, frame_shape)
        val = evaluate(mid_snapped)
        record(mid_snapped, val)

        if val < target:  # not severe enough yet, move toward worst
            a = mid
        else:
            b = mid
        if cfg["integer"] and abs(snap(dist_type, a, frame_shape) - snap(dist_type, b, frame_shape)) <= 1:
            break

    return best["param"], best["value"]


def load_video_paths(csv_path, root_dir):
    paths = []
    with open(csv_path) as f:
        for line in f.read().splitlines():
            if not line.strip():
                continue
            dataset_name, rel_path, _input_length, _unused, _text = line.split(",")
            paths.append(os.path.join(root_dir, dataset_name, rel_path))
    return paths


def bgr_to_lpips_tensor(frame_bgr):
    """H x W x C (BGR, uint8) -> 1 x 3 x H x W tensor in [-1, 1], RGB (as LPIPS expects)."""
    frame_rgb = frame_bgr[..., ::-1].astype(np.float32)
    tensor = torch.from_numpy(frame_rgb.copy()).permute(2, 0, 1).unsqueeze(0)
    return tensor / 127.5 - 1.0


def lpips_distance(loss_fn, clean_bgr_crop, distorted_bgr_crop):
    with torch.no_grad():
        d = loss_fn(bgr_to_lpips_tensor(clean_bgr_crop), bgr_to_lpips_tensor(distorted_bgr_crop))
    return d.item()


def evaluate_frame_type(loss_fn, dist_type, frame_clips_bgr, crop_size, param):
    func = get_distortion_function(dist_type)
    scores = []
    for clean_bgr in frame_clips_bgr:
        distorted_bgr = func(clean_bgr.copy(), param)
        scores.append(lpips_distance(
            loss_fn, center_crop(clean_bgr, crop_size), center_crop(distorted_bgr, crop_size),
        ))
    return float(np.mean(scores))


def evaluate_vc(loss_fn, video_path, frame_idxs, crop_size, param):
    fd, out_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        video_compression(video_path, out_path, param)
        distorted_video = load_video(out_path)
        clean_video = load_video(video_path)
        scores = []
        for i in frame_idxs:
            j = min(i, distorted_video.shape[0] - 1)
            clean_bgr = convert_tensor_to_cv2_format(clean_video[i:i + 1])[0]
            distorted_bgr = convert_tensor_to_cv2_format(distorted_video[j:j + 1])[0]
            scores.append(lpips_distance(
                loss_fn, center_crop(clean_bgr, crop_size), center_crop(distorted_bgr, crop_size),
            ))
        return float(np.mean(scores))
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)


def calibrate_one(video_path, dist_type, target_lpips, num_frames, crop_size, iterations, loss_fn):
    video = load_video(video_path)
    frame_shape = video.shape
    t = video.shape[0]
    idxs = np.linspace(0, t - 1, min(num_frames, t)).astype(int)

    if dist_type == "VC":
        def evaluate(param):
            return evaluate_vc(loss_fn, video_path, idxs, crop_size, param)
    else:
        frame_clips_bgr = [convert_tensor_to_cv2_format(video[i:i + 1])[0] for i in idxs]

        def evaluate(param):
            return evaluate_frame_type(loss_fn, dist_type, frame_clips_bgr, crop_size, param)

    return bisection_search(dist_type, evaluate, target_lpips, iterations, frame_shape)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    loss_fn = lpips.LPIPS(net="alex")
    loss_fn.eval()

    video_paths = load_video_paths(args.csv_path, args.root_dir)
    if args.limit is not None:
        video_paths = video_paths[:args.limit]

    params, achieved = [], []
    for i, video_path in enumerate(video_paths, start=1):
        try:
            best_param, best_lpips = calibrate_one(
                video_path, args.type, args.target_lpips, args.num_frames,
                args.crop_size, args.iterations, loss_fn,
            )
        except Exception as e:
            print(f"[{i}/{len(video_paths)}] WARNING: skipping {video_path} ({e})")
            continue

        params.append(best_param)
        achieved.append(best_lpips)
        print(f"[{i}/{len(video_paths)}] {os.path.basename(video_path)}: "
              f"param={best_param}  LPIPS={best_lpips:.4f}")

    params = np.array(params, dtype=np.float64)
    achieved = np.array(achieved, dtype=np.float64)

    print(f"\n{len(params)}/{len(video_paths)} videos processed")
    print(f"type={args.type}  target LPIPS(AlexNet)={args.target_lpips:.4f}  (crop={args.crop_size})")
    if len(params) == 0:
        print("No videos succeeded - nothing to report.")
        return
    print(f"param:          mean={params.mean():.4f}  sd={params.std():.4f}")
    print(f"achieved LPIPS: mean={achieved.mean():.4f}  sd={achieved.std():.4f}")

    if args.type in FRAME_DISTORTION_TYPES:
        presets = {lvl: get_distortion_parameter(args.type, lvl) for lvl in range(1, 6)}
        closest_lvl = min(presets, key=lambda lvl: abs(presets[lvl] - params.mean()))
        print(f"   closest preset severity level to the mean param: "
              f"{closest_lvl} (preset param={presets[closest_lvl]})")


if __name__ == "__main__":
    main()
