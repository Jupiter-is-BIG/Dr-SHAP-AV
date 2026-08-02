import os
import random

import sentencepiece
import torch
import torchaudio
import torchvision

from .video_distortion import distortion_vid, FRAME_DISTORTION_TYPES

class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)


class AdaptiveTimeMask(torch.nn.Module):
    def __init__(self, window, stride):
        super().__init__()
        self.window = window
        self.stride = stride

    def forward(self, x):
        # x: [T, ...]
        cloned = x.clone()
        length = cloned.size(0)
        n_mask = int((length + self.stride - 0.1) // self.stride)
        ts = torch.randint(0, self.window, size=(n_mask, 2))
        for t, t_end in ts:
            if length - t <= 0:
                continue
            t_start = random.randrange(0, length - t)
            if t_start == t_start + t:
                continue
            t_end += t_start
            cloned[t_start:t_end] = 0
        return cloned


class AddNoise(torch.nn.Module):
    def __init__(
        self,
        noise_path,
        snr_target=None,
    ):
        super().__init__()
        self.snr_levels = [snr_target] if snr_target else [-5, 0, 5, 10, 15, 20, 999999]
        self.noise, sample_rate = torchaudio.load(noise_path)
        assert sample_rate == 16000

    def forward(self, speech):
        # speech: T x 1
        # return: T x 1
        speech = speech.t()
        start_idx = random.randint(0, self.noise.shape[1] - speech.shape[1])
        noise_segment = self.noise[:, start_idx : start_idx + speech.shape[1]]
        snr_level = torch.tensor([random.choice(self.snr_levels)])
        noisy_speech = torchaudio.functional.add_noise(speech, noise_segment, snr_level)
        return noisy_speech.t()


class VideoDistortion(torch.nn.Module):
    """
    Applies a fixed-type/fixed-severity visual corruption (color, blur,
    block-wise, or compression-style) to a raw T x C x H x W clip, mirroring
    AddNoise's role for audio: it lets us sweep visual degradation the same
    way --decode-snr-target sweeps acoustic degradation.
    """
    def __init__(self, dist_type, dist_level=3):
        super().__init__()
        assert dist_type in FRAME_DISTORTION_TYPES + ["random"], (
            f"dist_type must be one of {FRAME_DISTORTION_TYPES + ['random']}, got {dist_type!r}. "
            "'VC' (video compression) is not supported here since a per-sample ffmpeg subprocess "
            "is too slow inside a DataLoader; call video_distortion.distortion_vid(..., dist_type='VC', "
            "vid_in_path=...) directly instead."
        )
        self.dist_type = dist_type
        self.dist_level = dist_level

    def forward(self, video):
        # video: T x C x H x W, RGB, pre-normalization (0-255 range).
        return distortion_vid(video, dist_type=self.dist_type, dist_level=self.dist_level)


class VideoTransform:
    def __init__(self, subset, dist_type=None, dist_level=3):
        distortion = (
            [VideoDistortion(dist_type, dist_level)]
            if dist_type not in (None, "none")
            else []
        )
        if subset == "train":
            self.video_pipeline = torch.nn.Sequential(
                *distortion,
                FunctionalModule(lambda x: x / 255.0),
                torchvision.transforms.RandomCrop(88),
                torchvision.transforms.Grayscale(),
                AdaptiveTimeMask(10, 25),
                torchvision.transforms.Normalize(0.421, 0.165),
            )
        elif subset == "val" or subset == "test":
            self.video_pipeline = torch.nn.Sequential(
                *distortion,
                FunctionalModule(lambda x: x / 255.0),
                torchvision.transforms.CenterCrop(88),
                torchvision.transforms.Grayscale(),
                torchvision.transforms.Normalize(0.421, 0.165),
            )

    def __call__(self, sample):
        # sample: T x C x H x W
        # rtype: T x 1 x H x W
        return self.video_pipeline(sample)


class AudioTransform:
    def __init__(self, subset, noise_type, snr_target=None):
        match noise_type:
            case "babble":
                noise_filename = "babble_noise.wav"
            case "music":
                noise_filename = "music-hd-0001.wav"
            case "speech":
                noise_filename = "speech-us-gov-0002.wav"
            case "sound":
                noise_filename = "noise-sound-bible-0014.wav"

        noise_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), noise_filename
                    )

        if subset == "train":
            self.audio_pipeline = torch.nn.Sequential(
                AdaptiveTimeMask(6400, 16000),
                AddNoise(noise_path = noise_path),
                FunctionalModule(
                    lambda x: torch.nn.functional.layer_norm(x, x.shape, eps=1e-8)
                ),
            )
        elif subset == "val" or subset == "test":
            self.audio_pipeline = torch.nn.Sequential(
                AddNoise(noise_path=noise_path, snr_target=snr_target)
                if snr_target is not None
                else FunctionalModule(lambda x: x),
                FunctionalModule(
                    lambda x: torch.nn.functional.layer_norm(x, x.shape, eps=1e-8)
                ),
            )

    def __call__(self, sample):
        # sample: T x 1
        # rtype: T x 1
        return self.audio_pipeline(sample)