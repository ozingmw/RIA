from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class LiteKWSPreprocessor:
    """Lightweight log-mel front-end for on-device KWS."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 24,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int | None = None,
        target_frames: int = 200,
        f_min: float = 50.0,
        f_max: float | None = None,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.n_mels = int(n_mels)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length) if win_length is not None else int(n_fft)
        self.target_frames = int(target_frames)
        self.f_min = float(f_min)
        self.f_max = float(f_max) if f_max is not None else self.sample_rate / 2.0

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
            center=False,  # causal-friendly
            power=2.0,
        )

    @property
    def target_samples(self) -> int:
        return (self.target_frames - 1) * self.hop_length + self.win_length

    def __call__(self, audio_np: np.ndarray) -> torch.Tensor:
        if audio_np.dtype != np.int16:
            audio_np = audio_np.astype(np.int16, copy=False)

        # int16 -> float32 in [-1, 1)
        x = torch.from_numpy(audio_np.astype(np.float32) / 32768.0).unsqueeze(0)

        mel = self.mel_transform(x)  # (1, n_mels, T)
        mel = torch.log1p(mel)

        # Keep the most recent frames for streaming-style windows.
        if mel.shape[-1] < self.target_frames:
            mel = F.pad(mel, (0, self.target_frames - mel.shape[-1]))
        else:
            mel = mel[..., -self.target_frames :]

        return mel.unsqueeze(1)  # (1, 1, n_mels, target_frames)


class CausalConv2d(nn.Module):
    """Causal convolution over time (last dimension)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] = (3, 3),
        stride: tuple[int, int] = (1, 1),
        dilation: tuple[int, int] = (1, 1),
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.groups = groups
        self.bias = bias

        self.pad_freq = (kernel_size[0] - 1) // 2 * dilation[0]
        self.pad_time = (kernel_size[1] - 1) * dilation[1]

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad only on the left in time to keep causality.
        x = F.pad(x, (self.pad_time, 0, self.pad_freq, self.pad_freq))
        return self.conv(x)


class DSConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] = (3, 3),
        stride: tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__()
        self.depthwise = CausalConv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class TinyDSCNN(nn.Module):
    """Tiny DS-CNN optimized for low-latency on-device KWS."""

    def __init__(
        self,
        num_classes: int = 2,
        channels: tuple[int, int, int, int] = (8, 16, 24, 32),
        dropout: float = 0.1,
        use_softmax: bool = False,
    ) -> None:
        super().__init__()
        c0, c1, c2, c3 = channels

        self.front = nn.Sequential(
            CausalConv2d(1, c0, kernel_size=(3, 3), stride=(1, 1), bias=False),
            nn.BatchNorm2d(c0),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.Sequential(
            DSConvBlock(c0, c1, stride=(2, 2)),
            DSConvBlock(c1, c2, stride=(1, 2)),
            DSConvBlock(c2, c3, stride=(1, 2)),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c3, num_classes),
        )
        self.use_softmax = use_softmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.front(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = self.classifier(x)
        if self.use_softmax:
            x = F.softmax(x, dim=1)
        return x


@dataclass
class SlidingWindowConfig:
    sample_rate: int = 16000
    window_ms: int = 2000
    hop_ms: int = 100

    @property
    def window_samples(self) -> int:
        return int(self.sample_rate * self.window_ms / 1000)

    @property
    def hop_samples(self) -> int:
        return int(self.sample_rate * self.hop_ms / 1000)


class SlidingWindowKWS:
    """Simple sliding-window inference wrapper for streaming audio."""

    def __init__(
        self,
        model: nn.Module,
        preprocessor: LiteKWSPreprocessor,
        config: SlidingWindowConfig | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.model = model.eval()
        self.preprocessor = preprocessor
        self.config = config or SlidingWindowConfig(
            sample_rate=preprocessor.sample_rate
        )
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self.model.to(self.device)
        self.buffer = np.zeros(0, dtype=np.int16)

    def reset(self) -> None:
        self.buffer = np.zeros(0, dtype=np.int16)

    def accept_audio(self, chunk: np.ndarray) -> list[float]:
        if chunk is None or len(chunk) == 0:
            return []

        if chunk.dtype != np.int16:
            chunk = chunk.astype(np.int16, copy=False)

        self.buffer = np.concatenate([self.buffer, chunk])
        results: list[float] = []

        while self.buffer.shape[0] >= self.config.window_samples:
            window = self.buffer[: self.config.window_samples]
            self.buffer = self.buffer[self.config.hop_samples :]

            mel = self.preprocessor(window).to(self.device)
            with torch.no_grad():
                logits = self.model(mel)
                if logits.shape[1] == 1:
                    prob = torch.sigmoid(logits)[0, 0].item()
                else:
                    prob = torch.softmax(logits, dim=1)[0, 1].item()
            results.append(prob)

        return results
