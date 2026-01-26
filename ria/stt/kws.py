import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class KWSPreprocessor:
    """
    Keyword Spotting용 오디오 전처리기
    입력: int16 numpy audio (16kHz, mono)
    출력: torch tensor (1, 1, 40, 100) <- (batch, channel, mel_bins, time_steps)
    """

    # Mel Spectrogram 변환기 (한 번만 생성)
    # 소리를 이미지로 바꿈
    def __init__(self):
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            n_mels=40
        )


    def __call__(self, audio_np):
        # int16 → float32 정규화
        audio_np = audio_np.astype("float32") / 32768.0

        # numpy → torch (batch dim 추가)
        x = torch.from_numpy(audio_np).unsqueeze(0)  # (1, N)

        # Mel Spectrogram
        mel = self.mel_transform(x)                  # (1, 40, T)
        mel = torch.log1p(mel)

        # DS-CNN 기준 time dimension 고정
        if mel.shape[-1] < 100:
            mel = F.pad(mel, (0, 100 - mel.shape[-1]))
        else:
            mel = mel[..., :100]

        # channel dim 추가
        mel = mel.unsqueeze(1)                       # (1, 1, 40, 100)

        return mel



# DS-CNN 모델
class DSCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Depthwise Convolution
            nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=(3,3),
                padding=1,
                groups=1
            ),
            nn.BatchNorm2d(1),
            nn.ReLU(),

            # Pointwise Convolution (1x1)
            nn.Conv2d(1, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Downsampling
            nn.MaxPool2d((2,2)),

            # 두번째 DS-Conv 블록
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(64, 2)  # [keyword, non-keyword] 클래스
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)