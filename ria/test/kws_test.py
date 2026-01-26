import numpy as np
import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ria.stt.kws import KWSPreprocessor, DSCNN

# 1. 가짜 오디오 생성 (1초짜리)
sample_rate = 16000
dummy_audio = np.random.randint(
    low=-32768,
    high=32767,
    size=sample_rate,
    dtype=np.int16
)

# 2. 전처리기
preprocessor = KWSPreprocessor()
mel = preprocessor(dummy_audio)

print("Mel shape:", mel.shape)
# 기대값: (1, 1, 40, 100)

# 3. 모델
model = DSCNN()
model.eval()

# 4. forward
with torch.no_grad():
    output = model(mel)

print("Model output:", output)
print("Output shape:", output.shape)
# 기대값: (1, 2)
