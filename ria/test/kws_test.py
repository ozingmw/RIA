import numpy as np
import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ria.stt.kws import KWSPreprocessor, DSCNN
from ria.stt.kws_advanced import LiteKWSPreprocessor, TinyDSCNN

# 1. 가짜 오디오 생성 (2초짜리)
sample_rate = 16000
duration_seconds = 2
dummy_audio = np.random.randint(
    low=-32768,
    high=32767,
    size=sample_rate * duration_seconds,
    dtype=np.int16
)

# 2. 전처리기
preprocessor = KWSPreprocessor()
mel = preprocessor(dummy_audio)

print("Mel shape:", mel.shape)
# 기대값: (1, 1, 40, 200)

# 3. 모델
model = DSCNN()
model.eval()

# 4. forward
with torch.no_grad():
    output = model(mel)

print("Model output:", output)
print("Output shape:", output.shape)
# 기대값: (1, 2)

# --- Advanced KWS sanity check ---
adv_preprocessor = LiteKWSPreprocessor(sample_rate=sample_rate, n_mels=24)
adv_mel = adv_preprocessor(dummy_audio)

print("Adv Mel shape:", adv_mel.shape)
# 기대값: (1, 1, 24, 200)

adv_model = TinyDSCNN(num_classes=2, use_softmax=True)
adv_model.eval()

with torch.no_grad():
    adv_output = adv_model(adv_mel)

print("Adv Model output:", adv_output)
print("Adv Output shape:", adv_output.shape)
# 기대값: (1, 2)
