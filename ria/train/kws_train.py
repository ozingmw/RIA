import csv
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import yaml
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ria.stt.kws import DSCNN, KWSPreprocessor

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac", ".wma"}
# 학습 설정 파일은 스크립트와 같은 폴더에 둔다.
CONFIG_PATH = Path(__file__).resolve().parent / "kws_config.yaml"


@dataclass
class DatasetSpec:
    items: list[tuple[Path, int]]
    label_to_index: dict[str, int]


class KWSDataset(Dataset):
    def __init__(
        self,
        items: list[tuple[Path, int]],
        preprocessor: KWSPreprocessor,
        target_samples: int,
        random_crop: bool,
    ):
        self.items = items
        self.preprocessor = preprocessor
        self.target_samples = target_samples
        self.random_crop = random_crop

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        path, label = self.items[index]
        audio = load_audio(path)
        audio = trim_or_pad(audio, self.target_samples, self.random_crop)
        mel = self.preprocessor(audio)
        return mel.squeeze(0), label


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_audio_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    files = [p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTS]
    return sorted(files)


def load_labels_csv(csv_path: Path) -> DatasetSpec:
    items: list[tuple[Path, int]] = []
    label_to_index: dict[str, int] = {}
    with csv_path.open(newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            first = row[0].strip()
            if not first or first.startswith("#"):
                continue
            if first.lower() in {"path", "file"}:
                continue
            if len(row) < 2:
                continue
            rel_path = Path(first)
            label_name = row[1].strip()
            if label_name not in label_to_index:
                label_to_index[label_name] = len(label_to_index)
            label = label_to_index[label_name]
            # labels.csv 기준 상대 경로를 실제 파일 경로로 해석
            path = rel_path if rel_path.is_absolute() else csv_path.parent / rel_path
            items.append((path, label))
    missing = [path for path, _ in items if not path.exists()]
    if missing:
        missing_text = "\n".join(str(p) for p in missing[:5])
        raise FileNotFoundError(f"Missing dataset files:\n{missing_text}")
    return DatasetSpec(items=items, label_to_index=label_to_index)


def load_from_subdirs(split_dir: Path) -> DatasetSpec:
    items: list[tuple[Path, int]] = []
    label_to_index: dict[str, int] = {}
    class_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])
    for class_dir in class_dirs:
        label_name = class_dir.name
        if label_name not in label_to_index:
            label_to_index[label_name] = len(label_to_index)
        label = label_to_index[label_name]
        for path in list_audio_files(class_dir):
            items.append((path, label))
    return DatasetSpec(items=items, label_to_index=label_to_index)


def build_dataset_spec(split_dir: Path) -> DatasetSpec:
    labels_csv = split_dir / "labels.csv"
    if labels_csv.exists():
        return load_labels_csv(labels_csv)
    class_dirs = (
        [p for p in split_dir.iterdir() if p.is_dir()] if split_dir.exists() else []
    )
    if class_dirs:
        return load_from_subdirs(split_dir)
    audio_files = list_audio_files(split_dir)
    if audio_files:
        raise ValueError(
            f"Cannot infer labels in {split_dir}. Use class subfolders or labels.csv."
        )
    return DatasetSpec(items=[], label_to_index={})


def remap_items(
    items: list[tuple[Path, int]],
    src_label_to_index: dict[str, int],
    dst_label_to_index: dict[str, int],
) -> list[tuple[Path, int]]:
    index_to_label = {idx: name for name, idx in src_label_to_index.items()}
    remapped: list[tuple[Path, int]] = []
    for path, label in items:
        label_name = index_to_label[label]
        if label_name not in dst_label_to_index:
            raise RuntimeError(f"Unknown label in evaluation set: {label_name}")
        remapped.append((path, dst_label_to_index[label_name]))
    return remapped


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def parse_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def parse_num_workers(value: object) -> int:
    if value is None:
        return 0 if os.name == "nt" else 2
    if isinstance(value, str) and value.strip().lower() == "auto":
        return 0 if os.name == "nt" else 2
    return int(value)


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open() as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("Config root must be a mapping.")
    return config


def load_audio(path: Path, target_sr: int = 16000) -> np.ndarray:
    try:
        waveform, sample_rate = torchaudio.load(str(path))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to load audio: {path}. "
            "Convert to wav or install a backend that supports this format."
        ) from exc
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        # KWS 전처리기는 16kHz 기준이므로 리샘플링
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)
    samples = waveform.squeeze(0).cpu().numpy()
    samples = np.clip(samples, -1.0, 1.0)
    samples = (samples * 32768.0).astype(np.int16)
    return samples


def trim_or_pad(samples: np.ndarray, target_len: int, random_crop: bool) -> np.ndarray:
    current_len = samples.shape[0]
    if current_len == target_len:
        return samples
    if current_len > target_len:
        if random_crop:
            # 학습 시에는 랜덤 크롭으로 증강
            start = random.randint(0, current_len - target_len)
        else:
            # 평가/검증에서는 중앙 크롭
            start = max(0, (current_len - target_len) // 2)
        return samples[start : start + target_len]
    pad = target_len - current_len
    return np.pad(samples, (0, pad), mode="constant")


def accuracy_from_probs(probs: torch.Tensor, labels: torch.Tensor) -> float:
    preds = probs.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return correct / labels.numel()


def compute_class_weights(
    items: list[tuple[Path, int]], num_classes: int
) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.float32)
    for _, label in items:
        counts[label] += 1.0
    weights = counts.sum() / np.maximum(counts, 1.0)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    class_weights: torch.Tensor | None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_acc = 0.0
    total_count = 0

    for batch, labels in loader:
        batch = batch.to(device)
        labels = labels.to(device)
        # 모델 출력은 Softmax 확률이므로 log로 NLL 사용
        probs = model(batch).clamp(min=1e-6)
        loss = F.nll_loss(probs.log(), labels, weight=class_weights)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_acc += accuracy_from_probs(probs, labels) * batch_size
        total_count += batch_size

    if total_count == 0:
        return 0.0, 0.0
    return total_loss / total_count, total_acc / total_count


def main() -> None:
    # YAML 설정 로드 (경로는 repo root 기준으로 해석)
    config = load_config(CONFIG_PATH)
    data_root = resolve_path(config.get("data_root", "ria/train/datasets"))
    out_dir = resolve_path(config.get("out_dir", "ria/train/outputs"))
    epochs = int(config.get("epochs", 20))
    batch_size = int(config.get("batch_size", 32))
    lr = float(config.get("lr", 1e-3))
    val_split = float(config.get("val_split", 0.1))
    seed = int(config.get("seed", 42))
    num_workers = parse_num_workers(config.get("num_workers", "auto"))
    force_cpu = parse_bool(config.get("force_cpu"), False)
    use_class_weight = parse_bool(config.get("use_class_weight"), False)

    seed_everything(seed)

    train_dir = data_root / "train"
    test_dir = data_root / "test"

    # labels.csv 또는 클래스 폴더 구조 둘 다 지원
    train_spec = build_dataset_spec(train_dir)
    test_spec = build_dataset_spec(test_dir)

    if not train_spec.items:
        raise RuntimeError(f"No training data found in {train_dir}")

    label_to_index = train_spec.label_to_index
    test_items: list[tuple[Path, int]] = []
    if test_spec.items:
        if set(test_spec.label_to_index.keys()) != set(label_to_index.keys()):
            raise RuntimeError("Train/test label sets do not match.")
        test_items = remap_items(
            test_spec.items, test_spec.label_to_index, label_to_index
        )

    num_classes = len(label_to_index)
    if num_classes != 2:
        raise RuntimeError(f"Expected 2 classes, found {num_classes}.")

    train_items = list(train_spec.items)
    val_items: list[tuple[Path, int]] = []
    if test_items:
        val_items = list(test_items)
    else:
        random.shuffle(train_items)
        val_size = max(1, int(len(train_items) * val_split))
        if len(train_items) <= val_size:
            raise RuntimeError("Not enough data to create a validation split.")
        val_items = train_items[:val_size]
        train_items = train_items[val_size:]

    preprocessor = KWSPreprocessor()
    target_samples = 16000

    train_dataset = KWSDataset(
        train_items, preprocessor, target_samples, random_crop=True
    )
    val_dataset = KWSDataset(val_items, preprocessor, target_samples, random_crop=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    device = torch.device(
        "cpu" if force_cpu or not torch.cuda.is_available() else "cuda"
    )
    model = DSCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    class_weights = None
    if use_class_weight:
        class_weights = compute_class_weights(train_items, num_classes).to(device)

    out_dir.mkdir(parents=True, exist_ok=True)
    label_list = [None] * len(label_to_index)
    for name, idx in label_to_index.items():
        label_list[idx] = name
    with (out_dir / "labels.json").open("w") as handle:
        json.dump(label_list, handle, indent=2)

    best_val = -1.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, device, optimizer, class_weights
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, device, optimizer=None, class_weights=class_weights
        )
        print(
            f"Epoch {epoch:02d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val:
            best_val = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "labels": label_list,
                    "sample_rate": 16000,
                    "mel_bins": 40,
                    "time_steps": 100,
                },
                out_dir / "kws_model_best.pt",
            )

    torch.save(
        {
            "model_state": model.state_dict(),
            "labels": label_list,
            "sample_rate": 16000,
            "mel_bins": 40,
            "time_steps": 100,
        },
        out_dir / "kws_model_last.pt",
    )


if __name__ == "__main__":
    main()
