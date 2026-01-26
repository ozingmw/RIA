from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from unicodedata import normalize

import numpy as np
import onnxruntime as ort

AVAILABLE_LANGS = {"en", "ko", "es", "pt", "fr"}


@dataclass
class Style:
    ttl: np.ndarray
    dp: np.ndarray


class UnicodeProcessor:
    def __init__(self, unicode_indexer_path: str):
        with open(unicode_indexer_path, "r", encoding="utf-8") as f:
            self.indexer = json.load(f)

    def _preprocess_text(self, text: str, lang: str) -> str:
        text = normalize("NFKD", text)

        emoji_pattern = re.compile(
            "[\U0001f600-\U0001f64f"
            "\U0001f300-\U0001f5ff"
            "\U0001f680-\U0001f6ff"
            "\U0001f700-\U0001f77f"
            "\U0001f780-\U0001f7ff"
            "\U0001f800-\U0001f8ff"
            "\U0001f900-\U0001f9ff"
            "\U0001fa00-\U0001fa6f"
            "\U0001fa70-\U0001faff"
            "\u2600-\u26ff"
            "\u2700-\u27bf"
            "\U0001f1e6-\U0001f1ff]+",
            flags=re.UNICODE,
        )
        text = emoji_pattern.sub("", text)

        replacements = {
            "–": "-",
            "‑": "-",
            "—": "-",
            "_": " ",
            "\u201c": '"',
            "\u201d": '"',
            "\u2018": "'",
            "\u2019": "'",
            "´": "'",
            "`": "'",
            "[": " ",
            "]": " ",
            "|": " ",
            "/": " ",
            "#": " ",
            "→": " ",
            "←": " ",
        }
        for k, v in replacements.items():
            text = text.replace(k, v)

        text = re.sub(r"[♥☆♡©\\]", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        if not re.search(r"[.!?;:,'\"')\]}…。」』】〉》›»]$", text):
            text += "."

        if lang not in AVAILABLE_LANGS:
            raise ValueError(f"지원하지 않는 언어: {lang}")

        return f"<{lang}>" + text + f"</{lang}>"

    def _text_to_unicode_values(self, text: str) -> np.ndarray:
        return np.array([ord(ch) for ch in text], dtype=np.uint16)

    def __call__(
        self, text_list: list[str], lang_list: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        text_list = [
            self._preprocess_text(t, lang) for t, lang in zip(text_list, lang_list)
        ]
        text_ids_lengths = np.array([len(t) for t in text_list], dtype=np.int64)
        max_len = int(text_ids_lengths.max()) if text_ids_lengths.size else 0
        text_ids = np.zeros((len(text_list), max_len), dtype=np.int64)
        for i, text in enumerate(text_list):
            unicode_vals = self._text_to_unicode_values(text)
            if isinstance(self.indexer, dict):
                indices = [self.indexer.get(str(val), 0) for val in unicode_vals]
            elif isinstance(self.indexer, list):
                indices = [
                    self.indexer[val] if val < len(self.indexer) else 0
                    for val in unicode_vals
                ]
            else:
                raise TypeError("unicode_indexer.json 형식이 지원되지 않습니다.")
            text_ids[i, : len(indices)] = np.array(indices, dtype=np.int64)
        text_mask = length_to_mask(text_ids_lengths)
        return text_ids, text_mask


class TextToSpeech:
    def __init__(
        self,
        cfgs: dict[str, Any],
        text_processor: UnicodeProcessor,
        dp_ort: ort.InferenceSession,
        text_enc_ort: ort.InferenceSession,
        vector_est_ort: ort.InferenceSession,
        vocoder_ort: ort.InferenceSession,
    ):
        self.cfgs = cfgs
        self.text_processor = text_processor
        self.dp_ort = dp_ort
        self.text_enc_ort = text_enc_ort
        self.vector_est_ort = vector_est_ort
        self.vocoder_ort = vocoder_ort
        self.sample_rate = int(cfgs["ae"]["sample_rate"])
        self.base_chunk_size = int(cfgs["ae"]["base_chunk_size"])
        self.chunk_compress_factor = int(cfgs["ttl"]["chunk_compress_factor"])
        self.ldim = int(cfgs["ttl"]["latent_dim"])

    def sample_noisy_latent(
        self, duration: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        bsz = len(duration)
        wav_len_max = int(duration.max() * self.sample_rate)
        wav_lengths = (duration * self.sample_rate).astype(np.int64)
        chunk_size = self.base_chunk_size * self.chunk_compress_factor
        latent_len = int((wav_len_max + chunk_size - 1) / chunk_size)
        latent_dim = self.ldim * self.chunk_compress_factor
        noisy_latent = np.random.randn(bsz, latent_dim, latent_len).astype(np.float32)
        latent_mask = get_latent_mask(
            wav_lengths, self.base_chunk_size, self.chunk_compress_factor
        )
        noisy_latent = noisy_latent * latent_mask
        return noisy_latent, latent_mask

    def _infer(
        self,
        text_list: list[str],
        lang_list: list[str],
        style: Style,
        total_step: int,
        speed: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(text_list) != style.ttl.shape[0]:
            raise ValueError("텍스트 개수와 스타일 개수가 일치해야 합니다.")
        text_ids, text_mask = self.text_processor(text_list, lang_list)
        dur_onnx, *_ = self.dp_ort.run(
            None, {"text_ids": text_ids, "style_dp": style.dp, "text_mask": text_mask}
        )
        dur_onnx = dur_onnx / speed
        text_emb_onnx, *_ = self.text_enc_ort.run(
            None, {"text_ids": text_ids, "style_ttl": style.ttl, "text_mask": text_mask}
        )
        xt, latent_mask = self.sample_noisy_latent(dur_onnx)
        total_step_np = np.array([total_step] * len(text_list), dtype=np.float32)
        for step in range(total_step):
            current_step = np.array([step] * len(text_list), dtype=np.float32)
            xt, *_ = self.vector_est_ort.run(
                None,
                {
                    "noisy_latent": xt,
                    "text_emb": text_emb_onnx,
                    "style_ttl": style.ttl,
                    "text_mask": text_mask,
                    "latent_mask": latent_mask,
                    "current_step": current_step,
                    "total_step": total_step_np,
                },
            )
        wav, *_ = self.vocoder_ort.run(None, {"latent": xt})
        return wav, dur_onnx

    def __call__(
        self,
        text: str,
        lang: str,
        style: Style,
        total_step: int,
        speed: float,
        max_chunk_length: int,
        silence_duration: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if style.ttl.shape[0] != 1:
            raise ValueError("단일 텍스트는 단일 스타일만 지원합니다.")
        text_list = chunk_text(text, max_len=max_chunk_length)
        if not text_list:
            return np.zeros((1, 0), dtype=np.float32), np.array([0.0], dtype=np.float32)
        wav_cat = None
        dur_cat = None
        for chunk in text_list:
            wav, dur_onnx = self._infer([chunk], [lang], style, total_step, speed)
            if wav_cat is None:
                wav_cat = wav
                dur_cat = dur_onnx
            else:
                silence = np.zeros(
                    (1, int(silence_duration * self.sample_rate)), dtype=np.float32
                )
                wav_cat = np.concatenate([wav_cat, silence, wav], axis=1)
                dur_cat = dur_cat + dur_onnx + silence_duration
        return wav_cat, dur_cat


def length_to_mask(lengths: np.ndarray, max_len: int | None = None) -> np.ndarray:
    if lengths.size == 0:
        return np.zeros((0, 1, 0), dtype=np.float32)
    max_len = int(max_len or lengths.max())
    ids = np.arange(0, max_len)
    mask = (ids < np.expand_dims(lengths, axis=1)).astype(np.float32)
    return mask.reshape(-1, 1, max_len)


def get_latent_mask(
    wav_lengths: np.ndarray,
    base_chunk_size: int,
    chunk_compress_factor: int,
) -> np.ndarray:
    latent_size = base_chunk_size * chunk_compress_factor
    latent_lengths = (wav_lengths + latent_size - 1) // latent_size
    return length_to_mask(latent_lengths)


def chunk_text(text: str, max_len: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not sentence:
            continue
        if len(current) + len(sentence) + 1 <= max_len:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def _load_onnx(
    onnx_path: str, opts: ort.SessionOptions, providers: list[str]
) -> ort.InferenceSession:
    return ort.InferenceSession(onnx_path, sess_options=opts, providers=providers)


def _load_onnx_all(
    onnx_dir: str, opts: ort.SessionOptions, providers: list[str]
) -> tuple[
    ort.InferenceSession,
    ort.InferenceSession,
    ort.InferenceSession,
    ort.InferenceSession,
]:
    dp_onnx_path = str(Path(onnx_dir) / "duration_predictor.onnx")
    text_enc_onnx_path = str(Path(onnx_dir) / "text_encoder.onnx")
    vector_est_onnx_path = str(Path(onnx_dir) / "vector_estimator.onnx")
    vocoder_onnx_path = str(Path(onnx_dir) / "vocoder.onnx")
    dp_ort = _load_onnx(dp_onnx_path, opts, providers)
    text_enc_ort = _load_onnx(text_enc_onnx_path, opts, providers)
    vector_est_ort = _load_onnx(vector_est_onnx_path, opts, providers)
    vocoder_ort = _load_onnx(vocoder_onnx_path, opts, providers)
    return dp_ort, text_enc_ort, vector_est_ort, vocoder_ort


def _load_cfgs(onnx_dir: str) -> dict[str, Any]:
    cfg_path = Path(onnx_dir) / "tts.json"
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_text_processor(onnx_dir: str) -> UnicodeProcessor:
    unicode_indexer_path = str(Path(onnx_dir) / "unicode_indexer.json")
    return UnicodeProcessor(unicode_indexer_path)


@lru_cache(maxsize=2)
def load_text_to_speech(onnx_dir: str, use_gpu: bool = False) -> TextToSpeech:
    opts = ort.SessionOptions()
    if use_gpu:
        raise NotImplementedError("GPU 모드는 아직 검증되지 않았습니다.")
    providers = ["CPUExecutionProvider"]
    cfgs = _load_cfgs(onnx_dir)
    dp_ort, text_enc_ort, vector_est_ort, vocoder_ort = _load_onnx_all(
        onnx_dir, opts, providers
    )
    text_processor = _load_text_processor(onnx_dir)
    return TextToSpeech(
        cfgs, text_processor, dp_ort, text_enc_ort, vector_est_ort, vocoder_ort
    )


@lru_cache(maxsize=8)
def load_voice_style(voice_style_path: str) -> Style:
    with open(voice_style_path, "r", encoding="utf-8") as f:
        voice_style = json.load(f)
    ttl_dims = voice_style["style_ttl"]["dims"]
    dp_dims = voice_style["style_dp"]["dims"]
    ttl_data = np.array(voice_style["style_ttl"]["data"], dtype=np.float32).flatten()
    dp_data = np.array(voice_style["style_dp"]["data"], dtype=np.float32).flatten()
    ttl_style = ttl_data.reshape(1, ttl_dims[1], ttl_dims[2])
    dp_style = dp_data.reshape(1, dp_dims[1], dp_dims[2])
    return Style(ttl_style, dp_style)
