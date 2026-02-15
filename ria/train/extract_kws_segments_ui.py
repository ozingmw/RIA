#!/usr/bin/env python
"""raw_data 폴더 전체 파일을 파형 UI로 선택 구간만 추출해 extract_data에 저장."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio

parser = argparse.ArgumentParser(description="raw_data 폴더 전체 파일에서 구간 추출")
parser.add_argument("--sample-rate", type=int, default=16000)
parser.add_argument("--max-seconds", type=float, default=2.0)
args = parser.parse_args()

root = Path(__file__).resolve().parent
raw_root = root / "raw_data"
extract_root = root / "extract_data"
extract_root.mkdir(parents=True, exist_ok=True)

audio_exts = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac"}
files = sorted(
    p for p in raw_root.rglob("*") if p.suffix.lower() in audio_exts and p.is_file()
)
if not files:
    raise FileNotFoundError(f"raw_data 안에 오디오 파일이 없습니다: {raw_root}")

for file_idx, input_path in enumerate(files):
    try:
        wave, sr = torchaudio.load(str(input_path))
    except Exception as exc:
        print(f"[skip] load failed: {input_path} ({exc})")
        continue

    if wave.shape[0] > 1:
        wave = wave.mean(dim=0, keepdim=True)
    if sr != args.sample_rate:
        wave = torchaudio.transforms.Resample(sr, args.sample_rate)(wave)
        sr = args.sample_rate

    wave = np.clip(wave[0].cpu().numpy().astype(np.float32), -1.0, 1.0)
    if wave.size == 0:
        print(f"[skip] empty audio: {input_path}")
        continue

    time = np.arange(wave.size) / sr
    max_len = wave.size / sr

    try:
        rel = input_path.relative_to(raw_root)
        out_dir = extract_root / rel.parent
    except ValueError:
        out_dir = extract_root
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{input_path.stem}_selected.wav"

    if max_len <= 1e-6:
        print(f"[skip] too short: {input_path}")
        continue

    start = 0.0
    end = min(max_len, args.max_seconds)
    state = {
        "drag": None,
        "tolerance": 0.05,
        "running": True,
        "saved": False,
        "start": start,
        "end": end,
    }

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time, wave, lw=1)
    ax.set_xlim(0, max_len)
    ax.set_ylim(
        wave.min() * 1.2 if wave.size else -1.0, wave.max() * 1.2 if wave.size else 1.0
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("amplitude")
    ax.set_title(
        f"[{file_idx + 1}/{len(files)}] {input_path.name} | [{state['start']:.2f}s, {state['end']:.2f}s] (dur: {state['end'] - state['start']:.2f}s)"
    )
    l1 = ax.axvline(state["start"], color="r", lw=2)
    l2 = ax.axvline(state["end"], color="g", lw=2)
    shade = ax.axvspan(state["start"], state["end"], color="blue", alpha=0.15)

    def on_motion(event):
        if event.inaxes is not ax or event.xdata is None:
            return
        if event.button != 1:
            state["drag"] = None
            return
        x = float(np.clip(event.xdata, 0.0, max_len))
        if state["drag"] is None:
            if abs(x - state["start"]) <= state["tolerance"]:
                state["drag"] = "start"
            elif abs(x - state["end"]) <= state["tolerance"]:
                state["drag"] = "end"
            else:
                state["drag"] = (
                    "start" if x <= (state["start"] + state["end"]) / 2 else "end"
                )

        if state["drag"] == "start":
            state["start"] = max(0.0, min(x, state["end"]))
            if state["end"] - state["start"] > args.max_seconds:
                state["start"] = max(0.0, state["end"] - args.max_seconds)
        else:
            state["end"] = min(max_len, max(x, state["start"]))
            if state["end"] - state["start"] > args.max_seconds:
                state["end"] = state["start"] + args.max_seconds
                if state["end"] > max_len:
                    state["end"] = max_len
                    state["start"] = max(0.0, state["end"] - args.max_seconds)

        state["start"] = max(0.0, min(state["start"], max_len))
        state["end"] = max(state["start"] + 1e-6, min(state["end"], max_len))
        l1.set_xdata([state["start"], state["start"]])
        l2.set_xdata([state["end"], state["end"]])
        y0, y1 = ax.get_ylim()
        shade.set_xy(
            [
                (state["start"], y0),
                (state["start"], y1),
                (state["end"], y1),
                (state["end"], y0),
            ]
        )
        ax.set_title(
            f"[{file_idx + 1}/{len(files)}] {input_path.name} | [{state['start']:.2f}s, {state['end']:.2f}s] (dur: {state['end'] - state['start']:.2f}s) | s:저장, n:다음, q:종료"
        )
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in {"q", "escape"}:
            state["running"] = False
            plt.close(fig)
            state["saved"] = False
            raise SystemExit

        if event.key in {"n", "right", "enter"}:
            state["running"] = False
            plt.close(fig)
            return

        if event.key not in {"s", "space"}:
            return

        sidx = int(round(state["start"] * sr))
        eidx = int(round(state["end"] * sr))
        clip = wave[sidx:eidx]
        if clip.size == 0:
            print("선택 구간이 비어있습니다.")
            return
        if (eidx - sidx) / sr > args.max_seconds:
            print("선택 구간이 2초보다 깁니다.")
            return
        torchaudio.save(
            str(output_path), torch.from_numpy(clip.reshape(1, -1)), sample_rate=sr
        )
        print(f"saved: {output_path} ({(eidx - sidx) / sr:.3f}s)")
        state["running"] = False
        state["saved"] = True
        plt.close(fig)

    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    if not state["running"]:
        continue
