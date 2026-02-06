#!/usr/bin/env bash
set -euo pipefail

src="${1:-/Users/jh/Desktop/workspace/RIA/ria/train/raw_data}"
dest="${2:-/Users/jh/Desktop/workspace/RIA/ria/train/datasets}"

find "$src" -type f -name "*.m4a" | while read -r f; do
  rel="${f#$src/}"
  out="$dest/${rel%.m4a}.wav"
  mkdir -p "$(dirname "$out")"
  ffmpeg -y -i "$f" \
    -ar 16000 -ac 1 -c:a pcm_s16le \
    "$out"
done
