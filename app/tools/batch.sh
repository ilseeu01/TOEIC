#!/bin/bash
# mp3 폴더의 아직 분할 안 된 파일을 하나씩 처리(기본 GPU, WHISPER_DEVICE=cpu 로 바꿀 수 있음)
cd "$(dirname "$0")/.."
PY=/home/capstone/miniconda3/envs/jimenglish/bin/python
ls ../mp3/*.mp3 | while read f; do
  n=$(basename "$f" .mp3)
  [ -f "data/segments/$n.json" ] && continue
  echo "$f"
done | WHISPER_DEVICE=${WHISPER_DEVICE:-cuda} xargs -P 1 -I{} bash -c 'n=$(basename "{}" .mp3); '"$PY"' tools/segment.py "{}" "data/segments/$n.json" --transcript "data/transcripts/$n.json" > "data/logs/$n.log" 2>&1; echo "$n: $(tail -1 "data/logs/$n.log")"'
