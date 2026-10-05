"""분할 검증: 모든 구간의 앞 4초를 다시 받아써서 해당 번호 안내로 시작하는지, 길이·순서가 정상인지 확인"""
import glob, json, re, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import segment as S
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio

dev = os.environ.get("WHISPER_DEVICE", "cpu")
# 분할에 쓴 small.en 과 다른, 더 큰 모델로 독립 확인
m = WhisperModel(os.environ.get("VERIFY_MODEL", "large-v3"), device=dev, compute_type="float16" if dev == "cuda" else "int8")
LIM = {1: (12, 45), 2: (8, 35), 3: (50, 130), 4: (50, 130)}
total = bad = 0
for f in sorted(glob.glob("data/segments/*.json")):
    d = json.load(open(f))
    a = decode_audio(os.path.join("..", "mp3", d["file"]), sampling_rate=16000)
    prev_end, probs = 0, []
    for it in d["items"]:
        n = it["numbers"][0]
        total += 1
        st, en = it["start"], it["end"]
        if st is None:
            probs.append(f"{n}: 없음"); continue
        L = en - st
        lo, hi = LIM[it["part"]]
        if not (lo <= L <= hi): probs.append(f"{n}: 길이 {L:.1f}s")
        if st < prev_end - 0.01: probs.append(f"{n}: 앞 구간과 겹침")
        prev_end = en
        ok, text = S.starts_with_cue(m, a, st, "group" if len(it["numbers"]) > 1 else "q", n)
        if not ok: probs.append(f"{n}({it['source']}): 시작이 '{text[:50]}'")
    bad += len(probs)
    print(f"{d['file']}: {'OK' if not probs else ' / '.join(probs)}", flush=True)
print(f"문제 항목 {total}개 중 문제점 {bad}개")
