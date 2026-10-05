"""토익 LC MP3 → 문제별 구간 자동 분할.

1) faster-whisper(로컬, CPU)로 받아쓰기(단어 시각 포함) — 결과는 캐시
2) "Number 34", "Questions 44 through 46", "Part 3" 안내를 찾아 경계로 사용
3) 받아쓰기가 놓친 번호는 앞뒤 번호 사이의 무음 뒤 말소리 시작점을 짧게 다시 받아써서 찾음
4) Part 1·2 = 문제 하나씩, Part 3·4 = 대화 + 질문 3개 묶음

사용: python segment.py <mp3> <out.json> [--transcript cache.json]
"""
import json, os, re, sys, time, hashlib
import numpy as np

SR = 16000
WORDS = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
GROUPS = [(a, a + 2) for a in range(32, 99, 3)]  # 32-34 … 98-100
LEAD = re.compile(r"^(\d{1,3})(?:[.,)]|\s|$)")  # "Number" 이 빠진 "2. Look at…", "18 This…"


def log(*a):
    print(*a, flush=True)


def norm(w):
    return re.sub(r"[^a-z0-9\-]", "", w.lower())


def parse_num(toks, i):
    """toks[i:] 에서 숫자 하나 → (값, 사용한 토큰 수)"""
    if i >= len(toks):
        return None, 0
    t = toks[i]
    if re.fullmatch(r"\d{1,3}", t):
        return int(t), 1
    if t in WORDS:
        return WORDS.index(t), 1
    m = re.fullmatch(r"([a-z]+)-([a-z]+)", t)
    if m and m.group(1) in TENS and m.group(2) in WORDS[1:10]:
        return TENS[m.group(1)] + WORDS.index(m.group(2)), 1
    if t in TENS:
        nx = toks[i + 1].lstrip("-") if i + 1 < len(toks) else ""  # "twenty" "two" / "twenty" "-two"
        if nx in WORDS[1:10]:
            return TENS[t] + WORDS.index(nx), 2
        return TENS[t], 1
    if t == "hundred" or (t in ("one", "a") and i + 1 < len(toks) and toks[i + 1] == "hundred"):
        return 100, 2 if t != "hundred" else 1
    return None, 0


def find_cues(words):
    """words: [(start, end, raw)] → cues [(t, kind, a, b)] kind=q|group|stop"""
    toks = [norm(w[2]) for w in words]
    cues = []
    for i, t in enumerate(toks):
        prev = toks[i - 1] if i else ""
        if t in ("question", "questions"):
            # "Questions 41 through 43" / "Question 41-43"
            j = i + 1
            if j < len(toks) and re.fullmatch(r"\d{1,3}-\d{1,3}", toks[j]):
                a, b = map(int, toks[j].split("-"))
                cues.append((words[i][0], "group", a, b))
                continue
            a, k = parse_num(toks, j)
            nxt = toks[j + k] if a and j + k < len(toks) else ""
            if re.fullmatch(r"-\d{1,3}", nxt):  # "32" "-34"
                b = int(nxt[1:])
                if 0 < b - a <= 3:
                    cues.append((words[i][0], "group", a, b))
            elif nxt in ("through", "thru", "to", "and"):
                b, _ = parse_num(toks, j + k + 1)
                if b and 0 < b - a <= 3:
                    cues.append((words[i][0], "group", a, b))
        elif (t == "number" or (t == "no" and words[i][2].strip().lower() == "no.")) and prev not in ("marked", "question"):
            n, _ = parse_num(toks, i + 1)
            if n and 1 <= n <= 100:
                cues.append((words[i][0], "q", n, n))
        elif t == "part" and i + 1 < len(toks) and (toks[i + 1] in ("1", "2", "3", "4", "one", "two", "three", "four")):
            cues.append((words[i][0], "stop", 0, 0))
        elif t == "go" and " ".join(toks[i : i + 6]).startswith("go on to the next page"):
            cues.append((words[i][0], "stop", 0, 0))
        elif t in ("end",) and " ".join(toks[i : i + 5]).startswith("end of the listening"):
            cues.append((words[i][0], "stop", 0, 0))
    return cues


def starts_with_cue(model, audio, st, kind, n):
    """st 부터 다시 받아써서 해당 안내(kind=q: Number n, group: Questions n through …)로 시작하는지.
    Whisper 가 짧은 조각에서 첫 마디를 가끔 빠뜨리므로 8초·5초 두 길이로 확인"""
    for L in (8.0, 5.0):
        r, _ = model.transcribe(audio[int(st * SR): int((st + L) * SR)], beam_size=5, condition_on_previous_text=False, without_timestamps=True)
        text = " ".join(x.text for x in r).strip()
        toks = [norm(x) for x in text.split()]
        lead = LEAD.match(text)
        if lead:
            toks = ["number", lead.group(1)] + toks[1:]
        if any(c[1] == kind and c[2] == n for c in find_cues([(0, 0, x) for x in toks[:6]])):
            return True, text
    return False, text


def silences(audio):
    hop = SR // 50  # 20ms
    n = len(audio) // hop
    rms = np.sqrt(np.mean(audio[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    floor, speech = np.percentile(db, 5), np.percentile(db, 90)
    thr = floor + max(6, (speech - floor) * 0.3)
    quiet = db < thr
    out, s = [], None
    for i, q in enumerate(np.append(quiet, False)):
        if q and s is None:
            s = i
        elif not q and s is not None:
            if i - s >= 20:  # 0.4s
                out.append((s / 50, i / 50))
            s = None
    return out


def main():
    src, out = sys.argv[1], sys.argv[2]
    cache = sys.argv[sys.argv.index("--transcript") + 1] if "--transcript" in sys.argv else None
    t0 = time.time()
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio

    sha = hashlib.sha256(open(src, "rb").read()).hexdigest()
    audio = decode_audio(src, sampling_rate=SR)
    dur = len(audio) / SR
    # WHISPER_DEVICE=cuda 면 GPU(float16), 기본은 CPU(int8)
    dev = os.environ.get("WHISPER_DEVICE", "cpu")
    model = WhisperModel("small.en", device=dev, compute_type="float16" if dev == "cuda" else "int8", cpu_threads=8)

    segs = None
    if cache:
        try:
            segs = json.load(open(cache))["segments"]
            log("받아쓰기 캐시 사용")
        except Exception:
            segs = None
    if segs is None:
        res, _ = model.transcribe(audio, word_timestamps=True, beam_size=1, condition_on_previous_text=False)
        segs = []
        for s in res:
            segs.append({"start": s.start, "end": s.end, "text": s.text.strip(),
                         "words": [{"start": w.start, "end": w.end, "word": w.word.strip()} for w in (s.words or [])]})
            if len(segs) % 40 == 0:
                log(f"PROGRESS {min(99, int(s.end / dur * 90))}")
        if cache:
            json.dump({"segments": segs}, open(cache, "w"), ensure_ascii=False)
    log(f"PROGRESS 90")

    words = []
    for s in segs:
        ws = s.get("words") or []
        if ws:
            words += [(w["start"], w["end"], w["word"]) for w in ws]
        else:
            words += [(s["start"], s["end"], w) for w in s["text"].split()]
        # 받아쓰기가 'Number'를 빠뜨린 "6. Look at the picture…"
        m = re.match(r"^(\d{1,3})[.)](?:\s|$)", s["text"])
        if m and int(m.group(1)) <= 31:
            words.append((s["start"], s["start"], "number"))
            words.append((s["start"], s["start"], m.group(1)))
    words.sort(key=lambda w: w[0])
    cues = find_cues(words)
    sil = silences(audio)

    def recheck(t_from, t_to, pattern, min_gap):
        """t_from~t_to 사이, min_gap 초 이상 무음 뒤 말소리 시작점을 짧게 다시 받아써서 pattern 을 찾음"""
        cands = [(e, e - s) for s, e in sil if t_from < e < t_to and e - s >= min_gap]
        cands.sort(key=lambda x: -x[1])
        for onset, _ in cands[:16]:
            clip = audio[int(max(0, onset - 0.3) * SR): int((onset + 4.0) * SR)]
            r, _ = model.transcribe(clip, beam_size=5, condition_on_previous_text=False, without_timestamps=True)
            text = " ".join(x.text for x in r).strip()
            toks = [norm(x) for x in text.split()]
            lead = LEAD.match(text)
            if lead:
                toks = ["number", lead.group(1)] + toks[1:]
            for c in find_cues([(onset, onset, x) for x in toks]):
                if pattern(c):
                    return onset, text.strip()
        return None, None

    # 번호별 첫 안내(같은 번호가 여러 번 잡히면 첫 번째)
    qcue, gcue, numcue = {}, {}, {}
    for t, k, a, b in cues:
        if k == "group" and (a, b) in GROUPS and a not in gcue:
            gcue[a] = (t, "asr")
    # Part 3 시작(첫 묶음 안내) 전의 "Number n" 만 Part 1·2 문제로 보고, 시간 순서가 번호 순서와 맞는 것만 쓴다
    # (대화 속 "room number one" 같은 언급 배제)
    part3 = min([t for t, _ in gcue.values()] + [dur])
    def hits_of(n, lo):
        return [t for t, k, a, _ in cues if k == "q" and a == n and lo < t < part3]

    last = -1.0
    for n in range(1, 32):
        hits = hits_of(n, last)
        # 다음 번호(n+1, n+2) 안내보다 뒤에 있는 n 은 버린다(잘못 받아쓴 번호가 순서를 망치지 않게)
        later = [h[0] for h in (hits_of(n + 1, last), hits_of(n + 2, last)) if h]
        if later:
            hits = [t for t in hits if t < min(later)]
        if hits:
            qcue[n] = (hits[0], "asr")
            last = hits[0]
    for t, k, a, b in cues:
        if k == "q" and a >= 32 and t > part3 - 60 and a not in numcue:
            numcue[a] = t
    stops = sorted(t for t, k, _, _ in cues if k == "stop")

    def longest_gap_onset(lo, hi):
        c = [(e, e - s) for s, e in sil if lo < e < hi]
        return max(c, key=lambda x: x[1])[0] if c else None

    # Part 1·2 에서 놓친 번호 보충
    for n in range(1, 32):
        if n in qcue:
            continue
        # 앞뒤로 확인된 번호 사이에서 "Number n" 을 다시 받아써서 찾는다(파트 안내 문구로 창을 자르지 않음)
        lo = max([qcue[m][0] for m in qcue if m < n], default=0)
        hi = min([qcue[m][0] for m in qcue if m > n] + ([gcue[32][0]] if 32 in gcue else []) + [dur])
        t, text = recheck(lo + 1, hi, lambda c: c[1] == "q" and c[2] == n, 1.0)
        if t is not None:
            qcue[n] = (t, "recheck")
            log(f"보충 {n}번 @{t:.1f} ← '{text}'")
            continue
        # 앞뒤 번호가 모두 확인됐으면 그 사이 가장 긴 무음(정답 고르는 쉼) 뒤를 시작으로 추정.
        # 파트가 바뀌는 자리(6→7)는 'Part 2 Directions' 안내 뒤에서만 찾는다.
        if n - 1 in qcue and n + 1 in qcue:
            lo2, hi2 = lo + 3, hi - 1
            inside = [x for x in stops if lo2 < x < hi2]
            if n == 7 and inside:
                lo2 = inside[-1] + 3
            t = longest_gap_onset(lo2, hi2)
            if t is not None:
                qcue[n] = (t, "estimated")
                log(f"추정 {n}번 @{t:.1f}")
                continue
        log(f"못 찾음 {n}번")

    # Part 3·4 묶음 보충: 'Questions a through b' 는 'Number a' 안내보다 앞(대화 앞)에 있다
    for a, b in GROUPS:
        if a in gcue:
            continue
        hi = min([gcue[m][0] for m in gcue if m > a] + ([numcue[a]] if a in numcue else []) + [dur])
        lo = max([gcue[m][0] for m in gcue if m < a] + ([numcue[a - 1] + 3] if a - 1 in numcue else []) + [x for x in stops if x < hi] + [0])
        t, text = recheck(lo, hi, lambda c: c[1] == "group" and c[2] == a, 0.6)
        if t is not None:
            gcue[a] = (t, "recheck")
            log(f"보충 {a}~{b}번 @{t:.1f} ← '{text}'")
            continue
        if a in numcue:
            # 대화는 최소 20초 → 'Number a' 20초 이전 구간에서 가장 긴 무음 뒤를 시작으로 추정
            t = longest_gap_onset(lo + 1, numcue[a] - 20)
            if t is not None:
                gcue[a] = (t, "estimated")
                log(f"추정 {a}~{b}번 @{t:.1f}")
                continue
        log(f"못 찾음 {a}~{b}번")

    # 시작점: 토익 음원은 문제 안내 직전에 항상 정답 고르는 긴 쉼(4~8초)이 있다.
    # 안내 시각 근처(3초 전~1초 후)에서 끝나는 가장 긴 무음의 끝 = 그 문제의 시작(0.3초 여유)
    def place(t, kind=None, n=None):
        near = [(e, e - s_) for s_, e in sil if t - 3 <= e <= t + 1.0]
        if near:
            e = max(near, key=lambda x: x[1])[0]
            return max(0.0, e - 0.3), True
        return max(0.0, t - 0.3), False

    items = []
    for n in range(1, 32):
        part = 1 if n <= 6 else 2
        if n not in qcue:
            items.append({"numbers": [n], "part": part, "start": None, "end": None, "source": "missing", "verified": False})
            continue
        t, srcs = qcue[n]
        st, ok = place(t, "q", n)
        items.append({"numbers": [n], "part": part, "start": st, "source": srcs, "verified": ok})
    for a, b in GROUPS:
        part = 3 if a < 71 else 4
        if a not in gcue:
            items.append({"numbers": list(range(a, b + 1)), "part": part, "start": None, "end": None, "source": "missing", "verified": False})
            continue
        t, srcs = gcue[a]
        st, ok = place(t, "group", a)
        items.append({"numbers": list(range(a, b + 1)), "part": part, "start": st, "source": srcs, "verified": ok})

    # 끝점: 다음 구간 시작 또는 파트 안내 직전. 정답 고르는 긴 쉼은 0.5초만 남긴다
    starts = sorted(i["start"] for i in items if i["start"] is not None)
    for it in items:
        if it["start"] is None:
            continue
        st = it["start"]
        nxt = min([x for x in starts if x > st + 0.5] + [x for x in stops if x > st + 3] + [dur])
        end = nxt
        for s_, e in sil:
            if s_ > st and s_ < nxt <= e + 0.3:
                end = min(nxt, s_ + 0.5)
        it["start"], it["end"] = round(st, 2), round(end, 2)
    unv = [i["numbers"][0] for i in items if i["start"] is not None and not i["verified"]]
    log(f"시작점 근처에 쉼이 없어 안내 0.3초 전으로 둔 번호: {unv}")

    json.dump({"file": src.split("/")[-1], "sha256": sha, "duration": round(dur, 3), "items": items,
               "createdAt": int(time.time() * 1000), "model": "faster-whisper small.en"}, open(out, "w"), ensure_ascii=False, indent=1)
    miss = [i["numbers"][0] for i in items if i["start"] is None]
    log(f"PROGRESS 100")
    log(f"완료 {time.time() - t0:.0f}초, 누락 {miss}")


if __name__ == "__main__":
    main()
