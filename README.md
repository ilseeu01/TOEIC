# 토익 LC 문제별 듣기

번호를 누르면 그 문제의 원음만 재생하는 웹앱입니다.
문제 구간은 음성인식(Whisper)으로 미리 잘라 두었고(`app/data/segments/`), 재생할 때는 AI 를 쓰지 않습니다.

- Part 1·2: 문제 하나씩
- Part 3·4: 대화 + 질문 3개 묶음 (44·45·46 중 아무 번호나 누르면 44~46 묶음 재생)
- 잘못 잘린 곳: 하단 '구간 조정'에서 시작·끝 ±1초

## 노트북에서 실행

1. [Node.js](https://nodejs.org) 18 이상 설치
2. 터미널(Windows 는 PowerShell)에서
   ```bash
   git clone https://github.com/ilseeu01/TOEIC.git
   cd TOEIC/app
   npm install
   npm start
   ```
3. 브라우저에서 http://127.0.0.1:5190

GPU 는 필요 없습니다. `mp3/` 의 TEST 02~10 은 이미 분할되어 있어 바로 쓸 수 있습니다.

## 폴더

```
mp3/                      원본 MP3 (분할 결과는 파일 이름으로 연결되므로 이름을 바꾸지 마세요)
app/server/index.js       서버(목록·재생·구간 조정 저장)
app/public/index.html     화면
app/data/segments/        자동 분할 결과(문제별 시작·끝 시각)
app/data/adjust/          '구간 조정'으로 고친 값
app/data/transcripts/     Whisper 받아쓰기(다시 분할할 때 재사용)
app/tools/segment.py      자동 분할
app/tools/verify.py       분할 검증(더 큰 모델로 각 구간 시작을 다시 받아써서 확인)
```

## 새 MP3 분할

`mp3/` 에 새 파일을 넣으면, Whisper 가 설치된 컴퓨터에서는 서버가 자동으로 분할합니다.
Whisper 가 없으면 화면에 '분할 안 됨'으로 표시되고 재생만 됩니다.

Whisper 설정(Python 3.9+):
```bash
pip install faster-whisper
# Windows 예: set PYTHON=C:\Python311\python.exe   / macOS·Linux 예: export PYTHON=$(which python3)
npm start
```
또는 직접 실행: `python tools/segment.py "../mp3/새파일.mp3" "data/segments/새파일.json" --transcript "data/transcripts/새파일.json"`
(GPU 가 있으면 `WHISPER_DEVICE=cuda`, 없으면 CPU 로 파일당 수 분)

## 분할 검증 결과 (2026-10-05)

TEST 02~10 9개 파일 × 54구간(문제 31 + 묶음 23) = 486구간을 large-v3 로 다시 받아써서 확인: 문제점 0개.
받아쓰기가 번호를 놓쳐 무음 위치로 추정한 곳 3개(TEST 07 11번, TEST 08 20번, TEST 09 29번)도 확인 통과.
