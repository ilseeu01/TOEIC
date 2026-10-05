// 토익 LC 문제별 듣기 서버
//  - mp3 폴더 목록, 파일 스트리밍
//  - 자동 분할 결과(data/segments/<이름>.json) 제공. 분할 안 된 파일은 자동으로 하나씩 처리
//  - 사용자가 ±1초 조정한 값(data/adjust/<이름>.json) 저장
import express from 'express'
import { spawn, spawnSync } from 'node:child_process'
import fs from 'node:fs/promises'
import { existsSync, statSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const APP = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const HOST = process.env.HOST ?? '127.0.0.1'
const PORT = Number(process.env.PORT ?? 5190)
const MP3_DIR = path.resolve(process.env.MP3_DIR ?? path.join(APP, '..', 'mp3'))
const DATA = path.join(APP, 'data')
// 자동 분할용 Python(faster-whisper 설치된 환경). 없으면 자동 분할을 끄고 재생만 한다(노트북 등)
const PY = process.env.PYTHON ?? '/home/capstone/miniconda3/envs/jimenglish/bin/python'
const CAN_SPLIT = existsSync(PY)
for (const d of ['segments', 'transcripts', 'adjust', 'logs']) await fs.mkdir(path.join(DATA, d), { recursive: true })

const base = (name) => name.replace(/\.mp3$/i, '')
const segFile = (name) => path.join(DATA, 'segments', `${base(name)}.json`)
const logFile = (name) => path.join(DATA, 'logs', `${base(name)}.log`)
const adjFile = (name) => path.join(DATA, 'adjust', `${base(name)}.json`)

async function listMp3() {
  const names = await fs.readdir(MP3_DIR).catch(() => [])
  return names.filter((n) => /\.mp3$/i.test(n)).sort((a, b) => a.localeCompare(b, 'ko', { numeric: true }))
}

/** 로그가 최근에 갱신되고 있으면 처리 중(서버 큐나 tools/batch.sh 어느 쪽이든) */
function running(name) {
  const f = logFile(name)
  if (!existsSync(f) || existsSync(segFile(name))) return null
  const st = statSync(f)
  if (Date.now() - st.mtimeMs > 5 * 60_000) return null
  return st
}

async function progressOf(name) {
  try {
    const log = await fs.readFile(logFile(name), 'utf8')
    const m = [...log.matchAll(/PROGRESS (\d+)/g)].pop()
    return m ? Number(m[1]) : 0
  } catch {
    return 0
  }
}

// 분할 안 된 파일을 하나씩 처리하는 큐
let busy = false
async function pump() {
  if (busy || !CAN_SPLIT) return
  // tools/batch.sh 등 다른 곳에서 분할 중이면 끝날 때까지 기다린다(같은 파일 중복 처리 방지)
  if (process.platform !== 'win32' && spawnSync('pgrep', ['-f', 'tools/segment.py']).status === 0) return
  for (const name of await listMp3()) {
    if (existsSync(segFile(name)) || running(name)) continue
    // 아직 복사 중인 파일은 건너뜀
    if (Date.now() - statSync(path.join(MP3_DIR, name)).mtimeMs < 15_000) continue
    busy = true
    const out = await fs.open(logFile(name), 'w')
    const p = spawn(PY, [path.join(APP, 'tools', 'segment.py'), path.join(MP3_DIR, name), segFile(name), '--transcript', path.join(DATA, 'transcripts', `${base(name)}.json`)], { env: { ...process.env, WHISPER_DEVICE: process.env.WHISPER_DEVICE ?? 'cuda' }, stdio: ['ignore', out.fd, out.fd] })
    p.on('error', (e) => console.error('자동 분할 실행 실패:', e.message))
    p.on('close', async () => {
      await out.close()
      busy = false
      pump()
    })
    return
  }
}
setInterval(pump, 30_000)
setTimeout(pump, 2_000)

const app = express()
app.disable('x-powered-by')

app.get('/api/tests', async (_req, res) => {
  const out = []
  for (const name of await listMp3()) {
    let status = 'queued'
    let progress = 0
    let missing = []
    if (existsSync(segFile(name))) {
      status = 'ready'
      try {
        const s = JSON.parse(await fs.readFile(segFile(name), 'utf8'))
        missing = s.items.filter((i) => i.start == null).flatMap((i) => i.numbers)
      } catch {
        status = 'error'
      }
    } else if (running(name)) {
      status = 'processing'
      progress = await progressOf(name)
    } else if (!CAN_SPLIT) {
      status = 'unsplit'
    }
    out.push({ name, status, progress, missing })
  }
  res.json(out)
})

app.get('/api/tests/:name', async (req, res) => {
  const name = req.params.name
  if (!(await listMp3()).includes(name) || !existsSync(segFile(name))) return res.status(404).send('분할 결과가 없습니다')
  const seg = JSON.parse(await fs.readFile(segFile(name), 'utf8'))
  const adj = JSON.parse(await fs.readFile(adjFile(name), 'utf8').catch(() => '{}'))
  // 자동 결과(auto)와 사용자가 고친 값(adjusted)을 구분해서 내려준다
  seg.items = seg.items.map((i) => {
    const a = adj[i.numbers[0]]
    return a ? { ...i, auto: { start: i.start, end: i.end }, start: a.start, end: a.end, adjusted: true } : i
  })
  res.json(seg)
})

app.post('/api/tests/:name/adjust', express.json(), async (req, res) => {
  const name = req.params.name
  if (!(await listMp3()).includes(name)) return res.status(404).send('없는 파일')
  const { key, start, end, reset } = req.body ?? {}
  if (!Number.isInteger(key)) return res.status(400).send('key 필요')
  const adj = JSON.parse(await fs.readFile(adjFile(name), 'utf8').catch(() => '{}'))
  if (reset) delete adj[key]
  else {
    if (!(Number.isFinite(start) && Number.isFinite(end) && end > start && start >= 0)) return res.status(400).send('구간이 올바르지 않습니다')
    adj[key] = { start: Math.round(start * 100) / 100, end: Math.round(end * 100) / 100 }
  }
  await fs.writeFile(adjFile(name), JSON.stringify(adj, null, 1))
  res.json({ ok: true })
})

app.get('/api/audio/:name', async (req, res, next) => {
  const name = req.params.name
  if (!(await listMp3()).includes(name)) return res.status(404).send('없는 파일')
  res.sendFile(path.join(MP3_DIR, name), { headers: { 'Content-Type': 'audio/mpeg' } }, (e) => e && !res.headersSent && next(e))
})

app.use(express.static(path.join(APP, 'public')))

app.listen(PORT, HOST, () => console.log(`토익 LC 문제별 듣기: http://${HOST}:${PORT}  (mp3: ${MP3_DIR}, 자동 분할: ${CAN_SPLIT ? '켬' : '꺼짐 - Whisper 없음'})`))
