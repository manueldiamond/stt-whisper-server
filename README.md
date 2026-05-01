# STT Whisper Server

Small HTTP speech-to-text server for offloading Whisper inference from a slow local machine to a faster server.

The container keeps the `faster-whisper` model loaded in a long-running FastAPI process, so requests avoid per-request model startup cost.

Audio is accepted for processing only. The server does not save transcripts and deletes the temporary uploaded audio after each request. By default temp audio is written under `/dev/shm` (tmpfs memory) inside the container.

## Model

Default model: `base`

This is one tier above `tiny`.

## Attribution

This repo is only a wrapper/API server around third-party speech-to-text tooling. I do **not** own the native `faster-whisper`, Whisper, CTranslate2, FFmpeg, or related model/code implementations. See their upstream projects and licenses.

## Run

```bash
docker compose up -d --build
```

Server listens on port `5279`.

Health check:

```bash
curl http://SERVER_IP:5279/health
```

Transcribe:

```bash
curl -X POST \
  -F "audio=@sample.wav" \
  http://SERVER_IP:5279/transcribe
```

With optional token auth and model override:

```bash
cp .env.example .env
# edit .env, set STT_API_TOKEN privately
# edit .env, set MODEL_SIZE=small/medium/turbo if wanted

docker compose up -d --build

curl -X POST \
  -H "X-STT-Token: YOUR_PRIVATE_TOKEN" \
  -F "audio=@sample.wav" \
  http://SERVER_IP:5279/transcribe
```

## Config

Set via Docker Compose environment variables:

| Variable | Default |
|---|---|
| `MODEL_SIZE` | `base` |
| `DEVICE` | `cpu` |
| `COMPUTE_TYPE` | `int8` |
| `LANGUAGE` | `en` |
| `BEAM_SIZE` | `1` |
| `VAD_FILTER` | `true` |
| `VAD_MIN_SILENCE_DURATION_MS` | `500` |
| `STT_API_TOKEN` | empty/off |
| `MAX_CONCURRENT_TRANSCRIPTIONS` | `3` |
| `MODEL_NUM_WORKERS` | same as `MAX_CONCURRENT_TRANSCRIPTIONS` |
| `TMP_DIR` | `/dev/shm` |
| `LOG_LEVEL` | `INFO` |

## Server notes

Expose only on trusted networks or use firewall/VPN/reverse proxy auth. If `STT_API_TOKEN` is empty, anyone who can reach the port can submit audio.

Concurrency: the API accepts concurrent requests. Transcription runs in a bounded thread pool controlled by `MAX_CONCURRENT_TRANSCRIPTIONS`. Default is `3`, so up to 3 requests transcribe in parallel; extra requests wait instead of spawning unlimited CPU-heavy work.

Logs include model config on startup and request timing per transcription. They do not log transcript text or auth tokens.

```bash
docker compose logs -f
```

Look for lines like:

```text
loading_model model_size=turbo device=cpu compute_type=int8 ...
transcribe_start request_id=... model_size=turbo ...
transcribe_done request_id=... model_size=turbo audio_duration=... text_chars=... transcribe_seconds=...
```
