# STT Whisper Server

Small HTTP speech-to-text server for offloading Whisper inference from a slow local machine to a faster server.

The container keeps the `faster-whisper` model loaded in a long-running FastAPI process, so requests avoid per-request model startup cost.

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

With optional token auth:

```bash
cp .env.example .env
# edit .env, set STT_API_TOKEN privately

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

## Server notes

Expose only on trusted networks or use firewall/VPN/reverse proxy auth. If `STT_API_TOKEN` is empty, anyone who can reach the port can submit audio.
