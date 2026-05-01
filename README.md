# STT Whisper Server

Small HTTP speech-to-text server for offloading Whisper inference from a slow local machine to a faster server.

The server accepts `model=...` per request. It keeps only one Whisper model loaded in RAM. Same-model requests can run in parallel. If a different model is requested, the server waits for active transcriptions to finish, unloads the old model, loads the requested one, then processes.

Audio is accepted for processing only. The server does not save transcripts and deletes the temporary uploaded audio after each request. By default temp audio is written under `/dev/shm` (tmpfs memory) inside the container.

## Model behavior

Default model if client sends no `model` field: `base`.

Per-request override:

```bash
curl -X POST \
  -F "model=turbo" \
  -F "audio=@sample.wav" \
  http://SERVER_IP:5279/transcribe
```

Allowed built-ins:

```text
tiny tiny.en base base.en small small.en medium medium.en
large-v1 large-v2 large-v3 large
distil-small.en distil-medium.en distil-large-v2 distil-large-v3 distil-large-v3.5
large-v3-turbo turbo
```

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
# edit .env, set DEFAULT_MODEL/PRELOAD_MODEL/ALLOWED_MODELS if wanted

docker compose up -d --build

curl -X POST \
  -H "X-STT-Token: YOUR_PRIVATE_TOKEN" \
  -F "model=small.en" \
  -F "audio=@sample.wav" \
  http://SERVER_IP:5279/transcribe
```

## Config

Set via Docker Compose environment variables:

| Variable | Default |
|---|---|
| `MODEL_SIZE` | `base` legacy fallback |
| `DEFAULT_MODEL` | `base` |
| `PRELOAD_MODEL` | empty/off |
| `ALLOWED_MODELS` | built-in list |
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

Concurrency: the API accepts concurrent requests. Transcription runs in a bounded thread pool controlled by `MAX_CONCURRENT_TRANSCRIPTIONS`. Default is `3`, so up to 3 same-model requests transcribe in parallel. Different-model requests wait until active jobs finish, then switch the single loaded model.

Logs include model config, model switching, and request timing. They do not log transcript text or auth tokens.

```bash
docker compose logs -f
```

Look for lines like:

```text
server_start default_model=base preload_model=none ...
model_switch_start request_id=... from_model=base to_model=turbo
model_loaded model_size=turbo load_seconds=...
request_received request_id=... requested_model=turbo loaded_model=turbo
transcribe_start request_id=... model_size=turbo ...
transcribe_done request_id=... model_size=turbo audio_duration=... text_chars=... transcribe_seconds=...
```
