import asyncio
import logging
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from faster_whisper import WhisperModel

MODEL_SIZE = os.getenv("MODEL_SIZE", "base")
DEVICE = os.getenv("DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")
LANGUAGE = os.getenv("LANGUAGE", "en")
BEAM_SIZE = int(os.getenv("BEAM_SIZE", "1"))
VAD_FILTER = os.getenv("VAD_FILTER", "true").lower() in {"1", "true", "yes", "on"}
VAD_MIN_SILENCE_DURATION_MS = int(os.getenv("VAD_MIN_SILENCE_DURATION_MS", "500"))
STT_API_TOKEN = os.getenv("STT_API_TOKEN", "")
MAX_CONCURRENT_TRANSCRIPTIONS = int(os.getenv("MAX_CONCURRENT_TRANSCRIPTIONS", "3"))
MODEL_NUM_WORKERS = int(os.getenv("MODEL_NUM_WORKERS", str(MAX_CONCURRENT_TRANSCRIPTIONS)))
TMP_DIR = os.getenv("TMP_DIR", "/dev/shm" if Path("/dev/shm").exists() else "/tmp")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("stt-whisper-server")

app = FastAPI(title="STT Whisper Server", version="0.4.0")
model: Optional[WhisperModel] = None
transcribe_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TRANSCRIPTIONS)
transcribe_executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_TRANSCRIPTIONS)
active_transcriptions = 0


def require_token(x_stt_token: Optional[str]) -> None:
    if STT_API_TOKEN and x_stt_token != STT_API_TOKEN:
        raise HTTPException(status_code=401, detail="invalid token")


@app.on_event("startup")
def load_model() -> None:
    global model
    logger.info(
        "loading_model model_size=%s device=%s compute_type=%s language=%s beam_size=%s vad_filter=%s vad_min_silence_ms=%s max_concurrency=%s model_num_workers=%s tmp_dir=%s",
        MODEL_SIZE,
        DEVICE,
        COMPUTE_TYPE,
        LANGUAGE,
        BEAM_SIZE,
        VAD_FILTER,
        VAD_MIN_SILENCE_DURATION_MS,
        MAX_CONCURRENT_TRANSCRIPTIONS,
        MODEL_NUM_WORKERS,
        TMP_DIR,
    )
    started = time.monotonic()
    model = WhisperModel(
        MODEL_SIZE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        num_workers=MODEL_NUM_WORKERS,
    )
    logger.info("model_loaded model_size=%s load_seconds=%.3f", MODEL_SIZE, time.monotonic() - started)


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "model_size": MODEL_SIZE,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "language": LANGUAGE,
        "max_concurrent_transcriptions": MAX_CONCURRENT_TRANSCRIPTIONS,
        "model_num_workers": MODEL_NUM_WORKERS,
        "active_transcriptions": active_transcriptions,
        "tmp_dir": TMP_DIR,
        "persists_audio": False,
        "persists_transcripts": False,
    }


def run_transcription(tmp_path: Path, request_id: str, audio_bytes: int) -> dict:
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    kwargs = {
        "language": LANGUAGE,
        "beam_size": BEAM_SIZE,
    }
    if VAD_FILTER:
        kwargs["vad_filter"] = True
        kwargs["vad_parameters"] = {
            "min_silence_duration_ms": VAD_MIN_SILENCE_DURATION_MS,
        }

    logger.info(
        "transcribe_start request_id=%s model_size=%s device=%s compute_type=%s audio_bytes=%s",
        request_id,
        MODEL_SIZE,
        DEVICE,
        COMPUTE_TYPE,
        audio_bytes,
    )
    started = time.monotonic()
    segments, info = model.transcribe(str(tmp_path), **kwargs)
    text = "".join(segment.text for segment in segments).strip()
    elapsed = time.monotonic() - started
    logger.info(
        "transcribe_done request_id=%s model_size=%s audio_duration=%s text_chars=%s transcribe_seconds=%.3f",
        request_id,
        MODEL_SIZE,
        getattr(info, "duration", None),
        len(text),
        elapsed,
    )

    return {
        "text": text,
        "language": getattr(info, "language", LANGUAGE),
        "language_probability": getattr(info, "language_probability", None),
        "duration": getattr(info, "duration", None),
        "model_size": MODEL_SIZE,
        "transcribe_seconds": round(elapsed, 3),
    }


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    x_stt_token: Optional[str] = Header(default=None),
) -> dict:
    global active_transcriptions
    require_token(x_stt_token)

    request_id = uuid4().hex[:12]
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    tmp_dir = Path(TMP_DIR)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    received_started = time.monotonic()
    audio_bytes = 0
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=tmp_dir) as tmp:
        tmp_path = Path(tmp.name)
        while chunk := await audio.read(1024 * 1024):
            audio_bytes += len(chunk)
            tmp.write(chunk)
    logger.info(
        "request_received request_id=%s filename=%s audio_bytes=%s receive_seconds=%.3f model_size=%s",
        request_id,
        audio.filename,
        audio_bytes,
        time.monotonic() - received_started,
        MODEL_SIZE,
    )

    try:
        wait_started = time.monotonic()
        async with transcribe_semaphore:
            wait_seconds = time.monotonic() - wait_started
            active_transcriptions += 1
            logger.info(
                "request_admitted request_id=%s active_transcriptions=%s wait_seconds=%.3f max_concurrency=%s",
                request_id,
                active_transcriptions,
                wait_seconds,
                MAX_CONCURRENT_TRANSCRIPTIONS,
            )
            try:
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(transcribe_executor, run_transcription, tmp_path, request_id, audio_bytes)
            finally:
                active_transcriptions -= 1
                logger.info("request_finished request_id=%s active_transcriptions=%s", request_id, active_transcriptions)
    finally:
        # No server-side audio retention. File is temporary only and always deleted.
        tmp_path.unlink(missing_ok=True)
