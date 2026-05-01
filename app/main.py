import asyncio
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

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

app = FastAPI(title="STT Whisper Server", version="0.3.0")
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
    model = WhisperModel(
        MODEL_SIZE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        num_workers=MODEL_NUM_WORKERS,
    )


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


def run_transcription(tmp_path: Path) -> dict:
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

    started = time.monotonic()
    segments, info = model.transcribe(str(tmp_path), **kwargs)
    text = "".join(segment.text for segment in segments).strip()
    elapsed = time.monotonic() - started

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

    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    tmp_dir = Path(TMP_DIR)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=tmp_dir) as tmp:
        tmp_path = Path(tmp.name)
        while chunk := await audio.read(1024 * 1024):
            tmp.write(chunk)

    try:
        async with transcribe_semaphore:
            active_transcriptions += 1
            try:
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(transcribe_executor, run_transcription, tmp_path)
            finally:
                active_transcriptions -= 1
    finally:
        # No server-side audio retention. File is temporary only and always deleted.
        tmp_path.unlink(missing_ok=True)
