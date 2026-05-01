import asyncio
import gc
import logging
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from faster_whisper import WhisperModel

LEGACY_MODEL_SIZE = os.getenv("MODEL_SIZE", "base")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", LEGACY_MODEL_SIZE)
ALLOWED_MODELS = {
    item.strip()
    for item in os.getenv(
        "ALLOWED_MODELS",
        "tiny,tiny.en,base,base.en,small,small.en,medium,medium.en,large-v1,large-v2,large-v3,large,distil-small.en,distil-medium.en,distil-large-v2,distil-large-v3,distil-large-v3.5,large-v3-turbo,turbo",
    ).split(",")
    if item.strip()
}
PRELOAD_MODEL = os.getenv("PRELOAD_MODEL", "").strip()
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

app = FastAPI(title="STT Whisper Server", version="0.5.0")
loaded_model: Optional[WhisperModel] = None
loaded_model_name: Optional[str] = None
switch_target: Optional[str] = None
active_transcriptions = 0
model_condition = asyncio.Condition()
transcribe_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TRANSCRIPTIONS)
transcribe_executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_TRANSCRIPTIONS)


def require_token(x_stt_token: Optional[str]) -> None:
    if STT_API_TOKEN and x_stt_token != STT_API_TOKEN:
        raise HTTPException(status_code=401, detail="invalid token")


def normalize_model(requested_model: Optional[str]) -> str:
    model_name = (requested_model or DEFAULT_MODEL).strip()
    if model_name not in ALLOWED_MODELS:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "model not allowed",
                "requested_model": model_name,
                "allowed_models": sorted(ALLOWED_MODELS),
            },
        )
    return model_name


def create_model(model_name: str) -> WhisperModel:
    logger.info(
        "loading_model model_size=%s device=%s compute_type=%s language=%s beam_size=%s vad_filter=%s vad_min_silence_ms=%s max_concurrency=%s model_num_workers=%s tmp_dir=%s",
        model_name,
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
        model_name,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        num_workers=MODEL_NUM_WORKERS,
    )
    logger.info("model_loaded model_size=%s load_seconds=%.3f", model_name, time.monotonic() - started)
    return model


@app.on_event("startup")
async def startup() -> None:
    logger.info(
        "server_start default_model=%s preload_model=%s allowed_models=%s",
        DEFAULT_MODEL,
        PRELOAD_MODEL or "none",
        ",".join(sorted(ALLOWED_MODELS)),
    )
    if PRELOAD_MODEL:
        model_name = normalize_model(PRELOAD_MODEL)
        await switch_to_model(model_name, request_id="startup", count_as_active=False)


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "default_model": DEFAULT_MODEL,
        "loaded_model": loaded_model_name,
        "switch_target": switch_target,
        "allowed_models": sorted(ALLOWED_MODELS),
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


async def switch_to_model(model_name: str, request_id: str, count_as_active: bool = True) -> WhisperModel:
    global active_transcriptions, loaded_model, loaded_model_name, switch_target

    while True:
        async with model_condition:
            if loaded_model is not None and loaded_model_name == model_name and switch_target is None:
                if count_as_active:
                    active_transcriptions += 1
                logger.info(
                    "model_reuse request_id=%s model_size=%s active_transcriptions=%s",
                    request_id,
                    model_name,
                    active_transcriptions,
                )
                return loaded_model

            if switch_target is not None:
                logger.info(
                    "model_switch_wait request_id=%s requested_model=%s loaded_model=%s switch_target=%s active_transcriptions=%s",
                    request_id,
                    model_name,
                    loaded_model_name,
                    switch_target,
                    active_transcriptions,
                )
                await model_condition.wait()
                continue

            switch_target = model_name
            while active_transcriptions > 0:
                logger.info(
                    "model_switch_wait_active request_id=%s requested_model=%s loaded_model=%s active_transcriptions=%s",
                    request_id,
                    model_name,
                    loaded_model_name,
                    active_transcriptions,
                )
                await model_condition.wait()
            old_model = loaded_model_name
            loaded_model = None
            loaded_model_name = None
            gc.collect()
            logger.info("model_unloaded request_id=%s old_model=%s", request_id, old_model)
            break

    logger.info("model_switch_start request_id=%s from_model=%s to_model=%s", request_id, old_model, model_name)
    new_model = await asyncio.to_thread(create_model, model_name)

    async with model_condition:
        loaded_model = new_model
        loaded_model_name = model_name
        switch_target = None
        if count_as_active:
            active_transcriptions += 1
        logger.info(
            "model_switch_done request_id=%s from_model=%s to_model=%s active_transcriptions=%s",
            request_id,
            old_model,
            model_name,
            active_transcriptions,
        )
        model_condition.notify_all()
        return loaded_model


async def release_model(request_id: str) -> None:
    global active_transcriptions
    async with model_condition:
        active_transcriptions = max(0, active_transcriptions - 1)
        logger.info("request_finished request_id=%s active_transcriptions=%s", request_id, active_transcriptions)
        model_condition.notify_all()


def run_transcription(model_ref: WhisperModel, model_name: str, tmp_path: Path, request_id: str, audio_bytes: int) -> dict:
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
        model_name,
        DEVICE,
        COMPUTE_TYPE,
        audio_bytes,
    )
    started = time.monotonic()
    segments, info = model_ref.transcribe(str(tmp_path), **kwargs)
    text = "".join(segment.text for segment in segments).strip()
    elapsed = time.monotonic() - started
    logger.info(
        "transcribe_done request_id=%s model_size=%s audio_duration=%s text_chars=%s transcribe_seconds=%.3f",
        request_id,
        model_name,
        getattr(info, "duration", None),
        len(text),
        elapsed,
    )

    return {
        "text": text,
        "language": getattr(info, "language", LANGUAGE),
        "language_probability": getattr(info, "language_probability", None),
        "duration": getattr(info, "duration", None),
        "model_size": model_name,
        "transcribe_seconds": round(elapsed, 3),
    }


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    model: Optional[str] = Form(default=None),
    x_stt_token: Optional[str] = Header(default=None),
) -> dict:
    require_token(x_stt_token)
    requested_model = normalize_model(model)

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
        "request_received request_id=%s filename=%s audio_bytes=%s receive_seconds=%.3f requested_model=%s loaded_model=%s",
        request_id,
        audio.filename,
        audio_bytes,
        time.monotonic() - received_started,
        requested_model,
        loaded_model_name,
    )

    try:
        wait_started = time.monotonic()
        async with transcribe_semaphore:
            wait_seconds = time.monotonic() - wait_started
            model_ref = await switch_to_model(requested_model, request_id=request_id)
            logger.info(
                "request_admitted request_id=%s requested_model=%s loaded_model=%s active_transcriptions=%s wait_seconds=%.3f max_concurrency=%s",
                request_id,
                requested_model,
                loaded_model_name,
                active_transcriptions,
                wait_seconds,
                MAX_CONCURRENT_TRANSCRIPTIONS,
            )
            try:
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(
                    transcribe_executor,
                    run_transcription,
                    model_ref,
                    requested_model,
                    tmp_path,
                    request_id,
                    audio_bytes,
                )
            finally:
                await release_model(request_id)
    finally:
        # No server-side audio retention. File is temporary only and always deleted.
        tmp_path.unlink(missing_ok=True)
