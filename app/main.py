import os
import tempfile
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

app = FastAPI(title="STT Whisper Server", version="0.1.0")
model: Optional[WhisperModel] = None


def require_token(x_stt_token: Optional[str]) -> None:
    if STT_API_TOKEN and x_stt_token != STT_API_TOKEN:
        raise HTTPException(status_code=401, detail="invalid token")


@app.on_event("startup")
def load_model() -> None:
    global model
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "model_size": MODEL_SIZE,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "language": LANGUAGE,
    }


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    x_stt_token: Optional[str] = Header(default=None),
) -> dict:
    require_token(x_stt_token)

    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        while chunk := await audio.read(1024 * 1024):
            tmp.write(chunk)

    try:
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

        segments, info = model.transcribe(str(tmp_path), **kwargs)
        text = "".join(segment.text for segment in segments).strip()

        return {
            "text": text,
            "language": getattr(info, "language", LANGUAGE),
            "language_probability": getattr(info, "language_probability", None),
            "duration": getattr(info, "duration", None),
            "model_size": MODEL_SIZE,
        }
    finally:
        tmp_path.unlink(missing_ok=True)
