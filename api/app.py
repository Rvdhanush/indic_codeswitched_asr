"""
FastAPI inference endpoint for Tamil-English code-switched ASR.

Endpoints:
    GET  /health           — liveness check
    POST /transcribe       — transcribe uploaded audio
    POST /analyze          — transcribe + failure analysis (requires reference)
    GET  /model/info       — loaded model metadata

Usage:
    uvicorn api.app:app --host 0.0.0.0 --port 8000

Environment:
    MODEL_PATH   — path to fine-tuned checkpoint (default: checkpoints/best_model)
    HF_TOKEN     — HuggingFace token (needed if base model is gated)
    BASE_MODEL   — base Whisper model ID (default: openai/whisper-small)
"""

import io
import os
import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import librosa
import soundfile as sf
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

from evaluation.metrics import analyze_failures, tag_segment_type

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_PATH = Path(os.getenv("MODEL_PATH", "checkpoints/best_model"))
BASE_MODEL = os.getenv("BASE_MODEL", "openai/whisper-small")
HF_TOKEN = os.getenv("HF_TOKEN")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16_000

# ---------------------------------------------------------------------------
# Model loader (singleton, loaded at startup)
# ---------------------------------------------------------------------------

_processor: Optional[WhisperProcessor] = None
_model: Optional[WhisperForConditionalGeneration] = None
_model_source: str = "not loaded"


def _load_model():
    global _processor, _model, _model_source

    if MODEL_PATH.exists():
        logger.info(f"Loading fine-tuned model from {MODEL_PATH}")
        _processor = WhisperProcessor.from_pretrained(
            str(MODEL_PATH), token=HF_TOKEN
        )
        base = WhisperForConditionalGeneration.from_pretrained(
            BASE_MODEL,
            token=HF_TOKEN,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        )
        _model = PeftModel.from_pretrained(base, str(MODEL_PATH))
        _model_source = f"fine-tuned LoRA ({MODEL_PATH})"
    else:
        logger.warning(
            f"Fine-tuned checkpoint not found at {MODEL_PATH}. "
            f"Falling back to base model: {BASE_MODEL}"
        )
        _processor = WhisperProcessor.from_pretrained(
            BASE_MODEL, token=HF_TOKEN
        )
        _model = WhisperForConditionalGeneration.from_pretrained(
            BASE_MODEL,
            token=HF_TOKEN,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        )
        _model_source = f"base model fallback ({BASE_MODEL})"

    _model.config.forced_decoder_ids = None
    _model.config.suppress_tokens = []
    _model = _model.to(DEVICE)
    _model.eval()
    logger.info(f"Model ready on {DEVICE}: {_model_source}")


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _load_audio(data: bytes, filename: str) -> np.ndarray:
    """
    Load audio bytes → mono float32 array at SAMPLE_RATE.
    Handles WAV, MP3, FLAC, OGG via librosa.
    """
    suffix = Path(filename).suffix.lower() or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        audio, sr = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return audio.astype(np.float32)


def _transcribe(audio: np.ndarray) -> str:
    inputs = _processor(
        audio, sampling_rate=SAMPLE_RATE, return_tensors="pt"
    ).input_features.to(DEVICE)

    if DEVICE == "cuda":
        inputs = inputs.half()

    with torch.no_grad():
        predicted_ids = _model.generate(
            inputs,
            language="ta",
            task="transcribe",
            max_new_tokens=256,
        )

    return _processor.batch_decode(
        predicted_ids, skip_special_tokens=True
    )[0].strip()


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class TranscribeResponse(BaseModel):
    transcript: str
    segment_type: str
    model_source: str
    device: str


class AnalyzeResponse(BaseModel):
    transcript: str
    reference: str
    segment_type: str
    wer: float
    cer: float
    switch_boundary_count: int
    failure_type: str
    reference_word_count: int
    hypothesis_word_count: int
    model_source: str
    device: str


class ModelInfoResponse(BaseModel):
    model_source: str
    base_model: str
    device: str
    model_path: str


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Indic Code-Switched ASR",
    description="Tamil-English code-switched speech recognition with failure analysis",
    version="1.0.0",
)


@app.on_event("startup")
def startup():
    _load_model()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    return ModelInfoResponse(
        model_source=_model_source,
        base_model=BASE_MODEL,
        device=DEVICE,
        model_path=str(MODEL_PATH),
    )


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(audio: UploadFile = File(...)):
    """
    Transcribe an uploaded audio file.

    Accepts: WAV, MP3, FLAC, OGG (any format librosa can read)
    Returns: transcript, detected segment type, model metadata
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio file")

    try:
        waveform = _load_audio(raw, audio.filename or "audio.wav")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not decode audio: {e}")

    transcript = _transcribe(waveform)
    segment_type = tag_segment_type(transcript)

    return TranscribeResponse(
        transcript=transcript,
        segment_type=segment_type,
        model_source=_model_source,
        device=DEVICE,
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    audio: UploadFile = File(...),
    reference: str = Form(...),
):
    """
    Transcribe audio and run full failure analysis against a reference transcript.

    Accepts: audio file + reference transcript (form field)
    Returns: transcript, WER/CER, failure category, segment type
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio file")

    try:
        waveform = _load_audio(raw, audio.filename or "audio.wav")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not decode audio: {e}")

    transcript = _transcribe(waveform)
    metrics = analyze_failures(reference, transcript)

    return AnalyzeResponse(
        transcript=transcript,
        reference=reference,
        segment_type=metrics["segment_type"],
        wer=metrics["wer"],
        cer=metrics["cer"],
        switch_boundary_count=metrics["switch_boundary_count"],
        failure_type=metrics["failure_type"],
        reference_word_count=metrics["reference_word_count"],
        hypothesis_word_count=metrics["hypothesis_word_count"],
        model_source=_model_source,
        device=DEVICE,
    )
