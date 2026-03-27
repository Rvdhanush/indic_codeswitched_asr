"""
Tests for api/app.py

Uses FastAPI TestClient with the model loading patched out —
no GPU, no HuggingFace downloads required.
"""

import io
import json
import numpy as np
import pytest
import soundfile as sf
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav_bytes(duration_s: float = 1.0, sr: int = 16_000) -> bytes:
    """Generate a minimal valid WAV file in memory."""
    samples = np.zeros(int(sr * duration_s), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV")
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# App fixture — patches model loading so tests run without weights
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """
    Returns a TestClient with:
    - _load_model() no-opped (sets a mock model + processor)
    - _transcribe() returning a fixed string
    """
    with patch("api.app._load_model") as mock_load:
        def fake_load():
            import api.app as app_module
            app_module._processor = MagicMock()
            app_module._model = MagicMock()
            app_module._model_source = "mock-model (test)"
            # Make generate() return tensor-like; batch_decode returns transcript
            app_module._processor.batch_decode.return_value = [
                "vanakkam this is a test"
            ]

        mock_load.side_effect = fake_load

        with patch("api.app._transcribe", return_value="vanakkam this is a test"):
            from api.app import app
            with TestClient(app) as c:
                yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_status_ok(self, client):
        resp = client.get("/health")
        assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# /model/info
# ---------------------------------------------------------------------------

class TestModelInfo:
    def test_returns_200(self, client):
        resp = client.get("/model/info")
        assert resp.status_code == 200

    def test_required_fields(self, client):
        data = client.get("/model/info").json()
        for field in ("model_source", "base_model", "device", "model_path"):
            assert field in data


# ---------------------------------------------------------------------------
# /transcribe
# ---------------------------------------------------------------------------

class TestTranscribe:
    def test_valid_wav_returns_200(self, client):
        wav = _make_wav_bytes()
        resp = client.post(
            "/transcribe",
            files={"audio": ("test.wav", wav, "audio/wav")},
        )
        assert resp.status_code == 200

    def test_response_has_transcript(self, client):
        wav = _make_wav_bytes()
        data = client.post(
            "/transcribe",
            files={"audio": ("test.wav", wav, "audio/wav")},
        ).json()
        assert "transcript" in data
        assert isinstance(data["transcript"], str)

    def test_response_has_segment_type(self, client):
        wav = _make_wav_bytes()
        data = client.post(
            "/transcribe",
            files={"audio": ("test.wav", wav, "audio/wav")},
        ).json()
        assert data["segment_type"] in (
            "monolingual_tamil", "monolingual_english", "code_switched"
        )

    def test_response_has_model_source(self, client):
        wav = _make_wav_bytes()
        data = client.post(
            "/transcribe",
            files={"audio": ("test.wav", wav, "audio/wav")},
        ).json()
        assert "model_source" in data

    def test_empty_file_returns_400(self, client):
        resp = client.post(
            "/transcribe",
            files={"audio": ("empty.wav", b"", "audio/wav")},
        )
        assert resp.status_code == 400

    def test_corrupt_audio_returns_422(self, client):
        resp = client.post(
            "/transcribe",
            files={"audio": ("bad.wav", b"not audio data at all", "audio/wav")},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /analyze
# ---------------------------------------------------------------------------

class TestAnalyze:
    def test_valid_request_returns_200(self, client):
        wav = _make_wav_bytes()
        resp = client.post(
            "/analyze",
            files={"audio": ("test.wav", wav, "audio/wav")},
            data={"reference": "vanakkam this is a test"},
        )
        assert resp.status_code == 200

    def test_response_has_wer(self, client):
        wav = _make_wav_bytes()
        data = client.post(
            "/analyze",
            files={"audio": ("test.wav", wav, "audio/wav")},
            data={"reference": "vanakkam this is a test"},
        ).json()
        assert "wer" in data
        assert 0.0 <= data["wer"] <= 1.0

    def test_response_has_failure_type(self, client):
        wav = _make_wav_bytes()
        data = client.post(
            "/analyze",
            files={"audio": ("test.wav", wav, "audio/wav")},
            data={"reference": "vanakkam this is a test"},
        ).json()
        valid = {
            "SUBSTITUTION_SWITCH", "DELETION_PROPER_NOUN",
            "SUBSTITUTION_NUMBER", "LANGUAGE_CONFUSION", "INSERTION_FILLER",
        }
        assert data["failure_type"] in valid

    def test_response_mirrors_reference(self, client):
        wav = _make_wav_bytes()
        ref = "vanakkam this is a test"
        data = client.post(
            "/analyze",
            files={"audio": ("test.wav", wav, "audio/wav")},
            data={"reference": ref},
        ).json()
        assert data["reference"] == ref

    def test_perfect_match_gives_zero_wer(self, client):
        """When transcript == reference, WER should be 0."""
        wav = _make_wav_bytes()
        ref = "vanakkam this is a test"  # matches mock _transcribe return value
        data = client.post(
            "/analyze",
            files={"audio": ("test.wav", wav, "audio/wav")},
            data={"reference": ref},
        ).json()
        assert data["wer"] == 0.0

    def test_missing_reference_returns_422(self, client):
        wav = _make_wav_bytes()
        resp = client.post(
            "/analyze",
            files={"audio": ("test.wav", wav, "audio/wav")},
            # no reference field
        )
        assert resp.status_code == 422
