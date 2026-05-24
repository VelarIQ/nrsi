"""Tests for the NRS Audio Generation Service — NRSIAudioBackend.

Exercises PVS-4 cache keys, safety gates, request model validation,
audio validation, /healthz, and /v1/feedback validation.
The nrs_synthesis module is mocked since it may not be installed.
"""

from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _load_audio_gen():
    mock_nrs_synthesis = MagicMock()
    mock_nrs_synthesis.SR = 48000

    _extra_mocks = {
        "nrs_synthesis": mock_nrs_synthesis,
        "scipy": MagicMock(),
        "scipy.signal": MagicMock(),
        "wgpu": MagicMock(),
    }

    with patch.dict(sys.modules, _extra_mocks):
        spec = importlib.util.spec_from_file_location(
            "audio_gen_main",
            str(
                Path(__file__).resolve().parent.parent
                / "services"
                / "audio-gen"
                / "main.py"
            ),
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["audio_gen_main"] = mod
        spec.loader.exec_module(mod)
    return mod


_ag = _load_audio_gen()

NRSIAudioBackend = _ag.NRSIAudioBackend
TTSRequest = _ag.TTSRequest
MusicGenRequest = _ag.MusicGenRequest
SFXRequest = _ag.SFXRequest
FeedbackRequest = _ag.FeedbackRequest
GEN_SAFETY_GATE = _ag.GEN_SAFETY_GATE
ValidationError = _ag.ValidationError
nrsi_raw = _ag.nrsi_raw
_app = _ag.app

from fastapi.testclient import TestClient


def _make_wav(samples: np.ndarray, sr: int = 22050) -> bytes:
    int_samples = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
    data = int_samples.tobytes()
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + len(data), b"WAVE",
        b"fmt ", 16, 1, 1, sr, sr * 2, 2, 16,
        b"data", len(data),
    )
    return header + data


@pytest.fixture()
def backend():
    be = NRSIAudioBackend()
    be.loaded = True
    return be


def test_audio_validation_short(backend):
    short_bytes = b"RIFF" + b"\x00" * 40
    passed, score = backend._validate_audio(short_bytes)
    assert score <= 0.5


def test_audio_validation_good_wav(backend):
    t = np.linspace(0, 2, 22050 * 2, dtype=np.float32)
    good = np.sin(2 * np.pi * 440 * t) * 0.5
    wav = _make_wav(good)
    passed, score = backend._validate_audio(wav)
    assert passed is True
    assert score > 0


def test_pvs_cache_key_deterministic(backend):
    sig1 = backend._pvs.signature("tts:hello:default:wav")
    sig2 = backend._pvs.signature("tts:hello:default:wav")
    assert sig1 == sig2

    sig_diff = backend._pvs.signature("tts:goodbye:default:wav")
    assert sig1 != sig_diff


def test_safety_gate_rejects_empty():
    raw_input = nrsi_raw({"prompt": ""})
    with pytest.raises(ValidationError):
        GEN_SAFETY_GATE.process(raw_input)


def test_safety_gate_passes_valid():
    raw_input = nrsi_raw({"prompt": "A calm piano melody"})
    result = GEN_SAFETY_GATE.process(raw_input)
    assert result.is_validated


def test_tts_request_defaults():
    req = TTSRequest(text="Hello world")
    assert req.voice == "default"
    assert req.speed == 1.0
    assert req.language == "en"
    assert req.format == "wav"


def test_musicgen_request_defaults():
    req = MusicGenRequest(prompt="Epic orchestral")
    assert req.duration_seconds == 10.0
    assert req.temperature == 1.0


def test_sfx_request_defaults():
    req = SFXRequest(prompt="Thunder clap")
    assert req.duration_seconds == 5.0


def test_healthz_returns_200():
    client = TestClient(_app, raise_server_exceptions=False)
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"


def test_feedback_request_valid():
    req = FeedbackRequest(job_id="abc123", quality_rating=4)
    assert req.quality_rating == 4


def test_feedback_request_rejects_out_of_range():
    with pytest.raises(Exception):
        FeedbackRequest(job_id="abc123", quality_rating=0)
    with pytest.raises(Exception):
        FeedbackRequest(job_id="abc123", quality_rating=6)
