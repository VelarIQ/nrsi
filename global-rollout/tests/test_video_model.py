"""Tests for the NRS Video Analysis Service — VideoAnalyzer, helpers, endpoints.

Exercises scene detection, motion analysis, ffprobe parsing, PVS-4 caching,
MEDIA_INPUT_GATE validation, nrsi_trusted wrapping, and the /healthz endpoint.
All subprocess and heavy-model calls are mocked.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from nrsi.core.types import NRSIData, TrustLevel, trusted as nrsi_trusted


# ---------------------------------------------------------------------------
# Import the video-model module from its hyphenated directory via importlib
# ---------------------------------------------------------------------------

def _load_video_model():
    """Load services/video-model/main.py while mocking heavy deps."""
    mock_transformers = MagicMock()
    mock_transformers.pipeline = MagicMock(return_value=MagicMock(return_value=[]))

    mod_name = "video_model_main"
    with patch.dict(sys.modules, {"transformers": mock_transformers}):
        spec = importlib.util.spec_from_file_location(
            mod_name,
            str(Path(__file__).resolve().parent.parent / "services" / "video-model" / "main.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
    return mod


_vm = _load_video_model()

VideoAnalyzer = _vm.VideoAnalyzer
_ffprobe_info = _vm._ffprobe_info
_video_input_not_empty = _vm._video_input_not_empty
MEDIA_INPUT_GATE = _vm.MEDIA_INPUT_GATE
_app = _vm.app

from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers — create deterministic synthetic frames on disk
# ---------------------------------------------------------------------------

def _save_solid_frame(path: str, rgb: tuple) -> str:
    img = Image.new("RGB", (320, 240), rgb)
    img.save(path, format="PNG")
    return path


@pytest.fixture()
def identical_frames(tmp_path):
    paths = []
    for i in range(3):
        p = str(tmp_path / f"frame_{i:06d}.png")
        _save_solid_frame(p, (100, 100, 100))
        paths.append(p)
    return paths


@pytest.fixture()
def contrasting_frames(tmp_path):
    colours = [(0, 0, 0), (255, 255, 255), (0, 0, 0)]
    paths = []
    for i, c in enumerate(colours):
        p = str(tmp_path / f"frame_{i:06d}.png")
        _save_solid_frame(p, c)
        paths.append(p)
    return paths


@pytest.fixture()
def analyzer():
    mock_transformers = MagicMock()
    mock_transformers.pipeline = MagicMock(return_value=MagicMock(return_value=[]))
    with patch.dict(sys.modules, {"transformers": mock_transformers}):
        return VideoAnalyzer()


# ---------------------------------------------------------------------------
# 1-3  Scene Detection
# ---------------------------------------------------------------------------

class TestSceneDetection:
    def test_empty_keyframes_returns_empty(self, analyzer):
        assert analyzer.detect_scenes([], fps=30.0, interval=30) == []

    def test_identical_frames_single_scene(self, analyzer, identical_frames):
        scenes = analyzer.detect_scenes(identical_frames, fps=30.0, interval=30)
        assert len(scenes) == 1
        assert scenes[0]["scene_index"] == 0

    def test_contrasting_frames_detect_boundary(self, analyzer, contrasting_frames):
        scenes = analyzer.detect_scenes(
            contrasting_frames, fps=30.0, interval=30, threshold=0.05
        )
        assert len(scenes) >= 2, "Should detect at least one scene boundary"

    def test_max_scenes_respected(self, analyzer, contrasting_frames):
        scenes = analyzer.detect_scenes(
            contrasting_frames, fps=30.0, interval=30, max_scenes=1, threshold=0.01
        )
        assert len(scenes) <= 2


# ---------------------------------------------------------------------------
# 4-6  Motion Analysis
# ---------------------------------------------------------------------------

class TestMotionAnalysis:
    def test_static_frames(self, analyzer, identical_frames):
        result = analyzer.analyze_motion(identical_frames, interval=30, fps=30.0)
        assert result["overall_motion_level"] == "static"
        assert result["trend"] == "stable"

    def test_high_motion_frames(self, analyzer, contrasting_frames):
        result = analyzer.analyze_motion(contrasting_frames, interval=30, fps=30.0)
        assert result["overall_motion_level"] in ("medium", "high")

    def test_single_frame_returns_static(self, analyzer, tmp_path):
        p = str(tmp_path / "solo.png")
        _save_solid_frame(p, (128, 128, 128))
        result = analyzer.analyze_motion([p], interval=30, fps=30.0)
        assert result["overall_motion_level"] == "static"
        assert result["motion_timeline"] == []


# ---------------------------------------------------------------------------
# 7-8  ffprobe helper (subprocess mocked)
# ---------------------------------------------------------------------------

class TestFfprobeInfo:
    def test_parses_mock_output(self):
        fake_json = json.dumps({
            "streams": [
                {
                    "codec_type": "video",
                    "width": 1920, "height": 1080,
                    "r_frame_rate": "30/1", "codec_name": "h264",
                },
                {"codec_type": "audio", "codec_name": "aac"},
            ],
            "format": {"duration": "120.5", "size": "50000000"},
        })
        mock_result = MagicMock(returncode=0, stdout=fake_json, stderr="")
        with patch("subprocess.run", return_value=mock_result):
            info = _ffprobe_info("/fake/video.mp4")
        assert info["width"] == 1920
        assert info["height"] == 1080
        assert info["fps"] == 30.0
        assert info["has_audio"] is True
        assert info["codec"] == "h264"

    def test_ffprobe_failure_returns_empty(self):
        mock_result = MagicMock(returncode=1, stdout="", stderr="error")
        with patch("subprocess.run", return_value=mock_result):
            info = _ffprobe_info("/nonexistent")
        assert info == {}


# ---------------------------------------------------------------------------
# 9-10  PVS-4 Caching
# ---------------------------------------------------------------------------

class TestPVS4Caching:
    def test_cache_store_and_hit(self, analyzer):
        key = analyzer.cache_key(b"video_bytes", "reqhash")
        analyzer.cache_store(key, {"scenes": [1, 2, 3]})
        cached = analyzer.cache_lookup(key)
        assert cached is not None
        assert cached["scenes"] == [1, 2, 3]

    def test_cache_miss(self, analyzer):
        result = analyzer.cache_lookup("nonexistent_key")
        assert result is None


# ---------------------------------------------------------------------------
# 11-13  MEDIA_INPUT_GATE
# ---------------------------------------------------------------------------

class TestMediaInputGate:
    def test_rejects_empty_input(self):
        assert _video_input_not_empty({"video_data": "", "video_url": None}) is False

    def test_accepts_video_data(self):
        assert _video_input_not_empty({"video_data": "abc"}) is True

    def test_accepts_video_url(self):
        assert _video_input_not_empty({"video_url": "https://example.com/v.mp4"}) is True


# ---------------------------------------------------------------------------
# 14  nrsi_trusted wrapping
# ---------------------------------------------------------------------------

class TestNRSITrusted:
    def test_trusted_fields(self):
        output = nrsi_trusted(
            value={"hello": "world"},
            confidence=0.9,
            gate_name="test_gate",
        )
        assert isinstance(output, NRSIData)
        assert output.trust_level == TrustLevel.TRUSTED
        assert 0.0 <= output.confidence <= 1.0
        assert len(output.provenance) >= 1


# ---------------------------------------------------------------------------
# 15  /healthz endpoint
# ---------------------------------------------------------------------------

class TestHealthzEndpoint:
    def test_healthz_returns_200(self):
        client = TestClient(_app, raise_server_exceptions=False)
        resp = client.get("/healthz")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["service"] == "video-model"
