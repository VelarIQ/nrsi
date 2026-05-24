"""Tests for the NRS Artifact Store — storage, signing, retrieval, token verification.

Uses a temporary directory for ARTIFACT_LOCAL_DIR so nothing touches the real
filesystem, and no external services are needed.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

# Patch env vars BEFORE importing the module so the store uses tmpdir
_ORIG_DIR = os.environ.get("ARTIFACT_LOCAL_DIR", "")
_ORIG_KEY = os.environ.get("ARTIFACT_SIGNING_KEY", "")


def _load_artifact_store(tmp_dir: str):
    """(Re-)import the module with ARTIFACT_LOCAL_DIR pointing at *tmp_dir*."""
    import importlib
    import importlib.util

    os.environ["ARTIFACT_LOCAL_DIR"] = tmp_dir
    os.environ["ARTIFACT_BACKEND"] = "local"

    mod_name = "artifact_store"
    spec = importlib.util.spec_from_file_location(
        mod_name,
        str(
            Path(__file__).resolve().parent.parent
            / "services"
            / "nrs-worker"
            / "artifact_store.py"
        ),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def store_mod(tmp_path):
    """Return the freshly-loaded artifact_store module using a tmpdir."""
    mod = _load_artifact_store(str(tmp_path))
    yield mod
    # Restore original env
    if _ORIG_DIR:
        os.environ["ARTIFACT_LOCAL_DIR"] = _ORIG_DIR
    else:
        os.environ.pop("ARTIFACT_LOCAL_DIR", None)


@pytest.fixture()
def store(store_mod):
    return store_mod.ArtifactStore()


@pytest.fixture()
def sample_data():
    return b"hello world this is test audio data for the artifact store"


# ---------------------------------------------------------------------------
# 1  store() creates a file and returns ArtifactRef
# ---------------------------------------------------------------------------

def test_store_creates_file(store, store_mod, sample_data, tmp_path):
    ref = store.store(sample_data, content_type="audio/wav")
    assert isinstance(ref, store_mod.ArtifactRef)
    matching = list(Path(tmp_path).glob(f"{ref.artifact_id}*"))
    assert len(matching) == 1, "Expected exactly one file written"


# ---------------------------------------------------------------------------
# 2  ArtifactRef has valid sha256 hash
# ---------------------------------------------------------------------------

def test_artifact_ref_sha256(store, sample_data):
    ref = store.store(sample_data, content_type="audio/wav")
    expected = hashlib.sha256(sample_data).hexdigest()
    assert ref.sha256 == expected


# ---------------------------------------------------------------------------
# 3  Correct content_type and size_bytes
# ---------------------------------------------------------------------------

def test_content_type_and_size(store, sample_data):
    ref = store.store(sample_data, content_type="image/png")
    assert ref.content_type == "image/png"
    assert ref.size_bytes == len(sample_data)


# ---------------------------------------------------------------------------
# 4  get_bytes() retrieves stored data
# ---------------------------------------------------------------------------

def test_get_bytes(store, sample_data):
    ref = store.store(sample_data, content_type="audio/wav")
    retrieved = store.get_bytes(ref.artifact_id)
    assert retrieved == sample_data


# ---------------------------------------------------------------------------
# 5  Signed URL contains expires and token
# ---------------------------------------------------------------------------

def test_signed_url_format(store, sample_data):
    ref = store.store(sample_data, content_type="audio/wav")
    assert "expires=" in ref.url
    assert "token=" in ref.url


# ---------------------------------------------------------------------------
# 6  verify_token() accepts valid token
# ---------------------------------------------------------------------------

def test_verify_valid_token(store, sample_data):
    ref = store.store(sample_data, content_type="audio/wav")
    url = ref.url
    qs = url.split("?", 1)[1]
    params = dict(p.split("=", 1) for p in qs.split("&"))
    filename = url.split("?")[0].rsplit("/", 1)[-1]
    assert store.verify_token(filename, int(params["expires"]), params["token"])


# ---------------------------------------------------------------------------
# 7  verify_token() rejects expired token
# ---------------------------------------------------------------------------

def test_reject_expired_token(store, sample_data):
    ref = store.store(sample_data, content_type="audio/wav")
    url = ref.url
    filename = url.split("?")[0].rsplit("/", 1)[-1]
    past_expires = int(time.time()) - 9999
    token = store._generate_token(filename, past_expires)
    assert store.verify_token(filename, past_expires, token) is False


# ---------------------------------------------------------------------------
# 8  verify_token() rejects tampered token
# ---------------------------------------------------------------------------

def test_reject_tampered_token(store, sample_data):
    ref = store.store(sample_data, content_type="audio/wav")
    url = ref.url
    qs = url.split("?", 1)[1]
    params = dict(p.split("=", 1) for p in qs.split("&"))
    filename = url.split("?")[0].rsplit("/", 1)[-1]
    tampered = "a" * 32
    assert store.verify_token(filename, int(params["expires"]), tampered) is False


# ---------------------------------------------------------------------------
# 9  _ext_for_content_type returns correct extensions
# ---------------------------------------------------------------------------

def test_ext_for_content_type(store_mod):
    fn = store_mod._ext_for_content_type
    assert fn("image/png") == ".png"
    assert fn("audio/wav") == ".wav"
    assert fn("video/mp4") == ".mp4"
    assert fn("application/pdf") == ".pdf"
    assert fn("application/octet-stream") == ".bin"


# ---------------------------------------------------------------------------
# 10  store with metadata preserves provenance
# ---------------------------------------------------------------------------

def test_store_preserves_metadata(store, sample_data):
    meta = {"source": "test-gen", "model": "nrs-audio-v1"}
    ref = store.store(sample_data, content_type="audio/wav", metadata=meta)
    assert ref.nrsi_provenance == meta


# ---------------------------------------------------------------------------
# 11  Unique artifact IDs across stores
# ---------------------------------------------------------------------------

def test_unique_ids(store):
    d1 = b"data_one"
    d2 = b"data_two"
    ref1 = store.store(d1, content_type="audio/wav")
    ref2 = store.store(d2, content_type="audio/wav")
    assert ref1.artifact_id != ref2.artifact_id


# ---------------------------------------------------------------------------
# 12  get_bytes returns None for unknown artifact
# ---------------------------------------------------------------------------

def test_get_bytes_unknown(store):
    assert store.get_bytes("nonexistent_artifact_id") is None
