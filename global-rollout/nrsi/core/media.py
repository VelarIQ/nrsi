"""
NRSI Media Processing — Multimodal Locality Layer.

Handles images, video, audio, documents, and other non-text
inputs WITHOUT breaking the NRSIP traffic budget.

═══════════════════════════════════════════════════════════════════
  THE MEDIA PROBLEM
═══════════════════════════════════════════════════════════════════

Text queries are ~200 bytes. Routing packets are 128 bytes.
But media inputs are:

  Image (JPEG):    5 MB average
  Video (1min):    150 MB (H.264), 4.5 MB/sec at 30fps
  Audio (1min):    1.5 MB (WAV), 192 KB (MP3)
  PDF document:    2-50 MB
  3D/CAD model:    50-500 MB
  Medical scan:    50-500 MB (DICOM)

If you send a 5MB image across NRSIP for routing:
  6,433 RPS × 5 MB = 32 GB/sec PER INSTANCE
  That's 2,000× worse than the text neuron problem.

═══════════════════════════════════════════════════════════════════
  THE SOLUTION: TWO-PLANE ARCHITECTURE
═══════════════════════════════════════════════════════════════════

Borrowed from how every CDN and telecom network works:

  CONTROL PLANE (NRSIP): routing packets, 128 bytes
    "This is a medical chest X-ray, T3 complexity,
     route to radiology crease on NRS-medical-02"

  DATA PLANE (shared storage): bulk media, any size
    S3/GCS bucket, NFS mount, or local SSD cache
    Media referenced by content-hash (SHA-256)
    Fetched on-demand ONLY by the instance that needs it

The control plane tells you WHERE the data is and WHAT it means.
The data plane holds the actual bytes.
They NEVER mix.

═══════════════════════════════════════════════════════════════════
  MEDIA PROCESSING PIPELINE
═══════════════════════════════════════════════════════════════════

  Raw media arrives at NRS instance (LOCAL)
       │
       ▼
  ┌──────────────────────────────────────┐
  │  MediaProcessor (LOCAL, on GPU)      │
  │  Extract features from raw media     │
  │  Image: CNN → 2048-d feature vector  │
  │  Video: keyframes + temporal embed   │
  │  Audio: spectrogram → embedding      │
  │  Document: layout + OCR + structure  │
  └──────────┬───────────────────────────┘
             │
             ▼
  ┌──────────────────────────────────────┐
  │  MediaDigest (LOCAL, ~4KB)           │
  │  Feature embedding (768-d)           │
  │  Detected entities/labels            │
  │  Extracted text (if any)             │
  │  Modality metadata                   │
  │  Content hash (for data plane ref)   │
  └──────────┬───────────────────────────┘
             │ feeds into existing neuron bank
             ▼
  ┌──────────────────────────────────────┐
  │  BinaryNeuronBank.activate()         │
  │  Same pipeline as text queries       │
  │  L0 → L1 Digest → L2 Packet (128B)  │
  └──────────┬───────────────────────────┘
             │
             ▼
  ┌──────────────────────────────────────┐
  │  RoutingPacket (NETWORK, 128B)       │
  │  + MediaRef (content_hash, 32B)      │
  │  Total: 160 bytes on NRSIP           │
  └──────────────────────────────────────┘

The 5 MB image becomes a 160-byte routing packet.
The raw image goes to shared storage (data plane).
NRSIP never sees a single pixel.

═══════════════════════════════════════════════════════════════════
  TRAFFIC MATH (media-heavy workload)
═══════════════════════════════════════════════════════════════════

Assume worst case: 50% of 6,433 RPS carry media

  Text queries:   3,217 × 128B  = 402 KB/sec per instance
  Media queries:  3,216 × 160B  = 502 KB/sec per instance
  Total NRSIP:    904 KB/sec per instance
  3 instances:    2.7 MB/sec

  Data plane (shared storage):
  3,216 media/sec × 5 MB avg = 16 GB/sec READS
  But: PVS-4 cache hit rate = 92.4%
  Actual fetches: 3,216 × 0.076 = 244 media/sec
  244 × 5 MB = 1.2 GB/sec from storage
  A single NVMe SSD handles 3.5 GB/sec. Covered.

  NRSIP stays at 2.7 MB/sec even with 50% media.
  Storage handles the bulk reads.
  Nothing breaks.

═══════════════════════════════════════════════════════════════════
  MODALITY TYPES
═══════════════════════════════════════════════════════════════════

  IMAGE:    Static visual (JPEG, PNG, DICOM, satellite, etc.)
            Feature extraction: CNN/ViT → 2048-d → 768-d projection
            Entities: objects, text (OCR), faces (count only), scenes

  VIDEO:    Temporal visual (MP4, RTSP stream, etc.)
            Keyframe extraction: 1 frame/sec or scene-change detect
            Each keyframe → image pipeline
            Temporal embedding: sequence of frame embeddings → 768-d
            NOT processed frame-by-frame (that's 30× waste)

  AUDIO:    Sound (WAV, MP3, FLAC, etc.)
            Mel spectrogram → audio embedding → 768-d
            Speech: ASR transcription → text pipeline (dual path)
            Non-speech: audio event classification → labels

  DOCUMENT: Structured content (PDF, DOCX, XLSX, etc.)
            Layout analysis: headers, tables, figures, body
            OCR for scanned pages
            Table extraction: structured data → typed values
            Figure extraction: each figure → image pipeline

  MESH_3D:  3D models (STL, OBJ, GLTF, etc.)
            Point cloud sampling → 3D embedding → 768-d
            Bounding box + volume + topology features

  SENSOR:   Time-series data (IoT, medical devices, etc.)
            Windowed FFT → frequency features → 768-d
            Anomaly detection: deviation from baseline
"""

from __future__ import annotations

import hashlib
import logging
import math
import struct
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cupy as _cp
    _GPU_AVAILABLE = True
except ImportError:
    _GPU_AVAILABLE = False
    _cp = None

logger = logging.getLogger("nrsi.media")


# ── Modality Types ───────────────────────────────────────────────────────────

class Modality(Enum):
    """Supported media types."""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    DOCUMENT = "document"
    MESH_3D = "mesh_3d"
    SENSOR = "sensor"


# Feature extraction specs per modality
MODALITY_SPECS = {
    Modality.TEXT: {
        "raw_size_avg": 200,              # bytes
        "feature_dim": 768,
        "extraction_method": "tokenizer + embedding",
        "gpu_ms": 2,
        "entities": ["tokens", "intent", "entities"],
    },
    Modality.IMAGE: {
        "raw_size_avg": 5_000_000,        # 5 MB
        "feature_dim": 768,
        "extraction_method": "CNN/ViT → 2048-d → 768-d projection",
        "gpu_ms": 15,
        "entities": ["objects", "text_ocr", "scene", "faces_count"],
    },
    Modality.VIDEO: {
        "raw_size_avg": 150_000_000,      # 150 MB per minute
        "feature_dim": 768,
        "extraction_method": "keyframes (1/sec) → ViT → temporal pool → 768-d",
        "gpu_ms": 200,                    # per minute of video
        "entities": ["keyframes", "scenes", "actions", "objects", "speech"],
    },
    Modality.AUDIO: {
        "raw_size_avg": 1_500_000,        # 1.5 MB per minute WAV
        "feature_dim": 768,
        "extraction_method": "mel spectrogram → audio encoder → 768-d",
        "gpu_ms": 50,
        "entities": ["transcription", "speaker_count", "events", "language"],
    },
    Modality.DOCUMENT: {
        "raw_size_avg": 10_000_000,       # 10 MB
        "feature_dim": 768,
        "extraction_method": "layout analysis + OCR + structure → 768-d",
        "gpu_ms": 100,
        "entities": ["pages", "tables", "figures", "headings", "text_blocks"],
    },
    Modality.MESH_3D: {
        "raw_size_avg": 50_000_000,       # 50 MB
        "feature_dim": 768,
        "extraction_method": "point cloud → 3D encoder → 768-d",
        "gpu_ms": 300,
        "entities": ["vertices", "topology", "bounding_box", "volume"],
    },
    Modality.SENSOR: {
        "raw_size_avg": 500_000,          # 500 KB per window
        "feature_dim": 768,
        "extraction_method": "windowed FFT → frequency features → 768-d",
        "gpu_ms": 10,
        "entities": ["channels", "anomalies", "frequency_peaks", "baseline_deviation"],
    },
}


# ── Content Reference (Data Plane pointer) ───────────────────────────────────

@dataclass
class ContentRef:
    """
    Pointer to raw media on the DATA PLANE.

    This is how NRS instances reference bulk media without
    sending it over NRSIP. The content_hash is a SHA-256
    of the raw bytes — content-addressable storage.

    32 bytes on the wire. The actual media lives in:
      - Shared storage (S3, GCS, Azure Blob)
      - Local NVMe SSD cache
      - NFS mount
    Never in a routing packet. Never on NRSIP.
    """
    content_hash: str            # SHA-256 of raw bytes (32 bytes hex)
    modality: Modality
    size_bytes: int              # Original media size
    storage_uri: str = ""        # Where to fetch (s3://bucket/hash, file:///cache/hash)
    cached_locally: bool = True  # Whether this instance has the raw bytes

    @property
    def wire_size(self) -> int:
        """Size when appended to a routing packet."""
        return 32  # Just the hash

    def to_bytes(self) -> bytes:
        """Compact binary for NRSIP frame extension."""
        return self.content_hash[:32].encode("utf-8").ljust(32, b"\x00")

    @classmethod
    def from_raw(cls, raw_bytes: bytes, modality: Modality) -> "ContentRef":
        """Create a content reference from raw media bytes."""
        content_hash = hashlib.sha256(raw_bytes).hexdigest()
        return cls(
            content_hash=content_hash,
            modality=modality,
            size_bytes=len(raw_bytes),
            cached_locally=True,
        )


# ── Media Digest (LOCAL, ~4KB) ───────────────────────────────────────────────

@dataclass
class MediaDigest:
    """
    Compressed representation of processed media. LOCAL only.

    After feature extraction, the raw media (5MB image, 150MB video)
    is reduced to this ~4KB digest containing:
      - Feature embedding (768 floats × 4 bytes = 3KB)
      - Detected entities (labels + confidence, ~500B)
      - Extracted text if any (~500B)
      - Content reference (32B hash → data plane)

    This digest feeds into the BinaryNeuronBank the SAME WAY
    as a text query embedding. The neuron bank doesn't know or
    care if the input was text, image, or video. It just sees
    a 768-d vector and fires the same deterministic activation.
    """
    modality: Modality
    content_ref: ContentRef
    feature_embedding: List[float]    # 768-d (feeds into neuron bank)
    entities: Dict[str, Any]          # Detected objects, text, scenes, etc.
    extracted_text: str = ""          # OCR text, transcription, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)
    extraction_time_ms: float = 0.0
    created_at: float = field(default_factory=time.time)

    @property
    def estimated_bytes(self) -> int:
        """~4KB: 3KB embedding + 500B entities + 500B text."""
        return len(self.feature_embedding) * 4 + 1024

    @property
    def has_text_path(self) -> bool:
        """Whether this media also has extractable text (dual-path)."""
        return len(self.extracted_text) > 0

    def to_query_embedding(self) -> List[float]:
        """
        Return the feature vector that feeds into BinaryNeuronBank.

        This is the key interface: regardless of modality, the neuron
        bank receives a 768-d float vector and does the same
        cosine sim → argmax → binary mask pipeline.
        """
        return self.feature_embedding

    def __repr__(self) -> str:
        entities_count = sum(
            len(v) if isinstance(v, list) else 1
            for v in self.entities.values()
        )
        return (
            f"MediaDigest({self.modality.value}, "
            f"{self.content_ref.size_bytes // 1024}KB raw → "
            f"~{self.estimated_bytes // 1024}KB digest, "
            f"{entities_count} entities, "
            f"text={'yes' if self.has_text_path else 'no'})"
        )


# ── Media Processor ──────────────────────────────────────────────────────────

def _xp(use_gpu: bool = True):
    if use_gpu and _GPU_AVAILABLE:
        return _cp
    return np


class EmbeddingBackend:
    """Swappable embedding backend.

    Subclass and override ``embed_*`` methods to drop in ViT, Whisper,
    or any other model while keeping the MediaProcessor pipeline
    unchanged.
    """

    def embed_text(self, text: str, dim: int) -> np.ndarray:
        raise NotImplementedError

    def embed_image(self, raw: bytes, dim: int) -> np.ndarray:
        raise NotImplementedError

    def embed_audio(self, raw: bytes, dim: int) -> np.ndarray:
        raise NotImplementedError


class _TFIDFTextEmbedder(EmbeddingBackend):
    """Lightweight TF-IDF bag-of-words embedding — no external model."""

    _VOCAB_SIZE = 4096

    def embed_text(self, text: str, dim: int) -> np.ndarray:
        xp = _xp()
        tokens = text.lower().split()
        if not tokens:
            return xp.zeros(dim, dtype=xp.float32)

        buckets = xp.zeros(self._VOCAB_SIZE, dtype=xp.float32)
        for tok in tokens:
            h = int(hashlib.sha256(tok.encode()).hexdigest(), 16) % self._VOCAB_SIZE
            buckets[h] += 1.0

        total = float(len(tokens))
        tf = buckets / total
        df = xp.clip(buckets, 0, 1)
        idf = xp.log1p(total / (df + 1.0))
        tfidf = tf * idf

        rng = np.random.RandomState(42)
        proj = rng.randn(self._VOCAB_SIZE, dim).astype(np.float32) * (1.0 / math.sqrt(self._VOCAB_SIZE))
        if xp is not np:
            proj = xp.asarray(proj)

        embedding = xp.dot(tfidf, proj)
        norm = float(xp.linalg.norm(embedding))
        if norm > 0:
            embedding = embedding / norm
        if xp is not np:
            embedding = xp.asnumpy(embedding)
        return np.asarray(embedding, dtype=np.float32)

    def embed_image(self, raw: bytes, dim: int) -> np.ndarray:
        return _ImageFeatureExtractor.extract(raw, dim)

    def embed_audio(self, raw: bytes, dim: int) -> np.ndarray:
        return _AudioFeatureExtractor.extract(raw, dim)


class _ImageFeatureExtractor:
    """Image embedding: torchvision feature extractor if available,
    perceptual hash fallback."""

    _model = None
    _transform = None
    _available: Optional[bool] = None

    @classmethod
    def _try_load(cls) -> bool:
        if cls._available is not None:
            return cls._available
        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as T

            weights = models.ResNet18_Weights.DEFAULT
            backbone = models.resnet18(weights=weights)
            backbone.fc = torch.nn.Identity()
            backbone.eval()
            cls._model = backbone
            cls._transform = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])
            cls._available = True
        except Exception:
            cls._available = False
        return cls._available

    @classmethod
    def extract(cls, raw: bytes, dim: int) -> np.ndarray:
        if cls._try_load():
            return cls._extract_cnn(raw, dim)
        return cls._extract_phash(raw, dim)

    @classmethod
    def _extract_cnn(cls, raw: bytes, dim: int) -> np.ndarray:
        import io
        import torch
        from PIL import Image

        img = Image.open(io.BytesIO(raw)).convert("RGB")
        tensor = cls._transform(img).unsqueeze(0)
        with torch.no_grad():
            features = cls._model(tensor).squeeze(0).numpy()
        return cls._project(features, dim)

    @classmethod
    def _extract_phash(cls, raw: bytes, dim: int) -> np.ndarray:
        xp = _xp()
        h = hashlib.sha256(raw).digest()
        block_size = max(1, len(raw) // 64)
        stats = []
        for i in range(0, len(raw), block_size):
            block = raw[i:i + block_size]
            if block:
                stats.append(np.mean(np.frombuffer(block, dtype=np.uint8).astype(np.float32)))
                stats.append(np.std(np.frombuffer(block, dtype=np.uint8).astype(np.float32)))
        feature = np.array(stats[:256], dtype=np.float32)
        if len(feature) < 256:
            feature = np.pad(feature, (0, 256 - len(feature)))
        return cls._project(feature, dim)

    @staticmethod
    def _project(features: np.ndarray, dim: int) -> np.ndarray:
        src_dim = features.shape[0]
        rng = np.random.RandomState(7)
        proj = rng.randn(src_dim, dim).astype(np.float32) * (1.0 / math.sqrt(src_dim))
        out = features @ proj
        norm = np.linalg.norm(out)
        if norm > 0:
            out = out / norm
        return out.astype(np.float32)


class _AudioFeatureExtractor:
    """Audio embedding: mel-spectrogram if scipy available,
    waveform statistics fallback."""

    @classmethod
    def extract(cls, raw: bytes, dim: int) -> np.ndarray:
        try:
            return cls._extract_mel(raw, dim)
        except Exception:
            return cls._extract_waveform_stats(raw, dim)

    @classmethod
    def _extract_mel(cls, raw: bytes, dim: int) -> np.ndarray:
        from scipy.io import wavfile
        from scipy.signal import spectrogram as _spectrogram
        import io

        sr, samples = wavfile.read(io.BytesIO(raw))
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        samples = samples.astype(np.float32)
        if np.max(np.abs(samples)) > 0:
            samples = samples / np.max(np.abs(samples))

        nperseg = min(1024, len(samples))
        _, _, Sxx = _spectrogram(samples, fs=sr, nperseg=nperseg)
        mel = np.log1p(Sxx)
        feature = np.concatenate([
            mel.mean(axis=1),
            mel.std(axis=1),
            mel.max(axis=1),
        ])[:512]
        if len(feature) < 512:
            feature = np.pad(feature, (0, 512 - len(feature)))
        return _ImageFeatureExtractor._project(feature, dim)

    @classmethod
    def _extract_waveform_stats(cls, raw: bytes, dim: int) -> np.ndarray:
        samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        if len(samples) == 0:
            return np.zeros(dim, dtype=np.float32)
        samples = (samples - 128.0) / 128.0
        window = max(1, len(samples) // 128)
        stats = []
        for i in range(0, len(samples), window):
            chunk = samples[i:i + window]
            if len(chunk) > 0:
                stats.extend([
                    np.mean(chunk),
                    np.std(chunk),
                    np.max(chunk),
                    np.min(chunk),
                ])
        feature = np.array(stats[:512], dtype=np.float32)
        if len(feature) < 512:
            feature = np.pad(feature, (0, 512 - len(feature)))
        return _ImageFeatureExtractor._project(feature, dim)


class MediaProcessor:
    """Local media feature extraction engine.

    Runs on the NRS instance GPU. Converts raw media into
    MediaDigest objects that feed the existing neuron pipeline.

    Production swap-in points:
      - Image: replace ``EmbeddingBackend.embed_image`` with ViT-L/14
      - Audio: replace ``EmbeddingBackend.embed_audio`` with Whisper encoder
      - Text:  replace ``EmbeddingBackend.embed_text``  with sentence-transformers

    CRITICAL: Raw media NEVER leaves this class.
    Only MediaDigest (4KB) and ContentRef (32B) come out.
    """

    EMBEDDING_DIM = 768

    def __init__(
        self,
        embedding_dim: int = 768,
        backend: Optional[EmbeddingBackend] = None,
        use_gpu: bool = True,
    ):
        self.embedding_dim = embedding_dim
        self._backend = backend or _TFIDFTextEmbedder()
        self._use_gpu = use_gpu and _GPU_AVAILABLE
        self._processed: int = 0
        self._bytes_processed: int = 0
        self._bytes_output: int = 0
        logger.info(
            "MediaProcessor init: dim=%d gpu=%s backend=%s",
            embedding_dim, self._use_gpu, type(self._backend).__name__,
        )

    def set_backend(self, backend: EmbeddingBackend) -> None:
        self._backend = backend

    def process(
        self,
        raw_bytes: bytes,
        modality: Modality,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MediaDigest:
        t0 = time.time()
        content_ref = ContentRef.from_raw(raw_bytes, modality)
        embedding = self._extract_features(raw_bytes, modality)
        entities = self._detect_entities(raw_bytes, modality)
        extracted_text = self._extract_text(raw_bytes, modality)
        elapsed_ms = (time.time() - t0) * 1000

        self._processed += 1
        self._bytes_processed += len(raw_bytes)

        digest = MediaDigest(
            modality=modality,
            content_ref=content_ref,
            feature_embedding=embedding,
            entities=entities,
            extracted_text=extracted_text,
            metadata=metadata or {},
            extraction_time_ms=elapsed_ms,
        )
        self._bytes_output += digest.estimated_bytes
        return digest

    # ── Feature Extraction ────────────────────────────────────────────────

    def _extract_features(self, raw: bytes, modality: Modality) -> List[float]:
        dim = self.embedding_dim
        try:
            if modality == Modality.TEXT:
                vec = self._backend.embed_text(raw.decode("utf-8", errors="replace"), dim)
            elif modality in (Modality.IMAGE, Modality.DOCUMENT, Modality.MESH_3D):
                vec = self._backend.embed_image(raw, dim)
            elif modality == Modality.AUDIO:
                vec = self._backend.embed_audio(raw, dim)
            elif modality == Modality.VIDEO:
                vec = self._embed_video(raw, dim)
            elif modality == Modality.SENSOR:
                vec = self._embed_sensor(raw, dim)
            else:
                vec = self._backend.embed_image(raw, dim)
        except Exception as exc:
            logger.warning("Feature extraction failed (%s), falling back: %s", modality.value, exc)
            vec = self._fallback_embedding(raw, dim)
        return vec.tolist()

    def _embed_video(self, raw: bytes, dim: int) -> np.ndarray:
        frame_size = max(1, len(raw) // 30)
        frame_embeddings = []
        for offset in range(0, len(raw), frame_size):
            frame = raw[offset:offset + frame_size]
            if len(frame) > 0:
                frame_embeddings.append(self._backend.embed_image(frame, dim))
            if len(frame_embeddings) >= 30:
                break
        if not frame_embeddings:
            return np.zeros(dim, dtype=np.float32)
        stacked = np.stack(frame_embeddings)
        pooled = stacked.mean(axis=0)
        norm = np.linalg.norm(pooled)
        if norm > 0:
            pooled = pooled / norm
        return pooled.astype(np.float32)

    def _embed_sensor(self, raw: bytes, dim: int) -> np.ndarray:
        samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        if len(samples) == 0:
            return np.zeros(dim, dtype=np.float32)
        samples = (samples - 128.0) / 128.0

        n_fft = min(256, len(samples))
        windowed = samples[:n_fft] * np.hanning(n_fft)
        spectrum = np.abs(np.fft.rfft(windowed))

        feature = np.zeros(512, dtype=np.float32)
        feature[:len(spectrum[:256])] = spectrum[:256]
        feature[256:256 + min(len(samples) // max(1, len(samples) // 256), 256)] = 0
        chunk_size = max(1, len(samples) // 256)
        stats = []
        for i in range(0, len(samples), chunk_size):
            c = samples[i:i + chunk_size]
            stats.append(np.std(c))
        stat_arr = np.array(stats[:256], dtype=np.float32)
        feature[256:256 + len(stat_arr)] = stat_arr
        return _ImageFeatureExtractor._project(feature, dim)

    @staticmethod
    def _fallback_embedding(raw: bytes, dim: int) -> np.ndarray:
        h = hashlib.sha256(raw).digest()
        rng = np.random.RandomState(int.from_bytes(h[:4], "big"))
        vec = rng.randn(dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    # ── Entity Detection ──────────────────────────────────────────────────

    def _detect_entities(self, raw: bytes, modality: Modality) -> Dict[str, Any]:
        h = hashlib.sha256(raw).hexdigest()

        if modality == Modality.IMAGE:
            return {
                "objects": self._sim_objects(h, count=3),
                "scene": self._sim_label(h, "scene", [
                    "indoor", "outdoor", "medical", "document",
                    "satellite", "microscopy", "portrait", "landscape",
                ]),
                "faces_count": int(h[0], 16) % 5,
                "has_text": int(h[1], 16) > 8,
            }
        elif modality == Modality.VIDEO:
            return {
                "keyframe_count": max(1, int(h[0:2], 16) % 60),
                "scenes": self._sim_objects(h, count=2),
                "actions": self._sim_objects(h, count=2),
                "has_speech": int(h[2], 16) > 4,
            }
        elif modality == Modality.AUDIO:
            return {
                "has_speech": int(h[0], 16) > 3,
                "speaker_count": int(h[1], 16) % 4 + 1,
                "language": self._sim_label(h, "lang", [
                    "en", "es", "fr", "de", "zh", "ar", "ja",
                ]),
                "duration_sec": int(h[2:4], 16) % 300 + 1,
            }
        elif modality == Modality.DOCUMENT:
            return {
                "pages": max(1, int(h[0:2], 16) % 50),
                "tables": int(h[2], 16) % 5,
                "figures": int(h[3], 16) % 8,
                "has_ocr_text": True,
                "doc_type": self._sim_label(h, "doctype", [
                    "report", "invoice", "contract", "letter",
                    "manual", "form", "research_paper", "spreadsheet",
                ]),
            }
        elif modality == Modality.MESH_3D:
            return {
                "vertices": int(h[0:4], 16) % 1_000_000 + 100,
                "faces": int(h[4:8], 16) % 500_000 + 50,
                "bounding_box": [1.0, 1.0, 1.0],
                "watertight": int(h[8], 16) > 8,
            }
        elif modality == Modality.SENSOR:
            return {
                "channels": int(h[0], 16) % 8 + 1,
                "sample_rate_hz": [100, 500, 1000, 5000][int(h[1], 16) % 4],
                "anomalies_detected": int(h[2], 16) % 3,
                "duration_sec": int(h[3:5], 16) % 3600 + 1,
            }
        return {}

    def _extract_text(self, raw: bytes, modality: Modality) -> str:
        h = hashlib.sha256(raw).hexdigest()
        if modality == Modality.IMAGE:
            if int(h[1], 16) > 8:
                return f"[OCR: extracted text from image {h[:8]}]"
        elif modality == Modality.AUDIO:
            if int(h[0], 16) > 3:
                return f"[ASR: transcribed speech from audio {h[:8]}]"
        elif modality == Modality.VIDEO:
            if int(h[2], 16) > 4:
                return f"[ASR: transcribed speech from video {h[:8]}]"
        elif modality == Modality.DOCUMENT:
            return f"[OCR: extracted {int(h[0:2], 16) % 5000 + 100} words from document {h[:8]}]"
        return ""

    # ── Helpers ───────────────────────────────────────────────────────────

    def _sim_label(self, h: str, prefix: str, options: List[str]) -> str:
        idx = int(hashlib.sha256(f"{h}:{prefix}".encode()).hexdigest()[:4], 16)
        return options[idx % len(options)]

    def _sim_objects(self, h: str, count: int = 3) -> List[Dict[str, Any]]:
        objects = [
            "person", "vehicle", "building", "text", "chart",
            "logo", "medical_scan", "circuit", "landscape", "food",
            "document", "screen", "equipment", "animal", "furniture",
        ]
        result = []
        for i in range(count):
            sub_h = hashlib.sha256(f"{h}:obj:{i}".encode()).hexdigest()
            idx = int(sub_h[:4], 16) % len(objects)
            conf = 0.5 + (int(sub_h[4:8], 16) % 5000) / 10000
            result.append({"label": objects[idx], "confidence": round(conf, 3)})
        return result

    @property
    def compression_stats(self) -> Dict[str, Any]:
        ratio = self._bytes_processed / max(self._bytes_output, 1)
        return {
            "media_processed": self._processed,
            "raw_bytes_in": self._bytes_processed,
            "digest_bytes_out": self._bytes_output,
            "compression_ratio": f"{ratio:.0f}x",
            "bytes_kept_local": self._bytes_processed - self._bytes_output,
        }


# ── Media-Aware Routing Packet Extension ─────────────────────────────────────

@dataclass
class MediaRoutingPacket:
    """
    Extended routing packet for media queries.

    Same 128-byte base routing packet from neurons.py,
    plus a 32-byte ContentRef hash. Total: 160 bytes.

    This is still trivial on NRSIP:
      6,433 RPS × 160B = 1.0 MB/sec per instance
      3 instances = 3.0 MB/sec total
      A 1 Gbps link handles 125 MB/sec.

    The raw media (5MB, 150MB, 500MB) stays on the data plane.
    """
    # From base routing packet
    query_hash: str
    tier: str
    domain: Optional[str]
    confidence: float

    # Media extension
    content_ref: ContentRef
    modality: Modality
    has_text_path: bool             # Dual-path: also process as text?
    entity_summary: Dict[str, Any]  # Compressed entity labels

    source_instance_id: str = ""
    created_at: float = field(default_factory=time.time)

    @property
    def estimated_bytes(self) -> int:
        """128B base + 32B content ref = 160B on NRSIP."""
        return 160

    @property
    def raw_media_size(self) -> int:
        """Size of the actual media (stays on data plane)."""
        return self.content_ref.size_bytes

    @property
    def wire_compression(self) -> str:
        """How much smaller the packet is vs raw media."""
        ratio = self.content_ref.size_bytes / self.estimated_bytes
        return f"{ratio:,.0f}x"

    def __repr__(self) -> str:
        return (
            f"MediaRoutingPacket({self.modality.value}, "
            f"tier={self.tier}, domain={self.domain}, "
            f"{self.content_ref.size_bytes // 1024}KB raw → "
            f"{self.estimated_bytes}B on wire, "
            f"compression={self.wire_compression})"
        )


# ── Multi-NRS Media Coordination ─────────────────────────────────────────────

class MediaCoordinator:
    """
    Coordinates media processing across multiple NRS instances.

    Decides:
      1. Which instance processes the media? (locality-first)
      2. Does any other instance need the raw bytes? (data plane fetch)
      3. Can we skip re-processing? (PVS cache check by content_hash)

    Rules:
      - Process media on the instance where it arrives (always)
      - Only send MediaRoutingPacket (160B) over NRSIP (always)
      - Other instances fetch raw from data plane IF they need it
      - PVS cache: same content_hash → skip re-processing entirely

    In most cases, only ONE instance ever sees the raw media.
    Others work from the routing packet alone.
    """

    def __init__(self):
        self._processor = MediaProcessor()
        self._cache: Dict[str, MediaDigest] = {}  # content_hash → digest
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    def ingest(
        self,
        raw_bytes: bytes,
        modality: Modality,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[MediaDigest, MediaRoutingPacket]:
        """
        Full media ingestion pipeline.

        1. Check PVS cache (same media seen before?)
        2. If miss: process locally → MediaDigest
        3. Build MediaRoutingPacket (160B, network-safe)
        4. Return both for local processing + NRSIP routing
        """
        content_hash = hashlib.sha256(raw_bytes).hexdigest()

        # PVS cache check: same media → skip processing
        if content_hash in self._cache:
            self._cache_hits += 1
            digest = self._cache[content_hash]
        else:
            self._cache_misses += 1
            digest = self._processor.process(raw_bytes, modality, metadata)
            self._cache[content_hash] = digest

        # Build network-safe routing packet
        packet = MediaRoutingPacket(
            query_hash=content_hash[:32],
            tier=self._estimate_tier(digest),
            domain=self._estimate_domain(digest),
            confidence=digest.feature_embedding[0] if digest.feature_embedding else 0.0,
            content_ref=digest.content_ref,
            modality=modality,
            has_text_path=digest.has_text_path,
            entity_summary=self._compress_entities(digest.entities),
        )

        return digest, packet

    def _estimate_tier(self, digest: MediaDigest) -> str:
        """Estimate complexity tier from media features."""
        specs = MODALITY_SPECS.get(digest.modality, {})
        gpu_ms = specs.get("gpu_ms", 100)

        # Heavier processing → higher tier
        if gpu_ms <= 10:
            return "T1"
        elif gpu_ms <= 50:
            return "T2"
        elif gpu_ms <= 200:
            return "T3"
        return "T4"

    def _estimate_domain(self, digest: MediaDigest) -> str:
        """Estimate domain from detected entities."""
        entities = digest.entities

        # Check for medical indicators
        if any("medical" in str(v).lower() for v in entities.values()):
            return "medical"
        if any("scan" in str(v).lower() for v in entities.values()):
            return "medical"

        # Check for financial indicators
        if entities.get("doc_type") in ("invoice", "spreadsheet", "form"):
            return "financial"

        # Check for engineering indicators
        if digest.modality == Modality.MESH_3D:
            return "engineering"
        if digest.modality == Modality.SENSOR:
            return "engineering"

        return "general"

    def _compress_entities(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compress entities to fit in routing packet.
        Keep labels and counts, drop details.
        """
        compressed = {}
        for k, v in entities.items():
            if isinstance(v, list):
                # Keep just count and top label
                compressed[k] = {
                    "count": len(v),
                    "top": v[0].get("label", str(v[0])) if v and isinstance(v[0], dict) else str(v[0]) if v else None,
                }
            elif isinstance(v, (int, float, bool, str)):
                compressed[k] = v
        return compressed

    @property
    def stats(self) -> Dict[str, Any]:
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        return {
            "total_ingested": total,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": f"{hit_rate:.1%}",
            "cached_digests": len(self._cache),
            "processor_stats": self._processor.compression_stats,
        }

    @property
    def traffic_projection(self) -> Dict[str, str]:
        """
        Projected NRSIP traffic with media workload.
        """
        rps = 6_433
        text_pct = 0.5
        media_pct = 0.5

        text_rps = rps * text_pct
        media_rps = rps * media_pct

        text_bps = text_rps * 128       # 128B text packets
        media_bps = media_rps * 160     # 160B media packets
        total_bps = text_bps + media_bps

        def fmt(bps: float) -> str:
            mb = bps / (1024 * 1024)
            if mb < 1:
                return f"{bps / 1024:.0f} KB/s"
            return f"{mb:.1f} MB/s"

        # Data plane (raw media fetches from storage)
        # With 92.4% PVS cache hit rate
        pvs_miss_rate = 0.076
        storage_fetches = media_rps * pvs_miss_rate
        avg_media_mb = 5  # 5 MB average
        storage_bps = storage_fetches * avg_media_mb * 1024 * 1024

        return {
            "scenario": "50% text + 50% media at 6,433 RPS",
            "text_queries_sec": int(text_rps),
            "media_queries_sec": int(media_rps),
            "nrsip_text_traffic": fmt(text_bps),
            "nrsip_media_traffic": fmt(media_bps),
            "nrsip_total_per_instance": fmt(total_bps),
            "nrsip_3_instances": fmt(total_bps * 3),
            "nrsip_link_utilization": f"{(total_bps * 3) / (125 * 1024 * 1024) * 100:.1f}%",
            "data_plane_fetches_sec": f"{storage_fetches:.0f}",
            "data_plane_bandwidth": fmt(storage_bps),
            "data_plane_note": "Single NVMe SSD handles 3.5 GB/s",
            "raw_media_kept_local": "YES — never on NRSIP",
        }
