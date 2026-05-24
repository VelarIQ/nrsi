"""
NRSI Streaming — Live & VOD Media Pipeline with NRS Integration.

Handles 8K+ live streams, video production, and scaled media
delivery WITHOUT routing raw media bytes through NRSIP.

═══════════════════════════════════════════════════════════════════
  THE STREAMING PROBLEM
═══════════════════════════════════════════════════════════════════

8K video (7680×4320 @ 60fps):
  Raw pixels:     48 Gbps per stream
  H.265 encoded:  50-100 Mbps per stream
  H.266/VVC:      25-50 Mbps per stream

100 concurrent 8K streams:
  Encoded:        5-10 Gbps continuous
  Raw:            4.8 Tbps (absurd, never transmitted raw)

If you try to push ANY of this through NRSIP routing:
  Even 1 stream at 50 Mbps > entire NRSIP budget (2.7 MB/sec)
  NRSIP is a CONTROL PLANE. It carries 160-byte packets.
  Streaming media is a DATA PLANE problem.

═══════════════════════════════════════════════════════════════════
  THREE-PLANE ARCHITECTURE
═══════════════════════════════════════════════════════════════════

NRS adds a third plane for streaming:

  CONTROL PLANE (NRSIP):     128-160 byte routing packets
    "Stream seg-00047 contains medical imagery, T3,
     route to radiology crease, H_score=0.94"

  DATA PLANE (object storage): bulk media at rest
    S3/GCS/Azure Blob for stored segments
    Content-addressed by SHA-256 hash
    PVS cache: don't reprocess identical segments

  STREAM PLANE (dedicated):   live media in motion
    RTMP/SRT/WebRTC ingest
    HLS/DASH adaptive bitrate output
    GPU transcoding pipeline (8K → 4K → 1080p → 720p → 480p)
    CDN edge distribution
    COMPLETELY SEPARATE from NRSIP

The stream plane handles the heavy bytes.
NRS taps into it at segment boundaries for AI validation.
NRSIP never sees a single pixel.

═══════════════════════════════════════════════════════════════════
  HOW NRS INTEGRATES WITH STREAMING
═══════════════════════════════════════════════════════════════════

NRS doesn't replace FFmpeg, GStreamer, or your CDN.
NRS validates WHAT THE STREAM CONTAINS and ensures
AI-generated content about the stream is hallucination-free.

  Live stream → Segmenter (2-6 sec chunks)
       │
       ├──→ Stream Plane: transcode + CDN delivery
       │    (raw bytes, high bandwidth, NRS doesn't touch)
       │
       └──→ NRS Tap: keyframe extraction (1 per segment)
            │
            ├── MediaProcessor: CNN → 768-d embedding
            ├── Entity detection: objects, text, faces, scenes
            ├── Content classification: medical, financial, etc.
            ├── Safety validation: prohibited content check
            ├── H_score: hallucination check on AI annotations
            │
            └── RoutingPacket (160B) → NRSIP
                "Segment validated, safe, medical domain, T2"

NRS processes ONE keyframe per segment (not every frame).
At 2-sec segments: 0.5 keyframes/sec per stream.
100 streams × 0.5 = 50 NRS queries/sec from streaming.
That's < 1% of NRS capacity (6,433 RPS).

═══════════════════════════════════════════════════════════════════
  RESOLUTION LADDER
═══════════════════════════════════════════════════════════════════

  Resolution    Pixels/frame   H.265 bitrate    H.266/VVC
  ──────────    ────────────   ─────────────    ─────────
  480p          640×480        1.5 Mbps         0.8 Mbps
  720p          1280×720       3 Mbps           1.5 Mbps
  1080p         1920×1080      8 Mbps           4 Mbps
  1440p (2K)    2560×1440      16 Mbps          8 Mbps
  2160p (4K)    3840×2160      35 Mbps          18 Mbps
  4320p (8K)    7680×4320      80 Mbps          40 Mbps
  8640p (16K)   15360×8640     200 Mbps         100 Mbps

  Adaptive bitrate: serve highest quality the client can handle.
  NRS validates at NATIVE resolution (highest available).
  Transcoding ladder is GPU-bound, not NRS-bound.

═══════════════════════════════════════════════════════════════════
  SCALING MATH
═══════════════════════════════════════════════════════════════════

Per-stream costs:

  Stream Plane (GPU transcode):
    8K H.265 encode: 1 GPU per stream (NVENC)
    Adaptive ladder (8K→4K→1080p→720p→480p): 1.5 GPU per stream
    100 streams: 150 GPUs for transcoding

  Data Plane (CDN):
    8K per viewer: 80 Mbps
    1000 viewers × 80 Mbps = 80 Gbps CDN egress
    Standard CDN problem, not NRS-specific

  NRS Validation (control plane):
    100 streams × 0.5 keyframes/sec = 50 queries/sec
    50 × 160B = 8 KB/sec on NRSIP
    NRS capacity: 6,433 RPS
    Streaming uses 0.8% of NRS capacity

  Total NRSIP with streaming + regular queries:
    Regular: 6,383 RPS × 160B = 1.0 MB/sec
    Streaming: 50 RPS × 160B = 8 KB/sec
    Total: 1.0 MB/sec (streaming is rounding error)
"""

from __future__ import annotations

import hashlib
import struct
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable

from nrsi.core.media import (
    Modality, MediaProcessor, MediaDigest,
    MediaRoutingPacket, ContentRef, MODALITY_SPECS,
)


# ── Resolution & Codec Specs ─────────────────────────────────────────────────

class Resolution(Enum):
    """Standard video resolutions."""
    RES_480P = "480p"
    RES_720P = "720p"
    RES_1080P = "1080p"
    RES_1440P = "1440p"       # 2K
    RES_2160P = "2160p"       # 4K UHD
    RES_4320P = "4320p"       # 8K UHD
    RES_8640P = "8640p"       # 16K (future)


class Codec(Enum):
    """Video codecs."""
    H264 = "h264"             # AVC — legacy, wide compat
    H265 = "h265"             # HEVC — current standard
    H266 = "h266"             # VVC — next gen, 50% better compression
    AV1 = "av1"               # Royalty-free, competitive with H.266
    VP9 = "vp9"               # Google/YouTube standard


class IngestProtocol(Enum):
    """Live stream ingest protocols."""
    RTMP = "rtmp"             # Legacy, widely supported
    SRT = "srt"               # Secure Reliable Transport — low latency
    WEBRTC = "webrtc"         # Sub-second latency
    RTSP = "rtsp"             # IP cameras, surveillance
    NDI = "ndi"               # Professional broadcast
    RIST = "rist"             # Reliable Internet Stream Transport


class DeliveryProtocol(Enum):
    """Stream delivery protocols."""
    HLS = "hls"               # Apple — widest compatibility
    DASH = "dash"             # MPEG-DASH — adaptive bitrate
    CMAF = "cmaf"             # Common Media Application Format
    WEBRTC = "webrtc"         # Ultra-low latency delivery
    LL_HLS = "ll-hls"         # Low-Latency HLS


# Resolution specifications
RESOLUTION_SPECS = {
    Resolution.RES_480P: {
        "width": 640, "height": 480, "pixels": 307_200,
        "h265_mbps": 1.5, "h266_mbps": 0.8, "av1_mbps": 0.9,
        "gpu_encode_fraction": 0.05,  # Fraction of one GPU
    },
    Resolution.RES_720P: {
        "width": 1280, "height": 720, "pixels": 921_600,
        "h265_mbps": 3.0, "h266_mbps": 1.5, "av1_mbps": 1.8,
        "gpu_encode_fraction": 0.1,
    },
    Resolution.RES_1080P: {
        "width": 1920, "height": 1080, "pixels": 2_073_600,
        "h265_mbps": 8.0, "h266_mbps": 4.0, "av1_mbps": 5.0,
        "gpu_encode_fraction": 0.2,
    },
    Resolution.RES_1440P: {
        "width": 2560, "height": 1440, "pixels": 3_686_400,
        "h265_mbps": 16.0, "h266_mbps": 8.0, "av1_mbps": 10.0,
        "gpu_encode_fraction": 0.35,
    },
    Resolution.RES_2160P: {
        "width": 3840, "height": 2160, "pixels": 8_294_400,
        "h265_mbps": 35.0, "h266_mbps": 18.0, "av1_mbps": 22.0,
        "gpu_encode_fraction": 0.6,
    },
    Resolution.RES_4320P: {
        "width": 7680, "height": 4320, "pixels": 33_177_600,
        "h265_mbps": 80.0, "h266_mbps": 40.0, "av1_mbps": 50.0,
        "gpu_encode_fraction": 1.0,  # Full GPU for 8K encode
    },
    Resolution.RES_8640P: {
        "width": 15360, "height": 8640, "pixels": 132_710_400,
        "h265_mbps": 200.0, "h266_mbps": 100.0, "av1_mbps": 120.0,
        "gpu_encode_fraction": 2.5,  # Needs multiple GPUs
    },
}

# Standard adaptive bitrate ladders
ABR_LADDERS = {
    "8k_full": [
        Resolution.RES_4320P,  # 8K source
        Resolution.RES_2160P,  # 4K
        Resolution.RES_1080P,  # 1080p
        Resolution.RES_720P,   # 720p
        Resolution.RES_480P,   # 480p fallback
    ],
    "4k_standard": [
        Resolution.RES_2160P,
        Resolution.RES_1080P,
        Resolution.RES_720P,
        Resolution.RES_480P,
    ],
    "1080p_web": [
        Resolution.RES_1080P,
        Resolution.RES_720P,
        Resolution.RES_480P,
    ],
    "low_latency": [
        Resolution.RES_1080P,
        Resolution.RES_720P,
    ],
}


# ── Stream Segment ───────────────────────────────────────────────────────────

@dataclass
class StreamSegment:
    """
    A chunk of a live or VOD stream.

    Streams are segmented at 2-6 second boundaries for:
      1. Adaptive bitrate switching
      2. CDN caching
      3. NRS validation (1 keyframe per segment)

    The segment itself lives on the STREAM PLANE.
    Only its metadata + NRS validation result cross NRSIP.
    """
    segment_id: str
    stream_id: str
    sequence_number: int
    duration_sec: float              # Typically 2-6 seconds
    resolution: Resolution
    codec: Codec
    size_bytes: int                  # Encoded segment size
    keyframe_bytes: Optional[bytes] = None  # Extracted keyframe for NRS
    pts_start: float = 0.0          # Presentation timestamp start
    pts_end: float = 0.0
    created_at: float = field(default_factory=time.time)

    # NRS validation results (filled after processing)
    nrs_validated: bool = False
    nrs_domain: Optional[str] = None
    nrs_tier: Optional[str] = None
    nrs_h_score: Optional[float] = None
    nrs_entities: Dict[str, Any] = field(default_factory=dict)
    nrs_safety_pass: bool = True

    @property
    def bitrate_mbps(self) -> float:
        """Actual bitrate of this segment."""
        if self.duration_sec <= 0:
            return 0.0
        return (self.size_bytes * 8) / (self.duration_sec * 1_000_000)

    def __repr__(self) -> str:
        status = "validated" if self.nrs_validated else "pending"
        return (
            f"Segment({self.stream_id}:seq{self.sequence_number}, "
            f"{self.resolution.value}, {self.duration_sec}s, "
            f"{self.bitrate_mbps:.1f}Mbps, nrs={status})"
        )


# ── Stream Configuration ─────────────────────────────────────────────────────

@dataclass
class StreamConfig:
    """
    Configuration for a live or VOD stream.

    Defines the ingest source, transcoding ladder,
    delivery method, and NRS validation settings.
    """
    stream_id: str
    source_resolution: Resolution = Resolution.RES_4320P
    source_codec: Codec = Codec.H265
    source_fps: int = 60

    # Ingest
    ingest_protocol: IngestProtocol = IngestProtocol.SRT
    ingest_url: str = ""

    # Transcoding
    target_codec: Codec = Codec.H265
    abr_ladder: str = "8k_full"       # Key into ABR_LADDERS
    segment_duration_sec: float = 4.0  # HLS/DASH segment length
    keyframe_interval_sec: float = 2.0 # GOP size

    # Delivery
    delivery_protocol: DeliveryProtocol = DeliveryProtocol.HLS
    cdn_origin_url: str = ""
    max_latency_sec: float = 6.0      # Target glass-to-glass

    # NRS integration
    nrs_validation_enabled: bool = True
    nrs_keyframes_per_segment: int = 1  # How many keyframes NRS inspects
    nrs_safety_check: bool = True       # Block segments that fail safety
    nrs_content_tagging: bool = True    # Auto-tag content domain

    @property
    def ladder_resolutions(self) -> List[Resolution]:
        return ABR_LADDERS.get(self.abr_ladder, ABR_LADDERS["4k_standard"])

    @property
    def source_bitrate_mbps(self) -> float:
        spec = RESOLUTION_SPECS.get(self.source_resolution, {})
        codec_key = f"{self.target_codec.value}_mbps"
        if codec_key not in spec:
            codec_key = "h265_mbps"
        return spec.get(codec_key, 50.0)

    @property
    def total_ladder_bitrate_mbps(self) -> float:
        """Total bitrate across all ABR rungs."""
        total = 0.0
        codec_key = f"{self.target_codec.value}_mbps"
        for res in self.ladder_resolutions:
            spec = RESOLUTION_SPECS.get(res, {})
            if codec_key not in spec:
                codec_key = "h265_mbps"
            total += spec.get(codec_key, 5.0)
        return total

    @property
    def gpu_cost(self) -> float:
        """GPU units needed for full ABR transcoding."""
        total = 0.0
        for res in self.ladder_resolutions:
            spec = RESOLUTION_SPECS.get(res, {})
            total += spec.get("gpu_encode_fraction", 0.2)
        return total


# ── Stream Pipeline ──────────────────────────────────────────────────────────

class StreamPipeline:
    """
    Live & VOD streaming pipeline with NRS integration.

    Manages the full lifecycle:
      1. INGEST:    Receive live stream (RTMP/SRT/WebRTC)
      2. SEGMENT:   Chunk into 2-6 sec segments
      3. TRANSCODE: GPU encode ABR ladder (8K→4K→1080p→720p→480p)
      4. NRS TAP:   Extract keyframe → MediaProcessor → validate
      5. DELIVER:   Push to CDN via HLS/DASH/CMAF

    Stream plane handles heavy bytes (transcode, CDN).
    NRS control plane handles validation (160B packets).
    They NEVER mix.

    Scaling:
      1 stream (8K):     80 Mbps stream plane, 80B/sec NRSIP
      10 streams (8K):   800 Mbps stream plane, 800B/sec NRSIP
      100 streams (8K):  8 Gbps stream plane, 8 KB/sec NRSIP
      1000 streams (8K): 80 Gbps stream plane, 80 KB/sec NRSIP

    NRSIP traffic from streaming is always negligible.
    Stream plane scales independently (more GPUs, more CDN edge).
    """

    def __init__(self):
        self._streams: Dict[str, StreamConfig] = {}
        self._segments: Dict[str, List[StreamSegment]] = {}
        self._media_processor = MediaProcessor()

        # Stats
        self._segments_processed: int = 0
        self._segments_validated: int = 0
        self._segments_blocked: int = 0
        self._total_stream_bytes: int = 0
        self._total_nrsip_bytes: int = 0

    def register_stream(self, config: StreamConfig) -> Dict[str, Any]:
        """
        Register a new live or VOD stream.

        Returns resource requirements and NRSIP traffic projection.
        """
        self._streams[config.stream_id] = config
        self._segments[config.stream_id] = []

        # Calculate resource requirements
        resources = self._calculate_resources(config)
        return resources

    def ingest_segment(
        self,
        stream_id: str,
        segment_data: bytes,
        sequence_number: int,
        duration_sec: float = 4.0,
    ) -> StreamSegment:
        """
        Ingest a stream segment.

        In production:
          1. Segment arrives from RTMP/SRT demuxer
          2. Keyframe extracted by GPU decoder
          3. Segment queued for ABR transcoding (stream plane)
          4. Keyframe sent to NRS for validation (control plane)

        Here: simulates the pipeline with hash-based processing.
        """
        config = self._streams.get(stream_id)
        if not config:
            raise ValueError(f"Stream {stream_id} not registered")

        # Create segment record
        seg_id = hashlib.sha256(
            f"{stream_id}:{sequence_number}".encode()
        ).hexdigest()[:16]

        segment = StreamSegment(
            segment_id=seg_id,
            stream_id=stream_id,
            sequence_number=sequence_number,
            duration_sec=duration_sec,
            resolution=config.source_resolution,
            codec=config.target_codec,
            size_bytes=len(segment_data),
        )

        # Extract keyframe for NRS (simulated: first 50KB of segment)
        keyframe_size = min(50_000, len(segment_data))
        segment.keyframe_bytes = segment_data[:keyframe_size]

        self._total_stream_bytes += len(segment_data)

        # NRS validation (if enabled)
        if config.nrs_validation_enabled and segment.keyframe_bytes:
            self._validate_segment(segment, config)

        # Store
        self._segments[stream_id].append(segment)
        self._segments_processed += 1

        return segment

    def _validate_segment(self, segment: StreamSegment, config: StreamConfig):
        """
        NRS validation of a stream segment via keyframe.

        This is the ONLY point where streaming touches NRS.
        One keyframe per segment → MediaProcessor → routing packet.
        Total NRSIP cost: 160 bytes per segment.
        """
        if not segment.keyframe_bytes:
            return

        # Process keyframe through media pipeline
        digest = self._media_processor.process(
            segment.keyframe_bytes,
            Modality.IMAGE,
            metadata={
                "source": "stream_keyframe",
                "stream_id": segment.stream_id,
                "sequence": segment.sequence_number,
                "resolution": segment.resolution.value,
            },
        )

        # Fill NRS validation results
        segment.nrs_validated = True
        segment.nrs_domain = self._detect_domain(digest)
        segment.nrs_tier = self._estimate_tier(digest)
        segment.nrs_entities = digest.entities
        segment.nrs_h_score = self._compute_h_score(digest)

        # Safety check
        if config.nrs_safety_check:
            segment.nrs_safety_pass = self._safety_check(digest)
            if not segment.nrs_safety_pass:
                self._segments_blocked += 1

        self._segments_validated += 1
        self._total_nrsip_bytes += 160  # One routing packet per segment

    def _detect_domain(self, digest: MediaDigest) -> str:
        """Domain detection from keyframe features."""
        entities = digest.entities
        if any("medical" in str(v).lower() for v in entities.values()):
            return "medical"
        if any("chart" in str(v).lower() or "document" in str(v).lower()
               for v in entities.values()):
            return "financial"
        return "general"

    def _estimate_tier(self, digest: MediaDigest) -> str:
        """Tier estimation from keyframe complexity."""
        entity_count = sum(
            len(v) if isinstance(v, list) else 1
            for v in digest.entities.values()
        )
        if entity_count <= 3:
            return "T1"
        elif entity_count <= 6:
            return "T2"
        elif entity_count <= 10:
            return "T3"
        return "T4"

    def _compute_h_score(self, digest: MediaDigest) -> float:
        """Simulated H_score for keyframe validation."""
        h = hashlib.sha256(
            digest.content_ref.content_hash.encode()
        ).digest()
        return 0.7 + (struct.unpack(">H", h[:2])[0] / 65535) * 0.3

    def _safety_check(self, digest: MediaDigest) -> bool:
        """
        Content safety validation.

        Production: NSFW detection, prohibited content,
        deepfake detection, violence classification, etc.
        Simulation: hash-based pass/fail.
        """
        h = hashlib.sha256(
            digest.content_ref.content_hash.encode()
        ).hexdigest()
        # 98% pass rate in simulation
        return int(h[0], 16) > 0

    def _calculate_resources(self, config: StreamConfig) -> Dict[str, Any]:
        """
        Calculate full resource requirements for a stream.

        Three independent scaling axes:
          1. Stream plane (GPU transcode + CDN bandwidth)
          2. Data plane (segment storage)
          3. Control plane (NRS validation, negligible)
        """
        source_spec = RESOLUTION_SPECS.get(config.source_resolution, {})
        codec_key = f"{config.target_codec.value}_mbps"
        if codec_key not in source_spec:
            codec_key = "h265_mbps"

        # Stream plane: transcoding GPUs
        gpu_total = config.gpu_cost

        # Stream plane: bandwidth per viewer
        per_viewer_mbps = source_spec.get(codec_key, 50.0)

        # Data plane: segment storage
        segment_mb = (per_viewer_mbps * config.segment_duration_sec) / 8
        segments_per_hour = 3600 / config.segment_duration_sec
        storage_gb_per_hour = (segment_mb * segments_per_hour * len(config.ladder_resolutions)) / 1024

        # Control plane: NRS queries
        nrs_queries_per_sec = config.nrs_keyframes_per_segment / config.segment_duration_sec
        nrsip_bytes_per_sec = nrs_queries_per_sec * 160

        # Ladder detail
        ladder_detail = []
        for res in config.ladder_resolutions:
            spec = RESOLUTION_SPECS.get(res, {})
            bitrate = spec.get(codec_key, 5.0)
            gpu_frac = spec.get("gpu_encode_fraction", 0.2)
            ladder_detail.append({
                "resolution": res.value,
                "bitrate_mbps": bitrate,
                "gpu_fraction": gpu_frac,
                "segment_size_mb": round((bitrate * config.segment_duration_sec) / 8, 2),
            })

        return {
            "stream_id": config.stream_id,
            "source": f"{config.source_resolution.value} @ {config.source_fps}fps",
            "codec": config.target_codec.value,
            "segment_duration_sec": config.segment_duration_sec,

            "stream_plane": {
                "gpu_transcode_units": round(gpu_total, 2),
                "source_bitrate_mbps": per_viewer_mbps,
                "total_ladder_bitrate_mbps": round(config.total_ladder_bitrate_mbps, 1),
                "abr_rungs": len(config.ladder_resolutions),
                "ladder": ladder_detail,
            },

            "data_plane": {
                "storage_gb_per_hour": round(storage_gb_per_hour, 1),
                "segment_count_per_hour": int(segments_per_hour),
            },

            "control_plane_nrs": {
                "nrs_queries_per_sec": round(nrs_queries_per_sec, 2),
                "nrsip_bytes_per_sec": int(nrsip_bytes_per_sec),
                "nrsip_note": "Negligible — less than a single TCP keepalive",
                "nrs_capacity_used": f"{nrs_queries_per_sec / 6433 * 100:.3f}%",
            },
        }

    def scale_projection(
        self,
        concurrent_streams: int = 100,
        resolution: Resolution = Resolution.RES_4320P,
        codec: Codec = Codec.H265,
        viewers_per_stream: int = 1000,
        segment_duration_sec: float = 4.0,
    ) -> Dict[str, Any]:
        """
        Project resource requirements at scale.

        Shows the three planes independently and
        proves NRSIP stays negligible at any stream count.
        """
        spec = RESOLUTION_SPECS.get(resolution, {})
        codec_key = f"{codec.value}_mbps"
        if codec_key not in spec:
            codec_key = "h265_mbps"

        source_mbps = spec.get(codec_key, 50.0)
        gpu_per_stream = spec.get("gpu_encode_fraction", 1.0) * 2.0  # Full ABR ladder

        # Stream plane
        total_gpu = concurrent_streams * gpu_per_stream
        ingest_gbps = (concurrent_streams * source_mbps) / 1000
        cdn_gbps = (concurrent_streams * viewers_per_stream * source_mbps) / 1000

        # Data plane
        segment_mb = (source_mbps * segment_duration_sec) / 8
        storage_tb_day = (concurrent_streams * segment_mb * (86400 / segment_duration_sec) * 5) / (1024 * 1024)  # 5 ABR rungs

        # Control plane
        nrs_qps = concurrent_streams / segment_duration_sec
        nrsip_kbps = (nrs_qps * 160 * 8) / 1000
        nrs_pct = nrs_qps / 6433 * 100

        def fmt_gbps(gbps: float) -> str:
            if gbps < 1:
                return f"{gbps * 1000:.0f} Mbps"
            elif gbps >= 1000:
                return f"{gbps / 1000:.1f} Tbps"
            return f"{gbps:.1f} Gbps"

        return {
            "scenario": f"{concurrent_streams} × {resolution.value} streams, {viewers_per_stream} viewers each",

            "stream_plane_gpu": {
                "gpus_transcode": f"{total_gpu:.0f} GPUs",
                "gpu_type": "NVIDIA A100/H100 (NVENC)",
                "ingest_bandwidth": fmt_gbps(ingest_gbps),
                "cdn_egress": fmt_gbps(cdn_gbps),
                "cdn_note": "Standard CDN problem — Cloudflare/Akamai/Fastly",
            },

            "data_plane_storage": {
                "storage_per_day": f"{storage_tb_day:.1f} TB",
                "segment_size_avg": f"{segment_mb:.1f} MB",
                "segments_per_sec": f"{concurrent_streams / segment_duration_sec:.0f}",
            },

            "control_plane_nrsip": {
                "nrs_queries_per_sec": f"{nrs_qps:.1f}",
                "nrsip_bandwidth": f"{nrsip_kbps:.1f} Kbps",
                "nrs_capacity_used": f"{nrs_pct:.1f}%",
                "packet_size": "160 bytes",
                "verdict": "NEGLIGIBLE" if nrs_pct < 10 else "SIGNIFICANT" if nrs_pct < 50 else "HEAVY",
            },

            "independence": {
                "stream_plane_bottleneck": "GPU count + CDN edge capacity",
                "data_plane_bottleneck": "Storage IOPS + capacity",
                "nrs_bottleneck": "None — streaming is < 5% of NRS capacity",
                "scaling_method": "Add GPUs and CDN edge independently of NRS",
            },
        }

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "active_streams": len(self._streams),
            "segments_processed": self._segments_processed,
            "segments_validated": self._segments_validated,
            "segments_blocked": self._segments_blocked,
            "total_stream_bytes": self._total_stream_bytes,
            "total_nrsip_bytes": self._total_nrsip_bytes,
            "nrsip_vs_stream_ratio": (
                f"{self._total_nrsip_bytes / max(self._total_stream_bytes, 1) * 100:.6f}%"
            ),
            "processor_stats": self._media_processor.compression_stats,
        }
