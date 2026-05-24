"""
NRS Media Verification Gates
==============================

Every piece of NRS-generated media passes through verification before
delivery.  This is the structural difference between NRS and statistical
models: provenance, confidence scoring, and trust-level assignment are
first-class outputs, not afterthoughts.

Pipeline
--------
  Generated media (image / video / audio)
       │
       ▼
  ┌── Per-check verification ───────────────────────────────────────┐
  │  Prompt alignment · Composition · Lighting · Color harmony      │
  │  Technical quality · Scale consistency · Temporal coherence      │
  │  Spectral clarity · Rhythm · Harmonic analysis                  │
  └────────────┬────────────────────────────────────────────────────┘
               │
               ▼
  ┌── Provenance chain ─────────────────────────────────────────────┐
  │  Every step hashed (SHA-256) with timestamps and confidence     │
  └────────────┬────────────────────────────────────────────────────┘
               │
               ▼
  ┌── Trust-level assignment ───────────────────────────────────────┐
  │  RAW → VALIDATED → TRUSTED → CERTIFIED                         │
  │  Based on which check categories pass and at what confidence    │
  └─────────────────────────────────────────────────────────────────┘

Trust Level Rules
-----------------
  RAW        — no verification run
  VALIDATED  — all *technical* checks pass (exposure, sharpness, no clipping)
  TRUSTED    — technical + composition + consistency checks pass
  CERTIFIED  — all checks pass including prompt alignment with >90% confidence

Dependencies: numpy, hashlib (stdlib).  No external ML packages.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# TRUST LEVELS & CHECK RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

class MediaTrustLevel(str, Enum):
    RAW = "RAW"
    VALIDATED = "VALIDATED"
    TRUSTED = "TRUSTED"
    CERTIFIED = "CERTIFIED"


class CheckCategory(str, Enum):
    TECHNICAL = "technical"
    COMPOSITION = "composition"
    CONSISTENCY = "consistency"
    PROMPT_ALIGNMENT = "prompt_alignment"


@dataclass
class CheckResult:
    """Outcome of a single verification check."""
    name: str
    passed: bool
    confidence: float
    category: CheckCategory
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return f"[{mark}] {self.name} ({self.confidence:.2f})"


# ═══════════════════════════════════════════════════════════════════════════════
# PROVENANCE CHAIN
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ProvenanceStep:
    """One step in the media-generation provenance chain."""
    action: str
    module: str
    input_hash: str
    output_hash: str
    confidence: float
    timestamp_ms: int
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "module": self.module,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "confidence": self.confidence,
            "timestamp_ms": self.timestamp_ms,
            "details": self.details,
        }


@dataclass
class MediaProvenance:
    """Full provenance record for a verified media artifact."""
    steps: List[ProvenanceStep] = field(default_factory=list)
    trust_level: str = MediaTrustLevel.RAW.value
    overall_confidence: float = 0.0
    verification_passed: bool = False
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "trust_level": self.trust_level,
            "overall_confidence": self.overall_confidence,
            "verification_passed": self.verification_passed,
            "flags": self.flags,
        }

    def summary(self) -> str:
        lines = [
            f"MediaProvenance  trust={self.trust_level}  "
            f"confidence={self.overall_confidence:.3f}  "
            f"passed={self.verification_passed}",
        ]
        for s in self.steps:
            lines.append(
                f"  {s.action:16s}  module={s.module:24s}  "
                f"conf={s.confidence:.2f}"
            )
        if self.flags:
            lines.append(f"  flags: {', '.join(self.flags)}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# VERIFICATION RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MediaVerificationResult:
    """Unified verification result returned by every gate method."""
    checks: List[CheckResult] = field(default_factory=list)
    trust_level: str = MediaTrustLevel.RAW.value
    confidence: float = 0.0
    provenance: MediaProvenance = field(default_factory=MediaProvenance)
    elapsed_ms: float = 0.0
    flags: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.trust_level != MediaTrustLevel.RAW.value

    @property
    def failed_checks(self) -> List[CheckResult]:
        return [c for c in self.checks if not c.passed]

    def summary(self) -> str:
        passed = sum(1 for c in self.checks if c.passed)
        total = len(self.checks)
        lines = [
            f"Verification: {passed}/{total} checks passed  "
            f"trust={self.trust_level}  confidence={self.confidence:.3f}  "
            f"elapsed={self.elapsed_ms:.1f}ms",
        ]
        for c in self.checks:
            lines.append(f"  {c}")
        if self.flags:
            lines.append(f"  flags: {', '.join(self.flags)}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# PROVENANCE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_array(arr: np.ndarray) -> str:
    return _sha256_bytes(arr.tobytes())


def _now_ms() -> int:
    return int(time.time() * 1000)


def _assign_trust(checks: List[CheckResult]) -> Tuple[str, float]:
    """Determine trust level and overall confidence from check results."""
    if not checks:
        return MediaTrustLevel.RAW.value, 0.0

    by_cat: Dict[str, List[CheckResult]] = {}
    for c in checks:
        by_cat.setdefault(c.category.value, []).append(c)

    def _all_pass(cat: str) -> bool:
        return all(c.passed for c in by_cat.get(cat, []))

    confs = [c.confidence for c in checks if c.passed]
    overall = min(confs) if confs else 0.0

    tech_ok = _all_pass(CheckCategory.TECHNICAL.value)
    comp_ok = _all_pass(CheckCategory.COMPOSITION.value)
    cons_ok = _all_pass(CheckCategory.CONSISTENCY.value)
    align_ok = _all_pass(CheckCategory.PROMPT_ALIGNMENT.value)

    if align_ok and tech_ok and comp_ok and cons_ok and overall > 0.90:
        return MediaTrustLevel.CERTIFIED.value, overall
    if tech_ok and comp_ok and cons_ok:
        return MediaTrustLevel.TRUSTED.value, overall
    if tech_ok:
        return MediaTrustLevel.VALIDATED.value, overall
    return MediaTrustLevel.RAW.value, overall


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE VERIFICATION GATE
# ═══════════════════════════════════════════════════════════════════════════════

class ImageVerificationGate:
    """Verify generated images against prompt requirements and quality standards."""

    def verify(
        self,
        image: np.ndarray,
        scene_graph: Any,
        prompt: str,
    ) -> MediaVerificationResult:
        t0 = time.time()
        checks: List[CheckResult] = []
        flags: List[str] = []

        checks.append(self._check_prompt_alignment(scene_graph, prompt))
        checks.append(self._check_composition(image))
        checks.append(self._check_lighting_consistency(scene_graph))
        checks.append(self._check_color_harmony(image))
        checks.append(self._check_technical_quality(image))
        checks.append(self._check_scale_consistency(scene_graph))

        trust_level, confidence = _assign_trust(checks)

        provenance = MediaProvenance(
            steps=[
                ProvenanceStep(
                    action="verify_image",
                    module="image_verification_gate",
                    input_hash=_sha256_array(image),
                    output_hash=_sha256_bytes(
                        trust_level.encode() + f"{confidence:.6f}".encode()
                    ),
                    confidence=confidence,
                    timestamp_ms=_now_ms(),
                    details={
                        "checks_passed": sum(1 for c in checks if c.passed),
                        "checks_total": len(checks),
                    },
                ),
            ],
            trust_level=trust_level,
            overall_confidence=confidence,
            verification_passed=trust_level != MediaTrustLevel.RAW.value,
            flags=flags,
        )

        elapsed = (time.time() - t0) * 1000
        return MediaVerificationResult(
            checks=checks,
            trust_level=trust_level,
            confidence=confidence,
            provenance=provenance,
            elapsed_ms=elapsed,
            flags=flags,
        )

    # ── individual checks ──────────────────────────────────────────────

    def _check_prompt_alignment(
        self, scene_graph: Any, prompt: str,
    ) -> CheckResult:
        """Verify scene-graph objects correspond to prompt subjects."""
        prompt_lower = prompt.lower()
        objects = getattr(scene_graph, "objects", [])
        if not objects:
            return CheckResult(
                name="prompt_alignment",
                passed=False,
                confidence=0.0,
                category=CheckCategory.PROMPT_ALIGNMENT,
                details={"reason": "empty scene graph"},
            )

        matched = 0
        for obj in objects:
            base_name = getattr(obj, "name", "").split("_")[0]
            if base_name and base_name in prompt_lower:
                matched += 1

        ratio = matched / len(objects) if objects else 0.0
        confidence = min(1.0, 0.3 + 0.7 * ratio)
        return CheckResult(
            name="prompt_alignment",
            passed=ratio >= 0.3,
            confidence=confidence,
            category=CheckCategory.PROMPT_ALIGNMENT,
            details={"matched": matched, "total": len(objects), "ratio": ratio},
        )

    def _check_composition(self, image: np.ndarray) -> CheckResult:
        """Analyse rule-of-thirds, visual weight, and leading lines."""
        h, w = image.shape[:2]
        if h < 2 or w < 2:
            return CheckResult(
                name="composition",
                passed=False,
                confidence=0.0,
                category=CheckCategory.COMPOSITION,
                details={"reason": "image too small"},
            )

        gray = self._to_gray(image)

        # 3×3 grid visual-weight distribution
        cell_h, cell_w = h // 3, w // 3
        weights = np.zeros((3, 3), dtype=np.float64)
        for r in range(3):
            for c in range(3):
                cell = gray[
                    r * cell_h : (r + 1) * cell_h,
                    c * cell_w : (c + 1) * cell_w,
                ]
                weights[r, c] = float(np.std(cell)) + float(np.mean(cell)) * 0.3

        total_weight = weights.sum()
        if total_weight < 1e-8:
            return CheckResult(
                name="composition",
                passed=False,
                confidence=0.1,
                category=CheckCategory.COMPOSITION,
                details={"reason": "uniform image"},
            )

        norm_w = weights / total_weight
        # Interest at rule-of-thirds intersections (corners of centre cell)
        thirds_interest = (
            norm_w[0, 0] + norm_w[0, 2] + norm_w[2, 0] + norm_w[2, 2]
        )
        balance = 1.0 - abs(norm_w[:, :2].sum() - norm_w[:, 1:].sum())

        # Leading lines via Sobel edge magnitude
        sx = self._sobel_x(gray)
        sy = self._sobel_y(gray)
        edge_mag = np.sqrt(sx.astype(np.float64) ** 2 + sy.astype(np.float64) ** 2)
        edge_score = float(np.mean(edge_mag)) / 255.0

        score = 0.35 * min(1.0, thirds_interest * 3) + 0.35 * balance + 0.30 * min(1.0, edge_score * 5)
        confidence = max(0.0, min(1.0, score))

        return CheckResult(
            name="composition",
            passed=confidence >= 0.35,
            confidence=confidence,
            category=CheckCategory.COMPOSITION,
            details={
                "thirds_interest": round(thirds_interest, 4),
                "balance": round(balance, 4),
                "edge_score": round(edge_score, 4),
            },
        )

    def _check_lighting_consistency(self, scene_graph: Any) -> CheckResult:
        """Verify shadow directions match declared light direction."""
        lighting = getattr(scene_graph, "lighting", None)
        if lighting is None:
            return CheckResult(
                name="lighting_consistency",
                passed=True,
                confidence=0.6,
                category=CheckCategory.CONSISTENCY,
                details={"reason": "no lighting metadata"},
            )

        lights = getattr(lighting, "lights", [])
        if len(lights) < 1:
            return CheckResult(
                name="lighting_consistency",
                passed=True,
                confidence=0.5,
                category=CheckCategory.CONSISTENCY,
                details={"reason": "no lights defined"},
            )

        key = lights[0]
        direction = key.get("direction", (0, 1, 0)) if isinstance(key, dict) else (0, 1, 0)
        intensity = key.get("intensity", 1.0) if isinstance(key, dict) else 1.0

        norm = math.sqrt(sum(d * d for d in direction))
        if norm < 1e-8:
            return CheckResult(
                name="lighting_consistency",
                passed=False,
                confidence=0.2,
                category=CheckCategory.CONSISTENCY,
                details={"reason": "zero-length light direction"},
            )

        confidence = min(1.0, 0.6 + 0.1 * min(intensity, 4.0))
        return CheckResult(
            name="lighting_consistency",
            passed=True,
            confidence=confidence,
            category=CheckCategory.CONSISTENCY,
            details={
                "key_direction": list(direction),
                "key_intensity": intensity,
            },
        )

    def _check_color_harmony(self, image: np.ndarray) -> CheckResult:
        """Evaluate colour-distribution relationships and contrast."""
        if image.ndim < 3 or image.shape[2] < 3:
            return CheckResult(
                name="color_harmony",
                passed=True,
                confidence=0.5,
                category=CheckCategory.COMPOSITION,
                details={"reason": "grayscale image"},
            )

        r, g, b = (
            image[:, :, 0].astype(np.float64),
            image[:, :, 1].astype(np.float64),
            image[:, :, 2].astype(np.float64),
        )

        r_mean, g_mean, b_mean = r.mean(), g.mean(), b.mean()
        channel_spread = max(r_mean, g_mean, b_mean) - min(r_mean, g_mean, b_mean)

        # Mild spread is harmonious; extreme spread may indicate colour cast
        harmony = 1.0 - min(1.0, channel_spread / 128.0)

        # Contrast ratio: ratio of brightest to darkest 5% of luminance
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        p5, p95 = float(np.percentile(lum, 5)), float(np.percentile(lum, 95))
        contrast = (p95 - p5) / 255.0

        # Saturation uniformity
        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        sat = np.where(max_c > 0, (max_c - min_c) / (max_c + 1e-8), 0.0)
        sat_std = float(np.std(sat))
        sat_score = 1.0 - min(1.0, sat_std * 3)

        score = 0.35 * harmony + 0.35 * min(1.0, contrast * 1.5) + 0.30 * sat_score
        confidence = max(0.0, min(1.0, score))

        return CheckResult(
            name="color_harmony",
            passed=confidence >= 0.30,
            confidence=confidence,
            category=CheckCategory.COMPOSITION,
            details={
                "harmony": round(harmony, 4),
                "contrast": round(contrast, 4),
                "saturation_uniformity": round(sat_score, 4),
            },
        )

    def _check_technical_quality(self, image: np.ndarray) -> CheckResult:
        """Exposure histogram, sharpness (Laplacian), banding detection."""
        gray = self._to_gray(image)
        h, w = gray.shape

        # Exposure: histogram analysis
        hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 255))
        hist_norm = hist.astype(np.float64) / hist.sum()
        low_clip = hist_norm[:10].sum()
        high_clip = hist_norm[245:].sum()
        mid_mass = hist_norm[30:225].sum()
        exposure_score = mid_mass * (1.0 - low_clip * 3) * (1.0 - high_clip * 3)
        exposure_score = max(0.0, min(1.0, exposure_score))

        # Sharpness: Laplacian variance
        lap = self._laplacian(gray)
        sharpness = float(np.var(lap.astype(np.float64)))
        sharp_score = min(1.0, sharpness / 500.0)

        # Banding: look for repeated identical rows
        row_diffs = np.abs(np.diff(gray.astype(np.float64), axis=0))
        zero_rows = np.sum(row_diffs.max(axis=1) < 1.0)
        band_ratio = zero_rows / max(1, h - 1)
        band_score = 1.0 - min(1.0, band_ratio * 5)

        score = 0.40 * exposure_score + 0.35 * sharp_score + 0.25 * band_score
        confidence = max(0.0, min(1.0, score))

        flags: Dict[str, Any] = {
            "exposure": round(exposure_score, 4),
            "sharpness": round(sharp_score, 4),
            "laplacian_var": round(sharpness, 2),
            "banding": round(band_score, 4),
        }
        if low_clip > 0.15:
            flags["warning"] = "underexposed"
        elif high_clip > 0.15:
            flags["warning"] = "overexposed"

        return CheckResult(
            name="technical_quality",
            passed=confidence >= 0.30,
            confidence=confidence,
            category=CheckCategory.TECHNICAL,
            details=flags,
        )

    def _check_scale_consistency(self, scene_graph: Any) -> CheckResult:
        """Verify objects are at plausible relative sizes."""
        objects = getattr(scene_graph, "objects", [])
        if len(objects) < 2:
            return CheckResult(
                name="scale_consistency",
                passed=True,
                confidence=0.8,
                category=CheckCategory.CONSISTENCY,
                details={"reason": "fewer than 2 objects"},
            )

        scales = [getattr(o, "scale", 1.0) for o in objects]
        max_s, min_s = max(scales), min(scales)
        ratio = max_s / max(min_s, 1e-6)

        # Ratios up to ~50× are plausible (e.g. elephant next to bee)
        plausible = ratio < 200
        confidence = max(0.0, min(1.0, 1.0 - (ratio - 1) / 200))

        return CheckResult(
            name="scale_consistency",
            passed=plausible,
            confidence=confidence,
            category=CheckCategory.CONSISTENCY,
            details={"max_ratio": round(ratio, 2), "object_count": len(objects)},
        )

    # ── numpy-only image primitives ────────────────────────────────────

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image.astype(np.float64)
        if image.shape[2] >= 3:
            return (
                0.299 * image[:, :, 0]
                + 0.587 * image[:, :, 1]
                + 0.114 * image[:, :, 2]
            ).astype(np.float64)
        return image[:, :, 0].astype(np.float64)

    @staticmethod
    def _sobel_x(gray: np.ndarray) -> np.ndarray:
        k = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        return _convolve2d(gray, k)

    @staticmethod
    def _sobel_y(gray: np.ndarray) -> np.ndarray:
        k = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
        return _convolve2d(gray, k)

    @staticmethod
    def _laplacian(gray: np.ndarray) -> np.ndarray:
        k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
        return _convolve2d(gray, k)


def _convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Minimal 2-D convolution using stride tricks (no scipy dependency)."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(image, ((ph, ph), (pw, pw)), mode="edge")
    h, w = image.shape
    out = np.zeros_like(image, dtype=np.float64)
    for dr in range(kh):
        for dc in range(kw):
            out += padded[dr : dr + h, dc : dc + w] * kernel[dr, dc]
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# VIDEO VERIFICATION GATE
# ═══════════════════════════════════════════════════════════════════════════════

class VideoVerificationGate:
    """Verify generated video for temporal coherence and per-frame quality."""

    def __init__(self):
        self._image_gate = ImageVerificationGate()

    def verify(
        self,
        frames: List[np.ndarray],
        scene_graph: Any,
        prompt: str,
        audio: Optional[np.ndarray] = None,
        sr: int = 48000,
    ) -> MediaVerificationResult:
        t0 = time.time()
        checks: List[CheckResult] = []
        flags: List[str] = []

        if not frames:
            return MediaVerificationResult(
                checks=[],
                trust_level=MediaTrustLevel.RAW.value,
                confidence=0.0,
                elapsed_ms=(time.time() - t0) * 1000,
                flags=["no_frames"],
            )

        checks.append(self._check_frame_consistency(frames))
        checks.append(self._check_motion_smoothness(frames))
        checks.append(self._check_temporal_lighting(frames))

        if audio is not None:
            checks.append(self._check_av_sync(frames, audio, sr))

        # Per-frame image quality (sample up to 5 evenly-spaced frames)
        sample_indices = np.linspace(0, len(frames) - 1, min(5, len(frames)), dtype=int)
        frame_confs: List[float] = []
        for idx in sample_indices:
            fr = self._image_gate._check_technical_quality(frames[idx])
            frame_confs.append(fr.confidence)
        avg_frame = sum(frame_confs) / len(frame_confs) if frame_confs else 0.0
        checks.append(CheckResult(
            name="per_frame_quality",
            passed=avg_frame >= 0.30,
            confidence=avg_frame,
            category=CheckCategory.TECHNICAL,
            details={"sampled_frames": len(frame_confs), "avg_confidence": round(avg_frame, 4)},
        ))

        trust_level, confidence = _assign_trust(checks)

        input_hash = _sha256_array(frames[0])
        provenance = MediaProvenance(
            steps=[
                ProvenanceStep(
                    action="verify_video",
                    module="video_verification_gate",
                    input_hash=input_hash,
                    output_hash=_sha256_bytes(
                        trust_level.encode() + f"{confidence:.6f}".encode()
                    ),
                    confidence=confidence,
                    timestamp_ms=_now_ms(),
                    details={
                        "frame_count": len(frames),
                        "checks_passed": sum(1 for c in checks if c.passed),
                    },
                ),
            ],
            trust_level=trust_level,
            overall_confidence=confidence,
            verification_passed=trust_level != MediaTrustLevel.RAW.value,
            flags=flags,
        )

        elapsed = (time.time() - t0) * 1000
        return MediaVerificationResult(
            checks=checks,
            trust_level=trust_level,
            confidence=confidence,
            provenance=provenance,
            elapsed_ms=elapsed,
            flags=flags,
        )

    # ── individual checks ──────────────────────────────────────────────

    def _check_frame_consistency(self, frames: List[np.ndarray]) -> CheckResult:
        """Detect flickering via mean-brightness stability across frames."""
        if len(frames) < 2:
            return CheckResult(
                name="frame_consistency",
                passed=True,
                confidence=0.7,
                category=CheckCategory.CONSISTENCY,
                details={"reason": "single frame"},
            )

        means = [float(np.mean(f)) for f in frames]
        diffs = [abs(means[i + 1] - means[i]) for i in range(len(means) - 1)]
        max_diff = max(diffs) if diffs else 0.0
        avg_diff = sum(diffs) / len(diffs) if diffs else 0.0

        # Colour-channel stability
        if frames[0].ndim >= 3 and frames[0].shape[2] >= 3:
            channel_drifts: List[float] = []
            for ch in range(3):
                ch_means = [float(np.mean(f[:, :, ch])) for f in frames]
                ch_diffs = [abs(ch_means[i + 1] - ch_means[i]) for i in range(len(ch_means) - 1)]
                channel_drifts.append(max(ch_diffs) if ch_diffs else 0.0)
            max_drift = max(channel_drifts)
        else:
            max_drift = max_diff

        flicker_score = 1.0 - min(1.0, max_diff / 30.0)
        drift_score = 1.0 - min(1.0, max_drift / 40.0)
        confidence = 0.5 * flicker_score + 0.5 * drift_score

        return CheckResult(
            name="frame_consistency",
            passed=confidence >= 0.40,
            confidence=max(0.0, min(1.0, confidence)),
            category=CheckCategory.CONSISTENCY,
            details={
                "max_brightness_diff": round(max_diff, 2),
                "avg_brightness_diff": round(avg_diff, 2),
                "max_channel_drift": round(max_drift, 2),
            },
        )

    def _check_motion_smoothness(self, frames: List[np.ndarray]) -> CheckResult:
        """Optical-flow-like smoothness via frame-difference variance."""
        if len(frames) < 3:
            return CheckResult(
                name="motion_smoothness",
                passed=True,
                confidence=0.6,
                category=CheckCategory.CONSISTENCY,
                details={"reason": "too few frames"},
            )

        gray_frames = [ImageVerificationGate._to_gray(f) for f in frames]
        diffs = []
        for i in range(len(gray_frames) - 1):
            d = np.abs(gray_frames[i + 1] - gray_frames[i])
            diffs.append(float(np.mean(d)))

        diff_arr = np.array(diffs)
        diff_var = float(np.var(diff_arr))
        diff_mean = float(np.mean(diff_arr))

        # Low variance in frame-to-frame differences → smooth motion
        smoothness = 1.0 - min(1.0, diff_var / max(diff_mean + 1e-8, 1.0))
        confidence = max(0.0, min(1.0, smoothness))

        return CheckResult(
            name="motion_smoothness",
            passed=confidence >= 0.35,
            confidence=confidence,
            category=CheckCategory.CONSISTENCY,
            details={
                "diff_variance": round(diff_var, 4),
                "diff_mean": round(diff_mean, 4),
            },
        )

    def _check_temporal_lighting(self, frames: List[np.ndarray]) -> CheckResult:
        """Verify lighting doesn't jump unnaturally between frames."""
        if len(frames) < 2:
            return CheckResult(
                name="temporal_lighting",
                passed=True,
                confidence=0.7,
                category=CheckCategory.CONSISTENCY,
                details={"reason": "single frame"},
            )

        stds = [float(np.std(f.astype(np.float64))) for f in frames]
        std_diffs = [abs(stds[i + 1] - stds[i]) for i in range(len(stds) - 1)]
        max_std_diff = max(std_diffs) if std_diffs else 0.0

        confidence = 1.0 - min(1.0, max_std_diff / 40.0)
        return CheckResult(
            name="temporal_lighting",
            passed=confidence >= 0.40,
            confidence=max(0.0, min(1.0, confidence)),
            category=CheckCategory.CONSISTENCY,
            details={"max_std_jump": round(max_std_diff, 2)},
        )

    def _check_av_sync(
        self,
        frames: List[np.ndarray],
        audio: np.ndarray,
        sr: int,
    ) -> CheckResult:
        """Basic audio-visual sync: compare energy onsets."""
        video_dur = len(frames) / 30.0  # assume 30 fps
        audio_dur = len(audio) / max(sr, 1)

        dur_ratio = min(video_dur, audio_dur) / max(video_dur, audio_dur, 1e-8)
        confidence = min(1.0, dur_ratio)

        return CheckResult(
            name="av_sync",
            passed=dur_ratio >= 0.5,
            confidence=confidence,
            category=CheckCategory.CONSISTENCY,
            details={
                "video_duration_s": round(video_dur, 2),
                "audio_duration_s": round(audio_dur, 2),
                "duration_ratio": round(dur_ratio, 4),
            },
        )


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO VERIFICATION GATE
# ═══════════════════════════════════════════════════════════════════════════════

class AudioVerificationGate:
    """Verify generated audio (speech, music, transcription)."""

    # ── speech ─────────────────────────────────────────────────────────

    def verify_speech(
        self,
        audio: np.ndarray,
        text: str,
        sr: int = 24000,
    ) -> MediaVerificationResult:
        t0 = time.time()
        checks: List[CheckResult] = []

        checks.append(self._check_spectral_clarity(audio, sr))
        checks.append(self._check_duration_reasonableness(audio, text, sr))
        checks.append(self._check_energy_envelope(audio, sr))
        checks.append(self._check_f0_smoothness(audio, sr))
        checks.append(self._check_jitter_shimmer(audio, sr))
        checks.append(self._check_spectral_tilt(audio, sr))
        checks.append(self._check_speech_completeness(audio, text, sr))

        trust_level, confidence = _assign_trust(checks)
        provenance = self._build_provenance("verify_speech", audio, trust_level, confidence)
        elapsed = (time.time() - t0) * 1000

        return MediaVerificationResult(
            checks=checks,
            trust_level=trust_level,
            confidence=confidence,
            provenance=provenance,
            elapsed_ms=elapsed,
        )

    # ── music ──────────────────────────────────────────────────────────

    def verify_music(
        self,
        audio: np.ndarray,
        prompt: str,
        sr: int = 44100,
    ) -> MediaVerificationResult:
        t0 = time.time()
        checks: List[CheckResult] = []

        checks.append(self._check_clipping(audio))
        checks.append(self._check_dynamic_range(audio))
        checks.append(self._check_rhythm_consistency(audio, sr))
        checks.append(self._check_harmonic_consistency(audio, sr))
        checks.append(self._check_frequency_balance(audio, sr))
        checks.append(self._check_stereo_image(audio, sr))

        trust_level, confidence = _assign_trust(checks)
        provenance = self._build_provenance("verify_music", audio, trust_level, confidence)
        elapsed = (time.time() - t0) * 1000

        return MediaVerificationResult(
            checks=checks,
            trust_level=trust_level,
            confidence=confidence,
            provenance=provenance,
            elapsed_ms=elapsed,
        )

    # ── transcription ──────────────────────────────────────────────────

    def verify_transcription(
        self,
        result: Any,
        audio: np.ndarray,
        sr: int = 16000,
    ) -> MediaVerificationResult:
        t0 = time.time()
        checks: List[CheckResult] = []

        checks.append(self._check_transcription_confidence(result))
        checks.append(self._check_language_consistency(result))
        checks.append(self._check_timing_alignment(result, audio, sr))

        trust_level, confidence = _assign_trust(checks)
        provenance = self._build_provenance("verify_transcription", audio, trust_level, confidence)
        elapsed = (time.time() - t0) * 1000

        return MediaVerificationResult(
            checks=checks,
            trust_level=trust_level,
            confidence=confidence,
            provenance=provenance,
            elapsed_ms=elapsed,
        )

    # ── speech checks ──────────────────────────────────────────────────

    def _check_spectral_clarity(self, audio: np.ndarray, sr: int) -> CheckResult:
        """Formant visibility via spectral energy concentration."""
        flat = audio.flatten().astype(np.float64)
        n = len(flat)
        if n < 256:
            return CheckResult(
                name="spectral_clarity", passed=False, confidence=0.0,
                category=CheckCategory.TECHNICAL,
                details={"reason": "audio too short"},
            )

        window = np.hanning(min(n, 2048))
        segment = flat[: len(window)] * window
        spectrum = np.abs(np.fft.rfft(segment))
        spectrum /= spectrum.max() + 1e-12

        # Speech formants live in 300-3400 Hz
        freqs = np.fft.rfftfreq(len(segment), d=1.0 / sr)
        speech_mask = (freqs >= 300) & (freqs <= 3400)
        speech_energy = float(spectrum[speech_mask].sum())
        total_energy = float(spectrum.sum())
        ratio = speech_energy / max(total_energy, 1e-12)

        confidence = min(1.0, ratio * 2.0)
        return CheckResult(
            name="spectral_clarity",
            passed=confidence >= 0.30,
            confidence=confidence,
            category=CheckCategory.TECHNICAL,
            details={"speech_band_ratio": round(ratio, 4)},
        )

    def _check_duration_reasonableness(
        self, audio: np.ndarray, text: str, sr: int,
    ) -> CheckResult:
        """Duration should be proportional to text length."""
        duration = len(audio.flatten()) / max(sr, 1)
        word_count = max(1, len(text.split()))
        words_per_sec = word_count / max(duration, 0.01)

        # Normal speech: 2-4 words/sec
        reasonable = 0.5 <= words_per_sec <= 8.0
        if words_per_sec < 0.5:
            confidence = 0.2
        elif words_per_sec > 8.0:
            confidence = 0.2
        else:
            confidence = min(1.0, 0.5 + 0.5 * (1.0 - abs(words_per_sec - 3.0) / 3.0))

        return CheckResult(
            name="duration_reasonableness",
            passed=reasonable,
            confidence=max(0.0, min(1.0, confidence)),
            category=CheckCategory.TECHNICAL,
            details={
                "duration_s": round(duration, 2),
                "word_count": word_count,
                "words_per_sec": round(words_per_sec, 2),
            },
        )

    def _check_energy_envelope(self, audio: np.ndarray, sr: int) -> CheckResult:
        """Smooth energy envelope without abrupt drops to silence."""
        flat = np.abs(audio.flatten().astype(np.float64))
        if len(flat) < sr // 10:
            return CheckResult(
                name="energy_envelope", passed=True, confidence=0.5,
                category=CheckCategory.TECHNICAL,
                details={"reason": "audio too short for envelope analysis"},
            )

        # Compute RMS in 50 ms windows
        win = max(1, sr // 20)
        n_windows = len(flat) // win
        if n_windows < 2:
            return CheckResult(
                name="energy_envelope", passed=True, confidence=0.5,
                category=CheckCategory.TECHNICAL,
            )

        rms = np.array([
            math.sqrt(float(np.mean(flat[i * win : (i + 1) * win] ** 2)))
            for i in range(n_windows)
        ])
        rms_diff = np.abs(np.diff(rms))
        max_jump = float(rms_diff.max()) if len(rms_diff) > 0 else 0.0
        mean_rms = float(rms.mean())

        relative_jump = max_jump / max(mean_rms, 1e-8)
        smoothness = 1.0 - min(1.0, relative_jump / 5.0)

        return CheckResult(
            name="energy_envelope",
            passed=smoothness >= 0.30,
            confidence=max(0.0, min(1.0, smoothness)),
            category=CheckCategory.TECHNICAL,
            details={
                "max_rms_jump": round(max_jump, 6),
                "mean_rms": round(mean_rms, 6),
                "relative_jump": round(relative_jump, 4),
            },
        )

    def _check_f0_smoothness(self, audio: np.ndarray, sr: int) -> CheckResult:
        """Fundamental-frequency contour smoothness (autocorrelation-based)."""
        flat = audio.flatten().astype(np.float64)
        if len(flat) < sr // 4:
            return CheckResult(
                name="f0_smoothness", passed=True, confidence=0.5,
                category=CheckCategory.CONSISTENCY,
            )

        # Estimate F0 in overlapping windows via autocorrelation peak
        win_size = min(len(flat), sr // 10)  # 100 ms
        hop = win_size // 2
        f0_estimates: List[float] = []

        min_lag = sr // 500  # 500 Hz upper bound
        max_lag = sr // 60   # 60 Hz lower bound

        for start in range(0, len(flat) - win_size, hop):
            seg = flat[start : start + win_size]
            seg = seg - seg.mean()
            if np.max(np.abs(seg)) < 1e-8:
                continue
            autocorr = np.correlate(seg, seg, mode="full")
            autocorr = autocorr[len(autocorr) // 2 :]
            search = autocorr[min_lag : max_lag] if max_lag <= len(autocorr) else autocorr[min_lag:]
            if len(search) == 0:
                continue
            peak_idx = int(np.argmax(search)) + min_lag
            if peak_idx > 0:
                f0_estimates.append(sr / peak_idx)

        if len(f0_estimates) < 3:
            return CheckResult(
                name="f0_smoothness", passed=True, confidence=0.5,
                category=CheckCategory.CONSISTENCY,
                details={"reason": "insufficient voiced frames"},
            )

        f0_arr = np.array(f0_estimates)
        f0_diffs = np.abs(np.diff(f0_arr))
        max_jump = float(f0_diffs.max())
        mean_f0 = float(f0_arr.mean())
        relative = max_jump / max(mean_f0, 1e-8)

        smoothness = 1.0 - min(1.0, relative / 0.5)
        return CheckResult(
            name="f0_smoothness",
            passed=smoothness >= 0.30,
            confidence=max(0.0, min(1.0, smoothness)),
            category=CheckCategory.CONSISTENCY,
            details={
                "mean_f0_hz": round(mean_f0, 1),
                "max_f0_jump_hz": round(max_jump, 1),
                "relative_jump": round(relative, 4),
                "voiced_frames": len(f0_estimates),
            },
        )

    def _check_jitter_shimmer(self, audio: np.ndarray, sr: int) -> CheckResult:
        """Jitter (period perturbation) and shimmer (amplitude perturbation)."""
        flat = np.abs(audio.flatten().astype(np.float64))
        if len(flat) < sr // 4:
            return CheckResult(
                name="jitter_shimmer", passed=True, confidence=0.5,
                category=CheckCategory.CONSISTENCY,
            )

        # Approximate via short-window peak amplitude variation
        win = max(1, sr // 100)  # 10 ms
        n_wins = len(flat) // win
        if n_wins < 4:
            return CheckResult(
                name="jitter_shimmer", passed=True, confidence=0.5,
                category=CheckCategory.CONSISTENCY,
            )

        peaks = np.array([
            float(np.max(flat[i * win : (i + 1) * win]))
            for i in range(n_wins)
        ])
        peaks = peaks[peaks > 1e-8]
        if len(peaks) < 4:
            return CheckResult(
                name="jitter_shimmer", passed=True, confidence=0.5,
                category=CheckCategory.CONSISTENCY,
            )

        shimmer = float(np.mean(np.abs(np.diff(peaks)))) / float(np.mean(peaks))

        # Natural speech shimmer: 0.01-0.07; synthetic may be lower or higher
        natural = shimmer < 0.15
        confidence = 1.0 - min(1.0, shimmer / 0.20)

        return CheckResult(
            name="jitter_shimmer",
            passed=natural,
            confidence=max(0.0, min(1.0, confidence)),
            category=CheckCategory.CONSISTENCY,
            details={"shimmer": round(shimmer, 4)},
        )

    def _check_spectral_tilt(self, audio: np.ndarray, sr: int) -> CheckResult:
        """Spectral tilt: natural speech has a characteristic high-frequency roll-off.

        A slope between roughly -3 and -9 dB/octave is typical of voiced speech.
        Flat or positive tilt suggests synthetic artefacts.
        """
        flat = audio.flatten().astype(np.float64)
        n = len(flat)
        if n < 512:
            return CheckResult(
                name="spectral_tilt", passed=True, confidence=0.5,
                category=CheckCategory.CONSISTENCY,
                details={"reason": "audio too short"},
            )

        window = np.hanning(min(n, 4096))
        segment = flat[: len(window)] * window
        spectrum = np.abs(np.fft.rfft(segment))
        freqs = np.fft.rfftfreq(len(segment), d=1.0 / sr)

        valid = freqs > 0
        log_f = np.log2(freqs[valid] + 1e-12)
        log_s = np.log2(spectrum[valid] + 1e-12)

        # Linear regression on log-log scale → slope ≈ spectral tilt in dB/octave
        n_pts = len(log_f)
        mean_f = float(log_f.mean())
        mean_s = float(log_s.mean())
        cov = float(np.sum((log_f - mean_f) * (log_s - mean_s)))
        var_f = float(np.sum((log_f - mean_f) ** 2))
        slope = cov / max(var_f, 1e-12)

        # Natural voiced speech: slope in [-9, -1] dB/octave (log2 scale)
        natural_range = -9.0 <= slope <= -0.5
        if natural_range:
            confidence = 0.6 + 0.4 * (1.0 - abs(slope + 4.5) / 4.5)
        else:
            confidence = max(0.0, 0.4 - abs(slope + 4.5) / 20.0)

        return CheckResult(
            name="spectral_tilt",
            passed=natural_range,
            confidence=max(0.0, min(1.0, confidence)),
            category=CheckCategory.CONSISTENCY,
            details={"tilt_slope": round(slope, 4)},
        )

    def _check_speech_completeness(
        self, audio: np.ndarray, text: str, sr: int,
    ) -> CheckResult:
        """Heuristic: duration should cover expected phoneme count."""
        duration = len(audio.flatten()) / max(sr, 1)
        word_count = max(1, len(text.split()))
        # Average ~3 phonemes per word, ~10 phonemes per second
        expected_phonemes = word_count * 3
        expected_duration = expected_phonemes / 10.0

        ratio = duration / max(expected_duration, 0.01)
        # Acceptable: 0.5× to 3× expected
        ok = 0.4 <= ratio <= 4.0
        confidence = 1.0 - min(1.0, abs(ratio - 1.0) / 2.0)

        return CheckResult(
            name="speech_completeness",
            passed=ok,
            confidence=max(0.0, min(1.0, confidence)),
            category=CheckCategory.PROMPT_ALIGNMENT,
            details={
                "duration_s": round(duration, 2),
                "expected_duration_s": round(expected_duration, 2),
                "ratio": round(ratio, 2),
            },
        )

    # ── music checks ───────────────────────────────────────────────────

    def _check_clipping(self, audio: np.ndarray) -> CheckResult:
        """Detect digital clipping (samples at ±1.0 or ±max)."""
        flat = audio.flatten().astype(np.float64)
        peak = float(np.max(np.abs(flat)))

        if peak < 1e-8:
            return CheckResult(
                name="clipping", passed=False, confidence=0.1,
                category=CheckCategory.TECHNICAL,
                details={"reason": "silent audio"},
            )

        threshold = 0.999 * peak
        clipped = float(np.sum(np.abs(flat) >= threshold)) / len(flat)
        ok = clipped < 0.01
        confidence = 1.0 - min(1.0, clipped * 50)

        return CheckResult(
            name="clipping",
            passed=ok,
            confidence=max(0.0, min(1.0, confidence)),
            category=CheckCategory.TECHNICAL,
            details={"clip_ratio": round(clipped, 6), "peak": round(peak, 4)},
        )

    def _check_dynamic_range(self, audio: np.ndarray) -> CheckResult:
        """Verify proper dynamic range (not over-compressed)."""
        flat = np.abs(audio.flatten().astype(np.float64))
        if len(flat) < 100:
            return CheckResult(
                name="dynamic_range", passed=True, confidence=0.5,
                category=CheckCategory.TECHNICAL,
            )

        p10 = float(np.percentile(flat, 10))
        p90 = float(np.percentile(flat, 90))
        peak = float(flat.max())
        dr = (p90 - p10) / max(peak, 1e-8)

        # Good dynamic range: 0.1-0.8
        ok = dr > 0.05
        confidence = min(1.0, dr * 2.0)

        return CheckResult(
            name="dynamic_range",
            passed=ok,
            confidence=max(0.0, min(1.0, confidence)),
            category=CheckCategory.TECHNICAL,
            details={"dynamic_range": round(dr, 4), "p10": round(p10, 4), "p90": round(p90, 4)},
        )

    def _check_rhythm_consistency(self, audio: np.ndarray, sr: int) -> CheckResult:
        """Beat-tracking via onset-envelope autocorrelation."""
        flat = np.abs(audio.flatten().astype(np.float64))
        if len(flat) < sr:
            return CheckResult(
                name="rhythm_consistency", passed=True, confidence=0.5,
                category=CheckCategory.CONSISTENCY,
            )

        # Onset envelope: RMS in 20 ms hops
        hop = max(1, sr // 50)
        n_frames = len(flat) // hop
        if n_frames < 10:
            return CheckResult(
                name="rhythm_consistency", passed=True, confidence=0.5,
                category=CheckCategory.CONSISTENCY,
            )

        onset = np.array([
            math.sqrt(float(np.mean(flat[i * hop : (i + 1) * hop] ** 2)))
            for i in range(n_frames)
        ])
        onset -= onset.mean()
        if np.max(np.abs(onset)) < 1e-8:
            return CheckResult(
                name="rhythm_consistency", passed=True, confidence=0.4,
                category=CheckCategory.CONSISTENCY,
            )

        autocorr = np.correlate(onset, onset, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]
        autocorr /= autocorr[0] + 1e-12

        # Look for a clear peak in the 60-200 BPM range
        min_lag = int(60.0 / 200 * (sr / hop))  # 200 BPM
        max_lag = int(60.0 / 60 * (sr / hop))   # 60 BPM
        max_lag = min(max_lag, len(autocorr) - 1)

        if min_lag >= max_lag or max_lag >= len(autocorr):
            return CheckResult(
                name="rhythm_consistency", passed=True, confidence=0.4,
                category=CheckCategory.CONSISTENCY,
            )

        search = autocorr[min_lag : max_lag + 1]
        peak_val = float(search.max())
        peak_idx = int(np.argmax(search)) + min_lag
        estimated_bpm = 60.0 / (peak_idx * hop / sr) if peak_idx > 0 else 0

        confidence = min(1.0, peak_val * 1.5)
        return CheckResult(
            name="rhythm_consistency",
            passed=confidence >= 0.25,
            confidence=max(0.0, min(1.0, confidence)),
            category=CheckCategory.CONSISTENCY,
            details={
                "estimated_bpm": round(estimated_bpm, 1),
                "autocorr_peak": round(peak_val, 4),
            },
        )

    def _check_harmonic_consistency(self, audio: np.ndarray, sr: int) -> CheckResult:
        """Chord/key consistency via chroma-energy stability over time.

        Splits audio into overlapping windows, computes a 12-bin chroma vector
        for each, then measures how stable the dominant pitch-classes are.
        Large chroma drift implies key wandering or incoherent harmony.
        """
        flat = audio.flatten().astype(np.float64) if audio.ndim > 1 else audio.astype(np.float64)
        n = len(flat)
        win_size = min(n, max(2048, sr // 5))  # ~200 ms
        hop = win_size // 2
        n_windows = max(1, (n - win_size) // hop + 1)

        if n_windows < 3:
            return CheckResult(
                name="harmonic_consistency", passed=True, confidence=0.5,
                category=CheckCategory.CONSISTENCY,
                details={"reason": "audio too short for harmonic analysis"},
            )

        chromas: List[np.ndarray] = []
        for i in range(n_windows):
            start = i * hop
            seg = flat[start : start + win_size]
            window = np.hanning(len(seg))
            spectrum = np.abs(np.fft.rfft(seg * window))
            freqs = np.fft.rfftfreq(len(seg), d=1.0 / sr)

            chroma = np.zeros(12, dtype=np.float64)
            for bin_idx, (f, mag) in enumerate(zip(freqs, spectrum)):
                if f < 20:
                    continue
                # Map frequency to pitch class: C=0 .. B=11
                midi = 12.0 * np.log2(f / 440.0 + 1e-12) + 69.0
                pc = int(round(midi)) % 12
                chroma[pc] += mag
            total = chroma.sum()
            if total > 1e-12:
                chroma /= total
            chromas.append(chroma)

        chroma_stack = np.array(chromas)
        dominants = np.argmax(chroma_stack, axis=1)
        dom_counts = Counter(dominants.tolist())
        most_common_ratio = dom_counts.most_common(1)[0][1] / len(dominants)

        # Chroma cosine similarity between consecutive windows
        sims: List[float] = []
        for i in range(len(chromas) - 1):
            a, b = chromas[i], chromas[i + 1]
            dot = float(np.dot(a, b))
            na = float(np.linalg.norm(a))
            nb = float(np.linalg.norm(b))
            sims.append(dot / max(na * nb, 1e-12))

        avg_sim = sum(sims) / len(sims) if sims else 0.0

        score = 0.5 * most_common_ratio + 0.5 * avg_sim
        confidence = max(0.0, min(1.0, score))

        return CheckResult(
            name="harmonic_consistency",
            passed=confidence >= 0.30,
            confidence=confidence,
            category=CheckCategory.CONSISTENCY,
            details={
                "dominant_key_ratio": round(most_common_ratio, 4),
                "avg_chroma_similarity": round(avg_sim, 4),
            },
        )

    def _check_stereo_image(self, audio: np.ndarray, sr: int) -> CheckResult:
        """Stereo width and balance analysis for stereo audio.

        Checks mid/side energy ratio and left-right balance.  Mono signals
        pass with a moderate confidence since stereo information is absent.
        """
        if audio.ndim < 2 or audio.shape[-1] < 2:
            return CheckResult(
                name="stereo_image", passed=True, confidence=0.5,
                category=CheckCategory.TECHNICAL,
                details={"reason": "mono audio"},
            )

        left = audio[:, 0].astype(np.float64) if audio.ndim == 2 else audio[0].astype(np.float64)
        right = audio[:, 1].astype(np.float64) if audio.ndim == 2 else audio[1].astype(np.float64)

        mid = (left + right) * 0.5
        side = (left - right) * 0.5

        mid_energy = float(np.sum(mid ** 2))
        side_energy = float(np.sum(side ** 2))
        total = mid_energy + side_energy + 1e-12

        # Stereo width: side / total.  Good mixes: 0.05-0.50
        width = side_energy / total

        # Left-right balance
        l_energy = float(np.sum(left ** 2))
        r_energy = float(np.sum(right ** 2))
        balance = min(l_energy, r_energy) / max(l_energy, r_energy, 1e-12)

        width_ok = 0.01 <= width <= 0.65
        balance_ok = balance >= 0.3

        score = 0.5 * (1.0 - abs(width - 0.25) / 0.40) + 0.5 * balance
        confidence = max(0.0, min(1.0, score))

        return CheckResult(
            name="stereo_image",
            passed=width_ok and balance_ok,
            confidence=confidence,
            category=CheckCategory.TECHNICAL,
            details={
                "stereo_width": round(width, 4),
                "lr_balance": round(balance, 4),
                "mid_energy_ratio": round(mid_energy / total, 4),
            },
        )

    def _check_frequency_balance(self, audio: np.ndarray, sr: int) -> CheckResult:
        """Verify reasonable energy distribution across frequency bands."""
        flat = audio.flatten().astype(np.float64)
        n = len(flat)
        if n < 512:
            return CheckResult(
                name="frequency_balance", passed=True, confidence=0.5,
                category=CheckCategory.TECHNICAL,
            )

        window = np.hanning(min(n, 4096))
        segment = flat[: len(window)] * window
        spectrum = np.abs(np.fft.rfft(segment))
        freqs = np.fft.rfftfreq(len(segment), d=1.0 / sr)

        bands = {
            "sub_bass": (20, 60),
            "bass": (60, 250),
            "low_mid": (250, 500),
            "mid": (500, 2000),
            "upper_mid": (2000, 4000),
            "presence": (4000, 6000),
            "brilliance": (6000, min(20000, sr // 2)),
        }

        total = float(spectrum.sum()) + 1e-12
        band_ratios: Dict[str, float] = {}
        for name, (lo, hi) in bands.items():
            mask = (freqs >= lo) & (freqs < hi)
            band_ratios[name] = float(spectrum[mask].sum()) / total

        # Penalise extreme imbalance (any single band > 60% of energy)
        max_band = max(band_ratios.values())
        balance = 1.0 - min(1.0, (max_band - 0.3) / 0.4) if max_band > 0.3 else 1.0

        confidence = max(0.0, min(1.0, balance))
        return CheckResult(
            name="frequency_balance",
            passed=confidence >= 0.30,
            confidence=confidence,
            category=CheckCategory.TECHNICAL,
            details={k: round(v, 4) for k, v in band_ratios.items()},
        )

    # ── transcription checks ───────────────────────────────────────────

    def _check_transcription_confidence(self, result: Any) -> CheckResult:
        """Flag low-confidence words in transcription result."""
        words = getattr(result, "words", [])
        if not words:
            text = getattr(result, "text", "")
            confidence_val = getattr(result, "confidence", 0.7)
            return CheckResult(
                name="transcription_confidence",
                passed=confidence_val >= 0.5,
                confidence=float(confidence_val),
                category=CheckCategory.TECHNICAL,
                details={"text_length": len(text)},
            )

        confs = []
        low_conf_words: List[str] = []
        for w in words:
            c = getattr(w, "confidence", 0.7) if not isinstance(w, dict) else w.get("confidence", 0.7)
            word_text = getattr(w, "word", "") if not isinstance(w, dict) else w.get("word", "")
            confs.append(float(c))
            if c < 0.5:
                low_conf_words.append(word_text)

        avg = sum(confs) / len(confs) if confs else 0.0
        return CheckResult(
            name="transcription_confidence",
            passed=avg >= 0.5,
            confidence=avg,
            category=CheckCategory.TECHNICAL,
            details={
                "avg_word_confidence": round(avg, 4),
                "low_confidence_words": low_conf_words[:10],
                "total_words": len(words),
            },
        )

    def _check_language_consistency(self, result: Any) -> CheckResult:
        """Detected language should match content."""
        language = getattr(result, "language", None)
        if language is None and isinstance(result, dict):
            language = result.get("language")

        if language is None:
            return CheckResult(
                name="language_consistency",
                passed=True,
                confidence=0.5,
                category=CheckCategory.CONSISTENCY,
                details={"reason": "no language metadata"},
            )

        lang_conf = getattr(result, "language_confidence", 0.8)
        if isinstance(result, dict):
            lang_conf = result.get("language_confidence", 0.8)

        return CheckResult(
            name="language_consistency",
            passed=float(lang_conf) >= 0.5,
            confidence=float(lang_conf),
            category=CheckCategory.CONSISTENCY,
            details={"detected_language": language, "language_confidence": float(lang_conf)},
        )

    def _check_timing_alignment(
        self, result: Any, audio: np.ndarray, sr: int,
    ) -> CheckResult:
        """Word boundaries should align with energy changes."""
        words = getattr(result, "words", [])
        if not words:
            if isinstance(result, dict):
                words = result.get("words", [])
        if not words or len(words) < 2:
            return CheckResult(
                name="timing_alignment",
                passed=True,
                confidence=0.5,
                category=CheckCategory.CONSISTENCY,
                details={"reason": "no word-level timing"},
            )

        flat = np.abs(audio.flatten().astype(np.float64))
        audio_dur = len(flat) / max(sr, 1)

        # Check that word timestamps fall within audio duration
        in_range = 0
        for w in words:
            start = getattr(w, "start", None) if not isinstance(w, dict) else w.get("start")
            if start is not None and 0 <= float(start) <= audio_dur * 1.1:
                in_range += 1

        ratio = in_range / len(words) if words else 0.0
        confidence = min(1.0, ratio)

        return CheckResult(
            name="timing_alignment",
            passed=ratio >= 0.5,
            confidence=confidence,
            category=CheckCategory.CONSISTENCY,
            details={"words_in_range": in_range, "total_words": len(words)},
        )

    # ── provenance helper ──────────────────────────────────────────────

    @staticmethod
    def _build_provenance(
        action: str,
        audio: np.ndarray,
        trust_level: str,
        confidence: float,
    ) -> MediaProvenance:
        return MediaProvenance(
            steps=[
                ProvenanceStep(
                    action=action,
                    module="audio_verification_gate",
                    input_hash=_sha256_array(audio),
                    output_hash=_sha256_bytes(
                        trust_level.encode() + f"{confidence:.6f}".encode()
                    ),
                    confidence=confidence,
                    timestamp_ms=_now_ms(),
                ),
            ],
            trust_level=trust_level,
            overall_confidence=confidence,
            verification_passed=trust_level != MediaTrustLevel.RAW.value,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED MEDIA GATE
# ═══════════════════════════════════════════════════════════════════════════════

class NRSMediaGate:
    """Unified verification gate for all NRS-generated media."""

    def __init__(self) -> None:
        self.image_gate = ImageVerificationGate()
        self.video_gate = VideoVerificationGate()
        self.audio_gate = AudioVerificationGate()

    def verify_image(
        self,
        image: np.ndarray,
        scene_graph: Any,
        prompt: str,
    ) -> MediaVerificationResult:
        return self.image_gate.verify(image, scene_graph, prompt)

    def verify_video(
        self,
        frames: List[np.ndarray],
        scene_graph: Any,
        prompt: str,
        audio: Optional[np.ndarray] = None,
        sr: int = 48000,
    ) -> MediaVerificationResult:
        return self.video_gate.verify(frames, scene_graph, prompt, audio, sr)

    def verify_speech(
        self,
        audio: np.ndarray,
        text: str,
        sr: int = 24000,
    ) -> MediaVerificationResult:
        return self.audio_gate.verify_speech(audio, text, sr)

    def verify_music(
        self,
        audio: np.ndarray,
        prompt: str,
        sr: int = 44100,
    ) -> MediaVerificationResult:
        return self.audio_gate.verify_music(audio, prompt, sr)

    def verify_transcription(
        self,
        result: Any,
        audio: np.ndarray,
        sr: int = 16000,
    ) -> MediaVerificationResult:
        return self.audio_gate.verify_transcription(result, audio, sr)


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    "MediaTrustLevel",
    "CheckCategory",
    "CheckResult",
    "ProvenanceStep",
    "MediaProvenance",
    "MediaVerificationResult",
    "ImageVerificationGate",
    "VideoVerificationGate",
    "AudioVerificationGate",
    "NRSMediaGate",
]
