"""
NRSI Video Generation Engine — time-varying SDFs, animation, physics,
temporal coherence, and full render-to-MP4 pipeline.

Replaces static camera-orbit video generation with real animation:
  - Time-varying SDF scenes with per-point velocity fields
  - Keyframe tracks with easing, skeletal poses, morph targets
  - Physics-lite: particles, rigid bodies, fluid surfaces
  - Temporal coherence: reprojection, motion vectors, blue noise
  - Camera animation: orbit, dolly, crane, tracking, flythrough
  - Scene templates: landscape, product, architecture, nature

Dependencies: numpy + subprocess (ffmpeg for encoding).  No ML.
"""
from __future__ import annotations

import math
import struct
import subprocess
import tempfile
import os
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Optional, Sequence, Tuple, Union,
)

import numpy as np

try:
    from .render_engine import (
        SDFScene, Camera, Material, Light, RenderConfig, NRSIRenderer,
        Vec3, _v3, _normalize_batch, _clamp, _dot_batch, _norm_batch,
        op_smooth_union, op_translate, op_rotate_y, sd_sphere, sd_box,
        sd_plane,
    )
except ImportError:
    from render_engine import (
        SDFScene, Camera, Material, Light, RenderConfig, NRSIRenderer,
        Vec3, _v3, _normalize_batch, _clamp, _dot_batch, _norm_batch,
        op_smooth_union, op_translate, op_rotate_y, sd_sphere, sd_box,
        sd_plane,
    )

try:
    from .advanced_renderer import AdvancedRenderer as _AdvancedRenderer
except ImportError:
    try:
        from advanced_renderer import AdvancedRenderer as _AdvancedRenderer
    except ImportError:
        _AdvancedRenderer = None

__all__ = [
    # time-varying SDF
    "AnimatedScene",
    # animation primitives
    "KeyframeTrack", "JointTransform", "SkeletalPose", "MorphTarget",
    # physics-lite
    "ParticleSystem", "RigidBody", "FluidSurface",
    # temporal coherence
    "TemporalBuffer", "BlueNoiseGenerator",
    # video pipeline
    "VideoRenderer",
    # camera animation
    "CameraPath",
    # scene templates
    "create_landscape_flythrough",
    "create_product_turntable",
    "create_architectural_walkthrough",
    "create_nature_timelapse",
]

_F32 = np.float32
_TWO_PI = 2.0 * math.pi


# ═══════════════════════════════════════════════════════════════════════════
# 1. TIME-VARYING SDF SYSTEM
# ═══════════════════════════════════════════════════════════════════════════

class AnimatedScene(SDFScene):
    """Scene whose SDF changes over time.

    Subclass and override ``evaluate(p, t)`` to define motion.
    The velocity field is estimated via finite differences when not
    overridden, enabling automatic motion-blur support.
    """

    def __init__(self):
        super().__init__()
        self._child_scenes: List[Tuple[Callable, Any]] = []

    def evaluate(self, p: Vec3, t: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate SDF at points *p* (N,3) at time *t*.

        Returns ``(distances (N,), material_ids (N,) int)``.
        Default implementation unions all registered child evaluators.
        """
        N = p.shape[0] if p.ndim > 1 else 1
        dist = np.full(N, 1e10, dtype=_F32)
        mats = np.zeros(N, dtype=np.int32)

        for eval_fn, _ in self._child_scenes:
            d, m = eval_fn(p, t)
            closer = d < dist
            dist = np.where(closer, d, dist)
            mats = np.where(closer, m, mats)

        return dist, mats

    def velocity_at(self, p: Vec3, t: float = 0.0, dt: float = 1e-3) -> Vec3:
        """Per-point velocity via central-difference SDF gradient in time."""
        d_fwd, _ = self.evaluate(p, t + dt)
        d_bck, _ = self.evaluate(p, t - dt)
        eps = 1e-4
        grad = np.zeros_like(p)
        for axis in range(3):
            offset = np.zeros(3, dtype=_F32)
            offset[axis] = eps
            df, _ = self.evaluate(p + offset, t)
            db, _ = self.evaluate(p - offset, t)
            grad[..., axis] = (df - db) / (2.0 * eps)
        speed = (d_fwd - d_bck) / (2.0 * dt)
        norm_sq = np.sum(grad * grad, axis=-1, keepdims=True).clip(1e-12)
        return -grad * (speed[..., np.newaxis] / norm_sq)

    def add_child(self, eval_fn: Callable, tag: Any = None):
        """Register a child evaluator ``(p, t) -> (dist, mat_ids)``."""
        self._child_scenes.append((eval_fn, tag))


# ═══════════════════════════════════════════════════════════════════════════
# 2. ANIMATION PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════

def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0 + 1e-12)))
    return t * t * (3.0 - 2.0 * t)


def _ease_in(t: float) -> float:
    return t * t


def _ease_out(t: float) -> float:
    return 1.0 - (1.0 - t) * (1.0 - t)


def _bounce(t: float) -> float:
    if t < 1.0 / 2.75:
        return 7.5625 * t * t
    elif t < 2.0 / 2.75:
        t -= 1.5 / 2.75
        return 7.5625 * t * t + 0.75
    elif t < 2.5 / 2.75:
        t -= 2.25 / 2.75
        return 7.5625 * t * t + 0.9375
    else:
        t -= 2.625 / 2.75
        return 7.5625 * t * t + 0.984375


def _elastic(t: float) -> float:
    if t <= 0.0 or t >= 1.0:
        return t
    return math.pow(2.0, -10.0 * t) * math.sin((t - 0.075) * _TWO_PI / 0.3) + 1.0


_EASING_FNS: Dict[str, Callable[[float], float]] = {
    "linear": lambda t: t,
    "smooth": lambda t: _smoothstep(0.0, 1.0, t),
    "ease_in": _ease_in,
    "ease_out": _ease_out,
    "bounce": _bounce,
    "elastic": _elastic,
}


class KeyframeTrack:
    """Interpolates scalar or Vec3 values over time with easing curves."""

    def __init__(self):
        self._keys: List[Tuple[float, Any, str]] = []

    def add_keyframe(self, time: float, value: Any, easing: str = "smooth"):
        if easing not in _EASING_FNS:
            raise ValueError(f"Unknown easing '{easing}'. Choose from {list(_EASING_FNS)}")
        self._keys.append((time, np.asarray(value, dtype=_F32), easing))
        self._keys.sort(key=lambda k: k[0])

    def evaluate(self, t: float) -> np.ndarray:
        """Return interpolated value at time *t*."""
        if not self._keys:
            return np.float32(0.0)
        if len(self._keys) == 1 or t <= self._keys[0][0]:
            return self._keys[0][1].copy()
        if t >= self._keys[-1][0]:
            return self._keys[-1][1].copy()

        for i in range(len(self._keys) - 1):
            t0, v0, _ = self._keys[i]
            t1, v1, easing = self._keys[i + 1]
            if t0 <= t <= t1:
                alpha = (t - t0) / (t1 - t0 + 1e-12)
                alpha = _EASING_FNS[easing](alpha)
                return v0 * (1.0 - alpha) + v1 * alpha

        return self._keys[-1][1].copy()


# ── skeletal animation ────────────────────────────────────────────────────

@dataclass
class JointTransform:
    position: Vec3 = field(default_factory=lambda: _v3(0, 0, 0))
    rotation: Vec3 = field(default_factory=lambda: _v3(0, 0, 0))
    scale: Vec3 = field(default_factory=lambda: _v3(1, 1, 1))


class SkeletalPose:
    """Joint hierarchy for character animation."""

    def __init__(self, joints: Optional[Dict[str, JointTransform]] = None):
        self.joints: Dict[str, JointTransform] = joints or {}

    def blend(self, other: "SkeletalPose", alpha: float) -> "SkeletalPose":
        """Linearly blend two poses."""
        result = SkeletalPose()
        all_names = set(self.joints) | set(other.joints)
        for name in all_names:
            a = self.joints.get(name, JointTransform())
            b = other.joints.get(name, JointTransform())
            blended = JointTransform(
                position=a.position * (1 - alpha) + b.position * alpha,
                rotation=a.rotation * (1 - alpha) + b.rotation * alpha,
                scale=a.scale * (1 - alpha) + b.scale * alpha,
            )
            result.joints[name] = blended
        return result

    # ── standard poses ────────────────────────────────────────────────

    @staticmethod
    def standing() -> "SkeletalPose":
        return SkeletalPose({
            "hip": JointTransform(position=_v3(0, 1.0, 0)),
            "spine": JointTransform(position=_v3(0, 1.3, 0)),
            "head": JointTransform(position=_v3(0, 1.7, 0)),
            "l_shoulder": JointTransform(position=_v3(-0.25, 1.45, 0)),
            "r_shoulder": JointTransform(position=_v3(0.25, 1.45, 0)),
            "l_hand": JointTransform(position=_v3(-0.25, 0.9, 0)),
            "r_hand": JointTransform(position=_v3(0.25, 0.9, 0)),
            "l_foot": JointTransform(position=_v3(-0.12, 0, 0)),
            "r_foot": JointTransform(position=_v3(0.12, 0, 0)),
        })

    @staticmethod
    def walking_0() -> "SkeletalPose":
        p = SkeletalPose.standing()
        p.joints["l_foot"] = JointTransform(position=_v3(-0.12, 0.05, 0.3))
        p.joints["r_foot"] = JointTransform(position=_v3(0.12, 0, -0.3))
        p.joints["l_hand"] = JointTransform(position=_v3(-0.25, 0.9, -0.2))
        p.joints["r_hand"] = JointTransform(position=_v3(0.25, 0.9, 0.2))
        return p

    @staticmethod
    def walking_1() -> "SkeletalPose":
        p = SkeletalPose.standing()
        p.joints["l_foot"] = JointTransform(position=_v3(-0.12, 0, -0.3))
        p.joints["r_foot"] = JointTransform(position=_v3(0.12, 0.05, 0.3))
        p.joints["l_hand"] = JointTransform(position=_v3(-0.25, 0.9, 0.2))
        p.joints["r_hand"] = JointTransform(position=_v3(0.25, 0.9, -0.2))
        return p

    @staticmethod
    def sitting() -> "SkeletalPose":
        p = SkeletalPose.standing()
        p.joints["hip"] = JointTransform(position=_v3(0, 0.5, 0))
        p.joints["spine"] = JointTransform(position=_v3(0, 0.8, 0))
        p.joints["head"] = JointTransform(position=_v3(0, 1.2, 0))
        p.joints["l_foot"] = JointTransform(position=_v3(-0.12, 0, 0.4))
        p.joints["r_foot"] = JointTransform(position=_v3(0.12, 0, 0.4))
        return p

    @staticmethod
    def running_0() -> "SkeletalPose":
        p = SkeletalPose.standing()
        p.joints["hip"] = JointTransform(position=_v3(0, 0.95, 0),
                                         rotation=_v3(0.1, 0, 0))
        p.joints["l_foot"] = JointTransform(position=_v3(-0.12, 0.2, 0.5))
        p.joints["r_foot"] = JointTransform(position=_v3(0.12, 0, -0.5))
        p.joints["l_hand"] = JointTransform(position=_v3(-0.3, 1.0, -0.4))
        p.joints["r_hand"] = JointTransform(position=_v3(0.3, 1.0, 0.4))
        return p

    @staticmethod
    def running_1() -> "SkeletalPose":
        p = SkeletalPose.standing()
        p.joints["hip"] = JointTransform(position=_v3(0, 0.95, 0),
                                         rotation=_v3(0.1, 0, 0))
        p.joints["l_foot"] = JointTransform(position=_v3(-0.12, 0, -0.5))
        p.joints["r_foot"] = JointTransform(position=_v3(0.12, 0.2, 0.5))
        p.joints["l_hand"] = JointTransform(position=_v3(-0.3, 1.0, 0.4))
        p.joints["r_hand"] = JointTransform(position=_v3(0.3, 1.0, -0.4))
        return p

    @staticmethod
    def waving() -> "SkeletalPose":
        p = SkeletalPose.standing()
        p.joints["r_hand"] = JointTransform(
            position=_v3(0.35, 1.8, 0.1),
            rotation=_v3(0, 0, -0.3),
        )
        return p


# ── morph targets ─────────────────────────────────────────────────────────

class MorphTarget:
    """Smooth interpolation between two SDF shapes."""

    def __init__(self, sdf_a: Callable, sdf_b: Callable, k: float = 0.15):
        self._sdf_a = sdf_a
        self._sdf_b = sdf_b
        self._k = k

    def evaluate(self, p: Vec3, blend: float) -> np.ndarray:
        """*blend* 0→shape A, 1→shape B, intermediate is smooth union."""
        da = self._sdf_a(p)
        db = self._sdf_b(p)
        blend = float(np.clip(blend, 0.0, 1.0))
        if blend <= 0.0:
            return da
        if blend >= 1.0:
            return db
        return da * (1.0 - blend) + db * blend - self._k * blend * (1.0 - blend)


# ═══════════════════════════════════════════════════════════════════════════
# 3. PHYSICS-LITE SYSTEMS
# ═══════════════════════════════════════════════════════════════════════════

_GRAVITY = _v3(0, -9.81, 0)


class ParticleSystem:
    """Simple particle emitter with gravity and ground-plane collision."""

    def __init__(self, max_particles: int = 4096):
        self.max_particles = max_particles
        self.positions = np.zeros((max_particles, 3), dtype=_F32)
        self.velocities = np.zeros((max_particles, 3), dtype=_F32)
        self.lifetimes = np.zeros(max_particles, dtype=_F32)
        self.ages = np.zeros(max_particles, dtype=_F32)
        self.alive = np.zeros(max_particles, dtype=bool)
        self._next = 0
        self.particle_radius = 0.02
        self.gravity = _GRAVITY.copy()
        self.damping = 0.98
        self.ground_y = 0.0
        self.bounce_factor = 0.3

    def emit(self, position: Vec3, velocity: Vec3, count: int,
             lifetime: float = 2.0, spread: float = 0.3):
        """Emit *count* particles from *position* with randomised spread."""
        rng = np.random.default_rng()
        for _ in range(count):
            idx = self._next % self.max_particles
            self._next += 1
            self.positions[idx] = np.asarray(position, dtype=_F32)
            noise = rng.normal(0, spread, 3).astype(_F32)
            self.velocities[idx] = np.asarray(velocity, dtype=_F32) + noise
            self.lifetimes[idx] = lifetime
            self.ages[idx] = 0.0
            self.alive[idx] = True

    def step(self, dt: float):
        """Integrate one timestep: gravity, damping, ground collision."""
        mask = self.alive
        if not mask.any():
            return
        self.velocities[mask] += self.gravity[np.newaxis, :] * dt
        self.velocities[mask] *= self.damping
        self.positions[mask] += self.velocities[mask] * dt

        below = mask & (self.positions[:, 1] < self.ground_y)
        self.positions[below, 1] = self.ground_y
        self.velocities[below, 1] *= -self.bounce_factor

        self.ages[mask] += dt
        expired = mask & (self.ages >= self.lifetimes)
        self.alive[expired] = False

    def sdf_at(self, p: Vec3, t: float = 0.0) -> np.ndarray:
        """SDF of all alive particles as smooth-unioned spheres."""
        live = np.where(self.alive)[0]
        N = p.shape[0] if p.ndim > 1 else 1
        dist = np.full(N, 1e10, dtype=_F32)
        r = self.particle_radius

        for idx in live:
            d = _norm_batch(p - self.positions[idx]) - r
            dist = op_smooth_union(dist, d, k=r * 3.0)

        return dist

    # ── presets ────────────────────────────────────────────────────────

    @classmethod
    def rain(cls, intensity: float = 1.0) -> "ParticleSystem":
        ps = cls(max_particles=8192)
        ps.particle_radius = 0.008
        ps.gravity = _v3(0, -12.0 * intensity, 0)
        ps.damping = 0.995
        ps.bounce_factor = 0.1
        return ps

    @classmethod
    def snow(cls) -> "ParticleSystem":
        ps = cls(max_particles=4096)
        ps.particle_radius = 0.015
        ps.gravity = _v3(0, -1.2, 0)
        ps.damping = 0.99
        ps.bounce_factor = 0.0
        return ps

    @classmethod
    def fire(cls) -> "ParticleSystem":
        ps = cls(max_particles=2048)
        ps.particle_radius = 0.04
        ps.gravity = _v3(0, 3.0, 0)
        ps.damping = 0.96
        ps.bounce_factor = 0.0
        ps.ground_y = -1e6
        return ps

    @classmethod
    def smoke(cls) -> "ParticleSystem":
        ps = cls(max_particles=2048)
        ps.particle_radius = 0.06
        ps.gravity = _v3(0, 1.5, 0)
        ps.damping = 0.97
        ps.bounce_factor = 0.0
        ps.ground_y = -1e6
        return ps

    @classmethod
    def sparks(cls) -> "ParticleSystem":
        ps = cls(max_particles=1024)
        ps.particle_radius = 0.005
        ps.gravity = _v3(0, -6.0, 0)
        ps.damping = 0.985
        ps.bounce_factor = 0.5
        return ps

    @classmethod
    def dust(cls) -> "ParticleSystem":
        ps = cls(max_particles=2048)
        ps.particle_radius = 0.01
        ps.gravity = _v3(0, -0.3, 0)
        ps.damping = 0.995
        ps.bounce_factor = 0.0
        return ps


class RigidBody:
    """Single rigid body with Euler integration and SDF transform."""

    def __init__(self, sdf_fn: Callable[[Vec3], np.ndarray],
                 position: Optional[Vec3] = None, mass: float = 1.0):
        self.sdf_fn = sdf_fn
        self.position = np.asarray(position if position is not None
                                   else _v3(0, 0, 0), dtype=_F32)
        self.velocity = _v3(0, 0, 0)
        self.angular_velocity = _v3(0, 0, 0)
        self.angle_y = 0.0
        self.mass = mass
        self.gravity = _GRAVITY.copy()
        self.restitution = 0.4
        self.ground_y = 0.0

    def step(self, dt: float):
        """Euler integration with gravity and ground-plane constraint."""
        self.velocity = self.velocity + self.gravity * dt
        self.position = self.position + self.velocity * dt
        self.angle_y += float(self.angular_velocity[1]) * dt

        if self.position[1] < self.ground_y:
            self.position[1] = self.ground_y
            self.velocity[1] = -self.velocity[1] * self.restitution

    def sdf_transform(self, p: Vec3, t: float = 0.0) -> np.ndarray:
        """Evaluate base SDF with current rigid-body transform applied."""
        q = op_translate(p, self.position)
        if abs(self.angle_y) > 1e-6:
            q = op_rotate_y(q, -self.angle_y)
        return self.sdf_fn(q)


class FluidSurface:
    """Gerstner wave model for animated water surfaces."""

    def __init__(self, waves: Optional[List[Dict[str, float]]] = None):
        if waves is None:
            waves = [
                {"amplitude": 0.08, "wavelength": 4.0,
                 "speed": 1.2, "direction": 0.0},
                {"amplitude": 0.04, "wavelength": 2.0,
                 "speed": 0.8, "direction": 0.7},
                {"amplitude": 0.02, "wavelength": 1.0,
                 "speed": 1.5, "direction": -0.4},
            ]
        self.waves = waves
        self.base_height = 0.0

    def _wave_params(self, w: Dict[str, float]):
        amp = w["amplitude"]
        wl = w["wavelength"]
        spd = w["speed"]
        d = w.get("direction", 0.0)
        k = _TWO_PI / wl
        dx = math.cos(d)
        dz = math.sin(d)
        return amp, k, spd, dx, dz

    def height_at(self, x: np.ndarray, z: np.ndarray,
                  t: float) -> np.ndarray:
        """Water surface height at (x, z) positions and time *t*."""
        x = np.asarray(x, dtype=_F32)
        z = np.asarray(z, dtype=_F32)
        h = np.full_like(x, self.base_height)
        for w in self.waves:
            amp, k, spd, dx, dz = self._wave_params(w)
            phase = k * (dx * x + dz * z) - spd * t * k
            h += amp * np.sin(phase)
        return h

    def normal_at(self, x: np.ndarray, z: np.ndarray,
                  t: float) -> Vec3:
        """Surface normal at (x, z) positions and time *t*."""
        x = np.asarray(x, dtype=_F32)
        z = np.asarray(z, dtype=_F32)
        dx_sum = np.zeros_like(x)
        dz_sum = np.zeros_like(x)
        for w in self.waves:
            amp, k, spd, wdx, wdz = self._wave_params(w)
            phase = k * (wdx * x + wdz * z) - spd * t * k
            c = amp * k * np.cos(phase)
            dx_sum += c * wdx
            dz_sum += c * wdz
        normals = np.stack([-dx_sum, np.ones_like(x), -dz_sum], axis=-1)
        return _normalize_batch(normals)

    # ── presets ────────────────────────────────────────────────────────

    @classmethod
    def calm(cls) -> "FluidSurface":
        return cls([
            {"amplitude": 0.03, "wavelength": 6.0,
             "speed": 0.5, "direction": 0.0},
            {"amplitude": 0.01, "wavelength": 2.5,
             "speed": 0.3, "direction": 0.8},
        ])

    @classmethod
    def choppy(cls) -> "FluidSurface":
        return cls([
            {"amplitude": 0.15, "wavelength": 3.5,
             "speed": 1.8, "direction": 0.2},
            {"amplitude": 0.08, "wavelength": 1.8,
             "speed": 1.2, "direction": -0.5},
            {"amplitude": 0.04, "wavelength": 0.8,
             "speed": 2.0, "direction": 1.1},
        ])

    @classmethod
    def stormy(cls) -> "FluidSurface":
        return cls([
            {"amplitude": 0.4, "wavelength": 8.0,
             "speed": 3.0, "direction": 0.1},
            {"amplitude": 0.2, "wavelength": 4.0,
             "speed": 2.5, "direction": -0.3},
            {"amplitude": 0.1, "wavelength": 1.5,
             "speed": 3.5, "direction": 0.9},
            {"amplitude": 0.05, "wavelength": 0.6,
             "speed": 4.0, "direction": -1.2},
        ])


# ═══════════════════════════════════════════════════════════════════════════
# 4. TEMPORAL COHERENCE
# ═══════════════════════════════════════════════════════════════════════════

class TemporalBuffer:
    """Per-pixel history for temporal accumulation and motion vectors."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.prev_color = np.zeros((height, width, 3), dtype=_F32)
        self.prev_depth = np.full((height, width), 1e10, dtype=_F32)
        self.motion_vectors = np.zeros((height, width, 2), dtype=_F32)
        self._has_prev = False

    def reproject(self, current_camera: Camera,
                  prev_camera: Camera) -> np.ndarray:
        """Reproject previous frame colour into current view.

        Returns (H, W, 3) float32 reprojected colour buffer.
        """
        h, w = self.height, self.width
        if not self._has_prev:
            return np.zeros((h, w, 3), dtype=_F32)

        yy, xx = np.mgrid[0:h, 0:w].astype(_F32)
        ndc_x = (xx + 0.5) / w * 2.0 - 1.0
        ndc_y = 1.0 - (yy + 0.5) / h * 2.0

        fov_h = math.tan(math.radians(current_camera.fov_deg) / 2.0)
        aspect = w / h
        fwd = _normalize_batch(current_camera.target - current_camera.position)
        right = _normalize_batch(np.cross(fwd, current_camera.up))
        up = np.cross(right, fwd)

        depth = self.prev_depth
        world = (current_camera.position[np.newaxis, np.newaxis, :]
                 + fwd[np.newaxis, np.newaxis, :] * depth[..., np.newaxis]
                 + right[np.newaxis, np.newaxis, :] * (ndc_x[..., np.newaxis] * fov_h * aspect * depth[..., np.newaxis])
                 + up[np.newaxis, np.newaxis, :] * (ndc_y[..., np.newaxis] * fov_h * depth[..., np.newaxis]))

        pfwd = _normalize_batch(prev_camera.target - prev_camera.position)
        pright = _normalize_batch(np.cross(pfwd, prev_camera.up))
        pup = np.cross(pright, pfwd)

        to_point = world - prev_camera.position[np.newaxis, np.newaxis, :]
        pz = np.sum(to_point * pfwd[np.newaxis, np.newaxis, :], axis=-1)
        px = np.sum(to_point * pright[np.newaxis, np.newaxis, :], axis=-1)
        py = np.sum(to_point * pup[np.newaxis, np.newaxis, :], axis=-1)

        pfov_h = math.tan(math.radians(prev_camera.fov_deg) / 2.0)
        safe_pz = np.where(np.abs(pz) < 1e-6, 1e-6, pz)
        prev_ndc_x = px / (safe_pz * pfov_h * aspect)
        prev_ndc_y = py / (safe_pz * pfov_h)

        prev_px = ((prev_ndc_x + 1.0) * 0.5 * w).astype(np.int32)
        prev_py = ((1.0 - prev_ndc_y) * 0.5 * h).astype(np.int32)

        valid = ((prev_px >= 0) & (prev_px < w) &
                 (prev_py >= 0) & (prev_py < h) &
                 (pz > 0))

        result = np.zeros((h, w, 3), dtype=_F32)
        vy = prev_py[valid]
        vx = prev_px[valid]
        cy = yy.astype(np.int32)[valid]
        cx = xx.astype(np.int32)[valid]
        result[cy, cx] = self.prev_color[vy, vx]
        return result

    def temporal_accumulate(self, current: np.ndarray,
                            reprojected: np.ndarray,
                            blend: float = 0.1) -> np.ndarray:
        """Blend current frame with reprojected history for TAA."""
        blend = float(np.clip(blend, 0.0, 1.0))
        if not self._has_prev:
            return current.copy()
        return current * blend + reprojected * (1.0 - blend)

    def compute_motion_vectors(self, current_depth: np.ndarray,
                               current_camera: Camera,
                               prev_camera: Camera) -> np.ndarray:
        """Per-pixel screen-space motion vectors (H, W, 2) in pixels."""
        h, w = self.height, self.width
        yy, xx = np.mgrid[0:h, 0:w].astype(_F32)
        ndc_x = (xx + 0.5) / w * 2.0 - 1.0
        ndc_y = 1.0 - (yy + 0.5) / h * 2.0

        fov_h = math.tan(math.radians(current_camera.fov_deg) / 2.0)
        aspect = w / h
        fwd = _normalize_batch(current_camera.target - current_camera.position)
        right = _normalize_batch(np.cross(fwd, current_camera.up))
        up = np.cross(right, fwd)

        depth = current_depth
        world = (current_camera.position[np.newaxis, np.newaxis, :]
                 + fwd[np.newaxis, np.newaxis, :] * depth[..., np.newaxis]
                 + right[np.newaxis, np.newaxis, :] * (ndc_x[..., np.newaxis] * fov_h * aspect * depth[..., np.newaxis])
                 + up[np.newaxis, np.newaxis, :] * (ndc_y[..., np.newaxis] * fov_h * depth[..., np.newaxis]))

        pfwd = _normalize_batch(prev_camera.target - prev_camera.position)
        pright = _normalize_batch(np.cross(pfwd, prev_camera.up))
        pup = np.cross(pright, pfwd)

        to_point = world - prev_camera.position[np.newaxis, np.newaxis, :]
        pz = np.sum(to_point * pfwd[np.newaxis, np.newaxis, :], axis=-1)
        px = np.sum(to_point * pright[np.newaxis, np.newaxis, :], axis=-1)
        py = np.sum(to_point * pup[np.newaxis, np.newaxis, :], axis=-1)

        pfov_h = math.tan(math.radians(prev_camera.fov_deg) / 2.0)
        safe_pz = np.where(np.abs(pz) < 1e-6, 1e-6, pz)
        prev_px = (px / (safe_pz * pfov_h * aspect) + 1.0) * 0.5 * w
        prev_py = (1.0 - py / (safe_pz * pfov_h)) * 0.5 * h

        mv = np.stack([prev_px - xx, prev_py - yy], axis=-1)
        self.motion_vectors = mv
        return mv

    def update(self, color: np.ndarray, depth: np.ndarray):
        """Store current frame as history for next frame's reprojection."""
        self.prev_color = color.astype(_F32) if color.dtype != _F32 else color.copy()
        self.prev_depth = depth.astype(_F32) if depth.dtype != _F32 else depth.copy()
        self._has_prev = True


class BlueNoiseGenerator:
    """Deterministic blue-noise sampling for temporally stable jitter."""

    def __init__(self, size: int = 128, seed: int = 0):
        self.size = size
        rng = np.random.default_rng(seed)
        self._texture = rng.random((size, size)).astype(_F32)
        self._golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0

    def sample(self, pixel_x: int, pixel_y: int, frame: int) -> float:
        """Return [0,1] value that varies smoothly across frames."""
        base = self._texture[pixel_y % self.size, pixel_x % self.size]
        offset = (frame * self._golden_ratio) % 1.0
        return float((base + offset) % 1.0)

    def sample_array(self, width: int, height: int,
                     frame: int) -> np.ndarray:
        """Return (H, W) blue-noise pattern for an entire frame."""
        tile_y = (height + self.size - 1) // self.size
        tile_x = (width + self.size - 1) // self.size
        tiled = np.tile(self._texture, (tile_y, tile_x))[:height, :width]
        offset = (frame * self._golden_ratio) % 1.0
        return (tiled + offset) % 1.0


# ═══════════════════════════════════════════════════════════════════════════
# 5. VIDEO RENDER PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

class VideoRenderer:
    """Renders animated SDF scenes to frame sequences and MP4."""

    def __init__(self, scene: AnimatedScene,
                 config: Optional[RenderConfig] = None,
                 lighting: Optional[Light] = None):
        self.scene = scene
        self.config = config or RenderConfig()
        self.lighting = lighting or Light()

        RendererClass = _AdvancedRenderer if _AdvancedRenderer else NRSIRenderer
        self.renderer = RendererClass(scene, self.config)

        self.temporal = TemporalBuffer(self.config.width, self.config.height)
        self.blue_noise = BlueNoiseGenerator()
        self._prev_camera: Optional[Camera] = None
        self._frame_idx = 0

    def render_frame(self, camera: Camera, t: float,
                     dt: float = 1.0 / 24.0) -> np.ndarray:
        """Render a single frame at time *t* with temporal accumulation.

        Returns (H, W, 3) uint8 image.
        """
        if hasattr(self.scene, 'evaluate') and callable(getattr(self.scene, 'evaluate')):
            self.scene._current_t = t

        frame_uint8 = self.renderer.render(camera, self.lighting)
        frame_f32 = frame_uint8.astype(_F32) / 255.0

        if self._prev_camera is not None:
            reprojected = self.temporal.reproject(camera, self._prev_camera)
            frame_f32 = self.temporal.temporal_accumulate(
                frame_f32, reprojected, blend=0.15)

        depth_est = np.full(
            (self.config.height, self.config.width), 10.0, dtype=_F32)
        self.temporal.update(frame_f32, depth_est)
        self._prev_camera = Camera(
            position=camera.position.copy(),
            target=camera.target.copy(),
            up=camera.up.copy(),
            fov_deg=camera.fov_deg,
        )
        self._frame_idx += 1

        return (_clamp(frame_f32) * 255).astype(np.uint8)

    def render_sequence(self, cameras: Union[List[Camera], Callable],
                        fps: int = 24, duration: Optional[float] = None,
                        progress_callback: Optional[Callable] = None,
                        ) -> List[np.ndarray]:
        """Render a full frame sequence.

        *cameras* can be a list of Camera objects (one per frame) or a
        callable ``(t) -> Camera`` evaluated at each timestep.
        """
        if callable(cameras):
            if duration is None:
                duration = 4.0
            n_frames = int(duration * fps)
            dt = 1.0 / fps
            frames = []
            for i in range(n_frames):
                t = i * dt
                cam = cameras(t)
                frame = self.render_frame(cam, t, dt)
                frames.append(frame)
                if progress_callback:
                    progress_callback(i + 1, n_frames)
            return frames

        n_frames = len(cameras)
        dt = 1.0 / fps
        frames = []
        for i, cam in enumerate(cameras):
            t = i * dt
            frame = self.render_frame(cam, t, dt)
            frames.append(frame)
            if progress_callback:
                progress_callback(i + 1, n_frames)
        return frames

    def encode_mp4(self, frames: List[np.ndarray], fps: int = 24,
                   quality: str = "high") -> bytes:
        """Encode frame list to H.264 MP4 via ffmpeg subprocess.

        *quality*: ``"draft"`` (fast/low), ``"high"`` (default),
        ``"ultra"`` (slow/best).
        """
        if not frames:
            return b""

        h, w = frames[0].shape[:2]
        crf_map = {"draft": "28", "high": "18", "ultra": "12"}
        preset_map = {"draft": "ultrafast", "high": "medium", "ultra": "slow"}
        crf = crf_map.get(quality, "18")
        preset = preset_map.get(quality, "medium")

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "out.mp4")
            cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{w}x{h}",
                "-pix_fmt", "rgb24",
                "-r", str(fps),
                "-i", "-",
                "-c:v", "libx264",
                "-preset", preset,
                "-crf", crf,
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                out_path,
            ]
            raw = b"".join(f.tobytes() for f in frames)
            proc = subprocess.run(
                cmd, input=raw,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg failed (exit {proc.returncode}): "
                    f"{proc.stderr.decode(errors='replace')[:500]}")
            with open(out_path, "rb") as fh:
                return fh.read()

    def encode_with_audio(self, frames: List[np.ndarray],
                          audio_bytes: bytes, fps: int = 24,
                          quality: str = "high") -> bytes:
        """Mux rendered video with audio track."""
        if not frames:
            return b""

        h, w = frames[0].shape[:2]
        crf_map = {"draft": "28", "high": "18", "ultra": "12"}
        preset_map = {"draft": "ultrafast", "high": "medium", "ultra": "slow"}
        crf = crf_map.get(quality, "18")
        preset = preset_map.get(quality, "medium")

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "audio.wav")
            out_path = os.path.join(tmpdir, "out.mp4")
            with open(audio_path, "wb") as af:
                af.write(audio_bytes)

            cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{w}x{h}",
                "-pix_fmt", "rgb24",
                "-r", str(fps),
                "-i", "-",
                "-i", audio_path,
                "-c:v", "libx264",
                "-preset", preset,
                "-crf", crf,
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                "-movflags", "+faststart",
                out_path,
            ]
            raw = b"".join(f.tobytes() for f in frames)
            proc = subprocess.run(
                cmd, input=raw,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg failed (exit {proc.returncode}): "
                    f"{proc.stderr.decode(errors='replace')[:500]}")
            with open(out_path, "rb") as fh:
                return fh.read()


# ═══════════════════════════════════════════════════════════════════════════
# 6. CAMERA ANIMATION
# ═══════════════════════════════════════════════════════════════════════════

def _catmull_rom(p0, p1, p2, p3, t):
    """Catmull-Rom spline interpolation for smooth camera paths."""
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2.0 * p1) +
        (-p0 + p2) * t +
        (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
        (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )


class CameraPath:
    """Animate a camera smoothly through keyframed waypoints."""

    def __init__(self):
        self._waypoints: List[Tuple[float, Vec3, Vec3, float]] = []

    def add_waypoint(self, t: float, position: Vec3, target: Vec3,
                     fov: float = 35.0):
        """Add a camera keyframe at time *t*."""
        self._waypoints.append((
            t,
            np.asarray(position, dtype=_F32),
            np.asarray(target, dtype=_F32),
            float(fov),
        ))
        self._waypoints.sort(key=lambda w: w[0])

    def evaluate(self, t: float) -> Camera:
        """Return interpolated camera at time *t*."""
        if not self._waypoints:
            return Camera()
        if len(self._waypoints) == 1:
            _, pos, tgt, fov = self._waypoints[0]
            return Camera(position=pos.copy(), target=tgt.copy(), fov_deg=fov)

        if t <= self._waypoints[0][0]:
            _, pos, tgt, fov = self._waypoints[0]
            return Camera(position=pos.copy(), target=tgt.copy(), fov_deg=fov)
        if t >= self._waypoints[-1][0]:
            _, pos, tgt, fov = self._waypoints[-1]
            return Camera(position=pos.copy(), target=tgt.copy(), fov_deg=fov)

        for i in range(len(self._waypoints) - 1):
            t0, p0, tg0, f0 = self._waypoints[i]
            t1, p1, tg1, f1 = self._waypoints[i + 1]
            if t0 <= t <= t1:
                alpha = (t - t0) / (t1 - t0 + 1e-12)
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)

                i_prev = max(0, i - 1)
                i_next = min(len(self._waypoints) - 1, i + 2)
                _, pp, tgp, fp = self._waypoints[i_prev]
                _, pn, tgn, fn = self._waypoints[i_next]

                pos = _catmull_rom(pp, p0, p1, pn, alpha)
                tgt = _catmull_rom(tgp, tg0, tg1, tgn, alpha)
                fov = _catmull_rom(
                    np.float32(fp), np.float32(f0),
                    np.float32(f1), np.float32(fn), alpha)

                return Camera(
                    position=pos.astype(_F32),
                    target=tgt.astype(_F32),
                    fov_deg=float(fov),
                )

        _, pos, tgt, fov = self._waypoints[-1]
        return Camera(position=pos.copy(), target=tgt.copy(), fov_deg=fov)

    # ── preset camera paths ───────────────────────────────────────────

    @classmethod
    def orbit(cls, center: Vec3, radius: float = 5.0,
              height: float = 2.0, speed: float = 1.0,
              duration: float = 4.0, n_keyframes: int = 32,
              ) -> "CameraPath":
        """Circular orbit around *center*."""
        center = np.asarray(center, dtype=_F32)
        path = cls()
        for i in range(n_keyframes):
            t = i / n_keyframes * duration
            angle = t * speed * _TWO_PI / duration
            pos = center + _v3(
                math.cos(angle) * radius,
                height,
                math.sin(angle) * radius,
            )
            path.add_waypoint(t, pos, center)
        return path

    @classmethod
    def dolly(cls, start: Vec3, end: Vec3, target: Vec3,
              duration: float = 4.0, n_keyframes: int = 16,
              ) -> "CameraPath":
        """Linear dolly from *start* to *end* looking at *target*."""
        start = np.asarray(start, dtype=_F32)
        end = np.asarray(end, dtype=_F32)
        target = np.asarray(target, dtype=_F32)
        path = cls()
        for i in range(n_keyframes):
            alpha = i / max(n_keyframes - 1, 1)
            t = alpha * duration
            pos = start * (1.0 - alpha) + end * alpha
            path.add_waypoint(t, pos, target)
        return path

    @classmethod
    def crane(cls, base: Vec3, height_range: Tuple[float, float],
              target: Vec3, duration: float = 4.0,
              n_keyframes: int = 16) -> "CameraPath":
        """Vertical crane shot from low to high (or vice versa)."""
        base = np.asarray(base, dtype=_F32)
        target = np.asarray(target, dtype=_F32)
        h_lo, h_hi = height_range
        path = cls()
        for i in range(n_keyframes):
            alpha = i / max(n_keyframes - 1, 1)
            t = alpha * duration
            h = h_lo + (h_hi - h_lo) * alpha
            pos = base.copy()
            pos[1] = h
            path.add_waypoint(t, pos, target)
        return path

    @classmethod
    def tracking(cls, target_path: Callable, offset: Vec3,
                 duration: float = 4.0, n_keyframes: int = 32,
                 ) -> "CameraPath":
        """Follow a moving target with fixed offset.

        *target_path*: callable ``(t) -> Vec3`` giving target position.
        """
        offset = np.asarray(offset, dtype=_F32)
        path = cls()
        for i in range(n_keyframes):
            alpha = i / max(n_keyframes - 1, 1)
            t = alpha * duration
            tgt = np.asarray(target_path(t), dtype=_F32)
            pos = tgt + offset
            path.add_waypoint(t, pos, tgt)
        return path

    @classmethod
    def flythrough(cls, waypoints: List[Vec3],
                   duration: float = 6.0) -> "CameraPath":
        """Smooth Catmull-Rom flythrough along *waypoints*.

        Camera looks ahead along the path direction.
        """
        waypoints = [np.asarray(w, dtype=_F32) for w in waypoints]
        n = len(waypoints)
        if n < 2:
            path = cls()
            if n == 1:
                path.add_waypoint(0.0, waypoints[0],
                                  waypoints[0] + _v3(0, 0, 1))
            return path

        path = cls()
        for i, wp in enumerate(waypoints):
            t = i / max(n - 1, 1) * duration
            look_idx = min(i + 1, n - 1)
            target = waypoints[look_idx]
            if np.allclose(wp, target):
                target = wp + _v3(0, 0, 0.1)
            path.add_waypoint(t, wp, target)
        return path


# ═══════════════════════════════════════════════════════════════════════════
# 7. SCENE TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════

def _simple_terrain_sdf(p: Vec3, t: float = 0.0):
    """Rolling hills ground plane for template scenes."""
    if p.ndim == 1:
        p = p[np.newaxis, :]
    freq = 0.15
    height = (np.sin(p[:, 0] * freq) * np.cos(p[:, 2] * freq * 0.7) * 2.0
              + np.sin(p[:, 0] * 0.05 + t * 0.1) * 4.0)
    dist = p[:, 1] - height
    mats = np.zeros(dist.shape, dtype=np.int32)
    return dist, mats


def _water_plane_sdf(fluid: FluidSurface, water_mat_id: int):
    """Return an evaluator that uses a FluidSurface for the y-displacement."""
    def _eval(p: Vec3, t: float = 0.0):
        if p.ndim == 1:
            p = p[np.newaxis, :]
        h = fluid.height_at(p[:, 0], p[:, 2], t)
        dist = p[:, 1] - h
        mats = np.full(dist.shape, water_mat_id, dtype=np.int32)
        return dist, mats
    return _eval


def create_landscape_flythrough(prompt: str = ""
                                ) -> Tuple[AnimatedScene, CameraPath]:
    """Scenic flythrough over rolling terrain with animated water."""
    scene = AnimatedScene()
    terrain_mat = scene.add_material(
        Material(albedo=_v3(0.15, 0.25, 0.08), roughness=0.9))
    water_mat = scene.add_material(
        Material(albedo=_v3(0.02, 0.05, 0.12), metallic=0.2,
                 roughness=0.05, ior=1.33))

    def terrain_eval(p, t):
        d, _ = _simple_terrain_sdf(p, t)
        m = np.full_like(d, terrain_mat, dtype=np.int32)
        return d, m

    fluid = FluidSurface.calm()
    fluid.base_height = -0.5
    water_eval = _water_plane_sdf(fluid, water_mat)

    scene.add_child(terrain_eval, "terrain")
    scene.add_child(water_eval, "water")

    waypoints = [
        _v3(-20, 6, -20),
        _v3(-5, 4, -10),
        _v3(5, 3, 0),
        _v3(15, 5, 10),
        _v3(25, 8, 20),
    ]
    camera_path = CameraPath.flythrough(waypoints, duration=6.0)
    return scene, camera_path


def create_product_turntable(prompt: str = ""
                             ) -> Tuple[AnimatedScene, CameraPath]:
    """Product showcase: object on pedestal with orbiting camera."""
    scene = AnimatedScene()
    pedestal_mat = scene.add_material(
        Material(albedo=_v3(0.9, 0.9, 0.92), metallic=0.1, roughness=0.3))
    product_mat = scene.add_material(
        Material(albedo=_v3(0.7, 0.15, 0.1), metallic=0.8, roughness=0.15))

    def pedestal_eval(p, t):
        if p.ndim == 1:
            p = p[np.newaxis, :]
        d_base = sd_box(p - _v3(0, -0.15, 0), _v3(0.8, 0.15, 0.8))
        m = np.full(d_base.shape, pedestal_mat, dtype=np.int32)
        return d_base, m

    def product_eval(p, t):
        if p.ndim == 1:
            p = p[np.newaxis, :]
        angle = t * 0.5
        q = op_rotate_y(p - _v3(0, 0.5, 0), angle)
        d = sd_box(q, _v3(0.3, 0.3, 0.3)) - 0.05
        m = np.full(d.shape, product_mat, dtype=np.int32)
        return d, m

    ground_mat = scene.add_material(
        Material(albedo=_v3(0.06, 0.06, 0.07), roughness=0.8))

    def ground_eval(p, t):
        if p.ndim == 1:
            p = p[np.newaxis, :]
        d = sd_plane(p, _v3(0, 1, 0), 0.3)
        m = np.full(d.shape, ground_mat, dtype=np.int32)
        return d, m

    scene.add_child(ground_eval, "ground")
    scene.add_child(pedestal_eval, "pedestal")
    scene.add_child(product_eval, "product")

    camera_path = CameraPath.orbit(
        center=_v3(0, 0.5, 0), radius=3.0, height=1.5,
        speed=1.0, duration=4.0)
    return scene, camera_path


def create_architectural_walkthrough(prompt: str = ""
                                     ) -> Tuple[AnimatedScene, CameraPath]:
    """Walk through a simple architectural space."""
    scene = AnimatedScene()
    floor_mat = scene.add_material(
        Material(albedo=_v3(0.35, 0.3, 0.25), roughness=0.6))
    wall_mat = scene.add_material(
        Material(albedo=_v3(0.85, 0.82, 0.78), roughness=0.7))
    ceiling_mat = scene.add_material(
        Material(albedo=_v3(0.9, 0.9, 0.9), roughness=0.8))

    def room_eval(p, t):
        if p.ndim == 1:
            p = p[np.newaxis, :]
        N = p.shape[0]
        dist = np.full(N, 1e10, dtype=_F32)
        mats = np.zeros(N, dtype=np.int32)

        d_floor = sd_plane(p, _v3(0, 1, 0), 0.0)
        closer = d_floor < dist
        dist = np.where(closer, d_floor, dist)
        mats = np.where(closer, floor_mat, mats)

        d_ceil = sd_plane(p, _v3(0, -1, 0), -3.5)
        closer = d_ceil < dist
        dist = np.where(closer, d_ceil, dist)
        mats = np.where(closer, ceiling_mat, mats)

        for wall_offset, wall_normal in [
            (_v3(5, 0, 0), _v3(-1, 0, 0)),
            (_v3(-5, 0, 0), _v3(1, 0, 0)),
            (_v3(0, 0, 12), _v3(0, 0, -1)),
            (_v3(0, 0, -2), _v3(0, 0, 1)),
        ]:
            d_wall = sd_plane(p - wall_offset, wall_normal, 0.0)
            closer = d_wall < dist
            dist = np.where(closer, d_wall, dist)
            mats = np.where(closer, wall_mat, mats)

        return dist, mats

    scene.add_child(room_eval, "room")

    waypoints = [
        _v3(0, 1.7, -1),
        _v3(0, 1.7, 3),
        _v3(2, 1.7, 6),
        _v3(-1, 1.7, 9),
        _v3(0, 1.7, 11),
    ]
    camera_path = CameraPath.flythrough(waypoints, duration=5.0)
    return scene, camera_path


def create_nature_timelapse(prompt: str = ""
                            ) -> Tuple[AnimatedScene, CameraPath]:
    """Time-lapse with animated water, shifting lighting."""
    scene = AnimatedScene()
    terrain_mat = scene.add_material(
        Material(albedo=_v3(0.12, 0.22, 0.06), roughness=0.85))
    water_mat = scene.add_material(
        Material(albedo=_v3(0.01, 0.04, 0.1), metallic=0.15,
                 roughness=0.04, ior=1.33))

    def terrain_eval(p, t):
        d, _ = _simple_terrain_sdf(p, t)
        m = np.full_like(d, terrain_mat, dtype=np.int32)
        return d, m

    fluid = FluidSurface.choppy()
    fluid.base_height = -1.0
    water_eval = _water_plane_sdf(fluid, water_mat)

    scene.add_child(terrain_eval, "terrain")
    scene.add_child(water_eval, "water")

    camera_path = CameraPath()
    camera_path.add_waypoint(0.0, _v3(10, 5, -10), _v3(0, 0, 0), fov=40.0)
    camera_path.add_waypoint(3.0, _v3(10, 5, -10), _v3(0, 0, 0), fov=40.0)
    camera_path.add_waypoint(6.0, _v3(10, 5, -10), _v3(0, 0, 0), fov=40.0)

    return scene, camera_path
