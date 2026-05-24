"""
NRSI Render Engine — SDF ray marching renderer with PBR shading.

Core rendering pipeline for the NRSI inference engine:
  - SDFScene holds SDF evaluator, materials, lights
  - NRSIRenderer performs sphere-traced ray marching with normal estimation,
    soft shadows, ambient occlusion, Phong/Cook-Torrance shading, sky model,
    fog, ACES tonemapping, and gamma correction
  - NumPy CPU path with transparent CuPy GPU acceleration

All array math is vectorised over (N,3) float32 point batches.
"""
from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

try:
    import cupy as _cp
    GPU_AVAILABLE = True
except ImportError:
    _cp = None
    GPU_AVAILABLE = False


def _get_xp(use_gpu: bool = True):
    if use_gpu and GPU_AVAILABLE:
        return _cp
    return np


Vec3 = np.ndarray

_F32 = np.float32

# ═══════════════════════════════════════════════════════════════════════════════
# VECTOR UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════


def _v3(x: float, y: float, z: float) -> Vec3:
    return np.array([x, y, z], dtype=_F32)


def _dot_batch(a: Vec3, b: Vec3) -> np.ndarray:
    return np.einsum("...i,...i->...", a, b)


def _norm_batch(v: Vec3) -> np.ndarray:
    return np.sqrt(_dot_batch(v, v).clip(1e-12))


def _normalize_batch(v: Vec3) -> Vec3:
    n = _norm_batch(v)
    if v.ndim == 1:
        return v / max(float(n), 1e-9)
    return v / n[..., np.newaxis].clip(1e-9)


def _clamp(x, lo=0.0, hi=1.0):
    return np.clip(x, lo, hi)


def _reflect_batch(incident: Vec3, normal: Vec3) -> Vec3:
    d = _dot_batch(incident, normal)
    if incident.ndim == 1:
        return incident - 2.0 * d * normal
    return incident - 2.0 * d[..., np.newaxis] * normal


# ═══════════════════════════════════════════════════════════════════════════════
# SDF PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════════


def sd_sphere(p: Vec3, radius: float = 1.0) -> np.ndarray:
    return _norm_batch(p) - radius


def sd_box(p: Vec3, half_extents: Vec3 = None) -> np.ndarray:
    h = np.asarray(half_extents if half_extents is not None else _v3(0.5, 0.5, 0.5), dtype=_F32)
    q = np.abs(p) - h
    return _norm_batch(np.maximum(q, 0.0)) + np.minimum(np.max(q, axis=-1), 0.0)


def sd_plane(p: Vec3, normal: Vec3 = None, offset: float = 0.0) -> np.ndarray:
    n = np.asarray(normal if normal is not None else _v3(0.0, 1.0, 0.0), dtype=_F32)
    return _dot_batch(p, n) + offset


# ═══════════════════════════════════════════════════════════════════════════════
# SDF OPERATORS
# ═══════════════════════════════════════════════════════════════════════════════


def op_smooth_union(d1: np.ndarray, d2: np.ndarray, k: float = 0.1) -> np.ndarray:
    h = _clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0)
    return d2 * (1.0 - h) + d1 * h - k * h * (1.0 - h)


def op_translate(p: Vec3, offset) -> Vec3:
    return p - np.asarray(offset, dtype=_F32)


def op_rotate_y(p: Vec3, angle: float) -> Vec3:
    c, s = math.cos(angle), math.sin(angle)
    x = p[..., 0] * c + p[..., 2] * s
    z = -p[..., 0] * s + p[..., 2] * c
    return np.stack([x, p[..., 1], z], axis=-1).astype(_F32)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Material:
    albedo: Vec3 = field(default_factory=lambda: _v3(0.8, 0.8, 0.8))
    roughness: float = 0.5
    metallic: float = 0.0
    ior: float = 1.5
    emission: Vec3 = field(default_factory=lambda: _v3(0.0, 0.0, 0.0))


@dataclass
class Camera:
    position: Vec3 = field(default_factory=lambda: _v3(0.0, 2.0, -5.0))
    target: Vec3 = field(default_factory=lambda: _v3(0.0, 0.0, 0.0))
    up: Vec3 = field(default_factory=lambda: _v3(0.0, 1.0, 0.0))
    fov_deg: float = 50.0

    def ray_directions(self, width: int, height: int) -> Vec3:
        fwd = _normalize_batch(self.target - self.position)
        right = _normalize_batch(np.cross(fwd, self.up))
        up = np.cross(right, fwd)

        aspect = width / height
        fov_h = math.tan(math.radians(self.fov_deg) / 2.0)

        yy, xx = np.mgrid[0:height, 0:width]
        ndc_x = ((xx.ravel() + 0.5) / width * 2.0 - 1.0).astype(_F32) * fov_h * aspect
        ndc_y = (1.0 - (yy.ravel() + 0.5) / height * 2.0).astype(_F32) * fov_h

        dirs = (fwd[np.newaxis, :]
                + right[np.newaxis, :] * ndc_x[:, np.newaxis]
                + up[np.newaxis, :] * ndc_y[:, np.newaxis])
        return _normalize_batch(dirs)


@dataclass
class Light:
    direction: Vec3 = field(default_factory=lambda: _normalize_batch(_v3(0.5, 0.8, -0.3)))
    color: Vec3 = field(default_factory=lambda: _v3(1.0, 0.98, 0.95))
    intensity: float = 1.5


@dataclass
class RenderConfig:
    width: int = 512
    height: int = 512
    max_march_steps: int = 128
    max_march_dist: float = 100.0
    surface_eps: float = 0.001
    normal_eps: float = 0.0005
    shadow_steps: int = 32
    shadow_softness: float = 8.0
    ao_steps: int = 5
    ao_step_size: float = 0.08
    ao_strength: float = 0.6
    fog_density: float = 0.02
    fog_start: float = 10.0
    sky_zenith: Vec3 = field(default_factory=lambda: _v3(0.15, 0.25, 0.55))
    sky_horizon: Vec3 = field(default_factory=lambda: _v3(0.6, 0.7, 0.85))
    exposure: float = 1.2
    gamma: float = 2.2


# ═══════════════════════════════════════════════════════════════════════════════
# SDF SCENE
# ═══════════════════════════════════════════════════════════════════════════════


class SDFScene:
    """Container for SDF evaluator and materials.

    Override ``evaluate`` or register child evaluators via ``add_child``.
    """

    def __init__(self):
        self.materials: List[Material] = [Material()]
        self._evaluators: List[Tuple[Callable, int]] = []
        self._current_t: float = 0.0

    def add_material(self, mat: Material) -> int:
        self.materials.append(mat)
        return len(self.materials) - 1

    def add_primitive(self, sdf_fn: Callable[[Vec3], np.ndarray],
                      material_id: int = 0):
        self._evaluators.append((sdf_fn, material_id))

    def evaluate(self, p: Vec3) -> Tuple[np.ndarray, np.ndarray]:
        if p.ndim == 1:
            p = p[np.newaxis, :]
        N = p.shape[0]
        dist = np.full(N, 1e10, dtype=_F32)
        mats = np.zeros(N, dtype=np.int32)

        for sdf_fn, mat_id in self._evaluators:
            d = sdf_fn(p)
            closer = d < dist
            dist = np.where(closer, d, dist)
            mats = np.where(closer, mat_id, mats)

        return dist, mats


# ═══════════════════════════════════════════════════════════════════════════════
# RAY MARCHING RENDERER
# ═══════════════════════════════════════════════════════════════════════════════


class NRSIRenderer:
    """Sphere-traced SDF renderer with PBR shading pipeline."""

    def __init__(self, scene: SDFScene, config: RenderConfig = None):
        self.scene = scene
        self.cfg = config or RenderConfig()

    # ── ray marching ──────────────────────────────────────────────────────

    def _march(self, origins: Vec3, dirs: Vec3
               ) -> Tuple[np.ndarray, Vec3, np.ndarray]:
        N = origins.shape[0]
        t = np.full(N, self.cfg.surface_eps, dtype=_F32)
        hit = np.zeros(N, dtype=bool)
        active = np.ones(N, dtype=bool)
        mat_ids = np.zeros(N, dtype=np.int32)

        for _ in range(self.cfg.max_march_steps):
            if not active.any():
                break
            idx = np.where(active)[0]
            pos = origins[idx] + dirs[idx] * t[idx, np.newaxis]
            d, m = self.scene.evaluate(pos)

            t[idx] += d
            converged = d < self.cfg.surface_eps
            hit[idx[converged]] = True
            mat_ids[idx[converged]] = m[converged]
            active[idx[converged]] = False
            active[idx[t[idx] > self.cfg.max_march_dist]] = False

        positions = origins + dirs * t[:, np.newaxis]
        return hit, positions, mat_ids

    # ── normal estimation (central differences) ──────────────────────────

    def _normals(self, p: Vec3) -> Vec3:
        eps = self.cfg.normal_eps
        e = np.array([eps, 0.0], dtype=_F32)

        d_x = (self.scene.evaluate(p + np.array([e[0], e[1], e[1]]))[0]
                - self.scene.evaluate(p - np.array([e[0], e[1], e[1]]))[0])
        d_y = (self.scene.evaluate(p + np.array([e[1], e[0], e[1]]))[0]
                - self.scene.evaluate(p - np.array([e[1], e[0], e[1]]))[0])
        d_z = (self.scene.evaluate(p + np.array([e[1], e[1], e[0]]))[0]
                - self.scene.evaluate(p - np.array([e[1], e[1], e[0]]))[0])

        normals = np.stack([d_x, d_y, d_z], axis=-1)
        return _normalize_batch(normals)

    # ── soft shadows (sphere-traced penumbra) ────────────────────────────

    def _soft_shadow(self, origins: Vec3, light_dir: Vec3,
                     max_dist: float = 40.0) -> np.ndarray:
        if light_dir.ndim == 1:
            l_dir = np.broadcast_to(light_dir, origins.shape).copy()
        else:
            l_dir = light_dir

        N = origins.shape[0]
        shadow = np.ones(N, dtype=_F32)
        t = np.full(N, self.cfg.surface_eps * 10, dtype=_F32)
        active = np.ones(N, dtype=bool)

        for _ in range(self.cfg.shadow_steps):
            if not active.any():
                break
            idx = np.where(active)[0]
            pos = origins[idx] + l_dir[idx] * t[idx, np.newaxis]
            d, _ = self.scene.evaluate(pos)

            shadow[idx] = np.minimum(
                shadow[idx],
                self.cfg.shadow_softness * d / t[idx].clip(0.001))
            t[idx] += d.clip(self.cfg.surface_eps)

            active[idx[d < self.cfg.surface_eps * 0.5]] = False
            active[idx[t[idx] > max_dist]] = False

        return _clamp(shadow, 0.0, 1.0)

    # ── ambient occlusion ────────────────────────────────────────────────

    def _ao(self, p: Vec3, n: Vec3) -> np.ndarray:
        N = p.shape[0]
        occ = np.zeros(N, dtype=_F32)
        scale = 1.0

        for i in range(1, self.cfg.ao_steps + 1):
            step = i * self.cfg.ao_step_size
            sample_p = p + n * step
            d, _ = self.scene.evaluate(sample_p)
            occ += (step - d.clip(0)) * scale
            scale *= 0.5

        return _clamp(1.0 - occ * self.cfg.ao_strength, 0.0, 1.0)

    # ── sky model ────────────────────────────────────────────────────────

    def _sky_color(self, dirs: Vec3, sun_dir: Vec3 = None) -> Vec3:
        if sun_dir is None:
            sun_dir = _normalize_batch(_v3(0.5, 0.8, -0.3))
        y = dirs[..., 1] if dirs.ndim > 1 else dirs[1]
        t = _clamp(y * 0.5 + 0.5, 0.0, 1.0)

        if dirs.ndim == 1:
            sky = self.cfg.sky_horizon * (1.0 - t) + self.cfg.sky_zenith * t
            sun_dot = max(float(_dot_batch(dirs, sun_dir)), 0.0)
            sun_glow = math.pow(max(sun_dot, 0.0), 32.0) * 0.8
            sky = sky + _v3(1.0, 0.9, 0.7) * sun_glow
            return sky

        sky = (self.cfg.sky_horizon[np.newaxis, :] * (1.0 - t[..., np.newaxis])
               + self.cfg.sky_zenith[np.newaxis, :] * t[..., np.newaxis])
        sun_dot = _clamp(_dot_batch(dirs, sun_dir), 0.0, 1.0)
        sun_glow = np.power(sun_dot, 32.0) * 0.8
        sky = sky + _v3(1.0, 0.9, 0.7)[np.newaxis, :] * sun_glow[..., np.newaxis]
        return sky.astype(_F32)

    # ── fog ───────────────────────────────────────────────────────────────

    def _apply_fog(self, color: Vec3, dist: np.ndarray,
                   sky_color: Vec3) -> Vec3:
        fog_factor = 1.0 - np.exp(-self.cfg.fog_density *
                                   np.maximum(dist - self.cfg.fog_start, 0.0))
        fog_factor = _clamp(fog_factor, 0.0, 1.0)
        if color.ndim == 1:
            return color * (1.0 - fog_factor) + sky_color * fog_factor
        return (color * (1.0 - fog_factor[..., np.newaxis])
                + sky_color * fog_factor[..., np.newaxis])

    # ── ACES tonemapping ─────────────────────────────────────────────────

    @staticmethod
    def _tonemap_aces(x: np.ndarray) -> np.ndarray:
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        return _clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)

    # ── shading ──────────────────────────────────────────────────────────

    def _shade(self, p: Vec3, n: Vec3, view_dir: Vec3,
               mat: Material, light: Light, ao: np.ndarray,
               shadow: np.ndarray) -> Vec3:
        N = p.shape[0]
        albedo = np.broadcast_to(mat.albedo, (N, 3)).astype(_F32)
        roughness = max(mat.roughness, 0.04)
        a2 = roughness ** 4

        l_dir = np.broadcast_to(light.direction, (N, 3)).astype(_F32)
        v = -view_dir

        ndl = _clamp(_dot_batch(n, l_dir), 0.0, 1.0)
        ndv = _clamp(_dot_batch(n, v), 0.001, 1.0)

        h_vec = _normalize_batch(l_dir + v)
        ndh = _clamp(_dot_batch(n, h_vec), 0.0, 1.0)
        vdh = _clamp(_dot_batch(v, h_vec), 0.0, 1.0)

        denom = ndh * ndh * (a2 - 1.0) + 1.0
        D = a2 / (np.pi * denom * denom + 1e-7)

        F0 = 0.04 * (1.0 - mat.metallic) + mat.metallic
        F = F0 + (1.0 - F0) * np.power(1.0 - vdh, 5.0)

        k = (roughness + 1.0) ** 2 / 8.0
        G = (ndv / (ndv * (1.0 - k) + k)) * (ndl / (ndl * (1.0 - k) + k))

        spec = (D * F * G) / (4.0 * ndv * ndl + 0.001)
        kd = (1.0 - F) * (1.0 - mat.metallic)

        radiance = light.color * light.intensity
        lit = radiance[np.newaxis, :] * ndl[:, np.newaxis] * shadow[:, np.newaxis]

        diffuse = kd[:, np.newaxis] * albedo / np.pi
        color = (diffuse + spec[:, np.newaxis]) * lit

        ambient = self.cfg.sky_horizon * 0.15
        color += albedo * ambient[np.newaxis, :] * ao[:, np.newaxis]

        if np.any(mat.emission > 0):
            color += np.broadcast_to(mat.emission, (N, 3)).astype(_F32)

        return color

    # ── main render ──────────────────────────────────────────────────────

    def render(self, camera: Camera, lighting=None) -> np.ndarray:
        """Render the scene from *camera*. Returns (H, W, 3) uint8 image."""
        if lighting is None:
            lighting = Light()

        w, h = self.cfg.width, self.cfg.height
        N = w * h

        origins = np.broadcast_to(camera.position, (N, 3)).astype(_F32).copy()
        dirs = camera.ray_directions(w, h)

        hit, pos, mat_ids = self._march(origins, dirs)

        color = np.zeros((N, 3), dtype=_F32)

        miss = ~hit
        if miss.any():
            sun_dir = _normalize_batch(lighting.direction)
            color[miss] = self._sky_color(dirs[miss], sun_dir)

        if hit.any():
            h_pos = pos[hit]
            h_dirs = dirs[hit]
            h_mat_ids = mat_ids[hit]

            normals = self._normals(h_pos)
            ao = self._ao(h_pos, normals)

            shadow_origin = h_pos + normals * self.cfg.surface_eps * 3.0
            shadow = self._soft_shadow(shadow_origin, lighting.direction)

            unique_mats = np.unique(h_mat_ids)
            h_color = np.zeros((hit.sum(), 3), dtype=_F32)

            for mid in unique_mats:
                if mid < 0 or mid >= len(self.scene.materials):
                    continue
                mask = h_mat_ids == mid
                if not mask.any():
                    continue
                mat = self.scene.materials[mid]
                h_color[mask] = self._shade(
                    h_pos[mask], normals[mask], h_dirs[mask],
                    mat, lighting, ao[mask], shadow[mask])

            dist = _norm_batch(h_pos - origins[hit])
            sun_dir = _normalize_batch(lighting.direction)
            sky_at_hit = self._sky_color(h_dirs, sun_dir)
            h_color = self._apply_fog(h_color, dist, sky_at_hit)

            color[hit] = h_color

        img = color.reshape(h, w, 3)
        img = img * self.cfg.exposure
        img = self._tonemap_aces(img)
        img = np.power(_clamp(img), 1.0 / self.cfg.gamma)

        return (_clamp(img) * 255).astype(np.uint8)


__all__ = [
    "Vec3", "_v3", "_dot_batch", "_norm_batch", "_normalize_batch",
    "_reflect_batch", "_clamp", "_get_xp", "GPU_AVAILABLE",
    "sd_sphere", "sd_box", "sd_plane",
    "op_smooth_union", "op_translate", "op_rotate_y",
    "Material", "Camera", "Light", "RenderConfig",
    "SDFScene", "NRSIRenderer",
]
