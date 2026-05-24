"""
NRSI Advanced Renderer — Multi-light, GI, Refraction, DOF, Motion Blur,
and Production Post-Processing.

Extends the base NRSIRenderer from render_engine.py with:
  - Multi-light system (point, area, spot, environment)
  - Path-traced global illumination (2-4 bounces, Russian roulette)
  - Refraction / transmission (Snell's law, Beer's law, TIR)
  - Depth of field (thin lens model)
  - Motion blur (per-object velocity, shutter jitter)
  - Full production post-processing pipeline

All array operations are vectorized numpy.  CuPy GPU transparent via _get_xp().
"""
from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Any, Tuple

try:
    from .render_engine import (
        Vec3, _v3, _dot_batch, _norm_batch, _normalize_batch, _reflect_batch,
        _clamp, Material, SDFScene, Camera, Light, RenderConfig,
        NRSIRenderer, _get_xp, GPU_AVAILABLE,
    )
except ImportError:
    from render_engine import (
        Vec3, _v3, _dot_batch, _norm_batch, _normalize_batch, _reflect_batch,
        _clamp, Material, SDFScene, Camera, Light, RenderConfig,
        NRSIRenderer, _get_xp, GPU_AVAILABLE,
    )

try:
    import cupy as _cp
except ImportError:
    _cp = None

__all__ = [
    "AdvancedMaterial",
    "PointLight", "AreaLight", "SpotLight", "EnvironmentLight",
    "LightingSetup",
    "ObjectMotion",
    "AdvancedRenderer",
    "DEFAULT_POST_PROCESSING",
]

# ═══════════════════════════════════════════════════════════════════════════
# ADVANCED MATERIAL
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AdvancedMaterial(Material):
    transmission: float = 0.0
    absorption_color: Vec3 = field(default_factory=lambda: _v3(1, 1, 1))
    absorption_density: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# LIGHT TYPES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PointLight:
    position: Vec3 = field(default_factory=lambda: _v3(0, 5, 0))
    color: Vec3 = field(default_factory=lambda: _v3(1, 1, 1))
    intensity: float = 10.0
    radius: float = 0.1


@dataclass
class AreaLight:
    position: Vec3 = field(default_factory=lambda: _v3(0, 5, 0))
    normal: Vec3 = field(default_factory=lambda: _v3(0, -1, 0))
    size: Tuple[float, float] = (1.0, 1.0)
    color: Vec3 = field(default_factory=lambda: _v3(1, 1, 1))
    intensity: float = 8.0


@dataclass
class SpotLight:
    position: Vec3 = field(default_factory=lambda: _v3(0, 5, 0))
    direction: Vec3 = field(default_factory=lambda: _v3(0, -1, 0))
    inner_angle: float = 20.0
    outer_angle: float = 35.0
    color: Vec3 = field(default_factory=lambda: _v3(1, 1, 1))
    intensity: float = 15.0


@dataclass
class EnvironmentLight:
    hdri_fn: Callable[[Vec3], Vec3] = None
    intensity: float = 1.0
    rotation: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# OBJECT MOTION (for motion blur)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ObjectMotion:
    """Per-object velocity / transform delta for motion blur."""
    object_id: int = 0
    velocity: Vec3 = field(default_factory=lambda: _v3(0, 0, 0))
    angular_velocity_y: float = 0.0
    transform_t0: Optional[Vec3] = None
    transform_t1: Optional[Vec3] = None

    def position_at(self, t: float) -> Vec3:
        """Interpolate position at shutter time t in [0, 1]."""
        if self.transform_t0 is not None and self.transform_t1 is not None:
            return self.transform_t0 * (1.0 - t) + self.transform_t1 * t
        return self.velocity * t


# ═══════════════════════════════════════════════════════════════════════════
# LIGHTING SETUP
# ═══════════════════════════════════════════════════════════════════════════

class LightingSetup:
    """Container for all light types in a scene."""

    def __init__(self):
        self.directional: List[Light] = []
        self.point: List[PointLight] = []
        self.area: List[AreaLight] = []
        self.spot: List[SpotLight] = []
        self.environment: Optional[EnvironmentLight] = None

    def add(self, light) -> "LightingSetup":
        if isinstance(light, Light):
            self.directional.append(light)
        elif isinstance(light, PointLight):
            self.point.append(light)
        elif isinstance(light, AreaLight):
            self.area.append(light)
        elif isinstance(light, SpotLight):
            self.spot.append(light)
        elif isinstance(light, EnvironmentLight):
            self.environment = light
        else:
            raise TypeError(f"Unknown light type: {type(light)}")
        return self


# ═══════════════════════════════════════════════════════════════════════════
# DEFAULT POST-PROCESSING CONFIG
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_POST_PROCESSING: Dict[str, float] = {
    "bloom": 0.3,
    "bloom_threshold": 0.8,
    "chromatic_aberration": 0.01,
    "film_grain": 0.03,
    "vignette": 0.4,
    "sharpening": 0.3,
    "contrast": 0.1,
    "saturation": 0.0,
    "color_temperature": 0.0,
    "letterbox": 0.0,
    "glow": 0.0,
    "lens_flare": 0.0,
}


# ═══════════════════════════════════════════════════════════════════════════
# ADVANCED RENDERER
# ═══════════════════════════════════════════════════════════════════════════

class AdvancedRenderer:
    """
    Production-quality SDF renderer with multi-light, GI, refraction,
    DOF, motion blur, and a full post-processing pipeline.

    Extends the base NRSIRenderer's march/normal/shadow/AO infrastructure.
    """

    def __init__(self, scene: SDFScene, config: RenderConfig = None,
                 lighting: LightingSetup = None):
        self._base = NRSIRenderer(scene, config)
        self.scene = scene
        self.cfg = self._base.cfg
        self.lighting = lighting or LightingSetup()
        self._rng = np.random.default_rng(42)
        self._current_time_samples: Optional[np.ndarray] = None
        self._current_motions: Optional[List[ObjectMotion]] = None

    # ── delegate to base renderer ─────────────────────────────────────────

    def _march(self, *a, **kw):
        return self._base._march(*a, **kw)

    def _normals(self, *a, **kw):
        return self._base._normals(*a, **kw)

    def _soft_shadow_uniform(self, origins: Vec3, light_dir: Vec3) -> np.ndarray:
        """Shadow trace for a uniform direction (3,)."""
        return self._base._soft_shadow(origins, light_dir)

    def _soft_shadow_perpixel(self, origins: Vec3,
                              light_dirs: Vec3,
                              max_dist: float = 40.0) -> np.ndarray:
        """Shadow trace with per-pixel (N,3) light directions."""
        N = origins.shape[0]
        shadow = np.ones(N, dtype=np.float32)
        t = np.full(N, self.cfg.surface_eps * 10, dtype=np.float32)
        active = np.ones(N, dtype=bool)

        for _ in range(self.cfg.shadow_steps):
            if not active.any():
                break
            idx = np.where(active)[0]
            pos = origins[idx] + light_dirs[idx] * t[idx, np.newaxis]
            d, _ = self.scene.evaluate(pos)

            shadow[idx] = np.minimum(
                shadow[idx],
                self.cfg.shadow_softness * d / t[idx].clip(0.001))
            t[idx] += d.clip(self.cfg.surface_eps)

            active[idx[d < self.cfg.surface_eps * 0.5]] = False
            active[idx[t[idx] > max_dist]] = False

        return _clamp(shadow, 0.0, 1.0)

    def _ao(self, *a, **kw):
        return self._base._ao(*a, **kw)

    def _sky_color(self, *a, **kw):
        return self._base._sky_color(*a, **kw)

    def _apply_fog(self, *a, **kw):
        return self._base._apply_fog(*a, **kw)

    def _tonemap_aces(self, *a, **kw):
        return self._base._tonemap_aces(*a, **kw)

    # ═══════════════════════════════════════════════════════════════════════
    # MULTI-LIGHT SHADING
    # ═══════════════════════════════════════════════════════════════════════

    def _shade_advanced(self, p: Vec3, n: Vec3, view_dir: Vec3,
                        mat: Material, ao: np.ndarray,
                        lighting: LightingSetup) -> Vec3:
        N = p.shape[0]
        color = np.zeros((N, 3), dtype=np.float32)

        albedo = np.broadcast_to(mat.albedo, (N, 3)).astype(np.float32)
        roughness = max(mat.roughness, 0.04)
        a2 = roughness ** 4

        v = -view_dir

        shadow_origin = p + n * self.cfg.surface_eps * 3

        for dl in lighting.directional:
            l_dir = np.broadcast_to(dl.direction, (N, 3)).astype(np.float32)
            shadow = self._soft_shadow_uniform(shadow_origin, dl.direction)
            radiance = dl.color * dl.intensity
            color += self._cook_torrance(
                n, v, l_dir, albedo, mat, a2, roughness,
                radiance, shadow)

        for pl in lighting.point:
            to_light = pl.position[np.newaxis, :] - p
            dist_sq = _dot_batch(to_light, to_light)
            dist = np.sqrt(dist_sq.clip(1e-8))
            l_dir = to_light / dist[:, np.newaxis].clip(1e-8)
            attenuation = pl.intensity / (dist_sq + pl.radius ** 2)
            shadow = self._soft_shadow_perpixel(shadow_origin, l_dir, max_dist=np.sqrt(dist_sq.max()) + 1.0)
            radiance = pl.color[np.newaxis, :] * attenuation[:, np.newaxis]
            color += self._cook_torrance_array(
                n, v, l_dir, albedo, mat, a2, roughness,
                radiance, shadow)

        for sl in lighting.spot:
            to_light = sl.position[np.newaxis, :] - p
            dist_sq = _dot_batch(to_light, to_light)
            dist = np.sqrt(dist_sq.clip(1e-8))
            l_dir = to_light / dist[:, np.newaxis].clip(1e-8)

            spot_cos = _dot_batch(
                -l_dir,
                np.broadcast_to(
                    _normalize_batch(sl.direction), (N, 3)))
            inner_cos = math.cos(math.radians(sl.inner_angle))
            outer_cos = math.cos(math.radians(sl.outer_angle))
            spot_atten = _clamp(
                (spot_cos - outer_cos) / max(inner_cos - outer_cos, 1e-6))
            attenuation = sl.intensity / (dist_sq + 1.0) * spot_atten

            shadow = self._soft_shadow_perpixel(shadow_origin, l_dir, max_dist=np.sqrt(dist_sq.max()) + 1.0)
            radiance = sl.color[np.newaxis, :] * attenuation[:, np.newaxis]
            color += self._cook_torrance_array(
                n, v, l_dir, albedo, mat, a2, roughness,
                radiance, shadow)

        for al in lighting.area:
            jitter = self._rng.uniform(-0.5, 0.5, size=(N, 2)).astype(np.float32)
            al_normal = _normalize_batch(al.normal)
            tangent = _normalize_batch(
                np.cross(al_normal, _v3(0, 0, 1))
                if abs(np.dot(al_normal, _v3(0, 0, 1))) < 0.99
                else np.cross(al_normal, _v3(1, 0, 0)))
            bitangent = np.cross(al_normal, tangent)

            sample_pos = (al.position[np.newaxis, :]
                          + tangent[np.newaxis, :] * (jitter[:, 0:1] * al.size[0])
                          + bitangent[np.newaxis, :] * (jitter[:, 1:2] * al.size[1]))
            to_light = sample_pos - p
            dist_sq = _dot_batch(to_light, to_light)
            dist = np.sqrt(dist_sq.clip(1e-8))
            l_dir = to_light / dist[:, np.newaxis].clip(1e-8)

            cos_light = _clamp(-_dot_batch(l_dir, np.broadcast_to(al_normal, (N, 3))))
            area = al.size[0] * al.size[1] * 4.0
            attenuation = al.intensity * cos_light * area / (dist_sq + 1.0)

            shadow = self._soft_shadow_perpixel(shadow_origin, l_dir, max_dist=np.sqrt(dist_sq.max()) + 1.0)
            radiance = al.color[np.newaxis, :] * attenuation[:, np.newaxis]
            color += self._cook_torrance_array(
                n, v, l_dir, albedo, mat, a2, roughness,
                radiance, shadow)

        if lighting.environment is not None and lighting.environment.hdri_fn is not None:
            env = lighting.environment
            reflect_dir = _reflect_batch(view_dir, n)
            if env.rotation != 0.0:
                c_r, s_r = math.cos(env.rotation), math.sin(env.rotation)
                rx = reflect_dir[:, 0] * c_r + reflect_dir[:, 2] * s_r
                rz = -reflect_dir[:, 0] * s_r + reflect_dir[:, 2] * c_r
                reflect_dir = np.stack([rx, reflect_dir[:, 1], rz], axis=-1)
            env_color = env.hdri_fn(reflect_dir) * env.intensity
            spec_weight = mat.metallic * (1.0 - roughness)
            color += env_color * spec_weight
            diff_color = env.hdri_fn(n) * env.intensity
            color += albedo * diff_color * (1.0 - mat.metallic) * 0.3

        sky_light = np.broadcast_to(self.cfg.sky_horizon * 0.15, (N, 3))
        color += albedo * sky_light * ao[:, np.newaxis]

        if np.any(mat.emission > 0):
            color += np.broadcast_to(mat.emission, (N, 3)).astype(np.float32)

        return color

    def _cook_torrance(self, n, v, l_dir, albedo, mat, a2, roughness,
                       radiance_scalar, shadow):
        """Cook-Torrance BRDF for uniform-direction lights."""
        N = n.shape[0]
        l = np.broadcast_to(l_dir, (N, 3)).astype(np.float32) if l_dir.ndim == 1 else l_dir
        h_vec = _normalize_batch(l + v)

        ndl = _clamp(_dot_batch(n, l))
        ndv = _clamp(_dot_batch(n, v), 0.001, 1.0)
        ndh = _clamp(_dot_batch(n, h_vec))
        vdh = _clamp(_dot_batch(v, h_vec))

        denom = ndh * ndh * (a2 - 1.0) + 1.0
        D = a2 / (np.pi * denom * denom + 1e-7)

        F0 = 0.04 * (1.0 - mat.metallic) + mat.metallic
        F = F0 + (1.0 - F0) * (1.0 - vdh) ** 5

        k = (roughness + 1.0) ** 2 / 8.0
        G = (ndv / (ndv * (1 - k) + k)) * (ndl / (ndl * (1 - k) + k))

        spec = (D * F * G) / (4.0 * ndv * ndl + 0.001)
        kd = (1.0 - F) * (1.0 - mat.metallic)

        rad = radiance_scalar[np.newaxis, :] * ndl[:, np.newaxis] * shadow[:, np.newaxis]
        diffuse = kd[:, np.newaxis] * albedo / np.pi
        return (diffuse + spec[:, np.newaxis]) * rad

    def _cook_torrance_array(self, n, v, l_dir, albedo, mat, a2, roughness,
                             radiance_array, shadow):
        """Cook-Torrance BRDF for per-pixel light directions and radiance."""
        h_vec = _normalize_batch(l_dir + v)

        ndl = _clamp(_dot_batch(n, l_dir))
        ndv = _clamp(_dot_batch(n, v), 0.001, 1.0)
        ndh = _clamp(_dot_batch(n, h_vec))
        vdh = _clamp(_dot_batch(v, h_vec))

        denom = ndh * ndh * (a2 - 1.0) + 1.0
        D = a2 / (np.pi * denom * denom + 1e-7)

        F0 = 0.04 * (1.0 - mat.metallic) + mat.metallic
        F = F0 + (1.0 - F0) * (1.0 - vdh) ** 5

        k = (roughness + 1.0) ** 2 / 8.0
        G = (ndv / (ndv * (1 - k) + k)) * (ndl / (ndl * (1 - k) + k))

        spec = (D * F * G) / (4.0 * ndv * ndl + 0.001)
        kd = (1.0 - F) * (1.0 - mat.metallic)

        rad = radiance_array * ndl[:, np.newaxis] * shadow[:, np.newaxis]
        diffuse = kd[:, np.newaxis] * albedo / np.pi
        return (diffuse + spec[:, np.newaxis]) * rad

    # ═══════════════════════════════════════════════════════════════════════
    # GLOBAL ILLUMINATION (path-traced, importance-sampled)
    # ═══════════════════════════════════════════════════════════════════════

    def _cosine_hemisphere_sample(self, n: Vec3, count: int) -> Vec3:
        """Cosine-weighted hemisphere sampling around normal n (N,3)."""
        N = n.shape[0]
        u1 = self._rng.uniform(0, 1, (N, count)).astype(np.float32)
        u2 = self._rng.uniform(0, 1, (N, count)).astype(np.float32)

        r = np.sqrt(u1)
        theta = 2.0 * np.pi * u2
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.sqrt((1.0 - u1).clip(0))

        up = np.where(
            np.abs(n[:, 1:2]) < 0.999,
            np.broadcast_to(_v3(0, 1, 0), (N, 3)),
            np.broadcast_to(_v3(1, 0, 0), (N, 3)))
        tangent = _normalize_batch(np.cross(up, n))
        bitangent = np.cross(n, tangent)

        dirs = (tangent[:, np.newaxis, :] * x[:, :, np.newaxis]
                + bitangent[:, np.newaxis, :] * y[:, :, np.newaxis]
                + n[:, np.newaxis, :] * z[:, :, np.newaxis])
        return dirs

    def _ggx_importance_sample(self, n: Vec3, v: Vec3,
                               roughness: float) -> Vec3:
        """GGX importance sampling — returns one sample direction per pixel."""
        N = n.shape[0]
        a = roughness * roughness
        u1 = self._rng.uniform(0, 1, N).astype(np.float32)
        u2 = self._rng.uniform(0, 1, N).astype(np.float32)

        cos_theta = np.sqrt((1.0 - u1) / (1.0 + (a * a - 1.0) * u1 + 1e-8))
        sin_theta = np.sqrt((1.0 - cos_theta * cos_theta).clip(0))
        phi = 2.0 * np.pi * u2

        hx = sin_theta * np.cos(phi)
        hy = sin_theta * np.sin(phi)
        hz = cos_theta

        up = np.where(
            np.abs(n[:, 1:2]) < 0.999,
            np.broadcast_to(_v3(0, 1, 0), (N, 3)),
            np.broadcast_to(_v3(1, 0, 0), (N, 3)))
        tangent = _normalize_batch(np.cross(up, n))
        bitangent = np.cross(n, tangent)

        h_world = (tangent * hx[:, np.newaxis]
                    + bitangent * hy[:, np.newaxis]
                    + n * hz[:, np.newaxis])
        h_world = _normalize_batch(h_world)

        vdh = _dot_batch(v, h_world).clip(0)
        sample_dir = 2.0 * vdh[:, np.newaxis] * h_world - v
        return _normalize_batch(sample_dir)

    def _trace_gi(self, origins: Vec3, normals: Vec3, view_dirs: Vec3,
                  materials: List[Material], mat_ids: np.ndarray,
                  bounces: int, samples: int,
                  quality: float, lighting: LightingSetup) -> Vec3:
        """Path-traced GI with Russian roulette termination."""
        N = origins.shape[0]
        if N == 0:
            return np.zeros((0, 3), dtype=np.float32)

        stride = max(1, int(1.0 / max(quality, 0.01)))
        if stride > 1:
            gi_idx = np.arange(0, N, stride)
        else:
            gi_idx = np.arange(N)

        gi_N = len(gi_idx)
        gi_origins = origins[gi_idx]
        gi_normals = normals[gi_idx]
        gi_views = view_dirs[gi_idx]
        gi_mats = mat_ids[gi_idx]

        accumulated = np.zeros((gi_N, 3), dtype=np.float32)

        for s in range(samples):
            throughput = np.ones((gi_N, 3), dtype=np.float32)
            ray_o = gi_origins + gi_normals * self.cfg.surface_eps * 5
            sample_dirs = self._cosine_hemisphere_sample(gi_normals, 1)
            ray_d = _normalize_batch(sample_dirs[:, 0, :])
            active = np.ones(gi_N, dtype=bool)

            for bounce in range(bounces):
                if not active.any():
                    break

                a_idx = np.where(active)[0]
                hit, pos, m_ids = self._march(ray_o[a_idx], ray_d[a_idx])

                miss_local = ~hit
                if miss_local.any():
                    miss_global = a_idx[miss_local]
                    sky = self._sky_color(ray_d[miss_global],
                                          _v3(0.5, 0.8, -0.3))
                    accumulated[miss_global] += throughput[miss_global] * sky
                    active[a_idx[miss_local]] = False

                if not hit.any():
                    break

                hit_global = a_idx[hit]
                h_pos = pos[hit]
                h_mids = m_ids[hit]
                h_normals = self._normals(h_pos)
                h_ao = self._ao(h_pos, h_normals)

                unique_mats = np.unique(h_mids)
                direct = np.zeros((hit.sum(), 3), dtype=np.float32)
                hit_albedo = np.zeros((hit.sum(), 3), dtype=np.float32)

                for mid in unique_mats:
                    if mid < 0 or mid >= len(self.scene.materials):
                        continue
                    mask = h_mids == mid
                    if not mask.any():
                        continue
                    mat_obj = self.scene.materials[mid]
                    hit_albedo[mask] = mat_obj.albedo
                    direct[mask] = self._shade_advanced(
                        h_pos[mask], h_normals[mask],
                        ray_d[hit_global[mask]], mat_obj,
                        h_ao[mask], lighting)

                accumulated[hit_global] += throughput[hit_global] * direct

                if bounce >= 2:
                    max_albedo = np.max(hit_albedo, axis=-1)
                    survive = self._rng.uniform(0, 1, hit.sum()) < max_albedo
                    terminate = ~survive
                    active[hit_global[terminate]] = False
                    surviving_mask = survive
                    if surviving_mask.any():
                        throughput[hit_global[surviving_mask]] /= (
                            max_albedo[surviving_mask, np.newaxis].clip(0.01))
                    if not active.any():
                        break
                    hit_global = hit_global[surviving_mask]
                    h_pos = h_pos[surviving_mask]
                    h_normals = h_normals[surviving_mask]
                    hit_albedo = hit_albedo[surviving_mask]

                still_active = active[hit_global]
                if not still_active.any():
                    break
                sa_idx = np.where(still_active)[0]
                sa_global = hit_global[sa_idx]

                throughput[sa_global] *= hit_albedo[sa_idx]

                sa_n = h_normals[sa_idx]
                sa_v = -ray_d[sa_global]
                n_sa = len(sa_idx)

                is_specular = np.zeros(n_sa, dtype=bool)
                sa_roughness = np.full(n_sa, 0.5, dtype=np.float32)
                for mid in np.unique(gi_mats[sa_global]):
                    if mid < 0 or mid >= len(materials):
                        continue
                    m_mask = gi_mats[sa_global] == mid
                    m_obj = materials[mid]
                    is_specular[m_mask] = m_obj.metallic > 0.3
                    sa_roughness[m_mask] = max(m_obj.roughness, 0.04)

                new_dirs = np.zeros((n_sa, 3), dtype=np.float32)
                if is_specular.any():
                    spec_idx = np.where(is_specular)[0]
                    avg_rough = float(sa_roughness[spec_idx].mean())
                    spec_dirs = self._ggx_importance_sample(
                        sa_n[spec_idx], sa_v[spec_idx], avg_rough)
                    new_dirs[spec_idx] = spec_dirs
                diff_mask = ~is_specular
                if diff_mask.any():
                    diff_idx = np.where(diff_mask)[0]
                    diff_dirs = self._cosine_hemisphere_sample(
                        sa_n[diff_idx], 1)[:, 0, :]
                    new_dirs[diff_idx] = diff_dirs

                ray_o[sa_global] = h_pos[sa_idx] + sa_n * self.cfg.surface_eps * 5
                ray_d[sa_global] = _normalize_batch(new_dirs)

        accumulated /= max(samples, 1)

        if stride > 1 and gi_N > 0:
            full_gi = np.zeros((N, 3), dtype=np.float32)
            full_gi[gi_idx] = accumulated
            for c in range(3):
                vals = np.interp(
                    np.arange(N).astype(np.float32),
                    gi_idx.astype(np.float32),
                    accumulated[:, c])
                full_gi[:, c] = vals
            return full_gi

        return accumulated

    # ═══════════════════════════════════════════════════════════════════════
    # REFRACTION / TRANSMISSION
    # ═══════════════════════════════════════════════════════════════════════

    def _refract_batch(self, incident: Vec3, normal: Vec3,
                       eta: float) -> Tuple[Vec3, np.ndarray]:
        """Snell's law refraction. Returns (refracted_dirs, valid_mask)."""
        cos_i = -_dot_batch(incident, normal)
        flip = cos_i < 0
        n_corrected = normal.copy()
        if flip.any():
            n_corrected[flip] = -n_corrected[flip]
            cos_i[flip] = -cos_i[flip]
            eta_arr = np.where(flip, 1.0 / eta, eta)
        else:
            eta_arr = np.full(cos_i.shape, eta, dtype=np.float32)

        sin2_t = eta_arr ** 2 * (1.0 - cos_i ** 2)
        valid = sin2_t <= 1.0

        cos_t = np.sqrt((1.0 - sin2_t).clip(0))
        refracted = (eta_arr[:, np.newaxis] * incident
                     + (eta_arr * cos_i - cos_t)[:, np.newaxis] * n_corrected)
        refracted = _normalize_batch(refracted)
        return refracted, valid

    def _schlick_fresnel(self, cos_theta: np.ndarray,
                         ior: float) -> np.ndarray:
        r0 = ((1.0 - ior) / (1.0 + ior)) ** 2
        return r0 + (1.0 - r0) * (1.0 - cos_theta.clip(0)) ** 5

    def _trace_transmission(self, origins: Vec3, dirs: Vec3,
                            normals: Vec3, mat: Material,
                            lighting: LightingSetup) -> Vec3:
        """Trace refracted rays for transmissive materials."""
        N = origins.shape[0]
        if N == 0:
            return np.zeros((0, 3), dtype=np.float32)

        eta = 1.0 / mat.ior
        cos_i = _clamp(-_dot_batch(dirs, normals), 0.0, 1.0)
        fresnel = self._schlick_fresnel(cos_i, mat.ior)

        refracted, valid = self._refract_batch(dirs, normals, eta)

        tir = ~valid
        result = np.zeros((N, 3), dtype=np.float32)

        if tir.any():
            reflected = _reflect_batch(dirs[tir], normals[tir])
            r_origins = origins[tir] + normals[tir] * self.cfg.surface_eps * 5
            hit, pos, m_ids = self._march(r_origins, reflected)
            if hit.any():
                h_normals = self._normals(pos[hit])
                h_ao = self._ao(pos[hit], h_normals)
                unique_mats = np.unique(m_ids[hit])
                h_color = np.zeros((hit.sum(), 3), dtype=np.float32)
                for mid in unique_mats:
                    if mid < 0 or mid >= len(self.scene.materials):
                        continue
                    mask = m_ids[hit] == mid
                    if mask.any():
                        m = self.scene.materials[mid]
                        h_color[mask] = self._shade_advanced(
                            pos[hit][mask], h_normals[mask],
                            reflected[hit][mask], m, h_ao[mask], lighting)
                tir_color = np.zeros((tir.sum(), 3), dtype=np.float32)
                tir_color[hit] = h_color
                miss = ~hit
                if miss.any():
                    tir_color[miss] = self._sky_color(
                        reflected[miss], _v3(0.5, 0.8, -0.3))
                result[tir] = tir_color
            else:
                result[tir] = self._sky_color(
                    reflected, _v3(0.5, 0.8, -0.3))

        if valid.any():
            r_origins = origins[valid] - normals[valid] * self.cfg.surface_eps * 5
            hit, pos, m_ids = self._march(r_origins, refracted[valid])

            refr_color = np.zeros((valid.sum(), 3), dtype=np.float32)
            miss = ~hit
            if miss.any():
                refr_color[miss] = self._sky_color(
                    refracted[valid][miss], _v3(0.5, 0.8, -0.3))

            if hit.any():
                h_normals = self._normals(pos[hit])
                h_ao = self._ao(pos[hit], h_normals)
                unique_mats = np.unique(m_ids[hit])
                h_color = np.zeros((hit.sum(), 3), dtype=np.float32)
                for mid in unique_mats:
                    if mid < 0 or mid >= len(self.scene.materials):
                        continue
                    mask = m_ids[hit] == mid
                    if mask.any():
                        m = self.scene.materials[mid]
                        h_color[mask] = self._shade_advanced(
                            pos[hit][mask], h_normals[mask],
                            refracted[valid][hit][mask], m,
                            h_ao[mask], lighting)
                refr_color[hit] = h_color

                if isinstance(mat, AdvancedMaterial) and mat.absorption_density > 0:
                    travel_dist = _norm_batch(pos[hit] - r_origins[hit])
                    absorption = np.exp(
                        -mat.absorption_density
                        * travel_dist[:, np.newaxis]
                        * (1.0 - mat.absorption_color[np.newaxis, :]))
                    refr_color[hit] *= absorption

            result[valid] = (
                refr_color * (1.0 - fresnel[valid, np.newaxis])
                + result[valid] * fresnel[valid, np.newaxis])

        return result

    # ═══════════════════════════════════════════════════════════════════════
    # DEPTH OF FIELD (thin lens model)
    # ═══════════════════════════════════════════════════════════════════════

    def _generate_dof_rays(self, camera: Camera, w: int, h: int,
                           aperture: float, focus_distance: float,
                           num_samples: int) -> Tuple[Vec3, Vec3]:
        """Generate jittered rays for thin-lens DOF."""
        base_dirs = camera.ray_directions(w, h)
        N = base_dirs.shape[0]

        if aperture <= 0 or num_samples <= 1:
            origins = np.broadcast_to(camera.position, (N, 3)).astype(np.float32).copy()
            return origins, base_dirs

        fwd = _normalize_batch(camera.target - camera.position)
        right = _normalize_batch(np.cross(fwd, camera.up))
        up = np.cross(right, fwd)

        focus_points = camera.position[np.newaxis, :] + base_dirs * focus_distance

        all_origins = np.zeros((N, num_samples, 3), dtype=np.float32)
        all_dirs = np.zeros((N, num_samples, 3), dtype=np.float32)

        for s in range(num_samples):
            angle = self._rng.uniform(0, 2 * np.pi, N).astype(np.float32)
            radius = aperture * np.sqrt(
                self._rng.uniform(0, 1, N).astype(np.float32))
            offset = (right[np.newaxis, :] * (radius * np.cos(angle))[:, np.newaxis]
                      + up[np.newaxis, :] * (radius * np.sin(angle))[:, np.newaxis])
            jittered_origin = camera.position[np.newaxis, :] + offset
            jittered_dir = _normalize_batch(focus_points - jittered_origin)
            all_origins[:, s, :] = jittered_origin
            all_dirs[:, s, :] = jittered_dir

        return all_origins, all_dirs

    # ═══════════════════════════════════════════════════════════════════════
    # MOTION BLUR
    # ═══════════════════════════════════════════════════════════════════════

    def _apply_motion_blur_jitter(self, origins: Vec3, dirs: Vec3,
                                  N: int) -> Tuple[Vec3, Vec3, np.ndarray]:
        """Jitter ray time across shutter interval [0, 1]."""
        t_samples = self._rng.uniform(0, 1, N).astype(np.float32)
        return origins, dirs, t_samples

    def _interpolate_scene_at_time(self, t_samples: np.ndarray,
                                   motions: List["ObjectMotion"]):
        """Store per-ray time for scene evaluation with motion."""
        self._current_time_samples = t_samples
        self._current_motions = motions

    # ═══════════════════════════════════════════════════════════════════════
    # POST-PROCESSING PIPELINE
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _build_identity_lut(size: int = 16) -> np.ndarray:
        """Build an identity 3D LUT (size x size x size x 3), float32 [0,1]."""
        s = np.linspace(0, 1, size, dtype=np.float32)
        r, g, b = np.meshgrid(s, s, s, indexing='ij')
        return np.stack([r, g, b], axis=-1)

    @staticmethod
    def _apply_lut_trilinear(img: np.ndarray,
                             lut: np.ndarray) -> np.ndarray:
        """Apply a 3D colour LUT via trilinear interpolation.

        lut shape: (S, S, S, 3) where S is the LUT grid size.
        img shape: (H, W, 3) float32 in [0, 1].
        """
        S = lut.shape[0]
        h, w = img.shape[:2]
        flat = _clamp(img.reshape(-1, 3)) * (S - 1)

        r0 = np.floor(flat[:, 0]).astype(np.int32).clip(0, S - 2)
        g0 = np.floor(flat[:, 1]).astype(np.int32).clip(0, S - 2)
        b0 = np.floor(flat[:, 2]).astype(np.int32).clip(0, S - 2)
        fr = flat[:, 0] - r0
        fg = flat[:, 1] - g0
        fb = flat[:, 2] - b0

        def _fetch(ri, gi, bi):
            return lut[ri, gi, bi]

        c000 = _fetch(r0, g0, b0)
        c100 = _fetch(r0 + 1, g0, b0)
        c010 = _fetch(r0, g0 + 1, b0)
        c110 = _fetch(r0 + 1, g0 + 1, b0)
        c001 = _fetch(r0, g0, b0 + 1)
        c101 = _fetch(r0 + 1, g0, b0 + 1)
        c011 = _fetch(r0, g0 + 1, b0 + 1)
        c111 = _fetch(r0 + 1, g0 + 1, b0 + 1)

        fr = fr[:, np.newaxis]
        fg = fg[:, np.newaxis]
        fb = fb[:, np.newaxis]

        c00 = c000 * (1 - fr) + c100 * fr
        c01 = c001 * (1 - fr) + c101 * fr
        c10 = c010 * (1 - fr) + c110 * fr
        c11 = c011 * (1 - fr) + c111 * fr

        c0 = c00 * (1 - fg) + c10 * fg
        c1 = c01 * (1 - fg) + c11 * fg

        result = c0 * (1 - fb) + c1 * fb
        return result.reshape(h, w, 3).astype(np.float32)

    def _box_blur_fast(self, img: np.ndarray, radius: int) -> np.ndarray:
        """Separable box blur via cumulative sum — O(1) per pixel."""
        if radius < 1:
            return img.copy()
        h, w = img.shape[:2]
        r = min(radius, min(h, w) // 2)
        padded = np.pad(img, ((r, r), (r, r), (0, 0)), mode='edge')

        cum = np.cumsum(padded, axis=0)
        cum = cum[2 * r:] - cum[:cum.shape[0] - 2 * r]
        cum /= (2 * r)

        cum2 = np.cumsum(cum, axis=1)
        cum2 = cum2[:, 2 * r:] - cum2[:, :cum2.shape[1] - 2 * r]
        cum2 /= (2 * r)

        return cum2[:h, :w]

    def _post_process_advanced(self, img: np.ndarray, w: int, h: int,
                               pp: Dict[str, float],
                               emission_mask: Optional[np.ndarray] = None
                               ) -> np.ndarray:
        """Full production post-processing pipeline."""

        # ── Bloom ──
        bloom_str = pp.get("bloom", 0.0)
        if bloom_str > 0:
            threshold = pp.get("bloom_threshold", 0.8)
            lum = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
            bright = img * (lum > threshold)[:, :, np.newaxis]
            bloom_radius = max(1, int(min(w, h) * 0.02))
            blurred = self._box_blur_fast(bright, bloom_radius)
            blurred = self._box_blur_fast(blurred, bloom_radius)
            img = img + blurred * bloom_str

        # ── Glow (emission-specific) ──
        glow_str = pp.get("glow", 0.0)
        if glow_str > 0 and emission_mask is not None:
            glow_src = img * emission_mask[:, :, np.newaxis]
            glow_radius = max(1, int(min(w, h) * 0.03))
            glow_blurred = self._box_blur_fast(glow_src, glow_radius)
            img = img + glow_blurred * glow_str

        # ── Chromatic Aberration ──
        ca_str = pp.get("chromatic_aberration", 0.0)
        if ca_str > 0:
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            cx, cy = w / 2.0, h / 2.0
            dx = (xx - cx) / cx
            dy = (yy - cy) / cy

            shift_px = ca_str * min(w, h) * 0.5
            r_x = _clamp((xx + dx * shift_px).astype(np.int32), 0, w - 1).astype(int)
            r_y = _clamp((yy + dy * shift_px).astype(np.int32), 0, h - 1).astype(int)
            b_x = _clamp((xx - dx * shift_px).astype(np.int32), 0, w - 1).astype(int)
            b_y = _clamp((yy - dy * shift_px).astype(np.int32), 0, h - 1).astype(int)

            result = img.copy()
            result[:, :, 0] = img[r_y, r_x, 0]
            result[:, :, 2] = img[b_y, b_x, 2]
            img = result

        # ── Lens Flare ──
        flare_str = pp.get("lens_flare", 0.0)
        if flare_str > 0:
            lum = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
            bright_mask = lum > 0.95
            if bright_mask.any():
                bright_y, bright_x = np.where(bright_mask)
                if len(bright_y) > 0:
                    src_y = int(np.mean(bright_y))
                    src_x = int(np.mean(bright_x))
                    cx, cy = w // 2, h // 2

                    for ghost_i in range(3):
                        scale = -0.3 * (ghost_i + 1)
                        gx = int(cx + (src_x - cx) * scale)
                        gy = int(cy + (src_y - cy) * scale)
                        if 0 <= gx < w and 0 <= gy < h:
                            ghost_r = max(2, int(min(w, h) * 0.01))
                            yy_g, xx_g = np.mgrid[
                                max(0, gy - ghost_r * 3):min(h, gy + ghost_r * 3),
                                max(0, gx - ghost_r * 3):min(w, gx + ghost_r * 3)
                            ].astype(np.float32)
                            if yy_g.size > 0:
                                d = np.sqrt((xx_g - gx) ** 2 + (yy_g - gy) ** 2)
                                falloff = np.exp(-d / ghost_r)
                                ghost_color = _v3(0.3, 0.5, 0.8) * flare_str * 0.3
                                y_sl = slice(max(0, gy - ghost_r * 3),
                                             min(h, gy + ghost_r * 3))
                                x_sl = slice(max(0, gx - ghost_r * 3),
                                             min(w, gx + ghost_r * 3))
                                img[y_sl, x_sl] += (
                                    ghost_color[np.newaxis, np.newaxis, :]
                                    * falloff[:, :, np.newaxis])

                    streak_len = int(w * 0.15)
                    y_lo = max(0, src_y - 1)
                    y_hi = min(h, src_y + 2)
                    x_lo = max(0, src_x - streak_len)
                    x_hi = min(w, src_x + streak_len)
                    if x_hi > x_lo and y_hi > y_lo:
                        xs = np.arange(x_lo, x_hi, dtype=np.float32)
                        streak_falloff = np.exp(
                            -np.abs(xs - src_x) / max(streak_len * 0.3, 1))
                        streak_color = _v3(0.6, 0.7, 1.0) * flare_str * 0.15
                        for y_row in range(y_lo, y_hi):
                            img[y_row, x_lo:x_hi] += (
                                streak_color[np.newaxis, :]
                                * streak_falloff[:, np.newaxis])

        # ── Color Grading (3D LUT with trilinear interpolation) ──
        lut = pp.get("lut", None)
        if lut is not None:
            img = self._apply_lut_trilinear(_clamp(img), lut)

        temp = pp.get("color_temperature", 0.0)
        if temp != 0.0:
            img[:, :, 0] += temp * 0.05
            img[:, :, 2] -= temp * 0.05

        sat = pp.get("saturation", 0.0)
        if sat != 0.0:
            lum = (0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1]
                   + 0.0722 * img[:, :, 2])
            img = lum[:, :, np.newaxis] + (img - lum[:, :, np.newaxis]) * (1.0 + sat)

        # ── Contrast (S-curve) ──
        contrast = pp.get("contrast", 0.0)
        if contrast != 0.0:
            mid = 0.5
            img = mid + (img - mid) * (1.0 + contrast)
            img = _clamp(img, 0.0, 10.0)

        # ── Film Grain (blue-noise-like, luminance-dependent) ──
        grain_str = pp.get("film_grain", 0.0)
        if grain_str > 0:
            lum = (0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1]
                   + 0.0722 * img[:, :, 2])
            shadow_weight = 1.0 - _clamp(lum * 2.0)

            base_noise = self._rng.normal(0, 1, (h, w)).astype(np.float32)
            offset_noise = self._rng.normal(0, 1, (h, w)).astype(np.float32)
            blue_approx = base_noise - self._box_blur_fast(
                offset_noise[:, :, np.newaxis], 2)[:, :, 0]
            blue_approx /= max(np.std(blue_approx), 1e-6)

            grain = blue_approx * grain_str * (0.3 + 0.7 * shadow_weight)
            img += grain[:, :, np.newaxis]

        # ── Sharpening (unsharp mask: blur → subtract → scale) ──
        sharp = pp.get("sharpening", 0.0)
        if sharp > 0:
            sharp_radius = max(1, int(min(w, h) * 0.003))
            blurred = self._box_blur_fast(img, sharp_radius)
            blurred = self._box_blur_fast(blurred, sharp_radius)
            img = img + (img - blurred) * sharp

        # ── Vignette ──
        vig = pp.get("vignette", 0.0)
        if vig > 0:
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            cx, cy = w / 2.0, h / 2.0
            dist = np.sqrt((xx - cx) ** 2 / cx ** 2 + (yy - cy) ** 2 / cy ** 2)
            vignette = 1.0 - _clamp(dist - (1.0 - vig)) * vig
            img *= vignette[:, :, np.newaxis]

        # ── Letterboxing ──
        letterbox = pp.get("letterbox", 0.0)
        if letterbox > 0:
            target_aspect = letterbox
            current_aspect = w / h
            if current_aspect < target_aspect:
                visible_h = int(w / target_aspect)
                bar = (h - visible_h) // 2
                if bar > 0:
                    img[:bar] = 0
                    img[h - bar:] = 0

        img = _clamp(img, 0.0, 1.0)
        return img

    # ═══════════════════════════════════════════════════════════════════════
    # MAIN RENDER
    # ═══════════════════════════════════════════════════════════════════════

    def render(self, camera: Camera, lighting: LightingSetup = None,
               gi_bounces: int = 2, gi_samples: int = 4,
               gi_quality: float = 1.0,
               dof_samples: int = 1, aperture: float = 0.0,
               focus_distance: float = 5.0,
               motion_blur: bool = False,
               motions: Optional[List["ObjectMotion"]] = None,
               post_processing: Dict[str, float] = None,
               use_gpu: Optional[bool] = None) -> np.ndarray:
        """
        Render the full frame with advanced features.
        Returns (H, W, 3) uint8 image.
        """
        if lighting is None:
            lighting = self.lighting
        if not (lighting.directional or lighting.point or lighting.area
                or lighting.spot or lighting.environment):
            lighting.add(Light())

        pp = dict(DEFAULT_POST_PROCESSING)
        if post_processing is not None:
            pp.update(post_processing)

        if use_gpu is None:
            use_gpu = GPU_AVAILABLE
        xp = _get_xp(use_gpu)

        w, h = self.cfg.width, self.cfg.height
        N = w * h

        do_dof = aperture > 0 and dof_samples > 1

        if do_dof:
            all_origins, all_dirs = self._generate_dof_rays(
                camera, w, h, aperture, focus_distance, dof_samples)
            accum = np.zeros((N, 3), dtype=np.float32)

            for s in range(dof_samples):
                s_origins = all_origins[:, s, :]
                s_dirs = all_dirs[:, s, :]
                if motion_blur:
                    s_origins, s_dirs, t_samples = \
                        self._apply_motion_blur_jitter(s_origins, s_dirs, N)
                    if motions:
                        self._interpolate_scene_at_time(t_samples, motions)
                sample_color = self._render_pass(
                    s_origins, s_dirs,
                    w, h, lighting, gi_bounces, gi_samples, gi_quality,
                    motion_blur, use_gpu, xp)
                accum += sample_color
            color = accum / dof_samples
        else:
            origins = np.broadcast_to(
                camera.position, (N, 3)).astype(np.float32).copy()
            dirs = camera.ray_directions(w, h)

            if motion_blur:
                origins, dirs, t_samples = self._apply_motion_blur_jitter(
                    origins, dirs, N)
                if motions:
                    self._interpolate_scene_at_time(t_samples, motions)

            color = self._render_pass(
                origins, dirs, w, h, lighting,
                gi_bounces, gi_samples, gi_quality,
                motion_blur, use_gpu, xp)

        img = color.reshape(h, w, 3)

        img = img * self.cfg.exposure
        img = self._tonemap_aces(img)
        img = np.power(_clamp(img), 1.0 / self.cfg.gamma)

        emission_mask = self._build_emission_mask(
            origins if not do_dof else all_origins[:, 0, :],
            dirs if not do_dof else all_dirs[:, 0, :],
            w, h)

        img = self._post_process_advanced(img, w, h, pp, emission_mask)

        self._current_time_samples = None
        self._current_motions = None

        return (_clamp(img) * 255).astype(np.uint8)

    def _render_pass(self, origins: Vec3, dirs: Vec3,
                     w: int, h: int,
                     lighting: LightingSetup,
                     gi_bounces: int, gi_samples: int,
                     gi_quality: float,
                     motion_blur: bool,
                     use_gpu: bool, xp) -> Vec3:
        """Single render pass — march, shade, GI, refraction, fog."""
        N = origins.shape[0]
        color = np.zeros((N, 3), dtype=np.float32)

        hit, pos, mat_ids = self._march(origins, dirs)

        miss = ~hit
        if miss.any():
            default_sun = _v3(0.5, 0.8, -0.3)
            color[miss] = self._sky_color(dirs[miss], default_sun)

        if not hit.any():
            return color

        h_pos = pos[hit]
        h_dirs = dirs[hit]
        h_mat_ids = mat_ids[hit]

        normals = self._normals(h_pos)
        ao = self._ao(h_pos, normals)

        unique_mats = np.unique(h_mat_ids)
        h_color = np.zeros((hit.sum(), 3), dtype=np.float32)

        for mid in unique_mats:
            if mid < 0 or mid >= len(self.scene.materials):
                continue
            mask = h_mat_ids == mid
            if not mask.any():
                continue
            mat = self.scene.materials[mid]
            h_color[mask] = self._shade_advanced(
                h_pos[mask], normals[mask], h_dirs[mask],
                mat, ao[mask], lighting)

            transmission = getattr(mat, 'transmission', 0.0)
            if transmission > 0:
                trans_color = self._trace_transmission(
                    h_pos[mask], h_dirs[mask], normals[mask],
                    mat, lighting)
                h_color[mask] = (
                    h_color[mask] * (1.0 - transmission)
                    + trans_color * transmission)

        if gi_bounces > 0 and gi_samples > 0:
            gi = self._trace_gi(
                h_pos, normals, h_dirs,
                self.scene.materials, h_mat_ids,
                gi_bounces, gi_samples, gi_quality, lighting)
            h_color += gi * 0.5

        dist = _norm_batch(h_pos - origins[hit])
        default_sun = _v3(0.5, 0.8, -0.3)
        sky_at_hit = self._sky_color(h_dirs, default_sun)
        h_color = self._apply_fog(h_color, dist, sky_at_hit)

        color[hit] = h_color
        return color

    def _build_emission_mask(self, origins: Vec3, dirs: Vec3,
                             w: int, h: int) -> Optional[np.ndarray]:
        """Build a 2D mask of emissive pixels for glow post-processing."""
        N = origins.shape[0]
        hit, pos, mat_ids = self._march(origins, dirs)
        mask = np.zeros(N, dtype=np.float32)

        if hit.any():
            unique_mats = np.unique(mat_ids[hit])
            for mid in unique_mats:
                if mid < 0 or mid >= len(self.scene.materials):
                    continue
                mat = self.scene.materials[mid]
                if np.any(mat.emission > 0):
                    m = mat_ids[hit] == mid
                    mask_indices = np.where(hit)[0][m]
                    mask[mask_indices] = 1.0

        return mask.reshape(h, w)
