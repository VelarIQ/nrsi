"""
NRSI GPU Renderer — PyTorch-accelerated SDF ray marching.

Pure GPU rendering path using torch tensors for all ray operations:
  - Batch ray marching with vectorised SDF evaluation
  - Parallel central-difference normal estimation
  - Cook-Torrance PBR shading on device
  - Post-processing (ACES tonemapping, gamma) entirely on GPU
  - Framebuffer as torch tensor, convertible to PIL Image / numpy

Falls back to CPU tensors when no GPU is detected.
"""
from __future__ import annotations

import math
from typing import Callable, List, Optional, Tuple

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    _TORCH_AVAILABLE = False

try:
    from .render_engine import (
        SDFScene, Camera, Material, Light, RenderConfig,
        Vec3, _v3, _clamp, _normalize_batch,
    )
except ImportError:
    from render_engine import (
        SDFScene, Camera, Material, Light, RenderConfig,
        Vec3, _v3, _clamp, _normalize_batch,
    )

__all__ = ["GPURenderer", "gpu_available", "get_torch_device"]

_F32 = np.float32


def _detect_device() -> str:
    if not _TORCH_AVAILABLE:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


_DEVICE: Optional[str] = None


def get_torch_device() -> str:
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = _detect_device()
    return _DEVICE


def gpu_available() -> bool:
    return get_torch_device() != "cpu"


def _to_torch(arr: np.ndarray, device: str) -> "torch.Tensor":
    return torch.from_numpy(np.ascontiguousarray(arr).astype(_F32)).to(device)


def _to_numpy(t: "torch.Tensor") -> np.ndarray:
    return t.detach().cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════════
# TORCH VECTOR HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def _tdot(a: "torch.Tensor", b: "torch.Tensor") -> "torch.Tensor":
    return (a * b).sum(dim=-1)


def _tnorm(v: "torch.Tensor") -> "torch.Tensor":
    return v.norm(dim=-1).clamp(min=1e-12)


def _tnormalize(v: "torch.Tensor") -> "torch.Tensor":
    return v / _tnorm(v).unsqueeze(-1).clamp(min=1e-9)


def _treflect(incident: "torch.Tensor", normal: "torch.Tensor") -> "torch.Tensor":
    d = _tdot(incident, normal)
    return incident - 2.0 * d.unsqueeze(-1) * normal


# ═══════════════════════════════════════════════════════════════════════════════
# GPU RENDERER
# ═══════════════════════════════════════════════════════════════════════════════


class GPURenderer:
    """SDF renderer that runs the full ray marching pipeline on GPU via PyTorch.

    SDF evaluation still calls the scene's numpy-based ``evaluate`` but the
    data transfer is batched: rays and framebuffer live as torch tensors on
    the target device.
    """

    def __init__(self, scene: SDFScene, config: RenderConfig = None,
                 device: Optional[str] = None):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for GPURenderer")
        self.scene = scene
        self.cfg = config or RenderConfig()
        self.device = device or get_torch_device()

    # ── SDF bridge (numpy scene evaluate → torch) ────────────────────────

    def _evaluate_sdf(self, p: "torch.Tensor"
                      ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        p_np = _to_numpy(p)
        d_np, m_np = self.scene.evaluate(p_np)
        d = torch.from_numpy(d_np.astype(_F32)).to(self.device)
        m = torch.from_numpy(m_np.astype(np.int32)).to(self.device)
        return d, m

    # ── ray marching ─────────────────────────────────────────────────────

    def _march(self, origins: "torch.Tensor", dirs: "torch.Tensor"
               ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        N = origins.shape[0]
        t = torch.full((N,), self.cfg.surface_eps,
                        dtype=torch.float32, device=self.device)
        hit = torch.zeros(N, dtype=torch.bool, device=self.device)
        active = torch.ones(N, dtype=torch.bool, device=self.device)
        mat_ids = torch.zeros(N, dtype=torch.int32, device=self.device)

        for _ in range(self.cfg.max_march_steps):
            if not active.any():
                break
            idx = active.nonzero(as_tuple=True)[0]
            pos = origins[idx] + dirs[idx] * t[idx].unsqueeze(-1)

            d, m = self._evaluate_sdf(pos)

            t[idx] += d
            converged = d < self.cfg.surface_eps
            conv_idx = idx[converged]
            hit[conv_idx] = True
            mat_ids[conv_idx] = m[converged]
            active[conv_idx] = False

            escaped = t[idx] > self.cfg.max_march_dist
            active[idx[escaped]] = False

        positions = origins + dirs * t.unsqueeze(-1)
        return hit, positions, mat_ids

    # ── normals ──────────────────────────────────────────────────────────

    def _normals(self, p: "torch.Tensor") -> "torch.Tensor":
        eps = self.cfg.normal_eps
        offsets = torch.tensor([
            [eps, 0, 0], [-eps, 0, 0],
            [0, eps, 0], [0, -eps, 0],
            [0, 0, eps], [0, 0, -eps],
        ], dtype=torch.float32, device=self.device)

        d_px, _ = self._evaluate_sdf(p + offsets[0])
        d_nx, _ = self._evaluate_sdf(p + offsets[1])
        d_py, _ = self._evaluate_sdf(p + offsets[2])
        d_ny, _ = self._evaluate_sdf(p + offsets[3])
        d_pz, _ = self._evaluate_sdf(p + offsets[4])
        d_nz, _ = self._evaluate_sdf(p + offsets[5])

        normals = torch.stack([d_px - d_nx, d_py - d_ny, d_pz - d_nz], dim=-1)
        return _tnormalize(normals)

    # ── soft shadows ─────────────────────────────────────────────────────

    def _soft_shadow(self, origins: "torch.Tensor",
                     light_dir: "torch.Tensor",
                     max_dist: float = 40.0) -> "torch.Tensor":
        if light_dir.dim() == 1:
            light_dir = light_dir.unsqueeze(0).expand_as(origins)

        N = origins.shape[0]
        shadow = torch.ones(N, dtype=torch.float32, device=self.device)
        t = torch.full((N,), self.cfg.surface_eps * 10,
                        dtype=torch.float32, device=self.device)
        active = torch.ones(N, dtype=torch.bool, device=self.device)

        for _ in range(self.cfg.shadow_steps):
            if not active.any():
                break
            idx = active.nonzero(as_tuple=True)[0]
            pos = origins[idx] + light_dir[idx] * t[idx].unsqueeze(-1)
            d, _ = self._evaluate_sdf(pos)

            shadow[idx] = torch.minimum(
                shadow[idx],
                self.cfg.shadow_softness * d / t[idx].clamp(min=0.001))
            t[idx] += d.clamp(min=self.cfg.surface_eps)

            occluded = d < self.cfg.surface_eps * 0.5
            active[idx[occluded]] = False
            escaped = t[idx] > max_dist
            active[idx[escaped]] = False

        return shadow.clamp(0.0, 1.0)

    # ── ambient occlusion ────────────────────────────────────────────────

    def _ao(self, p: "torch.Tensor", n: "torch.Tensor") -> "torch.Tensor":
        N = p.shape[0]
        occ = torch.zeros(N, dtype=torch.float32, device=self.device)
        scale = 1.0

        for i in range(1, self.cfg.ao_steps + 1):
            step = i * self.cfg.ao_step_size
            sample_p = p + n * step
            d, _ = self._evaluate_sdf(sample_p)
            occ += (step - d.clamp(min=0)) * scale
            scale *= 0.5

        return (1.0 - occ * self.cfg.ao_strength).clamp(0.0, 1.0)

    # ── sky model ────────────────────────────────────────────────────────

    def _sky_color(self, dirs: "torch.Tensor",
                   sun_dir: "torch.Tensor") -> "torch.Tensor":
        y = dirs[:, 1]
        t = (y * 0.5 + 0.5).clamp(0.0, 1.0)

        zenith = _to_torch(self.cfg.sky_zenith, self.device)
        horizon = _to_torch(self.cfg.sky_horizon, self.device)

        sky = horizon.unsqueeze(0) * (1.0 - t.unsqueeze(-1)) + \
              zenith.unsqueeze(0) * t.unsqueeze(-1)

        sun_dot = _tdot(dirs, sun_dir.unsqueeze(0).expand_as(dirs)).clamp(0.0, 1.0)
        sun_glow = sun_dot.pow(32.0) * 0.8
        sun_color = torch.tensor([1.0, 0.9, 0.7],
                                  dtype=torch.float32, device=self.device)
        sky = sky + sun_color.unsqueeze(0) * sun_glow.unsqueeze(-1)
        return sky

    # ── shading ──────────────────────────────────────────────────────────

    def _shade(self, p: "torch.Tensor", n: "torch.Tensor",
               view_dir: "torch.Tensor", mat: Material,
               light: Light, ao: "torch.Tensor",
               shadow: "torch.Tensor") -> "torch.Tensor":
        N = p.shape[0]
        albedo = _to_torch(mat.albedo, self.device).unsqueeze(0).expand(N, 3)
        roughness = max(mat.roughness, 0.04)
        a2 = roughness ** 4

        l_dir = _to_torch(light.direction, self.device).unsqueeze(0).expand(N, 3)
        v = -view_dir

        ndl = _tdot(n, l_dir).clamp(0.0, 1.0)
        ndv = _tdot(n, v).clamp(0.001, 1.0)

        h_vec = _tnormalize(l_dir + v)
        ndh = _tdot(n, h_vec).clamp(0.0, 1.0)
        vdh = _tdot(v, h_vec).clamp(0.0, 1.0)

        denom = ndh * ndh * (a2 - 1.0) + 1.0
        D = a2 / (math.pi * denom * denom + 1e-7)

        F0 = 0.04 * (1.0 - mat.metallic) + mat.metallic
        F = F0 + (1.0 - F0) * (1.0 - vdh).pow(5.0)

        k = (roughness + 1.0) ** 2 / 8.0
        G = (ndv / (ndv * (1.0 - k) + k)) * (ndl / (ndl * (1.0 - k) + k))

        spec = (D * F * G) / (4.0 * ndv * ndl + 0.001)
        kd = (1.0 - F) * (1.0 - mat.metallic)

        l_color = _to_torch(light.color, self.device)
        radiance = l_color * light.intensity
        lit = radiance.unsqueeze(0) * ndl.unsqueeze(-1) * shadow.unsqueeze(-1)

        diffuse = kd.unsqueeze(-1) * albedo / math.pi
        color = (diffuse + spec.unsqueeze(-1)) * lit

        ambient = _to_torch(self.cfg.sky_horizon * 0.15, self.device)
        color = color + albedo * ambient.unsqueeze(0) * ao.unsqueeze(-1)

        emission = _to_torch(mat.emission, self.device)
        if emission.sum() > 0:
            color = color + emission.unsqueeze(0).expand(N, 3)

        return color

    # ── fog ───────────────────────────────────────────────────────────────

    def _apply_fog(self, color: "torch.Tensor", dist: "torch.Tensor",
                   sky_color: "torch.Tensor") -> "torch.Tensor":
        fog_factor = 1.0 - torch.exp(
            -self.cfg.fog_density * (dist - self.cfg.fog_start).clamp(min=0))
        fog_factor = fog_factor.clamp(0.0, 1.0)
        return color * (1.0 - fog_factor.unsqueeze(-1)) + \
               sky_color * fog_factor.unsqueeze(-1)

    # ── ACES tonemapping ─────────────────────────────────────────────────

    @staticmethod
    def _tonemap_aces(x: "torch.Tensor") -> "torch.Tensor":
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        return ((x * (a * x + b)) / (x * (c * x + d) + e)).clamp(0.0, 1.0)

    # ── gamma correction ─────────────────────────────────────────────────

    def _gamma_correct(self, x: "torch.Tensor") -> "torch.Tensor":
        return x.clamp(0.0, 1.0).pow(1.0 / self.cfg.gamma)

    # ── main render ──────────────────────────────────────────────────────

    def render(self, camera: Camera, lighting: Light = None) -> np.ndarray:
        """Render via GPU ray march. Returns (H, W, 3) uint8 numpy array."""
        if lighting is None:
            lighting = Light()

        w, h = self.cfg.width, self.cfg.height
        N = w * h

        cam_pos = _to_torch(camera.position, self.device)
        dirs_np = camera.ray_directions(w, h)
        dirs = _to_torch(dirs_np, self.device)
        origins = cam_pos.unsqueeze(0).expand(N, 3).contiguous()

        hit, pos, mat_ids = self._march(origins, dirs)

        color = torch.zeros(N, 3, dtype=torch.float32, device=self.device)

        miss = ~hit
        if miss.any():
            sun_dir = _to_torch(
                _normalize_batch(lighting.direction), self.device)
            color[miss] = self._sky_color(dirs[miss], sun_dir)

        if hit.any():
            h_pos = pos[hit]
            h_dirs = dirs[hit]
            h_mat_ids = mat_ids[hit]

            normals = self._normals(h_pos)
            ao = self._ao(h_pos, normals)

            shadow_origin = h_pos + normals * self.cfg.surface_eps * 3.0
            sun_dir_t = _to_torch(lighting.direction, self.device)
            shadow = self._soft_shadow(shadow_origin, sun_dir_t)

            unique_mats = h_mat_ids.unique()
            h_color = torch.zeros(hit.sum(), 3,
                                   dtype=torch.float32, device=self.device)

            for mid_t in unique_mats:
                mid = int(mid_t.item())
                if mid < 0 or mid >= len(self.scene.materials):
                    continue
                mask = h_mat_ids == mid_t
                if not mask.any():
                    continue
                mat = self.scene.materials[mid]
                h_color[mask] = self._shade(
                    h_pos[mask], normals[mask], h_dirs[mask],
                    mat, lighting, ao[mask], shadow[mask])

            dist = _tnorm(h_pos - origins[hit])
            sun_dir = _to_torch(
                _normalize_batch(lighting.direction), self.device)
            sky_at_hit = self._sky_color(h_dirs, sun_dir)
            h_color = self._apply_fog(h_color, dist, sky_at_hit)

            color[hit] = h_color

        color = color.reshape(h, w, 3)
        color = color * self.cfg.exposure
        color = self._tonemap_aces(color)
        color = self._gamma_correct(color)

        return (_to_numpy(color.clamp(0.0, 1.0)) * 255).astype(np.uint8)

    def render_to_pil(self, camera: Camera,
                      lighting: Light = None) -> "Image.Image":
        """Render and return a PIL Image."""
        from PIL import Image
        arr = self.render(camera, lighting)
        return Image.fromarray(arr, mode="RGB")

    def render_to_tensor(self, camera: Camera,
                         lighting: Light = None) -> "torch.Tensor":
        """Render and return the framebuffer as a (H, W, 3) float32 torch tensor."""
        if lighting is None:
            lighting = Light()

        w, h = self.cfg.width, self.cfg.height
        N = w * h

        cam_pos = _to_torch(camera.position, self.device)
        dirs_np = camera.ray_directions(w, h)
        dirs = _to_torch(dirs_np, self.device)
        origins = cam_pos.unsqueeze(0).expand(N, 3).contiguous()

        hit_mask, pos, mat_ids = self._march(origins, dirs)

        color = torch.zeros(N, 3, dtype=torch.float32, device=self.device)

        miss = ~hit_mask
        if miss.any():
            sun_dir = _to_torch(
                _normalize_batch(lighting.direction), self.device)
            color[miss] = self._sky_color(dirs[miss], sun_dir)

        if hit_mask.any():
            h_pos = pos[hit_mask]
            h_dirs = dirs[hit_mask]
            h_mat_ids = mat_ids[hit_mask]
            normals = self._normals(h_pos)
            ao = self._ao(h_pos, normals)
            shadow_origin = h_pos + normals * self.cfg.surface_eps * 3.0
            sun_dir_t = _to_torch(lighting.direction, self.device)
            shadow = self._soft_shadow(shadow_origin, sun_dir_t)

            h_color = torch.zeros(hit_mask.sum(), 3,
                                   dtype=torch.float32, device=self.device)
            for mid_t in h_mat_ids.unique():
                mid = int(mid_t.item())
                if mid < 0 or mid >= len(self.scene.materials):
                    continue
                mask = h_mat_ids == mid_t
                if not mask.any():
                    continue
                mat = self.scene.materials[mid]
                h_color[mask] = self._shade(
                    h_pos[mask], normals[mask], h_dirs[mask],
                    mat, lighting, ao[mask], shadow[mask])

            dist = _tnorm(h_pos - origins[hit_mask])
            sun_dir = _to_torch(
                _normalize_batch(lighting.direction), self.device)
            sky_at_hit = self._sky_color(h_dirs, sun_dir)
            h_color = self._apply_fog(h_color, dist, sky_at_hit)
            color[hit_mask] = h_color

        color = color.reshape(h, w, 3)
        color = color * self.cfg.exposure
        color = self._tonemap_aces(color)
        color = self._gamma_correct(color)
        return color.clamp(0.0, 1.0)
