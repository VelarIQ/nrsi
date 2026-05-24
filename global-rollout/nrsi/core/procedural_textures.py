"""
NRSI Procedural Texture Engine — noise-driven PBR material synthesis.

All noise functions operate on (N,3) float32 position arrays and return
(N,) or (N,3) results with no Python-level per-pixel loops.  Gradient
noise uses deterministic hash-based lattice gradients for reproducibility.

Integration: ``sample_material`` bridges ``ProceduralMaterial`` to the
render_engine ``Material`` dataclass by evaluating position-dependent
PBR channels over arbitrary hit-point arrays.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, Tuple, Dict

__all__ = [
    # noise
    "perlin_3d",
    "simplex_3d",
    "fbm",
    "turbulence",
    "domain_warp",
    "voronoi_3d",
    "ridged_noise",
    # material
    "ProceduralMaterial",
    "sample_material",
    # weathering
    "apply_weathering",
    "apply_dust",
    "apply_rust",
]

# ═══════════════════════════════════════════════════════════════════════════
# HASH UTILITIES — deterministic, seed-aware, vectorised
# ═══════════════════════════════════════════════════════════════════════════

_PRIME1 = np.int32(73856093)
_PRIME2 = np.int32(19349663)
_PRIME3 = np.int32(83492791)


def _hash3(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray,
           seed: int = 0) -> np.ndarray:
    """Deterministic integer hash of 3-int coords → uint32 array."""
    s = np.int32(seed * 1013)
    h = (ix.astype(np.int32) * _PRIME1) ^ (iy.astype(np.int32) * _PRIME2) ^ (iz.astype(np.int32) * _PRIME3) ^ s
    h = ((h >> np.int32(13)) ^ h).astype(np.int32)
    h = (h * (h * h * np.int32(15731) + np.int32(789221)) + np.int32(1376312589)).astype(np.int32)
    return h.view(np.uint32)


def _grad3(h: np.ndarray) -> np.ndarray:
    """Map hash → unit gradient vector on (N,3) from 12-direction table."""
    _TABLE = np.array([
        [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
        [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
        [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
    ], dtype=np.float32)
    return _TABLE[h % 12]


def _quintic(t: np.ndarray) -> np.ndarray:
    """Quintic smooth-step: 6t^5 - 15t^4 + 10t^3."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _lerp(a, b, t):
    return a + t * (b - a)


# ═══════════════════════════════════════════════════════════════════════════
# PERLIN 3D — classic gradient noise on integer lattice
# ═══════════════════════════════════════════════════════════════════════════

def perlin_3d(p: np.ndarray, seed: int = 0) -> np.ndarray:
    """3-D Perlin gradient noise.

    Parameters
    ----------
    p : (N, 3) float32
    seed : int

    Returns
    -------
    (N,) float in approximately [-1, 1].
    """
    p = np.asarray(p, dtype=np.float32)
    squeeze = p.ndim == 1
    if squeeze:
        p = p[np.newaxis]

    xi = np.floor(p[:, 0]).astype(np.int32)
    yi = np.floor(p[:, 1]).astype(np.int32)
    zi = np.floor(p[:, 2]).astype(np.int32)

    xf = p[:, 0] - xi.astype(np.float32)
    yf = p[:, 1] - yi.astype(np.float32)
    zf = p[:, 2] - zi.astype(np.float32)

    u = _quintic(xf)
    v = _quintic(yf)
    w = _quintic(zf)

    def _corner(dx, dy, dz):
        h = _hash3(xi + dx, yi + dy, zi + dz, seed)
        g = _grad3(h)
        d = np.stack([xf - dx, yf - dy, zf - dz], axis=-1)
        return np.einsum("ij,ij->i", g, d)

    c000 = _corner(0, 0, 0)
    c100 = _corner(1, 0, 0)
    c010 = _corner(0, 1, 0)
    c110 = _corner(1, 1, 0)
    c001 = _corner(0, 0, 1)
    c101 = _corner(1, 0, 1)
    c011 = _corner(0, 1, 1)
    c111 = _corner(1, 1, 1)

    x0 = _lerp(c000, c100, u)
    x1 = _lerp(c010, c110, u)
    x2 = _lerp(c001, c101, u)
    x3 = _lerp(c011, c111, u)

    y0 = _lerp(x0, x1, v)
    y1 = _lerp(x2, x3, v)

    result = _lerp(y0, y1, w)
    return result[0] if squeeze else result


# ═══════════════════════════════════════════════════════════════════════════
# SIMPLEX 3D — skew/unskew with 4-corner simplex traversal
# ═══════════════════════════════════════════════════════════════════════════

_F3 = 1.0 / 3.0
_G3 = 1.0 / 6.0

_SIMPLEX_GRAD3 = np.array([
    [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
    [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
    [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
], dtype=np.float32)


def simplex_3d(p: np.ndarray, seed: int = 0) -> np.ndarray:
    """3-D Simplex gradient noise (Perlin-improved variant).

    Parameters
    ----------
    p : (N, 3) float32
    seed : int

    Returns
    -------
    (N,) float in approximately [-1, 1].
    """
    p = np.asarray(p, dtype=np.float32)
    squeeze = p.ndim == 1
    if squeeze:
        p = p[np.newaxis]

    N = p.shape[0]
    s = (p[:, 0] + p[:, 1] + p[:, 2]) * _F3
    i = np.floor(p[:, 0] + s).astype(np.int32)
    j = np.floor(p[:, 1] + s).astype(np.int32)
    k = np.floor(p[:, 2] + s).astype(np.int32)

    t = (i + j + k).astype(np.float32) * _G3
    x0 = p[:, 0] - (i.astype(np.float32) - t)
    y0 = p[:, 1] - (j.astype(np.float32) - t)
    z0 = p[:, 2] - (k.astype(np.float32) - t)

    i1 = np.zeros(N, dtype=np.int32)
    j1 = np.zeros(N, dtype=np.int32)
    k1 = np.zeros(N, dtype=np.int32)
    i2 = np.zeros(N, dtype=np.int32)
    j2 = np.zeros(N, dtype=np.int32)
    k2 = np.zeros(N, dtype=np.int32)

    ge_xy = (x0 >= y0)
    ge_xz = (x0 >= z0)
    ge_yz = (y0 >= z0)

    # x >= y >= z
    m = ge_xy & ge_xz & ge_yz
    i1[m] = 1; j2[m] = 1; i2[m] = 1
    # x >= z > y
    m = ge_xy & ge_xz & ~ge_yz
    i1[m] = 1; k2[m] = 1; i2[m] = 1
    # z > x >= y
    m = ge_xy & ~ge_xz
    k1[m] = 1; i2[m] = 1; k2[m] = 1
    # y > x, y >= z, x >= z
    m = ~ge_xy & ge_yz & ge_xz
    j1[m] = 1; i2[m] = 1; j2[m] = 1
    # y > x, y >= z, z > x
    m = ~ge_xy & ge_yz & ~ge_xz
    j1[m] = 1; k2[m] = 1; j2[m] = 1
    # z > y > x
    m = ~ge_xy & ~ge_yz
    k1[m] = 1; k2[m] = 1; j2[m] = 1

    x1 = x0 - i1.astype(np.float32) + _G3
    y1 = y0 - j1.astype(np.float32) + _G3
    z1 = z0 - k1.astype(np.float32) + _G3
    x2 = x0 - i2.astype(np.float32) + 2.0 * _G3
    y2 = y0 - j2.astype(np.float32) + 2.0 * _G3
    z2 = z0 - k2.astype(np.float32) + 2.0 * _G3
    x3 = x0 - 1.0 + 3.0 * _G3
    y3 = y0 - 1.0 + 3.0 * _G3
    z3 = z0 - 1.0 + 3.0 * _G3

    def _contrib(ix, iy, iz, dx, dy, dz):
        h = _hash3(ix, iy, iz, seed)
        g = _SIMPLEX_GRAD3[h % 12]
        t_val = 0.6 - dx * dx - dy * dy - dz * dz
        pos = t_val > 0
        out = np.zeros(N, dtype=np.float32)
        t_pos = t_val[pos]
        t2 = t_pos * t_pos
        out[pos] = t2 * t2 * (g[pos, 0] * dx[pos] + g[pos, 1] * dy[pos] + g[pos, 2] * dz[pos])
        return out

    n0 = _contrib(i, j, k, x0, y0, z0)
    n1 = _contrib(i + i1, j + j1, k + k1, x1, y1, z1)
    n2 = _contrib(i + i2, j + j2, k + k2, x2, y2, z2)
    n3 = _contrib(i + 1, j + 1, k + 1, x3, y3, z3)

    result = 32.0 * (n0 + n1 + n2 + n3)
    return result[0] if squeeze else result


# ═══════════════════════════════════════════════════════════════════════════
# COMPOSITE NOISE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def fbm(p: np.ndarray, octaves: int = 6, lacunarity: float = 2.0,
        gain: float = 0.5, noise_fn: Optional[Callable] = None,
        seed: int = 0) -> np.ndarray:
    """Fractal Brownian motion — layered noise with decreasing amplitude."""
    if noise_fn is None:
        noise_fn = perlin_3d
    p = np.asarray(p, dtype=np.float32).copy()
    squeeze = p.ndim == 1
    if squeeze:
        p = p[np.newaxis]

    total = np.zeros(p.shape[0], dtype=np.float32)
    amplitude = 1.0
    freq = 1.0
    for i in range(octaves):
        total += amplitude * noise_fn(p * freq, seed=seed + i)
        freq *= lacunarity
        amplitude *= gain
    return total[0] if squeeze else total


def turbulence(p: np.ndarray, octaves: int = 6, lacunarity: float = 2.0,
               gain: float = 0.5, seed: int = 0) -> np.ndarray:
    """Absolute-value turbulence — sum of |noise| per octave."""
    p = np.asarray(p, dtype=np.float32).copy()
    squeeze = p.ndim == 1
    if squeeze:
        p = p[np.newaxis]

    total = np.zeros(p.shape[0], dtype=np.float32)
    amplitude = 1.0
    freq = 1.0
    for i in range(octaves):
        total += amplitude * np.abs(perlin_3d(p * freq, seed=seed + i))
        freq *= lacunarity
        amplitude *= gain
    return total[0] if squeeze else total


def ridged_noise(p: np.ndarray, octaves: int = 6, lacunarity: float = 2.0,
                 gain: float = 0.5, seed: int = 0) -> np.ndarray:
    """Ridged multifractal — inverted absolute noise with signal feedback."""
    p = np.asarray(p, dtype=np.float32).copy()
    squeeze = p.ndim == 1
    if squeeze:
        p = p[np.newaxis]

    total = np.zeros(p.shape[0], dtype=np.float32)
    amplitude = 1.0
    freq = 1.0
    weight = 1.0
    for i in range(octaves):
        signal = 1.0 - np.abs(perlin_3d(p * freq, seed=seed + i))
        signal = signal * signal * weight
        weight = np.clip(signal, 0.0, 1.0) if np.isscalar(signal) else np.clip(signal, 0.0, 1.0)
        total += amplitude * signal
        freq *= lacunarity
        amplitude *= gain
    return total[0] if squeeze else total


def domain_warp(p: np.ndarray, strength: float = 0.5, octaves: int = 3,
                seed: int = 0) -> np.ndarray:
    """Domain warping — displaces input coords via noise for organic distortion.

    Returns warped positions (N,3) suitable for feeding into another noise call.
    """
    p = np.asarray(p, dtype=np.float32).copy()
    squeeze = p.ndim == 1
    if squeeze:
        p = p[np.newaxis]

    dx = fbm(p, octaves=octaves, seed=seed)
    dy = fbm(p + 5.2, octaves=octaves, seed=seed + 1)
    dz = fbm(p + 9.1, octaves=octaves, seed=seed + 2)
    warped = p + strength * np.stack([dx, dy, dz], axis=-1)
    return warped[0] if squeeze else warped


def voronoi_3d(p: np.ndarray, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Voronoi / Worley noise.

    Returns
    -------
    (dist, cell_id) : ((N,), (N,))
        dist — distance to nearest cell centre
        cell_id — deterministic integer id of the owning cell
    """
    p = np.asarray(p, dtype=np.float32)
    squeeze = p.ndim == 1
    if squeeze:
        p = p[np.newaxis]

    N = p.shape[0]
    base = np.floor(p).astype(np.int32)

    best_dist = np.full(N, 1e30, dtype=np.float32)
    best_cell = np.zeros(N, dtype=np.int32)

    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                ci = base[:, 0] + di
                cj = base[:, 1] + dj
                ck = base[:, 2] + dk
                h = _hash3(ci, cj, ck, seed)
                fx = (h & 0xFF).astype(np.float32) / 255.0
                h2 = _hash3(ci, cj, ck, seed + 7)
                fy = (h2 & 0xFF).astype(np.float32) / 255.0
                h3 = _hash3(ci, cj, ck, seed + 13)
                fz = (h3 & 0xFF).astype(np.float32) / 255.0

                cell_pos = np.stack([
                    ci.astype(np.float32) + fx,
                    cj.astype(np.float32) + fy,
                    ck.astype(np.float32) + fz,
                ], axis=-1)

                diff = p - cell_pos
                dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
                closer = dist < best_dist
                best_dist = np.where(closer, dist, best_dist)
                best_cell = np.where(closer, h.astype(np.int32), best_cell)

    if squeeze:
        return best_dist[0], best_cell[0]
    return best_dist, best_cell


# ═══════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _clamp01(x):
    return np.clip(x, 0.0, 1.0)


def _mix_color(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Lerp between two (N,3) colour arrays using (N,) blend factor."""
    t = t[..., np.newaxis] if t.ndim == 1 else t
    return a * (1.0 - t) + b * t


def _normal_from_noise(p: np.ndarray, noise_fn: Callable, eps: float = 0.001,
                       seed: int = 0) -> np.ndarray:
    """Central-difference normal perturbation from a scalar noise field."""
    ex = np.zeros_like(p); ex[:, 0] = eps
    ey = np.zeros_like(p); ey[:, 1] = eps
    ez = np.zeros_like(p); ez[:, 2] = eps
    dx = noise_fn(p + ex, seed=seed) - noise_fn(p - ex, seed=seed)
    dy = noise_fn(p + ey, seed=seed) - noise_fn(p - ey, seed=seed)
    dz = noise_fn(p + ez, seed=seed) - noise_fn(p - ez, seed=seed)
    grad = np.stack([dx, dy, dz], axis=-1)
    norms = np.sqrt(np.einsum("ij,ij->i", grad, grad)).clip(1e-9)
    return grad / norms[:, np.newaxis]


def _ensure_n3(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float32)
    if p.ndim == 1:
        p = p[np.newaxis]
    return p


# ═══════════════════════════════════════════════════════════════════════════
# PBR MATERIAL OUTPUT
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PBRResult:
    """Position-dependent PBR channels."""
    albedo: np.ndarray           # (N, 3)
    normal_perturbation: np.ndarray  # (N, 3)
    roughness: np.ndarray        # (N,)
    metallic: np.ndarray         # (N,)


# ═══════════════════════════════════════════════════════════════════════════
# PROCEDURAL MATERIAL
# ═══════════════════════════════════════════════════════════════════════════

class ProceduralMaterial:
    """Evaluable procedural material that produces PBR channels from positions.

    Instantiate via factory class-methods (``ProceduralMaterial.wood(...)``),
    then call the instance with an (N,3) position array to get a ``PBRResult``.
    """

    def __init__(self, eval_fn: Callable[[np.ndarray], PBRResult], name: str = "custom"):
        self._eval = eval_fn
        self.name = name

    def __call__(self, p: np.ndarray) -> PBRResult:
        return self._eval(_ensure_n3(p))

    def __repr__(self):
        return f"ProceduralMaterial({self.name!r})"

    # ── wood ──────────────────────────────────────────────────────────────

    @classmethod
    def wood(cls, species: str = "oak") -> "ProceduralMaterial":
        palette = {
            "oak":    (np.array([0.42, 0.26, 0.12], np.float32),
                       np.array([0.65, 0.43, 0.22], np.float32)),
            "walnut": (np.array([0.22, 0.13, 0.06], np.float32),
                       np.array([0.40, 0.25, 0.12], np.float32)),
            "pine":   (np.array([0.55, 0.40, 0.20], np.float32),
                       np.array([0.75, 0.60, 0.35], np.float32)),
            "cherry": (np.array([0.40, 0.15, 0.08], np.float32),
                       np.array([0.60, 0.30, 0.15], np.float32)),
        }
        dark, light = palette.get(species, palette["oak"])

        def _eval(p: np.ndarray) -> PBRResult:
            N = p.shape[0]
            r = np.sqrt(p[:, 0] ** 2 + p[:, 2] ** 2)
            rings = np.sin(r * 20.0 + fbm(p * 2.0, octaves=3, seed=42) * 4.0) * 0.5 + 0.5

            knot_noise = _clamp01(1.0 - turbulence(p * 0.5, octaves=3, seed=99))
            grain = _clamp01(rings * knot_noise)

            albedo = _mix_color(
                np.broadcast_to(dark, (N, 3)).copy(),
                np.broadcast_to(light, (N, 3)).copy(),
                grain,
            )
            roughness = np.full(N, 0.55, np.float32) + grain * 0.15
            metallic = np.full(N, 0.0, np.float32)
            normal = _normal_from_noise(p, lambda q, seed=0: perlin_3d(q * 8.0, seed=77))
            return PBRResult(albedo, normal * 0.3, roughness, metallic)

        return cls(_eval, f"wood/{species}")

    # ── marble ────────────────────────────────────────────────────────────

    @classmethod
    def marble(cls, color_a: Optional[np.ndarray] = None,
               color_b: Optional[np.ndarray] = None,
               vein_scale: float = 5.0) -> "ProceduralMaterial":
        ca = np.array(color_a if color_a is not None else [0.95, 0.93, 0.88], np.float32)
        cb = np.array(color_b if color_b is not None else [0.15, 0.12, 0.10], np.float32)

        def _eval(p: np.ndarray) -> PBRResult:
            N = p.shape[0]
            vein = np.sin(p[:, 0] * vein_scale + turbulence(p, octaves=5, seed=31) * 6.0) * 0.5 + 0.5
            albedo = _mix_color(
                np.broadcast_to(ca, (N, 3)).copy(),
                np.broadcast_to(cb, (N, 3)).copy(),
                vein,
            )
            roughness = np.full(N, 0.15, np.float32) + vein * 0.1
            metallic = np.full(N, 0.0, np.float32)
            normal = _normal_from_noise(p, lambda q, seed=0: perlin_3d(q * vein_scale, seed=31))
            return PBRResult(albedo, normal * 0.1, roughness, metallic)

        return cls(_eval, "marble")

    # ── stone ─────────────────────────────────────────────────────────────

    @classmethod
    def stone(cls, stone_type: str = "granite") -> "ProceduralMaterial":
        configs = {
            "granite": {
                "colors": [
                    np.array([0.55, 0.52, 0.50], np.float32),
                    np.array([0.35, 0.33, 0.30], np.float32),
                    np.array([0.70, 0.65, 0.60], np.float32),
                ],
                "roughness_base": 0.65,
                "scale": 8.0,
            },
            "sandstone": {
                "colors": [
                    np.array([0.76, 0.60, 0.42], np.float32),
                    np.array([0.60, 0.45, 0.30], np.float32),
                    np.array([0.85, 0.72, 0.55], np.float32),
                ],
                "roughness_base": 0.80,
                "scale": 4.0,
            },
            "slate": {
                "colors": [
                    np.array([0.25, 0.27, 0.30], np.float32),
                    np.array([0.18, 0.19, 0.22], np.float32),
                    np.array([0.35, 0.36, 0.38], np.float32),
                ],
                "roughness_base": 0.50,
                "scale": 6.0,
            },
            "cobblestone": {
                "colors": [
                    np.array([0.45, 0.42, 0.38], np.float32),
                    np.array([0.30, 0.28, 0.25], np.float32),
                    np.array([0.55, 0.50, 0.45], np.float32),
                ],
                "roughness_base": 0.85,
                "scale": 3.0,
            },
        }
        cfg = configs.get(stone_type, configs["granite"])

        def _eval(p: np.ndarray) -> PBRResult:
            N = p.shape[0]
            n1 = _clamp01(fbm(p * cfg["scale"], octaves=5, seed=10) * 0.5 + 0.5)
            n2 = _clamp01(fbm(p * cfg["scale"] * 2.0, octaves=3, seed=20) * 0.5 + 0.5)

            c0, c1, c2 = cfg["colors"]
            albedo = _mix_color(
                np.broadcast_to(c0, (N, 3)).copy(),
                np.broadcast_to(c1, (N, 3)).copy(),
                n1,
            )
            albedo = _mix_color(albedo, np.broadcast_to(c2, (N, 3)).copy(), n2 * 0.4)

            if stone_type == "cobblestone":
                vd, _ = voronoi_3d(p * 3.0, seed=55)
                edge = _clamp01(1.0 - vd * 4.0)
                albedo = albedo * (0.7 + 0.3 * edge[:, np.newaxis])

            roughness = np.full(N, cfg["roughness_base"], np.float32) + n1 * 0.1
            metallic = np.full(N, 0.0, np.float32)
            normal = _normal_from_noise(p, lambda q, seed=0: fbm(q * cfg["scale"], octaves=4, seed=10))
            return PBRResult(albedo, normal * 0.25, roughness, metallic)

        return cls(_eval, f"stone/{stone_type}")

    # ── metal ─────────────────────────────────────────────────────────────

    @classmethod
    def metal(cls, metal_type: str = "steel") -> "ProceduralMaterial":
        configs = {
            "steel":  {"color": np.array([0.55, 0.56, 0.58], np.float32), "rough": 0.30, "met": 0.95},
            "copper": {"color": np.array([0.72, 0.45, 0.20], np.float32), "rough": 0.35, "met": 0.90},
            "gold":   {"color": np.array([1.00, 0.76, 0.34], np.float32), "rough": 0.20, "met": 1.00},
            "rust":   {"color": np.array([0.45, 0.22, 0.08], np.float32), "rough": 0.85, "met": 0.30},
        }
        cfg = configs.get(metal_type, configs["steel"])

        def _eval(p: np.ndarray) -> PBRResult:
            N = p.shape[0]
            base = np.broadcast_to(cfg["color"], (N, 3)).copy()

            scratch = perlin_3d(p * 40.0, seed=60)
            brush = _clamp01(np.abs(scratch) * 2.0)
            albedo = base * (0.9 + 0.1 * brush[:, np.newaxis])

            if metal_type == "copper":
                patina_mask = _clamp01(fbm(p * 3.0, octaves=4, seed=70) * 0.5 + 0.3)
                patina_color = np.array([0.25, 0.55, 0.45], np.float32)
                albedo = _mix_color(albedo, np.broadcast_to(patina_color, (N, 3)).copy(), patina_mask)

            roughness = np.full(N, cfg["rough"], np.float32) + brush * 0.1
            metallic = np.full(N, cfg["met"], np.float32)
            normal = _normal_from_noise(p, lambda q, seed=0: perlin_3d(q * 40.0, seed=60))
            return PBRResult(albedo, normal * 0.15, roughness, metallic)

        return cls(_eval, f"metal/{metal_type}")

    # ── fabric ────────────────────────────────────────────────────────────

    @classmethod
    def fabric(cls, weave: str = "plain") -> "ProceduralMaterial":
        def _eval(p: np.ndarray) -> PBRResult:
            N = p.shape[0]
            scale = 30.0
            warp = np.sin(p[:, 0] * scale) * 0.5 + 0.5
            weft = np.sin(p[:, 2] * scale) * 0.5 + 0.5

            if weave == "twill":
                warp = np.sin((p[:, 0] + p[:, 2] * 0.5) * scale) * 0.5 + 0.5

            thread = _clamp01(warp * 0.5 + weft * 0.5 + perlin_3d(p * 50.0, seed=80) * 0.15)
            base_color = np.array([0.20, 0.22, 0.35], np.float32)
            albedo = np.broadcast_to(base_color, (N, 3)).copy() * (0.7 + 0.3 * thread[:, np.newaxis])

            fuzz = np.abs(perlin_3d(p * 100.0, seed=81)) * 0.3
            roughness = np.full(N, 0.85, np.float32) + fuzz
            metallic = np.full(N, 0.0, np.float32)
            normal = _normal_from_noise(p, lambda q, seed=0: perlin_3d(q * scale, seed=80))
            return PBRResult(albedo, normal * 0.2, _clamp01(roughness), metallic)

        return cls(_eval, f"fabric/{weave}")

    # ── water ─────────────────────────────────────────────────────────────

    @classmethod
    def water(cls, time: float = 0.0) -> "ProceduralMaterial":
        def _eval(p: np.ndarray) -> PBRResult:
            N = p.shape[0]
            pt = p.copy()
            pt[:, 1] += time * 0.3

            wave1 = np.sin(p[:, 0] * 3.0 + time * 2.0) * np.cos(p[:, 2] * 2.5 + time * 1.5)
            wave2 = fbm(pt * 2.0, octaves=4, seed=90) * 0.5
            surface = wave1 * 0.3 + wave2

            caustic = _clamp01(np.abs(perlin_3d(pt * 8.0, seed=91)) * 3.0)

            deep = np.array([0.02, 0.08, 0.18], np.float32)
            shallow = np.array([0.08, 0.25, 0.35], np.float32)
            depth_factor = _clamp01(surface * 0.5 + 0.5)
            albedo = _mix_color(
                np.broadcast_to(deep, (N, 3)).copy(),
                np.broadcast_to(shallow, (N, 3)).copy(),
                depth_factor,
            )
            albedo += caustic[:, np.newaxis] * 0.15

            roughness = np.full(N, 0.02, np.float32) + np.abs(surface) * 0.05
            metallic = np.full(N, 0.0, np.float32)
            normal = _normal_from_noise(pt, lambda q, seed=0: fbm(q * 2.0, octaves=4, seed=90))
            return PBRResult(_clamp01(albedo), normal * 0.6, _clamp01(roughness), metallic)

        return cls(_eval, "water")

    # ── skin ──────────────────────────────────────────────────────────────

    @classmethod
    def skin(cls) -> "ProceduralMaterial":
        def _eval(p: np.ndarray) -> PBRResult:
            N = p.shape[0]
            base = np.array([0.75, 0.56, 0.46], np.float32)

            sss_scatter = _clamp01(fbm(p * 5.0, octaves=4, seed=100) * 0.3 + 0.5)
            pores = _clamp01(1.0 - np.abs(perlin_3d(p * 60.0, seed=101)) * 4.0)

            albedo = np.broadcast_to(base, (N, 3)).copy()
            albedo *= (0.85 + 0.15 * sss_scatter[:, np.newaxis])
            albedo[:, 0] += sss_scatter * 0.08

            roughness = np.full(N, 0.45, np.float32) + (1.0 - pores) * 0.2
            metallic = np.full(N, 0.0, np.float32)
            normal = _normal_from_noise(p, lambda q, seed=0: perlin_3d(q * 60.0, seed=101))
            return PBRResult(_clamp01(albedo), normal * 0.15, _clamp01(roughness), metallic)

        return cls(_eval, "skin")

    # ── concrete ──────────────────────────────────────────────────────────

    @classmethod
    def concrete(cls, age: float = 0.5) -> "ProceduralMaterial":
        def _eval(p: np.ndarray) -> PBRResult:
            N = p.shape[0]
            base = np.array([0.55, 0.54, 0.52], np.float32)

            aggregate = _clamp01(fbm(p * 10.0, octaves=4, seed=110) * 0.5 + 0.5)
            cracks = _clamp01(1.0 - ridged_noise(p * 3.0, octaves=4, seed=111) * (0.5 + age))

            albedo = np.broadcast_to(base, (N, 3)).copy()
            albedo *= (0.8 + 0.2 * aggregate[:, np.newaxis])
            albedo *= (0.7 + 0.3 * cracks[:, np.newaxis])

            stain_mask = _clamp01(fbm(p * 2.0, octaves=3, seed=112) * age)
            stain_color = np.array([0.30, 0.28, 0.22], np.float32)
            albedo = _mix_color(albedo, np.broadcast_to(stain_color, (N, 3)).copy(), stain_mask)

            roughness = np.full(N, 0.80, np.float32) + aggregate * 0.1
            metallic = np.full(N, 0.0, np.float32)
            normal = _normal_from_noise(p, lambda q, seed=0: fbm(q * 10.0, octaves=4, seed=110))
            return PBRResult(_clamp01(albedo), normal * 0.3, _clamp01(roughness), metallic)

        return cls(_eval, "concrete")

    # ── brick ─────────────────────────────────────────────────────────────

    @classmethod
    def brick(cls, mortar_width: float = 0.05) -> "ProceduralMaterial":
        def _eval(p: np.ndarray) -> PBRResult:
            N = p.shape[0]
            bx = 0.25
            by = 0.08

            row = np.floor(p[:, 1] / by)
            offset = np.where(row.astype(np.int32) % 2 == 0, 0.0, bx * 0.5)
            u = np.mod(p[:, 0] + offset, bx)
            v = np.mod(p[:, 1], by)

            mortar_u = np.minimum(u, bx - u) < mortar_width
            mortar_v = np.minimum(v, by - v) < mortar_width
            is_mortar = mortar_u | mortar_v

            brick_color = np.array([0.55, 0.18, 0.10], np.float32)
            mortar_color = np.array([0.70, 0.68, 0.62], np.float32)

            variation = perlin_3d(p * 5.0, seed=120) * 0.1
            albedo = np.where(
                is_mortar[:, np.newaxis],
                np.broadcast_to(mortar_color, (N, 3)),
                np.broadcast_to(brick_color, (N, 3)),
            ).copy().astype(np.float32)
            albedo += variation[:, np.newaxis]

            roughness = np.where(is_mortar, 0.90, 0.70).astype(np.float32)
            metallic = np.full(N, 0.0, np.float32)
            normal = _normal_from_noise(p, lambda q, seed=0: perlin_3d(q * 5.0, seed=120))
            depth = np.where(is_mortar, -1.0, 0.0).astype(np.float32)
            n_perturb = normal * 0.2
            n_perturb[:, 1] += depth * 0.3
            return PBRResult(_clamp01(albedo), n_perturb, _clamp01(roughness), metallic)

        return cls(_eval, "brick")

    # ── grass ─────────────────────────────────────────────────────────────

    @classmethod
    def grass(cls) -> "ProceduralMaterial":
        def _eval(p: np.ndarray) -> PBRResult:
            N = p.shape[0]
            base_green = np.array([0.15, 0.35, 0.08], np.float32)
            tip_green = np.array([0.30, 0.55, 0.12], np.float32)
            dry = np.array([0.40, 0.38, 0.15], np.float32)

            density = _clamp01(fbm(p * 6.0, octaves=4, seed=130) * 0.5 + 0.6)
            height_var = _clamp01(perlin_3d(p * 15.0, seed=131) * 0.5 + 0.5)
            dry_mask = _clamp01(fbm(p * 2.0, octaves=3, seed=132) * 0.5 - 0.1)

            albedo = _mix_color(
                np.broadcast_to(base_green, (N, 3)).copy(),
                np.broadcast_to(tip_green, (N, 3)).copy(),
                height_var,
            )
            albedo = _mix_color(albedo, np.broadcast_to(dry, (N, 3)).copy(), dry_mask)
            albedo *= density[:, np.newaxis]

            roughness = np.full(N, 0.85, np.float32) + density * 0.1
            metallic = np.full(N, 0.0, np.float32)
            normal = _normal_from_noise(p, lambda q, seed=0: perlin_3d(q * 15.0, seed=131))
            return PBRResult(_clamp01(albedo), normal * 0.35, _clamp01(roughness), metallic)

        return cls(_eval, "grass")

    # ── clouds ────────────────────────────────────────────────────────────

    @classmethod
    def clouds(cls, time: float = 0.0) -> "ProceduralMaterial":
        def _eval(p: np.ndarray) -> PBRResult:
            N = p.shape[0]
            pt = p.copy()
            pt[:, 0] += time * 0.1

            density = _clamp01(fbm(pt * 1.5, octaves=6, seed=140) * 0.8 + 0.2)
            detail = _clamp01(turbulence(pt * 4.0, octaves=3, seed=141) * 0.5)
            cloud = _clamp01(density - detail * 0.3)

            white = np.array([0.95, 0.95, 0.97], np.float32)
            grey = np.array([0.60, 0.62, 0.65], np.float32)
            albedo = _mix_color(
                np.broadcast_to(grey, (N, 3)).copy(),
                np.broadcast_to(white, (N, 3)).copy(),
                cloud,
            )

            roughness = np.full(N, 0.95, np.float32)
            metallic = np.full(N, 0.0, np.float32)
            normal = np.zeros((N, 3), np.float32)
            normal[:, 1] = 1.0
            return PBRResult(albedo, normal, roughness, metallic)

        return cls(_eval, "clouds")

    # ── ice ───────────────────────────────────────────────────────────────

    @classmethod
    def ice(cls) -> "ProceduralMaterial":
        def _eval(p: np.ndarray) -> PBRResult:
            N = p.shape[0]
            base = np.array([0.70, 0.82, 0.92], np.float32)
            deep = np.array([0.30, 0.50, 0.70], np.float32)

            fractures = ridged_noise(p * 4.0, octaves=5, seed=150)
            fracture_mask = _clamp01(fractures * 0.6)
            depth_var = _clamp01(fbm(p * 2.0, octaves=3, seed=151) * 0.5 + 0.5)

            albedo = _mix_color(
                np.broadcast_to(base, (N, 3)).copy(),
                np.broadcast_to(deep, (N, 3)).copy(),
                depth_var,
            )
            albedo += fracture_mask[:, np.newaxis] * 0.15

            roughness = np.full(N, 0.08, np.float32) + fracture_mask * 0.3
            metallic = np.full(N, 0.0, np.float32)
            normal = _normal_from_noise(p, lambda q, seed=0: ridged_noise(q * 4.0, octaves=5, seed=150))
            return PBRResult(_clamp01(albedo), normal * 0.25, _clamp01(roughness), metallic)

        return cls(_eval, "ice")

    # ── lava ──────────────────────────────────────────────────────────────

    @classmethod
    def lava(cls, time: float = 0.0) -> "ProceduralMaterial":
        def _eval(p: np.ndarray) -> PBRResult:
            N = p.shape[0]
            pt = p.copy()
            pt[:, 0] += time * 0.05
            pt[:, 2] += time * 0.03

            flow = _clamp01(fbm(pt * 2.0, octaves=5, seed=160) * 0.5 + 0.5)
            crust = _clamp01(ridged_noise(p * 3.0, octaves=4, seed=161) * 0.7)

            hot = np.array([1.0, 0.35, 0.02], np.float32)
            cool = np.array([0.08, 0.03, 0.02], np.float32)
            glow_mask = _clamp01(flow - crust * 0.6)

            albedo = _mix_color(
                np.broadcast_to(cool, (N, 3)).copy(),
                np.broadcast_to(hot, (N, 3)).copy(),
                glow_mask,
            )

            roughness = np.full(N, 0.90, np.float32) - glow_mask * 0.6
            metallic = np.full(N, 0.0, np.float32)
            normal = _normal_from_noise(p, lambda q, seed=0: ridged_noise(q * 3.0, octaves=4, seed=161))
            return PBRResult(_clamp01(albedo), normal * 0.4, _clamp01(roughness), metallic)

        return cls(_eval, "lava")

    # ── sand ──────────────────────────────────────────────────────────────

    @classmethod
    def sand(cls) -> "ProceduralMaterial":
        def _eval(p: np.ndarray) -> PBRResult:
            N = p.shape[0]
            base = np.array([0.76, 0.70, 0.50], np.float32)
            dark = np.array([0.60, 0.52, 0.35], np.float32)

            ripples = np.sin(p[:, 0] * 15.0 + perlin_3d(p * 3.0, seed=170) * 3.0) * 0.5 + 0.5
            grain = _clamp01(perlin_3d(p * 80.0, seed=171) * 0.5 + 0.5)

            albedo = _mix_color(
                np.broadcast_to(base, (N, 3)).copy(),
                np.broadcast_to(dark, (N, 3)).copy(),
                ripples * 0.3,
            )
            albedo *= (0.9 + 0.1 * grain[:, np.newaxis])

            roughness = np.full(N, 0.85, np.float32) + grain * 0.1
            metallic = np.full(N, 0.0, np.float32)
            normal = _normal_from_noise(p, lambda q, seed=0: perlin_3d(q * 15.0, seed=170))
            return PBRResult(_clamp01(albedo), normal * 0.2, _clamp01(roughness), metallic)

        return cls(_eval, "sand")

    # ── bark ──────────────────────────────────────────────────────────────

    @classmethod
    def bark(cls, tree_type: str = "oak") -> "ProceduralMaterial":
        configs = {
            "oak":   {"color": np.array([0.30, 0.22, 0.14], np.float32), "ridge_scale": 8.0},
            "birch": {"color": np.array([0.82, 0.78, 0.72], np.float32), "ridge_scale": 4.0},
            "pine":  {"color": np.array([0.35, 0.20, 0.10], np.float32), "ridge_scale": 12.0},
        }
        cfg = configs.get(tree_type, configs["oak"])

        def _eval(p: np.ndarray) -> PBRResult:
            N = p.shape[0]
            ridges = ridged_noise(
                np.stack([p[:, 0] * 2.0, p[:, 1] * cfg["ridge_scale"], p[:, 2] * 2.0], axis=-1),
                octaves=5, seed=180,
            )
            ridge_mask = _clamp01(ridges * 0.5)

            base = np.broadcast_to(cfg["color"], (N, 3)).copy()
            dark = cfg["color"] * 0.5
            albedo = _mix_color(base, np.broadcast_to(dark, (N, 3)).copy(), ridge_mask)

            if tree_type == "birch":
                strips = _clamp01(np.sin(p[:, 1] * 40.0) * 0.5 + 0.3)
                dark_strip = np.array([0.15, 0.12, 0.10], np.float32)
                albedo = _mix_color(albedo, np.broadcast_to(dark_strip, (N, 3)).copy(), strips * 0.6)

            moss = _clamp01(fbm(p * 3.0, octaves=3, seed=181) - 0.3)
            moss_color = np.array([0.12, 0.25, 0.08], np.float32)
            albedo = _mix_color(albedo, np.broadcast_to(moss_color, (N, 3)).copy(), moss * 0.4)

            roughness = np.full(N, 0.90, np.float32) + ridge_mask * 0.08
            metallic = np.full(N, 0.0, np.float32)
            normal = _normal_from_noise(
                p,
                lambda q, seed=0: ridged_noise(
                    np.stack([q[:, 0] * 2.0, q[:, 1] * cfg["ridge_scale"], q[:, 2] * 2.0], axis=-1),
                    octaves=5, seed=180,
                ),
            )
            return PBRResult(_clamp01(albedo), normal * 0.45, _clamp01(roughness), metallic)

        return cls(_eval, f"bark/{tree_type}")


# ═══════════════════════════════════════════════════════════════════════════
# WEATHERING / AGING OVERLAYS
# ═══════════════════════════════════════════════════════════════════════════

def apply_weathering(albedo: np.ndarray, roughness: np.ndarray,
                     position: np.ndarray, age: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Add wear, grime, and edge damage to existing PBR channels.

    Returns (albedo, roughness) — both modified in-place copies.
    """
    position = _ensure_n3(position)
    albedo = albedo.copy()
    roughness = roughness.copy()

    grime = _clamp01(fbm(position * 4.0, octaves=4, seed=200) * age)
    grime_color = np.array([0.12, 0.10, 0.07], np.float32)
    N = position.shape[0]
    albedo = _mix_color(albedo, np.broadcast_to(grime_color, (N, 3)).copy(), grime)

    edge_wear = _clamp01(ridged_noise(position * 6.0, octaves=3, seed=201) * age * 1.5)
    albedo *= (1.0 - edge_wear[:, np.newaxis] * 0.3)

    roughness += grime * 0.2 + edge_wear * 0.15
    return _clamp01(albedo), _clamp01(roughness)


def apply_dust(albedo: np.ndarray, roughness: np.ndarray,
               position: np.ndarray, amount: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """Accumulate dust in crevices (upward-facing, concave areas).

    Returns (albedo, roughness).
    """
    position = _ensure_n3(position)
    albedo = albedo.copy()
    roughness = roughness.copy()

    N = position.shape[0]
    crevice = _clamp01(1.0 - voronoi_3d(position * 5.0, seed=210)[0] * 3.0)
    height_bias = _clamp01(position[:, 1] * 0.5 + 0.5)
    dust_mask = _clamp01(crevice * height_bias * amount * 2.0)

    dust_color = np.array([0.55, 0.52, 0.45], np.float32)
    albedo = _mix_color(albedo, np.broadcast_to(dust_color, (N, 3)).copy(), dust_mask)
    roughness += dust_mask * 0.25
    return _clamp01(albedo), _clamp01(roughness)


def apply_rust(albedo: np.ndarray, metallic: np.ndarray, roughness: np.ndarray,
               position: np.ndarray, amount: float = 0.3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply rust/corrosion to metallic surfaces.

    Returns (albedo, metallic, roughness).
    """
    position = _ensure_n3(position)
    albedo = albedo.copy()
    metallic = metallic.copy()
    roughness = roughness.copy()

    N = position.shape[0]
    rust_pattern = _clamp01(
        fbm(position * 5.0, octaves=5, seed=220) * 0.5
        + turbulence(position * 8.0, octaves=3, seed=221) * 0.3
    )
    edge_rust = _clamp01(ridged_noise(position * 4.0, octaves=3, seed=222) * 0.8)
    rust_mask = _clamp01((rust_pattern + edge_rust * 0.5) * amount * 2.0)

    rust_color_a = np.array([0.45, 0.18, 0.05], np.float32)
    rust_color_b = np.array([0.30, 0.12, 0.03], np.float32)
    rust_var = _clamp01(perlin_3d(position * 12.0, seed=223) * 0.5 + 0.5)
    rust_color = _mix_color(
        np.broadcast_to(rust_color_a, (N, 3)).copy(),
        np.broadcast_to(rust_color_b, (N, 3)).copy(),
        rust_var,
    )

    albedo = _mix_color(albedo, rust_color, rust_mask)
    metallic *= (1.0 - rust_mask * 0.9)
    roughness += rust_mask * 0.4
    return _clamp01(albedo), _clamp01(metallic), _clamp01(roughness)


# ═══════════════════════════════════════════════════════════════════════════
# RENDER ENGINE INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════

def sample_material(proc_mat: ProceduralMaterial,
                    positions: np.ndarray) -> Dict[str, np.ndarray]:
    """Evaluate a procedural material at hit positions for the render engine.

    Parameters
    ----------
    proc_mat : ProceduralMaterial
    positions : (N, 3) float32

    Returns
    -------
    dict with keys ``albedo`` (N,3), ``roughness`` (N,), ``metallic`` (N,),
    ``normal_offset`` (N,3) — ready for the renderer's per-pixel shading.
    """
    result = proc_mat(_ensure_n3(positions))
    return {
        "albedo": result.albedo,
        "roughness": result.roughness,
        "metallic": result.metallic,
        "normal_offset": result.normal_perturbation,
    }
