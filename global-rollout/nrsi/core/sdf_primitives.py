"""
NRSI SDF Primitive Library — 50+ shapes for NRS-native image/video generation.

Extends render_engine.py with geometric, organic, architectural, vehicle,
furniture, terrain, food, and tool primitives.  Every function is a pure
vectorised numpy operation over (N,3) float32 point arrays, returning (N,)
float32 signed distances.

Dependency: numpy only.  No circular imports — noise helpers are inlined.
"""
from __future__ import annotations

import math
import numpy as np
from typing import Tuple

# ═══════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════

Vec3 = np.ndarray

_F32 = np.float32


def _v3(x: float, y: float, z: float) -> Vec3:
    return np.array([x, y, z], dtype=_F32)


def _dot(a: Vec3, b: Vec3) -> np.ndarray:
    return np.einsum("...i,...i->...", a, b)


def _norm(v: Vec3) -> np.ndarray:
    return np.sqrt(_dot(v, v).clip(1e-12))


def _normalize(v: Vec3) -> Vec3:
    n = _norm(v)
    if v.ndim == 1:
        return v / max(n, 1e-9)
    return v / n[..., np.newaxis].clip(1e-9)


def _clamp(x, lo=0.0, hi=1.0):
    return np.clip(x, lo, hi)


def _mix(a, b, t):
    return a * (1.0 - t) + b * t


def _length2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sqrt(x * x + y * y)


def _max_comp(v: Vec3) -> np.ndarray:
    return np.max(v, axis=-1)


# ── hash-based noise (self-contained, no external deps) ──────────────────

def _fract(x):
    return x - np.floor(x)


def _hash_noise_3d(p: Vec3) -> np.ndarray:
    """Fast hash-based 3D value noise returning values in [0, 1]."""
    ip = np.floor(p).astype(_F32)
    fp = p - ip

    fp = fp * fp * (3.0 - 2.0 * fp)

    def _h(offset):
        q = ip + np.asarray(offset, dtype=_F32)
        dot_val = q[..., 0] * 127.1 + q[..., 1] * 311.7 + q[..., 2] * 74.7
        return _fract(np.sin(dot_val) * 43758.5453)

    c000 = _h([0, 0, 0])
    c100 = _h([1, 0, 0])
    c010 = _h([0, 1, 0])
    c110 = _h([1, 1, 0])
    c001 = _h([0, 0, 1])
    c101 = _h([1, 0, 1])
    c011 = _h([0, 1, 1])
    c111 = _h([1, 1, 1])

    fx, fy, fz = fp[..., 0], fp[..., 1], fp[..., 2]
    x0 = _mix(c000, c100, fx)
    x1 = _mix(c010, c110, fx)
    x2 = _mix(c001, c101, fx)
    x3 = _mix(c011, c111, fx)
    y0 = _mix(x0, x1, fy)
    y1 = _mix(x2, x3, fy)
    return _mix(y0, y1, fz).astype(_F32)


def _fbm(p: Vec3, octaves: int = 4) -> np.ndarray:
    """Fractional Brownian motion built on _hash_noise_3d."""
    val = np.zeros(p.shape[0], dtype=_F32)
    amp = 0.5
    freq = 1.0
    for _ in range(octaves):
        val += amp * _hash_noise_3d(p * freq)
        freq *= 2.0
        amp *= 0.5
    return val


# ═══════════════════════════════════════════════════════════════════════════
# BASE PRIMITIVES (duplicated locally to avoid import cycles)
# ═══════════════════════════════════════════════════════════════════════════

def _sd_sphere(p: Vec3, radius: float) -> np.ndarray:
    return _norm(p) - radius


def _sd_box(p: Vec3, half_extents: Vec3) -> np.ndarray:
    h = np.asarray(half_extents, dtype=_F32)
    q = np.abs(p) - h
    return _norm(np.maximum(q, 0.0)) + np.minimum(np.max(q, axis=-1), 0.0)


def _sd_capsule(p: Vec3, a: Vec3, b: Vec3, radius: float) -> np.ndarray:
    pa = p - a
    ba = b - a
    h = _clamp(_dot(pa, ba) / _dot(ba, ba), 0.0, 1.0)
    if pa.ndim > 1:
        return _norm(pa - ba * h[..., np.newaxis]) - radius
    return _norm(pa - ba * h) - radius


def _sd_cylinder(p: Vec3, radius: float, half_height: float) -> np.ndarray:
    d_xz = _length2(p[..., 0], p[..., 2]) - radius
    d_y = np.abs(p[..., 1]) - half_height
    outside = _length2(np.maximum(d_xz, 0.0), np.maximum(d_y, 0.0))
    inside = np.minimum(np.maximum(d_xz, d_y), 0.0)
    return outside + inside


def _sd_torus(p: Vec3, major_r: float, minor_r: float) -> np.ndarray:
    q_xz = _length2(p[..., 0], p[..., 2]) - major_r
    return _length2(q_xz, p[..., 1]) - minor_r


def _sd_ellipsoid(p: Vec3, radii: Vec3) -> np.ndarray:
    r = np.asarray(radii, dtype=_F32)
    k0 = _norm(p / r)
    k1 = _norm(p / (r * r))
    return k0 * (k0 - 1.0) / k1


# ── CSG ops (local copies) ───────────────────────────────────────────────

def _op_union(d1, d2):
    return np.minimum(d1, d2)


def _op_subtract(d1, d2):
    return np.maximum(d1, -d2)


def _op_intersect(d1, d2):
    return np.maximum(d1, d2)


def _op_smooth_union(d1, d2, k: float = 0.1):
    h = _clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0)
    return d2 * (1 - h) + d1 * h - k * h * (1 - h)


def _op_smooth_subtract(d1, d2, k: float = 0.1):
    h = _clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0)
    return _mix(d1, -d2, h) + k * h * (1.0 - h)


def _op_translate(p: Vec3, offset) -> Vec3:
    return p - np.asarray(offset, dtype=_F32)


def _op_rotate_y(p: Vec3, angle: float) -> Vec3:
    c, s = math.cos(angle), math.sin(angle)
    x = p[..., 0] * c + p[..., 2] * s
    z = -p[..., 0] * s + p[..., 2] * c
    return np.stack([x, p[..., 1], z], axis=-1).astype(_F32)


# ═══════════════════════════════════════════════════════════════════════════
# GEOMETRIC EXTENDED
# ═══════════════════════════════════════════════════════════════════════════

def sd_cone(p: Vec3, radius: float, height: float) -> np.ndarray:
    """Cone along Y axis with tip at (0, height, 0) and base radius at y=0."""
    q = np.stack([_length2(p[..., 0], p[..., 2]), p[..., 1]], axis=-1)
    tip = np.array([0.0, height], dtype=_F32)
    base_edge = np.array([radius, 0.0], dtype=_F32)
    cb = base_edge - tip
    qp = q - tip
    t = _clamp(_dot(qp, cb) / _dot(cb, cb), 0.0, 1.0)
    proj = tip + cb * t[..., np.newaxis]
    d_side = _norm(q - proj)
    d_base = _length2(np.maximum(_length2(q[..., 0], 0.0) - radius, 0.0),
                      np.maximum(-q[..., 1], 0.0))
    sign = np.where(
        (q[..., 0] * height - q[..., 1] * radius) > 0.0, 1.0, -1.0
    )
    return np.where(q[..., 1] < 0.0, d_base, sign * d_side).astype(_F32)


def sd_ellipsoid(p: Vec3, radii: Vec3) -> np.ndarray:
    """Ellipsoid centred at origin with semi-axes (rx, ry, rz)."""
    return _sd_ellipsoid(p, radii)


def sd_hexagonal_prism(p: Vec3, h_vec: Vec3) -> np.ndarray:
    """Hexagonal prism.  h_vec = (hex_radius, half_height)."""
    hv = np.asarray(h_vec, dtype=_F32)
    hr, hh = float(hv[0]), float(hv[1])
    k = np.array([-0.8660254, 0.5, 0.57735027], dtype=_F32)
    ap = np.abs(p)
    xy_shift = 2.0 * np.minimum(
        k[0] * ap[..., 0] + k[1] * ap[..., 2], 0.0
    )
    ax = ap[..., 0] + xy_shift * k[0]
    az = ap[..., 2] - xy_shift * k[1]
    ax = ax - np.clip(ax, -k[2] * hr, k[2] * hr)
    d_xz = _length2(ax, az - hr) * np.sign(az - hr)
    d_y = ap[..., 1] - hh
    return (np.minimum(np.maximum(d_xz, d_y), 0.0) +
            _length2(np.maximum(d_xz, 0.0), np.maximum(d_y, 0.0))).astype(_F32)


def sd_triangular_prism(p: Vec3, h_vec: Vec3) -> np.ndarray:
    """Triangular prism.  h_vec = (base_half_width, half_height)."""
    hv = np.asarray(h_vec, dtype=_F32)
    bw, hh = float(hv[0]), float(hv[1])
    q = np.abs(p)
    d1 = q[..., 2] - hh
    d2 = np.maximum(
        q[..., 0] * 0.866025 + p[..., 1] * 0.5,
        -p[..., 1]
    ) - bw * 0.5
    return (np.maximum(d1, d2) +
            np.minimum(np.maximum(d1, 0.0) + np.maximum(d2, 0.0), 0.0) * 0.0
            + _length2(np.maximum(d1, 0.0), np.maximum(d2, 0.0)) * 0.0
            ).astype(_F32)


def sd_pyramid(p: Vec3, height: float) -> np.ndarray:
    """Four-sided pyramid with base at y=0 and apex at y=height, base ±0.5."""
    h = float(height)
    a = np.abs(p[..., 0]) - 0.5
    b = np.abs(p[..., 2]) - 0.5
    m = np.maximum(a, b)
    q_x = np.where(b > a, np.abs(p[..., 2]) - 0.5, np.abs(p[..., 0]) - 0.5)
    q_y = p[..., 1]
    q_z = np.where(b > a, np.abs(p[..., 0]) - 0.5, np.abs(p[..., 2]) - 0.5)

    s = np.maximum(-q_x, 0.0)
    t = _clamp((q_z + q_x * h + q_y) / (2.0 * (1.0 + h * h)), 0.0, 1.0)

    a_vec_x = q_x + s
    a_vec_y = q_y - h
    b_vec_x = q_x - 1.0 + t
    b_vec_y = q_y - h * t

    d_a = a_vec_x * a_vec_x + a_vec_y * a_vec_y
    d_b = b_vec_x * b_vec_x + b_vec_y * b_vec_y

    d2 = np.where(
        np.minimum(a_vec_y, b_vec_y) < 0.0,
        np.minimum(d_a, d_b), 0.0
    )
    d_base = np.maximum(m, -q_y)
    sign = np.sign(np.maximum(q_z * q_y - q_x * h, q_x))
    return (np.maximum(d_base, 0.0) + np.sqrt(d2 + 1e-12) * sign).astype(_F32)


def sd_octahedron(p: Vec3, size: float) -> np.ndarray:
    """Exact octahedron SDF."""
    s = float(size)
    ap = np.abs(p)
    m = ap[..., 0] + ap[..., 1] + ap[..., 2] - s
    q = np.empty_like(p)
    mask1 = 3.0 * ap[..., 0] < m
    mask2 = 3.0 * ap[..., 1] < m
    mask3 = 3.0 * ap[..., 2] < m

    default = ap.copy()
    q[mask1] = ap[mask1]
    q[~mask1 & mask2] = ap[~mask1 & mask2][..., [1, 0, 2]]
    q[~mask1 & ~mask2 & mask3] = ap[~mask1 & ~mask2 & mask3][..., [2, 1, 0]]
    q[~mask1 & ~mask2 & ~mask3] = default[~mask1 & ~mask2 & ~mask3]

    k = _clamp(0.5 * (q[..., 2] - q[..., 1] + s), 0.0, s)
    d = _norm(np.stack([q[..., 0], q[..., 1] - s + k, q[..., 2] - k], axis=-1))
    return d.astype(_F32)


def sd_ring(p: Vec3, major_r: float, minor_r: float, thickness: float) -> np.ndarray:
    """Thick ring / washer in XZ plane."""
    d_torus = _sd_torus(p, major_r, minor_r)
    d_slab = np.abs(p[..., 1]) - thickness * 0.5
    return _op_intersect(d_torus, d_slab).astype(_F32)


def sd_arc(p: Vec3, angle: float, radius: float, thickness: float) -> np.ndarray:
    """Arc segment in XZ plane spanning ±angle/2 from +X axis."""
    half = angle * 0.5
    sc = np.array([math.sin(half), math.cos(half)], dtype=_F32)
    q = np.stack([np.abs(p[..., 0]), p[..., 2]], axis=-1)
    cond = sc[1] * q[..., 0] > sc[0] * q[..., 1]
    d_arc = np.where(
        cond,
        _length2(q[..., 0] - sc[0] * radius, q[..., 1] - sc[1] * radius),
        np.abs(_length2(q[..., 0], q[..., 1]) - radius)
    )
    d = _length2(d_arc, np.abs(p[..., 1])) - thickness
    return d.astype(_F32)


def sd_bezier(p: Vec3, a: Vec3, b: Vec3, c: Vec3, thickness: float = 0.05) -> np.ndarray:
    """Distance to a quadratic Bezier curve (a→b→c) with given thickness."""
    a, b, c = [np.asarray(v, dtype=_F32) for v in (a, b, c)]
    pa = p - a
    ba = b - a
    ca = c - 2.0 * b + a

    dot_ca_ca = _dot(ca, ca)
    dot_ca_ba = _dot(ca, ba)

    kx = _dot(pa, ca)
    ky = _dot(pa, ba)

    t = _clamp((kx * (-2.0 * dot_ca_ba) + ky * dot_ca_ca) /
               (dot_ca_ca * dot_ca_ca + 1e-10), 0.0, 1.0)
    if pa.ndim > 1:
        proj = a + 2.0 * t[..., np.newaxis] * ba + t[..., np.newaxis] ** 2 * ca
    else:
        proj = a + 2.0 * t * ba + t ** 2 * ca
    return (_norm(p - proj) - thickness).astype(_F32)


def sd_link(p: Vec3, length: float, r1: float, r2: float) -> np.ndarray:
    """Chain link oriented along Y."""
    le = float(length)
    q = p.copy()
    q[..., 1] = np.maximum(np.abs(q[..., 1]) - le, 0.0)
    d_xz = _length2(q[..., 0], q[..., 1]) - r1
    return (_length2(d_xz, q[..., 2]) - r2).astype(_F32)


# ═══════════════════════════════════════════════════════════════════════════
# OPERATORS EXTENDED
# ═══════════════════════════════════════════════════════════════════════════

def op_rotate_x(p: Vec3, angle: float) -> Vec3:
    """Rotate points around X axis."""
    c, s = math.cos(angle), math.sin(angle)
    y = p[..., 1] * c - p[..., 2] * s
    z = p[..., 1] * s + p[..., 2] * c
    return np.stack([p[..., 0], y, z], axis=-1).astype(_F32)


def op_rotate_z(p: Vec3, angle: float) -> Vec3:
    """Rotate points around Z axis."""
    c, s = math.cos(angle), math.sin(angle)
    x = p[..., 0] * c - p[..., 1] * s
    y = p[..., 0] * s + p[..., 1] * c
    return np.stack([x, y, p[..., 2]], axis=-1).astype(_F32)


def op_rotate_xyz(p: Vec3, angles: Vec3) -> Vec3:
    """Euler rotation (rx, ry, rz) applied in X→Y→Z order."""
    a = np.asarray(angles, dtype=_F32)
    q = op_rotate_x(p, float(a[0]))
    q = _op_rotate_y(q, float(a[1]))
    q = op_rotate_z(q, float(a[2]))
    return q


def op_twist(p: Vec3, amount: float) -> Vec3:
    """Twist deformation along Y axis."""
    angle = p[..., 1] * amount
    c, s = np.cos(angle), np.sin(angle)
    x = p[..., 0] * c - p[..., 2] * s
    z = p[..., 0] * s + p[..., 2] * c
    return np.stack([x, p[..., 1], z], axis=-1).astype(_F32)


def op_bend(p: Vec3, amount: float) -> Vec3:
    """Cheap bend deformation along X axis."""
    angle = p[..., 0] * amount
    c, s = np.cos(angle), np.sin(angle)
    x = c * p[..., 0] - s * p[..., 1]
    y = s * p[..., 0] + c * p[..., 1]
    return np.stack([x, y, p[..., 2]], axis=-1).astype(_F32)


def op_elongate(p: Vec3, h: Vec3) -> Tuple[Vec3, np.ndarray]:
    """Elongation: returns (modified_p, extra_distance) — add extra to SDF result."""
    hv = np.asarray(h, dtype=_F32)
    q = np.abs(p) - hv
    extra = np.minimum(_max_comp(np.minimum(np.abs(p), hv[np.newaxis, :])), 0.0)
    return np.maximum(q, 0.0).astype(_F32), extra.astype(_F32)


def op_round(d: np.ndarray, radius: float) -> np.ndarray:
    """Round any SDF by subtracting a radius."""
    return d - radius


def op_onion(d: np.ndarray, thickness: float) -> np.ndarray:
    """Hollow shell of any SDF."""
    return np.abs(d) - thickness


def op_repeat(p: Vec3, spacing: Vec3) -> Vec3:
    """Infinite repetition with given spacing per axis."""
    s = np.asarray(spacing, dtype=_F32)
    return (np.mod(p + 0.5 * s, s) - 0.5 * s).astype(_F32)


def op_mirror_x(p: Vec3) -> Vec3:
    """Mirror across the YZ plane (reflect X)."""
    out = p.copy()
    out[..., 0] = np.abs(p[..., 0])
    return out


def op_smooth_subtract(d1: np.ndarray, d2: np.ndarray, k: float = 0.1) -> np.ndarray:
    """Smooth subtraction (d1 minus d2)."""
    h = _clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0)
    return _mix(d1, -d2, h) + k * h * (1.0 - h)


def op_smooth_intersect(d1: np.ndarray, d2: np.ndarray, k: float = 0.1) -> np.ndarray:
    """Smooth intersection."""
    h = _clamp(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0)
    return _mix(d2, d1, h) + k * h * (1.0 - h)


# ═══════════════════════════════════════════════════════════════════════════
# HUMAN / ORGANIC
# ═══════════════════════════════════════════════════════════════════════════

def sd_human_torso(p: Vec3) -> np.ndarray:
    """Simplified torso: blended ellipsoids for chest and abdomen."""
    chest = _sd_ellipsoid(_op_translate(p, [0, 0.9, 0]), _v3(0.28, 0.25, 0.18))
    abdomen = _sd_ellipsoid(_op_translate(p, [0, 0.55, 0]), _v3(0.22, 0.2, 0.15))
    return _op_smooth_union(chest, abdomen, 0.15).astype(_F32)


def sd_human_head(p: Vec3) -> np.ndarray:
    """Head with basic cranium, jaw, and nose bumps."""
    cranium = _sd_ellipsoid(_op_translate(p, [0, 1.55, 0]), _v3(0.12, 0.14, 0.13))
    jaw = _sd_ellipsoid(_op_translate(p, [0, 1.42, 0.02]), _v3(0.09, 0.06, 0.08))
    nose = _sd_ellipsoid(_op_translate(p, [0, 1.5, -0.13]), _v3(0.02, 0.03, 0.03))
    head = _op_smooth_union(cranium, jaw, 0.06)
    return _op_smooth_union(head, nose, 0.03).astype(_F32)


def sd_human_limb(p: Vec3, length: float, radius_start: float,
                  radius_end: float) -> np.ndarray:
    """Tapered limb (capsule with varying radius approximated by two capsules)."""
    mid = length * 0.5
    r_mid = (radius_start + radius_end) * 0.5
    upper = _sd_capsule(p, _v3(0, 0, 0), _v3(0, -mid, 0), (radius_start + r_mid) * 0.5)
    lower = _sd_capsule(p, _v3(0, -mid, 0), _v3(0, -length, 0), (r_mid + radius_end) * 0.5)
    return _op_smooth_union(upper, lower, 0.02).astype(_F32)


def sd_human_hand(p: Vec3) -> np.ndarray:
    """Simplified hand: palm box + five finger capsules."""
    palm = _sd_ellipsoid(p, _v3(0.04, 0.02, 0.05))
    d = palm
    for i in range(5):
        angle = math.radians(-30 + i * 15)
        fx = math.sin(angle) * 0.04
        fz = -0.05 + math.cos(angle) * 0.0
        finger = _sd_capsule(p, _v3(fx, 0, -0.05), _v3(fx * 2.2, 0, -0.05 - 0.04), 0.008)
        d = _op_smooth_union(d, finger, 0.01)
    return d.astype(_F32)


def sd_human_figure(p: Vec3, pose: str = "standing") -> np.ndarray:
    """Complete human figure assembled from parts.  Poses: standing, sitting, walking."""
    d = sd_human_torso(p)
    d = _op_smooth_union(d, sd_human_head(p), 0.08)

    if pose == "sitting":
        leg_angle = -math.pi / 2
    elif pose == "walking":
        leg_angle = math.radians(20)
    else:
        leg_angle = 0.0

    for side in (-1, 1):
        arm_p = _op_translate(p, [side * 0.32, 1.0, 0])
        d = _op_smooth_union(d, sd_human_limb(arm_p, 0.55, 0.04, 0.03), 0.06)

        hip = _op_translate(p, [side * 0.12, 0.4, 0])
        if leg_angle != 0.0:
            hip = op_rotate_x(hip, leg_angle * side)
        d = _op_smooth_union(d, sd_human_limb(hip, 0.75, 0.06, 0.04), 0.06)

    return d.astype(_F32)


def sd_animal_body(p: Vec3, species: str = "cat") -> np.ndarray:
    """Basic four-legged animal.  Species: cat, dog, horse."""
    if species == "horse":
        body_rx, body_ry, body_rz = 0.6, 0.25, 0.2
        leg_len, leg_r = 0.5, 0.04
        head_r = 0.14
        neck_len = 0.3
    elif species == "dog":
        body_rx, body_ry, body_rz = 0.35, 0.15, 0.12
        leg_len, leg_r = 0.25, 0.03
        head_r = 0.1
        neck_len = 0.15
    else:
        body_rx, body_ry, body_rz = 0.2, 0.1, 0.1
        leg_len, leg_r = 0.15, 0.02
        head_r = 0.08
        neck_len = 0.1

    body = _sd_ellipsoid(p, _v3(body_rx, body_ry, body_rz))
    head = _sd_sphere(_op_translate(p, [-body_rx - neck_len, body_ry * 0.5, 0]), head_r)
    d = _op_smooth_union(body, head, 0.08)

    for sx in (-1, 1):
        for sz in (-1, 1):
            lp = _op_translate(p, [sx * body_rx * 0.6, -body_ry, sz * body_rz * 0.6])
            leg = _sd_capsule(lp, _v3(0, 0, 0), _v3(0, -leg_len, 0), leg_r)
            d = _op_smooth_union(d, leg, 0.04)

    return d.astype(_F32)


def sd_fish(p: Vec3, length: float = 0.5) -> np.ndarray:
    """Fish shape: tapered ellipsoid body + tail fin."""
    body = _sd_ellipsoid(p, _v3(length * 0.5, length * 0.15, length * 0.1))
    tail = _sd_ellipsoid(
        _op_translate(p, [length * 0.45, 0, 0]),
        _v3(length * 0.15, length * 0.2, length * 0.02)
    )
    return _op_smooth_union(body, tail, length * 0.08).astype(_F32)


def sd_bird(p: Vec3, wingspan: float = 1.0) -> np.ndarray:
    """Bird silhouette: body ellipsoid + two wing capsules + tail."""
    ws = float(wingspan)
    body = _sd_ellipsoid(p, _v3(ws * 0.15, ws * 0.08, ws * 0.08))
    head = _sd_sphere(_op_translate(p, [-ws * 0.18, ws * 0.04, 0]), ws * 0.06)
    d = _op_smooth_union(body, head, ws * 0.04)

    for side in (-1, 1):
        wing = _sd_capsule(
            p,
            _v3(0, 0, side * ws * 0.05),
            _v3(ws * 0.05, ws * 0.1, side * ws * 0.5),
            ws * 0.02
        )
        d = _op_smooth_union(d, wing, ws * 0.03)

    tail = _sd_ellipsoid(
        _op_translate(p, [ws * 0.2, 0, 0]),
        _v3(ws * 0.08, ws * 0.01, ws * 0.06)
    )
    return _op_smooth_union(d, tail, ws * 0.03).astype(_F32)


# ═══════════════════════════════════════════════════════════════════════════
# VEGETATION
# ═══════════════════════════════════════════════════════════════════════════

def sd_tree_trunk(p: Vec3, height: float = 2.0, radius: float = 0.1,
                  bend: float = 0.0) -> np.ndarray:
    """Tree trunk with taper and optional bend."""
    q = p.copy()
    if abs(bend) > 1e-6:
        q = op_bend(q, bend)
    taper = 1.0 - _clamp(p[..., 1] / height, 0.0, 1.0) * 0.6
    d_xz = _length2(q[..., 0], q[..., 2]) - radius * taper
    d_y = np.abs(q[..., 1] - height * 0.5) - height * 0.5
    outside = _length2(np.maximum(d_xz, 0.0), np.maximum(d_y, 0.0))
    inside = np.minimum(np.maximum(d_xz, d_y), 0.0)
    return (outside + inside).astype(_F32)


def sd_tree_canopy(p: Vec3, style: str = "deciduous") -> np.ndarray:
    """Tree canopy.  Styles: deciduous (sphere-ish), conifer (cone), palm (top spray)."""
    if style == "conifer":
        return sd_cone(_op_translate(p, [0, -1.5, 0]), 0.8, 2.0)
    elif style == "palm":
        d = _sd_sphere(p, 0.3)
        for angle_deg in range(0, 360, 45):
            a = math.radians(angle_deg)
            tip = _v3(math.cos(a) * 1.0, -0.3, math.sin(a) * 1.0)
            frond = _sd_capsule(p, _v3(0, 0, 0), tip, 0.04)
            d = _op_smooth_union(d, frond, 0.1)
        return d.astype(_F32)
    else:
        noise_disp = _hash_noise_3d(p * 3.0) * 0.15
        return (_sd_sphere(p, 1.0) - noise_disp).astype(_F32)


def sd_tree(p: Vec3, species: str = "oak") -> np.ndarray:
    """Complete tree.  Species: oak, pine, palm, birch, willow."""
    species_params = {
        "oak":    (3.0, 0.15, 0.0, "deciduous", 3.5),
        "pine":   (4.0, 0.1,  0.0, "conifer",   4.5),
        "palm":   (5.0, 0.12, 0.1, "palm",      5.5),
        "birch":  (4.0, 0.08, 0.0, "deciduous", 4.5),
        "willow": (3.0, 0.15, 0.05, "deciduous", 3.2),
    }
    trunk_h, trunk_r, trunk_bend, canopy_style, canopy_y = species_params.get(
        species, species_params["oak"]
    )
    trunk = sd_tree_trunk(p, trunk_h, trunk_r, trunk_bend)
    canopy = sd_tree_canopy(_op_translate(p, [0, -canopy_y, 0]), canopy_style)
    return _op_smooth_union(trunk, canopy, 0.2).astype(_F32)


def sd_bush(p: Vec3, density: float = 0.5) -> np.ndarray:
    """Bush / shrub: cluster of displaced spheres."""
    d = _sd_sphere(p, 0.4)
    noise = _hash_noise_3d(p * 4.0) * 0.15 * density
    for offset in [_v3(0.2, 0, 0.1), _v3(-0.15, 0.1, -0.1), _v3(0, 0.15, 0.2)]:
        lobe = _sd_sphere(_op_translate(p, offset), 0.25)
        d = _op_smooth_union(d, lobe, 0.15)
    return (d - noise).astype(_F32)


def sd_flower(p: Vec3, petals: int = 5, petal_size: float = 0.1) -> np.ndarray:
    """Flower with petals arranged around a centre disc."""
    centre = _sd_sphere(p, petal_size * 0.4)
    d = centre
    for i in range(petals):
        angle = 2.0 * math.pi * i / petals
        offset = _v3(math.cos(angle) * petal_size, 0, math.sin(angle) * petal_size)
        petal = _sd_ellipsoid(
            _op_translate(p, offset),
            _v3(petal_size * 0.5, petal_size * 0.15, petal_size * 0.3)
        )
        d = _op_smooth_union(d, petal, petal_size * 0.3)
    return d.astype(_F32)


def sd_grass_blade(p: Vec3, height: float = 0.3, bend_amount: float = 0.5) -> np.ndarray:
    """Single grass blade: thin tapered capsule with bend."""
    q = op_bend(p, bend_amount)
    taper = 1.0 - _clamp(q[..., 1] / height, 0.0, 1.0) * 0.9
    r = 0.005 * taper
    d_xz = _length2(q[..., 0], q[..., 2]) - r
    d_y = np.abs(q[..., 1] - height * 0.5) - height * 0.5
    return np.maximum(d_xz, d_y).astype(_F32)


# ═══════════════════════════════════════════════════════════════════════════
# ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════

def sd_wall(p: Vec3, width: float, height: float, thickness: float) -> np.ndarray:
    """Flat wall centred at origin."""
    return _sd_box(p, _v3(width * 0.5, height * 0.5, thickness * 0.5))


def sd_arch(p: Vec3, width: float, height: float, thickness: float) -> np.ndarray:
    """Architectural arch: rectangular base + semicircular top."""
    half_w = width * 0.5
    rect_h = height - half_w
    rect = _sd_box(
        _op_translate(p, [0, rect_h * 0.5, 0]),
        _v3(half_w, rect_h * 0.5, thickness * 0.5)
    )
    arch_centre = _op_translate(p, [0, rect_h, 0])
    arch_ring = _sd_torus(
        np.stack([arch_centre[..., 0], arch_centre[..., 2], arch_centre[..., 1]], axis=-1),
        half_w, thickness * 0.5
    )
    arch_top = _op_intersect(arch_ring, -arch_centre[..., 1])
    return _op_union(rect, arch_top).astype(_F32)


def sd_column(p: Vec3, radius: float, height: float,
              style: str = "doric") -> np.ndarray:
    """Column with capital.  Styles: doric, ionic, corinthian (simplified)."""
    shaft = _sd_cylinder(p, radius, height * 0.5)
    cap_h = height * 0.05
    cap_r = radius * 1.3 if style == "doric" else radius * 1.5
    capital = _sd_cylinder(
        _op_translate(p, [0, height * 0.5, 0]),
        cap_r, cap_h
    )
    base = _sd_cylinder(
        _op_translate(p, [0, -height * 0.5, 0]),
        cap_r, cap_h
    )
    d = _op_union(shaft, capital)
    d = _op_union(d, base)
    if style == "corinthian":
        for angle_deg in range(0, 360, 45):
            a = math.radians(angle_deg)
            leaf_p = _op_translate(p, [
                math.cos(a) * radius * 0.9,
                height * 0.42,
                math.sin(a) * radius * 0.9
            ])
            leaf = _sd_ellipsoid(leaf_p, _v3(0.02, 0.06, 0.02))
            d = _op_smooth_union(d, leaf, 0.02)
    return d.astype(_F32)


def sd_stairs(p: Vec3, num_steps: int, step_width: float,
              step_height: float, step_depth: float) -> np.ndarray:
    """Staircase ascending along +Z, steps stacked along +Y."""
    d = np.full(p.shape[0], 1e10, dtype=_F32)
    for i in range(num_steps):
        sp = _op_translate(p, [0, -i * step_height, -i * step_depth])
        step = _sd_box(sp, _v3(step_width * 0.5, step_height * 0.5, step_depth * 0.5))
        d = _op_union(d, step)
    return d.astype(_F32)


def sd_window(p: Vec3, width: float, height: float) -> np.ndarray:
    """Window opening (use with op_subtract to cut from a wall)."""
    return _sd_box(p, _v3(width * 0.5, height * 0.5, 0.5))


def sd_door(p: Vec3, width: float, height: float) -> np.ndarray:
    """Door opening with rounded top frame."""
    rect = _sd_box(
        _op_translate(p, [0, height * 0.4, 0]),
        _v3(width * 0.5, height * 0.4, 0.5)
    )
    arch_p = _op_translate(p, [0, height * 0.8, 0])
    arch = _sd_cylinder(
        np.stack([arch_p[..., 0], arch_p[..., 2], arch_p[..., 1]], axis=-1),
        width * 0.5, 0.5
    )
    return _op_union(rect, arch).astype(_F32)


def sd_roof(p: Vec3, width: float, depth: float, pitch: float) -> np.ndarray:
    """Pitched roof (triangular prism)."""
    peak_h = width * 0.5 * math.tan(pitch)
    q = p.copy()
    q[..., 1] = q[..., 1] - peak_h
    d_sides = np.abs(q[..., 0]) * math.cos(pitch) + q[..., 1] * math.sin(pitch)
    d_depth = np.abs(q[..., 2]) - depth * 0.5
    d_bottom = -q[..., 1] - peak_h
    return np.maximum(np.maximum(d_sides, d_depth), d_bottom).astype(_F32)


def sd_building(p: Vec3, width: float = 4.0, depth: float = 4.0,
                floors: int = 3, style: str = "modern") -> np.ndarray:
    """Complete building: stacked floor boxes + roof."""
    floor_h = 3.0
    total_h = floors * floor_h
    body = _sd_box(
        _op_translate(p, [0, total_h * 0.5, 0]),
        _v3(width * 0.5, total_h * 0.5, depth * 0.5)
    )
    roof_p = _op_translate(p, [0, total_h, 0])
    if style == "modern":
        roof = _sd_box(roof_p, _v3(width * 0.55, 0.15, depth * 0.55))
    else:
        roof = sd_roof(roof_p, width, depth, math.radians(30))
    d = _op_union(body, roof)

    for fl in range(floors):
        for wx in (-1, 1):
            win_p = _op_translate(p, [
                wx * width * 0.25,
                (fl + 0.5) * floor_h + floor_h * 0.2,
                -depth * 0.5
            ])
            win = sd_window(win_p, width * 0.15, floor_h * 0.4)
            d = _op_subtract(d, win)

    return d.astype(_F32)


# ═══════════════════════════════════════════════════════════════════════════
# VEHICLES
# ═══════════════════════════════════════════════════════════════════════════

def sd_car_body(p: Vec3, style: str = "sedan") -> np.ndarray:
    """Car body shell.  Styles: sedan, suv, sports."""
    if style == "suv":
        lower = _v3(1.0, 0.35, 0.5)
        upper = _v3(0.7, 0.35, 0.45)
        upper_y = 0.65
    elif style == "sports":
        lower = _v3(1.1, 0.2, 0.5)
        upper = _v3(0.5, 0.2, 0.42)
        upper_y = 0.35
    else:
        lower = _v3(1.0, 0.25, 0.48)
        upper = _v3(0.6, 0.25, 0.44)
        upper_y = 0.45

    d_lower = _sd_box(p, lower)
    d_lower = op_round(d_lower, 0.05)
    d_upper = _sd_box(_op_translate(p, [0, upper_y, 0]), upper)
    d_upper = op_round(d_upper, 0.08)
    return _op_smooth_union(d_lower, d_upper, 0.1).astype(_F32)


def sd_wheel(p: Vec3, radius: float = 0.3, width: float = 0.15) -> np.ndarray:
    """Wheel with tire: torus + disc."""
    tire = _sd_torus(
        np.stack([p[..., 0], p[..., 2], p[..., 1]], axis=-1),
        radius - width * 0.3, width * 0.3
    )
    disc = _sd_cylinder(
        np.stack([p[..., 0], p[..., 2], p[..., 1]], axis=-1),
        radius * 0.7, width * 0.3
    )
    return _op_union(tire, disc).astype(_F32)


def sd_boat(p: Vec3, length: float = 3.0) -> np.ndarray:
    """Boat hull: elongated ellipsoid with flat deck cut."""
    hull = _sd_ellipsoid(p, _v3(length * 0.5, length * 0.12, length * 0.18))
    deck_cut = p[..., 1] - length * 0.05
    return _op_subtract(hull, -deck_cut).astype(_F32)


def sd_airplane(p: Vec3, wingspan: float = 5.0) -> np.ndarray:
    """Airplane silhouette: fuselage + wings + tail."""
    ws = float(wingspan)
    fuselage = _sd_capsule(p, _v3(-ws * 0.4, 0, 0), _v3(ws * 0.4, 0, 0), ws * 0.04)
    wing = _sd_box(
        _op_translate(p, [ws * 0.05, 0, 0]),
        _v3(ws * 0.08, ws * 0.005, ws * 0.5)
    )
    wing = op_round(wing, ws * 0.005)
    tail_h = _sd_box(
        _op_translate(p, [ws * 0.35, 0, 0]),
        _v3(ws * 0.04, ws * 0.003, ws * 0.12)
    )
    tail_v = _sd_box(
        _op_translate(p, [ws * 0.35, ws * 0.05, 0]),
        _v3(ws * 0.04, ws * 0.05, ws * 0.003)
    )
    d = _op_union(fuselage, wing)
    d = _op_union(d, tail_h)
    d = _op_union(d, tail_v)
    return d.astype(_F32)


# ═══════════════════════════════════════════════════════════════════════════
# FURNITURE
# ═══════════════════════════════════════════════════════════════════════════

def sd_table(p: Vec3, width: float = 1.2, depth: float = 0.6,
             height: float = 0.75) -> np.ndarray:
    """Table with four legs."""
    top_thick = 0.04
    top = _sd_box(
        _op_translate(p, [0, height, 0]),
        _v3(width * 0.5, top_thick * 0.5, depth * 0.5)
    )
    d = top
    leg_r = 0.025
    for sx in (-1, 1):
        for sz in (-1, 1):
            lp = _op_translate(p, [
                sx * (width * 0.5 - 0.05),
                height * 0.5,
                sz * (depth * 0.5 - 0.05)
            ])
            leg = _sd_cylinder(lp, leg_r, height * 0.5 - top_thick * 0.5)
            d = _op_union(d, leg)
    return d.astype(_F32)


def sd_chair(p: Vec3) -> np.ndarray:
    """Simple chair: seat + four legs + backrest."""
    seat_h = 0.45
    seat = _sd_box(
        _op_translate(p, [0, seat_h, 0]),
        _v3(0.22, 0.02, 0.22)
    )
    d = seat
    leg_r = 0.018
    for sx in (-1, 1):
        for sz in (-1, 1):
            lp = _op_translate(p, [sx * 0.18, seat_h * 0.5, sz * 0.18])
            d = _op_union(d, _sd_cylinder(lp, leg_r, seat_h * 0.5))
    back = _sd_box(
        _op_translate(p, [0, seat_h + 0.25, -0.2]),
        _v3(0.2, 0.25, 0.015)
    )
    d = _op_union(d, back)
    for sx in (-1, 1):
        bp = _op_translate(p, [sx * 0.18, seat_h + 0.15, -0.2])
        d = _op_union(d, _sd_cylinder(bp, leg_r, 0.15))
    return d.astype(_F32)


def sd_bookshelf(p: Vec3, width: float = 0.8, height: float = 1.8,
                 shelves: int = 5) -> np.ndarray:
    """Bookshelf with evenly spaced shelves."""
    thick = 0.02
    d = _sd_box(
        _op_translate(p, [0, height * 0.5, 0]),
        _v3(width * 0.5, height * 0.5, 0.15)
    )
    interior = _sd_box(
        _op_translate(p, [0, height * 0.5, -thick]),
        _v3(width * 0.5 - thick, height * 0.5 - thick, 0.15 - thick)
    )
    d = _op_subtract(d, interior)

    for i in range(shelves):
        sy = (i + 1) * height / (shelves + 1)
        shelf = _sd_box(
            _op_translate(p, [0, sy, 0]),
            _v3(width * 0.5 - thick, thick * 0.5, 0.15 - thick)
        )
        d = _op_union(d, shelf)
    return d.astype(_F32)


def sd_lamp(p: Vec3, style: str = "desk") -> np.ndarray:
    """Lamp.  Styles: desk, floor."""
    if style == "floor":
        pole_h = 1.5
        shade_y = pole_h + 0.15
    else:
        pole_h = 0.4
        shade_y = pole_h + 0.08

    base = _sd_cylinder(_op_translate(p, [0, 0.01, 0]), 0.1, 0.01)
    pole = _sd_cylinder(_op_translate(p, [0, pole_h * 0.5, 0]), 0.012, pole_h * 0.5)
    shade = _sd_ellipsoid(
        _op_translate(p, [0, shade_y, 0]),
        _v3(0.12, 0.08, 0.12)
    )
    shade = op_onion(shade, 0.005)
    shade = _op_intersect(shade, -(p[..., 1] - shade_y + 0.02))
    d = _op_union(base, pole)
    d = _op_union(d, shade)
    return d.astype(_F32)


# ═══════════════════════════════════════════════════════════════════════════
# TERRAIN / ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════

def sd_terrain(p: Vec3, scale: float = 1.0, octaves: int = 4,
               height: float = 1.0) -> np.ndarray:
    """Heightfield terrain using FBM noise."""
    h = _fbm(p * scale, octaves) * height
    return (p[..., 1] - h).astype(_F32)


def sd_mountain(p: Vec3, height: float = 5.0, steepness: float = 2.0) -> np.ndarray:
    """Mountain peak: cone + noise displacement."""
    r_base = height / steepness
    cone = sd_cone(p, r_base, height)
    noise = _fbm(p * 2.0, 5) * height * 0.15
    return (cone - noise).astype(_F32)


def sd_water_surface(p: Vec3, time: float = 0.0, amplitude: float = 0.05,
                     frequency: float = 2.0) -> np.ndarray:
    """Animated water plane with wave displacement."""
    wave = (np.sin(p[..., 0] * frequency + time) *
            np.cos(p[..., 2] * frequency * 0.7 + time * 1.3)) * amplitude
    wave += np.sin(p[..., 0] * frequency * 2.3 + p[..., 2] * 1.7 + time * 0.8) * amplitude * 0.3
    return (p[..., 1] - wave).astype(_F32)


def sd_rock(p: Vec3, size: float = 0.5, seed: float = 0.0) -> np.ndarray:
    """Natural rock: displaced sphere with noise."""
    offset = _v3(seed * 17.3, seed * 7.1, seed * 31.7)
    noise = _fbm(p * (3.0 / size) + offset, 4) * size * 0.3
    return (_sd_sphere(p, size) - noise).astype(_F32)


def sd_cloud_volume(p: Vec3, time: float = 0.0,
                    density_threshold: float = 0.4) -> np.ndarray:
    """Cloud density field (negative = inside cloud)."""
    time_offset = _v3(time * 0.1, 0, time * 0.05)
    density = _fbm(p + time_offset, 5)
    return (density_threshold - density).astype(_F32)


def sd_cliff(p: Vec3, height: float = 5.0, roughness: float = 0.3) -> np.ndarray:
    """Cliff face along the XY plane."""
    base = p[..., 2]
    noise = _fbm(
        np.stack([p[..., 0] * 0.5, p[..., 1] * 0.3, p[..., 2] * 0.5], axis=-1),
        5
    ) * roughness
    cliff_mask = _clamp(p[..., 1] / height, 0.0, 1.0)
    return (base - noise * cliff_mask).astype(_F32)


# ═══════════════════════════════════════════════════════════════════════════
# FOOD
# ═══════════════════════════════════════════════════════════════════════════

def sd_apple(p: Vec3) -> np.ndarray:
    """Apple shape: sphere with top indent and stem."""
    body = _sd_sphere(p, 0.12)
    indent = _sd_sphere(_op_translate(p, [0, 0.12, 0]), 0.04)
    d = _op_smooth_subtract(body, indent, 0.03)
    stem = _sd_capsule(p, _v3(0, 0.11, 0), _v3(0.005, 0.16, 0), 0.004)
    return _op_union(d, stem).astype(_F32)


def sd_cup(p: Vec3, radius: float = 0.05, height: float = 0.1) -> np.ndarray:
    """Cup / mug: hollow cylinder + handle."""
    outer = _sd_cylinder(p, radius, height * 0.5)
    inner = _sd_cylinder(_op_translate(p, [0, 0.005, 0]), radius * 0.88, height * 0.5)
    d = _op_subtract(outer, inner)
    d = np.maximum(d, -(p[..., 1] - height * 0.5))
    handle = _sd_torus(
        np.stack([p[..., 2] + radius * 0.9, p[..., 1], p[..., 0]], axis=-1),
        radius * 0.5, radius * 0.08
    )
    handle = np.maximum(handle, -p[..., 2] - radius * 0.3)
    return _op_union(d, handle).astype(_F32)


def sd_plate(p: Vec3, radius: float = 0.15) -> np.ndarray:
    """Plate / shallow bowl."""
    outer = _sd_cylinder(p, radius, 0.008)
    bowl = _sd_sphere(_op_translate(p, [0, -radius * 1.5, 0]), radius * 1.55)
    return _op_intersect(outer, bowl).astype(_F32)


# ═══════════════════════════════════════════════════════════════════════════
# TOOLS / OBJECTS
# ═══════════════════════════════════════════════════════════════════════════

def sd_book(p: Vec3, width: float = 0.15, height: float = 0.22,
            thickness: float = 0.03) -> np.ndarray:
    """Book with slightly rounded spine."""
    body = _sd_box(p, _v3(width * 0.5, height * 0.5, thickness * 0.5))
    spine = _sd_cylinder(
        np.stack([p[..., 2], p[..., 1], p[..., 0] + width * 0.5], axis=-1),
        thickness * 0.5, height * 0.5
    )
    return _op_union(body, spine).astype(_F32)


def sd_sphere_detailed(p: Vec3, radius: float = 0.5,
                       displacement_scale: float = 0.05) -> np.ndarray:
    """Displaced sphere for planets, balls, etc."""
    base = _sd_sphere(p, radius)
    disp = _fbm(p * (4.0 / radius), 4) * displacement_scale
    return (base - disp).astype(_F32)


def sd_gear(p: Vec3, teeth: int = 12, radius: float = 0.3,
            thickness: float = 0.05) -> np.ndarray:
    """Gear / cog with teeth."""
    angle = np.arctan2(p[..., 2], p[..., 0])
    tooth_wave = np.cos(angle * teeth) * radius * 0.12
    r_xz = _length2(p[..., 0], p[..., 2])
    d_xz = r_xz - (radius + tooth_wave)
    d_y = np.abs(p[..., 1]) - thickness * 0.5
    outside = _length2(np.maximum(d_xz, 0.0), np.maximum(d_y, 0.0))
    inside = np.minimum(np.maximum(d_xz, d_y), 0.0)
    gear = outside + inside
    axle = _sd_cylinder(p, radius * 0.2, thickness * 0.5 + 0.01)
    return _op_subtract(gear, axle).astype(_F32)


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    # Geometric extended
    "sd_cone", "sd_ellipsoid", "sd_hexagonal_prism", "sd_triangular_prism",
    "sd_pyramid", "sd_octahedron", "sd_ring", "sd_arc", "sd_bezier", "sd_link",
    # Operators extended
    "op_rotate_x", "op_rotate_z", "op_rotate_xyz",
    "op_twist", "op_bend", "op_elongate", "op_round", "op_onion",
    "op_repeat", "op_mirror_x", "op_smooth_subtract", "op_smooth_intersect",
    # Human / organic
    "sd_human_torso", "sd_human_head", "sd_human_limb", "sd_human_hand",
    "sd_human_figure", "sd_animal_body", "sd_fish", "sd_bird",
    # Vegetation
    "sd_tree_trunk", "sd_tree_canopy", "sd_tree", "sd_bush",
    "sd_flower", "sd_grass_blade",
    # Architecture
    "sd_wall", "sd_arch", "sd_column", "sd_stairs", "sd_window", "sd_door",
    "sd_roof", "sd_building",
    # Vehicles
    "sd_car_body", "sd_wheel", "sd_boat", "sd_airplane",
    # Furniture
    "sd_table", "sd_chair", "sd_bookshelf", "sd_lamp",
    # Terrain / environment
    "sd_terrain", "sd_mountain", "sd_water_surface", "sd_rock",
    "sd_cloud_volume", "sd_cliff",
    # Food
    "sd_apple", "sd_cup", "sd_plate",
    # Tools / objects
    "sd_book", "sd_sphere_detailed", "sd_gear",
]
