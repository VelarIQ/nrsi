"""
NRS Creative Vision Engine
===========================

Deep visual intelligence that powers every aspect of media generation.
Not keyword matching — this is a full creative brain for visual arts.

Architecture
------------
  NLP Prompt
      │
      ▼
  ┌── Visual DNA ─────────────────────────────────────────────┐
  │  50+ dimensional style vector encoding every visual       │
  │  property: color temperature, contrast, bokeh, grain,     │
  │  lens distortion, atmospheric haze, motion blur,          │
  │  drama, elegance, intimacy, dynamism, …                   │
  └───────────┬───────────────────────────────────────────────┘
              │
              ├── Composition Intelligence                      
              │   Rule of thirds, golden ratio, leading lines,  
              │   visual weight, foreground interest, depth      
              │
              ├── Color Theory Engine                           
              │   Complementary, analogous, triadic, split      
              │   palettes. Emotional color mapping.             
              │
              ├── Lens Simulation                               
              │   Focal length → FOV, DOF, bokeh, distortion.   
              │   Named lens profiles (Hasselblad, Leica, etc.)  
              │
              ├── Time-of-Day Intelligence                      
              │   Every hour mapped to lighting params.          
              │   Golden, blue, midnight, high noon, overcast.   
              │
              ├── Weather System                                
              │   Fog, rain, snow, storm, clouds, haze, dust.   
              │   Each modifies lighting, color, atmosphere.     
              │
              ├── Material Intelligence                         
              │   Metal, glass, fabric, skin, water, wood,      
              │   concrete, carbon fiber. Reflection/refraction. 
              │
              ├── Emotional Mapping                             
              │   Mood → color temperature, contrast,           
              │   saturation, composition, lens choice.          
              │
              ├── Scene Architecture                            
              │   Foreground / midground / background layers.   
              │   Depth relationships and focal priorities.      
              │
              ├── Style Evolution (genetic)                     
              │   Mutation, crossover, selection.               
              │   Breeds new styles from validated parents.      
              │
              ├── Negative Prompt Learning                      
              │   Learns what NOT to do from rejected outputs.  
              │   Builds domain-specific anti-patterns.          
              │
              └── Adaptive Guidance                             
                  Learns optimal guidance_scale, steps,         
                  and CFG per style/subject from successes.     

Integration with NRSI Brain
---------------------------
  All creative knowledge feeds into:
    VLT L3: Persistent style patterns
    VLT L4: Validated ground truth (archival-quality presets)
    PVS-4:  Prompt → visual DNA instant matching
    Tuition: Feedback → corrected creative parameters
    Mesh:    Every creative output validated before storage
    Creative Lobe: Novel style combinations and analogies
"""

from __future__ import annotations

import hashlib
import math
import random
import time
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# VISUAL DNA — the genome of every visual style
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VisualDNA:
    """
    50+ dimensional vector encoding every visual property of an image.
    Each dimension is 0.0–1.0. Together they form the unique 'genetic code'
    of a visual style.

    Two images with similar VisualDNA will LOOK similar even if their
    text prompts are completely different. This is how CrossDomainTransfer
    works for visual content — strip the subject, keep the DNA.
    """

    # ── Color ──────────────────────────────────────────────
    color_temperature: float = 0.5        # 0=cool blue, 1=warm golden
    saturation: float = 0.6               # 0=desaturated/mono, 1=vivid
    vibrance: float = 0.5                 # selective saturation on muted tones
    hue_shift: float = 0.5                # 0=shifted cool, 0.5=neutral, 1=shifted warm
    color_harmony: float = 0.5            # 0=monochrome, 1=rich palette
    shadow_color_temp: float = 0.4        # shadow tint (cool shadows = classic cinema)
    highlight_color_temp: float = 0.6     # highlight tint (warm highlights = film look)

    # ── Exposure & Contrast ────────────────────────────────
    brightness: float = 0.5               # 0=dark, 1=bright
    contrast: float = 0.5                 # 0=flat, 1=extreme
    dynamic_range: float = 0.7            # 0=clipped, 1=full HDR
    black_level: float = 0.1              # crushed blacks for mood
    white_level: float = 0.9              # blown highlights for atmosphere
    midtone_contrast: float = 0.5         # clarity/punch in midtones
    exposure_compensation: float = 0.5    # 0=underexposed, 1=overexposed

    # ── Focus & Depth ──────────────────────────────────────
    depth_of_field: float = 0.5           # 0=deep (everything sharp), 1=ultra shallow
    bokeh_intensity: float = 0.3          # 0=none, 1=heavy bokeh circles
    bokeh_character: float = 0.5          # 0=busy/nervous, 1=smooth/creamy
    focus_distance: float = 0.5           # 0=near, 1=infinity
    tilt_shift: float = 0.0              # 0=normal, 1=miniature effect

    # ── Texture & Detail ───────────────────────────────────
    sharpness: float = 0.7               # 0=soft/dreamy, 1=razor sharp
    grain_amount: float = 0.1            # 0=clean digital, 1=heavy film grain
    grain_size: float = 0.3              # 0=fine grain, 1=coarse grain
    noise_character: float = 0.5         # 0=digital noise, 1=organic film grain
    micro_contrast: float = 0.6          # fine texture detail
    clarity: float = 0.6                 # local contrast enhancement
    texture_detail: float = 0.7          # pore-level, fabric-weave detail

    # ── Lens Character ─────────────────────────────────────
    focal_length: float = 0.5            # 0=ultra wide (14mm), 1=super tele (400mm)
    lens_distortion: float = 0.1         # 0=rectilinear, 1=fisheye
    chromatic_aberration: float = 0.05   # 0=none, 1=heavy color fringing
    vignette: float = 0.2               # 0=none, 1=heavy darkened corners
    lens_flare: float = 0.0             # 0=none, 1=prominent anamorphic flare
    anamorphic: float = 0.0             # 0=spherical, 1=full anamorphic squeeze

    # ── Atmosphere & Environment ───────────────────────────
    atmospheric_haze: float = 0.1        # 0=crystal clear, 1=heavy haze/fog
    volumetric_light: float = 0.2        # 0=none, 1=god rays everywhere
    dust_particles: float = 0.0          # 0=clean, 1=visible dust/particles
    rain_intensity: float = 0.0          # 0=dry, 1=heavy downpour
    snow_intensity: float = 0.0          # 0=none, 1=blizzard
    cloud_drama: float = 0.3            # 0=clear sky, 1=dramatic cloudscape
    fog_density: float = 0.0            # 0=none, 1=pea soup fog

    # ── Light Character ────────────────────────────────────
    light_direction: float = 0.5         # 0=backlit, 0.5=side, 1=front
    light_hardness: float = 0.5          # 0=ultra soft, 1=hard direct
    rim_light: float = 0.2              # 0=none, 1=strong edge light
    fill_ratio: float = 0.5             # 0=no fill (deep shadows), 1=full fill
    specular_intensity: float = 0.4      # 0=matte, 1=mirror reflections
    light_color_variation: float = 0.3   # 0=uniform, 1=mixed color sources

    # ── Composition & Framing ──────────────────────────────
    symmetry: float = 0.3               # 0=asymmetric, 1=perfectly symmetric
    rule_of_thirds: float = 0.7         # 0=centered, 1=strong thirds placement
    leading_lines: float = 0.3          # 0=no lines, 1=strong leading lines
    negative_space: float = 0.4         # 0=filled frame, 1=lots of negative space
    foreground_interest: float = 0.3    # 0=no foreground, 1=strong foreground element
    depth_layers: float = 0.5           # 0=flat, 1=distinct FG/MG/BG separation
    visual_weight_balance: float = 0.5  # 0=bottom heavy, 1=top heavy

    # ── Motion & Energy ────────────────────────────────────
    motion_blur: float = 0.0            # 0=frozen, 1=heavy motion streaks
    dynamism: float = 0.5               # 0=static/calm, 1=energetic/explosive
    camera_movement: float = 0.0        # 0=locked, 1=aggressive movement

    # ── Mood & Tone ────────────────────────────────────────
    drama: float = 0.5                  # 0=neutral, 1=maximum drama
    elegance: float = 0.5               # 0=raw/gritty, 1=refined/polished
    intimacy: float = 0.5               # 0=distant/epic, 1=close/personal
    mystery: float = 0.5               # 0=clear/obvious, 1=hidden/suggestive
    nostalgia: float = 0.0             # 0=modern/clean, 1=vintage/retro
    warmth: float = 0.5                # 0=cold/clinical, 1=warm/inviting
    tension: float = 0.3               # 0=relaxed, 1=high tension

    def to_dict(self) -> Dict[str, float]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    def to_vector(self) -> List[float]:
        return [getattr(self, k) for k in self.__dataclass_fields__]

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> VisualDNA:
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    def distance(self, other: VisualDNA) -> float:
        """Euclidean distance between two DNA vectors (lower = more similar)."""
        a, b = self.to_vector(), other.to_vector()
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def similarity(self, other: VisualDNA) -> float:
        """Cosine similarity (1.0 = identical, 0.0 = orthogonal)."""
        a, b = self.to_vector(), other.to_vector()
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x ** 2 for x in a)) or 1e-9
        mag_b = math.sqrt(sum(x ** 2 for x in b)) or 1e-9
        return dot / (mag_a * mag_b)

    def blend(self, other: VisualDNA, alpha: float = 0.5) -> VisualDNA:
        """Interpolate between two DNA vectors."""
        d = {}
        for k in self.__dataclass_fields__:
            va = getattr(self, k)
            vb = getattr(other, k)
            d[k] = va * (1 - alpha) + vb * alpha
        return VisualDNA.from_dict(d)


# ═══════════════════════════════════════════════════════════════════════════════
# STYLE DNA LIBRARY — every named style has a genetic code
# ═══════════════════════════════════════════════════════════════════════════════

STYLE_DNA: Dict[str, VisualDNA] = {
    "photorealistic": VisualDNA(
        color_temperature=0.5, saturation=0.5, contrast=0.5,
        sharpness=0.8, grain_amount=0.0, depth_of_field=0.4,
        drama=0.3, elegance=0.6, clarity=0.8, texture_detail=0.9,
        dynamic_range=0.8,
    ),
    "cinematic": VisualDNA(
        color_temperature=0.55, saturation=0.45, contrast=0.65,
        sharpness=0.6, grain_amount=0.25, depth_of_field=0.6,
        drama=0.75, elegance=0.7, anamorphic=0.7, lens_flare=0.3,
        vignette=0.35, shadow_color_temp=0.3, black_level=0.15,
        dynamic_range=0.85, volumetric_light=0.4, nostalgia=0.2,
        bokeh_character=0.8,
    ),
    "hyperreal": VisualDNA(
        color_temperature=0.5, saturation=0.7, contrast=0.6,
        sharpness=0.95, grain_amount=0.0, depth_of_field=0.3,
        drama=0.5, elegance=0.8, clarity=0.95, texture_detail=1.0,
        micro_contrast=0.9, specular_intensity=0.7, dynamic_range=0.95,
    ),
    "editorial": VisualDNA(
        color_temperature=0.45, saturation=0.4, contrast=0.55,
        sharpness=0.7, grain_amount=0.05, depth_of_field=0.55,
        drama=0.4, elegance=0.9, intimacy=0.6, negative_space=0.6,
        symmetry=0.5, light_hardness=0.4, fill_ratio=0.6,
    ),
    "dramatic": VisualDNA(
        color_temperature=0.4, saturation=0.5, contrast=0.85,
        sharpness=0.7, grain_amount=0.1, depth_of_field=0.5,
        drama=1.0, elegance=0.5, volumetric_light=0.6,
        rim_light=0.7, light_direction=0.2, fill_ratio=0.2,
        black_level=0.2, tension=0.8, shadow_color_temp=0.3,
    ),
    "moody": VisualDNA(
        color_temperature=0.35, saturation=0.35, contrast=0.7,
        brightness=0.35, sharpness=0.55, grain_amount=0.2,
        depth_of_field=0.5, drama=0.7, mystery=0.8, tension=0.6,
        vignette=0.4, fog_density=0.2, atmospheric_haze=0.3,
        nostalgia=0.3, warmth=0.3,
    ),
    "noir": VisualDNA(
        color_temperature=0.3, saturation=0.1, contrast=0.9,
        brightness=0.3, sharpness=0.6, grain_amount=0.3,
        drama=0.9, mystery=1.0, tension=0.9, vignette=0.5,
        light_hardness=0.8, fill_ratio=0.1, rim_light=0.5,
        shadow_color_temp=0.2, black_level=0.25, nostalgia=0.5,
    ),
    "golden_hour": VisualDNA(
        color_temperature=0.85, saturation=0.65, contrast=0.5,
        brightness=0.6, sharpness=0.6, grain_amount=0.05,
        drama=0.6, warmth=0.95, volumetric_light=0.5,
        lens_flare=0.3, atmospheric_haze=0.2, light_direction=0.3,
        highlight_color_temp=0.85, shadow_color_temp=0.5,
        elegance=0.7,
    ),
    "automotive": VisualDNA(
        color_temperature=0.5, saturation=0.55, contrast=0.6,
        sharpness=0.85, grain_amount=0.0, depth_of_field=0.4,
        drama=0.65, elegance=0.8, specular_intensity=0.8,
        micro_contrast=0.8, clarity=0.85, dynamism=0.6,
        leading_lines=0.5, depth_layers=0.7, rim_light=0.4,
        light_hardness=0.5,
    ),
    "aerial": VisualDNA(
        color_temperature=0.5, saturation=0.55, contrast=0.5,
        sharpness=0.7, depth_of_field=0.1, drama=0.5,
        atmospheric_haze=0.3, cloud_drama=0.5,
        negative_space=0.6, depth_layers=0.8,
        focal_length=0.2, brightness=0.6,
    ),
    "portrait": VisualDNA(
        color_temperature=0.55, saturation=0.45, contrast=0.45,
        sharpness=0.65, depth_of_field=0.8, bokeh_intensity=0.7,
        bokeh_character=0.9, drama=0.4, elegance=0.7, intimacy=0.9,
        focal_length=0.6, fill_ratio=0.55, light_hardness=0.35,
        texture_detail=0.8, warmth=0.6,
    ),
    "landscape": VisualDNA(
        color_temperature=0.5, saturation=0.6, contrast=0.55,
        sharpness=0.8, depth_of_field=0.1, drama=0.6,
        clarity=0.8, atmospheric_haze=0.2, cloud_drama=0.6,
        depth_layers=0.9, foreground_interest=0.7,
        dynamic_range=0.9, focal_length=0.2,
    ),
    "architectural": VisualDNA(
        color_temperature=0.45, saturation=0.4, contrast=0.55,
        sharpness=0.85, depth_of_field=0.15, drama=0.4,
        elegance=0.8, symmetry=0.7, leading_lines=0.8,
        lens_distortion=0.0, clarity=0.85, negative_space=0.5,
        focal_length=0.25,
    ),
    "product": VisualDNA(
        color_temperature=0.5, saturation=0.5, contrast=0.5,
        sharpness=0.9, depth_of_field=0.4, drama=0.2,
        elegance=0.85, clarity=0.9, specular_intensity=0.6,
        light_hardness=0.4, fill_ratio=0.7, negative_space=0.7,
        texture_detail=0.9,
    ),
    "fashion": VisualDNA(
        color_temperature=0.48, saturation=0.5, contrast=0.55,
        sharpness=0.7, depth_of_field=0.5, drama=0.5,
        elegance=0.95, intimacy=0.7, grain_amount=0.08,
        negative_space=0.5, bokeh_character=0.7,
        light_hardness=0.4, fill_ratio=0.5,
    ),
    "abstract": VisualDNA(
        color_temperature=0.5, saturation=0.7, contrast=0.7,
        sharpness=0.4, depth_of_field=0.3, drama=0.6,
        dynamism=0.7, motion_blur=0.3, chromatic_aberration=0.2,
        color_harmony=0.8, mystery=0.7,
    ),
    "vibrant": VisualDNA(
        color_temperature=0.55, saturation=0.9, vibrance=0.9,
        contrast=0.6, brightness=0.6, sharpness=0.7,
        drama=0.5, dynamism=0.6, color_harmony=0.9,
        clarity=0.7, warmth=0.6,
    ),
    "studio": VisualDNA(
        color_temperature=0.5, saturation=0.5, contrast=0.5,
        sharpness=0.85, depth_of_field=0.3, drama=0.3,
        elegance=0.8, light_hardness=0.45, fill_ratio=0.6,
        specular_intensity=0.5, clarity=0.8, texture_detail=0.85,
        negative_space=0.6,
    ),
    "macro": VisualDNA(
        color_temperature=0.5, saturation=0.6, contrast=0.5,
        sharpness=0.95, depth_of_field=0.95, bokeh_intensity=0.9,
        bokeh_character=0.8, drama=0.4, intimacy=1.0,
        texture_detail=1.0, micro_contrast=0.9, focal_length=0.5,
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# LENS PROFILES — real-world camera lens simulation
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LensProfile:
    """Simulates the optical character of a real lens."""
    name: str
    brand: str
    focal_length_mm: int
    max_aperture: float
    field_of_view_deg: float
    bokeh_quality: float           # 0=busy, 1=creamy
    distortion: float              # barrel (+) or pincushion (-)
    chromatic_aberration: float
    vignette_wide_open: float
    sharpness_center: float
    sharpness_edge: float
    flare_resistance: float
    character: str                 # warm, clinical, vintage, creamy, etc.
    prompt_hint: str


LENS_LIBRARY: Dict[str, LensProfile] = {
    "hasselblad_xcd_90": LensProfile(
        name="XCD 90mm f/3.2", brand="Hasselblad", focal_length_mm=90,
        max_aperture=3.2, field_of_view_deg=26, bokeh_quality=0.95,
        distortion=0.01, chromatic_aberration=0.02, vignette_wide_open=0.15,
        sharpness_center=0.98, sharpness_edge=0.92, flare_resistance=0.9,
        character="clinical precision with medium format creamy rendering",
        prompt_hint="shot on Hasselblad X2D, XCD 90mm f/3.2, medium format, "
                    "incredible detail, creamy bokeh",
    ),
    "leica_noctilux_50": LensProfile(
        name="Noctilux-M 50mm f/0.95", brand="Leica", focal_length_mm=50,
        max_aperture=0.95, field_of_view_deg=47, bokeh_quality=0.98,
        distortion=0.02, chromatic_aberration=0.05, vignette_wide_open=0.35,
        sharpness_center=0.88, sharpness_edge=0.72, flare_resistance=0.6,
        character="dreamy wide-open glow, legendary bokeh, painterly rendering",
        prompt_hint="shot on Leica M11, Noctilux 50mm f/0.95 wide open, "
                    "ethereal glow, impossibly shallow depth of field",
    ),
    "sony_gm_85": LensProfile(
        name="FE 85mm f/1.4 GM", brand="Sony", focal_length_mm=85,
        max_aperture=1.4, field_of_view_deg=29, bokeh_quality=0.92,
        distortion=0.01, chromatic_aberration=0.03, vignette_wide_open=0.2,
        sharpness_center=0.95, sharpness_edge=0.88, flare_resistance=0.85,
        character="razor sharp with silky smooth bokeh",
        prompt_hint="shot on Sony A7RV, 85mm f/1.4 GM, tack sharp subject, "
                    "butter smooth bokeh",
    ),
    "canon_rf_28_70": LensProfile(
        name="RF 28-70mm f/2L", brand="Canon", focal_length_mm=50,
        max_aperture=2.0, field_of_view_deg=47, bokeh_quality=0.88,
        distortion=0.03, chromatic_aberration=0.04, vignette_wide_open=0.25,
        sharpness_center=0.93, sharpness_edge=0.85, flare_resistance=0.8,
        character="versatile luxury zoom, creamy and sharp",
        prompt_hint="shot on Canon R5, RF 28-70mm f/2L, professional quality, "
                    "rich color rendering",
    ),
    "zeiss_otus_55": LensProfile(
        name="Otus 55mm f/1.4", brand="Zeiss", focal_length_mm=55,
        max_aperture=1.4, field_of_view_deg=43, bokeh_quality=0.94,
        distortion=0.005, chromatic_aberration=0.01, vignette_wide_open=0.18,
        sharpness_center=0.99, sharpness_edge=0.95, flare_resistance=0.92,
        character="clinical perfection, reference-grade rendering",
        prompt_hint="shot with Zeiss Otus 55mm f/1.4, reference-grade optics, "
                    "no optical compromises, surgical sharpness",
    ),
    "arri_master_prime_35": LensProfile(
        name="Master Prime 35mm T1.3", brand="ARRI", focal_length_mm=35,
        max_aperture=1.3, field_of_view_deg=63, bokeh_quality=0.9,
        distortion=0.02, chromatic_aberration=0.02, vignette_wide_open=0.12,
        sharpness_center=0.96, sharpness_edge=0.92, flare_resistance=0.95,
        character="cinema standard, controlled rendering, beautiful flare",
        prompt_hint="35mm film, ARRI Master Prime, cinematic rendering, "
                    "beautiful optical character",
    ),
    "cooke_s4_75": LensProfile(
        name="S4/i 75mm T2", brand="Cooke", focal_length_mm=75,
        max_aperture=2.0, field_of_view_deg=32, bokeh_quality=0.93,
        distortion=0.01, chromatic_aberration=0.03, vignette_wide_open=0.15,
        sharpness_center=0.92, sharpness_edge=0.87, flare_resistance=0.8,
        character="warm, organic, the famous Cooke Look",
        prompt_hint="shot on ARRI Alexa with Cooke S4/i 75mm, "
                    "warm organic rendering, the Cooke Look",
    ),
    "sigma_art_35": LensProfile(
        name="35mm f/1.4 DG DN Art", brand="Sigma", focal_length_mm=35,
        max_aperture=1.4, field_of_view_deg=63, bokeh_quality=0.85,
        distortion=0.02, chromatic_aberration=0.03, vignette_wide_open=0.2,
        sharpness_center=0.94, sharpness_edge=0.88, flare_resistance=0.82,
        character="modern sharp, versatile wide-angle with character",
        prompt_hint="shot with Sigma 35mm f/1.4 Art, sharp modern rendering, "
                    "versatile wide-angle perspective",
    ),
    "nikon_z_14_24": LensProfile(
        name="NIKKOR Z 14-24mm f/2.8 S", brand="Nikon", focal_length_mm=14,
        max_aperture=2.8, field_of_view_deg=114, bokeh_quality=0.7,
        distortion=0.08, chromatic_aberration=0.04, vignette_wide_open=0.3,
        sharpness_center=0.93, sharpness_edge=0.85, flare_resistance=0.88,
        character="ultra wide, dramatic perspective, sharp corner-to-corner",
        prompt_hint="shot on Nikon Z9, NIKKOR Z 14-24mm f/2.8, "
                    "ultra wide angle, dramatic perspective",
    ),
    "phase_one_150mp": LensProfile(
        name="Schneider 80mm f/2.8 LS", brand="Phase One", focal_length_mm=80,
        max_aperture=2.8, field_of_view_deg=54, bokeh_quality=0.9,
        distortion=0.005, chromatic_aberration=0.01, vignette_wide_open=0.1,
        sharpness_center=0.99, sharpness_edge=0.96, flare_resistance=0.93,
        character="150 megapixel resolution, absolute technical perfection",
        prompt_hint="shot on Phase One IQ4 150MP, Schneider 80mm, "
                    "150 megapixel resolution, pore-level detail",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# COLOR THEORY ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class ColorHarmony(Enum):
    COMPLEMENTARY = "complementary"
    ANALOGOUS = "analogous"
    TRIADIC = "triadic"
    SPLIT_COMPLEMENTARY = "split_complementary"
    TETRADIC = "tetradic"
    MONOCHROMATIC = "monochromatic"


@dataclass
class ColorPalette:
    """A mood-driven color palette."""
    name: str
    primary: str            # hex color
    secondary: str
    accent: str
    shadow_tint: str
    highlight_tint: str
    harmony: ColorHarmony
    prompt_fragment: str    # text to inject into prompt


MOOD_PALETTES: Dict[str, ColorPalette] = {
    "dramatic": ColorPalette(
        "Dramatic Shadows", "#1a1a2e", "#16213e", "#e94560", "#0a0a1a",
        "#ff6b6b", ColorHarmony.COMPLEMENTARY,
        "deep shadows, rich crimson accents, dark blue tones, "
        "dramatic color contrast",
    ),
    "serene": ColorPalette(
        "Serene Waters", "#a8d8ea", "#aa96da", "#fcbad3", "#dfe6e9",
        "#ffeaa7", ColorHarmony.ANALOGOUS,
        "soft pastel tones, tranquil blues and lavenders, "
        "gentle color palette, soothing atmosphere",
    ),
    "luxurious": ColorPalette(
        "Gold Standard", "#2c2c54", "#d4af37", "#c0392b", "#1a1a2e",
        "#f5e6cc", ColorHarmony.SPLIT_COMPLEMENTARY,
        "rich gold accents, deep navy, luxurious color palette, "
        "premium materials, opulent atmosphere",
    ),
    "energetic": ColorPalette(
        "Electric Surge", "#ff6b35", "#f7c59f", "#004e89", "#1a659e",
        "#ff9f1c", ColorHarmony.TRIADIC,
        "vibrant orange energy, electric blue contrast, "
        "high-saturation dynamic colors",
    ),
    "dark": ColorPalette(
        "Midnight Noir", "#0d0d0d", "#1a1a2e", "#4a0e0e", "#050505",
        "#2d2d2d", ColorHarmony.MONOCHROMATIC,
        "near-black tones, minimal color, deep shadows, "
        "noir atmosphere, monochromatic darkness",
    ),
    "romantic": ColorPalette(
        "Sunset Blush", "#e8a87c", "#d63384", "#85586f", "#3c1642",
        "#ffd1dc", ColorHarmony.ANALOGOUS,
        "warm blush tones, soft pinks and corals, "
        "romantic golden warmth, intimate color palette",
    ),
    "futuristic": ColorPalette(
        "Neon Grid", "#0d0221", "#0abde3", "#ee5a24", "#5f27cd",
        "#00d2d3", ColorHarmony.TETRADIC,
        "neon cyan and magenta, dark cyberpunk backdrop, "
        "fluorescent accents, futuristic color scheme",
    ),
    "neutral": ColorPalette(
        "Clean Neutral", "#f5f5f5", "#333333", "#888888", "#222222",
        "#ffffff", ColorHarmony.MONOCHROMATIC,
        "clean neutral tones, balanced grays, "
        "professional color balance",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# TIME-OF-DAY INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TimeOfDay:
    """Lighting parameters for a specific time of day."""
    name: str
    hour_range: Tuple[int, int]
    color_temperature: float
    brightness: float
    contrast: float
    shadow_color_temp: float
    highlight_color_temp: float
    atmospheric_haze: float
    volumetric_light: float
    drama: float
    prompt_fragment: str


TIME_PRESETS: Dict[str, TimeOfDay] = {
    "predawn": TimeOfDay(
        "Pre-Dawn", (4, 5), color_temperature=0.3, brightness=0.15,
        contrast=0.4, shadow_color_temp=0.25, highlight_color_temp=0.4,
        atmospheric_haze=0.4, volumetric_light=0.1, drama=0.5,
        prompt_fragment="pre-dawn twilight, deep blue sky, "
                        "first hints of light on the horizon",
    ),
    "blue_hour": TimeOfDay(
        "Blue Hour", (5, 6), color_temperature=0.25, brightness=0.25,
        contrast=0.45, shadow_color_temp=0.2, highlight_color_temp=0.35,
        atmospheric_haze=0.3, volumetric_light=0.15, drama=0.65,
        prompt_fragment="blue hour, deep cobalt sky, "
                        "city lights mixing with natural twilight",
    ),
    "golden_hour_morning": TimeOfDay(
        "Golden Hour (Morning)", (6, 8), color_temperature=0.8, brightness=0.5,
        contrast=0.5, shadow_color_temp=0.45, highlight_color_temp=0.85,
        atmospheric_haze=0.2, volumetric_light=0.5, drama=0.6,
        prompt_fragment="morning golden hour, warm golden light, "
                        "long shadows, magical atmosphere",
    ),
    "morning": TimeOfDay(
        "Late Morning", (8, 11), color_temperature=0.55, brightness=0.65,
        contrast=0.5, shadow_color_temp=0.4, highlight_color_temp=0.6,
        atmospheric_haze=0.1, volumetric_light=0.2, drama=0.3,
        prompt_fragment="late morning light, clean and bright, "
                        "pleasant natural daylight",
    ),
    "high_noon": TimeOfDay(
        "High Noon", (11, 14), color_temperature=0.5, brightness=0.8,
        contrast=0.7, shadow_color_temp=0.35, highlight_color_temp=0.55,
        atmospheric_haze=0.05, volumetric_light=0.1, drama=0.4,
        prompt_fragment="midday sun, harsh overhead lighting, "
                        "strong shadows, high contrast",
    ),
    "afternoon": TimeOfDay(
        "Afternoon", (14, 16), color_temperature=0.55, brightness=0.6,
        contrast=0.5, shadow_color_temp=0.4, highlight_color_temp=0.6,
        atmospheric_haze=0.1, volumetric_light=0.25, drama=0.35,
        prompt_fragment="afternoon light, warm and directional, "
                        "pleasant shadows",
    ),
    "golden_hour_evening": TimeOfDay(
        "Golden Hour (Evening)", (16, 18), color_temperature=0.85,
        brightness=0.45, contrast=0.55, shadow_color_temp=0.5,
        highlight_color_temp=0.9, atmospheric_haze=0.25,
        volumetric_light=0.6, drama=0.7,
        prompt_fragment="evening golden hour, rich warm light, "
                        "dramatic long shadows, golden everything",
    ),
    "sunset": TimeOfDay(
        "Sunset", (18, 19), color_temperature=0.9, brightness=0.35,
        contrast=0.6, shadow_color_temp=0.4, highlight_color_temp=0.95,
        atmospheric_haze=0.3, volumetric_light=0.7, drama=0.85,
        prompt_fragment="dramatic sunset, sky on fire, orange and purple, "
                        "silhouettes, spectacular clouds",
    ),
    "blue_hour_evening": TimeOfDay(
        "Blue Hour (Evening)", (19, 20), color_temperature=0.25,
        brightness=0.2, contrast=0.5, shadow_color_temp=0.2,
        highlight_color_temp=0.3, atmospheric_haze=0.35,
        volumetric_light=0.2, drama=0.7,
        prompt_fragment="evening blue hour, deep blue atmosphere, "
                        "warm artificial lights contrasting cool sky",
    ),
    "night": TimeOfDay(
        "Night", (20, 4), color_temperature=0.3, brightness=0.1,
        contrast=0.8, shadow_color_temp=0.2, highlight_color_temp=0.4,
        atmospheric_haze=0.2, volumetric_light=0.15, drama=0.6,
        prompt_fragment="nighttime, moonlight, city lights, "
                        "deep shadows, artificial light sources",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# WEATHER SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WeatherProfile:
    """How weather conditions affect visual parameters."""
    name: str
    fog_density: float
    rain_intensity: float
    snow_intensity: float
    cloud_drama: float
    atmospheric_haze: float
    brightness_mod: float          # multiplier on base brightness
    contrast_mod: float
    saturation_mod: float
    color_temperature_shift: float # additive shift
    prompt_fragment: str


WEATHER_PROFILES: Dict[str, WeatherProfile] = {
    "clear": WeatherProfile(
        "Clear Sky", 0.0, 0.0, 0.0, 0.1, 0.05, 1.0, 1.0, 1.0, 0.0,
        "clear sky, perfect visibility",
    ),
    "overcast": WeatherProfile(
        "Overcast", 0.05, 0.0, 0.0, 0.3, 0.15, 0.85, 0.7, 0.85, -0.05,
        "overcast sky, soft diffused light, even illumination, "
        "no harsh shadows",
    ),
    "cloudy_dramatic": WeatherProfile(
        "Dramatic Clouds", 0.0, 0.0, 0.0, 0.9, 0.1, 0.9, 1.1, 0.95, 0.0,
        "dramatic cloudscape, towering cumulonimbus, "
        "shafts of light breaking through clouds, epic sky",
    ),
    "fog": WeatherProfile(
        "Dense Fog", 0.8, 0.0, 0.0, 0.2, 0.7, 0.7, 0.4, 0.6, -0.05,
        "dense fog, mysterious atmosphere, silhouettes fading, "
        "limited visibility, ethereal",
    ),
    "mist": WeatherProfile(
        "Light Mist", 0.3, 0.0, 0.0, 0.2, 0.35, 0.85, 0.6, 0.8, -0.03,
        "light morning mist, soft atmosphere, gentle haze, "
        "dreamlike quality",
    ),
    "rain_light": WeatherProfile(
        "Light Rain", 0.1, 0.3, 0.0, 0.5, 0.2, 0.75, 0.8, 0.8, -0.05,
        "light rain, wet surfaces reflecting light, "
        "glistening streets, rain droplets",
    ),
    "rain_heavy": WeatherProfile(
        "Heavy Rain", 0.15, 0.9, 0.0, 0.7, 0.3, 0.55, 0.7, 0.7, -0.08,
        "heavy rain, downpour, splashing water, "
        "rain streaks, dramatic wet atmosphere",
    ),
    "storm": WeatherProfile(
        "Thunderstorm", 0.1, 0.7, 0.0, 1.0, 0.25, 0.4, 0.9, 0.6, -0.1,
        "thunderstorm, lightning, dark ominous clouds, "
        "dramatic atmosphere, powerful forces of nature",
    ),
    "snow_light": WeatherProfile(
        "Light Snow", 0.1, 0.0, 0.3, 0.4, 0.2, 0.9, 0.5, 0.7, -0.08,
        "light snowfall, gentle flakes, winter atmosphere, "
        "dusted surfaces, quiet and peaceful",
    ),
    "snow_heavy": WeatherProfile(
        "Blizzard", 0.3, 0.0, 0.9, 0.5, 0.5, 0.6, 0.4, 0.5, -0.1,
        "blizzard, heavy snowfall, low visibility, "
        "whiteout conditions, fierce winter storm",
    ),
    "dust": WeatherProfile(
        "Dust Storm", 0.2, 0.0, 0.0, 0.3, 0.6, 0.7, 0.6, 0.5, 0.1,
        "dust storm, sandy haze, orange-tinted atmosphere, "
        "reduced visibility, gritty particles",
    ),
    "haze_golden": WeatherProfile(
        "Golden Haze", 0.1, 0.0, 0.0, 0.2, 0.4, 0.9, 0.7, 0.9, 0.15,
        "golden haze, warm atmospheric diffusion, "
        "dreamy soft light, vintage feeling",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# MATERIAL INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MaterialProfile:
    """How to render specific material types convincingly."""
    name: str
    specular_intensity: float
    roughness: float
    metallic: float
    subsurface: float           # subsurface scattering (skin, wax, jade)
    reflectivity: float
    transparency: float
    ior: float                  # index of refraction
    prompt_keywords: List[str]


MATERIAL_PROFILES: Dict[str, MaterialProfile] = {
    "polished_metal": MaterialProfile(
        "Polished Metal", 0.95, 0.05, 1.0, 0.0, 0.9, 0.0, 2.5,
        ["mirror-polished metal", "chrome reflections", "metallic sheen",
         "polished to perfection"],
    ),
    "brushed_metal": MaterialProfile(
        "Brushed Metal", 0.7, 0.3, 1.0, 0.0, 0.5, 0.0, 2.3,
        ["brushed aluminum", "anodized metal", "satin metallic finish",
         "directional metal texture"],
    ),
    "carbon_fiber": MaterialProfile(
        "Carbon Fiber", 0.6, 0.2, 0.3, 0.0, 0.3, 0.0, 1.6,
        ["visible carbon fiber weave", "carbon fiber texture",
         "aerospace-grade carbon", "clear-coated carbon fiber pattern"],
    ),
    "automotive_paint": MaterialProfile(
        "Automotive Paint", 0.85, 0.08, 0.4, 0.0, 0.7, 0.0, 1.5,
        ["deep automotive paint", "multi-coat metallic paint",
         "showroom-quality finish", "clear coat reflections",
         "candy paint depth"],
    ),
    "glass": MaterialProfile(
        "Glass", 0.9, 0.02, 0.0, 0.0, 0.8, 0.9, 1.5,
        ["crystal clear glass", "glass reflections and refractions",
         "transparent material", "light passing through glass"],
    ),
    "skin": MaterialProfile(
        "Human Skin", 0.3, 0.6, 0.0, 0.8, 0.1, 0.0, 1.4,
        ["realistic skin texture", "subsurface scattering in skin",
         "visible pores", "natural skin tones", "skin translucency"],
    ),
    "fabric_silk": MaterialProfile(
        "Silk", 0.6, 0.3, 0.0, 0.2, 0.3, 0.0, 1.5,
        ["flowing silk fabric", "silk sheen", "luxurious silk texture",
         "silk draping and folds"],
    ),
    "fabric_leather": MaterialProfile(
        "Leather", 0.35, 0.55, 0.0, 0.1, 0.15, 0.0, 1.5,
        ["rich leather texture", "leather grain visible", "patina",
         "hand-stitched leather", "premium leather"],
    ),
    "water": MaterialProfile(
        "Water", 0.7, 0.1, 0.0, 0.3, 0.6, 0.85, 1.33,
        ["crystal clear water", "water reflections", "light caustics",
         "water surface tension", "refraction through water"],
    ),
    "wood": MaterialProfile(
        "Wood", 0.2, 0.7, 0.0, 0.05, 0.08, 0.0, 1.5,
        ["rich wood grain", "natural wood texture", "polished wood surface",
         "aged patina wood"],
    ),
    "concrete": MaterialProfile(
        "Concrete", 0.1, 0.9, 0.0, 0.0, 0.05, 0.0, 1.5,
        ["raw concrete texture", "brutalist concrete", "exposed aggregate",
         "weathered concrete surface"],
    ),
    "marble": MaterialProfile(
        "Marble", 0.7, 0.15, 0.0, 0.3, 0.5, 0.1, 1.5,
        ["polished marble veins", "carrara marble", "marble surface",
         "light penetrating marble", "translucent marble edges"],
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# COMPOSITION INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════════

class CompositionRule(Enum):
    RULE_OF_THIRDS = "rule_of_thirds"
    GOLDEN_RATIO = "golden_ratio"
    CENTERED = "centered"
    DIAGONAL = "diagonal"
    LEADING_LINES = "leading_lines"
    FRAME_WITHIN_FRAME = "frame_within_frame"
    SYMMETRY = "symmetry"
    FILL_THE_FRAME = "fill_the_frame"
    NEGATIVE_SPACE = "negative_space"
    TRIANGULAR = "triangular"
    S_CURVE = "s_curve"
    LAYERED_DEPTH = "layered_depth"


COMPOSITION_PROMPTS: Dict[CompositionRule, str] = {
    CompositionRule.RULE_OF_THIRDS:
        "subject placed at rule of thirds intersection, "
        "off-center composition, balanced visual weight",
    CompositionRule.GOLDEN_RATIO:
        "golden ratio spiral composition, fibonacci placement, "
        "naturally harmonious layout",
    CompositionRule.CENTERED:
        "centered composition, symmetrical framing, "
        "subject commanding the center",
    CompositionRule.DIAGONAL:
        "strong diagonal lines, dynamic composition, "
        "energy flowing corner to corner",
    CompositionRule.LEADING_LINES:
        "strong leading lines drawing the eye to subject, "
        "converging perspective lines, visual pathway",
    CompositionRule.FRAME_WITHIN_FRAME:
        "natural frame within frame, doorway or arch framing, "
        "layered framing composition",
    CompositionRule.SYMMETRY:
        "perfect bilateral symmetry, mirror-image composition, "
        "satisfying symmetrical balance",
    CompositionRule.FILL_THE_FRAME:
        "subject fills entire frame, intimate tight crop, "
        "no wasted space, maximum impact",
    CompositionRule.NEGATIVE_SPACE:
        "generous negative space, subject isolated in emptiness, "
        "minimalist composition, breathing room",
    CompositionRule.TRIANGULAR:
        "triangular composition, three-point visual structure, "
        "stable and dynamic triangular layout",
    CompositionRule.S_CURVE:
        "S-curve composition, winding path or line, "
        "elegant flowing visual movement",
    CompositionRule.LAYERED_DEPTH:
        "distinct foreground, midground, background layers, "
        "rich depth separation, environmental storytelling",
}


# ═══════════════════════════════════════════════════════════════════════════════
# EMOTIONAL MAPPING — mood → visual parameters
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EmotionalProfile:
    """Maps an emotion to concrete visual parameters."""
    name: str
    dna_overrides: Dict[str, float]
    composition_bias: CompositionRule
    palette: str
    lens_preference: str
    time_preference: str
    prompt_fragment: str


EMOTION_MAP: Dict[str, EmotionalProfile] = {
    "power": EmotionalProfile(
        "Power", {"contrast": 0.8, "drama": 0.9, "dynamism": 0.8,
                  "light_hardness": 0.7, "rim_light": 0.6,
                  "black_level": 0.2, "tension": 0.7},
        CompositionRule.DIAGONAL, "dramatic", "arri_master_prime_35",
        "golden_hour_evening",
        "powerful, commanding presence, imposing, monumental",
    ),
    "elegance": EmotionalProfile(
        "Elegance", {"elegance": 0.95, "saturation": 0.4,
                     "contrast": 0.5, "grain_amount": 0.05,
                     "depth_of_field": 0.6, "bokeh_character": 0.9,
                     "negative_space": 0.6},
        CompositionRule.GOLDEN_RATIO, "luxurious", "leica_noctilux_50",
        "golden_hour_evening",
        "refined elegance, tasteful, sophisticated, understated luxury",
    ),
    "serenity": EmotionalProfile(
        "Serenity", {"contrast": 0.35, "saturation": 0.4,
                     "brightness": 0.6, "warmth": 0.6,
                     "atmospheric_haze": 0.2, "tension": 0.0,
                     "drama": 0.2},
        CompositionRule.NEGATIVE_SPACE, "serene", "sony_gm_85",
        "morning",
        "peaceful, calming, tranquil atmosphere, gentle light",
    ),
    "mystery": EmotionalProfile(
        "Mystery", {"brightness": 0.25, "contrast": 0.75,
                    "mystery": 1.0, "fog_density": 0.4,
                    "fill_ratio": 0.15, "vignette": 0.4,
                    "saturation": 0.3},
        CompositionRule.NEGATIVE_SPACE, "dark", "cooke_s4_75",
        "blue_hour_evening",
        "mysterious, enigmatic, hidden elements, shadows concealing",
    ),
    "joy": EmotionalProfile(
        "Joy", {"brightness": 0.7, "saturation": 0.75,
                "vibrance": 0.8, "warmth": 0.7,
                "contrast": 0.45, "color_harmony": 0.9,
                "dynamism": 0.6},
        CompositionRule.RULE_OF_THIRDS, "energetic", "canon_rf_28_70",
        "golden_hour_morning",
        "joyful, uplifting, vibrant, full of life and color",
    ),
    "melancholy": EmotionalProfile(
        "Melancholy", {"saturation": 0.3, "brightness": 0.35,
                       "contrast": 0.6, "warmth": 0.3,
                       "grain_amount": 0.2, "nostalgia": 0.6,
                       "atmospheric_haze": 0.3},
        CompositionRule.NEGATIVE_SPACE, "dark", "cooke_s4_75",
        "blue_hour",
        "melancholic, wistful, bittersweet, solitary atmosphere",
    ),
    "awe": EmotionalProfile(
        "Awe", {"drama": 0.95, "contrast": 0.7,
                "dynamic_range": 0.95, "volumetric_light": 0.7,
                "depth_layers": 0.9, "cloud_drama": 0.8,
                "brightness": 0.5},
        CompositionRule.LAYERED_DEPTH, "dramatic", "nikon_z_14_24",
        "golden_hour_evening",
        "awe-inspiring, breathtaking, vast scale, overwhelming beauty",
    ),
    "tension": EmotionalProfile(
        "Tension", {"contrast": 0.85, "tension": 1.0,
                    "drama": 0.8, "fill_ratio": 0.1,
                    "light_hardness": 0.9, "black_level": 0.25,
                    "saturation": 0.3},
        CompositionRule.DIAGONAL, "dramatic", "arri_master_prime_35",
        "night",
        "tense, suspenseful, edge of your seat, ominous",
    ),
    "nostalgia": EmotionalProfile(
        "Nostalgia", {"nostalgia": 0.9, "grain_amount": 0.35,
                      "saturation": 0.4, "warmth": 0.65,
                      "sharpness": 0.5, "vignette": 0.3,
                      "chromatic_aberration": 0.08},
        CompositionRule.CENTERED, "romantic", "cooke_s4_75",
        "golden_hour_evening",
        "nostalgic, vintage feeling, memory-like quality, "
        "faded warmth, analog film aesthetic",
    ),
    "intimacy": EmotionalProfile(
        "Intimacy", {"depth_of_field": 0.9, "bokeh_intensity": 0.8,
                     "intimacy": 1.0, "warmth": 0.65,
                     "light_hardness": 0.2, "fill_ratio": 0.5,
                     "grain_amount": 0.08},
        CompositionRule.FILL_THE_FRAME, "romantic", "leica_noctilux_50",
        "golden_hour_evening",
        "intimate, personal, close and tender, soft focus background",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# SCENE ARCHITECTURE — layered depth decomposition
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SceneLayer:
    """A single depth layer in the scene."""
    name: str                   # "foreground", "midground", "background", "sky"
    depth_range: Tuple[float, float]   # 0=nearest, 1=infinity
    focus_priority: float       # 0=don't care, 1=primary focus
    detail_level: float         # how much detail this layer needs
    blur_amount: float          # depth-of-field blur for this layer
    elements: List[str]         # what's in this layer
    prompt_fragment: str


@dataclass
class SceneArchitecture:
    """Full layered decomposition of a scene."""
    layers: List[SceneLayer]
    primary_subject_layer: str
    composition: CompositionRule
    depth_complexity: float     # 0=flat, 1=many distinct depth planes
    environmental_storytelling: float  # 0=blank backdrop, 1=rich context

    def to_prompt_fragments(self) -> List[str]:
        fragments = []
        for layer in self.layers:
            if layer.elements:
                fragments.append(layer.prompt_fragment)
        fragments.append(COMPOSITION_PROMPTS.get(
            self.composition,
            COMPOSITION_PROMPTS[CompositionRule.RULE_OF_THIRDS]
        ))
        return fragments


# ═══════════════════════════════════════════════════════════════════════════════
# STYLE EVOLUTION — genetic algorithm for style breeding
# ═══════════════════════════════════════════════════════════════════════════════

class StyleEvolution:
    """
    Breeds new visual styles from validated parents using genetic principles.

    Mutation:  Random perturbation of individual DNA dimensions.
    Crossover: Combine dimensions from two parent styles.
    Selection: Keep styles that pass Mesh validation.

    This is how the system discovers novel aesthetics without
    any human designer or fine-tuning.
    """

    def __init__(self, mutation_rate: float = 0.15,
                 mutation_magnitude: float = 0.1):
        self._mutation_rate = mutation_rate
        self._mutation_mag = mutation_magnitude
        self._generation = 0
        self._population: Dict[str, VisualDNA] = {}
        self._fitness_scores: Dict[str, float] = {}

    def register(self, name: str, dna: VisualDNA,
                 fitness: float = 0.5):
        self._population[name] = dna
        self._fitness_scores[name] = fitness

    def mutate(self, parent: VisualDNA,
               magnitude: Optional[float] = None) -> VisualDNA:
        """Random perturbation of DNA dimensions."""
        mag = magnitude or self._mutation_mag
        child = deepcopy(parent)
        for k in child.__dataclass_fields__:
            if random.random() < self._mutation_rate:
                val = getattr(child, k)
                delta = random.gauss(0, mag)
                setattr(child, k, max(0.0, min(1.0, val + delta)))
        return child

    def crossover(self, parent_a: VisualDNA,
                  parent_b: VisualDNA) -> VisualDNA:
        """Combine dimensions from two parents."""
        child_dict = {}
        for k in parent_a.__dataclass_fields__:
            if random.random() < 0.5:
                child_dict[k] = getattr(parent_a, k)
            else:
                child_dict[k] = getattr(parent_b, k)
        return VisualDNA.from_dict(child_dict)

    def breed(self, parent_a_name: str,
              parent_b_name: str) -> VisualDNA:
        """Crossover + mutation from two named parents."""
        pa = self._population.get(parent_a_name)
        pb = self._population.get(parent_b_name)
        if pa is None or pb is None:
            raise ValueError(f"Unknown parent style: "
                             f"{parent_a_name} or {parent_b_name}")
        child = self.crossover(pa, pb)
        child = self.mutate(child, self._mutation_mag * 0.5)
        self._generation += 1
        return child

    def select_elite(self, top_n: int = 5) -> List[Tuple[str, VisualDNA]]:
        """Return the top-N styles by fitness."""
        ranked = sorted(self._fitness_scores.items(),
                        key=lambda x: x[1], reverse=True)
        return [(name, self._population[name])
                for name, _ in ranked[:top_n]
                if name in self._population]

    def evolve_generation(self, top_n: int = 5,
                          offspring_per_pair: int = 2) -> List[VisualDNA]:
        """Breed a new generation from elite parents."""
        elite = self.select_elite(top_n)
        if len(elite) < 2:
            return []
        offspring = []
        for i in range(len(elite)):
            for j in range(i + 1, len(elite)):
                for _ in range(offspring_per_pair):
                    child = self.crossover(elite[i][1], elite[j][1])
                    child = self.mutate(child)
                    offspring.append(child)
        self._generation += 1
        return offspring


# ═══════════════════════════════════════════════════════════════════════════════
# NEGATIVE PROMPT LEARNING
# ═══════════════════════════════════════════════════════════════════════════════

class NegativePromptLearner:
    """
    Learns what NOT to generate from rejected outputs and user feedback.
    Builds domain-specific anti-pattern libraries that strengthen over time.
    """

    UNIVERSAL_NEGATIVES = (
        "watermark, text, logo, signature, username, "
        "low quality, jpeg artifacts, compression artifacts, "
        "blurry, out of focus, duplicate, morbid, mutilated, "
        "deformed, disfigured, bad anatomy, bad proportions, "
        "extra limbs, cloned face, gross proportions, "
        "malformed limbs, missing arms, missing legs, "
        "extra arms, extra legs, fused fingers, "
        "too many fingers, long neck, poorly drawn, "
        "mutation, bad art, beginner, amateur"
    )

    DOMAIN_NEGATIVES: Dict[str, str] = {
        "vehicle": (
            "toy car, RC car, cartoon car, damaged car, "
            "rusty, dirty, cheap plastic, model car, "
            "distorted proportions, wrong wheel count"
        ),
        "person": (
            "extra fingers, extra hands, deformed face, "
            "asymmetric eyes, cross-eyed, bad teeth, "
            "plastic skin, mannequin, wax figure, "
            "uncanny valley, wrong number of fingers"
        ),
        "architecture": (
            "structurally impossible, floating building, "
            "impossible geometry, melting walls, "
            "distorted perspective, wrong scale"
        ),
        "nature": (
            "artificial looking, plastic plants, "
            "wrong leaf shapes, floating rocks, "
            "impossible terrain, oversaturated"
        ),
        "food": (
            "inedible looking, melted food, wrong colors, "
            "plastic food, artificial food, messy plating"
        ),
    }

    def __init__(self):
        self._learned: Dict[str, Set[str]] = {}
        self._rejection_count: Dict[str, int] = {}

    def learn_rejection(self, domain: str, issue: str):
        """Learn a new negative pattern from a rejected output."""
        if domain not in self._learned:
            self._learned[domain] = set()
        self._learned[domain].add(issue)
        self._rejection_count[domain] = \
            self._rejection_count.get(domain, 0) + 1

    def build_negative(self, domains: List[str],
                       style: str = "") -> str:
        """Build a comprehensive negative prompt for given domains."""
        parts = [self.UNIVERSAL_NEGATIVES]
        for domain in domains:
            if domain in self.DOMAIN_NEGATIVES:
                parts.append(self.DOMAIN_NEGATIVES[domain])
            if domain in self._learned:
                parts.append(", ".join(self._learned[domain]))
        return ", ".join(parts)

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "learned_patterns": {d: len(p)
                                 for d, p in self._learned.items()},
            "total_rejections": sum(self._rejection_count.values()),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ADAPTIVE GUIDANCE — learns optimal parameters from successes
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptiveGuidance:
    """
    Learns optimal guidance_scale, inference steps, and CFG per
    style/subject combination from successful generations.
    """

    def __init__(self):
        self._successes: Dict[str, List[Dict[str, float]]] = {}
        self._optimal: Dict[str, Dict[str, float]] = {}

    def record_success(self, style: str, subjects: List[str],
                       guidance_scale: float, steps: int,
                       quality_score: float):
        key = f"{style}:{','.join(sorted(subjects))}"
        if key not in self._successes:
            self._successes[key] = []
        self._successes[key].append({
            "guidance_scale": guidance_scale,
            "steps": steps,
            "quality_score": quality_score,
        })
        self._recompute_optimal(key)

    def get_optimal(self, style: str,
                    subjects: List[str]) -> Optional[Dict[str, float]]:
        key = f"{style}:{','.join(sorted(subjects))}"
        return self._optimal.get(key)

    def _recompute_optimal(self, key: str):
        records = self._successes.get(key, [])
        if len(records) < 2:
            return
        total_w = sum(r["quality_score"] for r in records) or 1.0
        self._optimal[key] = {
            "guidance_scale": sum(
                r["guidance_scale"] * r["quality_score"]
                for r in records
            ) / total_w,
            "steps": round(sum(
                r["steps"] * r["quality_score"]
                for r in records
            ) / total_w),
        }

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "tracked_combinations": len(self._successes),
            "optimized_combinations": len(self._optimal),
            "total_successes": sum(
                len(v) for v in self._successes.values()
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CREATIVE VISION ENGINE — the unified creative brain
# ═══════════════════════════════════════════════════════════════════════════════

class NRSICreativeVision:
    """
    The full visual creative intelligence of NRS.

    Integrates: Visual DNA, style evolution, composition rules,
    color theory, lens simulation, time/weather/material intelligence,
    emotional mapping, scene architecture, negative prompt learning,
    and adaptive guidance.

    Every generation passes through this engine to get:
      1. Complete Visual DNA for the target style
      2. Lens selection and prompt injection
      3. Color palette from emotional mapping
      4. Time-of-day and weather parameters
      5. Material-specific rendering hints
      6. Composition rule selection
      7. Scene layer decomposition
      8. Domain-specific negative prompts
      9. Adaptive guidance parameters from past successes
      10. Style evolution for novel aesthetics
    """

    def __init__(self):
        self.evolution = StyleEvolution()
        self.negative_learner = NegativePromptLearner()
        self.adaptive = AdaptiveGuidance()

        for name, dna in STYLE_DNA.items():
            self.evolution.register(name, dna, fitness=0.7)

    def get_visual_dna(self, style: str,
                       mood: Optional[str] = None) -> VisualDNA:
        """Get or synthesize a VisualDNA for the given style and mood."""
        base = STYLE_DNA.get(style, STYLE_DNA["photorealistic"])

        if mood and mood in EMOTION_MAP:
            emotion = EMOTION_MAP[mood]
            override = VisualDNA.from_dict(emotion.dna_overrides)
            base = base.blend(override, alpha=0.4)

        return base

    def select_lens(self, style: str, subjects: List[str],
                    mood: str) -> LensProfile:
        """Select the ideal lens for this scene."""
        if mood in EMOTION_MAP:
            pref = EMOTION_MAP[mood].lens_preference
            if pref in LENS_LIBRARY:
                return LENS_LIBRARY[pref]

        if "vehicle" in subjects:
            return LENS_LIBRARY["arri_master_prime_35"]
        if "person" in subjects:
            return LENS_LIBRARY["sony_gm_85"]
        if "nature" in subjects and style in ("landscape", "aerial"):
            return LENS_LIBRARY["nikon_z_14_24"]
        if "architecture" in subjects:
            return LENS_LIBRARY["sigma_art_35"]
        if "food" in subjects or style == "product":
            return LENS_LIBRARY["zeiss_otus_55"]
        if style == "cinematic":
            return LENS_LIBRARY["cooke_s4_75"]
        if style == "hyperreal":
            return LENS_LIBRARY["phase_one_150mp"]
        if style == "editorial" or style == "fashion":
            return LENS_LIBRARY["hasselblad_xcd_90"]

        return LENS_LIBRARY["canon_rf_28_70"]

    def select_palette(self, mood: str) -> ColorPalette:
        """Select color palette from mood."""
        return MOOD_PALETTES.get(mood, MOOD_PALETTES["neutral"])

    def select_time(self, mood: str,
                    prompt_lower: str) -> Optional[TimeOfDay]:
        """Detect or infer time of day."""
        for name, tod in TIME_PRESETS.items():
            for kw in name.replace("_", " ").split():
                if kw in prompt_lower and len(kw) > 3:
                    return tod

        time_keywords = {
            "sunset": "sunset", "sunrise": "golden_hour_morning",
            "golden hour": "golden_hour_evening",
            "blue hour": "blue_hour", "twilight": "blue_hour_evening",
            "night": "night", "midnight": "night",
            "noon": "high_noon", "morning": "morning",
            "afternoon": "afternoon", "dawn": "blue_hour",
            "dusk": "blue_hour_evening",
        }
        for kw, preset_name in time_keywords.items():
            if kw in prompt_lower:
                return TIME_PRESETS.get(preset_name)

        if mood in EMOTION_MAP:
            pref = EMOTION_MAP[mood].time_preference
            return TIME_PRESETS.get(pref)

        return None

    def select_weather(self, prompt_lower: str) -> Optional[WeatherProfile]:
        """Detect weather from prompt."""
        weather_keywords = {
            "fog": "fog", "foggy": "fog", "mist": "mist", "misty": "mist",
            "rain": "rain_light", "rainy": "rain_light",
            "downpour": "rain_heavy", "heavy rain": "rain_heavy",
            "storm": "storm", "thunder": "storm", "lightning": "storm",
            "snow": "snow_light", "snowing": "snow_light",
            "blizzard": "snow_heavy",
            "dust": "dust", "sandstorm": "dust",
            "haze": "haze_golden", "hazy": "haze_golden",
            "overcast": "overcast", "cloudy": "overcast",
            "dramatic clouds": "cloudy_dramatic",
            "clear sky": "clear", "clear": "clear",
        }
        for kw, profile_name in weather_keywords.items():
            if kw in prompt_lower:
                return WEATHER_PROFILES.get(profile_name)
        return None

    def detect_materials(self, subjects: List[str],
                         prompt_lower: str) -> List[MaterialProfile]:
        """Detect dominant materials in the scene."""
        found = []
        material_keywords = {
            "polished_metal": ["chrome", "polished", "mirror finish",
                               "stainless"],
            "carbon_fiber": ["carbon fiber", "carbon"],
            "automotive_paint": ["metallic paint", "car paint",
                                 "candy paint"],
            "glass": ["glass", "transparent", "crystal"],
            "skin": ["skin", "face", "portrait", "person"],
            "fabric_silk": ["silk", "satin", "flowing fabric"],
            "fabric_leather": ["leather"],
            "water": ["water", "ocean", "rain", "wet", "pool"],
            "wood": ["wood", "wooden", "timber"],
            "concrete": ["concrete", "brutalist", "cement"],
            "marble": ["marble"],
        }

        for mat_name, keywords in material_keywords.items():
            for kw in keywords:
                if kw in prompt_lower:
                    if mat_name in MATERIAL_PROFILES:
                        found.append(MATERIAL_PROFILES[mat_name])
                    break

        if "vehicle" in subjects and not found:
            found.append(MATERIAL_PROFILES["automotive_paint"])
            found.append(MATERIAL_PROFILES["polished_metal"])
            found.append(MATERIAL_PROFILES["glass"])

        return found

    def select_composition(self, subjects: List[str],
                           style: str, mood: str) -> CompositionRule:
        """Select optimal composition rule."""
        if mood in EMOTION_MAP:
            return EMOTION_MAP[mood].composition_bias

        if style == "architectural":
            return CompositionRule.SYMMETRY
        if style == "landscape":
            return CompositionRule.LAYERED_DEPTH
        if "vehicle" in subjects:
            return CompositionRule.LEADING_LINES
        if style == "portrait":
            return CompositionRule.RULE_OF_THIRDS
        if style == "product":
            return CompositionRule.CENTERED
        if style in ("macro", "editorial"):
            return CompositionRule.NEGATIVE_SPACE

        return CompositionRule.RULE_OF_THIRDS

    def decompose_scene(self, subjects: List[str], setting: str,
                        style: str) -> SceneArchitecture:
        """Build layered scene architecture."""
        layers = []

        if style in ("landscape", "aerial", "architectural"):
            layers.append(SceneLayer(
                "foreground", (0.0, 0.2), 0.6, 0.8, 0.1,
                ["foreground rocks", "flowers", "grass"],
                "detailed foreground element providing depth anchor",
            ))
            layers.append(SceneLayer(
                "midground", (0.2, 0.6), 1.0, 1.0, 0.0,
                subjects,
                f"sharp midground with {', '.join(subjects) or 'main subject'}",
            ))
            layers.append(SceneLayer(
                "background", (0.6, 0.9), 0.3, 0.5, 0.3,
                [setting],
                f"atmospheric background {setting}",
            ))
            layers.append(SceneLayer(
                "sky", (0.9, 1.0), 0.1, 0.4, 0.4,
                ["sky", "clouds"],
                "dramatic sky adding mood and context",
            ))
        else:
            layers.append(SceneLayer(
                "background", (0.5, 1.0), 0.2, 0.4, 0.5,
                [setting],
                f"soft out-of-focus {setting} background",
            ))
            layers.append(SceneLayer(
                "subject", (0.2, 0.5), 1.0, 1.0, 0.0,
                subjects,
                f"tack-sharp subject: {', '.join(subjects) or 'main element'}",
            ))

        comp = self.select_composition(subjects, style, "neutral")

        return SceneArchitecture(
            layers=layers,
            primary_subject_layer="midground" if len(layers) > 2 else "subject",
            composition=comp,
            depth_complexity=min(1.0, len(layers) / 4),
            environmental_storytelling=0.7 if setting != "studio" else 0.2,
        )

    # ── Master enrichment: the core creative pipeline ─────────────────

    def enrich(
        self,
        prompt: str,
        style: str,
        subjects: List[str],
        setting: str,
        mood: str,
        lighting: str,
    ) -> Dict[str, Any]:
        """
        Full creative enrichment pipeline. Returns everything needed
        to drive a generation at maximum creative intelligence.
        """
        lower = prompt.lower()

        dna = self.get_visual_dna(style, mood)
        lens = self.select_lens(style, subjects, mood)
        palette = self.select_palette(mood)
        time_of_day = self.select_time(mood, lower)
        weather = self.select_weather(lower)
        materials = self.detect_materials(subjects, lower)
        composition = self.select_composition(subjects, style, mood)
        scene_arch = self.decompose_scene(subjects, setting, style)

        prompt_fragments = [prompt]

        prompt_fragments.append(lens.prompt_hint)
        prompt_fragments.append(palette.prompt_fragment)
        prompt_fragments.append(
            COMPOSITION_PROMPTS.get(composition,
                                    COMPOSITION_PROMPTS[CompositionRule.RULE_OF_THIRDS])
        )

        if time_of_day:
            prompt_fragments.append(time_of_day.prompt_fragment)
        if weather:
            prompt_fragments.append(weather.prompt_fragment)

        for mat in materials[:3]:
            prompt_fragments.append(", ".join(mat.prompt_keywords[:2]))

        if mood in EMOTION_MAP:
            prompt_fragments.append(EMOTION_MAP[mood].prompt_fragment)

        for frag in scene_arch.to_prompt_fragments():
            prompt_fragments.append(frag)

        enriched_prompt = ", ".join(prompt_fragments)

        negative = self.negative_learner.build_negative(subjects, style)

        optimal = self.adaptive.get_optimal(style, subjects)

        if weather:
            dna.fog_density = max(dna.fog_density, weather.fog_density)
            dna.rain_intensity = max(dna.rain_intensity, weather.rain_intensity)
            dna.snow_intensity = max(dna.snow_intensity, weather.snow_intensity)
            dna.cloud_drama = max(dna.cloud_drama, weather.cloud_drama)
            dna.atmospheric_haze = max(dna.atmospheric_haze,
                                       weather.atmospheric_haze)
            dna.brightness = dna.brightness * weather.brightness_mod
            dna.contrast = dna.contrast * weather.contrast_mod
            dna.saturation = dna.saturation * weather.saturation_mod

        if time_of_day:
            dna.color_temperature = time_of_day.color_temperature
            dna.shadow_color_temp = time_of_day.shadow_color_temp
            dna.highlight_color_temp = time_of_day.highlight_color_temp
            dna.volumetric_light = max(dna.volumetric_light,
                                       time_of_day.volumetric_light)

        return {
            "enriched_prompt": enriched_prompt,
            "negative_prompt": negative,
            "visual_dna": dna,
            "lens": lens,
            "palette": palette,
            "time_of_day": time_of_day,
            "weather": weather,
            "materials": materials,
            "composition": composition,
            "scene_architecture": scene_arch,
            "adaptive_params": optimal,
            "dna_dimensions": dna.to_dict(),
        }

    def record_generation_feedback(
        self,
        style: str,
        subjects: List[str],
        guidance_scale: float,
        steps: int,
        quality_score: float,
        rejected: bool = False,
        rejection_reason: str = "",
    ):
        """Feed generation results back into the creative brain."""
        if rejected:
            for subj in subjects:
                self.negative_learner.learn_rejection(subj, rejection_reason)
        else:
            self.adaptive.record_success(
                style, subjects, guidance_scale, steps, quality_score
            )
            if quality_score > 0.8:
                dna = STYLE_DNA.get(style, STYLE_DNA["photorealistic"])
                self.evolution.register(
                    f"{style}_success_{int(time.time())}",
                    dna,
                    fitness=quality_score,
                )

    def breed_new_style(self, parent_a: str,
                        parent_b: str) -> VisualDNA:
        """Create a novel style by breeding two validated parents."""
        return self.evolution.breed(parent_a, parent_b)

    def evolve(self, top_n: int = 5) -> List[VisualDNA]:
        """Run one generation of style evolution."""
        return self.evolution.evolve_generation(top_n)

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "style_library": len(STYLE_DNA),
            "lens_library": len(LENS_LIBRARY),
            "color_palettes": len(MOOD_PALETTES),
            "time_presets": len(TIME_PRESETS),
            "weather_profiles": len(WEATHER_PROFILES),
            "material_profiles": len(MATERIAL_PROFILES),
            "composition_rules": len(COMPOSITION_PROMPTS),
            "emotion_profiles": len(EMOTION_MAP),
            "evolution": {
                "population": len(self.evolution._population),
                "generation": self.evolution._generation,
            },
            "negative_learner": self.negative_learner.stats,
            "adaptive_guidance": self.adaptive.stats,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    "NRSICreativeVision",
    "VisualDNA",
    "STYLE_DNA",
    "LensProfile",
    "LENS_LIBRARY",
    "ColorHarmony",
    "ColorPalette",
    "MOOD_PALETTES",
    "TimeOfDay",
    "TIME_PRESETS",
    "WeatherProfile",
    "WEATHER_PROFILES",
    "MaterialProfile",
    "MATERIAL_PROFILES",
    "CompositionRule",
    "COMPOSITION_PROMPTS",
    "EmotionalProfile",
    "EMOTION_MAP",
    "SceneLayer",
    "SceneArchitecture",
    "StyleEvolution",
    "NegativePromptLearner",
    "AdaptiveGuidance",
]
