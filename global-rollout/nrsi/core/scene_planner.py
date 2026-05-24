"""
NRS Scene Graph Planner
========================

Converts text prompts into structured scene descriptions for the SDF renderer.

Replaces keyword-bucket decomposition with NRS reasoning-driven scene
construction.  The planner performs:

  1. Semantic decomposition — subjects, setting, mood, time, weather, action,
     camera angle, style extracted via a comprehensive phrase taxonomy.
  2. Scene composition — rule-of-thirds placement, ground plane, depth layering,
     scale consistency.
  3. Lighting from mood / time-of-day — sun position, color temperature, fill.
  4. Camera from style / subject — FOV, elevation, distance, framing.
  5. Material assignment — subjects mapped to procedural materials.
  6. Environment — fog, ground, water, sky from setting keywords.

Dependencies: numpy, dataclasses (stdlib).  No ML, no external packages.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SceneObject:
    """A single entity placed in the scene."""
    name: str
    primitive: str
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: float = 1.0
    material: str = "default"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneLighting:
    """Full lighting rig for the scene."""
    lights: List[dict] = field(default_factory=list)
    ambient_color: Tuple[float, float, float] = (0.15, 0.15, 0.18)
    ambient_intensity: float = 0.3
    time_of_day: str = "noon"
    sky_type: str = "clear"


@dataclass
class SceneCamera:
    """Virtual camera placement and lens."""
    position: Tuple[float, float, float] = (0.0, 1.7, -8.0)
    target: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    fov_deg: float = 50.0
    style: str = "eye_level"


@dataclass
class SceneGraph:
    """Complete scene description ready for the SDF renderer."""
    objects: List[SceneObject] = field(default_factory=list)
    lighting: SceneLighting = field(default_factory=SceneLighting)
    camera: SceneCamera = field(default_factory=SceneCamera)
    environment: Dict[str, Any] = field(default_factory=dict)
    mood: str = "serene"
    post_processing: Dict[str, float] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# SUBJECT TAXONOMY — 200+ nouns → SDF primitive mappings
# ═══════════════════════════════════════════════════════════════════════════════

def _s(primitive: str, material: str, scale: float,
       params: Dict[str, Any] | None = None,
       category: str = "object") -> dict:
    return {
        "primitive": primitive,
        "default_material": material,
        "default_scale": scale,
        "default_params": params or {},
        "category": category,
    }


SUBJECT_MAP: Dict[str, dict] = {
    # ── humans ────────────────────────────────────────────────────────────
    "man":          _s("sd_human_figure", "skin", 1.0, {"pose": "standing", "gender": "male"}, "human"),
    "woman":        _s("sd_human_figure", "skin", 1.0, {"pose": "standing", "gender": "female"}, "human"),
    "person":       _s("sd_human_figure", "skin", 1.0, {"pose": "standing"}, "human"),
    "child":        _s("sd_human_figure", "skin", 0.65, {"pose": "standing", "age": "child"}, "human"),
    "boy":          _s("sd_human_figure", "skin", 0.7, {"pose": "standing", "gender": "male", "age": "child"}, "human"),
    "girl":         _s("sd_human_figure", "skin", 0.7, {"pose": "standing", "gender": "female", "age": "child"}, "human"),
    "baby":         _s("sd_human_figure", "skin", 0.35, {"pose": "sitting", "age": "infant"}, "human"),
    "elder":        _s("sd_human_figure", "skin", 0.95, {"pose": "standing", "age": "elderly"}, "human"),
    "athlete":      _s("sd_human_figure", "skin", 1.0, {"pose": "running", "build": "athletic"}, "human"),
    "dancer":       _s("sd_human_figure", "skin", 1.0, {"pose": "dancing"}, "human"),
    "soldier":      _s("sd_human_figure", "metal_armor", 1.0, {"pose": "standing", "gear": "military"}, "human"),
    "knight":       _s("sd_human_figure", "metal_steel", 1.0, {"pose": "standing", "gear": "armor"}, "human"),
    "samurai":      _s("sd_human_figure", "metal_steel", 1.0, {"pose": "standing", "gear": "samurai_armor"}, "human"),
    "astronaut":    _s("sd_human_figure", "fabric_synthetic", 1.0, {"pose": "standing", "gear": "spacesuit"}, "human"),
    "diver":        _s("sd_human_figure", "rubber", 1.0, {"pose": "standing", "gear": "wetsuit"}, "human"),
    "monk":         _s("sd_human_figure", "fabric_cotton", 1.0, {"pose": "sitting", "gear": "robe"}, "human"),
    "musician":     _s("sd_human_figure", "skin", 1.0, {"pose": "standing", "action": "playing"}, "human"),
    "couple":       _s("sd_human_figure", "skin", 1.0, {"pose": "standing", "count": 2}, "human"),

    # ── animals ───────────────────────────────────────────────────────────
    "dog":          _s("sd_animal_body", "fur_short", 0.5, {"species": "dog"}, "animal"),
    "cat":          _s("sd_animal_body", "fur_short", 0.3, {"species": "cat"}, "animal"),
    "horse":        _s("sd_animal_body", "fur_short", 1.6, {"species": "horse"}, "animal"),
    "cow":          _s("sd_animal_body", "fur_short", 1.4, {"species": "cow"}, "animal"),
    "sheep":        _s("sd_animal_body", "wool", 0.8, {"species": "sheep"}, "animal"),
    "deer":         _s("sd_animal_body", "fur_short", 1.3, {"species": "deer"}, "animal"),
    "wolf":         _s("sd_animal_body", "fur_thick", 0.8, {"species": "wolf"}, "animal"),
    "fox":          _s("sd_animal_body", "fur_thick", 0.4, {"species": "fox"}, "animal"),
    "bear":         _s("sd_animal_body", "fur_thick", 1.8, {"species": "bear"}, "animal"),
    "lion":         _s("sd_animal_body", "fur_short", 1.2, {"species": "lion"}, "animal"),
    "tiger":        _s("sd_animal_body", "fur_short", 1.1, {"species": "tiger"}, "animal"),
    "elephant":     _s("sd_animal_body", "skin_rough", 3.5, {"species": "elephant"}, "animal"),
    "giraffe":      _s("sd_animal_body", "skin_rough", 5.5, {"species": "giraffe"}, "animal"),
    "rabbit":       _s("sd_animal_body", "fur_soft", 0.25, {"species": "rabbit"}, "animal"),
    "eagle":        _s("sd_bird_body", "feathers", 0.8, {"species": "eagle"}, "animal"),
    "owl":          _s("sd_bird_body", "feathers", 0.4, {"species": "owl"}, "animal"),
    "hawk":         _s("sd_bird_body", "feathers", 0.5, {"species": "hawk"}, "animal"),
    "parrot":       _s("sd_bird_body", "feathers", 0.25, {"species": "parrot"}, "animal"),
    "swan":         _s("sd_bird_body", "feathers", 0.9, {"species": "swan"}, "animal"),
    "penguin":      _s("sd_bird_body", "feathers", 0.7, {"species": "penguin"}, "animal"),
    "dolphin":      _s("sd_fish_body", "skin_smooth", 2.0, {"species": "dolphin"}, "animal"),
    "whale":        _s("sd_fish_body", "skin_smooth", 15.0, {"species": "whale"}, "animal"),
    "shark":        _s("sd_fish_body", "skin_smooth", 3.0, {"species": "shark"}, "animal"),
    "fish":         _s("sd_fish_body", "scales", 0.3, {"species": "generic_fish"}, "animal"),
    "butterfly":    _s("sd_insect_body", "chitin", 0.05, {"species": "butterfly"}, "animal"),
    "bee":          _s("sd_insect_body", "chitin", 0.02, {"species": "bee"}, "animal"),
    "spider":       _s("sd_insect_body", "chitin", 0.04, {"species": "spider"}, "animal"),
    "snake":        _s("sd_snake_body", "scales", 0.8, {"species": "snake"}, "animal"),
    "turtle":       _s("sd_animal_body", "shell", 0.4, {"species": "turtle"}, "animal"),
    "frog":         _s("sd_animal_body", "skin_smooth", 0.08, {"species": "frog"}, "animal"),
    "dragon":       _s("sd_dragon_body", "scales_dragon", 8.0, {"species": "dragon"}, "animal"),
    "unicorn":      _s("sd_animal_body", "fur_short", 1.7, {"species": "unicorn"}, "animal"),

    # ── vehicles ──────────────────────────────────────────────────────────
    "car":          _s("sd_car_body", "metal_steel", 1.0, {"style": "sedan"}, "vehicle"),
    "sports car":   _s("sd_car_body", "metal_steel", 1.0, {"style": "sports"}, "vehicle"),
    "ferrari":      _s("sd_car_body", "metal_steel", 1.0, {"style": "sports", "brand": "ferrari"}, "vehicle"),
    "lamborghini":  _s("sd_car_body", "metal_steel", 1.0, {"style": "sports", "brand": "lamborghini"}, "vehicle"),
    "porsche":      _s("sd_car_body", "metal_steel", 1.0, {"style": "sports", "brand": "porsche"}, "vehicle"),
    "suv":          _s("sd_car_body", "metal_steel", 1.1, {"style": "suv"}, "vehicle"),
    "truck":        _s("sd_truck_body", "metal_steel", 1.5, {"style": "pickup"}, "vehicle"),
    "semi truck":   _s("sd_truck_body", "metal_steel", 3.0, {"style": "semi"}, "vehicle"),
    "bus":          _s("sd_bus_body", "metal_steel", 2.5, {"style": "city"}, "vehicle"),
    "van":          _s("sd_car_body", "metal_steel", 1.3, {"style": "van"}, "vehicle"),
    "motorcycle":   _s("sd_motorcycle_body", "metal_chrome", 0.7, {"style": "sport"}, "vehicle"),
    "bicycle":      _s("sd_bicycle_body", "metal_steel", 0.6, {}, "vehicle"),
    "scooter":      _s("sd_motorcycle_body", "plastic", 0.5, {"style": "scooter"}, "vehicle"),
    "train":        _s("sd_train_body", "metal_steel", 4.0, {"style": "passenger"}, "vehicle"),
    "airplane":     _s("sd_airplane_body", "metal_aluminum", 15.0, {"style": "commercial"}, "vehicle"),
    "helicopter":   _s("sd_helicopter_body", "metal_steel", 4.0, {}, "vehicle"),
    "boat":         _s("sd_boat_body", "fiberglass", 3.0, {"style": "sailboat"}, "vehicle"),
    "ship":         _s("sd_ship_body", "metal_steel", 25.0, {"style": "cargo"}, "vehicle"),
    "yacht":        _s("sd_boat_body", "fiberglass", 8.0, {"style": "luxury"}, "vehicle"),
    "submarine":    _s("sd_submarine_body", "metal_steel", 10.0, {}, "vehicle"),
    "rocket":       _s("sd_rocket_body", "metal_steel", 20.0, {}, "vehicle"),
    "spaceship":    _s("sd_spaceship_body", "metal_steel", 12.0, {"style": "sci_fi"}, "vehicle"),
    "tank":         _s("sd_tank_body", "metal_armor", 3.0, {}, "vehicle"),
    "tractor":      _s("sd_tractor_body", "metal_steel", 2.0, {}, "vehicle"),
    "ambulance":    _s("sd_car_body", "metal_steel", 1.3, {"style": "ambulance"}, "vehicle"),
    "fire truck":   _s("sd_truck_body", "metal_steel", 2.5, {"style": "fire_truck"}, "vehicle"),
    "police car":   _s("sd_car_body", "metal_steel", 1.0, {"style": "police"}, "vehicle"),
    "taxi":         _s("sd_car_body", "metal_steel", 1.0, {"style": "taxi"}, "vehicle"),
    "race car":     _s("sd_car_body", "metal_steel", 1.0, {"style": "formula"}, "vehicle"),

    # ── buildings ─────────────────────────────────────────────────────────
    "house":        _s("sd_building", "stone_brick", 1.0, {"style": "residential", "floors": 2}, "building"),
    "cabin":        _s("sd_building", "wood_log", 1.0, {"style": "cabin", "floors": 1}, "building"),
    "cottage":      _s("sd_building", "stone_brick", 0.8, {"style": "cottage", "floors": 1}, "building"),
    "mansion":      _s("sd_building", "stone_marble", 2.0, {"style": "mansion", "floors": 3}, "building"),
    "castle":       _s("sd_castle", "stone_granite", 5.0, {"style": "medieval"}, "building"),
    "palace":       _s("sd_building", "stone_marble", 6.0, {"style": "palace"}, "building"),
    "skyscraper":   _s("sd_building", "glass_steel", 1.0, {"style": "modern", "floors": 40}, "building"),
    "tower":        _s("sd_tower", "stone_granite", 3.0, {"style": "watchtower"}, "building"),
    "lighthouse":   _s("sd_tower", "stone_white", 2.5, {"style": "lighthouse"}, "building"),
    "church":       _s("sd_building", "stone_granite", 2.0, {"style": "church"}, "building"),
    "cathedral":    _s("sd_building", "stone_granite", 4.0, {"style": "gothic_cathedral"}, "building"),
    "mosque":       _s("sd_building", "stone_marble", 3.0, {"style": "mosque"}, "building"),
    "temple":       _s("sd_building", "stone_granite", 3.0, {"style": "temple"}, "building"),
    "pyramid":      _s("sd_pyramid", "stone_sandstone", 6.0, {}, "building"),
    "barn":         _s("sd_building", "wood_plank", 1.5, {"style": "barn"}, "building"),
    "windmill":     _s("sd_windmill", "stone_white", 2.0, {}, "building"),
    "factory":      _s("sd_building", "metal_steel", 2.5, {"style": "industrial"}, "building"),
    "warehouse":    _s("sd_building", "metal_corrugated", 2.0, {"style": "warehouse"}, "building"),
    "bridge":       _s("sd_bridge", "metal_steel", 3.0, {"style": "suspension"}, "building"),
    "dam":          _s("sd_dam", "concrete", 8.0, {}, "building"),
    "stadium":      _s("sd_building", "concrete", 5.0, {"style": "stadium"}, "building"),
    "hospital":     _s("sd_building", "concrete", 2.0, {"style": "hospital", "floors": 6}, "building"),
    "school":       _s("sd_building", "stone_brick", 1.5, {"style": "school", "floors": 3}, "building"),
    "office":       _s("sd_building", "glass_steel", 1.0, {"style": "office", "floors": 10}, "building"),
    "apartment":    _s("sd_building", "concrete", 1.0, {"style": "apartment", "floors": 8}, "building"),
    "ruins":        _s("sd_ruins", "stone_weathered", 2.0, {"style": "ancient"}, "building"),
    "hut":          _s("sd_building", "wood_plank", 0.6, {"style": "hut", "floors": 1}, "building"),
    "igloo":        _s("sd_dome", "ice", 1.0, {"style": "igloo"}, "building"),
    "tent":         _s("sd_tent", "fabric_canvas", 1.0, {}, "building"),
    "gazebo":       _s("sd_gazebo", "wood_oak", 1.0, {}, "building"),

    # ── nature: trees ─────────────────────────────────────────────────────
    "tree":         _s("sd_tree", "bark_generic", 1.0, {"species": "deciduous"}, "nature"),
    "oak":          _s("sd_tree", "bark_oak", 1.0, {"species": "oak"}, "nature"),
    "pine":         _s("sd_tree", "bark_pine", 1.2, {"species": "pine"}, "nature"),
    "birch":        _s("sd_tree", "bark_birch", 0.9, {"species": "birch"}, "nature"),
    "willow":       _s("sd_tree", "bark_willow", 1.0, {"species": "willow"}, "nature"),
    "maple":        _s("sd_tree", "bark_maple", 1.0, {"species": "maple"}, "nature"),
    "cherry blossom": _s("sd_tree", "bark_cherry", 0.8, {"species": "cherry", "bloom": True}, "nature"),
    "palm":         _s("sd_tree", "bark_palm", 1.0, {"species": "palm"}, "nature"),
    "palm tree":    _s("sd_tree", "bark_palm", 1.0, {"species": "palm"}, "nature"),
    "redwood":      _s("sd_tree", "bark_redwood", 3.0, {"species": "redwood"}, "nature"),
    "bamboo":       _s("sd_bamboo_cluster", "bamboo", 0.6, {}, "nature"),
    "cactus":       _s("sd_cactus", "plant_succulent", 0.5, {"species": "saguaro"}, "nature"),
    "bush":         _s("sd_bush", "leaf_green", 0.4, {}, "nature"),
    "shrub":        _s("sd_bush", "leaf_green", 0.3, {}, "nature"),
    "hedge":        _s("sd_hedge", "leaf_green", 0.5, {}, "nature"),
    "vine":         _s("sd_vine", "leaf_green", 0.3, {}, "nature"),
    "fern":         _s("sd_fern", "leaf_green", 0.3, {}, "nature"),
    "moss":         _s("sd_ground_cover", "moss", 0.05, {}, "nature"),
    "mushroom":     _s("sd_mushroom", "organic", 0.1, {}, "nature"),
    "flower":       _s("sd_flower", "petal", 0.15, {"species": "generic"}, "nature"),
    "rose":         _s("sd_flower", "petal_red", 0.15, {"species": "rose"}, "nature"),
    "sunflower":    _s("sd_flower", "petal_yellow", 0.3, {"species": "sunflower"}, "nature"),
    "tulip":        _s("sd_flower", "petal", 0.15, {"species": "tulip"}, "nature"),
    "lily":         _s("sd_flower", "petal_white", 0.12, {"species": "lily"}, "nature"),
    "daisy":        _s("sd_flower", "petal_white", 0.1, {"species": "daisy"}, "nature"),
    "lavender":     _s("sd_flower_cluster", "petal_purple", 0.2, {"species": "lavender"}, "nature"),
    "grass":        _s("sd_ground_cover", "grass", 0.05, {}, "nature"),
    "wheat":        _s("sd_crop_field", "straw", 0.5, {"crop": "wheat"}, "nature"),
    "corn":         _s("sd_crop_field", "leaf_green", 0.8, {"crop": "corn"}, "nature"),

    # ── nature: terrain / water ───────────────────────────────────────────
    "mountain":     _s("sd_mountain", "stone_granite", 1.0, {}, "nature"),
    "hill":         _s("sd_hill", "grass", 0.5, {}, "nature"),
    "cliff":        _s("sd_cliff", "stone_granite", 1.0, {}, "nature"),
    "valley":       _s("sd_valley", "grass", 1.0, {}, "nature"),
    "canyon":       _s("sd_canyon", "stone_sandstone", 1.0, {}, "nature"),
    "volcano":      _s("sd_volcano", "stone_basalt", 1.5, {}, "nature"),
    "island":       _s("sd_island", "sand", 1.0, {}, "nature"),
    "cave":         _s("sd_cave", "stone_granite", 1.0, {}, "nature"),
    "river":        _s("sd_river", "water", 1.0, {}, "nature"),
    "lake":         _s("sd_lake", "water", 1.0, {}, "nature"),
    "pond":         _s("sd_lake", "water", 0.3, {}, "nature"),
    "ocean":        _s("sd_ocean", "water_deep", 1.0, {}, "nature"),
    "waterfall":    _s("sd_waterfall", "water", 1.0, {}, "nature"),
    "stream":       _s("sd_river", "water", 0.3, {}, "nature"),
    "beach":        _s("sd_beach", "sand", 1.0, {}, "nature"),
    "desert":       _s("sd_desert", "sand", 1.0, {}, "nature"),
    "dune":         _s("sd_dune", "sand", 1.0, {}, "nature"),
    "glacier":      _s("sd_glacier", "ice", 2.0, {}, "nature"),
    "iceberg":      _s("sd_iceberg", "ice", 3.0, {}, "nature"),
    "rock":         _s("sd_rock", "stone_granite", 0.5, {}, "nature"),
    "boulder":      _s("sd_rock", "stone_granite", 1.5, {}, "nature"),
    "pebble":       _s("sd_rock", "stone_smooth", 0.05, {}, "nature"),
    "crystal":      _s("sd_crystal", "crystal", 0.3, {}, "nature"),
    "coral":        _s("sd_coral", "coral", 0.4, {}, "nature"),

    # ── furniture ─────────────────────────────────────────────────────────
    "chair":        _s("sd_chair", "wood_oak", 0.5, {"style": "dining"}, "furniture"),
    "armchair":     _s("sd_chair", "fabric_leather", 0.6, {"style": "armchair"}, "furniture"),
    "sofa":         _s("sd_sofa", "fabric_leather", 0.8, {}, "furniture"),
    "couch":        _s("sd_sofa", "fabric_cotton", 0.8, {}, "furniture"),
    "table":        _s("sd_table", "wood_oak", 0.5, {"style": "dining"}, "furniture"),
    "desk":         _s("sd_table", "wood_oak", 0.5, {"style": "desk"}, "furniture"),
    "bed":          _s("sd_bed", "fabric_cotton", 0.6, {}, "furniture"),
    "bookshelf":    _s("sd_bookshelf", "wood_oak", 0.8, {}, "furniture"),
    "cabinet":      _s("sd_cabinet", "wood_oak", 0.7, {}, "furniture"),
    "wardrobe":     _s("sd_cabinet", "wood_oak", 0.9, {"style": "wardrobe"}, "furniture"),
    "lamp":         _s("sd_lamp", "metal_brass", 0.3, {}, "furniture"),
    "chandelier":   _s("sd_chandelier", "metal_brass", 0.6, {}, "furniture"),
    "mirror":       _s("sd_mirror", "glass", 0.5, {}, "furniture"),
    "rug":          _s("sd_rug", "fabric_wool", 0.05, {}, "furniture"),
    "piano":        _s("sd_piano", "wood_ebony", 1.0, {}, "furniture"),
    "fireplace":    _s("sd_fireplace", "stone_brick", 0.8, {}, "furniture"),
    "bathtub":      _s("sd_bathtub", "ceramic", 0.6, {}, "furniture"),
    "sink":         _s("sd_sink", "ceramic", 0.3, {}, "furniture"),
    "toilet":       _s("sd_toilet", "ceramic", 0.3, {}, "furniture"),
    "stove":        _s("sd_stove", "metal_steel", 0.5, {}, "furniture"),
    "refrigerator": _s("sd_refrigerator", "metal_steel", 0.7, {}, "furniture"),
    "bench":        _s("sd_bench", "wood_oak", 0.5, {}, "furniture"),
    "stool":        _s("sd_stool", "wood_oak", 0.3, {}, "furniture"),
    "throne":       _s("sd_chair", "wood_oak", 0.8, {"style": "throne"}, "furniture"),
    "fountain":     _s("sd_fountain", "stone_marble", 1.0, {}, "furniture"),
    "statue":       _s("sd_statue", "stone_marble", 1.2, {}, "furniture"),
    "clock":        _s("sd_clock", "wood_oak", 0.3, {}, "furniture"),
    "vase":         _s("sd_vase", "ceramic", 0.2, {}, "furniture"),

    # ── food ──────────────────────────────────────────────────────────────
    "apple":        _s("sd_sphere_fruit", "fruit_red", 0.04, {"fruit": "apple"}, "food"),
    "orange":       _s("sd_sphere_fruit", "fruit_orange", 0.04, {"fruit": "orange"}, "food"),
    "banana":       _s("sd_banana", "fruit_yellow", 0.04, {}, "food"),
    "grape":        _s("sd_grape_cluster", "fruit_purple", 0.03, {}, "food"),
    "watermelon":   _s("sd_sphere_fruit", "fruit_green", 0.15, {"fruit": "watermelon"}, "food"),
    "pineapple":    _s("sd_pineapple", "fruit_yellow", 0.12, {}, "food"),
    "strawberry":   _s("sd_sphere_fruit", "fruit_red", 0.02, {"fruit": "strawberry"}, "food"),
    "bread":        _s("sd_bread_loaf", "bread_crust", 0.1, {}, "food"),
    "cake":         _s("sd_cake", "frosting", 0.15, {}, "food"),
    "pizza":        _s("sd_pizza", "food_surface", 0.2, {}, "food"),
    "burger":       _s("sd_burger", "food_surface", 0.06, {}, "food"),
    "sushi":        _s("sd_sushi", "food_surface", 0.03, {}, "food"),
    "wine glass":   _s("sd_wine_glass", "glass", 0.12, {}, "food"),
    "cup":          _s("sd_cup", "ceramic", 0.06, {}, "food"),
    "mug":          _s("sd_cup", "ceramic", 0.06, {"style": "mug"}, "food"),
    "bottle":       _s("sd_bottle", "glass", 0.12, {}, "food"),
    "bowl":         _s("sd_bowl", "ceramic", 0.08, {}, "food"),
    "plate":        _s("sd_plate", "ceramic", 0.12, {}, "food"),
    "pot":          _s("sd_pot", "metal_steel", 0.12, {}, "food"),
    "pan":          _s("sd_pan", "metal_steel", 0.15, {}, "food"),
    "cheese":       _s("sd_cheese_wedge", "food_surface", 0.06, {}, "food"),
    "egg":          _s("sd_egg", "shell_white", 0.03, {}, "food"),

    # ── objects / props ───────────────────────────────────────────────────
    "sword":        _s("sd_sword", "metal_steel", 0.5, {}, "object"),
    "shield":       _s("sd_shield", "metal_steel", 0.4, {}, "object"),
    "axe":          _s("sd_axe", "metal_steel", 0.4, {}, "object"),
    "bow":          _s("sd_bow", "wood_yew", 0.5, {}, "object"),
    "arrow":        _s("sd_arrow", "wood_ash", 0.3, {}, "object"),
    "spear":        _s("sd_spear", "wood_ash", 0.7, {}, "object"),
    "staff":        _s("sd_staff", "wood_oak", 0.7, {}, "object"),
    "wand":         _s("sd_wand", "wood_oak", 0.15, {}, "object"),
    "crown":        _s("sd_crown", "metal_gold", 0.1, {}, "object"),
    "ring":         _s("sd_ring", "metal_gold", 0.01, {}, "object"),
    "gem":          _s("sd_gem", "crystal", 0.02, {}, "object"),
    "diamond":      _s("sd_gem", "diamond", 0.02, {}, "object"),
    "key":          _s("sd_key", "metal_brass", 0.04, {}, "object"),
    "chest":        _s("sd_chest", "wood_oak", 0.3, {}, "object"),
    "barrel":       _s("sd_barrel", "wood_oak", 0.4, {}, "object"),
    "crate":        _s("sd_crate", "wood_plank", 0.3, {}, "object"),
    "box":          _s("sd_box_prop", "cardboard", 0.2, {}, "object"),
    "ball":         _s("sd_sphere_prop", "rubber", 0.1, {}, "object"),
    "globe":        _s("sd_sphere_prop", "painted", 0.15, {"style": "globe"}, "object"),
    "telescope":    _s("sd_telescope", "metal_brass", 0.4, {}, "object"),
    "compass":      _s("sd_compass", "metal_brass", 0.04, {}, "object"),
    "lantern":      _s("sd_lantern", "metal_brass", 0.15, {}, "object"),
    "candle":       _s("sd_candle", "wax", 0.1, {}, "object"),
    "torch":        _s("sd_torch", "wood_ash", 0.2, {}, "object"),
    "campfire":     _s("sd_campfire", "wood_ash", 0.3, {}, "object"),
    "flag":         _s("sd_flag", "fabric_cotton", 0.5, {}, "object"),
    "banner":       _s("sd_banner", "fabric_cotton", 0.6, {}, "object"),
    "rope":         _s("sd_rope", "hemp", 0.1, {}, "object"),
    "chain":        _s("sd_chain", "metal_steel", 0.1, {}, "object"),
    "anchor":       _s("sd_anchor", "metal_steel", 0.5, {}, "object"),
    "wheel":        _s("sd_wheel", "wood_oak", 0.4, {}, "object"),
    "gear":         _s("sd_gear", "metal_steel", 0.2, {}, "object"),
    "book":         _s("sd_book", "leather", 0.1, {}, "object"),
    "scroll":       _s("sd_scroll", "parchment", 0.1, {}, "object"),
    "map":          _s("sd_map", "parchment", 0.15, {}, "object"),
    "painting":     _s("sd_painting", "canvas", 0.3, {}, "object"),
    "guitar":       _s("sd_guitar", "wood_spruce", 0.4, {}, "object"),
    "drum":         _s("sd_drum", "wood_oak", 0.3, {}, "object"),
    "violin":       _s("sd_violin", "wood_maple", 0.25, {}, "object"),
    "trumpet":      _s("sd_trumpet", "metal_brass", 0.2, {}, "object"),
    "bell":         _s("sd_bell", "metal_bronze", 0.3, {}, "object"),
    "umbrella":     _s("sd_umbrella", "fabric_nylon", 0.4, {}, "object"),
    "hat":          _s("sd_hat", "fabric_felt", 0.12, {}, "object"),
    "mask":         _s("sd_mask", "ceramic", 0.1, {}, "object"),
    "skull":        _s("sd_skull", "bone", 0.1, {}, "object"),
    "cross":        _s("sd_cross", "wood_oak", 0.5, {}, "object"),
    "tombstone":    _s("sd_tombstone", "stone_granite", 0.5, {}, "object"),
    "sign":         _s("sd_sign", "wood_plank", 0.4, {}, "object"),
    "mailbox":      _s("sd_mailbox", "metal_steel", 0.3, {}, "object"),
    "streetlight":  _s("sd_streetlight", "metal_steel", 1.5, {}, "object"),
    "traffic light": _s("sd_traffic_light", "metal_steel", 1.2, {}, "object"),
    "telephone pole": _s("sd_pole", "wood_treated", 3.0, {}, "object"),
    "satellite dish": _s("sd_dish", "metal_steel", 0.5, {}, "object"),
    "robot":        _s("sd_robot", "metal_steel", 1.0, {"style": "humanoid"}, "object"),
    "drone":        _s("sd_drone", "plastic", 0.3, {}, "object"),
    "camera":       _s("sd_camera", "metal_steel", 0.08, {}, "object"),
    "computer":     _s("sd_computer", "plastic", 0.2, {}, "object"),
    "phone":        _s("sd_phone", "glass", 0.06, {}, "object"),
    "television":   _s("sd_screen", "plastic", 0.4, {}, "object"),
    "screen":       _s("sd_screen", "plastic", 0.3, {}, "object"),
}

_SUBJECT_COUNT = len(SUBJECT_MAP)
assert _SUBJECT_COUNT >= 200, (
    f"SUBJECT_MAP has {_SUBJECT_COUNT} entries, need >= 200"
)


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTITY PARSING
# ═══════════════════════════════════════════════════════════════════════════════

_QUANTITY_WORDS: Dict[str, Tuple[int, int]] = {
    "a":        (1, 1),
    "an":       (1, 1),
    "one":      (1, 1),
    "single":   (1, 1),
    "couple":   (2, 2),
    "pair":     (2, 2),
    "two":      (2, 2),
    "three":    (3, 3),
    "few":      (3, 5),
    "several":  (3, 5),
    "four":     (4, 4),
    "five":     (5, 5),
    "six":      (6, 6),
    "some":     (4, 7),
    "many":     (6, 12),
    "group":    (6, 12),
    "herd":     (8, 15),
    "flock":    (8, 15),
    "pack":     (5, 10),
    "crowd":    (15, 30),
    "army":     (20, 50),
    "forest":   (20, 50),
    "field":    (15, 40),
    "row":      (5, 8),
    "line":     (4, 7),
    "cluster":  (5, 10),
    "dozen":    (12, 12),
}

_SIZE_MODIFIERS: Dict[str, float] = {
    "tiny":     0.3,
    "miniature": 0.35,
    "small":    0.6,
    "little":   0.6,
    "medium":   1.0,
    "large":    1.5,
    "big":      1.5,
    "huge":     2.5,
    "massive":  3.5,
    "enormous": 4.0,
    "giant":    5.0,
    "colossal": 6.0,
    "towering": 3.0,
    "tall":     1.8,
    "short":    0.7,
}


# ═══════════════════════════════════════════════════════════════════════════════
# SPATIAL RELATIONSHIP OFFSETS
# ═══════════════════════════════════════════════════════════════════════════════

_SPATIAL_OFFSETS: Dict[str, Tuple[float, float, float]] = {
    "on":           (0.0,  1.0,  0.0),
    "on top of":    (0.0,  1.2,  0.0),
    "above":        (0.0,  2.0,  0.0),
    "over":         (0.0,  2.0,  0.0),
    "under":        (0.0, -0.5,  0.0),
    "below":        (0.0, -1.0,  0.0),
    "beneath":      (0.0, -0.5,  0.0),
    "next to":      (1.5,  0.0,  0.0),
    "beside":       (1.5,  0.0,  0.0),
    "near":         (2.0,  0.0,  0.5),
    "behind":       (0.0,  0.0,  2.0),
    "in front of":  (0.0,  0.0, -2.0),
    "left of":      (-1.5, 0.0,  0.0),
    "right of":     (1.5,  0.0,  0.0),
    "between":      (0.0,  0.0,  0.0),
    "inside":       (0.0,  0.0,  0.0),
    "around":       (2.0,  0.0,  0.0),
    "across from":  (0.0,  0.0, -3.0),
    "along":        (0.0,  0.0,  1.0),
}


# ═══════════════════════════════════════════════════════════════════════════════
# MOOD / TIME / CAMERA LOOKUP TABLES
# ═══════════════════════════════════════════════════════════════════════════════

_MOOD_KEYWORDS: Dict[str, str] = {
    "serene": "serene", "peaceful": "serene", "calm": "serene", "tranquil": "serene",
    "quiet": "serene", "gentle": "serene", "still": "serene",
    "dramatic": "dramatic", "epic": "dramatic", "intense": "dramatic",
    "powerful": "dramatic", "bold": "dramatic",
    "energetic": "energetic", "vibrant": "energetic", "lively": "energetic",
    "dynamic": "energetic", "exciting": "energetic",
    "mysterious": "mysterious", "enigmatic": "mysterious", "eerie": "mysterious",
    "haunting": "mysterious", "spooky": "mysterious", "creepy": "mysterious",
    "warm": "warm", "cozy": "warm", "inviting": "warm", "comfortable": "warm",
    "homely": "warm", "welcoming": "warm",
    "cold": "cold", "frozen": "cold", "icy": "cold", "frigid": "cold",
    "winter": "cold", "arctic": "cold",
    "romantic": "warm", "nostalgic": "warm", "dreamy": "mysterious",
    "dark": "mysterious", "gloomy": "mysterious", "somber": "cold",
    "cheerful": "energetic", "happy": "energetic", "joyful": "energetic",
    "melancholic": "cold", "sad": "cold", "lonely": "cold",
    "majestic": "dramatic", "grand": "dramatic", "magnificent": "dramatic",
    "horror": "mysterious", "terrifying": "mysterious",
    "futuristic": "energetic", "cyberpunk": "energetic", "neon": "energetic",
}

_TIME_KEYWORDS: Dict[str, str] = {
    "dawn": "dawn", "sunrise": "dawn", "first light": "dawn",
    "morning": "morning", "early morning": "dawn",
    "noon": "noon", "midday": "noon", "high noon": "noon",
    "afternoon": "afternoon",
    "sunset": "sunset", "sundown": "sunset", "dusk": "dusk",
    "evening": "dusk", "twilight": "dusk",
    "night": "night", "nighttime": "night", "dark": "night",
    "midnight": "midnight", "late night": "midnight",
    "golden hour": "sunset",
    "blue hour": "dusk",
    "overcast": "noon",
}

_CAMERA_KEYWORDS: Dict[str, str] = {
    "aerial": "bird_eye", "bird's eye": "bird_eye", "overhead": "bird_eye",
    "top down": "bird_eye", "from above": "bird_eye", "drone": "bird_eye",
    "low angle": "worm_eye", "worm's eye": "worm_eye", "from below": "worm_eye",
    "ground level": "worm_eye",
    "cinematic": "cinematic", "film": "cinematic", "movie": "cinematic",
    "wide shot": "landscape", "panoramic": "landscape", "panorama": "landscape",
    "wide angle": "landscape",
    "close up": "portrait", "closeup": "portrait", "macro": "portrait",
    "portrait": "portrait", "headshot": "portrait",
    "eye level": "eye_level", "straight on": "eye_level",
}


# ═══════════════════════════════════════════════════════════════════════════════
# LIGHTING PRESETS
# ═══════════════════════════════════════════════════════════════════════════════

_LIGHTING_PRESETS: Dict[str, Dict[str, Any]] = {
    "dawn": {
        "key": {"type": "directional", "direction": (0.3, 0.15, -0.8),
                "color": (1.0, 0.55, 0.3), "intensity": 1.2},
        "fill": {"type": "directional", "direction": (-0.5, 0.3, 0.5),
                 "color": (0.3, 0.35, 0.6), "intensity": 0.3},
        "ambient_color": (0.15, 0.12, 0.2),
        "ambient_intensity": 0.25,
        "sky_type": "sunset_gradient",
    },
    "morning": {
        "key": {"type": "directional", "direction": (0.5, 0.6, -0.5),
                "color": (1.0, 0.95, 0.85), "intensity": 2.0},
        "fill": {"type": "directional", "direction": (-0.3, 0.4, 0.3),
                 "color": (0.5, 0.6, 0.8), "intensity": 0.5},
        "ambient_color": (0.2, 0.22, 0.28),
        "ambient_intensity": 0.35,
        "sky_type": "clear",
    },
    "noon": {
        "key": {"type": "directional", "direction": (0.0, 0.95, -0.1),
                "color": (1.0, 1.0, 0.98), "intensity": 2.8},
        "fill": {"type": "directional", "direction": (0.0, 0.3, 0.5),
                 "color": (0.5, 0.55, 0.7), "intensity": 0.4},
        "ambient_color": (0.25, 0.25, 0.3),
        "ambient_intensity": 0.4,
        "sky_type": "clear",
    },
    "afternoon": {
        "key": {"type": "directional", "direction": (-0.5, 0.5, -0.4),
                "color": (1.0, 0.95, 0.8), "intensity": 2.2},
        "fill": {"type": "directional", "direction": (0.4, 0.3, 0.3),
                 "color": (0.45, 0.5, 0.65), "intensity": 0.45},
        "ambient_color": (0.22, 0.22, 0.26),
        "ambient_intensity": 0.35,
        "sky_type": "clear",
    },
    "sunset": {
        "key": {"type": "directional", "direction": (-0.8, 0.15, -0.3),
                "color": (1.0, 0.5, 0.15), "intensity": 1.8},
        "fill": {"type": "directional", "direction": (0.5, 0.3, 0.5),
                 "color": (0.25, 0.2, 0.45), "intensity": 0.3},
        "ambient_color": (0.2, 0.12, 0.15),
        "ambient_intensity": 0.25,
        "sky_type": "sunset_gradient",
    },
    "dusk": {
        "key": {"type": "directional", "direction": (-0.6, 0.08, -0.5),
                "color": (0.6, 0.35, 0.5), "intensity": 0.8},
        "fill": {"type": "directional", "direction": (0.3, 0.2, 0.4),
                 "color": (0.15, 0.15, 0.35), "intensity": 0.2},
        "ambient_color": (0.08, 0.08, 0.15),
        "ambient_intensity": 0.2,
        "sky_type": "cloudy",
    },
    "night": {
        "key": {"type": "directional", "direction": (0.3, 0.6, -0.4),
                "color": (0.4, 0.45, 0.7), "intensity": 0.5},
        "fill": {"type": "directional", "direction": (-0.3, 0.2, 0.3),
                 "color": (0.1, 0.1, 0.2), "intensity": 0.1},
        "ambient_color": (0.03, 0.03, 0.06),
        "ambient_intensity": 0.1,
        "sky_type": "clear",
    },
    "midnight": {
        "key": {"type": "directional", "direction": (0.2, 0.7, -0.3),
                "color": (0.25, 0.3, 0.55), "intensity": 0.3},
        "fill": {"type": "directional", "direction": (-0.2, 0.15, 0.4),
                 "color": (0.05, 0.05, 0.12), "intensity": 0.05},
        "ambient_color": (0.02, 0.02, 0.04),
        "ambient_intensity": 0.05,
        "sky_type": "clear",
    },
}

_MOOD_LIGHTING_ADJUST: Dict[str, Dict[str, Any]] = {
    "dramatic": {
        "key_intensity_mult": 1.4,
        "fill_intensity_mult": 0.3,
        "ambient_intensity_mult": 0.5,
    },
    "serene": {
        "key_intensity_mult": 0.8,
        "fill_intensity_mult": 1.5,
        "ambient_intensity_mult": 1.3,
    },
    "mysterious": {
        "key_intensity_mult": 0.5,
        "fill_intensity_mult": 0.4,
        "ambient_intensity_mult": 0.3,
    },
    "energetic": {
        "key_intensity_mult": 1.2,
        "fill_intensity_mult": 1.0,
        "ambient_intensity_mult": 1.1,
    },
    "warm": {
        "key_intensity_mult": 1.0,
        "fill_intensity_mult": 0.8,
        "ambient_intensity_mult": 1.0,
    },
    "cold": {
        "key_intensity_mult": 0.9,
        "fill_intensity_mult": 0.7,
        "ambient_intensity_mult": 0.6,
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# CAMERA STYLE PRESETS
# ═══════════════════════════════════════════════════════════════════════════════

_CAMERA_PRESETS: Dict[str, Dict[str, Any]] = {
    "eye_level":  {"elevation": 1.7, "distance_mult": 1.0, "fov_deg": 50.0},
    "bird_eye":   {"elevation": 15.0, "distance_mult": 1.5, "fov_deg": 60.0},
    "worm_eye":   {"elevation": 0.15, "distance_mult": 0.8, "fov_deg": 35.0},
    "cinematic":  {"elevation": 0.8, "distance_mult": 1.2, "fov_deg": 32.0},
    "portrait":   {"elevation": 1.6, "distance_mult": 0.4, "fov_deg": 85.0},
    "landscape":  {"elevation": 3.0, "distance_mult": 2.0, "fov_deg": 65.0},
}


# ═══════════════════════════════════════════════════════════════════════════════
# POST-PROCESSING PRESETS
# ═══════════════════════════════════════════════════════════════════════════════

_POST_PRESETS: Dict[str, Dict[str, float]] = {
    "serene":     {"bloom": 0.15, "vignette": 0.1, "saturation": 0.9, "contrast": 0.9, "grain": 0.0},
    "dramatic":   {"bloom": 0.3, "vignette": 0.35, "saturation": 0.85, "contrast": 1.3, "grain": 0.05},
    "energetic":  {"bloom": 0.2, "vignette": 0.15, "saturation": 1.2, "contrast": 1.1, "grain": 0.0},
    "mysterious": {"bloom": 0.1, "vignette": 0.45, "saturation": 0.6, "contrast": 1.15, "grain": 0.08},
    "warm":       {"bloom": 0.2, "vignette": 0.2, "saturation": 1.05, "contrast": 1.0, "grain": 0.02},
    "cold":       {"bloom": 0.05, "vignette": 0.25, "saturation": 0.7, "contrast": 1.1, "grain": 0.03},
}


# ═══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT KEYWORD → SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

_ENVIRONMENT_KEYWORDS: Dict[str, Dict[str, Any]] = {
    "forest":   {"ground_material": "grass", "fog_density": 0.02, "ground_type": "forest_floor"},
    "jungle":   {"ground_material": "mud", "fog_density": 0.04, "ground_type": "jungle_floor"},
    "desert":   {"ground_material": "sand", "fog_density": 0.005, "ground_type": "sand_dunes"},
    "beach":    {"ground_material": "sand", "fog_density": 0.01, "water": True, "ground_type": "beach"},
    "ocean":    {"ground_material": "water_deep", "fog_density": 0.015, "water": True, "ground_type": "ocean"},
    "lake":     {"ground_material": "grass", "fog_density": 0.01, "water": True, "ground_type": "lakeside"},
    "river":    {"ground_material": "grass", "fog_density": 0.01, "water": True, "ground_type": "riverbank"},
    "mountain": {"ground_material": "stone_granite", "fog_density": 0.008, "ground_type": "rocky"},
    "snow":     {"ground_material": "snow", "fog_density": 0.015, "ground_type": "snow_covered"},
    "tundra":   {"ground_material": "snow", "fog_density": 0.02, "ground_type": "tundra"},
    "swamp":    {"ground_material": "mud", "fog_density": 0.06, "water": True, "ground_type": "swamp"},
    "meadow":   {"ground_material": "grass", "fog_density": 0.005, "ground_type": "meadow"},
    "field":    {"ground_material": "grass", "fog_density": 0.005, "ground_type": "field"},
    "garden":   {"ground_material": "grass", "fog_density": 0.005, "ground_type": "garden"},
    "park":     {"ground_material": "grass", "fog_density": 0.005, "ground_type": "park"},
    "city":     {"ground_material": "asphalt", "fog_density": 0.01, "ground_type": "urban"},
    "street":   {"ground_material": "asphalt", "fog_density": 0.008, "ground_type": "road"},
    "road":     {"ground_material": "asphalt", "fog_density": 0.008, "ground_type": "road"},
    "highway":  {"ground_material": "asphalt", "fog_density": 0.006, "ground_type": "highway"},
    "village":  {"ground_material": "cobblestone", "fog_density": 0.01, "ground_type": "village"},
    "cave":     {"ground_material": "stone_granite", "fog_density": 0.08, "ground_type": "cave"},
    "dungeon":  {"ground_material": "stone_granite", "fog_density": 0.06, "ground_type": "dungeon"},
    "space":    {"ground_material": "none", "fog_density": 0.0, "ground_type": "void", "sky_type": "starfield"},
    "underwater": {"ground_material": "sand", "fog_density": 0.1, "water": True, "ground_type": "seabed"},
    "volcano":  {"ground_material": "stone_basalt", "fog_density": 0.04, "ground_type": "volcanic"},
    "ruins":    {"ground_material": "stone_weathered", "fog_density": 0.02, "ground_type": "ruins"},
    "graveyard": {"ground_material": "grass", "fog_density": 0.04, "ground_type": "graveyard"},
    "farm":     {"ground_material": "dirt", "fog_density": 0.005, "ground_type": "farmland"},
    "harbor":   {"ground_material": "wood_plank", "fog_density": 0.015, "water": True, "ground_type": "dock"},
    "castle":   {"ground_material": "stone_granite", "fog_density": 0.015, "ground_type": "castle_grounds"},
    "stadium":  {"ground_material": "grass", "fog_density": 0.005, "ground_type": "sports_field"},
    "studio":   {"ground_material": "concrete_smooth", "fog_density": 0.0, "ground_type": "studio"},
    "rooftop":  {"ground_material": "concrete", "fog_density": 0.008, "ground_type": "rooftop"},
    "bridge":   {"ground_material": "metal_steel", "fog_density": 0.01, "ground_type": "bridge"},
    "alley":    {"ground_material": "cobblestone", "fog_density": 0.02, "ground_type": "alley"},
    "market":   {"ground_material": "cobblestone", "fog_density": 0.01, "ground_type": "market"},
    "temple":   {"ground_material": "stone_marble", "fog_density": 0.01, "ground_type": "temple"},
    "library":  {"ground_material": "wood_oak", "fog_density": 0.0, "ground_type": "interior"},
    "classroom": {"ground_material": "linoleum", "fog_density": 0.0, "ground_type": "interior"},
    "kitchen":  {"ground_material": "tile", "fog_density": 0.0, "ground_type": "interior"},
    "bedroom":  {"ground_material": "wood_oak", "fog_density": 0.0, "ground_type": "interior"},
    "bathroom": {"ground_material": "tile", "fog_density": 0.0, "ground_type": "interior"},
    "office":   {"ground_material": "carpet", "fog_density": 0.0, "ground_type": "interior"},
}


# ═══════════════════════════════════════════════════════════════════════════════
# REFERENCE SCALES (metres) for consistency
# ═══════════════════════════════════════════════════════════════════════════════

_REFERENCE_SCALES: Dict[str, float] = {
    "human":    1.75,
    "animal":   1.0,
    "vehicle":  4.5,
    "building": 10.0,
    "nature":   8.0,
    "furniture": 0.8,
    "food":     0.1,
    "object":   0.3,
}


# ═══════════════════════════════════════════════════════════════════════════════
# SCENE GRAPH PLANNER
# ═══════════════════════════════════════════════════════════════════════════════

class SceneGraphPlanner:
    """Decomposes a text prompt into a fully-specified :class:`SceneGraph`.

    Pure NRS reasoning — no ML models, no external API calls.
    """

    def __init__(self, rng_seed: int | None = None):
        self._rng = np.random.default_rng(rng_seed)

    # ── public entry point ────────────────────────────────────────────────

    def plan(self, prompt: str) -> SceneGraph:
        """Convert *prompt* into a renderable :class:`SceneGraph`."""
        lower = prompt.lower().strip()
        tokens = lower.split()

        mood = self._detect_mood(lower)
        time_of_day = self._detect_time(lower)
        camera_style = self._detect_camera_style(lower)
        setting = self._detect_setting(lower)

        subjects = self._extract_subjects(lower, tokens)
        objects = self._compose_objects(subjects, lower)
        lighting = self._build_lighting(time_of_day, mood)
        camera = self._build_camera(camera_style, objects)
        environment = self._build_environment(setting, lower)
        post = _POST_PRESETS.get(mood, _POST_PRESETS["serene"]).copy()

        return SceneGraph(
            objects=objects,
            lighting=lighting,
            camera=camera,
            environment=environment,
            mood=mood,
            post_processing=post,
        )

    # ── semantic decomposition ────────────────────────────────────────────

    def _detect_mood(self, text: str) -> str:
        for keyword, mood in _MOOD_KEYWORDS.items():
            if keyword in text:
                return mood
        return "serene"

    def _detect_time(self, text: str) -> str:
        for phrase in sorted(_TIME_KEYWORDS, key=len, reverse=True):
            if phrase in text:
                return _TIME_KEYWORDS[phrase]
        return "noon"

    def _detect_camera_style(self, text: str) -> str:
        for phrase in sorted(_CAMERA_KEYWORDS, key=len, reverse=True):
            if phrase in text:
                return _CAMERA_KEYWORDS[phrase]
        return "eye_level"

    def _detect_setting(self, text: str) -> str:
        for phrase in sorted(_ENVIRONMENT_KEYWORDS, key=len, reverse=True):
            if phrase in text:
                return phrase
        return "field"

    # ── subject extraction ────────────────────────────────────────────────

    _IRREGULAR_PLURALS: Dict[str, str] = {
        "people": "person", "men": "man", "women": "woman",
        "children": "child", "mice": "mouse", "geese": "swan",
        "wolves": "wolf", "knives": "sword", "leaves": "fern",
        "deer": "deer", "sheep": "sheep", "fish": "fish",
        "oxen": "cow", "cacti": "cactus", "fungi": "mushroom",
    }

    def _normalize_plural(self, word: str) -> str | None:
        """Return the SUBJECT_MAP key if *word* is a known plural form."""
        if word in SUBJECT_MAP:
            return word
        if word in self._IRREGULAR_PLURALS:
            return self._IRREGULAR_PLURALS[word]
        for suffix, repl in [("ies", "y"), ("ves", "f"), ("ses", "s"),
                              ("es", ""), ("s", "")]:
            if word.endswith(suffix):
                candidate = word[:-len(suffix)] + repl
                if candidate in SUBJECT_MAP:
                    return candidate
        return None

    def _extract_subjects(
        self, text: str, tokens: List[str],
    ) -> List[Tuple[str, int, float]]:
        """Return list of (subject_key, count, size_modifier)."""
        found: List[Tuple[str, int, float]] = []

        expanded_keys: Dict[str, str] = {}
        for key in SUBJECT_MAP:
            expanded_keys[key] = key
        for word in re.findall(r"[a-z]+", text):
            if word not in expanded_keys:
                norm = self._normalize_plural(word)
                if norm:
                    expanded_keys[word] = norm

        sorted_keys = sorted(expanded_keys.keys(), key=len, reverse=True)
        remaining = text

        while remaining:
            best_key: str | None = None
            best_pos = len(remaining) + 1

            for key in sorted_keys:
                pos = remaining.find(key)
                if pos == -1:
                    continue
                at_word_boundary = (
                    (pos == 0 or not remaining[pos - 1].isalpha())
                    and (pos + len(key) >= len(remaining)
                         or not remaining[pos + len(key)].isalpha())
                )
                if not at_word_boundary:
                    continue
                if pos < best_pos or (pos == best_pos and len(key) > len(best_key or "")):
                    best_key = key
                    best_pos = pos

            if best_key is None:
                break

            resolved = expanded_keys.get(best_key, best_key)
            before = remaining[:best_pos].strip().split()
            count = self._parse_quantity(before)
            size_mod = self._parse_size(before)
            found.append((resolved, count, size_mod))
            remaining = remaining[best_pos + len(best_key):]

        if not found:
            found.append(("rock", 3, 1.0))

        return found

    def _parse_quantity(self, preceding_words: List[str]) -> int:
        for word in reversed(preceding_words):
            if word in _QUANTITY_WORDS:
                lo, hi = _QUANTITY_WORDS[word]
                return int(self._rng.integers(lo, hi + 1))
            if word.isdigit():
                return max(1, min(int(word), 50))
        return 1

    def _parse_size(self, preceding_words: List[str]) -> float:
        for word in reversed(preceding_words):
            if word in _SIZE_MODIFIERS:
                return _SIZE_MODIFIERS[word]
        return 1.0

    # ── scene composition ─────────────────────────────────────────────────

    def _compose_objects(
        self,
        subjects: List[Tuple[str, int, float]],
        text: str,
    ) -> List[SceneObject]:
        objects: List[SceneObject] = []
        total = sum(count for _, count, _ in subjects)

        thirds_x = [-2.0, 0.0, 2.0]
        thirds_z = [-1.5, 0.0, 1.5]

        obj_idx = 0
        for subj_key, count, size_mod in subjects:
            info = SUBJECT_MAP[subj_key]
            category = info["category"]
            ref_scale = _REFERENCE_SCALES.get(category, 1.0)
            base_scale = info["default_scale"] * size_mod

            for i in range(count):
                if total == 1:
                    px, pz = 0.0, 0.0
                elif total <= 3:
                    px = thirds_x[obj_idx % 3]
                    pz = 0.0
                else:
                    spread = min(total * 0.8, 15.0)
                    px = float(self._rng.uniform(-spread, spread))
                    pz = float(self._rng.uniform(-spread * 0.5, spread * 0.5))

                py = 0.0
                if category in ("animal", "human"):
                    py = 0.0
                elif category == "nature" and subj_key in ("mountain", "hill", "cliff"):
                    pz += 10.0 + float(self._rng.uniform(0, 5))
                    py = 0.0

                ry = float(self._rng.uniform(-0.3, 0.3))

                obj = SceneObject(
                    name=f"{subj_key}_{i}",
                    primitive=info["primitive"],
                    position=(px, py, pz),
                    rotation=(0.0, ry, 0.0),
                    scale=base_scale * ref_scale,
                    material=info["default_material"],
                    params={**info["default_params"]},
                )
                objects.append(obj)
                obj_idx += 1

        self._apply_spatial_relations(objects, text)
        return objects

    def _apply_spatial_relations(
        self, objects: List[SceneObject], text: str,
    ) -> None:
        """Adjust positions based on spatial prepositions found in *text*."""
        if len(objects) < 2:
            return
        for phrase in sorted(_SPATIAL_OFFSETS, key=len, reverse=True):
            if phrase in text:
                dx, dy, dz = _SPATIAL_OFFSETS[phrase]
                anchor = objects[0]
                ax, ay, az = anchor.position
                for obj in objects[1:]:
                    ox, oy, oz = obj.position
                    jitter_x = float(self._rng.uniform(-0.3, 0.3))
                    jitter_z = float(self._rng.uniform(-0.3, 0.3))
                    obj.position = (
                        ax + dx + jitter_x,
                        ay + dy,
                        az + dz + jitter_z,
                    )
                break

    # ── lighting ──────────────────────────────────────────────────────────

    def _build_lighting(self, time_of_day: str, mood: str) -> SceneLighting:
        preset = _LIGHTING_PRESETS.get(time_of_day, _LIGHTING_PRESETS["noon"])
        adjust = _MOOD_LIGHTING_ADJUST.get(mood, {})

        key = dict(preset["key"])
        fill = dict(preset["fill"])
        key["intensity"] *= adjust.get("key_intensity_mult", 1.0)
        fill["intensity"] *= adjust.get("fill_intensity_mult", 1.0)

        amb_color = preset["ambient_color"]
        amb_intensity = preset["ambient_intensity"] * adjust.get("ambient_intensity_mult", 1.0)

        return SceneLighting(
            lights=[key, fill],
            ambient_color=amb_color,
            ambient_intensity=amb_intensity,
            time_of_day=time_of_day,
            sky_type=preset["sky_type"],
        )

    # ── camera ────────────────────────────────────────────────────────────

    def _build_camera(
        self, style: str, objects: List[SceneObject],
    ) -> SceneCamera:
        preset = _CAMERA_PRESETS.get(style, _CAMERA_PRESETS["eye_level"])

        if objects:
            xs = [o.position[0] for o in objects]
            ys = [o.position[1] for o in objects]
            zs = [o.position[2] for o in objects]
            cx = (min(xs) + max(xs)) / 2.0
            cy = (min(ys) + max(ys)) / 2.0
            cz = (min(zs) + max(zs)) / 2.0

            span_x = max(xs) - min(xs) + 2.0
            span_z = max(zs) - min(zs) + 2.0
            max_scale = max(o.scale for o in objects)
            scene_radius = max(span_x, span_z, max_scale * 2.0) * 0.6

            cam_dist = max(scene_radius * preset["distance_mult"], 3.0) * 1.2
        else:
            cx, cy, cz = 0.0, 0.0, 0.0
            cam_dist = 8.0

        elev = preset["elevation"]
        cam_pos = (cx, elev, cz - cam_dist)
        cam_target = (cx, cy + elev * 0.3, cz)

        return SceneCamera(
            position=cam_pos,
            target=cam_target,
            fov_deg=preset["fov_deg"],
            style=style,
        )

    # ── environment ───────────────────────────────────────────────────────

    def _build_environment(self, setting: str, text: str) -> Dict[str, Any]:
        base = _ENVIRONMENT_KEYWORDS.get(setting, _ENVIRONMENT_KEYWORDS["field"]).copy()

        if "fog" in text or "foggy" in text:
            base["fog_density"] = max(base.get("fog_density", 0), 0.06)
        if "rain" in text or "rainy" in text:
            base["rain"] = True
        if "snow" in text or "snowy" in text:
            base["snow"] = True
        if "wind" in text or "windy" in text:
            base["wind_strength"] = 0.6

        return base


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    "SceneObject",
    "SceneLighting",
    "SceneCamera",
    "SceneGraph",
    "SceneGraphPlanner",
    "SUBJECT_MAP",
]
