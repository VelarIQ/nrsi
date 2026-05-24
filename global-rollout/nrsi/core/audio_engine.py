"""NRS-Native Audio Generation Engine — Complete Production Pipeline.

Pure-math, zero-ML audio generation covering:
  1. Grapheme-to-Phoneme (G2P) with 5000+ word CMU-style dictionary
  2. NRS Vocoder — source-filter speech synthesis with prosody engine
  3. Music Composition — theory-driven chord/melody/arrangement engine
  4. Physical Modeling Instruments — waveguide strings, piano, guitar,
     brass, woodwind, drums
  5. Audio Master Chain — multiband compression, EQ, reverb, limiter
  6. Speech Recognition — mel spectrogram, formant tracking, Viterbi

No external ML libraries.  numpy + scipy only.
"""
from __future__ import annotations

import io
import math
import os
import re
import struct
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import butter, sosfilt, lfilter, firwin, fftconvolve

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

SPEECH_SR = 24000
MUSIC_SR = 44100
N_MEL = 80
MEL_FMIN = 50.0
MEL_FMAX = 11000.0

_TWO_PI = 2.0 * math.pi


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PhonemeToken:
    phoneme: str
    stress: int = 0          # 0=none, 1=primary, 2=secondary
    duration_ms: float = 0.0
    is_pause: bool = False
    phrase_boundary: bool = False


@dataclass
class VoiceProfile:
    pitch_base: float = 130.0
    pitch_range: float = 0.3
    formant_shift: float = 1.0
    breathiness: float = 0.02
    vibrato_rate: float = 5.5
    vibrato_depth: float = 0.01
    jitter: float = 0.003
    shimmer: float = 0.02

    @classmethod
    def male_default(cls) -> VoiceProfile:
        return cls(pitch_base=120.0, pitch_range=0.25, formant_shift=1.0,
                   breathiness=0.015, vibrato_rate=5.2, vibrato_depth=0.008,
                   jitter=0.003, shimmer=0.02)

    @classmethod
    def female_default(cls) -> VoiceProfile:
        return cls(pitch_base=200.0, pitch_range=0.35, formant_shift=1.15,
                   breathiness=0.025, vibrato_rate=5.8, vibrato_depth=0.012,
                   jitter=0.002, shimmer=0.015)

    @classmethod
    def child(cls) -> VoiceProfile:
        return cls(pitch_base=280.0, pitch_range=0.4, formant_shift=1.25,
                   breathiness=0.03, vibrato_rate=6.0, vibrato_depth=0.015,
                   jitter=0.004, shimmer=0.025)

    @classmethod
    def deep_male(cls) -> VoiceProfile:
        return cls(pitch_base=85.0, pitch_range=0.2, formant_shift=0.85,
                   breathiness=0.01, vibrato_rate=4.8, vibrato_depth=0.006,
                   jitter=0.003, shimmer=0.02)

    @classmethod
    def narrator(cls) -> VoiceProfile:
        return cls(pitch_base=140.0, pitch_range=0.2, formant_shift=0.95,
                   breathiness=0.008, vibrato_rate=5.0, vibrato_depth=0.005,
                   jitter=0.002, shimmer=0.012)


@dataclass
class MusicScore:
    tempo: int = 120
    time_sig: Tuple[int, int] = (4, 4)
    key: str = "C"
    scale: str = "major"
    chords: List[Tuple[int, str, float]] = field(default_factory=list)
    melody: List[Tuple[int, float, float]] = field(default_factory=list)
    bass: List[Tuple[int, float, float]] = field(default_factory=list)
    drums: List[Tuple[str, float, float]] = field(default_factory=list)
    duration: float = 30.0


@dataclass
class WordResult:
    text: str
    start_time: float
    end_time: float
    confidence: float


@dataclass
class TranscriptionResult:
    text: str
    words: List[WordResult]
    trust_level: str
    language: str
    reasoning_chain: List[str]


# ═══════════════════════════════════════════════════════════════════════════════
# PHONEME DATA — ARPAbet formant targets
# ═══════════════════════════════════════════════════════════════════════════════

# (F1, F2, F3, BW1, BW2, BW3, voiced, fric_amp, nasal, base_dur_ms)
_PHONEME_PARAMS: Dict[str, Tuple] = {
    "AA": (730, 1090, 2440,  90, 110, 170, True,  0.0,  False, 100),
    "AE": (660, 1720, 2410,  80, 100, 170, True,  0.0,  False, 100),
    "AH": (520, 1190, 2390,  80, 100, 150, True,  0.0,  False,  70),
    "AO": (570,  840, 2410,  80, 100, 150, True,  0.0,  False, 100),
    "AW": (730, 1090, 2440,  90, 110, 170, True,  0.0,  False, 130),
    "AX": (500, 1500, 2490,  80, 100, 160, True,  0.0,  False,  50),
    "AY": (730, 1090, 2440,  90, 110, 170, True,  0.0,  False, 130),
    "EH": (530, 1840, 2480,  70, 100, 140, True,  0.0,  False,  85),
    "ER": (490, 1350, 1690,  80, 100, 160, True,  0.0,  False,  90),
    "EY": (400, 2100, 2660,  60, 100, 130, True,  0.0,  False, 110),
    "IH": (390, 1990, 2550,  60, 100, 140, True,  0.0,  False,  60),
    "IY": (270, 2290, 3010,  60, 100, 120, True,  0.0,  False,  90),
    "OW": (570,  840, 2410,  80, 100, 150, True,  0.0,  False, 110),
    "OY": (570,  840, 2410,  80, 100, 150, True,  0.0,  False, 130),
    "UH": (440, 1020, 2240,  70, 100, 140, True,  0.0,  False,  70),
    "UW": (300,  870, 2240,  70, 100, 130, True,  0.0,  False,  90),
    # Stops voiced
    "B":  (200, 1100, 2150,  60,  90, 150, True,  0.0,  False,  15),
    "D":  (200, 1600, 2600,  60,  90, 150, True,  0.0,  False,  12),
    "G":  (200, 1990, 2850,  60,  90, 150, True,  0.0,  False,  12),
    # Stops unvoiced
    "P":  (200, 1100, 2150,  60,  90, 150, False, 0.15, False,  15),
    "T":  (200, 1600, 2600,  60,  90, 150, False, 0.20, False,  12),
    "K":  (200, 1990, 2850,  60,  90, 150, False, 0.18, False,  12),
    # Fricatives unvoiced
    "F":  (400, 1400, 2400, 200, 300, 400, False, 0.45, False,  90),
    "S":  (400, 1400, 6000, 200, 300, 500, False, 0.55, False, 100),
    "SH": (400, 1800, 2600, 200, 300, 400, False, 0.50, False, 100),
    "TH": (400, 1400, 2200, 200, 300, 400, False, 0.35, False,  70),
    "HH": (500, 1500, 2500, 200, 300, 400, False, 0.30, False,  50),
    "CH": (400, 1800, 2600, 200, 300, 400, False, 0.50, False,  80),
    # Fricatives voiced
    "V":  (300, 1100, 2400, 200, 300, 400, True,  0.25, False,  70),
    "Z":  (300, 1400, 6000, 200, 300, 500, True,  0.35, False,  80),
    "ZH": (300, 1800, 2600, 200, 300, 400, True,  0.30, False,  70),
    "DH": (300, 1400, 2200, 200, 300, 400, True,  0.20, False,  50),
    "JH": (300, 1800, 2600, 200, 300, 400, True,  0.30, False,  80),
    # Nasals
    "M":  (300, 1100, 2200, 100, 150, 200, True,  0.0,  True,   80),
    "N":  (300, 1450, 2200, 100, 150, 200, True,  0.0,  True,   70),
    "NG": (300, 1990, 2850, 100, 150, 200, True,  0.0,  True,   70),
    # Liquids / glides
    "L":  (350, 1100, 2700,  80, 120, 160, True,  0.0,  False,  60),
    "R":  (350, 1300, 1600,  80, 120, 160, True,  0.0,  False,  60),
    "W":  (300,  700, 2200,  80, 120, 160, True,  0.0,  False,  40),
    "Y":  (270, 2290, 3010,  80, 120, 160, True,  0.0,  False,  40),
    # Silence / pause
    "SIL": (0, 0, 0, 0, 0, 0, False, 0.0, False, 60),
    "PAU": (0, 0, 0, 0, 0, 0, False, 0.0, False, 200),
}

_VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AX", "AY", "EH", "ER",
    "EY", "IH", "IY", "OW", "OY", "UH", "UW",
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. GRAPHEME-TO-PHONEME ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

# Compact dictionary: "word:PH0 PH1 PH2|word2:PH0 PH1"
# Stress markers: 0=no stress, 1=primary, 2=secondary on vowels
_DICT_DATA = (
    "a:AH0|the:DH AH0|and:AE1 N D|of:AH0 V|to:T UW1|in:IH1 N|is:IH1 Z|"
    "it:IH1 T|that:DH AE1 T|was:W AH1 Z|for:F AO1 R|on:AA1 N|are:AA1 R|"
    "with:W IH1 TH|as:AE1 Z|i:AY1|his:HH IH1 Z|they:DH EY1|be:B IY1|"
    "at:AE1 T|one:W AH1 N|have:HH AE1 V|this:DH IH1 S|from:F R AH1 M|"
    "or:AO1 R|had:HH AE1 D|by:B AY1|not:N AA1 T|but:B AH1 T|"
    "what:W AH1 T|all:AO1 L|were:W ER1|we:W IY1|when:W EH1 N|"
    "your:Y AO1 R|can:K AE1 N|said:S EH1 D|there:DH EH1 R|"
    "each:IY1 CH|which:W IH1 CH|do:D UW1|their:DH EH1 R|"
    "if:IH1 F|will:W IH1 L|up:AH1 P|other:AH1 DH ER0|about:AH0 B AW1 T|"
    "out:AW1 T|many:M EH1 N IY0|then:DH EH1 N|them:DH EH1 M|"
    "these:DH IY1 Z|so:S OW1|some:S AH1 M|her:HH ER1|"
    "would:W UH1 D|make:M EY1 K|like:L AY1 K|him:HH IH1 M|"
    "into:IH1 N T UW0|time:T AY1 M|has:HH AE1 Z|look:L UH1 K|"
    "two:T UW1|more:M AO1 R|write:R AY1 T|go:G OW1|"
    "see:S IY1|number:N AH1 M B ER0|no:N OW1|way:W EY1|"
    "could:K UH1 D|people:P IY1 P AH0 L|my:M AY1|than:DH AE1 N|"
    "first:F ER1 S T|water:W AO1 T ER0|been:B IH1 N|call:K AO1 L|"
    "who:HH UW1|oil:OY1 L|its:IH1 T S|now:N AW1|find:F AY1 N D|"
    "long:L AO1 NG|down:D AW1 N|day:D EY1|did:D IH1 D|get:G EH1 T|"
    "come:K AH1 M|made:M EY1 D|may:M EY1|part:P AA1 R T|"
    "over:OW1 V ER0|new:N UW1|sound:S AW1 N D|take:T EY1 K|"
    "only:OW1 N L IY0|little:L IH1 T AH0 L|work:W ER1 K|"
    "know:N OW1|place:P L EY1 S|year:Y IH1 R|live:L IH1 V|"
    "me:M IY1|back:B AE1 K|give:G IH1 V|most:M OW1 S T|"
    "very:V EH1 R IY0|after:AE1 F T ER0|thing:TH IH1 NG|"
    "our:AW1 ER0|just:JH AH1 S T|name:N EY1 M|good:G UH1 D|"
    "sentence:S EH1 N T AH0 N S|man:M AE1 N|think:TH IH1 NG K|"
    "say:S EY1|great:G R EY1 T|where:W EH1 R|help:HH EH1 L P|"
    "through:TH R UW1|much:M AH1 CH|before:B IH0 F AO1 R|"
    "line:L AY1 N|right:R AY1 T|too:T UW1|mean:M IY1 N|"
    "old:OW1 L D|any:EH1 N IY0|same:S EY1 M|tell:T EH1 L|"
    "boy:B OY1|follow:F AA1 L OW0|came:K EY1 M|want:W AA1 N T|"
    "show:SH OW1|also:AO1 L S OW0|around:ER0 AW1 N D|"
    "form:F AO1 R M|three:TH R IY1|small:S M AO1 L|"
    "set:S EH1 T|put:P UH1 T|end:EH1 N D|does:D AH1 Z|"
    "another:AH0 N AH1 DH ER0|well:W EH1 L|large:L AA1 R JH|"
    "must:M AH1 S T|big:B IH1 G|even:IY1 V AH0 N|such:S AH1 CH|"
    "because:B IH0 K AH1 Z|turn:T ER1 N|here:HH IY1 R|"
    "why:W AY1|ask:AE1 S K|went:W EH1 N T|men:M EH1 N|"
    "read:R IY1 D|need:N IY1 D|land:L AE1 N D|different:D IH1 F ER0 AH0 N T|"
    "home:HH OW1 M|us:AH1 S|move:M UW1 V|try:T R AY1|"
    "kind:K AY1 N D|hand:HH AE1 N D|picture:P IH1 K CH ER0|"
    "again:AH0 G EH1 N|change:CH EY1 N JH|off:AO1 F|play:P L EY1|"
    "spell:S P EH1 L|air:EH1 R|away:AH0 W EY1|animal:AE1 N AH0 M AH0 L|"
    "house:HH AW1 S|point:P OY1 N T|page:P EY1 JH|letter:L EH1 T ER0|"
    "mother:M AH1 DH ER0|answer:AE1 N S ER0|found:F AW1 N D|"
    "study:S T AH1 D IY0|still:S T IH1 L|learn:L ER1 N|"
    "should:SH UH1 D|america:AH0 M EH1 R IH0 K AH0|world:W ER1 L D|"
    "high:HH AY1|every:EH1 V R IY0|near:N IH1 R|add:AE1 D|"
    "food:F UW1 D|between:B IH0 T W IY1 N|own:OW1 N|"
    "below:B IH0 L OW1|country:K AH1 N T R IY0|plant:P L AE1 N T|"
    "last:L AE1 S T|school:S K UW1 L|father:F AA1 DH ER0|"
    "keep:K IY1 P|tree:T R IY1|never:N EH1 V ER0|start:S T AA1 R T|"
    "city:S IH1 T IY0|earth:ER1 TH|eye:AY1|light:L AY1 T|"
    "thought:TH AO1 T|head:HH EH1 D|under:AH1 N D ER0|"
    "story:S T AO1 R IY0|saw:S AO1|left:L EH1 F T|"
    "few:F Y UW1|while:W AY1 L|along:AH0 L AO1 NG|"
    "might:M AY1 T|close:K L OW1 Z|something:S AH1 M TH IH0 NG|"
    "seem:S IY1 M|next:N EH1 K S T|hard:HH AA1 R D|"
    "open:OW1 P AH0 N|example:IH0 G Z AE1 M P AH0 L|"
    "begin:B IH0 G IH1 N|life:L AY1 F|always:AO1 L W EY0 Z|"
    "those:DH OW1 Z|both:B OW1 TH|paper:P EY1 P ER0|"
    "together:T AH0 G EH1 DH ER0|got:G AA1 T|group:G R UW1 P|"
    "often:AO1 F AH0 N|run:R AH1 N|important:IH0 M P AO1 R T AH0 N T|"
    "until:AH0 N T IH1 L|children:CH IH1 L D R AH0 N|"
    "side:S AY1 D|feet:F IY1 T|car:K AA1 R|mile:M AY1 L|"
    "night:N AY1 T|walk:W AO1 K|white:W AY1 T|sea:S IY1|"
    "began:B IH0 G AE1 N|grow:G R OW1|took:T UH1 K|"
    "river:R IH1 V ER0|four:F AO1 R|carry:K AE1 R IY0|"
    "state:S T EY1 T|once:W AH1 N S|book:B UH1 K|"
    "hear:HH IY1 R|stop:S T AA1 P|without:W IH0 DH AW1 T|"
    "second:S EH1 K AH0 N D|late:L EY1 T|miss:M IH1 S|"
    "idea:AY0 D IY1 AH0|enough:IH0 N AH1 F|eat:IY1 T|"
    "face:F EY1 S|watch:W AA1 CH|far:F AA1 R|"
    "real:R IY1 L|almost:AO1 L M OW0 S T|let:L EH1 T|"
    "above:AH0 B AH1 V|girl:G ER1 L|sometimes:S AH1 M T AY2 M Z|"
    "mountain:M AW1 N T AH0 N|cut:K AH1 T|young:Y AH1 NG|"
    "talk:T AO1 K|soon:S UW1 N|list:L IH1 S T|"
    "song:S AO1 NG|being:B IY1 IH0 NG|leave:L IY1 V|"
    "family:F AE1 M AH0 L IY0|body:B AA1 D IY0|"
    "music:M Y UW1 Z IH0 K|color:K AH1 L ER0|"
    "stand:S T AE1 N D|sun:S AH1 N|question:K W EH1 S CH AH0 N|"
    "fish:F IH1 SH|area:EH1 R IY0 AH0|mark:M AA1 R K|"
    "dog:D AO1 G|horse:HH AO1 R S|bird:B ER1 D|"
    "problem:P R AA1 B L AH0 M|complete:K AH0 M P L IY1 T|"
    "room:R UW1 M|knew:N UW1|since:S IH1 N S|"
    "ever:EH1 V ER0|piece:P IY1 S|told:T OW1 L D|"
    "usually:Y UW1 ZH AH0 L IY0|didn't:D IH1 D AH0 N T|"
    "friends:F R EH1 N D Z|easy:IY1 Z IY0|heard:HH ER1 D|"
    "order:AO1 R D ER0|red:R EH1 D|door:D AO1 R|"
    "sure:SH UH1 R|become:B IH0 K AH1 M|top:T AA1 P|"
    "ship:SH IH1 P|across:AH0 K R AO1 S|today:T AH0 D EY1|"
    "during:D UH1 R IH0 NG|short:SH AO1 R T|better:B EH1 T ER0|"
    "best:B EH1 S T|however:HH AW0 EH1 V ER0|low:L OW1|"
    "hours:AW1 ER0 Z|black:B L AE1 K|products:P R AA1 D AH0 K T S|"
    "happened:HH AE1 P AH0 N D|whole:HH OW1 L|measure:M EH1 ZH ER0|"
    "remember:R IH0 M EH1 M B ER0|early:ER1 L IY0|"
    "reach:R IY1 CH|rest:R EH1 S T|nothing:N AH1 TH IH0 NG|"
    "able:EY1 B AH0 L|money:M AH1 N IY0|serve:S ER1 V|"
    "voice:V OY1 S|power:P AW1 ER0|town:T AW1 N|"
    "fine:F AY1 N|certain:S ER1 T AH0 N|fly:F L AY1|"
    "unit:Y UW1 N AH0 T|lead:L IY1 D|cry:K R AY1|"
    "dark:D AA1 R K|machine:M AH0 SH IY1 N|note:N OW1 T|"
    "wait:W EY1 T|plan:P L AE1 N|figure:F IH1 G Y ER0|"
    "star:S T AA1 R|box:B AA1 K S|noun:N AW1 N|"
    "field:F IY1 L D|interest:IH1 N T R AH0 S T|"
    "building:B IH1 L D IH0 NG|half:HH AE1 F|"
    "class:K L AE1 S|behind:B IH0 HH AY1 N D|"
    "clear:K L IY1 R|map:M AE1 P|table:T EY1 B AH0 L|"
    "hundred:HH AH1 N D R AH0 D|among:AH0 M AH1 NG|"
    "free:F R IY1|feel:F IY1 L|true:T R UW1|"
    "during:D UH1 R IH0 NG|full:F UH1 L|"
    "five:F AY1 V|six:S IH1 K S|seven:S EH1 V AH0 N|"
    "eight:EY1 T|nine:N AY1 N|ten:T EH1 N|"
    "yes:Y EH1 S|perhaps:P ER0 HH AE1 P S|"
    "possible:P AA1 S AH0 B AH0 L|actually:AE1 K CH UW0 AH0 L IY0|"
    "already:AO0 L R EH1 D IY0|believe:B IH0 L IY1 V|"
    "minute:M IH1 N AH0 T|continue:K AH0 N T IH1 N Y UW0|"
    "result:R IH0 Z AH1 L T|information:IH2 N F ER0 M EY1 SH AH0 N|"
    "program:P R OW1 G R AE2 M|language:L AE1 NG G W IH0 JH|"
    "computer:K AH0 M P Y UW1 T ER0|human:HH Y UW1 M AH0 N|"
    "reason:R IY1 Z AH0 N|level:L EH1 V AH0 L|"
    "inside:IH0 N S AY1 D|outside:AW1 T S AY1 D|"
    "toward:T AH0 W AO1 R D|within:W IH0 DH IH1 N|"
    "against:AH0 G EH1 N S T|listen:L IH1 S AH0 N|"
    "happen:HH AE1 P AH0 N|whether:W EH1 DH ER0|"
    "cannot:K AE1 N AA0 T|probably:P R AA1 B AH0 B L IY0|"
    "million:M IH1 L Y AH0 N|market:M AA1 R K AH0 T|"
    "industry:IH1 N D AH0 S T R IY0|report:R IH0 P AO1 R T|"
    "moment:M OW1 M AH0 N T|public:P AH1 B L IH0 K|"
    "known:N OW1 N|special:S P EH1 SH AH0 L|"
    "single:S IH1 NG G AH0 L|personal:P ER1 S AH0 N AH0 L|"
    "create:K R IY0 EY1 T|present:P R EH1 Z AH0 N T|"
    "value:V AE1 L Y UW0|develop:D IH0 V EH1 L AH0 P|"
    "provide:P R AH0 V AY1 D|service:S ER1 V AH0 S|"
    "general:JH EH1 N ER0 AH0 L|major:M EY1 JH ER0|"
    "data:D EY1 T AH0|process:P R AA1 S EH0 S|"
    "model:M AA1 D AH0 L|system:S IH1 S T AH0 M|"
    "network:N EH1 T W ER2 K|design:D IH0 Z AY1 N|"
    "digital:D IH1 JH AH0 T AH0 L|signal:S IH1 G N AH0 L|"
    "frequency:F R IY1 K W AH0 N S IY0|analysis:AH0 N AE1 L AH0 S IH0 S|"
    "response:R IH0 S P AA1 N S|control:K AH0 N T R OW1 L|"
    "output:AW1 T P UH2 T|input:IH1 N P UH2 T|"
    "pattern:P AE1 T ER0 N|performance:P ER0 F AO1 R M AH0 N S|"
    "energy:EH1 N ER0 JH IY0|natural:N AE1 CH ER0 AH0 L|"
    "physical:F IH1 Z IH0 K AH0 L|neural:N UH1 R AH0 L|"
    "audio:AO1 D IY0 OW0|video:V IH1 D IY0 OW0|"
    "image:IH1 M AH0 JH|render:R EH1 N D ER0|"
    "engine:EH1 N JH AH0 N|speech:S P IY1 CH|"
    "quality:K W AA1 L AH0 T IY0|generate:JH EH1 N ER0 EY2 T|"
    "synthesis:S IH1 N TH AH0 S IH0 S|harmonic:HH AA0 R M AA1 N IH0 K|"
    "spectral:S P EH1 K T R AH0 L|envelope:EH1 N V AH0 L OW2 P|"
    "resonance:R EH1 Z AH0 N AH0 N S|modulation:M AA2 JH AH0 L EY1 SH AH0 N|"
    "amplitude:AE1 M P L AH0 T UW2 D|waveform:W EY1 V F AO2 R M|"
    "formant:F AO1 R M AH0 N T|phoneme:F OW1 N IY2 M|"
    "prosody:P R AA1 S AH0 D IY0|acoustic:AH0 K UW1 S T IH0 K|"
    "technology:T EH0 K N AA1 L AH0 JH IY0|"
    "beautiful:B Y UW1 T AH0 F AH0 L|happy:HH AE1 P IY0|"
    "ready:R EH1 D IY0|done:D AH1 N|everything:EH1 V R IY0 TH IH2 NG|"
    "nothing:N AH1 TH IH0 NG|something:S AH1 M TH IH0 NG|"
    "morning:M AO1 R N IH0 NG|evening:IY1 V N IH0 NG|"
    "welcome:W EH1 L K AH0 M|please:P L IY1 Z|"
    "thank:TH AE1 NG K|thanks:TH AE1 NG K S|"
    "testing:T EH1 S T IH0 NG|hello:HH AH0 L OW1|"
    "sorry:S AA1 R IY0|love:L AH1 V|"
    "speed:S P IY1 D|drive:D R AY1 V|fast:F AE1 S T|"
    "woman:W UH1 M AH0 N|child:CH AY1 L D|"
    "word:W ER1 D|number:N AH1 M B ER0|"
    "really:R IY1 L IY0|understand:AH2 N D ER0 S T AE1 N D|"
    "himself:HH IH0 M S EH1 L F|government:G AH1 V ER0 N M AH0 N T|"
    "company:K AH1 M P AH0 N IY0|business:B IH1 Z N AH0 S|"
    "president:P R EH1 Z AH0 D AH0 N T|"
    "different:D IH1 F ER0 AH0 N T|"
    "education:EH2 JH AH0 K EY1 SH AH0 N|"
    "development:D IH0 V EH1 L AH0 P M AH0 N T|"
    "experience:IH0 K S P IH1 R IY0 AH0 N S|"
    "political:P AH0 L IH1 T AH0 K AH0 L|"
    "national:N AE1 SH AH0 N AH0 L|"
    "international:IH2 N T ER0 N AE1 SH AH0 N AH0 L|"
    "community:K AH0 M Y UW1 N AH0 T IY0|"
    "university:Y UW2 N AH0 V ER1 S AH0 T IY0|"
    "research:R IH0 S ER1 CH|available:AH0 V EY1 L AH0 B AH0 L|"
    "management:M AE1 N AH0 JH M AH0 N T|"
    "security:S IH0 K Y UH1 R AH0 T IY0|"
    "technology:T EH0 K N AA1 L AH0 JH IY0|"
    "environment:EH0 N V AY1 R AH0 N M AH0 N T|"
    "economic:IY2 K AH0 N AA1 M IH0 K|"
    "financial:F AH0 N AE1 N SH AH0 L|"
    "military:M IH1 L AH0 T EH2 R IY0|"
    "situation:S IH2 CH UW0 EY1 SH AH0 N|"
    "condition:K AH0 N D IH1 SH AH0 N|"
    "position:P AH0 Z IH1 SH AH0 N|"
    "attention:AH0 T EH1 N SH AH0 N|"
    "direction:D ER0 EH1 K SH AH0 N|"
    "production:P R AH0 D AH1 K SH AH0 N|"
    "population:P AA2 P Y AH0 L EY1 SH AH0 N|"
    "operation:AA2 P ER0 EY1 SH AH0 N|"
    "organization:AO2 R G AH0 N AH0 Z EY1 SH AH0 N|"
    "application:AE2 P L AH0 K EY1 SH AH0 N|"
    "communication:K AH0 M Y UW2 N AH0 K EY1 SH AH0 N|"
    "generation:JH EH2 N ER0 EY1 SH AH0 N|"
    "education:EH2 JH AH0 K EY1 SH AH0 N|"
    "information:IH2 N F ER0 M EY1 SH AH0 N|"
    "decision:D IH0 S IH1 ZH AH0 N|"
    "television:T EH1 L AH0 V IH2 ZH AH0 N|"
    "professional:P R AH0 F EH1 SH AH0 N AH0 L|"
    "traditional:T R AH0 D IH1 SH AH0 N AH0 L|"
    "individual:IH2 N D AH0 V IH1 JH UW0 AH0 L|"
    "particular:P ER0 T IH1 K Y AH0 L ER0|"
    "significant:S IH0 G N IH1 F AH0 K AH0 N T|"
    "american:AH0 M EH1 R AH0 K AH0 N|"
    "european:Y UH2 R AH0 P IY1 AH0 N|"
    "social:S OW1 SH AH0 L|medical:M EH1 D AH0 K AH0 L|"
    "local:L OW1 K AH0 L|legal:L IY1 G AH0 L|"
    "central:S EH1 N T R AH0 L|final:F AY1 N AH0 L|"
    "simple:S IH1 M P AH0 L|similar:S IH1 M AH0 L ER0|"
    "common:K AA1 M AH0 N|current:K ER1 AH0 N T|"
    "recent:R IY1 S AH0 N T|modern:M AA1 D ER0 N|"
    "future:F Y UW1 CH ER0|private:P R AY1 V AH0 T|"
    "various:V EH1 R IY0 AH0 S|entire:EH0 N T AY1 ER0|"
    "serious:S IH1 R IY0 AH0 S|possible:P AA1 S AH0 B AH0 L|"
    "necessary:N EH1 S AH0 S EH2 R IY0|"
    "difficult:D IH1 F AH0 K AH0 L T|"
    "wonderful:W AH1 N D ER0 F AH0 L|"
    "terrible:T EH1 R AH0 B AH0 L|"
    "excellent:EH1 K S AH0 L AH0 N T|"
    "impossible:IH0 M P AA1 S AH0 B AH0 L|"
    "responsible:R IH0 S P AA1 N S AH0 B AH0 L|"
    "comfortable:K AH1 M F ER0 T AH0 B AH0 L|"
    "interesting:IH1 N T R AH0 S T IH0 NG|"
    "absolutely:AE2 B S AH0 L UW1 T L IY0|"
    "especially:IH0 S P EH1 SH AH0 L IY0|"
    "immediately:IH0 M IY1 D IY0 AH0 T L IY0|"
    "eventually:IH0 V EH1 N CH UW0 AH0 L IY0|"
    "apparently:AH0 P EH1 R AH0 N T L IY0|"
    "obviously:AA1 B V IY0 AH0 S L IY0|"
    "basically:B EY1 S IH0 K L IY0|"
    "generally:JH EH1 N ER0 AH0 L IY0|"
    "specifically:S P AH0 S IH1 F IH0 K L IY0|"
    "particularly:P ER0 T IH1 K Y AH0 L ER0 L IY0|"
    "certainly:S ER1 T AH0 N L IY0|"
    "definitely:D EH1 F AH0 N AH0 T L IY0|"
    "completely:K AH0 M P L IY1 T L IY0|"
    "suddenly:S AH1 D AH0 N L IY0|"
    "quickly:K W IH1 K L IY0|slowly:S L OW1 L IY0|"
    "simply:S IH1 M P L IY0|clearly:K L IH1 R L IY0|"
    "exactly:IH0 G Z AE1 K T L IY0|"
    "directly:D ER0 EH1 K T L IY0|"
    "recently:R IY1 S AH0 N T L IY0|"
    "currently:K ER1 AH0 N T L IY0|"
    "previously:P R IY1 V IY0 AH0 S L IY0|"
    "actually:AE1 K CH UW0 AH0 L IY0|"
    "finally:F AY1 N AH0 L IY0|"
    "probably:P R AA1 B AH0 B L IY0|"
    "seriously:S IH1 R IY0 AH0 S L IY0|"
    # Numbers
    "zero:Z IH1 R OW0|eleven:IH0 L EH1 V AH0 N|"
    "twelve:T W EH1 L V|thirteen:TH ER1 T IY2 N|"
    "fourteen:F AO1 R T IY2 N|fifteen:F IH0 F T IY1 N|"
    "sixteen:S IH0 K S T IY1 N|seventeen:S EH2 V AH0 N T IY1 N|"
    "eighteen:EY1 T IY2 N|nineteen:N AY1 N T IY2 N|"
    "twenty:T W EH1 N T IY0|thirty:TH ER1 T IY0|"
    "forty:F AO1 R T IY0|fifty:F IH1 F T IY0|"
    "sixty:S IH1 K S T IY0|seventy:S EH1 V AH0 N T IY0|"
    "eighty:EY1 T IY0|ninety:N AY1 N T IY0|"
    "thousand:TH AW1 Z AH0 N D|"
    # Common names
    "john:JH AA1 N|james:JH EY1 M Z|robert:R AA1 B ER0 T|"
    "michael:M AY1 K AH0 L|william:W IH1 L Y AH0 M|"
    "david:D EY1 V IH0 D|richard:R IH1 CH ER0 D|"
    "joseph:JH OW1 S AH0 F|thomas:T AA1 M AH0 S|"
    "charles:CH AA1 R L Z|mary:M EH1 R IY0|"
    "patricia:P AH0 T R IH1 SH AH0|jennifer:JH EH1 N AH0 F ER0|"
    "elizabeth:IH0 L IH1 Z AH0 B AH0 TH|"
    "linda:L IH1 N D AH0|barbara:B AA1 R B ER0 AH0|"
    "susan:S UW1 Z AH0 N|jessica:JH EH1 S IH0 K AH0|"
    "sarah:S EH1 R AH0|karen:K EH1 R AH0 N|"
    "daniel:D AE1 N Y AH0 L|matthew:M AE1 TH Y UW0|"
    "anthony:AE1 N TH AH0 N IY0|mark:M AA1 R K|"
    "donald:D AA1 N AH0 L D|steven:S T IY1 V AH0 N|"
    "paul:P AO1 L|andrew:AE1 N D R UW0|"
    "joshua:JH AA1 SH UW0 AH0|kenneth:K EH1 N AH0 TH|"
    "kevin:K EH1 V AH0 N|brian:B R AY1 AH0 N|"
    "george:JH AO1 R JH|timothy:T IH1 M AH0 TH IY0|"
    "ronald:R AA1 N AH0 L D|edward:EH1 D W ER0 D|"
    "jason:JH EY1 S AH0 N|jeffrey:JH EH1 F R IY0|"
    "ryan:R AY1 AH0 N|jacob:JH EY1 K AH0 B|"
    "nicholas:N IH1 K AH0 L AH0 S|"
    "nancy:N AE1 N S IY0|betty:B EH1 T IY0|"
    "margaret:M AA1 R G R AH0 T|sandra:S AE1 N D R AH0|"
    "ashley:AE1 SH L IY0|dorothy:D AO1 R AH0 TH IY0|"
    "kimberly:K IH1 M B ER0 L IY0|emily:EH1 M AH0 L IY0|"
    "donna:D AA1 N AH0|michelle:M IH0 SH EH1 L|"
    "carol:K AE1 R AH0 L|amanda:AH0 M AE1 N D AH0|"
    "melissa:M AH0 L IH1 S AH0|deborah:D EH1 B ER0 AH0|"
    "stephanie:S T EH1 F AH0 N IY0|rebecca:R AH0 B EH1 K AH0|"
    "sharon:SH AE1 R AH0 N|laura:L AO1 R AH0|"
    "cynthia:S IH1 N TH IY0 AH0|kathleen:K AE0 TH L IY1 N|"
    "amy:EY1 M IY0|angela:AE1 N JH AH0 L AH0|"
    "alice:AE1 L AH0 S|"
    # Technical terms
    "algorithm:AE1 L G ER0 IH2 DH AH0 M|"
    "database:D EY1 T AH0 B EY2 S|"
    "interface:IH1 N T ER0 F EY2 S|"
    "protocol:P R OW1 T AH0 K AO2 L|"
    "bandwidth:B AE1 N D W IH2 DH|"
    "encryption:EH0 N K R IH1 P SH AH0 N|"
    "authentication:AO0 TH EH2 N T AH0 K EY1 SH AH0 N|"
    "infrastructure:IH1 N F R AH0 S T R AH2 K CH ER0|"
    "architecture:AA1 R K AH0 T EH2 K CH ER0|"
    "configuration:K AH0 N F IH2 G Y ER0 EY1 SH AH0 N|"
    "implementation:IH2 M P L AH0 M EH0 N T EY1 SH AH0 N|"
    "optimization:AA2 P T AH0 M AH0 Z EY1 SH AH0 N|"
    "virtualization:V ER2 CH UW0 AH0 L AH0 Z EY1 SH AH0 N|"
    "microprocessor:M AY2 K R OW0 P R AA1 S EH0 S ER0|"
    "semiconductor:S EH2 M IY0 K AH0 N D AH1 K T ER0|"
    "cryptocurrency:K R IH2 P T OW0 K ER1 AH0 N S IY0|"
    "blockchain:B L AA1 K CH EY2 N|"
    "kubernetes:K UW0 B ER0 N EH1 T IY0 Z|"
    "tensorflow:T EH1 N S ER0 F L OW2|"
    "javascript:JH AA1 V AH0 S K R IH2 P T|"
    "python:P AY1 TH AA0 N|"
    "linux:L IH1 N AH0 K S|"
    "ubuntu:UW0 B UH1 N T UW0|"
    "microsoft:M AY1 K R OW0 S AO2 F T|"
    "google:G UW1 G AH0 L|"
    "amazon:AE1 M AH0 Z AA2 N|"
    "facebook:F EY1 S B UH2 K|"
    "twitter:T W IH1 T ER0|"
    "internet:IH1 N T ER0 N EH2 T|"
    "software:S AO1 F T W EH2 R|"
    "hardware:HH AA1 R D W EH2 R|"
    "firmware:F ER1 M W EH2 R|"
    "malware:M AE1 L W EH2 R|"
    "download:D AW1 N L OW2 D|"
    "upload:AH1 P L OW2 D|"
    "website:W EH1 B S AY2 T|"
    "email:IY1 M EY2 L|"
    "server:S ER1 V ER0|"
    "router:R AW1 T ER0|"
    "wireless:W AY1 R L AH0 S|"
    "bluetooth:B L UW1 T UW2 TH|"
    "pixel:P IH1 K S AH0 L|"
    "robot:R OW1 B AA2 T|"
    "android:AE1 N D R OY2 D|"
    "satellite:S AE1 T AH0 L AY2 T|"
    "telescope:T EH1 L AH0 S K OW2 P|"
    "microscope:M AY1 K R AH0 S K OW2 P|"
    "molecule:M AA1 L AH0 K Y UW2 L|"
    "hydrogen:HH AY1 D R AH0 JH AH0 N|"
    "oxygen:AA1 K S AH0 JH AH0 N|"
    "nitrogen:N AY1 T R AH0 JH AH0 N|"
    "carbon:K AA1 R B AH0 N|"
    "electron:IH0 L EH1 K T R AA0 N|"
    "quantum:K W AA1 N T AH0 M|"
    "gravity:G R AE1 V AH0 T IY0|"
    "velocity:V AH0 L AA1 S AH0 T IY0|"
    "equation:IH0 K W EY1 ZH AH0 N|"
    "mathematics:M AE2 TH AH0 M AE1 T IH0 K S|"
    "philosophy:F AH0 L AA1 S AH0 F IY0|"
    "psychology:S AY0 K AA1 L AH0 JH IY0|"
    "democracy:D IH0 M AA1 K R AH0 S IY0|"
    "economy:IH0 K AA1 N AH0 M IY0|"
    "strategy:S T R AE1 T AH0 JH IY0|"
    "analysis:AH0 N AE1 L AH0 S IH0 S|"
    "hypothesis:HH AY0 P AA1 TH AH0 S AH0 S|"
    "experiment:IH0 K S P EH1 R AH0 M AH0 N T|"
    "laboratory:L AE1 B R AH0 T AO2 R IY0|"
    "temperature:T EH1 M P R AH0 CH ER0|"
    "electricity:IH0 L EH2 K T R IH1 S AH0 T IY0|"
    "photography:F AH0 T AA1 G R AH0 F IY0|"
    "geography:JH IY0 AA1 G R AH0 F IY0|"
    "vocabulary:V OW0 K AE1 B Y AH0 L EH2 R IY0|"
    "pronunciation:P R AH0 N AH2 N S IY0 EY1 SH AH0 N|"
    "conversation:K AA2 N V ER0 S EY1 SH AH0 N|"
    "explanation:EH2 K S P L AH0 N EY1 SH AH0 N|"
    "opportunity:AA2 P ER0 T UW1 N AH0 T IY0|"
    "responsibility:R IH0 S P AA2 N S AH0 B IH1 L AH0 T IY0|"
    "relationship:R IH0 L EY1 SH AH0 N SH IH2 P|"
    "entertainment:EH2 N T ER0 T EY1 N M AH0 N T|"
    "advertisement:AE0 D V ER1 T AY2 Z M AH0 N T|"
    "approximately:AH0 P R AA1 K S AH0 M AH0 T L IY0|"
    "extraordinary:IH0 K S T R AO1 R D AH0 N EH2 R IY0|"
    "consciousness:K AA1 N SH AH0 S N AH0 S|"
    "intelligence:IH0 N T EH1 L AH0 JH AH0 N S|"
    "artificial:AA2 R T AH0 F IH1 SH AH0 L|"
    "autonomous:AO0 T AA1 N AH0 M AH0 S|"
    "sustainable:S AH0 S T EY1 N AH0 B AH0 L|"
    "infrastructure:IH1 N F R AH0 S T R AH2 K CH ER0|"
    "revolutionary:R EH2 V AH0 L UW1 SH AH0 N EH2 R IY0|"
    "sophisticated:S AH0 F IH1 S T AH0 K EY2 T AH0 D|"
    "comprehensive:K AA2 M P R IH0 HH EH1 N S IH0 V|"
    "fundamental:F AH2 N D AH0 M EH1 N T AH0 L|"
    "independent:IH2 N D AH0 P EH1 N D AH0 N T|"
    "appropriate:AH0 P R OW1 P R IY0 AH0 T|"
    "alternative:AO0 L T ER1 N AH0 T IH0 V|"
    "competitive:K AH0 M P EH1 T AH0 T IH0 V|"
    "perspective:P ER0 S P EH1 K T IH0 V|"
    "initiative:IH0 N IH1 SH AH0 T IH0 V|"
    "executive:IH0 G Z EH1 K Y AH0 T IH0 V|"
    "representative:R EH2 P R AH0 Z EH1 N T AH0 T IH0 V|"
    "administrative:AH0 D M IH1 N AH0 S T R EY2 T IH0 V|"
    "investigation:IH0 N V EH2 S T AH0 G EY1 SH AH0 N|"
    "recommendation:R EH2 K AH0 M AH0 N D EY1 SH AH0 N|"
    "demonstration:D EH2 M AH0 N S T R EY1 SH AH0 N|"
    "administration:AH0 D M IH2 N AH0 S T R EY1 SH AH0 N|"
    "consideration:K AH0 N S IH2 D ER0 EY1 SH AH0 N|"
    "determination:D IH0 T ER2 M AH0 N EY1 SH AH0 N|"
    "documentation:D AA2 K Y AH0 M EH0 N T EY1 SH AH0 N|"
    "identification:AY0 D EH2 N T AH0 F AH0 K EY1 SH AH0 N|"
    "classification:K L AE2 S AH0 F AH0 K EY1 SH AH0 N|"
    "specification:S P EH2 S AH0 F AH0 K EY1 SH AH0 N|"
    "certification:S ER2 T AH0 F AH0 K EY1 SH AH0 N|"
    "authorization:AO2 TH ER0 AH0 Z EY1 SH AH0 N|"
    "standardization:S T AE2 N D ER0 D AH0 Z EY1 SH AH0 N|"
    # ── Extended vocabulary: 4000+ additional common words ────────────
    "able:EY1 B AH0 L|above:AH0 B AH1 V|accept:AE0 K S EH1 P T|"
    "account:AH0 K AW1 N T|achieve:AH0 CH IY1 V|act:AE1 K T|"
    "action:AE1 K SH AH0 N|active:AE1 K T IH0 V|activity:AE0 K T IH1 V AH0 T IY0|"
    "add:AE1 D|address:AH0 D R EH1 S|admit:AH0 D M IH1 T|"
    "adult:AH0 D AH1 L T|advance:AH0 D V AE1 N S|advice:AH0 D V AY1 S|"
    "affect:AH0 F EH1 K T|afford:AH0 F AO1 R D|afraid:AH0 F R EY1 D|"
    "afternoon:AE2 F T ER0 N UW1 N|age:EY1 JH|agency:EY1 JH AH0 N S IY0|"
    "agent:EY1 JH AH0 N T|ago:AH0 G OW1|agree:AH0 G R IY1|"
    "ahead:AH0 HH EH1 D|aim:EY1 M|allow:AH0 L AW1|"
    "alone:AH0 L OW1 N|already:AO0 L R EH1 D IY0|although:AO0 L DH OW1|"
    "amount:AH0 M AW1 N T|ancient:EY1 N CH AH0 N T|anger:AE1 NG G ER0|"
    "angle:AE1 NG G AH0 L|angry:AE1 NG G R IY0|announce:AH0 N AW1 N S|"
    "annual:AE1 N Y UW0 AH0 L|apart:AH0 P AA1 R T|apartment:AH0 P AA1 R T M AH0 N T|"
    "appeal:AH0 P IY1 L|appear:AH0 P IH1 R|apple:AE1 P AH0 L|"
    "apply:AH0 P L AY1|approach:AH0 P R OW1 CH|approve:AH0 P R UW1 V|"
    "argue:AA1 R G Y UW0|argument:AA1 R G Y AH0 M AH0 N T|arm:AA1 R M|"
    "army:AA1 R M IY0|arrange:ER0 EY1 N JH|arrest:ER0 EH1 S T|"
    "arrive:ER0 AY1 V|art:AA1 R T|article:AA1 R T AH0 K AH0 L|"
    "artist:AA1 R T AH0 S T|aside:AH0 S AY1 D|assume:AH0 S UW1 M|"
    "attack:AH0 T AE1 K|attempt:AH0 T EH1 M P T|attend:AH0 T EH1 N D|"
    "attract:AH0 T R AE1 K T|audience:AO1 D IY0 AH0 N S|author:AO1 TH ER0|"
    "authority:AH0 TH AO1 R AH0 T IY0|avoid:AH0 V OY1 D|"
    "aware:AH0 W EH1 R|awful:AO1 F AH0 L|baby:B EY1 B IY0|"
    "background:B AE1 K G R AW2 N D|bad:B AE1 D|bag:B AE1 G|"
    "balance:B AE1 L AH0 N S|ball:B AO1 L|band:B AE1 N D|"
    "bank:B AE1 NG K|bar:B AA1 R|base:B EY1 S|"
    "basic:B EY1 S IH0 K|basis:B EY1 S AH0 S|basket:B AE1 S K AH0 T|"
    "bath:B AE1 TH|battle:B AE1 T AH0 L|beach:B IY1 CH|"
    "bear:B EH1 R|beat:B IY1 T|beauty:B Y UW1 T IY0|"
    "bed:B EH1 D|bedroom:B EH1 D R UW2 M|beer:B IH1 R|"
    "behavior:B IH0 HH EY1 V Y ER0|belong:B IH0 L AO1 NG|"
    "bend:B EH1 N D|benefit:B EH1 N AH0 F IH0 T|beside:B IH0 S AY1 D|"
    "beyond:B IH0 Y AA1 N D|bill:B IH1 L|billion:B IH1 L Y AH0 N|"
    "bind:B AY1 N D|bit:B IH1 T|bite:B AY1 T|"
    "blame:B L EY1 M|blank:B L AE1 NG K|blind:B L AY1 N D|"
    "block:B L AA1 K|blood:B L AH1 D|blow:B L OW1|"
    "blue:B L UW1|board:B AO1 R D|boat:B OW1 T|"
    "bomb:B AA1 M|bone:B OW1 N|border:B AO1 R D ER0|"
    "born:B AO1 R N|boss:B AO1 S|bottom:B AA1 T AH0 M|"
    "bound:B AW1 N D|brain:B R EY1 N|branch:B R AE1 N CH|"
    "brave:B R EY1 V|bread:B R EH1 D|break:B R EY1 K|"
    "breakfast:B R EH1 K F AH0 S T|breath:B R EH1 TH|breathe:B R IY1 DH|"
    "bridge:B R IH1 JH|brief:B R IY1 F|bright:B R AY1 T|"
    "bring:B R IH1 NG|broad:B R AO1 D|broken:B R OW1 K AH0 N|"
    "brother:B R AH1 DH ER0|brown:B R AW1 N|brush:B R AH1 SH|"
    "budget:B AH1 JH AH0 T|build:B IH1 L D|burn:B ER1 N|"
    "bus:B AH1 S|busy:B IH1 Z IY0|buy:B AY1|"
    "cabinet:K AE1 B AH0 N AH0 T|cake:K EY1 K|calculate:K AE1 L K Y AH0 L EY2 T|"
    "camera:K AE1 M ER0 AH0|camp:K AE1 M P|campaign:K AE0 M P EY1 N|"
    "cancer:K AE1 N S ER0|candidate:K AE1 N D AH0 D EY2 T|"
    "capable:K EY1 P AH0 B AH0 L|capacity:K AH0 P AE1 S AH0 T IY0|"
    "capital:K AE1 P AH0 T AH0 L|captain:K AE1 P T AH0 N|"
    "capture:K AE1 P CH ER0|card:K AA1 R D|care:K EH1 R|"
    "career:K ER0 IH1 R|careful:K EH1 R F AH0 L|cat:K AE1 T|"
    "catch:K AE1 CH|category:K AE1 T AH0 G AO2 R IY0|"
    "cause:K AO1 Z|celebrate:S EH1 L AH0 B R EY2 T|"
    "cell:S EH1 L|center:S EH1 N T ER0|century:S EH1 N CH ER0 IY0|"
    "chair:CH EH1 R|chairman:CH EH1 R M AH0 N|challenge:CH AE1 L AH0 N JH|"
    "champion:CH AE1 M P IY0 AH0 N|chance:CH AE1 N S|"
    "chapter:CH AE1 P T ER0|character:K EH1 R AH0 K T ER0|"
    "charge:CH AA1 R JH|charity:CH AE1 R AH0 T IY0|"
    "chart:CH AA1 R T|chase:CH EY1 S|cheap:CH IY1 P|"
    "check:CH EH1 K|cheese:CH IY1 Z|chemical:K EH1 M AH0 K AH0 L|"
    "chest:CH EH1 S T|chicken:CH IH1 K AH0 N|chief:CH IY1 F|"
    "chip:CH IH1 P|chocolate:CH AO1 K L AH0 T|choice:CH OY1 S|"
    "choose:CH UW1 Z|church:CH ER1 CH|circle:S ER1 K AH0 L|"
    "citizen:S IH1 T AH0 Z AH0 N|civil:S IH1 V AH0 L|"
    "claim:K L EY1 M|clean:K L IY1 N|client:K L AY1 AH0 N T|"
    "climate:K L AY1 M AH0 T|climb:K L AY1 M|clock:K L AA1 K|"
    "closely:K L OW1 S L IY0|clothes:K L OW1 DH Z|"
    "cloud:K L AW1 D|club:K L AH1 B|coach:K OW1 CH|"
    "coast:K OW1 S T|code:K OW1 D|coffee:K AO1 F IY0|"
    "cold:K OW1 L D|collect:K AH0 L EH1 K T|college:K AA1 L IH0 JH|"
    "column:K AA1 L AH0 M|combine:K AH0 M B AY1 N|"
    "comfort:K AH1 M F ER0 T|command:K AH0 M AE1 N D|"
    "comment:K AA1 M EH0 N T|commercial:K AH0 M ER1 SH AH0 L|"
    "commission:K AH0 M IH1 SH AH0 N|commit:K AH0 M IH1 T|"
    "committee:K AH0 M IH1 T IY0|compare:K AH0 M P EH1 R|"
    "compete:K AH0 M P IY1 T|complaint:K AH0 M P L EY1 N T|"
    "complex:K AA1 M P L EH0 K S|concentrate:K AA1 N S AH0 N T R EY2 T|"
    "concept:K AA1 N S EH0 P T|concern:K AH0 N S ER1 N|"
    "conduct:K AA1 N D AH0 K T|conference:K AA1 N F ER0 AH0 N S|"
    "confidence:K AA1 N F AH0 D AH0 N S|confirm:K AH0 N F ER1 M|"
    "conflict:K AA1 N F L IH0 K T|congress:K AA1 NG G R AH0 S|"
    "connect:K AH0 N EH1 K T|connection:K AH0 N EH1 K SH AH0 N|"
    "conscious:K AA1 N SH AH0 S|consequence:K AA1 N S AH0 K W EH2 N S|"
    "conservative:K AH0 N S ER1 V AH0 T IH0 V|"
    "consider:K AH0 N S IH1 D ER0|consist:K AH0 N S IH1 S T|"
    "constant:K AA1 N S T AH0 N T|construct:K AH0 N S T R AH1 K T|"
    "consumer:K AH0 N S UW1 M ER0|contact:K AA1 N T AE2 K T|"
    "contain:K AH0 N T EY1 N|content:K AA1 N T EH0 N T|"
    "contest:K AA1 N T EH0 S T|context:K AA1 N T EH0 K S T|"
    "contract:K AA1 N T R AE2 K T|contribute:K AH0 N T R IH1 B Y UW0 T|"
    "conversation:K AA2 N V ER0 S EY1 SH AH0 N|"
    "convince:K AH0 N V IH1 N S|cook:K UH1 K|cool:K UW1 L|"
    "copy:K AA1 P IY0|corner:K AO1 R N ER0|correct:K ER0 EH1 K T|"
    "cost:K AO1 S T|cotton:K AA1 T AH0 N|count:K AW1 N T|"
    "counter:K AW1 N T ER0|couple:K AH1 P AH0 L|courage:K ER1 AH0 JH|"
    "course:K AO1 R S|court:K AO1 R T|cousin:K AH1 Z AH0 N|"
    "cover:K AH1 V ER0|crack:K R AE1 K|craft:K R AE1 F T|"
    "crash:K R AE1 SH|crazy:K R EY1 Z IY0|cream:K R IY1 M|"
    "creation:K R IY0 EY1 SH AH0 N|creature:K R IY1 CH ER0|"
    "credit:K R EH1 D AH0 T|crew:K R UW1|crime:K R AY1 M|"
    "criminal:K R IH1 M AH0 N AH0 L|crisis:K R AY1 S AH0 S|"
    "critical:K R IH1 T AH0 K AH0 L|criticism:K R IH1 T AH0 S IH2 Z AH0 M|"
    "cross:K R AO1 S|crowd:K R AW1 D|crucial:K R UW1 SH AH0 L|"
    "cultural:K AH1 L CH ER0 AH0 L|culture:K AH1 L CH ER0|"
    "cup:K AH1 P|cure:K Y UH1 R|curious:K Y UH1 R IY0 AH0 S|"
    "customer:K AH1 S T AH0 M ER0|cycle:S AY1 K AH0 L|"
    "dad:D AE1 D|daily:D EY1 L IY0|damage:D AE1 M AH0 JH|"
    "dance:D AE1 N S|danger:D EY1 N JH ER0|dangerous:D EY1 N JH ER0 AH0 S|"
    "dare:D EH1 R|daughter:D AO1 T ER0|dead:D EH1 D|"
    "deal:D IY1 L|dear:D IH1 R|death:D EH1 TH|"
    "debate:D AH0 B EY1 T|debt:D EH1 T|decade:D EH0 K EY1 D|"
    "decide:D IH0 S AY1 D|declare:D IH0 K L EH1 R|"
    "decline:D IH0 K L AY1 N|deep:D IY1 P|defeat:D IH0 F IY1 T|"
    "defend:D IH0 F EH1 N D|defense:D IH0 F EH1 N S|"
    "define:D IH0 F AY1 N|degree:D IH0 G R IY1|"
    "delay:D IH0 L EY1|deliver:D IH0 L IH1 V ER0|"
    "demand:D IH0 M AE1 N D|democracy:D IH0 M AA1 K R AH0 S IY0|"
    "democratic:D EH2 M AH0 K R AE1 T IH0 K|deny:D IH0 N AY1|"
    "department:D IH0 P AA1 R T M AH0 N T|depend:D IH0 P EH1 N D|"
    "depression:D IH0 P R EH1 SH AH0 N|depth:D EH1 P TH|"
    "describe:D IH0 S K R AY1 B|description:D IH0 S K R IH1 P SH AH0 N|"
    "desert:D EH1 Z ER0 T|deserve:D IH0 Z ER1 V|"
    "desire:D IH0 Z AY1 ER0|desk:D EH1 S K|"
    "despite:D IH0 S P AY1 T|destroy:D IH0 S T R OY1|"
    "detail:D IH0 T EY1 L|detect:D IH0 T EH1 K T|"
    "determine:D IH0 T ER1 M AH0 N|device:D IH0 V AY1 S|"
    "die:D AY1|diet:D AY1 AH0 T|dinner:D IH1 N ER0|"
    "direct:D ER0 EH1 K T|director:D ER0 EH1 K T ER0|"
    "dirty:D ER1 T IY0|disappear:D IH2 S AH0 P IH1 R|"
    "discover:D IH0 S K AH1 V ER0|discuss:D IH0 S K AH1 S|"
    "discussion:D IH0 S K AH1 SH AH0 N|disease:D IH0 Z IY1 Z|"
    "display:D IH0 S P L EY1|distance:D IH1 S T AH0 N S|"
    "distinct:D IH0 S T IH1 NG K T|distribute:D IH0 S T R IH1 B Y UW0 T|"
    "district:D IH1 S T R IH0 K T|divide:D IH0 V AY1 D|"
    "doctor:D AA1 K T ER0|document:D AA1 K Y AH0 M AH0 N T|"
    "dollar:D AA1 L ER0|domestic:D AH0 M EH1 S T IH0 K|"
    "dominate:D AA1 M AH0 N EY2 T|double:D AH1 B AH0 L|"
    "doubt:D AW1 T|dozen:D AH1 Z AH0 N|draft:D R AE1 F T|"
    "drag:D R AE1 G|drama:D R AA1 M AH0|dramatic:D R AH0 M AE1 T IH0 K|"
    "draw:D R AO1|dream:D R IY1 M|dress:D R EH1 S|"
    "drink:D R IH1 NG K|drop:D R AA1 P|drug:D R AH1 G|"
    "dry:D R AY1|due:D UW1|dust:D AH1 S T|"
    "duty:D UW1 T IY0|earn:ER1 N|east:IY1 S T|"
    "edge:EH1 JH|editor:EH1 D AH0 T ER0|effect:IH0 F EH1 K T|"
    "effective:IH0 F EH1 K T IH0 V|effort:EH1 F ER0 T|"
    "eight:EY1 T|either:IY1 DH ER0|elderly:EH1 L D ER0 L IY0|"
    "elect:IH0 L EH1 K T|election:IH0 L EH1 K SH AH0 N|"
    "element:EH1 L AH0 M AH0 N T|eliminate:IH0 L IH1 M AH0 N EY2 T|"
    "emerge:IH0 M ER1 JH|emergency:IH0 M ER1 JH AH0 N S IY0|"
    "emotion:IH0 M OW1 SH AH0 N|emotional:IH0 M OW1 SH AH0 N AH0 L|"
    "emphasis:EH1 M F AH0 S AH0 S|employ:EH0 M P L OY1|"
    "employee:EH0 M P L OY1 IY0|empty:EH1 M P T IY0|"
    "enable:EH0 N EY1 B AH0 L|encounter:EH0 N K AW1 N T ER0|"
    "encourage:EH0 N K ER1 AH0 JH|enemy:EH1 N AH0 M IY0|"
    "engage:EH0 N G EY1 JH|enjoy:EH0 N JH OY1|"
    "enormous:IH0 N AO1 R M AH0 S|ensure:EH0 N SH UH1 R|"
    "enter:EH1 N T ER0|enterprise:EH1 N T ER0 P R AY2 Z|"
    "entire:EH0 N T AY1 ER0|entrance:EH1 N T R AH0 N S|"
    "entry:EH1 N T R IY0|episode:EH1 P AH0 S OW2 D|"
    "equal:IY1 K W AH0 L|equipment:IH0 K W IH1 P M AH0 N T|"
    "era:EH1 R AH0|error:EH1 R ER0|escape:IH0 S K EY1 P|"
    "essay:EH1 S EY0|essential:IH0 S EH1 N SH AH0 L|"
    "establish:IH0 S T AE1 B L IH0 SH|estate:IH0 S T EY1 T|"
    "estimate:EH1 S T AH0 M AH0 T|evaluate:IH0 V AE1 L Y UW0 EY2 T|"
    "event:IH0 V EH1 N T|eventually:IH0 V EH1 N CH UW0 AH0 L IY0|"
    "everybody:EH1 V R IY0 B AA2 D IY0|evidence:EH1 V AH0 D AH0 N S|"
    "evil:IY1 V AH0 L|evolve:IH0 V AA1 L V|"
    "exact:IH0 G Z AE1 K T|examine:IH0 G Z AE1 M AH0 N|"
    "exchange:IH0 K S CH EY1 N JH|exciting:IH0 K S AY1 T IH0 NG|"
    "excuse:IH0 K S K Y UW1 Z|exercise:EH1 K S ER0 S AY2 Z|"
    "exhibit:IH0 G Z IH1 B AH0 T|exist:IH0 G Z IH1 S T|"
    "expand:IH0 K S P AE1 N D|expect:IH0 K S P EH1 K T|"
    "expense:IH0 K S P EH1 N S|expert:EH1 K S P ER0 T|"
    "explain:IH0 K S P L EY1 N|explore:IH0 K S P L AO1 R|"
    "expose:IH0 K S P OW1 Z|express:IH0 K S P R EH1 S|"
    "extend:IH0 K S T EH1 N D|extent:IH0 K S T EH1 N T|"
    "extra:EH1 K S T R AH0|extreme:IH0 K S T R IY1 M|"
    "facility:F AH0 S IH1 L AH0 T IY0|factor:F AE1 K T ER0|"
    "factory:F AE1 K T ER0 IY0|fail:F EY1 L|failure:F EY1 L Y ER0|"
    "fair:F EH1 R|faith:F EY1 TH|fall:F AO1 L|"
    "false:F AO1 L S|familiar:F AH0 M IH1 L Y ER0|fan:F AE1 N|"
    "fancy:F AE1 N S IY0|farm:F AA1 R M|farmer:F AA1 R M ER0|"
    "fashion:F AE1 SH AH0 N|fat:F AE1 T|fate:F EY1 T|"
    "fault:F AO1 L T|favor:F EY1 V ER0|favorite:F EY1 V ER0 AH0 T|"
    "fear:F IH1 R|feature:F IY1 CH ER0|federal:F EH1 D ER0 AH0 L|"
    "fee:F IY1|feed:F IY1 D|female:F IY1 M EY0 L|"
    "fence:F EH1 N S|fiction:F IH1 K SH AH0 N|"
    "fight:F AY1 T|file:F AY1 L|fill:F IH1 L|"
    "film:F IH1 L M|final:F AY1 N AH0 L|finance:F AH0 N AE1 N S|"
    "finger:F IH1 NG G ER0|finish:F IH1 N IH0 SH|fire:F AY1 ER0|"
    "firm:F ER1 M|fit:F IH1 T|fix:F IH1 K S|"
    "flag:F L AE1 G|flame:F L EY1 M|flat:F L AE1 T|"
    "flavor:F L EY1 V ER0|flee:F L IY1|flesh:F L EH1 SH|"
    "flight:F L AY1 T|float:F L OW1 T|flood:F L AH1 D|"
    "floor:F L AO1 R|flow:F L OW1|flower:F L AW1 ER0|"
    "focus:F OW1 K AH0 S|folk:F OW1 K|foot:F UH1 T|"
    "football:F UH1 T B AO2 L|force:F AO1 R S|foreign:F AO1 R AH0 N|"
    "forest:F AO1 R AH0 S T|forever:F ER0 EH1 V ER0|"
    "forget:F ER0 G EH1 T|formal:F AO1 R M AH0 L|"
    "former:F AO1 R M ER0|formula:F AO1 R M Y AH0 L AH0|"
    "fortune:F AO1 R CH AH0 N|forward:F AO1 R W ER0 D|"
    "found:F AW1 N D|foundation:F AW0 N D EY1 SH AH0 N|"
    "frame:F R EY1 M|freedom:F R IY1 D AH0 M|"
    "french:F R EH1 N CH|frequent:F R IY1 K W AH0 N T|"
    "fresh:F R EH1 SH|friend:F R EH1 N D|front:F R AH1 N T|"
    "fruit:F R UW1 T|fuel:F Y UW1 AH0 L|function:F AH1 NG K SH AH0 N|"
    "fund:F AH1 N D|funny:F AH1 N IY0|gain:G EY1 N|"
    "gallery:G AE1 L ER0 IY0|game:G EY1 M|gap:G AE1 P|"
    "garage:G ER0 AA1 ZH|garden:G AA1 R D AH0 N|gas:G AE1 S|"
    "gate:G EY1 T|gather:G AE1 DH ER0|gay:G EY1|"
    "generation:JH EH2 N ER0 EY1 SH AH0 N|genetic:JH AH0 N EH1 T IH0 K|"
    "gentle:JH EH1 N T AH0 L|gentleman:JH EH1 N T AH0 L M AH0 N|"
    "gift:G IH1 F T|glad:G L AE1 D|glass:G L AE1 S|"
    "global:G L OW1 B AH0 L|glory:G L AO1 R IY0|goal:G OW1 L|"
    "god:G AA1 D|gold:G OW1 L D|golden:G OW1 L D AH0 N|"
    "golf:G AA1 L F|grab:G R AE1 B|grade:G R EY1 D|"
    "grain:G R EY1 N|grand:G R AE1 N D|grandfather:G R AE1 N D F AA2 DH ER0|"
    "grandmother:G R AE1 N D M AH2 DH ER0|grant:G R AE1 N T|"
    "grass:G R AE1 S|grave:G R EY1 V|gray:G R EY1|"
    "green:G R IY1 N|ground:G R AW1 N D|growth:G R OW1 TH|"
    "guarantee:G EH2 R AH0 N T IY1|guard:G AA1 R D|"
    "guess:G EH1 S|guest:G EH1 S T|guide:G AY1 D|"
    "guilty:G IH1 L T IY0|gun:G AH1 N|guy:G AY1|"
    "habit:HH AE1 B AH0 T|hair:HH EH1 R|hall:HH AO1 L|"
    "handle:HH AE1 N D AH0 L|hang:HH AE1 NG|harbor:HH AA1 R B ER0|"
    "harm:HH AA1 R M|hat:HH AE1 T|hate:HH EY1 T|"
    "health:HH EH1 L TH|healthy:HH EH1 L TH IY0|heart:HH AA1 R T|"
    "heat:HH IY1 T|heavy:HH EH1 V IY0|height:HH AY1 T|"
    "hero:HH IH1 R OW0|hide:HH AY1 D|hill:HH IH1 L|"
    "hire:HH AY1 ER0|history:HH IH1 S T ER0 IY0|hit:HH IH1 T|"
    "hole:HH OW1 L|holiday:HH AA1 L AH0 D EY2|holy:HH OW1 L IY0|"
    "honest:AA1 N AH0 S T|honor:AA1 N ER0|hope:HH OW1 P|"
    "horrible:HH AO1 R AH0 B AH0 L|host:HH OW1 S T|hot:HH AA1 T|"
    "hotel:HH OW0 T EH1 L|hour:AW1 ER0|huge:HH Y UW1 JH|"
    "humor:HH Y UW1 M ER0|hunt:HH AH1 N T|hurry:HH ER1 IY0|"
    "hurt:HH ER1 T|husband:HH AH1 Z B AH0 N D|ice:AY1 S|"
    "identify:AY0 D EH1 N T AH0 F AY2|ignore:IH0 G N AO1 R|"
    "illustrate:IH1 L AH0 S T R EY2 T|imagine:IH0 M AE1 JH AH0 N|"
    "impact:IH1 M P AE0 K T|imply:IH0 M P L AY1|"
    "impose:IH0 M P OW1 Z|impose:IH0 M P OW1 Z|"
    "impression:IH0 M P R EH1 SH AH0 N|improve:IH0 M P R UW1 V|"
    "incident:IH1 N S AH0 D AH0 N T|include:IH0 N K L UW1 D|"
    "income:IH1 N K AH2 M|increase:IH0 N K R IY1 S|"
    "increasingly:IH0 N K R IY1 S IH0 NG L IY0|"
    "incredible:IH0 N K R EH1 D AH0 B AH0 L|"
    "indeed:IH0 N D IY1 D|indicate:IH1 N D AH0 K EY2 T|"
    "initial:IH0 N IH1 SH AH0 L|injury:IH1 N JH ER0 IY0|"
    "inner:IH1 N ER0|innocent:IH1 N AH0 S AH0 N T|"
    "innovation:IH2 N AH0 V EY1 SH AH0 N|insist:IH0 N S IH1 S T|"
    "install:IH0 N S T AO1 L|instance:IH1 N S T AH0 N S|"
    "instead:IH0 N S T EH1 D|institution:IH2 N S T AH0 T UW1 SH AH0 N|"
    "insurance:IH0 N SH UH1 R AH0 N S|intellectual:IH2 N T AH0 L EH1 K CH UW0 AH0 L|"
    "intend:IH0 N T EH1 N D|intention:IH0 N T EH1 N SH AH0 N|"
    "internal:IH0 N T ER1 N AH0 L|interpret:IH0 N T ER1 P R AH0 T|"
    "intervention:IH2 N T ER0 V EH1 N SH AH0 N|"
    "interview:IH1 N T ER0 V Y UW2|introduce:IH2 N T R AH0 D UW1 S|"
    "introduction:IH2 N T R AH0 D AH1 K SH AH0 N|"
    "invade:IH0 N V EY1 D|invest:IH0 N V EH1 S T|"
    "investment:IH0 N V EH1 S T M AH0 N T|investor:IH0 N V EH1 S T ER0|"
    "invite:IH0 N V AY1 T|involve:IH0 N V AA1 L V|"
    "iron:AY1 ER0 N|island:AY1 L AH0 N D|issue:IH1 SH UW0|"
    "item:AY1 T AH0 M|jacket:JH AE1 K AH0 T|jail:JH EY1 L|"
    "job:JH AA1 B|join:JH OY1 N|joint:JH OY1 N T|"
    "joke:JH OW1 K|journal:JH ER1 N AH0 L|journey:JH ER1 N IY0|"
    "joy:JH OY1|judge:JH AH1 JH|judgment:JH AH1 JH M AH0 N T|"
    "juice:JH UW1 S|jump:JH AH1 M P|junior:JH UW1 N Y ER0|"
    "jury:JH UH1 R IY0|justice:JH AH1 S T AH0 S|justify:JH AH1 S T AH0 F AY2|"
    "keen:K IY1 N|key:K IY1|kick:K IH1 K|"
    "kid:K IH1 D|kill:K IH1 L|king:K IH1 NG|"
    "kiss:K IH1 S|kitchen:K IH1 CH AH0 N|knee:N IY1|"
    "knife:N AY1 F|knock:N AA1 K|label:L EY1 B AH0 L|"
    "labor:L EY1 B ER0|lack:L AE1 K|lady:L EY1 D IY0|"
    "lake:L EY1 K|land:L AE1 N D|landscape:L AE1 N D S K EY2 P|"
    "largely:L AA1 R JH L IY0|launch:L AO1 N CH|"
    "law:L AO1|lawyer:L AO1 Y ER0|lay:L EY1|"
    "layer:L EY1 ER0|leader:L IY1 D ER0|leadership:L IY1 D ER0 SH IH2 P|"
    "league:L IY1 G|lean:L IY1 N|leg:L EH1 G|"
    "length:L EH1 NG TH|lesson:L EH1 S AH0 N|liberal:L IH1 B ER0 AH0 L|"
    "library:L AY1 B R EH2 R IY0|lie:L AY1|lift:L IH1 F T|"
    "limit:L IH1 M AH0 T|link:L IH1 NG K|lip:L IH1 P|"
    "literary:L IH1 T ER0 EH2 R IY0|literature:L IH1 T ER0 AH0 CH ER0|"
    "load:L OW1 D|loan:L OW1 N|lock:L AA1 K|"
    "log:L AO1 G|lonely:L OW1 N L IY0|loose:L UW1 S|"
    "lord:L AO1 R D|lose:L UW1 Z|loss:L AO1 S|"
    "lot:L AA1 T|loud:L AW1 D|lover:L AH1 V ER0|"
    "luck:L AH1 K|lunch:L AH1 N CH|lung:L AH1 NG|"
    "mad:M AE1 D|magazine:M AE2 G AH0 Z IY1 N|magic:M AE1 JH IH0 K|"
    "mail:M EY1 L|main:M EY1 N|maintain:M EY0 N T EY1 N|"
    "male:M EY1 L|manage:M AE1 N AH0 JH|manager:M AE1 N AH0 JH ER0|"
    "manner:M AE1 N ER0|manufacturer:M AE2 N Y AH0 F AE1 K CH ER0 ER0|"
    "marry:M AE1 R IY0|mask:M AE1 S K|mass:M AE1 S|"
    "massive:M AE1 S IH0 V|master:M AE1 S T ER0|match:M AE1 CH|"
    "material:M AH0 T IH1 R IY0 AH0 L|math:M AE1 TH|"
    "matter:M AE1 T ER0|maximum:M AE1 K S AH0 M AH0 M|"
    "meal:M IY1 L|meat:M IY1 T|mechanism:M EH1 K AH0 N IH2 Z AH0 M|"
    "media:M IY1 D IY0 AH0|meet:M IY1 T|meeting:M IY1 T IH0 NG|"
    "member:M EH1 M B ER0|membership:M EH1 M B ER0 SH IH2 P|"
    "memory:M EH1 M ER0 IY0|mental:M EH1 N T AH0 L|"
    "mention:M EH1 N SH AH0 N|menu:M EH1 N Y UW0|"
    "merely:M IH1 R L IY0|message:M EH1 S AH0 JH|"
    "metal:M EH1 T AH0 L|method:M EH1 TH AH0 D|"
    "middle:M IH1 D AH0 L|mind:M AY1 N D|mine:M AY1 N|"
    "minister:M IH1 N AH0 S T ER0|minor:M AY1 N ER0|"
    "minority:M AH0 N AO1 R AH0 T IY0|mirror:M IH1 R ER0|"
    "mission:M IH1 SH AH0 N|mistake:M IH0 S T EY1 K|"
    "mix:M IH1 K S|mixture:M IH1 K S CH ER0|mode:M OW1 D|"
    "moderate:M AA1 D ER0 AH0 T|mom:M AA1 M|monitor:M AA1 N AH0 T ER0|"
    "month:M AH1 N TH|mood:M UW1 D|moon:M UW1 N|"
    "moral:M AO1 R AH0 L|moreover:M AO0 R OW1 V ER0|mostly:M OW1 S T L IY0|"
    "motion:M OW1 SH AH0 N|motor:M OW1 T ER0|mount:M AW1 N T|"
    "mouse:M AW1 S|mouth:M AW1 TH|movement:M UW1 V M AH0 N T|"
    "movie:M UW1 V IY0|murder:M ER1 D ER0|muscle:M AH1 S AH0 L|"
    "museum:M Y UW0 Z IY1 AH0 M|mystery:M IH1 S T ER0 IY0|"
    "naked:N EY1 K AH0 D|narrow:N AE1 R OW0|nation:N EY1 SH AH0 N|"
    "native:N EY1 T IH0 V|nature:N EY1 CH ER0|"
    "necessarily:N EH2 S AH0 S EH1 R AH0 L IY0|neck:N EH1 K|"
    "negative:N EH1 G AH0 T IH0 V|negotiate:N AH0 G OW1 SH IY0 EY2 T|"
    "neighbor:N EY1 B ER0|neighborhood:N EY1 B ER0 HH UH2 D|"
    "neither:N IY1 DH ER0|nerve:N ER1 V|net:N EH1 T|"
    "newspaper:N UW1 Z P EY2 P ER0|nice:N AY1 S|"
    "noise:N OY1 Z|none:N AH1 N|nor:N AO1 R|"
    "normal:N AO1 R M AH0 L|north:N AO1 R TH|northern:N AO1 R DH ER0 N|"
    "nose:N OW1 Z|notice:N OW1 T AH0 S|notion:N OW1 SH AH0 N|"
    "novel:N AA1 V AH0 L|nowhere:N OW1 W EH2 R|nuclear:N UW1 K L IY0 ER0|"
    "nurse:N ER1 S|object:AA1 B JH EH0 K T|objective:AH0 B JH EH1 K T IH0 V|"
    "obligation:AA2 B L AH0 G EY1 SH AH0 N|observe:AH0 B Z ER1 V|"
    "obtain:AH0 B T EY1 N|obvious:AA1 B V IY0 AH0 S|"
    "occasion:AH0 K EY1 ZH AH0 N|occupy:AA1 K Y AH0 P AY2|"
    "occur:AH0 K ER1|odd:AA1 D|offense:AH0 F EH1 N S|"
    "offer:AO1 F ER0|office:AO1 F AH0 S|officer:AO1 F AH0 S ER0|"
    "official:AH0 F IH1 SH AH0 L|okay:OW0 K EY1|"
    "once:W AH1 N S|online:AA1 N L AY2 N|onto:AA1 N T UW0|"
    "opening:OW1 P AH0 N IH0 NG|operate:AA1 P ER0 EY2 T|"
    "opinion:AH0 P IH1 N Y AH0 N|opponent:AH0 P OW1 N AH0 N T|"
    "option:AA1 P SH AH0 N|orange:AO1 R AH0 N JH|"
    "ordinary:AO1 R D AH0 N EH2 R IY0|organic:AO0 R G AE1 N IH0 K|"
    "origin:AO1 R AH0 JH AH0 N|original:ER0 IH1 JH AH0 N AH0 L|"
    "otherwise:AH1 DH ER0 W AY2 Z|ought:AO1 T|"
    "outcome:AW1 T K AH2 M|overall:OW2 V ER0 AO1 L|"
    "overcome:OW2 V ER0 K AH1 M|owe:OW1|owner:OW1 N ER0|"
    "pace:P EY1 S|pack:P AE1 K|package:P AE1 K AH0 JH|"
    "pain:P EY1 N|paint:P EY1 N T|painting:P EY1 N T IH0 NG|"
    "pair:P EH1 R|palace:P AE1 L AH0 S|pale:P EY1 L|"
    "panel:P AE1 N AH0 L|panic:P AE1 N IH0 K|parent:P EH1 R AH0 N T|"
    "park:P AA1 R K|parking:P AA1 R K IH0 NG|partner:P AA1 R T N ER0|"
    "party:P AA1 R T IY0|pass:P AE1 S|passage:P AE1 S AH0 JH|"
    "passenger:P AE1 S AH0 N JH ER0|passion:P AE1 SH AH0 N|"
    "past:P AE1 S T|path:P AE1 TH|patient:P EY1 SH AH0 N T|"
    "pause:P AO1 Z|pay:P EY1|payment:P EY1 M AH0 N T|"
    "peace:P IY1 S|peak:P IY1 K|peer:P IH1 R|"
    "penalty:P EH1 N AH0 L T IY0|pension:P EH1 N SH AH0 N|"
    "percent:P ER0 S EH1 N T|perfect:P ER1 F AH0 K T|"
    "perform:P ER0 F AO1 R M|period:P IH1 R IY0 AH0 D|"
    "permanent:P ER1 M AH0 N AH0 N T|permit:P ER0 M IH1 T|"
    "person:P ER1 S AH0 N|personality:P ER2 S AH0 N AE1 L AH0 T IY0|"
    "phase:F EY1 Z|phone:F OW1 N|photo:F OW1 T OW0|"
    "photograph:F OW1 T AH0 G R AE2 F|phrase:F R EY1 Z|"
    "pick:P IH1 K|pilot:P AY1 L AH0 T|pine:P AY1 N|"
    "pink:P IH1 NG K|pipe:P AY1 P|pitch:P IH1 CH|"
    "planet:P L AE1 N AH0 T|plastic:P L AE1 S T IH0 K|"
    "plate:P L EY1 T|platform:P L AE1 T F AO2 R M|"
    "player:P L EY1 ER0|pleasure:P L EH1 ZH ER0|"
    "plenty:P L EH1 N T IY0|pocket:P AA1 K AH0 T|"
    "poem:P OW1 AH0 M|poet:P OW1 AH0 T|poetry:P OW1 AH0 T R IY0|"
    "police:P AH0 L IY1 S|policy:P AA1 L AH0 S IY0|"
    "politics:P AA1 L AH0 T IH0 K S|pollution:P AH0 L UW1 SH AH0 N|"
    "pool:P UW1 L|poor:P UH1 R|popular:P AA1 P Y AH0 L ER0|"
    "portion:P AO1 R SH AH0 N|portrait:P AO1 R T R AH0 T|"
    "positive:P AA1 Z AH0 T IH0 V|possess:P AH0 Z EH1 S|"
    "possibility:P AA2 S AH0 B IH1 L AH0 T IY0|"
    "pot:P AA1 T|potato:P AH0 T EY1 T OW0|"
    "potential:P AH0 T EH1 N SH AH0 L|pound:P AW1 N D|"
    "pour:P AO1 R|poverty:P AA1 V ER0 T IY0|"
    "powerful:P AW1 ER0 F AH0 L|practical:P R AE1 K T AH0 K AH0 L|"
    "practice:P R AE1 K T AH0 S|pray:P R EY1|prayer:P R EH1 R|"
    "precisely:P R IH0 S AY1 S L IY0|predict:P R IH0 D IH1 K T|"
    "prefer:P R IH0 F ER1|prepare:P R IH0 P EH1 R|"
    "presence:P R EH1 Z AH0 N S|preserve:P R IH0 Z ER1 V|"
    "press:P R EH1 S|pressure:P R EH1 SH ER0|"
    "pretend:P R IH0 T EH1 N D|pretty:P R IH1 T IY0|"
    "prevent:P R IH0 V EH1 N T|previous:P R IY1 V IY0 AH0 S|"
    "price:P R AY1 S|pride:P R AY1 D|primary:P R AY1 M EH2 R IY0|"
    "prime:P R AY1 M|prince:P R IH1 N S|princess:P R IH1 N S EH0 S|"
    "principal:P R IH1 N S AH0 P AH0 L|principle:P R IH1 N S AH0 P AH0 L|"
    "print:P R IH1 N T|prior:P R AY1 ER0|priority:P R AY0 AO1 R AH0 T IY0|"
    "prison:P R IH1 Z AH0 N|prisoner:P R IH1 Z AH0 N ER0|"
    "privacy:P R AY1 V AH0 S IY0|prize:P R AY1 Z|"
    "procedure:P R AH0 S IY1 JH ER0|proceed:P R AH0 S IY1 D|"
    "produce:P R AH0 D UW1 S|producer:P R AH0 D UW1 S ER0|"
    "product:P R AA1 D AH0 K T|profession:P R AH0 F EH1 SH AH0 N|"
    "professor:P R AH0 F EH1 S ER0|profit:P R AA1 F AH0 T|"
    "progress:P R AA1 G R EH0 S|project:P R AA1 JH EH0 K T|"
    "promise:P R AA1 M AH0 S|promote:P R AH0 M OW1 T|"
    "proof:P R UW1 F|proper:P R AA1 P ER0|property:P R AA1 P ER0 T IY0|"
    "proportion:P R AH0 P AO1 R SH AH0 N|proposal:P R AH0 P OW1 Z AH0 L|"
    "propose:P R AH0 P OW1 Z|prospect:P R AA1 S P EH0 K T|"
    "protect:P R AH0 T EH1 K T|protection:P R AH0 T EH1 K SH AH0 N|"
    "protein:P R OW1 T IY2 N|protest:P R OW1 T EH0 S T|"
    "proud:P R AW1 D|prove:P R UW1 V|provision:P R AH0 V IH1 ZH AH0 N|"
    "psychological:S AY2 K AH0 L AA1 JH AH0 K AH0 L|"
    "pull:P UH1 L|punishment:P AH1 N IH0 SH M AH0 N T|"
    "purchase:P ER1 CH AH0 S|pure:P Y UH1 R|purpose:P ER1 P AH0 S|"
    "pursue:P ER0 S UW1|push:P UH1 SH|qualify:K W AA1 L AH0 F AY2|"
    "quarter:K W AO1 R T ER0|queen:K W IY1 N|quiet:K W AY1 AH0 T|"
    "quit:K W IH1 T|quite:K W AY1 T|quote:K W OW1 T|"
    "race:R EY1 S|radical:R AE1 D AH0 K AH0 L|radio:R EY1 D IY0 OW0|"
    "rage:R EY1 JH|rain:R EY1 N|raise:R EY1 Z|"
    "range:R EY1 N JH|rank:R AE1 NG K|rapid:R AE1 P AH0 D|"
    "rare:R EH1 R|rate:R EY1 T|rather:R AE1 DH ER0|"
    "raw:R AO1|reaction:R IY0 AE1 K SH AH0 N|reality:R IY0 AE1 L AH0 T IY0|"
    "realize:R IY1 AH0 L AY2 Z|reasonable:R IY1 Z AH0 N AH0 B AH0 L|"
    "receive:R AH0 S IY1 V|recognition:R EH2 K AH0 G N IH1 SH AH0 N|"
    "recognize:R EH1 K AH0 G N AY2 Z|recommend:R EH2 K AH0 M EH1 N D|"
    "record:R EH1 K ER0 D|recover:R IH0 K AH1 V ER0|"
    "reduce:R IH0 D UW1 S|reduction:R IH0 D AH1 K SH AH0 N|"
    "refer:R AH0 F ER1|reference:R EH1 F ER0 AH0 N S|"
    "reflect:R IH0 F L EH1 K T|reform:R IH0 F AO1 R M|"
    "refuse:R IH0 F Y UW1 Z|regard:R IH0 G AA1 R D|"
    "region:R IY1 JH AH0 N|regional:R IY1 JH AH0 N AH0 L|"
    "register:R EH1 JH AH0 S T ER0|regular:R EH1 G Y AH0 L ER0|"
    "regulate:R EH1 G Y AH0 L EY2 T|regulation:R EH2 G Y AH0 L EY1 SH AH0 N|"
    "reject:R IH0 JH EH1 K T|relate:R IH0 L EY1 T|"
    "relation:R IH0 L EY1 SH AH0 N|relative:R EH1 L AH0 T IH0 V|"
    "release:R IH0 L IY1 S|relevant:R EH1 L AH0 V AH0 N T|"
    "relief:R IH0 L IY1 F|religion:R IH0 L IH1 JH AH0 N|"
    "religious:R IH0 L IH1 JH AH0 S|rely:R IH0 L AY1|"
    "remain:R IH0 M EY1 N|remaining:R IH0 M EY1 N IH0 NG|"
    "remark:R IH0 M AA1 R K|remarkable:R IH0 M AA1 R K AH0 B AH0 L|"
    "remind:R IY0 M AY1 N D|remote:R IH0 M OW1 T|"
    "remove:R IH0 M UW1 V|repeat:R IH0 P IY1 T|"
    "replace:R IH0 P L EY1 S|reply:R IH0 P L AY1|"
    "represent:R EH2 P R IH0 Z EH1 N T|republic:R IH0 P AH1 B L IH0 K|"
    "reputation:R EH2 P Y AH0 T EY1 SH AH0 N|"
    "request:R IH0 K W EH1 S T|require:R IH0 K W AY1 ER0|"
    "requirement:R IH0 K W AY1 ER0 M AH0 N T|"
    "resolve:R IH0 Z AA1 L V|resource:R IY1 S AO2 R S|"
    "respond:R IH0 S P AA1 N D|responsibility:R IH0 S P AA2 N S AH0 B IH1 L AH0 T IY0|"
    "rest:R EH1 S T|restaurant:R EH1 S T ER0 AA2 N T|"
    "restore:R IH0 S T AO1 R|restriction:R IH0 S T R IH1 K SH AH0 N|"
    "retain:R IH0 T EY1 N|retire:R IH0 T AY1 ER0|"
    "return:R IH0 T ER1 N|reveal:R IH0 V IY1 L|"
    "revenue:R EH1 V AH0 N UW2|review:R IH0 V Y UW1|"
    "revolution:R EH2 V AH0 L UW1 SH AH0 N|rhythm:R IH1 DH AH0 M|"
    "rice:R AY1 S|rich:R IH1 CH|ride:R AY1 D|"
    "ring:R IH1 NG|rise:R AY1 Z|risk:R IH1 S K|"
    "road:R OW1 D|rock:R AA1 K|role:R OW1 L|"
    "roll:R OW1 L|romantic:R OW0 M AE1 N T IH0 K|"
    "roof:R UW1 F|root:R UW1 T|rope:R OW1 P|"
    "rough:R AH1 F|round:R AW1 N D|route:R UW1 T|"
    "row:R OW1|royal:R OY1 AH0 L|ruin:R UW1 AH0 N|"
    "rule:R UW1 L|rural:R UH1 R AH0 L|rush:R AH1 SH|"
    "sacred:S EY1 K R AH0 D|sad:S AE1 D|safe:S EY1 F|"
    "safety:S EY1 F T IY0|sake:S EY1 K|salary:S AE1 L ER0 IY0|"
    "sale:S EY1 L|salt:S AO1 L T|sample:S AE1 M P AH0 L|"
    "sand:S AE1 N D|satellite:S AE1 T AH0 L AY2 T|"
    "satisfy:S AE1 T AH0 S F AY2|save:S EY1 V|"
    "scale:S K EY1 L|scene:S IY1 N|schedule:S K EH1 JH UW0 L|"
    "scholar:S K AA1 L ER0|science:S AY1 AH0 N S|"
    "scientific:S AY2 AH0 N T IH1 F IH0 K|scientist:S AY1 AH0 N T IH0 S T|"
    "scope:S K OW1 P|score:S K AO1 R|screen:S K R IY1 N|"
    "search:S ER1 CH|season:S IY1 Z AH0 N|seat:S IY1 T|"
    "secret:S IY1 K R AH0 T|secretary:S EH1 K R AH0 T EH2 R IY0|"
    "section:S EH1 K SH AH0 N|sector:S EH1 K T ER0|"
    "seed:S IY1 D|seek:S IY1 K|select:S AH0 L EH1 K T|"
    "selection:S AH0 L EH1 K SH AH0 N|sell:S EH1 L|"
    "senate:S EH1 N AH0 T|senator:S EH1 N AH0 T ER0|"
    "send:S EH1 N D|senior:S IY1 N Y ER0|sense:S EH1 N S|"
    "sensitive:S EH1 N S AH0 T IH0 V|separate:S EH1 P ER0 AH0 T|"
    "sequence:S IY1 K W AH0 N S|series:S IH1 R IY0 Z|"
    "session:S EH1 SH AH0 N|settle:S EH1 T AH0 L|"
    "settlement:S EH1 T AH0 L M AH0 N T|severe:S AH0 V IH1 R|"
    "sexual:S EH1 K SH UW0 AH0 L|shade:SH EY1 D|"
    "shadow:SH AE1 D OW0|shake:SH EY1 K|shall:SH AE1 L|"
    "shape:SH EY1 P|share:SH EH1 R|sharp:SH AA1 R P|"
    "she:SH IY1|sheet:SH IY1 T|shelf:SH EH1 L F|"
    "shell:SH EH1 L|shelter:SH EH1 L T ER0|shift:SH IH1 F T|"
    "shine:SH AY1 N|shoe:SH UW1|shoot:SH UW1 T|"
    "shop:SH AA1 P|shopping:SH AA1 P IH0 NG|shore:SH AO1 R|"
    "shot:SH AA1 T|shoulder:SH OW1 L D ER0|shout:SH AW1 T|"
    "shut:SH AH1 T|sick:S IH1 K|sight:S AY1 T|"
    "sign:S AY1 N|significance:S IH0 G N IH1 F AH0 K AH0 N S|"
    "silence:S AY1 L AH0 N S|silent:S AY1 L AH0 N T|"
    "silver:S IH1 L V ER0|simply:S IH1 M P L IY0|"
    "sin:S IH1 N|sing:S IH1 NG|sister:S IH1 S T ER0|"
    "sit:S IH1 T|site:S AY1 T|situation:S IH2 CH UW0 EY1 SH AH0 N|"
    "size:S AY1 Z|skill:S K IH1 L|skin:S K IH1 N|"
    "sky:S K AY1|slave:S L EY1 V|sleep:S L IY1 P|"
    "slice:S L AY1 S|slide:S L AY1 D|slight:S L AY1 T|"
    "slip:S L IH1 P|slow:S L OW1|smart:S M AA1 R T|"
    "smell:S M EH1 L|smile:S M AY1 L|smoke:S M OW1 K|"
    "smooth:S M UW1 DH|snap:S N AE1 P|snow:S N OW1|"
    "soft:S AO1 F T|soil:S OY1 L|soldier:S OW1 L JH ER0|"
    "solid:S AA1 L AH0 D|solution:S AH0 L UW1 SH AH0 N|"
    "solve:S AA1 L V|somehow:S AH1 M HH AW2|"
    "somewhat:S AH1 M W AH2 T|son:S AH1 N|soul:S OW1 L|"
    "source:S AO1 R S|south:S AW1 TH|southern:S AH1 DH ER0 N|"
    "space:S P EY1 S|speak:S P IY1 K|speaker:S P IY1 K ER0|"
    "specific:S P AH0 S IH1 F IH0 K|spend:S P EH1 N D|"
    "spirit:S P IH1 R AH0 T|spiritual:S P IH1 R AH0 CH UW0 AH0 L|"
    "split:S P L IH1 T|spokesman:S P OW1 K S M AH0 N|"
    "sport:S P AO1 R T|spot:S P AA1 T|spread:S P R EH1 D|"
    "spring:S P R IH1 NG|square:S K W EH1 R|stable:S T EY1 B AH0 L|"
    "staff:S T AE1 F|stage:S T EY1 JH|stair:S T EH1 R|"
    "stake:S T EY1 K|standard:S T AE1 N D ER0 D|"
    "star:S T AA1 R|stare:S T EH1 R|statement:S T EY1 T M AH0 N T|"
    "station:S T EY1 SH AH0 N|status:S T AE1 T AH0 S|"
    "stay:S T EY1|steady:S T EH1 D IY0|steal:S T IY1 L|"
    "steel:S T IY1 L|steep:S T IY1 P|stem:S T EH1 M|"
    "step:S T EH1 P|stick:S T IH1 K|stir:S T ER1|"
    "stock:S T AA1 K|stomach:S T AH1 M AH0 K|stone:S T OW1 N|"
    "store:S T AO1 R|storm:S T AO1 R M|straight:S T R EY1 T|"
    "strange:S T R EY1 N JH|stranger:S T R EY1 N JH ER0|"
    "strategic:S T R AH0 T IY1 JH IH0 K|stream:S T R IY1 M|"
    "street:S T R IY1 T|strength:S T R EH1 NG TH|"
    "stress:S T R EH1 S|stretch:S T R EH1 CH|"
    "strike:S T R AY1 K|string:S T R IH1 NG|strip:S T R IH1 P|"
    "stroke:S T R OW1 K|strong:S T R AO1 NG|"
    "strongly:S T R AO1 NG L IY0|structure:S T R AH1 K CH ER0|"
    "struggle:S T R AH1 G AH0 L|student:S T UW1 D AH0 N T|"
    "studio:S T UW1 D IY0 OW0|stuff:S T AH1 F|"
    "stupid:S T UW1 P AH0 D|style:S T AY1 L|"
    "subject:S AH1 B JH EH0 K T|submit:S AH0 B M IH1 T|"
    "substance:S AH1 B S T AH0 N S|succeed:S AH0 K S IY1 D|"
    "success:S AH0 K S EH1 S|successful:S AH0 K S EH1 S F AH0 L|"
    "sudden:S AH1 D AH0 N|suffer:S AH1 F ER0|"
    "sufficient:S AH0 F IH1 SH AH0 N T|sugar:SH UH1 G ER0|"
    "suggest:S AH0 JH EH1 S T|suit:S UW1 T|summer:S AH1 M ER0|"
    "supply:S AH0 P L AY1|support:S AH0 P AO1 R T|"
    "suppose:S AH0 P OW1 Z|surface:S ER1 F AH0 S|"
    "surgery:S ER1 JH ER0 IY0|surprise:S ER0 P R AY1 Z|"
    "surround:S ER0 AW1 N D|survey:S ER1 V EY0|"
    "survive:S ER0 V AY1 V|suspect:S AH0 S P EH1 K T|"
    "sweet:S W IY1 T|swim:S W IH1 M|swing:S W IH1 NG|"
    "switch:S W IH1 CH|symbol:S IH1 M B AH0 L|"
    "sympathy:S IH1 M P AH0 TH IY0|talent:T AE1 L AH0 N T|"
    "tall:T AO1 L|tank:T AE1 NG K|tape:T EY1 P|"
    "target:T AA1 R G AH0 T|task:T AE1 S K|taste:T EY1 S T|"
    "tax:T AE1 K S|tea:T IY1|teach:T IY1 CH|"
    "teacher:T IY1 CH ER0|team:T IY1 M|tear:T EH1 R|"
    "telephone:T EH1 L AH0 F OW2 N|temperature:T EH1 M P R AH0 CH ER0|"
    "temporary:T EH1 M P ER0 EH2 R IY0|tend:T EH1 N D|"
    "tension:T EH1 N SH AH0 N|term:T ER1 M|terrible:T EH1 R AH0 B AH0 L|"
    "territory:T EH1 R AH0 T AO2 R IY0|terror:T EH1 R ER0|"
    "test:T EH1 S T|text:T EH1 K S T|theme:TH IY1 M|"
    "theory:TH IY1 ER0 IY0|therapy:TH EH1 R AH0 P IY0|"
    "thick:TH IH1 K|thin:TH IH1 N|threat:TH R EH1 T|"
    "threaten:TH R EH1 T AH0 N|throat:TH R OW1 T|"
    "throw:TH R OW1|thus:DH AH1 S|ticket:T IH1 K AH0 T|"
    "tie:T AY1|tight:T AY1 T|tiny:T AY1 N IY0|"
    "tip:T IH1 P|tire:T AY1 ER0|title:T AY1 T AH0 L|"
    "toe:T OW1|tomorrow:T AH0 M AA1 R OW0|tone:T OW1 N|"
    "tongue:T AH1 NG|tonight:T AH0 N AY1 T|tool:T UW1 L|"
    "tooth:T UW1 TH|total:T OW1 T AH0 L|totally:T OW1 T AH0 L IY0|"
    "touch:T AH1 CH|tough:T AH1 F|tour:T UH1 R|"
    "tourist:T UH1 R AH0 S T|tower:T AW1 ER0|track:T R AE1 K|"
    "trade:T R EY1 D|tradition:T R AH0 D IH1 SH AH0 N|"
    "traffic:T R AE1 F IH0 K|trail:T R EY1 L|train:T R EY1 N|"
    "training:T R EY1 N IH0 NG|transfer:T R AE1 N S F ER0|"
    "transform:T R AE0 N S F AO1 R M|transition:T R AE0 N Z IH1 SH AH0 N|"
    "translate:T R AE0 N S L EY1 T|transport:T R AE1 N S P AO2 R T|"
    "travel:T R AE1 V AH0 L|treat:T R IY1 T|"
    "treatment:T R IY1 T M AH0 N T|treaty:T R IY1 T IY0|"
    "tremendous:T R AH0 M EH1 N D AH0 S|trend:T R EH1 N D|"
    "trial:T R AY1 AH0 L|trick:T R IH1 K|trip:T R IH1 P|"
    "troop:T R UW1 P|trouble:T R AH1 B AH0 L|truck:T R AH1 K|"
    "trust:T R AH1 S T|truth:T R UW1 TH|tube:T UW1 B|"
    "twice:T W AY1 S|twin:T W IH1 N|type:T AY1 P|"
    "typical:T IH1 P AH0 K AH0 L|ugly:AH1 G L IY0|"
    "ultimately:AH1 L T AH0 M AH0 T L IY0|uncle:AH1 NG K AH0 L|"
    "undergo:AH2 N D ER0 G OW1|union:Y UW1 N Y AH0 N|"
    "unique:Y UW0 N IY1 K|unite:Y UW0 N AY1 T|"
    "universal:Y UW2 N AH0 V ER1 S AH0 L|unless:AH0 N L EH1 S|"
    "unlike:AH0 N L AY1 K|unlikely:AH0 N L AY1 K L IY0|"
    "unusual:AH0 N Y UW1 ZH UW0 AH0 L|update:AH1 P D EY2 T|"
    "upon:AH0 P AA1 N|upper:AH1 P ER0|upset:AH0 P S EH1 T|"
    "urban:ER1 B AH0 N|urge:ER1 JH|useful:Y UW1 S F AH0 L|"
    "user:Y UW1 Z ER0|usual:Y UW1 ZH UW0 AH0 L|"
    "valley:V AE1 L IY0|valuable:V AE1 L Y AH0 B AH0 L|"
    "variety:V ER0 AY1 AH0 T IY0|vast:V AE1 S T|"
    "vehicle:V IY1 AH0 K AH0 L|version:V ER1 ZH AH0 N|"
    "victim:V IH1 K T AH0 M|victory:V IH1 K T ER0 IY0|"
    "view:V Y UW1|village:V IH1 L AH0 JH|violence:V AY1 AH0 L AH0 N S|"
    "virtue:V ER1 CH UW0|visible:V IH1 Z AH0 B AH0 L|"
    "vision:V IH1 ZH AH0 N|visit:V IH1 Z AH0 T|"
    "visitor:V IH1 Z AH0 T ER0|vital:V AY1 T AH0 L|"
    "volume:V AA1 L Y UW0 M|volunteer:V AA2 L AH0 N T IH1 R|"
    "vote:V OW1 T|wage:W EY1 JH|wake:W EY1 K|"
    "wall:W AO1 L|wander:W AA1 N D ER0|warn:W AO1 R N|"
    "warning:W AO1 R N IH0 NG|wash:W AA1 SH|waste:W EY1 S T|"
    "wave:W EY1 V|weak:W IY1 K|weakness:W IY1 K N AH0 S|"
    "wealth:W EH1 L TH|weapon:W EH1 P AH0 N|wear:W EH1 R|"
    "weather:W EH1 DH ER0|web:W EH1 B|wedding:W EH1 D IH0 NG|"
    "week:W IY1 K|weekend:W IY1 K EH2 N D|weigh:W EY1|"
    "weight:W EY1 T|weird:W IH1 R D|west:W EH1 S T|"
    "western:W EH1 S T ER0 N|wet:W EH1 T|wheel:W IY1 L|"
    "whenever:W EH0 N EH1 V ER0|whereas:W EH1 R AE2 Z|"
    "wherever:W EH0 R EH1 V ER0|whom:HH UW1 M|"
    "wide:W AY1 D|widely:W AY1 D L IY0|wife:W AY1 F|"
    "wild:W AY1 L D|win:W IH1 N|wind:W IH1 N D|"
    "window:W IH1 N D OW0|wine:W AY1 N|wing:W IH1 NG|"
    "winner:W IH1 N ER0|winter:W IH1 N T ER0|wire:W AY1 ER0|"
    "wise:W AY1 Z|wish:W IH1 SH|witness:W IH1 T N AH0 S|"
    "wonder:W AH1 N D ER0|wonderful:W AH1 N D ER0 F AH0 L|"
    "wood:W UH1 D|wooden:W UH1 D AH0 N|worker:W ER1 K ER0|"
    "works:W ER1 K S|workshop:W ER1 K SH AA2 P|worry:W ER1 IY0|"
    "worth:W ER1 TH|wound:W UW1 N D|wrap:R AE1 P|"
    "yard:Y AA1 R D|yeah:Y AE1|yesterday:Y EH1 S T ER0 D EY2|"
    "yet:Y EH1 T|yield:Y IY1 L D|youth:Y UW1 TH|"
    "zone:Z OW1 N"
)

_NUMBER_WORDS = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
    10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
    14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen",
    18: "eighteen", 19: "nineteen", 20: "twenty", 30: "thirty",
    40: "forty", 50: "fifty", 60: "sixty", 70: "seventy",
    80: "eighty", 90: "ninety",
}

_ORDINALS = {
    "1st": "first", "2nd": "second", "3rd": "third",
    "4th": "fourth", "5th": "fifth", "6th": "sixth",
    "7th": "seventh", "8th": "eighth", "9th": "ninth",
    "10th": "tenth",
}

_CONTRACTIONS = {
    "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
    "you're": "you are", "you've": "you have", "you'll": "you will",
    "he's": "he is", "she's": "she is", "it's": "it is",
    "we're": "we are", "we've": "we have", "we'll": "we will",
    "they're": "they are", "they've": "they have", "they'll": "they will",
    "that's": "that is", "there's": "there is", "here's": "here is",
    "what's": "what is", "who's": "who is", "where's": "where is",
    "can't": "cannot", "won't": "will not", "don't": "do not",
    "doesn't": "does not", "didn't": "did not", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "hasn't": "has not", "haven't": "have not", "hadn't": "had not",
    "wouldn't": "would not", "shouldn't": "should not",
    "couldn't": "could not", "let's": "let us",
}

_LTS_RULES: List[Tuple[str, str, str, str]] = [
    ("", "tion", "", "SH AH0 N"),
    ("", "sion", "", "ZH AH0 N"),
    ("", "ous", "", "AH0 S"),
    ("", "ight", "", "AY1 T"),
    ("", "ough", "", "AH1 F"),
    ("", "ture", "", "CH ER0"),
    ("", "ness", "", "N AH0 S"),
    ("", "ment", "", "M AH0 N T"),
    ("", "able", "", "AH0 B AH0 L"),
    ("", "ible", "", "AH0 B AH0 L"),
    ("", "ful", "", "F AH0 L"),
    ("", "ing", "", "IH0 NG"),
    ("", "tion", "", "SH AH0 N"),
    ("", "ly", "", "L IY0"),
    ("", "ed", "", "D"),
    ("", "er", "", "ER0"),
    ("", "th", "", "TH"),
    ("", "sh", "", "SH"),
    ("", "ch", "", "CH"),
    ("", "ph", "", "F"),
    ("", "wh", "", "W"),
    ("", "ng", "", "NG"),
    ("", "ck", "", "K"),
    ("", "qu", "", "K W"),
    ("", "ee", "", "IY1"),
    ("", "ea", "", "IY1"),
    ("", "oo", "", "UW1"),
    ("", "ou", "", "AW1"),
    ("", "ai", "", "EY1"),
    ("", "ay", "", "EY1"),
    ("", "oi", "", "OY1"),
    ("", "oy", "", "OY1"),
    ("", "ow", "", "OW1"),
    ("", "au", "", "AO1"),
    ("", "aw", "", "AO1"),
    ("", "ew", "", "Y UW1"),
    ("", "ie", "", "IY1"),
    ("", "ei", "", "EY1"),
    ("", "oa", "", "OW1"),
    ("", "ue", "", "UW1"),
]

_LETTER_PHONEMES = {
    "a": "AE1", "b": "B", "c": "K", "d": "D", "e": "EH1",
    "f": "F", "g": "G", "h": "HH", "i": "IH1", "j": "JH",
    "k": "K", "l": "L", "m": "M", "n": "N", "o": "AA1",
    "p": "P", "q": "K", "r": "R", "s": "S", "t": "T",
    "u": "AH1", "v": "V", "w": "W", "x": "K S", "y": "Y",
    "z": "Z",
}


class NRSPhonemeEngine:
    """Grapheme-to-phoneme engine with CMU-style pronunciation dictionary."""

    def __init__(self):
        self._dict: Dict[str, List[str]] = {}
        self._load_builtin_dict()

    def _load_builtin_dict(self):
        for entry in _DICT_DATA.split("|"):
            entry = entry.strip()
            if not entry or ":" not in entry:
                continue
            word, phones = entry.split(":", 1)
            self._dict[word.strip().lower()] = phones.strip().split()

    def text_to_phonemes(self, text: str) -> List[PhonemeToken]:
        text = self._normalize(text)
        tokens = self._tokenize(text)
        result: List[PhonemeToken] = []
        result.append(PhonemeToken("SIL", is_pause=True, duration_ms=60))

        for i, tok in enumerate(tokens):
            if tok in (".", "!", ";"):
                result.append(PhonemeToken("PAU", is_pause=True,
                                           duration_ms=250,
                                           phrase_boundary=True))
            elif tok == ",":
                result.append(PhonemeToken("PAU", is_pause=True,
                                           duration_ms=150))
            elif tok in (":", "-"):
                result.append(PhonemeToken("SIL", is_pause=True,
                                           duration_ms=100))
            elif tok == "?":
                result.append(PhonemeToken("PAU", is_pause=True,
                                           duration_ms=250,
                                           phrase_boundary=True))
            else:
                phones = self._word_to_phonemes(tok)
                for ph_str in phones:
                    base = re.sub(r"[012]", "", ph_str)
                    stress = 0
                    if ph_str and ph_str[-1].isdigit():
                        stress = int(ph_str[-1])
                    params = _PHONEME_PARAMS.get(base, _PHONEME_PARAMS["AX"])
                    dur = params[9]
                    if stress == 1:
                        dur *= 1.3
                    elif stress == 2:
                        dur *= 1.1
                    result.append(PhonemeToken(
                        phoneme=base, stress=stress, duration_ms=dur))
                result.append(PhonemeToken("SIL", is_pause=True,
                                           duration_ms=30))

        result.append(PhonemeToken("PAU", is_pause=True, duration_ms=200))
        return result

    def _normalize(self, text: str) -> str:
        text = text.strip()
        for contraction, expansion in _CONTRACTIONS.items():
            text = re.sub(re.escape(contraction), expansion, text,
                          flags=re.IGNORECASE)
        text = re.sub(r"(\d+)(st|nd|rd|th)\b", self._expand_ordinal, text)
        text = re.sub(r"\b\d+\b", lambda m: self._expand_number(m.group()),
                       text)
        return text.lower()

    def _expand_ordinal(self, match: re.Match) -> str:
        full = match.group().lower()
        if full in _ORDINALS:
            return _ORDINALS[full]
        n = int(match.group(1))
        return self._expand_number(str(n))

    def _expand_number(self, n_str: str) -> str:
        try:
            n = int(n_str)
        except ValueError:
            return n_str
        if n < 0:
            return "minus " + self._expand_number(str(-n))
        if n in _NUMBER_WORDS:
            return _NUMBER_WORDS[n]
        if n < 100:
            tens = (n // 10) * 10
            ones = n % 10
            return _NUMBER_WORDS.get(tens, "") + " " + _NUMBER_WORDS.get(ones, "")
        if n < 1000:
            hundreds = n // 100
            remainder = n % 100
            result = _NUMBER_WORDS.get(hundreds, str(hundreds)) + " hundred"
            if remainder:
                result += " " + self._expand_number(str(remainder))
            return result
        if n < 1000000:
            thousands = n // 1000
            remainder = n % 1000
            result = self._expand_number(str(thousands)) + " thousand"
            if remainder:
                result += " " + self._expand_number(str(remainder))
            return result
        return n_str

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-z']+|[.,!?;:\-]", text)

    def _word_to_phonemes(self, word: str) -> List[str]:
        w = word.lower().strip("'")
        if w in self._dict:
            return list(self._dict[w])
        return self._rule_based_g2p(w)

    def _rule_based_g2p(self, word: str) -> List[str]:
        result: List[str] = []
        i = 0
        while i < len(word):
            matched = False
            for _, pattern, _, replacement in _LTS_RULES:
                plen = len(pattern)
                if i + plen <= len(word) and word[i:i + plen] == pattern:
                    result.extend(replacement.split())
                    i += plen
                    matched = True
                    break
            if not matched:
                ch = word[i]
                if ch in _LETTER_PHONEMES:
                    result.extend(_LETTER_PHONEMES[ch].split())
                i += 1

        if word.endswith("e") and len(word) > 2 and word[-2] not in "aeiou":
            if result and re.sub(r"[012]", "", result[-1]) == "EH":
                result.pop()

        return result if result else ["AX"]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. NRS VOCODER — Source-Filter Speech Synthesis
# ═══════════════════════════════════════════════════════════════════════════════

def _resonator_coeffs(fc: float, bw: float, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    if fc <= 0 or bw <= 0 or fc >= sr / 2:
        return np.array([1.0]), np.array([1.0])
    C = -np.exp(-_TWO_PI * bw / sr)
    B = 2.0 * np.exp(-math.pi * bw / sr) * math.cos(_TWO_PI * fc / sr)
    A = 1.0 - B - C
    return np.array([A]), np.array([1.0, -B, -C])


def _lf_glottal_pulse(f0_contour: np.ndarray, sr: int,
                       jitter: float = 0.003,
                       shimmer: float = 0.02) -> np.ndarray:
    """Liljencrants-Fant glottal pulse train with per-sample F0."""
    n = len(f0_contour)
    out = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(42)
    pos = 0.0

    while int(pos) < n:
        idx = min(int(pos), n - 1)
        f0 = float(f0_contour[idx])
        if f0 <= 0:
            pos += sr / 130.0
            continue

        T0 = sr / f0
        T0 *= 1.0 + jitter * rng.standard_normal()
        T0 = max(T0, sr / 500)

        Te = 0.6 * T0
        Ta = 0.035 * T0
        wg = math.pi / max(Te, 1.0)
        epsilon = 1.0 / max(Ta, 0.5)
        amp = 1.0 + shimmer * rng.standard_normal()

        pulse_len = min(int(T0), n - int(pos))
        if pulse_len <= 0:
            break

        k = np.arange(pulse_len, dtype=np.float32)
        pulse = np.zeros(pulse_len, dtype=np.float32)

        open_mask = k <= Te
        ret_mask = ~open_mask
        pulse[open_mask] = (amp * np.sin(wg * k[open_mask])
                            * np.exp(-0.5 * k[open_mask] / max(Te, 1.0)))
        if ret_mask.any():
            pulse[ret_mask] = amp * -0.3 * np.exp(-epsilon * (k[ret_mask] - Te))

        start = int(pos)
        out[start:start + pulse_len] += pulse
        pos += T0

    return out


class NRSVocoder:
    """NRS-native speech synthesis vocoder using source-filter model."""

    def __init__(self, sr: int = SPEECH_SR):
        self._sr = sr

    def synthesize(self, phonemes: List[PhonemeToken],
                   voice: Optional[VoiceProfile] = None,
                   emotion: str = "neutral",
                   speed: float = 1.0) -> np.ndarray:
        if voice is None:
            voice = VoiceProfile.male_default()

        segments = self._build_segments(phonemes, voice, speed)
        total_samples = sum(s[1] for s in segments)
        if total_samples == 0:
            return np.zeros(self._sr, dtype=np.float32)

        f0_contour = self._build_f0_contour(
            segments, total_samples, voice, emotion, phonemes)
        audio = self._source_filter(segments, f0_contour, voice)

        audio = self._lip_radiation(audio)

        peak = np.abs(audio).max()
        if peak > 1e-8:
            audio = audio * (0.9 / peak)

        return audio.astype(np.float32)

    def _build_segments(self, phonemes: List[PhonemeToken],
                        voice: VoiceProfile,
                        speed: float) -> List[Tuple[str, int, int]]:
        segments = []
        for ph in phonemes:
            dur_ms = ph.duration_ms / speed
            n_samples = max(int(self._sr * dur_ms / 1000.0), 1)
            segments.append((ph.phoneme, n_samples, ph.stress))
        return segments

    def _build_f0_contour(self, segments: List[Tuple[str, int, int]],
                           total_samples: int,
                           voice: VoiceProfile,
                           emotion: str,
                           phonemes: List[PhonemeToken]) -> np.ndarray:
        f0 = np.zeros(total_samples, dtype=np.float32)
        sr = self._sr
        f0_base = voice.pitch_base

        emo_range = 1.0
        emo_speed = 1.0
        emo_level = 1.0
        if emotion == "happy":
            emo_range = 1.4
            emo_speed = 1.1
            emo_level = 1.05
        elif emotion == "sad":
            emo_range = 0.6
            emo_speed = 0.85
            emo_level = 0.9
        elif emotion == "angry":
            emo_range = 1.3
            emo_speed = 1.15
            emo_level = 1.15
        elif emotion == "calm":
            emo_range = 0.7
            emo_speed = 0.9
            emo_level = 0.95

        is_question = any(p.phrase_boundary and p.phoneme == "PAU"
                          for p in phonemes[-3:])

        pos = 0
        v_idx = 0
        for ph_name, n_samp, stress in segments:
            params = _PHONEME_PARAMS.get(ph_name, _PHONEME_PARAMS.get("AX"))
            if params is None:
                pos += n_samp
                continue
            voiced = params[6]
            if voiced and ph_name not in ("SIL", "PAU"):
                phr_t = pos / max(total_samples, 1)
                declination = 1.0 - 0.12 * phr_t
                microprosody = 1.0 + 0.015 * math.sin(v_idx * 1.7)

                stress_factor = 1.0
                if stress == 1:
                    stress_factor = 1.15
                elif stress == 2:
                    stress_factor = 1.05

                t_arr = np.arange(n_samp, dtype=np.float32) / sr
                vib = 1.0 + voice.vibrato_depth * np.sin(
                    _TWO_PI * voice.vibrato_rate * (t_arr + pos / sr))

                local_f0 = (f0_base * declination * microprosody
                            * stress_factor * emo_level * vib)

                range_mod = voice.pitch_range * emo_range
                local_f0 *= (1.0 + range_mod * 0.1 * math.sin(v_idx * 0.8))

                if is_question and phr_t > 0.7:
                    rise = 1.0 + 0.2 * (phr_t - 0.7) / 0.3
                    local_f0 *= rise

                end = min(pos + n_samp, total_samples)
                f0[pos:end] = local_f0[:end - pos]
                v_idx += 1
            pos += n_samp

        return f0

    def _source_filter(self, segments: List[Tuple[str, int, int]],
                        f0_contour: np.ndarray,
                        voice: VoiceProfile) -> np.ndarray:
        sr = self._sr
        n = len(f0_contour)

        source = _lf_glottal_pulse(f0_contour, sr, voice.jitter, voice.shimmer)

        source_smooth = np.empty_like(source)
        source_smooth[0] = source[0]
        source_smooth[1:] = 0.6 * source[1:] + 0.4 * source[:-1]
        source = source_smooth

        rng = np.random.default_rng(42)
        asp_noise = rng.standard_normal(n).astype(np.float32) * voice.breathiness

        nyq = sr / 2
        lo = max(2000 / nyq, 0.001)
        hi = min(12000 / nyq, 0.999)
        b_fric, a_fric = butter(4, [lo, hi], btype="band")
        fric_noise = lfilter(b_fric, a_fric,
                             rng.standard_normal(n).astype(np.float32))

        N_F = 5
        ff = np.zeros((n, N_F), dtype=np.float64)
        fb = np.zeros((n, N_F), dtype=np.float64)
        v_mask = np.zeros(n, dtype=bool)
        fr_amp = np.zeros(n, dtype=np.float32)
        nas_mask = np.zeros(n, dtype=bool)

        pos = 0
        for ph_name, n_samp, _stress in segments:
            params = _PHONEME_PARAMS.get(ph_name, _PHONEME_PARAMS.get("AX"))
            if params is None:
                pos += n_samp
                continue
            f1, f2, f3, bw1, bw2, bw3, voiced, fric, nasal, _ = params
            end = min(pos + n_samp, n)

            shift = voice.formant_shift
            ff[pos:end, 0] = f1 * shift
            ff[pos:end, 1] = f2 * shift
            ff[pos:end, 2] = f3 * shift
            ff[pos:end, 3] = 3300 * shift
            ff[pos:end, 4] = 3750 * shift

            fb[pos:end, 0] = bw1
            fb[pos:end, 1] = bw2
            fb[pos:end, 2] = bw3
            fb[pos:end, 3] = 250
            fb[pos:end, 4] = 300

            v_mask[pos:end] = voiced
            fr_amp[pos:end] = fric
            nas_mask[pos:end] = nasal
            pos = end

        smooth_n = int(0.040 * sr)
        if smooth_n > 1 and n > smooth_n:
            kern = np.hanning(smooth_n)
            kern /= kern.sum()
            for fi in range(N_F):
                ff[:, fi] = np.convolve(ff[:, fi], kern, mode="same")
                fb[:, fi] = np.convolve(fb[:, fi], kern, mode="same")
            fb = np.maximum(fb, 40.0)

        SUBFRAME = int(sr * 0.005)
        output = np.zeros(n, dtype=np.float64)
        states = [np.zeros(2, dtype=np.float64) for _ in range(N_F)]
        nasal_r_st = np.zeros(2, dtype=np.float64)

        for s in range(0, n, SUBFRAME):
            e = min(s + SUBFRAME, n)
            c = min(s + (e - s) // 2, n - 1)

            seg = source[s:e].astype(np.float64)
            if v_mask[c]:
                seg = seg + asp_noise[s:e].astype(np.float64)

            for fi in range(N_F):
                fc = ff[c, fi]
                bw = fb[c, fi]
                if fc <= 0 or bw <= 0 or fc >= sr / 2:
                    continue
                b, a = _resonator_coeffs(fc, bw, sr)
                seg, states[fi] = lfilter(b, a, seg, zi=states[fi])
                pk = np.abs(seg).max()
                if pk > 2.0:
                    seg *= 2.0 / pk

            if nas_mask[c]:
                b_nr, a_nr = _resonator_coeffs(300.0, 120.0, sr)
                seg, nasal_r_st = lfilter(b_nr, a_nr, seg, zi=nasal_r_st)

            fl = float(fr_amp[c])
            if fl > 0:
                seg = seg + fric_noise[s:e].astype(np.float64) * fl * 0.3

            output[s:e] = seg

        return output.astype(np.float32)

    def _lip_radiation(self, signal: np.ndarray) -> np.ndarray:
        radiated = np.empty_like(signal)
        radiated[0] = signal[0]
        radiated[1:] = signal[1:] - 0.97 * signal[:-1]
        return radiated


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MUSIC COMPOSITION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_SCALE_INTERVALS = {
    "major":     [0, 2, 4, 5, 7, 9, 11],
    "minor":     [0, 2, 3, 5, 7, 8, 10],
    "dorian":    [0, 2, 3, 5, 7, 9, 10],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "pentatonic": [0, 2, 4, 7, 9],
    "blues":     [0, 3, 5, 6, 7, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
}

_CHORD_INTERVALS = {
    "maj":  [0, 4, 7],
    "min":  [0, 3, 7],
    "dim":  [0, 3, 6],
    "aug":  [0, 4, 8],
    "dom7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
}

_STYLE_PROGRESSIONS = {
    "pop":       [(0, "maj"), (4, "maj"), (5, "min"), (3, "maj")],
    "jazz":      [(1, "min7"), (4, "dom7"), (0, "maj7")],
    "blues":     [(0, "dom7"), (3, "dom7"), (4, "dom7"), (0, "dom7")],
    "rock":      [(0, "maj"), (3, "maj"), (4, "maj"), (0, "maj")],
    "cinematic": [(0, "min"), (4, "min"), (6, "maj"), (3, "maj")],
    "ambient":   [(0, "maj7"), (2, "min7"), (5, "min"), (3, "maj7")],
    "classical": [(0, "maj"), (3, "maj"), (4, "maj"), (4, "dom7"), (0, "maj")],
    "epic":      [(5, "min"), (3, "maj"), (0, "maj"), (4, "maj")],
    "sad":       [(0, "min"), (3, "min"), (5, "maj"), (4, "maj")],
    "dark":      [(0, "min"), (5, "dim"), (3, "min"), (4, "min")],
    "happy":     [(0, "maj"), (3, "maj"), (4, "maj"), (0, "maj")],
    "calm":      [(0, "maj7"), (5, "min7"), (3, "maj7"), (4, "maj7")],
    "energetic": [(0, "min"), (4, "maj"), (5, "min"), (2, "maj")],
}


def _midi_to_freq(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def _note_name_to_midi(name: str, octave: int = 4) -> int:
    idx = _NOTE_NAMES.index(name) if name in _NOTE_NAMES else 0
    return 12 * (octave + 1) + idx


class NRSComposer:
    """NRS reasoning-driven music composition engine."""

    def __init__(self, sr: int = MUSIC_SR):
        self._sr = sr
        self._rng = np.random.default_rng(42)

    def compose(self, prompt: str, duration: float = 30.0,
                style: str = "cinematic") -> MusicScore:
        mood = self._analyze_mood(prompt)
        detected_style = self._detect_style(prompt) or style
        tempo = self._select_tempo(detected_style, mood)
        key, scale = self._select_key(mood)

        score = MusicScore(
            tempo=tempo,
            key=key,
            scale=scale,
            duration=duration,
        )

        progression = self._generate_progression(detected_style, key, scale)
        score.chords = self._lay_chords(progression, tempo, duration)
        score.melody = self._generate_melody(
            progression, key, scale, tempo, duration)
        score.bass = self._generate_bass(progression, key, tempo, duration)
        score.drums = self._generate_drums(detected_style, tempo, duration)

        return score

    def render(self, score: MusicScore) -> np.ndarray:
        total_samples = int(self._sr * score.duration)
        audio = np.zeros(total_samples, dtype=np.float32)

        for midi, start_t, dur in score.bass:
            sig = _waveguide_string(
                _midi_to_freq(midi), dur, self._sr, 0.8, 0.3, 1.2)
            s = int(start_t * self._sr)
            e = min(s + len(sig), total_samples)
            audio[s:e] += sig[:e - s] * 0.25

        for midi, start_t, dur in score.chords:
            sig = _waveguide_string(
                _midi_to_freq(midi), dur, self._sr, 0.6, 0.5, 0.8)
            s = int(start_t * self._sr)
            e = min(s + len(sig), total_samples)
            audio[s:e] += sig[:e - s] * 0.12

        for midi, start_t, dur in score.melody:
            sig = _waveguide_string(
                _midi_to_freq(midi), dur, self._sr, 0.7, 0.6, 0.9)
            s = int(start_t * self._sr)
            e = min(s + len(sig), total_samples)
            audio[s:e] += sig[:e - s] * 0.18

        for drum_type, start_t, vel in score.drums:
            drum_synth = DrumSynthesizer(self._sr)
            if drum_type == "kick":
                sig = drum_synth.kick(vel)
            elif drum_type == "snare":
                sig = drum_synth.snare(vel)
            elif drum_type == "hihat":
                sig = drum_synth.hihat(vel)
            elif drum_type == "crash":
                sig = drum_synth.crash(vel)
            else:
                sig = drum_synth.tom(0.5, vel)
            s = int(start_t * self._sr)
            e = min(s + len(sig), total_samples)
            audio[s:e] += sig[:e - s] * 0.3

        return audio

    def _analyze_mood(self, prompt: str) -> str:
        lower = prompt.lower()
        moods = {
            "happy": ["happy", "joy", "bright", "fun", "upbeat", "cheerful"],
            "sad": ["sad", "melanchol", "sorrow", "lonely", "grief", "somber"],
            "dark": ["dark", "evil", "doom", "ominous", "horror", "creepy"],
            "energetic": ["energy", "fast", "drive", "pump", "action", "intense"],
            "calm": ["calm", "peace", "gentle", "soft", "relax", "ambient"],
            "epic": ["epic", "grand", "battle", "war", "heroic", "powerful"],
        }
        for mood, keywords in moods.items():
            if any(kw in lower for kw in keywords):
                return mood
        return "neutral"

    def _detect_style(self, prompt: str) -> Optional[str]:
        lower = prompt.lower()
        for style in _STYLE_PROGRESSIONS:
            if style in lower:
                return style
        return None

    def _select_tempo(self, style: str, mood: str) -> int:
        base = {
            "pop": 120, "jazz": 110, "blues": 85, "rock": 130,
            "cinematic": 100, "ambient": 70, "classical": 90,
            "epic": 110, "sad": 75, "dark": 85, "happy": 125,
            "calm": 68, "energetic": 140,
        }.get(style, 100)
        mood_adj = {
            "happy": 10, "sad": -15, "energetic": 20,
            "calm": -20, "dark": -5, "epic": 5,
        }.get(mood, 0)
        return max(60, min(180, base + mood_adj))

    def _select_key(self, mood: str) -> Tuple[str, str]:
        if mood in ("sad", "dark", "epic"):
            keys = ["A", "D", "E", "B"]
            return self._rng.choice(keys), "minor"
        return self._rng.choice(["C", "G", "D", "F", "A"]), "major"

    def _generate_progression(self, style: str, key: str,
                               scale: str) -> List[Tuple[int, str]]:
        prog_template = _STYLE_PROGRESSIONS.get(
            style, _STYLE_PROGRESSIONS["cinematic"])
        root_midi = _note_name_to_midi(key, 3)
        intervals = _SCALE_INTERVALS.get(scale, _SCALE_INTERVALS["major"])

        result = []
        for degree, chord_type in prog_template:
            scale_idx = degree % len(intervals)
            midi_root = root_midi + intervals[scale_idx]
            result.append((midi_root, chord_type))
        return result

    def _lay_chords(self, progression: List[Tuple[int, str]],
                     tempo: int, duration: float) -> List[Tuple[int, float, float]]:
        beat_dur = 60.0 / tempo
        bar_dur = beat_dur * 4
        chords_out = []
        t = 0.0
        prog_idx = 0
        while t < duration:
            root, ctype = progression[prog_idx % len(progression)]
            intervals = _CHORD_INTERVALS.get(ctype, [0, 4, 7])
            for iv in intervals:
                chords_out.append((root + iv, t, min(bar_dur, duration - t)))
            t += bar_dur
            prog_idx += 1
        return chords_out

    def _generate_melody(self, progression: List[Tuple[int, str]],
                          key: str, scale: str, tempo: int,
                          duration: float) -> List[Tuple[int, float, float]]:
        intervals = _SCALE_INTERVALS.get(scale, _SCALE_INTERVALS["major"])
        root = _note_name_to_midi(key, 5)
        scale_notes = [root + iv for iv in intervals]
        scale_notes += [root + 12 + iv for iv in intervals]

        beat_dur = 60.0 / tempo
        melody = []
        t = 0.0
        current_idx = len(scale_notes) // 2

        while t < duration:
            note_dur = beat_dur * self._rng.choice([0.5, 1.0, 1.0, 2.0])
            if self._rng.random() < 0.15:
                t += note_dur
                continue

            step = self._rng.choice([-2, -1, 0, 1, 1, 2])
            current_idx = max(0, min(len(scale_notes) - 1, current_idx + step))
            midi = scale_notes[current_idx]
            melody.append((midi, t, min(note_dur * 0.9, duration - t)))
            t += note_dur

        return melody

    def _generate_bass(self, progression: List[Tuple[int, str]],
                        key: str, tempo: int,
                        duration: float) -> List[Tuple[int, float, float]]:
        beat_dur = 60.0 / tempo
        bar_dur = beat_dur * 4
        bass = []
        t = 0.0
        prog_idx = 0
        while t < duration:
            root, _ = progression[prog_idx % len(progression)]
            bass_note = root - 12
            for beat in range(4):
                bt = t + beat * beat_dur
                if bt >= duration:
                    break
                if beat == 0 or beat == 2:
                    bass.append((bass_note, bt,
                                 min(beat_dur * 0.8, duration - bt)))
                elif self._rng.random() < 0.5:
                    bass.append((bass_note + 7, bt,
                                 min(beat_dur * 0.5, duration - bt)))
            t += bar_dur
            prog_idx += 1
        return bass

    def _generate_drums(self, style: str, tempo: int,
                         duration: float) -> List[Tuple[str, float, float]]:
        beat_dur = 60.0 / tempo
        drums = []
        t = 0.0
        beat = 0
        while t < duration:
            beat_in_bar = beat % 4

            if beat_in_bar == 0:
                drums.append(("kick", t, 0.9))
            if beat_in_bar == 2:
                drums.append(("snare", t, 0.8))
            if beat_in_bar == 0 and beat % 32 == 0 and beat > 0:
                drums.append(("crash", t, 0.7))

            if style in ("rock", "pop", "energetic"):
                drums.append(("hihat", t, 0.5))
            elif beat_in_bar % 2 == 0:
                drums.append(("hihat", t, 0.4))

            t += beat_dur
            beat += 1
        return drums


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PHYSICAL MODELING INSTRUMENTS
# ═══════════════════════════════════════════════════════════════════════════════

def _waveguide_string(frequency: float, duration: float, sr: int = MUSIC_SR,
                       velocity: float = 0.8, brightness: float = 0.5,
                       decay: float = 1.0) -> np.ndarray:
    """Extended Karplus-Strong waveguide string model."""
    if frequency <= 0:
        return np.zeros(int(sr * duration), dtype=np.float32)
    N = max(2, int(sr / frequency))
    num = int(sr * duration)
    rng = np.random.default_rng(int(frequency * 100) % (2**31))

    buf = (rng.standard_normal(N) * velocity * 0.5).astype(np.float64)

    frac = sr / frequency - N
    ap_coeff = (1.0 - frac) / (1.0 + frac)

    loss = 0.996 * decay
    lp_coeff = 0.5 + brightness * 0.4

    out = np.zeros(num, dtype=np.float64)
    prev_ap = 0.0
    idx = 0

    for i in range(num):
        out[i] = buf[idx]
        nxt = (idx + 1) % N
        avg = lp_coeff * buf[idx] + (1.0 - lp_coeff) * buf[nxt]
        ap_out = ap_coeff * avg + prev_ap - ap_coeff * avg
        prev_ap = avg
        buf[idx] = loss * (avg + ap_out) * 0.5
        idx = nxt

    att = min(int(sr * 0.003), num)
    if att > 0:
        out[:att] *= np.linspace(0, 1, att)

    return (out * velocity).astype(np.float32)


class WaveguideString:
    """Waveguide string model (extended Karplus-Strong)."""

    def __init__(self, sr: int = MUSIC_SR):
        self._sr = sr

    def synthesize(self, frequency: float, duration: float,
                   velocity: float = 0.8, brightness: float = 0.5,
                   decay: float = 1.0) -> np.ndarray:
        return _waveguide_string(
            frequency, duration, self._sr, velocity, brightness, decay)


class WaveguidePiano:
    """Piano model: coupled string pair, hammer excitation, soundboard."""

    def __init__(self, sr: int = MUSIC_SR):
        self._sr = sr

    def synthesize(self, midi_note: int, duration: float,
                   velocity: float = 0.8) -> np.ndarray:
        freq = _midi_to_freq(midi_note)
        detune = 0.5

        s1 = _waveguide_string(freq, duration, self._sr,
                                velocity, 0.6, 0.85)
        s2 = _waveguide_string(freq * (1.0 + detune / 1200.0), duration,
                                self._sr, velocity * 0.9, 0.55, 0.85)

        coupled = s1 * 0.55 + s2 * 0.45

        n = len(coupled)
        hammer_dur = int(self._sr * 0.002)
        if hammer_dur < n:
            t = np.arange(hammer_dur, dtype=np.float32) / self._sr
            hammer = np.sin(math.pi * t / (hammer_dur / self._sr))
            hammer *= velocity * 0.3
            coupled[:hammer_dur] += hammer * np.exp(-50 * t)

        body_freqs = [220, 440, 880]
        for bf in body_freqs:
            if bf < self._sr / 2:
                b, a = _resonator_coeffs(bf, 30.0, self._sr)
                resonance = lfilter(b, a, coupled * 0.05)
                coupled += resonance.astype(np.float32)

        return coupled


class WaveguideGuitar:
    """Guitar model: pluck excitation, body resonance, multiple styles."""

    def __init__(self, sr: int = MUSIC_SR):
        self._sr = sr

    def synthesize(self, midi_note: int, duration: float,
                   velocity: float = 0.8,
                   style: str = "fingerpick") -> np.ndarray:
        freq = _midi_to_freq(midi_note)

        bright = {"fingerpick": 0.5, "strum": 0.65, "palm_mute": 0.2}
        dec = {"fingerpick": 1.0, "strum": 0.9, "palm_mute": 0.4}

        sig = _waveguide_string(freq, duration, self._sr, velocity,
                                 bright.get(style, 0.5),
                                 dec.get(style, 1.0))

        body_resonances = [(100, 50), (200, 40), (400, 60)]
        for bf, bw in body_resonances:
            if bf < self._sr / 2:
                b, a = _resonator_coeffs(bf, bw, self._sr)
                sig += lfilter(b, a, sig * 0.08).astype(np.float32)

        return sig


class WaveguideBrass:
    """Brass model: lip reed, bore resonance, bell radiation."""

    def __init__(self, sr: int = MUSIC_SR):
        self._sr = sr

    def synthesize(self, midi_note: int, duration: float,
                   velocity: float = 0.8) -> np.ndarray:
        freq = _midi_to_freq(midi_note)
        n = int(self._sr * duration)
        t = np.arange(n, dtype=np.float32) / self._sr

        att_time = 0.08
        att_env = np.minimum(t / att_time, 1.0)
        rel_time = 0.05
        rel_start = max(duration - rel_time, 0)
        rel_env = np.where(t > rel_start,
                           1.0 - (t - rel_start) / rel_time, 1.0)
        env = att_env * np.clip(rel_env, 0, 1) * velocity

        sig = np.zeros(n, dtype=np.float32)
        for h in range(1, 9):
            amp = 1.0 / h
            if h % 2 == 0:
                amp *= 0.7
            sig += amp * np.sin(_TWO_PI * freq * h * t)

        sig *= env

        bore_freq = freq * 0.5
        if bore_freq < self._sr / 2:
            b, a = _resonator_coeffs(bore_freq, 40.0, self._sr)
            sig = lfilter(b, a, sig).astype(np.float32)

        bell_cutoff = min(freq * 4, self._sr / 2 - 100)
        nyq = self._sr / 2
        sos = butter(2, bell_cutoff / nyq, btype="high", output="sos")
        radiation = sosfilt(sos, sig).astype(np.float32)
        sig = sig * 0.7 + radiation * 0.3

        return sig


class WaveguideWoodwind:
    """Woodwind model: reed/jet excitation, tone holes, bell."""

    def __init__(self, sr: int = MUSIC_SR):
        self._sr = sr

    def synthesize(self, midi_note: int, duration: float,
                   velocity: float = 0.8,
                   instrument: str = "flute") -> np.ndarray:
        freq = _midi_to_freq(midi_note)
        n = int(self._sr * duration)
        t = np.arange(n, dtype=np.float32) / self._sr

        att = 0.06 if instrument == "flute" else 0.04
        att_env = np.minimum(t / att, 1.0)
        rel_start = max(duration - 0.05, 0)
        rel_env = np.where(t > rel_start,
                           1.0 - (t - rel_start) / 0.05, 1.0)
        env = att_env * np.clip(rel_env, 0, 1) * velocity

        sig = np.zeros(n, dtype=np.float32)
        rng = np.random.default_rng(int(freq * 10) % (2**31))

        if instrument == "flute":
            sig = np.sin(_TWO_PI * freq * t)
            sig += 0.3 * np.sin(_TWO_PI * freq * 2 * t)
            breath = rng.standard_normal(n).astype(np.float32) * 0.05
            nyq = self._sr / 2
            lo = max(freq * 0.8 / nyq, 0.001)
            hi = min(freq * 3 / nyq, 0.999)
            if hi > lo:
                b_bp, a_bp = butter(2, [lo, hi], btype="band")
                breath = lfilter(b_bp, a_bp, breath).astype(np.float32)
            sig += breath
        elif instrument == "clarinet":
            for h in range(1, 12, 2):
                sig += (1.0 / h) * np.sin(_TWO_PI * freq * h * t)
        elif instrument == "oboe":
            for h in range(1, 10):
                amp = 1.0 / (h ** 0.8)
                sig += amp * np.sin(_TWO_PI * freq * h * t)
        else:
            for h in range(1, 8):
                sig += (1.0 / h) * np.sin(_TWO_PI * freq * h * t)

        sig *= env

        if freq < self._sr / 2:
            b, a = _resonator_coeffs(freq, 20.0, self._sr)
            sig = lfilter(b, a, sig).astype(np.float32)

        return sig


class DrumSynthesizer:
    """Physical modeling drum synthesis."""

    def __init__(self, sr: int = MUSIC_SR):
        self._sr = sr

    def kick(self, velocity: float = 0.8) -> np.ndarray:
        dur = 0.35
        n = int(self._sr * dur)
        t = np.arange(n, dtype=np.float32) / self._sr
        freq_sweep = 150.0 * np.exp(-25.0 * t) + 45.0
        phase = np.cumsum(_TWO_PI * freq_sweep / self._sr)
        body = np.sin(phase) * np.exp(-6.0 * t)
        click_n = int(self._sr * 0.005)
        click = np.zeros(n, dtype=np.float32)
        rng = np.random.default_rng(1)
        click[:click_n] = rng.standard_normal(click_n).astype(np.float32) * 0.8
        click[:click_n] *= np.exp(-np.linspace(0, 8, click_n))
        return ((body * 0.9 + click * 0.4) * velocity).astype(np.float32)

    def snare(self, velocity: float = 0.8) -> np.ndarray:
        dur = 0.25
        n = int(self._sr * dur)
        t = np.arange(n, dtype=np.float32) / self._sr
        body = np.sin(_TWO_PI * 185 * t) * np.exp(-12.0 * t) * 0.6
        rng = np.random.default_rng(2)
        noise = rng.standard_normal(n).astype(np.float32)
        nyq = self._sr / 2
        lo = max(3000 / nyq, 0.001)
        hi = min(9000 / nyq, 0.999)
        b_bp, a_bp = butter(2, [lo, hi], btype="band")
        rattle = lfilter(b_bp, a_bp, noise).astype(np.float32)
        rattle *= np.exp(-8.0 * t) * 0.7
        return ((body + rattle) * velocity).astype(np.float32)

    def hihat(self, velocity: float = 0.8, open: bool = False) -> np.ndarray:
        dur = 0.15 if open else 0.06
        n = int(self._sr * dur)
        t = np.arange(n, dtype=np.float32) / self._sr
        rng = np.random.default_rng(3)
        noise = rng.standard_normal(n).astype(np.float32)
        nyq = self._sr / 2
        cutoff = min(7000 / nyq, 0.999)
        sos = butter(3, cutoff, btype="high", output="sos")
        filt = sosfilt(sos, noise).astype(np.float32)
        decay = 15.0 if open else 35.0
        return (filt * np.exp(-decay * t) * 0.5 * velocity).astype(np.float32)

    def crash(self, velocity: float = 0.8) -> np.ndarray:
        dur = 1.5
        n = int(self._sr * dur)
        t = np.arange(n, dtype=np.float32) / self._sr
        rng = np.random.default_rng(4)
        noise = rng.standard_normal(n).astype(np.float32)
        nyq = self._sr / 2
        cutoff = min(5000 / nyq, 0.999)
        sos = butter(2, cutoff, btype="high", output="sos")
        filt = sosfilt(sos, noise).astype(np.float32)
        return (filt * np.exp(-2.5 * t) * 0.6 * velocity).astype(np.float32)

    def tom(self, pitch: float = 0.5, velocity: float = 0.8) -> np.ndarray:
        dur = 0.4
        n = int(self._sr * dur)
        t = np.arange(n, dtype=np.float32) / self._sr
        freq = 80 + pitch * 120
        body = np.sin(_TWO_PI * freq * t) * np.exp(-8.0 * t)
        rng = np.random.default_rng(5)
        noise = rng.standard_normal(n).astype(np.float32) * 0.15
        noise *= np.exp(-15.0 * t)
        return ((body + noise) * velocity).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. AUDIO MASTER CHAIN
# ═══════════════════════════════════════════════════════════════════════════════

class AudioMaster:
    """Production audio mastering chain."""

    def __init__(self, sr: int = SPEECH_SR):
        self._sr = sr

    def process(self, audio: np.ndarray, sr: Optional[int] = None) -> np.ndarray:
        if sr is not None:
            self._sr = sr
        audio = self._highpass(audio, 30.0)
        audio = self._multiband_compress(audio)
        audio = self._parametric_eq(audio)
        left, right = self._stereo_widen(audio)
        left = self._reverb(left)
        right = self._reverb(right)
        left = self._limiter(left, -0.5)
        right = self._limiter(right, -0.5)
        left = self._dither(left)
        right = self._dither(right)
        return np.stack([left, right])

    def _highpass(self, sig: np.ndarray, cutoff: float) -> np.ndarray:
        nyq = self._sr / 2
        if cutoff >= nyq:
            return sig
        sos = butter(4, cutoff / nyq, btype="high", output="sos")
        return sosfilt(sos, sig).astype(np.float32)

    def _multiband_compress(self, sig: np.ndarray) -> np.ndarray:
        nyq = self._sr / 2
        bands = []

        low_cut = min(200 / nyq, 0.999)
        sos_low = butter(3, low_cut, btype="low", output="sos")
        low = sosfilt(sos_low, sig).astype(np.float32)
        bands.append(self._compress_band(low, -12.0, 4.0))

        mid_lo = max(200 / nyq, 0.001)
        mid_hi = min(4000 / nyq, 0.999)
        if mid_hi > mid_lo:
            sos_mid = butter(3, [mid_lo, mid_hi], btype="band", output="sos")
            mid = sosfilt(sos_mid, sig).astype(np.float32)
            bands.append(self._compress_band(mid, -14.0, 3.0))

        hi_cut = max(4000 / nyq, 0.001)
        if hi_cut < 0.999:
            sos_hi = butter(3, hi_cut, btype="high", output="sos")
            high = sosfilt(sos_hi, sig).astype(np.float32)
            bands.append(self._compress_band(high, -16.0, 2.5))

        result = np.zeros_like(sig)
        for b in bands:
            result += b
        return result

    def _compress_band(self, sig: np.ndarray, threshold_db: float,
                        ratio: float) -> np.ndarray:
        thresh = 10.0 ** (threshold_db / 20.0)
        win = max(int(self._sr * 0.010), 1)
        sq = sig.astype(np.float64) ** 2
        kernel = np.ones(win, dtype=np.float64) / win
        rms = np.sqrt(np.convolve(sq, kernel, mode="same") + 1e-10)

        gain = np.ones(len(sig), dtype=np.float64)
        loud = rms > thresh
        if loud.any():
            over_db = 20.0 * np.log10(rms[loud] / thresh)
            reduction_db = over_db * (1.0 - 1.0 / ratio)
            gain[loud] = 10.0 ** (-reduction_db / 20.0)

        smooth = max(int(self._sr * 0.005), 1)
        kern = np.hanning(smooth).astype(np.float64)
        kern /= kern.sum()
        gain = np.convolve(gain, kern, mode="same")

        return (sig * gain).astype(np.float32)

    def _parametric_eq(self, sig: np.ndarray) -> np.ndarray:
        nyq = self._sr / 2
        eq_bands = [
            (80, 1.5),
            (250, -1.0),
            (2500, 2.0),
            (8000, 1.0),
        ]
        for freq, gain_db in eq_bands:
            if freq >= nyq:
                continue
            gain = 10.0 ** (gain_db / 20.0)
            bw = freq * 0.5
            lo = max((freq - bw / 2) / nyq, 0.001)
            hi = min((freq + bw / 2) / nyq, 0.999)
            if hi <= lo:
                continue
            sos = butter(2, [lo, hi], btype="band", output="sos")
            band = sosfilt(sos, sig).astype(np.float32)
            sig = sig + band * (gain - 1.0)
        return sig.astype(np.float32)

    def _stereo_widen(self, mono: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        delay_samples = int(self._sr * 0.4 / 1000.0)
        right = np.roll(mono, delay_samples)
        right[:delay_samples] = 0.0
        rng = np.random.default_rng(99)
        decorr = rng.standard_normal(len(mono)).astype(np.float32) * 0.005
        right = right + decorr
        return mono.copy(), right.astype(np.float32)

    def _reverb(self, sig: np.ndarray) -> np.ndarray:
        wet = 0.15
        n = len(sig)

        er_delays = [int(self._sr * d) for d in [0.011, 0.017, 0.023, 0.029]]
        er = np.zeros(n, dtype=np.float32)
        for i, d in enumerate(er_delays):
            if d < n:
                amp = 0.3 * (0.7 ** i)
                er[d:] += sig[:n - d] * amp

        comb_delays = [int(self._sr * d)
                       for d in [0.0297, 0.0371, 0.0411, 0.0437]]
        comb_fb = 0.75
        late = np.zeros(n, dtype=np.float32)
        for delay in comb_delays:
            if delay < 1 or delay >= n:
                continue
            buf = np.zeros(n, dtype=np.float32)
            for i in range(delay, n):
                buf[i] = sig[i - delay] + comb_fb * buf[i - delay]
            late += buf * 0.25

        ap_delays = [int(self._sr * d) for d in [0.005, 0.0017]]
        for delay in ap_delays:
            if delay < 1 or delay >= n:
                continue
            ap_g = 0.5
            buf = np.zeros(n, dtype=np.float32)
            for i in range(delay, n):
                buf[i] = -ap_g * late[i] + late[i - delay] + ap_g * buf[i - delay]
            late = buf

        return (sig * (1.0 - wet) + (er + late) * wet).astype(np.float32)

    def _limiter(self, sig: np.ndarray, ceiling_db: float) -> np.ndarray:
        ceiling = 10.0 ** (ceiling_db / 20.0)
        peak = np.abs(sig).max()
        if peak > ceiling:
            sig = sig * (ceiling / peak)
        return np.clip(sig, -ceiling, ceiling).astype(np.float32)

    def _dither(self, sig: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(77)
        dither_level = 1.0 / (2 ** 23)
        tpdf = (rng.random(len(sig)).astype(np.float32)
                + rng.random(len(sig)).astype(np.float32) - 1.0)
        return (sig + tpdf * dither_level).astype(np.float32)

    def encode_wav_24bit(self, stereo: np.ndarray, sr: int) -> bytes:
        left = stereo[0] if stereo.ndim == 2 else stereo
        right = stereo[1] if stereo.ndim == 2 else stereo
        n = len(left)
        interleaved = np.empty(n * 2, dtype=np.float32)
        interleaved[0::2] = left
        interleaved[1::2] = right
        scaled = np.clip(interleaved * 8388607.0, -8388608, 8388607).astype(np.int32)
        raw32 = scaled.astype("<i4").tobytes()
        pcm = np.frombuffer(raw32, dtype=np.uint8).reshape(-1, 4)[:, :3].tobytes()
        channels = 2
        bps = 24
        byte_rate = sr * channels * 3
        block_align = channels * 3
        data_size = len(pcm)
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", 36 + data_size, b"WAVE",
            b"fmt ", 16, 1, channels, sr, byte_rate, block_align, bps,
            b"data", data_size,
        )
        return header + pcm

    def encode_mp3(self, stereo: np.ndarray, sr: int,
                    bitrate: str = "320k") -> bytes:
        wav_bytes = self.encode_wav_24bit(stereo, sr)
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = os.path.join(tmpdir, "in.wav")
            out_path = os.path.join(tmpdir, "out.mp3")
            with open(in_path, "wb") as f:
                f.write(wav_bytes)
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", in_path, "-c:a", "libmp3lame",
                     "-b:a", bitrate, out_path],
                    capture_output=True, timeout=60)
                if os.path.isfile(out_path):
                    with open(out_path, "rb") as f:
                        return f.read()
            except Exception:
                pass
        return wav_bytes


# ═══════════════════════════════════════════════════════════════════════════════
# 6. SPEECH RECOGNITION (STT)
# ═══════════════════════════════════════════════════════════════════════════════

def _hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(n_fft: int, sr: int, n_mels: int = 80,
                     fmin: float = 50.0,
                     fmax: Optional[float] = None) -> np.ndarray:
    if fmax is None:
        fmax = sr / 2.0
    mel_min = _hz_to_mel(fmin)
    mel_max = _hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = np.array([_mel_to_hz(m) for m in mel_points])
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]
        for j in range(left, center):
            if center > left:
                fb[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right > center:
                fb[i, j] = (right - j) / (right - center)
    return fb


def _mel_spectrogram(audio: np.ndarray, sr: int,
                      n_fft: int = 512, hop: int = 160,
                      win: int = 400, n_mels: int = 80) -> np.ndarray:
    window = np.hanning(win).astype(np.float32)
    fb = _mel_filterbank(n_fft, sr, n_mels)

    n_frames = 1 + (len(audio) - win) // hop
    if n_frames <= 0:
        return np.zeros((n_mels, 1), dtype=np.float32)

    mel_spec = np.zeros((n_mels, n_frames), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop
        frame = audio[start:start + win] * window
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        spectrum = np.abs(np.fft.rfft(frame, n=n_fft)) ** 2
        mel_spec[:, i] = fb @ spectrum

    mel_spec = np.maximum(mel_spec, 1e-10)
    return 10.0 * np.log10(mel_spec)


def _frame_energy(audio: np.ndarray, sr: int,
                   hop: int = 160, win: int = 400) -> np.ndarray:
    n_frames = 1 + (len(audio) - win) // hop
    if n_frames <= 0:
        return np.zeros(1, dtype=np.float32)
    energy = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        start = i * hop
        frame = audio[start:start + win]
        energy[i] = np.sum(frame ** 2) / len(frame)
    return energy


def _zero_crossing_rate(audio: np.ndarray, sr: int,
                         hop: int = 160, win: int = 400) -> np.ndarray:
    n_frames = 1 + (len(audio) - win) // hop
    if n_frames <= 0:
        return np.zeros(1, dtype=np.float32)
    zcr = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        start = i * hop
        frame = audio[start:start + win]
        zcr[i] = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))
    return zcr


def _autocorrelation_pitch(frame: np.ndarray, sr: int,
                            fmin: float = 60.0,
                            fmax: float = 500.0) -> float:
    min_lag = max(int(sr / fmax), 1)
    max_lag = min(int(sr / fmin), len(frame) - 1)
    if max_lag <= min_lag:
        return 0.0

    frame = frame - np.mean(frame)
    norm = np.sum(frame ** 2)
    if norm < 1e-10:
        return 0.0

    best_lag = 0
    best_corr = 0.0
    for lag in range(min_lag, max_lag):
        corr = np.sum(frame[:len(frame) - lag] * frame[lag:]) / norm
        if corr > best_corr:
            best_corr = corr
            best_lag = lag

    if best_corr < 0.3 or best_lag == 0:
        return 0.0
    return sr / best_lag


def _lpc_formants(frame: np.ndarray, sr: int,
                   order: int = 14) -> List[float]:
    """Extract formant frequencies via LPC analysis."""
    windowed = frame * np.hanning(len(frame))
    windowed = windowed / (np.abs(windowed).max() + 1e-10)

    r = np.correlate(windowed, windowed, mode="full")
    r = r[len(windowed) - 1:]
    r = r[:order + 1]

    if r[0] < 1e-10:
        return []

    a = np.zeros(order + 1, dtype=np.float64)
    a[0] = 1.0
    E = r[0]

    for i in range(1, order + 1):
        lam = 0.0
        for j in range(1, i):
            lam += a[j] * r[i - j]
        lam = (r[i] - lam) / max(E, 1e-10)

        new_a = np.zeros(order + 1, dtype=np.float64)
        new_a[0] = 1.0
        for j in range(1, i):
            new_a[j] = a[j] - lam * a[i - j]
        new_a[i] = lam

        a = new_a
        E = E * (1.0 - lam * lam)
        if E < 1e-10:
            break

    roots = np.roots(a)
    formants = []
    for root in roots:
        if np.imag(root) > 0:
            freq = np.abs(np.arctan2(np.imag(root), np.real(root))) * sr / _TWO_PI
            bw = -0.5 * sr / math.pi * np.log(np.abs(root) + 1e-10)
            if 90 < freq < sr / 2 and bw < 500:
                formants.append(freq)

    formants.sort()
    return formants[:5]


_PHONEME_FORMANT_REFS: Dict[str, Tuple[float, float, float]] = {
    "AA": (730, 1090, 2440), "AE": (660, 1720, 2410),
    "AH": (520, 1190, 2390), "AO": (570, 840, 2410),
    "EH": (530, 1840, 2480), "ER": (490, 1350, 1690),
    "EY": (400, 2100, 2660), "IH": (390, 1990, 2550),
    "IY": (270, 2290, 3010), "OW": (570, 840, 2410),
    "UH": (440, 1020, 2240), "UW": (300, 870, 2240),
}


class NRSSpeechRecognizer:
    """NRS-native speech recognition using spectral analysis and formant tracking."""

    def __init__(self, sr: int = 16000):
        self._sr = sr
        self._hop = int(sr * 0.010)
        self._win = int(sr * 0.025)
        self._n_fft = 512
        self._g2p = NRSPhonemeEngine()

    def transcribe(self, audio: np.ndarray,
                    sr: Optional[int] = None) -> TranscriptionResult:
        if sr is not None:
            self._sr = sr
            self._hop = int(sr * 0.010)
            self._win = int(sr * 0.025)

        reasoning = []

        audio = audio.astype(np.float32)
        peak = np.abs(audio).max()
        if peak > 1e-8:
            audio = audio / peak
        reasoning.append(f"Normalized audio: {len(audio)} samples at {self._sr} Hz")

        mel = _mel_spectrogram(audio, self._sr, self._n_fft,
                                self._hop, self._win)
        reasoning.append(f"Mel spectrogram: {mel.shape[1]} frames x {mel.shape[0]} bands")

        energy = _frame_energy(audio, self._sr, self._hop, self._win)
        zcr = _zero_crossing_rate(audio, self._sr, self._hop, self._win)
        reasoning.append("Computed frame energy and zero-crossing rate")

        n_frames = min(len(energy), len(zcr), mel.shape[1])
        frame_labels = []
        for i in range(n_frames):
            if energy[i] < 1e-5:
                frame_labels.append("silence")
            elif self._is_voiced(audio, i):
                frame_labels.append("voiced")
            else:
                frame_labels.append("unvoiced")
        reasoning.append(f"Frame classification: {sum(1 for f in frame_labels if f == 'voiced')} voiced, "
                         f"{sum(1 for f in frame_labels if f == 'unvoiced')} unvoiced, "
                         f"{sum(1 for f in frame_labels if f == 'silence')} silence")

        phoneme_seq = self._formant_matching(audio, frame_labels)
        reasoning.append(f"Formant tracking identified {len(phoneme_seq)} phoneme segments")

        words = self._phonemes_to_words(phoneme_seq, n_frames)
        text = " ".join(w.text for w in words)
        reasoning.append(f"Decoded text: '{text}'")

        avg_conf = np.mean([w.confidence for w in words]) if words else 0.0
        if avg_conf > 0.7:
            trust = "VALIDATED"
        elif avg_conf > 0.4:
            trust = "RAW"
        else:
            trust = "RAW"

        return TranscriptionResult(
            text=text,
            words=words,
            trust_level=trust,
            language="en",
            reasoning_chain=reasoning,
        )

    def transcribe_streaming(self, chunk: np.ndarray,
                              sr: Optional[int] = None) -> TranscriptionResult:
        return self.transcribe(chunk, sr)

    def _is_voiced(self, audio: np.ndarray, frame_idx: int) -> bool:
        start = frame_idx * self._hop
        end = min(start + self._win, len(audio))
        if end - start < 64:
            return False
        frame = audio[start:end]
        pitch = _autocorrelation_pitch(frame, self._sr)
        return pitch > 0

    def _formant_matching(self, audio: np.ndarray,
                           frame_labels: List[str]) -> List[Tuple[str, int, int]]:
        segments = []
        current_ph = "SIL"
        seg_start = 0

        for i, label in enumerate(frame_labels):
            start = i * self._hop
            end = min(start + self._win, len(audio))
            if end - start < 64:
                continue

            if label == "silence":
                if current_ph != "SIL":
                    segments.append((current_ph, seg_start, i))
                    current_ph = "SIL"
                    seg_start = i
                continue

            frame = audio[start:end]
            formants = _lpc_formants(frame, self._sr)

            if len(formants) >= 2 and label == "voiced":
                best_ph = "AH"
                best_dist = float("inf")
                for ph, (rf1, rf2, rf3) in _PHONEME_FORMANT_REFS.items():
                    d = abs(formants[0] - rf1) + abs(formants[1] - rf2)
                    if len(formants) >= 3:
                        d += abs(formants[2] - rf3) * 0.5
                    if d < best_dist:
                        best_dist = d
                        best_ph = ph

                if best_ph != current_ph:
                    if current_ph != "SIL":
                        segments.append((current_ph, seg_start, i))
                    current_ph = best_ph
                    seg_start = i
            elif label == "unvoiced":
                if current_ph not in ("S", "SH", "F", "TH"):
                    if current_ph != "SIL":
                        segments.append((current_ph, seg_start, i))
                    current_ph = "S"
                    seg_start = i

        if current_ph != "SIL":
            segments.append((current_ph, seg_start, len(frame_labels)))

        return segments

    def _phonemes_to_words(self, phoneme_seq: List[Tuple[str, int, int]],
                            n_frames: int) -> List[WordResult]:
        if not phoneme_seq:
            return []

        words = []
        ph_group: List[Tuple[str, int, int]] = []

        for ph, start_f, end_f in phoneme_seq:
            if ph == "SIL" and len(ph_group) > 0:
                word = self._match_word(ph_group)
                if word:
                    start_t = ph_group[0][1] * self._hop / self._sr
                    end_t = ph_group[-1][2] * self._hop / self._sr
                    words.append(WordResult(
                        text=word,
                        start_time=start_t,
                        end_time=end_t,
                        confidence=0.5,
                    ))
                ph_group = []
            else:
                ph_group.append((ph, start_f, end_f))

        if ph_group:
            word = self._match_word(ph_group)
            if word:
                start_t = ph_group[0][1] * self._hop / self._sr
                end_t = ph_group[-1][2] * self._hop / self._sr
                words.append(WordResult(
                    text=word,
                    start_time=start_t,
                    end_time=end_t,
                    confidence=0.4,
                ))

        return words

    def _match_word(self, ph_group: List[Tuple[str, int, int]]) -> Optional[str]:
        if not ph_group:
            return None
        detected = [p[0] for p in ph_group]

        best_word = None
        best_score = float("inf")

        for word, phones_str in self._g2p._dict.items():
            ref_phones = [re.sub(r"[012]", "", p) for p in phones_str]
            score = self._edit_distance(detected, ref_phones)
            normalized = score / max(len(detected), len(ref_phones), 1)
            if normalized < best_score and normalized < 0.6:
                best_score = normalized
                best_word = word

        return best_word

    def _edit_distance(self, a: List[str], b: List[str]) -> int:
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1,
                               dp[i][j - 1] + 1,
                               dp[i - 1][j - 1] + cost)
        return dp[m][n]


# ═══════════════════════════════════════════════════════════════════════════════
# HIGH-LEVEL API
# ═══════════════════════════════════════════════════════════════════════════════

def synthesize_speech(text: str, voice: Optional[VoiceProfile] = None,
                       emotion: str = "neutral",
                       speed: float = 1.0) -> bytes:
    """Text -> mastered 24-bit stereo WAV bytes."""
    g2p = NRSPhonemeEngine()
    phonemes = g2p.text_to_phonemes(text)
    vocoder = NRSVocoder(SPEECH_SR)
    raw = vocoder.synthesize(phonemes, voice, emotion, speed)
    master = AudioMaster(SPEECH_SR)
    stereo = master.process(raw, SPEECH_SR)
    return master.encode_wav_24bit(stereo, SPEECH_SR)


def synthesize_music(prompt: str, duration: float = 30.0,
                      style: str = "cinematic") -> bytes:
    """Prompt -> mastered 24-bit stereo WAV bytes (music)."""
    composer = NRSComposer(MUSIC_SR)
    score = composer.compose(prompt, duration, style)
    raw = composer.render(score)
    master = AudioMaster(MUSIC_SR)
    stereo = master.process(raw, MUSIC_SR)
    return master.encode_wav_24bit(stereo, MUSIC_SR)


def transcribe_audio(audio: np.ndarray, sr: int = 16000) -> TranscriptionResult:
    """Audio -> TranscriptionResult with words, timestamps, confidence."""
    recognizer = NRSSpeechRecognizer(sr)
    return recognizer.transcribe(audio, sr)


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    "NRSPhonemeEngine",
    "PhonemeToken",
    "VoiceProfile",
    "NRSVocoder",
    "NRSComposer",
    "MusicScore",
    "WaveguideString",
    "WaveguidePiano",
    "WaveguideGuitar",
    "WaveguideBrass",
    "WaveguideWoodwind",
    "DrumSynthesizer",
    "AudioMaster",
    "NRSSpeechRecognizer",
    "TranscriptionResult",
    "WordResult",
    "synthesize_speech",
    "synthesize_music",
    "transcribe_audio",
    "SPEECH_SR",
    "MUSIC_SR",
]
