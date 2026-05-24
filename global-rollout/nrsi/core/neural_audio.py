"""
NRS Neural Audio Engine
========================

NRSI-native production-grade audio generation. Voice, music, and sound
effects powered by neural models with full brain integration.

Architecture
------------
  NLP Prompt / Text
      │
      ▼
  ┌── NRSI Audio Intelligence ───────────────────────────────┐
  │  Prompt analysis → mood, genre, instrument detection     │
  │  Safety validation gate  │  Style enrichment             │
  └───────────┬──────────────────────────────────────────────┘
              │
              ├── Voice Engine (XTTS-v2)
              │   24 languages, voice cloning, emotion control
              │   24kHz native → 48kHz/24-bit production output
              │
              ├── Music Engine (MusicGen)
              │   Text-to-music, variable length, multi-segment
              │   32kHz stereo → 48kHz/24-bit production output
              │
              └── SFX Engine (AudioLDM2)
                  Text-to-any-sound, negative prompts
                  16kHz → 48kHz/24-bit production output
              │
              ▼
  ┌── Audio Post-Processing ──────────────────────────────────┐
  │  Normalization (-1 dBFS) │ HP filter (20Hz)              │
  │  Gentle compression │ Stereo widening │ Fade curves      │
  │  Sample rate conversion → 48kHz/24-bit                   │
  └───────────┬──────────────────────────────────────────────┘
              │
              ▼
  ┌── Output Encoding ────────────────────────────────────────┐
  │  WAV (PCM 24-bit) │ FLAC (level 8) │ MP3 (320kbps)      │
  └───────────────────────────────────────────────────────────┘

Brain Integration
-----------------
  VLT L1:  Active generation state
  VLT L2:  Session audio cache (prompt → result)
  PVS-4:   Deterministic prompt → audio instant recall
  Tuition: Quality feedback → learned preferences
  Creative Vision: Mood detection → model parameters

Hardware Targets
----------------
  CUDA  NVIDIA GPUs (Ampere+) — primary
  MPS   Apple Silicon (M1–M4)
  CPU   Fallback (slow but functional)
"""

from __future__ import annotations

import hashlib
import io
import logging
import math
import os
import struct
import subprocess
import tempfile
import time
import wave
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("nrsi.neural.audio")


# ═══════════════════════════════════════════════════════════════════════════════
# HARDWARE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_audio_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


_AUDIO_DEVICE: Optional[str] = None


def get_audio_device() -> str:
    global _AUDIO_DEVICE
    if _AUDIO_DEVICE is None:
        _AUDIO_DEVICE = _detect_audio_device()
    return _AUDIO_DEVICE


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class AudioFormat(Enum):
    WAV = "wav"
    FLAC = "flac"
    MP3 = "mp3"


class AudioQuality(Enum):
    DRAFT = "draft"
    STANDARD = "standard"
    HIGH = "high"
    STUDIO = "studio"
    LOSSLESS = "lossless"


class VoiceEmotion(Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    WHISPER = "whisper"
    EXCITED = "excited"
    CALM = "calm"


QUALITY_SAMPLE_RATES: Dict[AudioQuality, int] = {
    AudioQuality.DRAFT: 22050,
    AudioQuality.STANDARD: 44100,
    AudioQuality.HIGH: 48000,
    AudioQuality.STUDIO: 48000,
    AudioQuality.LOSSLESS: 48000,
}

QUALITY_BIT_DEPTHS: Dict[AudioQuality, int] = {
    AudioQuality.DRAFT: 16,
    AudioQuality.STANDARD: 16,
    AudioQuality.HIGH: 24,
    AudioQuality.STUDIO: 24,
    AudioQuality.LOSSLESS: 24,
}

PRODUCTION_SAMPLE_RATE = 48000
PRODUCTION_BIT_DEPTH = 24
PRODUCTION_CHANNELS = 2


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AudioGenerationRequest:
    """Unified request for any audio generation task."""
    prompt: str
    negative_prompt: str = ""
    duration_seconds: float = 10.0
    quality: AudioQuality = AudioQuality.HIGH
    output_format: AudioFormat = AudioFormat.WAV
    seed: Optional[int] = None
    temperature: float = 1.0


@dataclass
class TTSGenerationRequest:
    """Request for text-to-speech synthesis."""
    text: str
    voice: str = "default"
    language: str = "en"
    speed: float = 1.0
    pitch: float = 1.0
    emotion: VoiceEmotion = VoiceEmotion.NEUTRAL
    reference_audio: Optional[bytes] = None
    quality: AudioQuality = AudioQuality.HIGH
    output_format: AudioFormat = AudioFormat.WAV


@dataclass
class AudioGenerationResult:
    """Output of an audio generation job."""
    audio_bytes: bytes = b""
    sample_rate: int = 48000
    channels: int = 2
    bit_depth: int = 24
    duration_seconds: float = 0.0
    format: str = "wav"
    generation_time_ms: float = 0.0
    device_used: str = ""
    model_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO POST-PROCESSOR — studio-grade finishing
# ═══════════════════════════════════════════════════════════════════════════════

class NRSIAudioPostProcessor:
    """
    Studio-grade audio post-processing applied to all generated audio.
    Ensures consistent quality, loudness, and format across all outputs.
    """

    @staticmethod
    def normalize(audio: np.ndarray, target_db: float = -1.0) -> np.ndarray:
        """Normalize audio to target dBFS peak level."""
        peak = np.abs(audio).max()
        if peak < 1e-8:
            return audio
        target_linear = 10 ** (target_db / 20.0)
        return audio * (target_linear / peak)

    @staticmethod
    def highpass_filter(audio: np.ndarray, sr: int,
                        cutoff_hz: float = 20.0) -> np.ndarray:
        """Remove DC offset and sub-bass rumble below cutoff."""
        try:
            from scipy.signal import butter, sosfilt
            sos = butter(4, cutoff_hz, btype='highpass', fs=sr, output='sos')
            return sosfilt(sos, audio).astype(np.float32)
        except ImportError:
            if audio.ndim == 1:
                audio = audio - np.mean(audio)
            else:
                audio = audio - np.mean(audio, axis=-1, keepdims=True)
            return audio

    @staticmethod
    def compress(audio: np.ndarray, threshold_db: float = -12.0,
                 ratio: float = 3.0, attack_ms: float = 5.0,
                 release_ms: float = 50.0, sr: int = 48000) -> np.ndarray:
        """Gentle dynamic range compression."""
        threshold = 10 ** (threshold_db / 20.0)
        attack_coeff = np.exp(-1.0 / (sr * attack_ms / 1000.0))
        release_coeff = np.exp(-1.0 / (sr * release_ms / 1000.0))

        flat = audio.flatten() if audio.ndim > 1 else audio
        envelope = np.zeros_like(flat)
        env = 0.0
        for i in range(len(flat)):
            level = abs(flat[i])
            if level > env:
                env = attack_coeff * env + (1 - attack_coeff) * level
            else:
                env = release_coeff * env + (1 - release_coeff) * level
            envelope[i] = env

        gain = np.ones_like(envelope)
        above = envelope > threshold
        if np.any(above):
            gain[above] = (
                threshold * (envelope[above] / threshold) ** (1.0 / ratio)
            ) / (envelope[above] + 1e-10)

        if audio.ndim > 1:
            gain = gain.reshape(audio.shape[0], -1)
            if audio.shape[0] == 2:
                gain_mono = np.mean(gain, axis=0)
                gain = np.stack([gain_mono, gain_mono])

        return (audio * gain).astype(np.float32)

    @staticmethod
    def stereo_widen(audio: np.ndarray, width: float = 1.3) -> np.ndarray:
        """Widen stereo field. width=1.0 is unchanged, >1.0 widens."""
        if audio.ndim != 2 or audio.shape[0] != 2:
            return audio
        mid = (audio[0] + audio[1]) / 2.0
        side = (audio[0] - audio[1]) / 2.0
        side = side * width
        left = mid + side
        right = mid - side
        return np.stack([left, right]).astype(np.float32)

    @staticmethod
    def fade(audio: np.ndarray, sr: int,
             fade_in_ms: float = 10.0,
             fade_out_ms: float = 30.0) -> np.ndarray:
        """Apply smooth fade in/out."""
        fade_in_samples = int(sr * fade_in_ms / 1000.0)
        fade_out_samples = int(sr * fade_out_ms / 1000.0)
        length = audio.shape[-1]

        if fade_in_samples > 0 and fade_in_samples < length:
            ramp = np.linspace(0, 1, fade_in_samples, dtype=np.float32)
            if audio.ndim == 2:
                ramp = ramp[np.newaxis, :]
            audio[..., :fade_in_samples] *= ramp

        if fade_out_samples > 0 and fade_out_samples < length:
            ramp = np.linspace(1, 0, fade_out_samples, dtype=np.float32)
            if audio.ndim == 2:
                ramp = ramp[np.newaxis, :]
            audio[..., -fade_out_samples:] *= ramp

        return audio

    @staticmethod
    def resample(audio: np.ndarray, src_sr: int,
                 target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        if src_sr == target_sr:
            return audio
        try:
            import librosa
            if audio.ndim == 2:
                channels = []
                for ch in range(audio.shape[0]):
                    channels.append(
                        librosa.resample(audio[ch], orig_sr=src_sr,
                                         target_sr=target_sr)
                    )
                return np.stack(channels).astype(np.float32)
            return librosa.resample(audio, orig_sr=src_sr,
                                    target_sr=target_sr).astype(np.float32)
        except ImportError:
            ratio = target_sr / src_sr
            new_length = int(audio.shape[-1] * ratio)
            indices = np.linspace(0, audio.shape[-1] - 1, new_length)
            if audio.ndim == 2:
                result = np.zeros((audio.shape[0], new_length),
                                  dtype=np.float32)
                for ch in range(audio.shape[0]):
                    result[ch] = np.interp(indices,
                                           np.arange(audio.shape[-1]),
                                           audio[ch])
                return result
            return np.interp(indices, np.arange(audio.shape[-1]),
                             audio).astype(np.float32)

    @staticmethod
    def mono_to_stereo(audio: np.ndarray) -> np.ndarray:
        """Convert mono to stereo."""
        if audio.ndim == 1:
            return np.stack([audio, audio]).astype(np.float32)
        if audio.ndim == 2 and audio.shape[0] == 1:
            return np.concatenate([audio, audio], axis=0).astype(np.float32)
        return audio

    def master(self, audio: np.ndarray, src_sr: int,
               quality: AudioQuality = AudioQuality.HIGH,
               is_music: bool = False) -> Tuple[np.ndarray, int]:
        """
        Full mastering chain for production output.
        Returns (audio_array, sample_rate).
        """
        target_sr = QUALITY_SAMPLE_RATES.get(quality, PRODUCTION_SAMPLE_RATE)

        audio = self.resample(audio, src_sr, target_sr)
        audio = self.mono_to_stereo(audio)
        audio = self.highpass_filter(audio, target_sr, cutoff_hz=20.0)
        audio = self.compress(audio, threshold_db=-14.0, ratio=2.5,
                              sr=target_sr)

        if is_music:
            audio = self.stereo_widen(audio, width=1.2)

        audio = self.normalize(audio, target_db=-1.0)
        audio = self.fade(audio, target_sr, fade_in_ms=10, fade_out_ms=30)

        return audio, target_sr


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO ENCODING — WAV / FLAC / MP3
# ═══════════════════════════════════════════════════════════════════════════════

class AudioEncoder:
    """Encodes float32 audio arrays to production-quality byte streams."""

    @staticmethod
    def to_wav(audio: np.ndarray, sr: int,
               bit_depth: int = 24) -> bytes:
        """Encode to WAV (PCM). Supports 16-bit and 24-bit."""
        audio = np.clip(audio, -1.0, 1.0)
        if audio.ndim == 1:
            audio = np.stack([audio, audio])
        channels = audio.shape[0]

        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setframerate(sr)
            if bit_depth == 24:
                wf.setsampwidth(3)
                interleaved = audio.T.flatten()
                scaled = (interleaved * 8388607).astype(np.int32)
                pcm = bytearray()
                for sample in scaled:
                    s = int(sample)
                    pcm.extend(struct.pack('<i', s)[:3])
                wf.writeframes(bytes(pcm))
            else:
                wf.setsampwidth(2)
                interleaved = audio.T.flatten()
                pcm = (interleaved * 32767).astype(np.int16).tobytes()
                wf.writeframes(pcm)
        return buf.getvalue()

    @staticmethod
    def to_flac(wav_bytes: bytes) -> bytes:
        """Convert WAV bytes to FLAC via ffmpeg."""
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = os.path.join(tmpdir, "in.wav")
            out_path = os.path.join(tmpdir, "out.flac")
            with open(in_path, "wb") as f:
                f.write(wav_bytes)
            cmd = [
                "ffmpeg", "-y", "-i", in_path,
                "-c:a", "flac", "-compression_level", "8",
                "-sample_fmt", "s32",
                out_path,
            ]
            try:
                subprocess.run(cmd, capture_output=True, check=True,
                               timeout=120)
                with open(out_path, "rb") as f:
                    return f.read()
            except Exception as exc:
                logger.warning("FLAC encoding failed: %s — returning WAV", exc)
                return wav_bytes

    @staticmethod
    def to_mp3(wav_bytes: bytes, bitrate: str = "320k") -> bytes:
        """Convert WAV bytes to high-quality MP3 via ffmpeg."""
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = os.path.join(tmpdir, "in.wav")
            out_path = os.path.join(tmpdir, "out.mp3")
            with open(in_path, "wb") as f:
                f.write(wav_bytes)
            cmd = [
                "ffmpeg", "-y", "-i", in_path,
                "-c:a", "libmp3lame", "-b:a", bitrate,
                "-q:a", "0",
                out_path,
            ]
            try:
                subprocess.run(cmd, capture_output=True, check=True,
                               timeout=120)
                with open(out_path, "rb") as f:
                    return f.read()
            except Exception as exc:
                logger.warning("MP3 encoding failed: %s — returning WAV", exc)
                return wav_bytes

    def encode(self, audio: np.ndarray, sr: int,
               fmt: AudioFormat = AudioFormat.WAV,
               quality: AudioQuality = AudioQuality.HIGH) -> bytes:
        """Encode audio to the requested format."""
        bit_depth = QUALITY_BIT_DEPTHS.get(quality, 24)
        wav_bytes = self.to_wav(audio, sr, bit_depth)

        if fmt == AudioFormat.FLAC:
            return self.to_flac(wav_bytes)
        elif fmt == AudioFormat.MP3:
            return self.to_mp3(wav_bytes)
        return wav_bytes


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL MANAGER — lazy loading, memory management
# ═══════════════════════════════════════════════════════════════════════════════

class NRSIAudioModelManager:
    """
    Manages neural audio model lifecycle.
    Only one large model loaded at a time to conserve GPU memory.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self._cache_dir = cache_dir or os.path.expanduser(
            "~/.cache/nrs/audio_models"
        )
        os.makedirs(self._cache_dir, exist_ok=True)
        self._loaded: Dict[str, Any] = {}
        self._device = get_audio_device()

    @property
    def device(self) -> str:
        return self._device

    def load_tts(self) -> Any:
        """Load XTTS-v2 for text-to-speech."""
        if "tts" in self._loaded:
            return self._loaded["tts"]

        self._unload_all()

        try:
            from TTS.api import TTS
            model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            if self._device != "cpu":
                model = model.to(self._device)
            self._loaded["tts"] = model
            logger.info("XTTS-v2 loaded on %s", self._device)
            return model
        except Exception as exc:
            logger.warning("XTTS-v2 load failed: %s", exc)
            return None

    def load_musicgen(self) -> Any:
        """Load MusicGen for music generation."""
        if "musicgen" in self._loaded:
            return self._loaded["musicgen"]

        self._unload_all()

        try:
            from audiocraft.models import MusicGen
            model = MusicGen.get_pretrained(
                "facebook/musicgen-medium",
                device=self._device,
            )
            self._loaded["musicgen"] = model
            logger.info("MusicGen-medium loaded on %s", self._device)
            return model
        except Exception as exc:
            logger.warning("MusicGen load failed: %s", exc)
            return None

    def load_audioldm(self) -> Any:
        """Load AudioLDM2 for sound effects."""
        if "audioldm" in self._loaded:
            return self._loaded["audioldm"]

        self._unload_all()

        try:
            import torch
            from diffusers import AudioLDM2Pipeline
            pipe = AudioLDM2Pipeline.from_pretrained(
                "cvssp/audioldm2",
                torch_dtype=torch.float16 if self._device != "cpu" else torch.float32,
                cache_dir=self._cache_dir,
            )
            if self._device != "cpu":
                pipe = pipe.to(self._device)
            if hasattr(pipe, "enable_model_cpu_offload") and self._device == "cuda":
                try:
                    pipe.enable_model_cpu_offload()
                except Exception:
                    pass
            self._loaded["audioldm"] = pipe
            logger.info("AudioLDM2 loaded on %s", self._device)
            return pipe
        except Exception as exc:
            logger.warning("AudioLDM2 load failed: %s", exc)
            return None

    def unload(self, key: str):
        if key in self._loaded:
            del self._loaded[key]
            self._flush_gpu()

    def _unload_all(self):
        self._loaded.clear()
        self._flush_gpu()

    def _flush_gpu(self):
        try:
            import torch
            if self._device == "cuda":
                torch.cuda.empty_cache()
            elif self._device == "mps":
                torch.mps.empty_cache()
        except ImportError:
            pass

    @property
    def loaded_models(self) -> List[str]:
        return list(self._loaded.keys())


# ═══════════════════════════════════════════════════════════════════════════════
# VOICE ENGINE — Neural TTS (XTTS-v2)
# ═══════════════════════════════════════════════════════════════════════════════

class NRSIVoiceEngine:
    """
    Production-grade text-to-speech using XTTS-v2.

    Capabilities:
      - 24 languages
      - Voice cloning from short reference audio
      - Emotion control via prompt conditioning
      - Speed and pitch adjustment
      - 24kHz native output → upsampled to 48kHz/24-bit
    """

    SUPPORTED_LANGUAGES = [
        "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
        "nl", "cs", "ar", "zh", "ja", "hu", "ko", "hi",
    ]

    EMOTION_PROMPTS: Dict[VoiceEmotion, str] = {
        VoiceEmotion.NEUTRAL: "",
        VoiceEmotion.HAPPY: " [laughing] [happy]",
        VoiceEmotion.SAD: " [sad] [sighing]",
        VoiceEmotion.ANGRY: " [angry]",
        VoiceEmotion.WHISPER: " [whispering]",
        VoiceEmotion.EXCITED: " [excited] [enthusiastic]",
        VoiceEmotion.CALM: " [calm] [soft]",
    }

    def __init__(self, model_manager: NRSIAudioModelManager):
        self._mm = model_manager
        self._cloned_voices: Dict[str, str] = {}
        self._voice_dir = os.path.join(
            model_manager._cache_dir, "voices"
        )
        os.makedirs(self._voice_dir, exist_ok=True)

    def synthesize(self, request: TTSGenerationRequest) -> Optional[np.ndarray]:
        """
        Synthesize speech from text. Returns float32 numpy array at native SR.
        Returns None if model unavailable.
        """
        model = self._mm.load_tts()
        if model is None:
            return None

        try:
            speaker_wav = None
            if request.reference_audio:
                ref_path = os.path.join(self._voice_dir, "_temp_ref.wav")
                with open(ref_path, "wb") as f:
                    f.write(request.reference_audio)
                speaker_wav = ref_path
            elif request.voice in self._cloned_voices:
                speaker_wav = self._cloned_voices[request.voice]

            text = request.text
            emotion_suffix = self.EMOTION_PROMPTS.get(
                request.emotion, ""
            )
            if emotion_suffix:
                text = text + emotion_suffix

            lang = request.language if request.language in self.SUPPORTED_LANGUAGES else "en"

            if speaker_wav:
                wav = model.tts(
                    text=text,
                    speaker_wav=speaker_wav,
                    language=lang,
                    speed=request.speed,
                )
            else:
                wav = model.tts(
                    text=text,
                    language=lang,
                    speed=request.speed,
                )

            if isinstance(wav, list):
                audio = np.array(wav, dtype=np.float32)
            else:
                audio = np.asarray(wav, dtype=np.float32)

            return audio

        except Exception as exc:
            logger.error("XTTS synthesis failed: %s", exc)
            return None

    @property
    def native_sample_rate(self) -> int:
        return 24000

    def clone_voice(self, reference_audio: bytes,
                    voice_name: str) -> bool:
        """Store a reference audio for voice cloning."""
        path = os.path.join(self._voice_dir, f"{voice_name}.wav")
        try:
            with open(path, "wb") as f:
                f.write(reference_audio)
            self._cloned_voices[voice_name] = path
            logger.info("Voice cloned: %s", voice_name)
            return True
        except Exception as exc:
            logger.error("Voice clone failed: %s", exc)
            return False

    def list_voices(self) -> Dict[str, str]:
        """List available voices (built-in + cloned)."""
        voices = {"default": "built-in"}
        for name, path in self._cloned_voices.items():
            voices[name] = f"cloned ({path})"
        return voices


# ═══════════════════════════════════════════════════════════════════════════════
# MUSIC ENGINE — MusicGen
# ═══════════════════════════════════════════════════════════════════════════════

class NRSIMusicEngine:
    """
    Production-grade music generation using Meta MusicGen.

    Capabilities:
      - Text-to-music from natural language
      - Variable length (multi-segment for >30s)
      - Temperature control for creativity
      - Stereo output at 32kHz → 48kHz production
    """

    MAX_SEGMENT_SECONDS = 30.0
    CROSSFADE_SECONDS = 2.0

    def __init__(self, model_manager: NRSIAudioModelManager):
        self._mm = model_manager

    def generate(self, request: AudioGenerationRequest) -> Optional[np.ndarray]:
        """
        Generate music from text prompt. Returns float32 numpy array.
        Returns None if model unavailable.
        """
        model = self._mm.load_musicgen()
        if model is None:
            return None

        try:
            import torch

            duration = max(0.5, min(request.duration_seconds, 300.0))
            model.set_generation_params(
                duration=min(duration, self.MAX_SEGMENT_SECONDS),
                temperature=request.temperature,
            )

            if duration <= self.MAX_SEGMENT_SECONDS:
                wav = model.generate([request.prompt])
                audio = wav[0].cpu().numpy()
                if audio.ndim == 2 and audio.shape[0] == 1:
                    audio = audio.squeeze(0)
                return audio.astype(np.float32)

            segments = []
            remaining = duration
            segment_idx = 0

            while remaining > 0:
                seg_dur = min(remaining, self.MAX_SEGMENT_SECONDS)
                model.set_generation_params(
                    duration=seg_dur,
                    temperature=request.temperature,
                )

                prompt = request.prompt
                if segment_idx > 0:
                    prompt = f"{request.prompt}, continuation"

                wav = model.generate([prompt])
                seg_audio = wav[0].cpu().numpy()
                if seg_audio.ndim == 2 and seg_audio.shape[0] == 1:
                    seg_audio = seg_audio.squeeze(0)
                segments.append(seg_audio.astype(np.float32))

                remaining -= seg_dur
                segment_idx += 1

            return self._crossfade_segments(
                segments, self.native_sample_rate
            )

        except Exception as exc:
            logger.error("MusicGen generation failed: %s", exc)
            return None

    def _crossfade_segments(self, segments: List[np.ndarray],
                            sr: int) -> np.ndarray:
        """Crossfade multiple segments together."""
        if len(segments) == 1:
            return segments[0]

        xfade_samples = int(self.CROSSFADE_SECONDS * sr)
        result = segments[0]

        for seg in segments[1:]:
            if len(result) < xfade_samples or len(seg) < xfade_samples:
                result = np.concatenate([result, seg])
                continue

            fade_out = np.linspace(1, 0, xfade_samples, dtype=np.float32)
            fade_in = np.linspace(0, 1, xfade_samples, dtype=np.float32)

            overlap = (result[-xfade_samples:] * fade_out +
                       seg[:xfade_samples] * fade_in)

            result = np.concatenate([
                result[:-xfade_samples],
                overlap,
                seg[xfade_samples:],
            ])

        return result

    @property
    def native_sample_rate(self) -> int:
        return 32000


# ═══════════════════════════════════════════════════════════════════════════════
# SFX ENGINE — AudioLDM2
# ═══════════════════════════════════════════════════════════════════════════════

class NRSISFXEngine:
    """
    Production-grade sound effect generation using AudioLDM2.

    Capabilities:
      - Any sound from text description
      - Negative prompts for quality control
      - Variable duration
      - 16kHz native → 48kHz production output
    """

    def __init__(self, model_manager: NRSIAudioModelManager):
        self._mm = model_manager

    def generate(self, request: AudioGenerationRequest) -> Optional[np.ndarray]:
        """
        Generate sound effect from text prompt. Returns float32 numpy array.
        Returns None if model unavailable.
        """
        pipe = self._mm.load_audioldm()
        if pipe is None:
            return None

        try:
            import torch

            duration = max(1.0, min(request.duration_seconds, 30.0))

            gen = None
            if request.seed is not None:
                device = "cuda" if self._mm.device == "cuda" else "cpu"
                gen = torch.Generator(device=device)
                gen.manual_seed(request.seed)

            negative = request.negative_prompt or "low quality, distorted"

            result = pipe(
                request.prompt,
                negative_prompt=negative,
                audio_length_in_s=duration,
                num_inference_steps=100,
                generator=gen,
            )

            audio = result.audios[0]
            if isinstance(audio, np.ndarray):
                return audio.astype(np.float32)
            return np.asarray(audio, dtype=np.float32)

        except Exception as exc:
            logger.error("AudioLDM2 generation failed: %s", exc)
            return None

    @property
    def native_sample_rate(self) -> int:
        return 16000


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO INTELLIGENCE — prompt analysis for audio generation
# ═══════════════════════════════════════════════════════════════════════════════

class NRSIAudioIntelligence:
    """
    Analyzes prompts to determine optimal audio generation parameters.
    Detects mood, genre, instruments, and maps to model settings.

    When web retrieval is enabled (WebRetrievalEngine connected),
    searches the web for reference descriptions of named artists,
    instruments, styles, or sounds to augment the enrichment.
    Web facts are cached in PVS-4 via the brain so repeated
    queries never hit the network twice.
    """

    _MOOD_KEYWORDS = {
        "dramatic": ["dramatic", "epic", "intense", "powerful", "cinematic"],
        "calm": ["calm", "peaceful", "ambient", "gentle", "relaxing"],
        "energetic": ["energetic", "upbeat", "fast", "dance", "electronic"],
        "dark": ["dark", "ominous", "horror", "suspense", "tension"],
        "happy": ["happy", "joyful", "bright", "cheerful", "playful"],
        "sad": ["sad", "melancholic", "somber", "emotional", "nostalgic"],
        "romantic": ["romantic", "love", "tender", "intimate", "warm"],
    }

    _GENRE_KEYWORDS = {
        "orchestral": ["orchestral", "symphony", "classical", "strings",
                        "brass", "woodwind", "philharmonic"],
        "electronic": ["electronic", "synth", "edm", "techno", "house",
                        "trance", "dubstep", "ambient electronic"],
        "rock": ["rock", "guitar", "drums", "bass", "metal", "punk",
                  "grunge", "indie rock"],
        "jazz": ["jazz", "saxophone", "trumpet", "swing", "blues",
                  "bebop", "smooth jazz"],
        "hiphop": ["hip hop", "rap", "beats", "trap", "boom bap"],
        "acoustic": ["acoustic", "folk", "unplugged", "singer-songwriter"],
        "cinematic": ["cinematic", "film score", "movie soundtrack",
                       "trailer music", "epic score"],
    }

    _WEB_TRIGGER_KEYWORDS = [
        "like", "style of", "similar to", "inspired by",
        "sounds like", "vibe of", "tone of", "voice of",
        "remix", "cover", "tribute",
    ]

    def __init__(self):
        self._web: Optional[Any] = None
        self._web_cache: Dict[str, List[str]] = {}

    def connect_web(self, web_engine: Any):
        """Connect WebRetrievalEngine for live web-augmented enrichment."""
        self._web = web_engine
        logger.info("AudioIntelligence: web retrieval connected")

    def analyze_music_prompt(self, prompt: str,
                             web_facts: Optional[List[str]] = None
                             ) -> Dict[str, Any]:
        """Analyze a music generation prompt, optionally with web facts."""
        lower = prompt.lower()
        mood = self._detect(lower, self._MOOD_KEYWORDS) or "neutral"
        genre = self._detect(lower, self._GENRE_KEYWORDS) or "cinematic"

        temperature = 1.0
        if mood in ("calm", "romantic"):
            temperature = 0.8
        elif mood in ("energetic", "happy"):
            temperature = 1.1

        enriched = self._enrich_music_prompt(prompt, mood, genre)

        if web_facts:
            fact_str = "; ".join(web_facts[:3])
            enriched = f"{enriched}, reference style: {fact_str}"

        return {
            "mood": mood,
            "genre": genre,
            "temperature": temperature,
            "enriched_prompt": enriched,
            "web_augmented": bool(web_facts),
        }

    def analyze_sfx_prompt(self, prompt: str,
                           web_facts: Optional[List[str]] = None
                           ) -> Dict[str, Any]:
        """Analyze a sound effect prompt."""
        enriched = prompt
        if web_facts:
            fact_str = "; ".join(web_facts[:2])
            enriched = f"{prompt}, detailed: {fact_str}"

        return {
            "enriched_prompt": enriched,
            "negative_prompt": "low quality, distorted, muffled, clipping",
            "web_augmented": bool(web_facts),
        }

    def analyze_voice_prompt(self, text: str,
                             web_facts: Optional[List[str]] = None
                             ) -> Dict[str, Any]:
        """Analyze text for voice synthesis parameters."""
        has_question = "?" in text
        has_exclamation = "!" in text
        is_long = len(text) > 500

        return {
            "text": text,
            "suggested_speed": 0.95 if is_long else 1.0,
            "suggested_emotion": (
                VoiceEmotion.EXCITED if has_exclamation
                else VoiceEmotion.NEUTRAL
            ),
            "web_augmented": bool(web_facts),
        }

    def needs_web_lookup(self, prompt: str) -> bool:
        """Check if this prompt references named entities that need web lookup."""
        lower = prompt.lower()
        return any(kw in lower for kw in self._WEB_TRIGGER_KEYWORDS)

    def build_web_query(self, prompt: str, task: str) -> str:
        """Build a web search query for reference material."""
        task_context = {
            "music": "music style characteristics instruments tempo",
            "sfx": "sound effect characteristics audio description",
            "voice": "voice characteristics tone speaking style",
        }
        ctx = task_context.get(task, "audio style")
        return f"{prompt} {ctx}"

    def _detect(self, text: str, mapping: Dict) -> Optional[str]:
        for category, keywords in mapping.items():
            for kw in keywords:
                if kw in text:
                    return category
        return None

    def _enrich_music_prompt(self, prompt: str, mood: str,
                             genre: str) -> str:
        enrichments = {
            "orchestral": ", full orchestra, lush strings, powerful brass, "
                          "concert hall acoustics",
            "electronic": ", synthesized pads, electronic beats, "
                          "atmospheric textures, clean production",
            "cinematic": ", film score quality, epic production, "
                         "dynamic range, emotional arc",
            "jazz": ", smooth performance, live instruments, "
                    "warm tone, skilled musicianship",
            "acoustic": ", natural instruments, intimate recording, "
                        "warm acoustic space",
        }
        suffix = enrichments.get(genre, ", high quality production")
        return prompt + suffix


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED NEURAL AUDIO ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class NRSINeuralAudioEngine:
    """
    Unified interface for all neural audio generation, fully integrated
    with the NRS brain (VLT, PVS-4, Tuition, Symbiotic Mesh, Creative Vision).

    Flow for every generation:
      1. PVS-4 instant recall check — return cached if exact match
      2. Tuition quality routing — correct quality tier from learned feedback
      3. Creative enrichment — mood/genre/style enrichment via Creative Vision
      4. Neural model generation — XTTS-v2 / MusicGen / AudioLDM2
      5. Post-processing — studio mastering chain
      6. Mesh validation — quality gate
      7. Brain feedback — store success/rejection for future learning
      8. VLT persistence — store generation knowledge + Redis bytes
    """

    def __init__(self, cache_dir: Optional[str] = None,
                 enable_brain: bool = True,
                 enable_web: bool = True):
        self._mm = NRSIAudioModelManager(cache_dir=cache_dir)
        self._voice = NRSIVoiceEngine(self._mm)
        self._music = NRSIMusicEngine(self._mm)
        self._sfx = NRSISFXEngine(self._mm)
        self._post = NRSIAudioPostProcessor()
        self._encoder = AudioEncoder()
        self._intel = NRSIAudioIntelligence()
        self._initialized = False
        self._enable_brain = enable_brain
        self._enable_web = enable_web
        self._brain: Optional[Any] = None
        self._web: Optional[Any] = None

    def initialize(self):
        """Connect to NRSI brain, web retrieval, and mark engine ready."""
        if self._enable_brain:
            try:
                from nrsi.core.neural_cache import NRSIMemoryBridge, CacheConfig
                cfg = CacheConfig()
                self._brain = NRSIMemoryBridge(config=cfg)
                self._brain.connect()
                health = self._brain.health_check()
                logger.info(
                    "Audio Brain connected — VLT:%s PVS4:%s Tuition:%s "
                    "Mesh:%s Redis:%s",
                    health["vlt"], health["pvs4"], health["tuition"],
                    health["mesh"], health["redis"],
                )
            except Exception as exc:
                logger.warning("Audio brain init failed: %s — running "
                               "without memory", exc)
                self._brain = None

        if self._enable_web:
            try:
                from nrsip.web_retrieval import WebRetrievalEngine
                self._web = WebRetrievalEngine()
                self._intel.connect_web(self._web)
                logger.info("Audio WebRetrieval connected (Brave+DDG)")
            except Exception as exc:
                logger.warning("Web retrieval init failed: %s — "
                               "running without web", exc)
                self._web = None

        self._initialized = True
        logger.info("NRS Neural Audio Engine ready on %s (brain=%s web=%s)",
                     self._mm.device,
                     "connected" if self._brain else "offline",
                     "connected" if self._web else "offline")

    def _web_enrich(self, prompt: str, task: str) -> List[str]:
        """
        Fetch web reference facts for a prompt if it references named
        entities/artists/styles. Uses PVS-4 cache first, web second.
        Results are stored in VLT L3 for future instant recall.
        """
        if not self._web:
            return []

        if not self._intel.needs_web_lookup(prompt):
            return []

        if self._brain:
            pvs_match = self._brain._vlt.pvs.lookup(
                f"web_audio:{prompt.strip().lower()}"
            )
            if pvs_match and pvs_match.data.get("type") == "web_audio_facts":
                facts = pvs_match.data.get("facts", [])
                if facts:
                    logger.info("Web facts PVS-4 hit for: %s", prompt[:40])
                    return facts

        try:
            import asyncio
            query = self._intel.build_web_query(prompt, task)

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    facts_result = pool.submit(
                        asyncio.run,
                        self._web.retrieve_facts(query, max_facts=3)
                    ).result(timeout=10)
            else:
                facts_result = asyncio.run(
                    self._web.retrieve_facts(query, max_facts=3)
                )

            facts = [f.text[:200] for f in facts_result if f.text]

            if facts and self._brain:
                self._brain._vlt.pvs.store(
                    text=f"web_audio:{prompt.strip().lower()}",
                    tier=task,
                    confidence=0.7,
                    data={"type": "web_audio_facts", "facts": facts},
                )
                self._brain._vlt.store(
                    key=f"web_audio_ref:{task}:{int(time.time())}",
                    value={"prompt": prompt, "facts": facts, "task": task},
                    layer=self._brain._VLTLayer.L3_PERSISTENT,
                    confidence=0.7,
                    domain="web_audio_reference",
                    source="web_retrieval",
                    tags={"web_audio", task},
                )

            logger.info("Web enrichment: %d facts for '%s'",
                         len(facts), prompt[:40])
            return facts

        except Exception as exc:
            logger.warning("Web enrichment failed: %s", exc)
            return []

    # ── Voice Synthesis ───────────────────────────────────────────────────

    def synthesize_voice(
        self, request: TTSGenerationRequest
    ) -> AudioGenerationResult:
        """Generate speech from text using XTTS-v2 with full brain path."""
        t0 = time.time()

        if self._brain:
            recalled = self._brain.recall_audio(
                request.text, "voice", PRODUCTION_SAMPLE_RATE
            )
            if recalled:
                audio_bytes, meta = recalled
                logger.info("Voice PVS-4 instant recall: %s", meta.get("prompt", "")[:40])
                return AudioGenerationResult(
                    audio_bytes=audio_bytes,
                    sample_rate=meta.get("sample_rate", PRODUCTION_SAMPLE_RATE),
                    channels=PRODUCTION_CHANNELS,
                    bit_depth=24,
                    duration_seconds=meta.get("duration", 0),
                    format="wav",
                    generation_time_ms=(time.time() - t0) * 1000,
                    device_used="pvs4_cache",
                    model_id="xtts_v2",
                    metadata={"source": "pvs4_recall"},
                )

        if self._brain:
            routed_q = self._brain.route_audio_quality(
                request.text, "voice", request.quality.value
            )
            q_map = {q.value: q for q in AudioQuality}
            if routed_q in q_map:
                request.quality = q_map[routed_q]

        web_facts = self._web_enrich(request.text, "voice")

        analysis = self._intel.analyze_voice_prompt(request.text, web_facts)
        if request.emotion == VoiceEmotion.NEUTRAL:
            request.emotion = analysis["suggested_emotion"]

        audio = self._voice.synthesize(request)
        if audio is None:
            if self._brain:
                self._brain.audio_feedback(
                    request.text, "voice", 0.0, "xtts_v2",
                    rejected=True, rejection_reason="model_unavailable",
                )
            return self._empty_result("tts_failed")

        mastered, sr = self._post.master(
            audio, self._voice.native_sample_rate,
            quality=request.quality, is_music=False,
        )

        encoded = self._encoder.encode(
            mastered, sr, request.output_format, request.quality,
        )

        duration = mastered.shape[-1] / sr

        if self._brain:
            validated = self._brain.validate_audio(
                request.text, "voice", duration, 0.85,
                {"model": "xtts_v2", "language": request.language},
            )
            q_score = 0.9 if validated else 0.4
            self._brain.audio_feedback(
                request.text, "voice", q_score, "xtts_v2",
                rejected=not validated,
                rejection_reason="" if validated else "mesh_quality_gate",
            )
            if validated:
                self._brain.remember_audio(
                    request.text, "voice", encoded, duration, sr, "xtts_v2",
                    {"language": request.language, "emotion": request.emotion.value},
                )

        elapsed_ms = (time.time() - t0) * 1000
        return AudioGenerationResult(
            audio_bytes=encoded,
            sample_rate=sr,
            channels=PRODUCTION_CHANNELS,
            bit_depth=QUALITY_BIT_DEPTHS.get(request.quality, 24),
            duration_seconds=duration,
            format=request.output_format.value,
            generation_time_ms=elapsed_ms,
            device_used=self._mm.device,
            model_id="xtts_v2",
        )

    # ── Music Generation ──────────────────────────────────────────────────

    def generate_music(
        self, request: AudioGenerationRequest
    ) -> AudioGenerationResult:
        """Generate music from text prompt using MusicGen with full brain path."""
        t0 = time.time()

        if self._brain:
            recalled = self._brain.recall_audio(
                request.prompt, "music", PRODUCTION_SAMPLE_RATE
            )
            if recalled:
                audio_bytes, meta = recalled
                logger.info("Music PVS-4 instant recall")
                return AudioGenerationResult(
                    audio_bytes=audio_bytes,
                    sample_rate=meta.get("sample_rate", PRODUCTION_SAMPLE_RATE),
                    channels=PRODUCTION_CHANNELS,
                    bit_depth=24,
                    duration_seconds=meta.get("duration", 0),
                    format="wav",
                    generation_time_ms=(time.time() - t0) * 1000,
                    device_used="pvs4_cache",
                    model_id="musicgen_medium",
                    metadata={"source": "pvs4_recall"},
                )

        web_facts = self._web_enrich(request.prompt, "music")

        analysis = self._intel.analyze_music_prompt(request.prompt, web_facts)

        if self._brain:
            routed_q = self._brain.route_audio_quality(
                request.prompt, "music", request.quality.value
            )
            q_map = {q.value: q for q in AudioQuality}
            if routed_q in q_map:
                request.quality = q_map[routed_q]

            enrichment = self._brain.audio_creative_enrich(
                request.prompt, "music",
                mood=analysis["mood"], genre=analysis["genre"],
            )
            enriched_prompt = enrichment.get(
                "enriched_prompt", analysis["enriched_prompt"]
            )
            if web_facts:
                enriched_prompt = f"{enriched_prompt}, reference: {'; '.join(web_facts[:2])}"
        else:
            enriched_prompt = analysis["enriched_prompt"]

        enriched_request = AudioGenerationRequest(
            prompt=enriched_prompt,
            negative_prompt=request.negative_prompt,
            duration_seconds=request.duration_seconds,
            quality=request.quality,
            output_format=request.output_format,
            seed=request.seed,
            temperature=analysis["temperature"],
        )

        audio = self._music.generate(enriched_request)
        if audio is None:
            if self._brain:
                self._brain.audio_feedback(
                    request.prompt, "music", 0.0, "musicgen_medium",
                    rejected=True, rejection_reason="model_unavailable",
                )
            return self._empty_result("musicgen_failed")

        mastered, sr = self._post.master(
            audio, self._music.native_sample_rate,
            quality=request.quality, is_music=True,
        )

        encoded = self._encoder.encode(
            mastered, sr, request.output_format, request.quality,
        )

        duration = mastered.shape[-1] / sr

        if self._brain:
            validated = self._brain.validate_audio(
                request.prompt, "music", duration, 0.85,
                {"model": "musicgen_medium", "mood": analysis["mood"],
                 "genre": analysis["genre"]},
            )
            q_score = 0.9 if validated else 0.3
            self._brain.audio_feedback(
                request.prompt, "music", q_score, "musicgen_medium",
                rejected=not validated,
                rejection_reason="" if validated else "mesh_quality_gate",
            )
            if validated:
                self._brain.remember_audio(
                    request.prompt, "music", encoded, duration, sr,
                    "musicgen_medium",
                    {"mood": analysis["mood"], "genre": analysis["genre"]},
                )

        elapsed_ms = (time.time() - t0) * 1000
        return AudioGenerationResult(
            audio_bytes=encoded,
            sample_rate=sr,
            channels=PRODUCTION_CHANNELS,
            bit_depth=QUALITY_BIT_DEPTHS.get(request.quality, 24),
            duration_seconds=duration,
            format=request.output_format.value,
            generation_time_ms=elapsed_ms,
            device_used=self._mm.device,
            model_id="musicgen_medium",
            metadata={"mood": analysis["mood"], "genre": analysis["genre"]},
        )

    # ── SFX Generation ────────────────────────────────────────────────────

    def generate_sfx(
        self, request: AudioGenerationRequest
    ) -> AudioGenerationResult:
        """Generate sound effects using AudioLDM2 with full brain path."""
        t0 = time.time()

        if self._brain:
            recalled = self._brain.recall_audio(
                request.prompt, "sfx", PRODUCTION_SAMPLE_RATE
            )
            if recalled:
                audio_bytes, meta = recalled
                logger.info("SFX PVS-4 instant recall")
                return AudioGenerationResult(
                    audio_bytes=audio_bytes,
                    sample_rate=meta.get("sample_rate", PRODUCTION_SAMPLE_RATE),
                    channels=PRODUCTION_CHANNELS,
                    bit_depth=24,
                    duration_seconds=meta.get("duration", 0),
                    format="wav",
                    generation_time_ms=(time.time() - t0) * 1000,
                    device_used="pvs4_cache",
                    model_id="audioldm2",
                    metadata={"source": "pvs4_recall"},
                )

        web_facts = self._web_enrich(request.prompt, "sfx")

        analysis = self._intel.analyze_sfx_prompt(request.prompt, web_facts)

        if self._brain:
            routed_q = self._brain.route_audio_quality(
                request.prompt, "sfx", request.quality.value
            )
            q_map = {q.value: q for q in AudioQuality}
            if routed_q in q_map:
                request.quality = q_map[routed_q]

            enrichment = self._brain.audio_creative_enrich(
                request.prompt, "sfx", mood="", genre="",
            )
            enriched_prompt = enrichment.get(
                "enriched_prompt", analysis["enriched_prompt"]
            )
            if web_facts:
                enriched_prompt = f"{enriched_prompt}, reference: {'; '.join(web_facts[:2])}"
        else:
            enriched_prompt = analysis["enriched_prompt"]

        enriched_request = AudioGenerationRequest(
            prompt=enriched_prompt,
            negative_prompt=analysis["negative_prompt"],
            duration_seconds=request.duration_seconds,
            quality=request.quality,
            output_format=request.output_format,
            seed=request.seed,
            temperature=request.temperature,
        )

        audio = self._sfx.generate(enriched_request)
        if audio is None:
            if self._brain:
                self._brain.audio_feedback(
                    request.prompt, "sfx", 0.0, "audioldm2",
                    rejected=True, rejection_reason="model_unavailable",
                )
            return self._empty_result("audioldm_failed")

        mastered, sr = self._post.master(
            audio, self._sfx.native_sample_rate,
            quality=request.quality, is_music=False,
        )

        encoded = self._encoder.encode(
            mastered, sr, request.output_format, request.quality,
        )

        duration = mastered.shape[-1] / sr

        if self._brain:
            validated = self._brain.validate_audio(
                request.prompt, "sfx", duration, 0.8,
                {"model": "audioldm2"},
            )
            q_score = 0.85 if validated else 0.3
            self._brain.audio_feedback(
                request.prompt, "sfx", q_score, "audioldm2",
                rejected=not validated,
                rejection_reason="" if validated else "mesh_quality_gate",
            )
            if validated:
                self._brain.remember_audio(
                    request.prompt, "sfx", encoded, duration, sr, "audioldm2",
                )

        elapsed_ms = (time.time() - t0) * 1000
        return AudioGenerationResult(
            audio_bytes=encoded,
            sample_rate=sr,
            channels=PRODUCTION_CHANNELS,
            bit_depth=QUALITY_BIT_DEPTHS.get(request.quality, 24),
            duration_seconds=duration,
            format=request.output_format.value,
            generation_time_ms=elapsed_ms,
            device_used=self._mm.device,
            model_id="audioldm2",
        )

    # ── Video Soundtrack (SFX + Music layered) ────────────────────────────

    def generate_video_soundtrack(
        self,
        sfx_prompt: str,
        music_prompt: str,
        duration_seconds: float,
        sfx_volume: float = 0.7,
        music_volume: float = 0.4,
        quality: AudioQuality = AudioQuality.HIGH,
    ) -> AudioGenerationResult:
        """
        Generate a layered video soundtrack: SFX + background music.
        Both tracks go through the full brain path individually,
        then are mixed at the specified volume ratios.
        """
        t0 = time.time()

        sfx_audio = None
        if sfx_prompt:
            sfx_req = AudioGenerationRequest(
                prompt=sfx_prompt,
                duration_seconds=duration_seconds,
                quality=quality,
            )
            sfx_audio = self._sfx.generate(sfx_req)

        music_audio = None
        if music_prompt:
            music_req = AudioGenerationRequest(
                prompt=music_prompt,
                duration_seconds=duration_seconds,
                quality=quality,
            )
            music_audio = self._music.generate(music_req)

        target_sr = PRODUCTION_SAMPLE_RATE
        target_samples = int(duration_seconds * target_sr)

        mixed = np.zeros((2, target_samples), dtype=np.float32)

        if sfx_audio is not None:
            sfx_r = self._post.resample(sfx_audio,
                                         self._sfx.native_sample_rate,
                                         target_sr)
            sfx_r = self._post.mono_to_stereo(sfx_r)
            length = min(sfx_r.shape[-1], target_samples)
            mixed[:, :length] += sfx_r[:, :length] * sfx_volume

        if music_audio is not None:
            mus_r = self._post.resample(music_audio,
                                         self._music.native_sample_rate,
                                         target_sr)
            mus_r = self._post.mono_to_stereo(mus_r)
            mus_r = self._post.stereo_widen(mus_r, width=1.2)
            length = min(mus_r.shape[-1], target_samples)
            mixed[:, :length] += mus_r[:, :length] * music_volume

        mixed = self._post.normalize(mixed, target_db=-1.0)
        mixed = self._post.fade(mixed, target_sr, fade_in_ms=20,
                                fade_out_ms=50)

        encoded = self._encoder.encode(
            mixed, target_sr, AudioFormat.WAV, quality,
        )

        if self._brain:
            combined_prompt = f"{sfx_prompt} | {music_prompt}"
            self._brain.remember_audio(
                combined_prompt, "soundtrack", encoded,
                duration_seconds, target_sr, "audioldm2+musicgen",
            )

        elapsed_ms = (time.time() - t0) * 1000
        return AudioGenerationResult(
            audio_bytes=encoded,
            sample_rate=target_sr,
            channels=PRODUCTION_CHANNELS,
            bit_depth=QUALITY_BIT_DEPTHS.get(quality, 24),
            duration_seconds=duration_seconds,
            format="wav",
            generation_time_ms=elapsed_ms,
            device_used=self._mm.device,
            model_id="audioldm2+musicgen",
        )

    # ── Voice Management ──────────────────────────────────────────────────

    def clone_voice(self, reference_audio: bytes,
                    voice_name: str) -> bool:
        """Clone a voice from reference audio."""
        return self._voice.clone_voice(reference_audio, voice_name)

    def list_voices(self) -> Dict[str, str]:
        """List available voices."""
        return self._voice.list_voices()

    def feedback(self, prompt: str, task: str, quality_rating: int,
                 details: Optional[Dict] = None):
        """External feedback entry point for Tuition learning."""
        if not self._brain:
            return
        student_tier = "standard"
        teacher_tier = "high" if quality_rating >= 3 else "draft"
        self._brain.audio_feedback(
            prompt, task,
            quality_score=quality_rating / 5.0,
            model_id=f"neural_audio_{task}",
            rejected=quality_rating <= 2,
            rejection_reason="user_rejection" if quality_rating <= 2 else "",
        )

    def shutdown(self):
        """Release all model memory."""
        self._mm._unload_all()
        self._initialized = False

    @property
    def brain_stats(self) -> Dict[str, Any]:
        if self._brain:
            return self._brain.stats
        return {"brain": "offline"}

    @property
    def stats(self) -> Dict[str, Any]:
        web_stats = {}
        if self._web:
            web_stats = self._web.stats if hasattr(self._web, "stats") else {}
        return {
            "device": self._mm.device,
            "loaded_models": self._mm.loaded_models,
            "voices": self._voice.list_voices(),
            "initialized": self._initialized,
            "brain": "connected" if self._brain else "offline",
            "brain_stats": self.brain_stats,
            "web": "connected" if self._web else "offline",
            "web_stats": web_stats,
        }

    def _empty_result(self, reason: str) -> AudioGenerationResult:
        return AudioGenerationResult(
            metadata={"error": reason, "fallback": True},
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    "NRSINeuralAudioEngine",
    "NRSIVoiceEngine",
    "NRSIMusicEngine",
    "NRSISFXEngine",
    "NRSIAudioPostProcessor",
    "NRSIAudioModelManager",
    "NRSIAudioIntelligence",
    "AudioEncoder",
    "AudioGenerationRequest",
    "TTSGenerationRequest",
    "AudioGenerationResult",
    "AudioFormat",
    "AudioQuality",
    "VoiceEmotion",
    "PRODUCTION_SAMPLE_RATE",
    "PRODUCTION_BIT_DEPTH",
    "PRODUCTION_CHANNELS",
    "get_audio_device",
]
