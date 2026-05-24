"""
NRSI Neurons — Peripheral Nervous System with Network Locality.

Binary neuron layer: 100M+ ON/OFF signal carriers.
Same query → same activation → deterministic output.

═══════════════════════════════════════════════════════════════════
  THE SCALING PROBLEM (why this file exists in this form)
═══════════════════════════════════════════════════════════════════

If you naively broadcast neuron activations across NRSIP:

  111K activations × 8 bytes = ~888KB per query
  6,433 RPS × 3 NRS instances = 19,299 queries/sec
  19,299 × 888KB = ~16 GB/sec of raw neuron traffic
  On 3 NRSIPs → network dies instantly

The human body solves this the same way:
  - Your peripheral neurons fire locally
  - Your brain compresses those signals
  - You send WORDS to other people, not raw neuron firings
  - Words are 10-100 bytes. Neuron firings are millions.

═══════════════════════════════════════════════════════════════════
  THE LOCALITY PRINCIPLE
═══════════════════════════════════════════════════════════════════

Three layers of signal compression:

  Layer 0: Raw Activations (NEVER leaves the NRS instance)
    111K neuron IDs + similarity scores
    ~888KB per query
    Lives in GPU memory, dies after T0 routing

  Layer 1: Activation Digest (LOCAL to NRS, shared between lobes)
    Domain distribution + top-k compressed + signature hash
    ~2KB per query (440x compression)
    Passed from neurons → T0 → lobes within ONE instance

  Layer 2: Routing Packet (THIS is what crosses NRSIP)
    Tier + domain + confidence + query hash
    ~128 bytes per query (6,937x compression vs raw)
    The ONLY thing that travels between NRS instances

Traffic comparison at 6,433 RPS × 3 instances:

  Without locality:  ~16 GB/sec  ← kills the network
  With Layer 1 only: ~37 MB/sec  ← manageable but wasteful
  With Layer 2 only: ~2.4 MB/sec ← trivial, TCP handles it

Layer 2 at 2.4 MB/sec is LESS traffic than a 1080p video stream.
Three NRS instances + three NRSIPs = ~7.2 MB/sec total.
A basic 1 Gbps link handles 125 MB/sec. We use 5.7% of it.

═══════════════════════════════════════════════════════════════════
  MULTI-NRS SCALING
═══════════════════════════════════════════════════════════════════

Single NRS:
  100M neurons, local activation, local T0 routing
  No network traffic. Everything in GPU memory.

2-3 NRS instances (connected via NRSIP):
  Each instance has its OWN 100M neurons (not shared)
  Neurons are instance-local, never synchronized
  Only Layer 2 routing packets cross NRSIP
  Use case: load balancing, domain specialization

10+ NRS instances (NRS cluster):
  Domain-sharded: each instance owns specific domain creases
  NRSIP routes queries to the RIGHT instance
  Neurons on each instance only fire for their domains
  Cross-instance traffic = Layer 2 packets only

100+ NRS instances (NRS mesh):
  Hierarchical: regional routers aggregate Layer 2 packets
  Leaf instances handle domain processing
  Spine instances handle cross-domain coordination
  Traffic stays logarithmic, not linear with instance count

═══════════════════════════════════════════════════════════════════
  ARCHITECTURE
═══════════════════════════════════════════════════════════════════

  Query arrives at NRS instance
       │
       ▼
  ┌──────────────────────────────┐
  │  BinaryNeuronBank (LOCAL)    │  Layer 0: 888KB
  │  100M neurons, 111K fire     │  GPU memory only
  │  Cosine sim → argmax → mask  │  Dies after routing
  └──────────┬───────────────────┘
             │ compress()
             ▼
  ┌──────────────────────────────┐
  │  ActivationDigest (LOCAL)    │  Layer 1: ~2KB
  │  Domain dist + top signals   │  Shared between lobes
  │  + signature hash            │  Within ONE instance
  └──────────┬───────────────────┘
             │ to_routing_packet()
             ▼
  ┌──────────────────────────────┐
  │  RoutingPacket (NETWORK)     │  Layer 2: ~128 bytes
  │  Tier + domain + confidence  │  Crosses NRSIP
  │  + query_hash                │  Between NRS instances
  └──────────────────────────────┘

The neuron bank is LOCAL. The routing packet is NETWORK.
Nothing in between ever leaves the instance.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import math
import random
import time

import numpy as np

try:
    import cupy as _cp
    _GPU_AVAILABLE = True
except ImportError:
    _GPU_AVAILABLE = False
    _cp = None


def _xp(use_gpu: bool = True):
    if use_gpu and _GPU_AVAILABLE:
        return _cp
    return np


# ── Constants ────────────────────────────────────────────────────────────────

# Production specs
DEFAULT_TOTAL_NEURONS = 20_000_000_000   # 20B neurons
DEFAULT_ACTIVE_K = 20_000_000            # 0.1% sparsity
DEFAULT_EMBEDDING_DIM = 1024             # Extended embedding dimension

# Layer sizes (bytes, approximate)
LAYER_0_SIZE_BYTES = DEFAULT_ACTIVE_K * 8    # ~160MB raw activations
LAYER_1_SIZE_BYTES = 2_048                    # ~2KB digest
LAYER_2_SIZE_BYTES = 128                      # ~128B routing packet


# ── Neuron Types ─────────────────────────────────────────────────────────────

class NeuronState(Enum):
    """Binary state — ON or OFF. No in-between. No probability."""
    OFF = 0
    ON = 1


class SignalLayer(Enum):
    """Which compression layer a signal is at."""
    RAW = 0        # Layer 0: raw activations (LOCAL, ~888KB)
    DIGEST = 1     # Layer 1: compressed digest (LOCAL, ~2KB)
    PACKET = 2     # Layer 2: routing packet (NETWORK, ~128B)


# ── Layer 0: Raw Activation Pattern ──────────────────────────────────────────

@dataclass
class NeuronActivation:
    """Record of a single neuron firing."""
    neuron_id: int
    state: NeuronState
    similarity: float
    domain_hint: Optional[str] = None


@dataclass
class ActivationPattern:
    """
    Layer 0 — Raw Activation Pattern.

    NEVER leaves the NRS instance. Lives in GPU memory.
    Dies after T0 routing extracts what it needs.

    Contains the full 111K active neuron IDs with similarity
    scores. This is ~888KB of data per query.
    """
    active_ids: List[int]
    total_neurons: int
    query_hash: str
    similarities: Dict[int, float] = field(default_factory=dict)
    domain_signals: Dict[str, int] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    layer: SignalLayer = SignalLayer.RAW

    @property
    def sparsity(self) -> float:
        if self.total_neurons == 0:
            return 0.0
        return len(self.active_ids) / self.total_neurons

    @property
    def active_count(self) -> int:
        return len(self.active_ids)

    @property
    def signature(self) -> str:
        """Deterministic signature of this activation pattern."""
        id_bytes = b"".join(struct.pack(">I", nid) for nid in sorted(self.active_ids))
        return hashlib.sha256(id_bytes).hexdigest()

    def domain_distribution(self) -> Dict[str, float]:
        """Fraction of active neurons per domain."""
        total = sum(self.domain_signals.values())
        if total == 0:
            return {}
        return {d: c / total for d, c in self.domain_signals.items()}

    @property
    def estimated_bytes(self) -> int:
        """Approximate size of this raw activation in memory."""
        return len(self.active_ids) * 8 + len(self.similarities) * 12

    def compress(self) -> "ActivationDigest":
        """
        Compress Layer 0 → Layer 1 (ActivationDigest).

        440x compression: ~888KB → ~2KB
        Keeps domain distribution, top-k signals, signature.
        Discards individual neuron IDs and scores.
        """
        dist = self.domain_distribution()

        top_signals = sorted(
            self.similarities.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]

        return ActivationDigest(
            signature=self.signature,
            query_hash=self.query_hash,
            active_count=self.active_count,
            total_neurons=self.total_neurons,
            sparsity=self.sparsity,
            domain_distribution=dist,
            top_signals=top_signals,
            mean_similarity=sum(self.similarities.values()) / max(len(self.similarities), 1),
            max_similarity=max(self.similarities.values()) if self.similarities else 0.0,
            created_at=self.created_at,
        )

    def __repr__(self) -> str:
        return (
            f"ActivationPattern(L0, active={self.active_count}/{self.total_neurons}, "
            f"~{self.estimated_bytes // 1024}KB)"
        )


# ── Layer 1: Activation Digest (LOCAL, between lobes) ───────────────────────

@dataclass
class ActivationDigest:
    """
    Layer 1 — Activation Digest.

    LOCAL to the NRS instance. Shared between T0 and lobes.
    Never crosses NRSIP. ~2KB per query.

    Contains everything T0 and the lobes need for routing
    and processing, without the raw neuron IDs.
    """
    signature: str
    query_hash: str
    active_count: int
    total_neurons: int
    sparsity: float
    domain_distribution: Dict[str, float]
    top_signals: List[Tuple[int, float]]
    mean_similarity: float
    max_similarity: float
    created_at: float = field(default_factory=time.time)
    layer: SignalLayer = SignalLayer.DIGEST

    @property
    def primary_domain(self) -> Optional[str]:
        """Strongest domain signal."""
        if not self.domain_distribution:
            return None
        return max(self.domain_distribution, key=self.domain_distribution.get)

    @property
    def domain_confidence(self) -> float:
        """How confident is the domain detection (0-1)."""
        if not self.domain_distribution:
            return 0.0
        values = sorted(self.domain_distribution.values(), reverse=True)
        if len(values) < 2:
            return values[0] if values else 0.0
        return values[0] - values[1]

    @property
    def estimated_bytes(self) -> int:
        return LAYER_1_SIZE_BYTES

    def to_routing_packet(
        self,
        tier: str = "T1",
        confidence: float = 0.0,
    ) -> "RoutingPacket":
        """
        Compress Layer 1 → Layer 2 (RoutingPacket).

        16x compression: ~2KB → ~128B
        This is what crosses NRSIP between NRS instances.
        """
        return RoutingPacket(
            query_hash=self.query_hash,
            activation_signature=self.signature[:16],
            tier=tier,
            domain=self.primary_domain,
            domain_confidence=self.domain_confidence,
            neuron_sparsity=self.sparsity,
            mean_similarity=self.mean_similarity,
            confidence=confidence,
            source_instance_id="",
            created_at=self.created_at,
        )

    def __repr__(self) -> str:
        dom = self.primary_domain or "none"
        return (
            f"ActivationDigest(L1, domain={dom}, "
            f"sparsity={self.sparsity:.4f}, ~{self.estimated_bytes}B)"
        )


# ── Layer 2: Routing Packet (NETWORK, crosses NRSIP) ────────────────────────

@dataclass
class RoutingPacket:
    """
    Layer 2 — Routing Packet.

    THE ONLY THING that crosses NRSIP between NRS instances.
    ~128 bytes. At 6,433 RPS = ~800 KB/sec per instance.
    Three instances on three NRSIPs = ~2.4 MB/sec total.
    A 1 Gbps link handles 125 MB/sec. We use < 2%.

    Contains just enough for another NRS instance to:
      1. Know what tier to process at
      2. Know what domain the query belongs to
      3. Verify it matches the originating activation
      4. Route to the correct crease

    Does NOT contain:
      - Individual neuron IDs (local only)
      - Similarity scores (local only)
      - Raw activation data (local only)
      - The actual query text (travels separately in NRSIP frame)
    """
    query_hash: str                    # 32 bytes
    activation_signature: str          # 16 bytes
    tier: str                          # 2 bytes
    domain: Optional[str]              # ~16 bytes
    domain_confidence: float           # 8 bytes
    neuron_sparsity: float             # 8 bytes
    mean_similarity: float             # 8 bytes
    confidence: float                  # 8 bytes
    source_instance_id: str = ""       # 16 bytes
    timestamp: float = field(default_factory=time.time)
    created_at: float = 0.0
    layer: SignalLayer = SignalLayer.PACKET

    @property
    def estimated_bytes(self) -> int:
        return LAYER_2_SIZE_BYTES

    def to_bytes(self) -> bytes:
        """
        Serialize to compact binary for NRSIP wire protocol.
        Fixed 128-byte frame for predictable network behavior.
        """
        parts = [
            self.query_hash[:32].encode("utf-8").ljust(32, b"\x00"),
            self.activation_signature[:16].encode("utf-8").ljust(16, b"\x00"),
            self.tier.encode("utf-8").ljust(4, b"\x00"),
            (self.domain or "").encode("utf-8")[:24].ljust(24, b"\x00"),
            struct.pack(">d", self.domain_confidence),
            struct.pack(">d", self.neuron_sparsity),
            struct.pack(">d", self.mean_similarity),
            struct.pack(">d", self.confidence),
            self.source_instance_id[:16].encode("utf-8").ljust(16, b"\x00"),
            struct.pack(">d", self.timestamp),
        ]
        frame = b"".join(parts)
        return frame[:128].ljust(128, b"\x00")

    @classmethod
    def from_bytes(cls, data: bytes) -> "RoutingPacket":
        """Deserialize from NRSIP wire format."""
        if len(data) < 128:
            data = data.ljust(128, b"\x00")
        return cls(
            query_hash=data[0:32].rstrip(b"\x00").decode("utf-8", errors="replace"),
            activation_signature=data[32:48].rstrip(b"\x00").decode("utf-8", errors="replace"),
            tier=data[48:52].rstrip(b"\x00").decode("utf-8", errors="replace"),
            domain=data[52:76].rstrip(b"\x00").decode("utf-8", errors="replace") or None,
            domain_confidence=struct.unpack(">d", data[76:84])[0],
            neuron_sparsity=struct.unpack(">d", data[84:92])[0],
            mean_similarity=struct.unpack(">d", data[92:100])[0],
            confidence=struct.unpack(">d", data[100:108])[0],
            source_instance_id=data[108:124].rstrip(b"\x00").decode("utf-8", errors="replace"),
            timestamp=struct.unpack(">d", data[124:132])[0] if len(data) >= 132 else time.time(),
        )

    def __repr__(self) -> str:
        return (
            f"RoutingPacket(L2, tier={self.tier}, "
            f"domain={self.domain}, conf={self.confidence:.2f}, "
            f"~{self.estimated_bytes}B)"
        )


# ── Embedding Engine ─────────────────────────────────────────────────────────

class NRSEmbeddingEngine:
    """TF-IDF weighted word-vector embeddings — transformer-free semantic vectors.

    Three-tier vector source (tried in order):
      1. Pre-trained GloVe/Word2Vec loaded from NRS_VECTORS_PATH or via gensim
      2. Subword hash vectors (FastText-like: words sharing n-grams get
         similar representations — deterministic, no external data)
      3. Character hash (legacy fallback, worst quality)

    Embedding process:
      - Tokenise input into words
      - Look up (or compute) each word's base vector
      - Weight by TF-IDF (term frequency × inverse document frequency)
      - Sum, L2-normalise → unit-norm output vector

    Same input always produces the same vector (deterministic).
    """

    _INTERNAL_DIM = 1024
    _NGRAM_RANGE = (3, 6)

    _LOG_IDF: Dict[str, float] = {}

    def __init__(self, dim: int = DEFAULT_EMBEDDING_DIM, seed: int = 42):
        self.dim = dim
        self._seed = seed
        self._rng = np.random.RandomState(seed)
        self._domain_seeds: Dict[str, np.ndarray] = {}

        self._word_vectors: Dict[str, np.ndarray] = {}
        self._word_cache_limit = 2_000_000
        self._external_loaded = False
        self._internal_dim = self._INTERNAL_DIM

        self._projection: Optional[np.ndarray] = None

        try:
            from nrsip.unicode_tokenizer import (
                tokenize as _utok,
                is_stop_word as _usw,
                subword_ngrams as _usng,
            )
            self._unicode_tokenize = _utok
            self._unicode_is_stop = _usw
            self._unicode_ngrams = _usng
        except ImportError:
            self._unicode_tokenize = None
            self._unicode_is_stop = None
            self._unicode_ngrams = None

        self._load_external_vectors()
        self._build_projection()
        self._build_idf()

    def _load_external_vectors(self) -> None:
        """Try loading pre-trained word vectors (file → spaCy → gensim → subword)."""
        import os
        path = os.environ.get("NRS_VECTORS_PATH", "")

        if path:
            try:
                self._load_glove_text(path)
                return
            except Exception:
                pass

        try:
            self._load_spacy_vectors()
            if self._external_loaded:
                return
        except Exception:
            pass

        try:
            import gensim.downloader as api  # type: ignore[import-untyped]
            model = api.load("glove-wiki-gigaword-300")
            for word in model.key_to_index:
                self._word_vectors[word] = model[word].astype(np.float32)
            self._internal_dim = model.vector_size
            self._external_loaded = True
            return
        except Exception:
            pass

    def _load_spacy_vectors(self) -> None:
        """Load word vectors from the best available spaCy model."""
        import re as _re
        import spacy
        _alpha_re = _re.compile(r'^[a-z]{2,}$')

        for model_name in ("en_core_web_lg", "en_core_web_md"):
            try:
                nlp = spacy.load(model_name, disable=["parser", "ner", "tagger"])
                vectors = nlp.vocab.vectors
                if vectors.shape[0] == 0:
                    continue
                self._internal_dim = vectors.shape[1]
                strings = nlp.vocab.strings
                count = 0
                for key in vectors.keys():
                    word = strings[key]
                    low = word.lower()
                    if low in self._word_vectors:
                        continue
                    if not _alpha_re.match(low):
                        continue
                    vec = nlp.vocab[word].vector
                    norm = np.linalg.norm(vec)
                    if norm < 1e-6:
                        continue
                    self._word_vectors[low] = vec.astype(np.float32)
                    count += 1
                if count > 0:
                    self._external_loaded = True
                    self._spacy_model = model_name
                return
            except OSError:
                continue

    def _load_glove_text(self, path: str) -> None:
        """Load GloVe-format text file (word vec0 vec1 ... vecN)."""
        count = 0
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                if len(parts) < 10:
                    continue
                word = parts[0]
                vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                self._word_vectors[word] = vec
                if count == 0:
                    self._internal_dim = len(vec)
                count += 1
        if count > 0:
            self._external_loaded = True

    def _build_projection(self) -> None:
        """Build random projection matrix internal_dim → output dim."""
        if self._internal_dim == self.dim:
            self._projection = None
            return
        rng = np.random.RandomState(self._seed + 7)
        mat = rng.randn(self._internal_dim, self.dim).astype(np.float32)
        mat /= np.sqrt(self._internal_dim)
        self._projection = mat

    def _build_idf(self) -> None:
        """Seed IDF cache for known stop words across all loaded languages."""
        if self._LOG_IDF:
            return
        try:
            from nrsip.unicode_tokenizer import STOP_WORDS
            for sw_set in STOP_WORDS.values():
                for word in sw_set:
                    self._LOG_IDF[word] = 0.1
        except ImportError:
            pass

    def _subword_vector(self, word: str) -> np.ndarray:
        """FastText-like: average n-gram hash vectors for a word.

        Uses Unicode-safe n-gram extraction so CJK, Devanagari, Arabic,
        Cyrillic, etc. all produce meaningful sub-token features.
        """
        if self._unicode_ngrams is not None:
            ngrams = self._unicode_ngrams(word, self._NGRAM_RANGE[0], self._NGRAM_RANGE[1])
        else:
            padded = f"<{word}>"
            ngrams = [word]
            for n in range(self._NGRAM_RANGE[0], self._NGRAM_RANGE[1] + 1):
                for i in range(len(padded) - n + 1):
                    ngrams.append(padded[i : i + n])
        vec = np.zeros(self._internal_dim, dtype=np.float32)
        for ng in ngrams:
            h = int(hashlib.sha256(ng.encode("utf-8")).hexdigest(), 16)
            rng = np.random.RandomState(h & 0x7FFFFFFF)
            vec += rng.randn(self._internal_dim).astype(np.float32)
        vec /= max(len(ngrams), 1)
        return vec

    def _word_vector(self, word: str) -> np.ndarray:
        """Get the vector for a single word (external or subword hash)."""
        if word in self._word_vectors:
            return self._word_vectors[word]
        vec = self._subword_vector(word)
        if len(self._word_vectors) < self._word_cache_limit:
            self._word_vectors[word] = vec
        return vec

    def _idf_weight(self, word: str) -> float:
        """IDF weight for a word. Stop words get 0.1, known words get
        cached IDF, unknown words get a high weight (rare = informative)."""
        if word in self._LOG_IDF:
            return self._LOG_IDF[word]
        if self._unicode_is_stop is not None:
            if self._unicode_is_stop(word):
                return 0.1
        if self._external_loaded and word in self._word_vectors:
            return 1.0
        return 2.0

    def _project(self, vec: np.ndarray) -> np.ndarray:
        """Project from internal_dim to output dim."""
        if self._projection is None:
            return vec
        return vec @ self._projection

    def embed(self, text: str) -> np.ndarray:
        """Embed text into a unit-norm vector via TF-IDF weighted word vectors.

        Supports all Unicode scripts: Latin, CJK, Arabic, Devanagari,
        Cyrillic, Hangul, Thai, and 150+ other languages.
        """
        if not text:
            return np.zeros(self.dim, dtype=np.float32)

        if self._unicode_tokenize is not None:
            tokens = self._unicode_tokenize(text)
        else:
            import re
            text_lower = text.lower().strip()
            tokens = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text_lower)
        if not tokens:
            return np.zeros(self.dim, dtype=np.float32)

        from collections import Counter
        tf = Counter(tokens)
        total = len(tokens)

        vec = np.zeros(self._internal_dim, dtype=np.float32)
        total_weight = 0.0
        for word, count in tf.items():
            wv = self._word_vector(word)
            w = (count / total) * self._idf_weight(word)
            vec += wv * w
            total_weight += w

        if total_weight > 1e-9:
            vec /= total_weight

        out = self._project(vec)

        norm = np.linalg.norm(out)
        if norm > 1e-9:
            out /= norm
        return out.astype(np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts. Returns (N, dim) array."""
        vecs = np.stack([self.embed(t) for t in texts])
        return vecs

    def register_domain_bias(self, domain: str, keywords: List[str]):
        """Pre-compute a domain centroid for domain-aware neuron init."""
        if not keywords:
            return
        vecs = self.embed_batch(keywords)
        centroid = vecs.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 1e-9:
            centroid /= norm
        self._domain_seeds[domain] = centroid


# ── Binary Neuron Bank ───────────────────────────────────────────────────────

class BinaryNeuronBank:
    """
    Peripheral Nervous System — Binary Neuron Array with real 768-d embeddings.

    INSTANCE-LOCAL. Neurons never leave this NRS instance.
    Each NRS instance has its OWN neuron bank.

    Activation process (deterministic):
      1. Query → embedded as vector q in R^768 via NRSEmbeddingEngine
      2. Cosine similarity: scores = q · N^T (GPU-backed via CuPy when available)
      3. Argmax selects top-k (NO sampling in deterministic mode)
      4. Selected = ON, rest = OFF
      5. Raw ActivationPattern (Layer 0, ~888KB) — LOCAL
      6. Compressed to ActivationDigest (Layer 1, ~2KB) — LOCAL
      7. Further compressed to RoutingPacket (Layer 2, ~128B) — NETWORK

    Scaling behavior:
      1 NRS:   0 network traffic (all local)
      3 NRS:   ~2.4 MB/sec routing packets on NRSIP
      10 NRS:  ~8 MB/sec
      100 NRS: ~80 MB/sec (still under 1 Gbps)
      1000 NRS: hierarchical routing, stays logarithmic
    """

    def __init__(
        self,
        total_neurons: int = 1_000_000,
        active_k: int = 1_000,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        instance_id: str = "",
        use_gpu: bool = True,
        embedding_engine: Optional[NRSEmbeddingEngine] = None,
    ):
        self.total_neurons = total_neurons
        self.active_k = min(active_k, total_neurons)
        self.embedding_dim = embedding_dim
        self.instance_id = instance_id or hashlib.sha256(
            str(time.time()).encode()
        ).hexdigest()[:12]
        self.use_gpu = use_gpu and _GPU_AVAILABLE
        self.embedder = embedding_engine or NRSEmbeddingEngine(dim=embedding_dim)

        self._neuron_embeddings: Optional[np.ndarray] = None
        self._neuron_embeddings_gpu = None
        self._domains: Dict[int, str] = {}
        self._domain_ranges: Dict[str, Tuple[int, int]] = {}
        self._registered: int = 0
        self._initialized = False

        self._activations: int = 0
        self._digests_created: int = 0
        self._packets_created: int = 0
        self._total_time_ms: float = 0.0
        self._raw_bytes_saved: int = 0

    def _initialize_embeddings(self):
        """Lazily initialize neuron embedding matrix on first use."""
        if self._initialized:
            return
        rng = np.random.RandomState(hash(self.instance_id) & 0x7FFFFFFF)
        emb = rng.randn(self.total_neurons, self.embedding_dim).astype(np.float32)
        for domain, (start, end) in self._domain_ranges.items():
            if domain in self.embedder._domain_seeds:
                centroid = self.embedder._domain_seeds[domain]
                emb[start:end] += centroid[np.newaxis, :] * 2.0
        norms = np.linalg.norm(emb, axis=1, keepdims=True).clip(1e-9)
        emb /= norms
        self._neuron_embeddings = emb
        if self.use_gpu:
            self._neuron_embeddings_gpu = _cp.asarray(emb)
        self._initialized = True

    def register_neurons(self, domain: str, count: int) -> Tuple[int, int]:
        """Register a block of neurons with a domain association."""
        start = self._registered
        end = min(start + count, self.total_neurons)
        for i in range(start, end):
            self._domains[i] = domain
        self._domain_ranges[domain] = (start, end)
        self._registered = end
        self._initialized = False
        return start, end

    def seed_domain(self, domain: str, keywords: List[str]):
        """Provide domain keywords so neurons in that domain cluster meaningfully."""
        self.embedder.register_domain_bias(domain, keywords)
        self._initialized = False

    def _query_hash(self, query: str) -> str:
        return hashlib.sha256(query.encode("utf-8")).hexdigest()

    def _embed_query(self, query: str) -> np.ndarray:
        return self.embedder.embed(query)

    def _cosine_similarity_gpu(self, query_vec: np.ndarray) -> np.ndarray:
        """GPU-accelerated cosine similarity: (1,d) · (N,d)^T → (N,)."""
        self._initialize_embeddings()
        q_gpu = _cp.asarray(query_vec.reshape(1, -1))
        scores = _cp.dot(q_gpu, self._neuron_embeddings_gpu.T).ravel()
        return _cp.asnumpy(scores)

    def _cosine_similarity_cpu(self, query_vec: np.ndarray) -> np.ndarray:
        """CPU cosine similarity: (1,d) · (N,d)^T → (N,)."""
        self._initialize_embeddings()
        scores = self._neuron_embeddings @ query_vec
        return scores

    def activate(self, query: str) -> ActivationPattern:
        """Full cosine-similarity activation. Returns Layer 0 (local)."""
        t0 = time.time()
        self._initialize_embeddings()
        qhash = self._query_hash(query)
        query_vec = self._embed_query(query)

        if self.use_gpu:
            scores = self._cosine_similarity_gpu(query_vec)
        else:
            scores = self._cosine_similarity_cpu(query_vec)

        top_indices = np.argpartition(scores, -self.active_k)[-self.active_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        active_ids = sorted(top_indices.tolist())
        similarities = {int(nid): float(scores[nid]) for nid in top_indices}

        domain_signals: Dict[str, int] = {}
        for nid in active_ids:
            domain = self._domains.get(nid, "unassigned")
            domain_signals[domain] = domain_signals.get(domain, 0) + 1

        elapsed_ms = (time.time() - t0) * 1000
        self._activations += 1
        self._total_time_ms += elapsed_ms

        return ActivationPattern(
            active_ids=active_ids,
            total_neurons=self.total_neurons,
            query_hash=qhash,
            similarities=similarities,
            domain_signals=domain_signals,
        )

    def activate_fast(self, query: str) -> ActivationPattern:
        """Same as activate() — with real embeddings, full cosine is already fast."""
        return self.activate(query)

    def activate_and_compress(self, query: str) -> ActivationDigest:
        """
        One-shot: activate → compress to Layer 1 digest.
        Typical production call. Raw pattern is discarded.
        """
        pattern = self.activate(query)
        digest = pattern.compress()
        self._digests_created += 1
        self._raw_bytes_saved += pattern.estimated_bytes - digest.estimated_bytes
        return digest

    def activate_stochastic(
        self,
        query: str,
        temperature: float = 1.0,
        top_k_multiplier: float = 1.0,
        noise_sigma: float = 0.1,
    ) -> ActivationDigest:
        """Temperature-scaled stochastic neuron activation for creative modes.

        Real 768-d cosine similarity with Gaussian noise + softmax sampling.
        Higher temperature = more diverse/novel neuron activation patterns.
        """
        t0 = time.time()
        self._initialize_embeddings()
        qhash = self._query_hash(query)
        query_vec = self._embed_query(query)

        if self.use_gpu:
            scores = self._cosine_similarity_gpu(query_vec)
        else:
            scores = self._cosine_similarity_cpu(query_vec)

        effective_k = max(1, int(self.active_k * top_k_multiplier))
        candidate_count = min(effective_k * 3, self.total_neurons)

        candidate_indices = np.argpartition(scores, -candidate_count)[-candidate_count:]
        candidate_scores = scores[candidate_indices].copy()

        noise = np.random.normal(0, noise_sigma * temperature, size=candidate_scores.shape)
        noisy_scores = np.clip(candidate_scores + noise, -1.0, 1.0)

        temp = max(0.01, temperature)
        max_s = noisy_scores.max()
        exp_scores = np.exp((noisy_scores - max_s) / temp)
        probs = exp_scores / exp_scores.sum()

        chosen = np.random.choice(
            len(candidate_indices), size=min(effective_k, len(candidate_indices)),
            replace=False, p=probs,
        )
        selected_indices = candidate_indices[chosen]

        active_ids = sorted(selected_indices.tolist())
        similarities = {int(nid): float(scores[nid]) for nid in selected_indices}

        domain_signals: Dict[str, int] = {}
        for nid in active_ids:
            domain = self._domains.get(nid, "unassigned")
            domain_signals[domain] = domain_signals.get(domain, 0) + 1

        pattern = ActivationPattern(
            active_ids=active_ids,
            total_neurons=self.total_neurons,
            query_hash=qhash,
            similarities=similarities,
            domain_signals=domain_signals,
        )
        digest = pattern.compress()

        elapsed_ms = (time.time() - t0) * 1000
        self._activations += 1
        self._digests_created += 1
        self._total_time_ms += elapsed_ms
        self._raw_bytes_saved += pattern.estimated_bytes - digest.estimated_bytes
        return digest

    def activate_for_mode(self, query: str, strategy) -> ActivationDigest:
        """Dispatch to deterministic or stochastic based on NeuronStrategy."""
        if getattr(strategy, "use_stochastic", False):
            return self.activate_stochastic(
                query,
                temperature=getattr(strategy, "temperature", 1.0),
                top_k_multiplier=getattr(strategy, "top_k_multiplier", 1.0),
                noise_sigma=getattr(strategy, "noise_sigma", 0.1),
            )
        return self.activate_and_compress(query)

    def full_pipeline(
        self,
        query: str,
        tier: str = "T1",
        confidence: float = 0.0,
    ) -> Tuple[ActivationDigest, RoutingPacket]:
        """
        Full pipeline: activate → digest (local) → packet (network).

        Returns both the digest (for local T0/lobe processing)
        and the routing packet (for NRSIP transmission).
        """
        digest = self.activate_and_compress(query)
        packet = digest.to_routing_packet(tier=tier, confidence=confidence)
        packet.source_instance_id = self.instance_id
        self._packets_created += 1
        self._raw_bytes_saved += digest.estimated_bytes - packet.estimated_bytes
        return digest, packet

    def query_similarity(self, query_a: str, query_b: str) -> float:
        """Cosine similarity between two query embeddings."""
        va = self._embed_query(query_a)
        vb = self._embed_query(query_b)
        return float(np.dot(va, vb))

    def find_similar_neurons(self, query: str, top_n: int = 20) -> List[Tuple[int, float, str]]:
        """Return top-N neurons with their similarity scores and domains."""
        self._initialize_embeddings()
        query_vec = self._embed_query(query)
        if self.use_gpu:
            scores = self._cosine_similarity_gpu(query_vec)
        else:
            scores = self._cosine_similarity_cpu(query_vec)
        top_idx = np.argpartition(scores, -top_n)[-top_n:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [(int(i), float(scores[i]), self._domains.get(i, "unassigned")) for i in top_idx]

    def embed_text(self, text: str) -> np.ndarray:
        """Expose embedding for external use (scene planner, audio feature matching)."""
        return self._embed_query(text)

    @property
    def compression_ratio(self) -> Dict[str, Any]:
        """Compression at each layer."""
        raw = LAYER_0_SIZE_BYTES
        return {
            "L0_raw_bytes": raw,
            "L1_digest_bytes": LAYER_1_SIZE_BYTES,
            "L2_packet_bytes": LAYER_2_SIZE_BYTES,
            "L0_to_L1_ratio": f"{raw / LAYER_1_SIZE_BYTES:.0f}x",
            "L0_to_L2_ratio": f"{raw / LAYER_2_SIZE_BYTES:.0f}x",
            "total_compression": f"{raw / LAYER_2_SIZE_BYTES:.0f}x",
        }

    @property
    def network_budget(self) -> Dict[str, str]:
        """Projected NRSIP traffic at production scale."""
        packet_size = LAYER_2_SIZE_BYTES
        rps = 6_433

        def fmt(mb: float) -> str:
            if mb < 1:
                return f"{mb*1024:.0f} KB/s"
            return f"{mb:.1f} MB/s"

        single = rps * packet_size / (1024 * 1024)
        return {
            "packet_size_bytes": packet_size,
            "queries_per_sec": rps,
            "1_nrs": "0 (all local)",
            "3_nrs": fmt(single * 3),
            "10_nrs": fmt(single * 10),
            "100_nrs": fmt(single * 100),
            "1gbps_capacity": "125 MB/s",
            "3_nrs_link_utilization": f"{(single * 3) / 125 * 100:.1f}%",
            "100_nrs_link_utilization": f"{(single * 100) / 125 * 100:.1f}%",
        }

    @property
    def stats(self) -> Dict[str, Any]:
        avg_ms = self._total_time_ms / self._activations if self._activations > 0 else 0
        return {
            "instance_id": self.instance_id,
            "total_neurons": self.total_neurons,
            "active_k": self.active_k,
            "target_sparsity": self.active_k / self.total_neurons if self.total_neurons > 0 else 0,
            "embedding_dim": self.embedding_dim,
            "embedding_type": "real_768d",
            "gpu_enabled": self.use_gpu,
            "registered_neurons": self._registered,
            "domains": len(set(self._domains.values())),
            "domain_seeded": len(self.embedder._domain_seeds),
            "activations": self._activations,
            "digests_created": self._digests_created,
            "packets_created": self._packets_created,
            "avg_activation_ms": round(avg_ms, 2),
            "raw_bytes_saved": self._raw_bytes_saved,
            "compression": self.compression_ratio,
        }

    def __repr__(self) -> str:
        gpu_tag = "GPU" if self.use_gpu else "CPU"
        return (
            f"BinaryNeuronBank(instance={self.instance_id[:8]}, "
            f"neurons={self.total_neurons}, "
            f"active_k={self.active_k}, "
            f"embed=768d-{gpu_tag}, "
            f"sparsity={self.active_k/self.total_neurons:.4f})"
        )
