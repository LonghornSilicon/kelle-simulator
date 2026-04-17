"""
AERP — Attention-based Eviction and Recomputation Policy (Section 4 of Kelle paper).

Key insight: instead of dropping evicted tokens entirely, Kelle stores their
smaller *input vectors* (d_model × 2 bytes) rather than the full KV pairs
(2 × num_heads × head_dim × num_layers bytes).  When a token is needed again,
KV is recomputed on the fly using the RSA.

Three classes of tokens in the cache:
  CACHED     — full KV stored in eDRAM  (preferred path)
  RECOMPUTE  — only input vector stored; KV recomputed on access
  EVICTED    — fully discarded (used for truly unimportant tokens)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
from .config import HardwareConfig, ModelConfig
from .systolic_evictor import SystolicEvictor


class TokenStatus(Enum):
    CACHED    = auto()
    RECOMPUTE = auto()
    EVICTED   = auto()


@dataclass
class CachedToken:
    token_id: int
    position: int
    status: TokenStatus
    kv_bytes: int        # bytes occupied in eDRAM (0 if RECOMPUTE/EVICTED)
    input_bytes: int     # bytes for stored input vector (only if RECOMPUTE)
    importance_score: float = 0.0

    @property
    def storage_bytes(self) -> int:
        if self.status == TokenStatus.CACHED:
            return self.kv_bytes
        elif self.status == TokenStatus.RECOMPUTE:
            return self.input_bytes
        return 0


@dataclass
class AERPStats:
    eviction_events: int = 0
    recompute_events: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    kv_bytes_saved_by_recompute: int = 0  # KV bytes not stored due to recompute path
    eviction_log: List[Tuple[int, int, str]] = field(default_factory=list)
    # (step, token_id, "evicted" | "recompute")


class AERP:
    """
    Manages the KV-cache token set with AERP eviction and recomputation.

    On every decode step:
      1. A new token is generated and its KV pair must be added.
      2. If the cache is at capacity N', the systolic evictor identifies the
         lowest-importance token.
      3. If that token's importance_ratio >= threshold → store input for recompute.
         Otherwise → full eviction.
      4. On the next time a RECOMPUTE token's KV is needed, a recompute event
         is triggered (adds RSA cycles).
    """

    def __init__(self, hw: HardwareConfig, model: ModelConfig,
                 evictor: SystolicEvictor):
        self.hw = hw
        self.model = model
        self.evictor = evictor

        self._tokens: Dict[int, CachedToken] = {}
        self._capacity = hw.kv_cache_capacity_tokens
        self.stats = AERPStats()

    # ── KV per-token sizes ────────────────────────────────────────────────────
    # The 4 MB eDRAM is a streaming per-layer buffer: it holds one transformer
    # layer's worth of KV vectors at a time.  Capacity and byte-budget checks
    # are therefore based on per-layer bytes, not all-layers bytes.

    def _kv_bytes(self) -> int:
        """Bytes for one token's KV in a single layer (eDRAM working-set unit)."""
        return self.model.kv_bytes_per_token_per_layer

    def _input_bytes(self) -> int:
        return self.model.input_bytes_per_token  # d_model × 2 bytes FP16

    # ── Cache access ──────────────────────────────────────────────────────────

    def access_token(self, token_id: int, step: int
                     ) -> Tuple[TokenStatus, int]:
        """
        Return (status, extra_rsa_cycles) for accessing token_id's KV cache.
        extra_rsa_cycles is non-zero only for RECOMPUTE tokens.
        """
        tok = self._tokens.get(token_id)
        if tok is None or tok.status == TokenStatus.EVICTED:
            self.stats.cache_misses += 1
            return TokenStatus.EVICTED, 0

        if tok.status == TokenStatus.CACHED:
            self.stats.cache_hits += 1
            return TokenStatus.CACHED, 0

        # RECOMPUTE path: trigger RSA recomputation
        self.stats.cache_hits += 1  # input vector is present
        self.stats.recompute_events += 1
        # Recompute cycles: single token through QKV projection per layer
        # = num_layers × matmul([1, d_model] × [d_model, 2×d_model])
        m = self.model
        recompute_cycles = (m.num_layers *
                            _matmul_cycles(1, m.d_model, 2 * m.d_model,
                                           self.hw.rsa_rows, self.hw.rsa_cols))
        return TokenStatus.RECOMPUTE, recompute_cycles

    # ── eDRAM byte budget ─────────────────────────────────────────────────────

    @property
    def _recompute_byte_budget(self) -> int:
        """Bytes available for input-vector storage after reserving N' KV slots."""
        return self.hw.edram_kvcache_bytes - self._capacity * self._kv_bytes()

    # ── Token insertion ───────────────────────────────────────────────────────

    def insert_token(self, token_id: int, position: int, step: int
                     ) -> Tuple[Optional[int], str, int]:
        """
        Insert a new token into the cache.

        Two-tier capacity management:
          1. KV capacity (N'): num_cached (CACHED-status only) must stay <= N'.
             Evict one CACHED token when full — either to RECOMPUTE or full eviction.
          2. eDRAM byte budget: RECOMPUTE tokens use input vectors (much smaller).
             When total input-vector bytes exceed the remaining eDRAM budget,
             fully evict the lowest-importance RECOMPUTE token.

        Returns (evicted_token_id, action, eviction_search_cycles).
        action in {"none", "evicted", "recompute", "recompute_overflow"}.
        """
        evicted_id: Optional[int] = None
        action = "none"
        search_cycles = 0

        # Tier 1: KV-pair capacity check (counts CACHED tokens only)
        if self.num_cached >= self._capacity:
            cached_ids = {tid for tid, t in self._tokens.items()
                         if t.status == TokenStatus.CACHED}
            evicted_id, search_cycles = self.evictor.find_eviction_candidate(cached_ids)
            if evicted_id is not None:
                action = self._evict_or_recompute(evicted_id, step)

        # Tier 2: eDRAM byte budget — purge one RECOMPUTE token if overflow
        recompute_bytes = sum(
            t.input_bytes for t in self._tokens.values()
            if t.status == TokenStatus.RECOMPUTE
        )
        if recompute_bytes > self._recompute_byte_budget:
            overflow_id = self._evict_cheapest_recompute(step)
            if overflow_id is not None and evicted_id is None:
                evicted_id = overflow_id
                action = "recompute_overflow"

        kv_b = self._kv_bytes()
        self._tokens[token_id] = CachedToken(
            token_id=token_id,
            position=position,
            status=TokenStatus.CACHED,
            kv_bytes=kv_b,
            input_bytes=self._input_bytes(),
        )
        # Evictor registration is handled by the simulator so that
        # num_cached and is_recent reflect the final cache state, not the
        # incremental insertion order during prefill.
        return evicted_id, action, search_cycles

    def _evict_cheapest_recompute(self, step: int) -> Optional[int]:
        """Fully evict the lowest-importance RECOMPUTE token (byte-budget overflow)."""
        candidates = [
            t for t in self._tokens.values()
            if t.status == TokenStatus.RECOMPUTE
        ]
        if not candidates:
            return None
        victim = min(candidates, key=lambda t: t.importance_score)
        self._tokens.pop(victim.token_id, None)
        self.evictor.remove_token(victim.token_id)
        self.stats.eviction_events += 1
        self.stats.eviction_log.append((step, victim.token_id, "evicted_overflow"))
        return victim.token_id

    def _evict_or_recompute(self, token_id: int, step: int) -> str:
        tok = self._tokens.get(token_id)
        if tok is None:
            return "none"

        if self.evictor.should_recompute(token_id):
            # Demote: keep input vector, drop KV
            saved = tok.kv_bytes
            tok.status = TokenStatus.RECOMPUTE
            tok.kv_bytes = 0
            self.stats.recompute_events += 1
            self.stats.kv_bytes_saved_by_recompute += saved
            self.stats.eviction_log.append((step, token_id, "recompute"))
            return "recompute"
        else:
            # Full eviction
            self._tokens.pop(token_id, None)
            self.evictor.remove_token(token_id)
            self.stats.eviction_events += 1
            self.stats.eviction_log.append((step, token_id, "evicted"))
            return "evicted"

    # ── Importance update ─────────────────────────────────────────────────────

    def update_importance(self, attention_weights: Dict[int, float]) -> None:
        for tid, score in attention_weights.items():
            if tid in self._tokens:
                self._tokens[tid].importance_score = score

    # ── Statistics helpers ────────────────────────────────────────────────────

    @property
    def num_cached(self) -> int:
        return sum(1 for t in self._tokens.values()
                   if t.status == TokenStatus.CACHED)

    @property
    def num_recompute(self) -> int:
        return sum(1 for t in self._tokens.values()
                   if t.status == TokenStatus.RECOMPUTE)

    @property
    def total_tokens(self) -> int:
        return len(self._tokens)

    @property
    def edram_bytes_used(self) -> int:
        return sum(t.storage_bytes for t in self._tokens.values())

    @property
    def cache_hit_rate(self) -> float:
        total = self.stats.cache_hits + self.stats.cache_misses
        return self.stats.cache_hits / total if total > 0 else 1.0

    def occupancy_snapshot(self) -> Dict[str, int]:
        return {
            "cached": self.num_cached,
            "recompute": self.num_recompute,
            "total": self.total_tokens,
            "edram_bytes": self.edram_bytes_used,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper (avoids circular import with rsa.py)
# ─────────────────────────────────────────────────────────────────────────────

def _matmul_cycles(M: int, K: int, N: int, rsa_rows: int, rsa_cols: int) -> int:
    """Throughput model: ceil(M*K*N / (rsa_rows*rsa_cols)) compute cycles."""
    import math
    macs = M * K * N
    return max(1, math.ceil(macs / (rsa_rows * rsa_cols)))
