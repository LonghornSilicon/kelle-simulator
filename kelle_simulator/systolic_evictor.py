"""
Systolic Evictor — Section 6 of the Kelle paper.

A lightweight hardware unit (0.06 mm², 0.028 W) attached to the RSA that:
  1. Accumulates per-token attention scores across layers and heads.
  2. Maintains a sorted view so the minimum-importance token is always known.
  3. Identifies which tokens are candidates for full eviction vs. recomputation
     (input-vector-only storage).

It operates in parallel with the RSA, adding 5-7% latency overhead only
when a new minimum-search is forced by an eviction trigger.

Internal structure (from the paper):
  • A column of registers holding current importance scores (one per cached token)
  • A register chain that propagates the running minimum from top to bottom
  • A comparator tree that updates the minimum in O(log N') cycles
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from .config import HardwareConfig, ModelConfig


@dataclass
class TokenImportance:
    token_id: int
    score: float           # accumulated attention-score sum across all heads/layers
    head_counts: int       # number of heads that considered this token "important"
    num_heads_total: int
    is_initial_token: bool = False
    is_recent_token: bool = False

    @property
    def importance_ratio(self) -> float:
        """Fraction of attention heads that scored this token above mean."""
        return self.head_counts / max(self.num_heads_total, 1)


class SystolicEvictor:
    """
    Tracks importance scores for up to N' cached tokens.

    Cycle costs:
      • Score update (after each attention computation): 1 cycle per token
        (pipelined alongside RSA softmax output)
      • Min-search for eviction candidate: ceil(log2(N')) cycles
        (comparator tree through the register chain)
    """

    def __init__(self, hw: HardwareConfig, model: ModelConfig):
        self.hw = hw
        self.model = model
        self._scores: Dict[int, TokenImportance] = {}
        self._step: int = 0

        # Cycle and energy accounting for the evictor itself
        self.total_evictor_cycles: int = 0
        self.total_evictor_energy_pj: float = 0.0

    # ── Token management ──────────────────────────────────────────────────────

    def register_token(self, token_id: int, position: int,
                       num_cached: int, recent_window: int,
                       initial_preserved: int) -> None:
        # Cap recent_window so at least half the cache is evictable
        effective_recent = min(recent_window, max(1, num_cached // 2))
        is_initial = (position < initial_preserved)
        is_recent  = (position >= num_cached - effective_recent)
        self._scores[token_id] = TokenImportance(
            token_id=token_id,
            score=0.0,
            head_counts=0,
            num_heads_total=self.model.num_heads * self.model.num_layers,
            is_initial_token=is_initial,
            is_recent_token=is_recent,
        )

    def remove_token(self, token_id: int) -> None:
        self._scores.pop(token_id, None)

    # ── Attention score accumulation ──────────────────────────────────────────

    def accumulate_scores(self, attention_weights: Dict[int, float],
                          layer: int, head: int) -> int:
        """
        Called after each attention head's softmax output.
        attention_weights: {token_id → attention weight for this query}.

        Returns the number of cycles consumed (1 cycle per token, pipelined).
        """
        if not self._scores:
            return 0

        n_tokens = len(attention_weights)
        mean_weight = 1.0 / n_tokens if n_tokens > 0 else 0.0

        for token_id, weight in attention_weights.items():
            if token_id in self._scores:
                self._scores[token_id].score += weight
                if weight > mean_weight:
                    self._scores[token_id].head_counts += 1

        # Pipelined: 1 cycle for the entire update (parallel comparisons)
        cycles = 1
        energy = cycles * self.hw.systolic_evictor_power_w / self.hw.clock_freq_hz * 1e12
        self.total_evictor_cycles += cycles
        self.total_evictor_energy_pj += energy
        return cycles

    # ── Eviction candidate search ─────────────────────────────────────────────

    def find_eviction_candidate(self, cached_token_ids=None) -> Tuple[Optional[int], int]:
        """
        Identify the token with the lowest importance score that is neither
        an initial token nor a recent token (those are always preserved).

        Protection status is recomputed dynamically from insertion order so
        that as new tokens arrive, old tokens correctly lose "recent" status.

        Returns (token_id_to_evict, cycles_consumed).
        Cycles = ceil(log2(N')) for the comparator-tree min-search.
        """
        n = len(self._scores)
        if n == 0:
            return None, 0

        search_cycles = max(1, math.ceil(math.log2(n + 1)))
        energy = search_cycles * self.hw.systolic_evictor_power_w / self.hw.clock_freq_hz * 1e12
        self.total_evictor_cycles += search_cycles
        self.total_evictor_energy_pj += energy

        # Recompute protection status based on current insertion order
        tokens_list = list(self._scores.keys())  # ordered by insertion
        num_cached = len(tokens_list)
        initial_preserved = self.hw.initial_tokens_preserved
        # At least half the cache must be evictable
        effective_recent = min(self.hw.recent_window_tokens, max(1, num_cached // 2))
        recent_threshold = num_cached - effective_recent

        candidates = []
        for i, tid in enumerate(tokens_list):
            is_initial = i < initial_preserved
            is_recent  = i >= recent_threshold
            if not is_initial and not is_recent:
                if cached_token_ids is None or tid in cached_token_ids:
                    candidates.append(self._scores[tid])

        if not candidates:
            return None, search_cycles

        victim = min(candidates, key=lambda t: t.score)
        return victim.token_id, search_cycles

    # ── Recomputation decision ────────────────────────────────────────────────

    def should_recompute(self, token_id: int) -> bool:
        """
        Return True if the token is important enough to warrant storing its
        input vector for on-demand KV recomputation (≥50% of heads found it
        above the mean — Section 4.2 of the paper).
        """
        tok = self._scores.get(token_id)
        if tok is None:
            return False
        return tok.importance_ratio >= self.hw.recompute_head_threshold

    # ── HST/LST classification for 2DRP ──────────────────────────────────────

    def get_hst_threshold(self) -> float:
        """Median importance score — tokens above are HST, below are LST."""
        scores = [t.score for t in self._scores.values()]
        if not scores:
            return 0.0
        scores.sort()
        mid = len(scores) // 2
        return scores[mid]

    def get_importance_map(self) -> Dict[int, float]:
        return {tid: t.score for tid, t in self._scores.items()}

    def is_hst(self, token_id: int) -> bool:
        threshold = self.get_hst_threshold()
        tok = self._scores.get(token_id)
        return tok is not None and tok.score >= threshold

    # ── Step bookkeeping ──────────────────────────────────────────────────────

    def update_recency(self, num_cached: int, recent_window: int,
                       initial_preserved: int) -> None:
        """Refresh is_recent / is_initial flags after a new token is added."""
        for tid, tok in self._scores.items():
            pos = list(self._scores.keys()).index(tid)
            tok.is_initial_token = pos < initial_preserved
            tok.is_recent_token  = pos >= (num_cached - recent_window)

    @property
    def num_tracked(self) -> int:
        return len(self._scores)
