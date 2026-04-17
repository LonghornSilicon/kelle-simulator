"""
eDRAM Controller with 2DRP (Two-Dimensional Adaptive Refresh Policy).

The two dimensions are:
  1. Token importance: High-Score Token (HST) vs Low-Score Token (LST)
  2. Bit position:     MSBs (bits [15:8]) vs LSBs (bits [7:0])

This yields four refresh groups, each with a distinct interval:
  HST-MSB  →  360 µs   (shortest, most frequent)
  LST-MSB  → 1440 µs
  HST-LSB  → 5400 µs
  LST-LSB  → 7200 µs   (longest, least frequent)

The controller also runs a memory-access scheduler that co-ordinates refresh
with normal read/write traffic to minimise bank conflicts.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from .config import HardwareConfig
from .memory import KVCacheEDRAM, AccessResult


# ─────────────────────────────────────────────────────────────────────────────
# Refresh group enum (string constants for clarity)
# ─────────────────────────────────────────────────────────────────────────────

HST_MSB = 'hst_msb'
HST_LSB = 'hst_lsb'
LST_MSB = 'lst_msb'
LST_LSB = 'lst_lsb'

ALL_GROUPS = (HST_MSB, HST_LSB, LST_MSB, LST_LSB)


@dataclass
class RefreshEvent:
    cycle: int
    group: str
    bank_id: int
    energy_pj: float
    duration_cycles: int


# ─────────────────────────────────────────────────────────────────────────────
# Per-token refresh tracker
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TokenRefreshState:
    token_id: int
    is_hst: bool  # high-score token?
    # Cycle at which each byte-group was last refreshed
    last_refresh_cycle: Dict[str, int] = field(default_factory=lambda: {
        g: 0 for g in ALL_GROUPS
    })

    def refresh_group_for_msb(self) -> str:
        return HST_MSB if self.is_hst else LST_MSB

    def refresh_group_for_lsb(self) -> str:
        return HST_LSB if self.is_hst else LST_LSB


# ─────────────────────────────────────────────────────────────────────────────
# 2DRP Controller
# ─────────────────────────────────────────────────────────────────────────────

class EDRAMController:
    """
    Tracks refresh deadlines for every cached token and issues refresh
    operations when the simulator advances the cycle counter.

    The controller interleaves refreshes with normal eDRAM traffic by
    scheduling them in otherwise-idle cycles (the 'refresh scheduler'
    from Section 5 of the paper).
    """

    def __init__(self, hw: HardwareConfig, kv_edram: KVCacheEDRAM):
        self.hw = hw
        self.kv_edram = kv_edram

        # token_id → refresh state
        self._states: Dict[int, TokenRefreshState] = {}

        # History of all refresh events (for energy accounting)
        self.refresh_log: List[RefreshEvent] = []

        # Cumulative refresh energy and cycles
        self.total_refresh_energy_pj: float = 0.0
        self.total_refresh_cycles: int = 0

        # Bit-flip error accounting
        self.total_bit_flips_msb: int = 0
        self.total_bit_flips_lsb: int = 0

    # ── Token lifecycle ───────────────────────────────────────────────────────

    def register_token(self, token_id: int, is_hst: bool, current_cycle: int) -> None:
        self._states[token_id] = TokenRefreshState(
            token_id=token_id,
            is_hst=is_hst,
            last_refresh_cycle={g: current_cycle for g in ALL_GROUPS},
        )

    def promote_to_hst(self, token_id: int) -> None:
        if token_id in self._states:
            self._states[token_id].is_hst = True

    def demote_to_lst(self, token_id: int) -> None:
        if token_id in self._states:
            self._states[token_id].is_hst = False

    def evict_token(self, token_id: int) -> None:
        self._states.pop(token_id, None)

    # ── Cycle advance: issue pending refreshes ────────────────────────────────

    def advance_to_cycle(self, current_cycle: int) -> Tuple[int, float]:
        """
        Check all tracked tokens for overdue refreshes up to current_cycle.
        Returns (extra_stall_cycles, total_energy_pj) for this advance step.
        """
        stall = 0
        energy = 0.0

        for token_id, state in list(self._states.items()):
            for is_msb in (True, False):
                group = (state.refresh_group_for_msb() if is_msb
                         else state.refresh_group_for_lsb())
                interval = self.hw.refresh_interval_cycles(group)
                last = state.last_refresh_cycle[group]

                overdue_count = (current_cycle - last) // interval
                if overdue_count > 0:
                    # Check if data may have been corrupted before refresh
                    elapsed_s = (current_cycle - last) * self.hw.ns_per_cycle * 1e-9
                    p_flip = self.hw.bit_flip_prob(elapsed_s)
                    # Estimate bit-flips (each KV element has num_heads*head_dim bits)
                    # Using a Poisson model: expected flips ≈ p_flip × num_bits
                    # For accounting purposes, track expected flips across all layers
                    if is_msb:
                        self.total_bit_flips_msb += int(p_flip * 8)
                    else:
                        self.total_bit_flips_lsb += int(p_flip * 8)

                    # Issue the refresh(es)
                    bank_id = token_id % (self.hw.num_edram_banks // 4)
                    for _ in range(overdue_count):
                        bank = self.kv_edram.banks[bank_id]
                        r = bank.refresh()
                        state.last_refresh_cycle[group] += interval

                        evt = RefreshEvent(
                            cycle=current_cycle,
                            group=group,
                            bank_id=bank_id,
                            energy_pj=r.energy_pj,
                            duration_cycles=r.cycles,
                        )
                        self.refresh_log.append(evt)
                        energy += r.energy_pj
                        # Refresh is pipelined with compute; only stall if
                        # the bank is needed for a concurrent access (simplified:
                        # assume 10% stall probability)
                        stall += int(r.cycles * 0.10)

        self.total_refresh_energy_pj += energy
        self.total_refresh_cycles += stall
        return stall, energy

    # ── Importance reclassification ───────────────────────────────────────────

    def reclassify_tokens(self, importance_scores: Dict[int, float],
                          threshold: float) -> None:
        """
        After each decode step, reclassify tokens as HST/LST based on updated
        attention-score accumulations, adjusting their refresh intervals.
        """
        for token_id, score in importance_scores.items():
            if token_id not in self._states:
                continue
            new_hst = score >= threshold
            if new_hst != self._states[token_id].is_hst:
                if new_hst:
                    self.promote_to_hst(token_id)
                else:
                    self.demote_to_lst(token_id)

    # ── Statistics ────────────────────────────────────────────────────────────

    @property
    def num_tracked_tokens(self) -> int:
        return len(self._states)

    @property
    def hst_count(self) -> int:
        return sum(1 for s in self._states.values() if s.is_hst)

    @property
    def lst_count(self) -> int:
        return sum(1 for s in self._states.values() if not s.is_hst)

    def refresh_energy_breakdown(self) -> Dict[str, float]:
        breakdown: Dict[str, float] = {g: 0.0 for g in ALL_GROUPS}
        for evt in self.refresh_log:
            breakdown[evt.group] += evt.energy_pj
        return breakdown

    def average_refresh_interval_ms(self) -> Dict[str, float]:
        """Effective average refresh intervals weighted by token counts."""
        hst = self.hst_count
        lst = self.lst_count
        total = hst + lst if (hst + lst) > 0 else 1
        return {
            'msb': ((hst * self.hw.refresh_hst_msb_s +
                     lst * self.hw.refresh_lst_msb_s) / total) * 1e3,
            'lsb': ((hst * self.hw.refresh_hst_lsb_s +
                     lst * self.hw.refresh_lst_lsb_s) / total) * 1e3,
        }
