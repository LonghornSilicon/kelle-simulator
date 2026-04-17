"""
Memory subsystem: 2 MB weight SRAM, 4 MB KV-cache eDRAM, 256 KB activation eDRAM,
16 GB off-chip DRAM.  All access methods return (cycles, energy_pJ).
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional
from .config import HardwareConfig, ModelConfig


# ─────────────────────────────────────────────────────────────────────────────
# Memory access result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AccessResult:
    cycles: int
    energy_pj: float
    bytes_transferred: int


def _access(bytes_count: int,
            bandwidth_bytes_per_cycle: float,
            latency_cycles: int,
            energy_pj_per_byte: float) -> AccessResult:
    transfer_cycles = math.ceil(bytes_count / bandwidth_bytes_per_cycle)
    total_cycles = max(latency_cycles, transfer_cycles)
    energy = bytes_count * energy_pj_per_byte
    return AccessResult(total_cycles, energy, bytes_count)


# ─────────────────────────────────────────────────────────────────────────────
# SRAM: 2 MB weight cache
# ─────────────────────────────────────────────────────────────────────────────

class WeightSRAM:
    """Stores INT8 model weights for on-chip layers."""

    def __init__(self, hw: HardwareConfig, model: ModelConfig):
        self.hw = hw
        self.model = model
        self._used_bytes: int = 0
        self._stats_reads: int = 0
        self._stats_writes: int = 0

    def read(self, num_bytes: int) -> AccessResult:
        self._stats_reads += 1
        return _access(num_bytes,
                       self.hw.sram_bytes_per_cycle,
                       self.hw.sram_latency_cycles,
                       self.hw.sram_energy_pj_per_byte)

    def write(self, num_bytes: int) -> AccessResult:
        self._stats_writes += 1
        clamped = min(self._used_bytes + num_bytes, self.hw.sram_size_bytes)
        self._used_bytes = clamped
        return _access(num_bytes,
                       self.hw.sram_bytes_per_cycle,
                       self.hw.sram_latency_cycles,
                       self.hw.sram_energy_pj_per_byte)

    @property
    def utilisation(self) -> float:
        return self._used_bytes / self.hw.sram_size_bytes

    def can_fit_layer(self, layer_idx: int) -> bool:
        """Check if a transformer layer's weights fit in remaining SRAM."""
        m = self.model
        # QKV projection: 3 × d_model × d_model
        # Output proj: d_model × d_model
        # FFN: d_model × d_ffn + d_ffn × d_model
        layer_w = (3 * m.d_model * m.d_model +
                   m.d_model * m.d_model +
                   m.d_model * m.d_ffn +
                   m.d_ffn * m.d_model) * (m.weight_bits // 8)
        return (self._used_bytes + layer_w) <= self.hw.sram_size_bytes


# ─────────────────────────────────────────────────────────────────────────────
# eDRAM bank — one of 32 banks in the KV-cache array
# ─────────────────────────────────────────────────────────────────────────────

class EDRAMBank:
    """
    One bank of the KV-cache eDRAM.  The 32 banks are split into four groups:
      banks  0-7  → Key MSBs (bits [15:8])
      banks  8-15 → Key LSBs (bits  [7:0])
      banks 16-23 → Value MSBs
      banks 24-31 → Value LSBs
    """

    def __init__(self, bank_id: int, hw: HardwareConfig):
        self.bank_id = bank_id
        self.hw = hw
        self.capacity_bytes = hw.edram_kvcache_bytes // hw.num_edram_banks

        # Map token_id → bytes stored in this bank
        self._tokens: dict[int, int] = {}

    @property
    def group(self) -> str:
        b = self.bank_id
        if b < 8:
            return 'key_msb'
        elif b < 16:
            return 'key_lsb'
        elif b < 24:
            return 'val_msb'
        else:
            return 'val_lsb'

    @property
    def used_bytes(self) -> int:
        return sum(self._tokens.values())

    @property
    def free_bytes(self) -> int:
        return self.capacity_bytes - self.used_bytes

    def store_token(self, token_id: int, num_bytes: int) -> AccessResult:
        self._tokens[token_id] = num_bytes
        return _access(num_bytes,
                       self.hw.edram_bytes_per_cycle,
                       self.hw.edram_latency_cycles,
                       self.hw.edram_energy_pj_per_byte)

    def load_token(self, token_id: int) -> AccessResult:
        num_bytes = self._tokens.get(token_id, 0)
        return _access(max(num_bytes, 1),
                       self.hw.edram_bytes_per_cycle,
                       self.hw.edram_latency_cycles,
                       self.hw.edram_energy_pj_per_byte)

    def evict_token(self, token_id: int) -> None:
        self._tokens.pop(token_id, None)

    def refresh(self) -> AccessResult:
        """Simulate a full-bank refresh (read + rewrite entire bank content)."""
        refresh_bytes = self.used_bytes if self.used_bytes > 0 else 64
        energy = refresh_bytes * self.hw.edram_energy_pj_per_byte * 2  # read+write
        cycles = math.ceil(refresh_bytes * 2 / self.hw.edram_bytes_per_cycle)
        return AccessResult(cycles, energy, refresh_bytes * 2)


# ─────────────────────────────────────────────────────────────────────────────
# KV-cache eDRAM array (32 banks)
# ─────────────────────────────────────────────────────────────────────────────

class KVCacheEDRAM:
    """4 MB KV-cache eDRAM.  Splits each token's KV into 4 byte-groups for 2DRP."""

    def __init__(self, hw: HardwareConfig, model: ModelConfig):
        self.hw = hw
        self.model = model
        self.banks = [EDRAMBank(i, hw) for i in range(hw.num_edram_banks)]
        # token_id → layer → stored bytes per bank-group
        self._token_layers: dict[int, set[int]] = {}

    # Bytes per token per layer for each half-word group
    def _group_bytes(self, layer: int) -> int:
        m = self.model
        # One group = one half (MSB or LSB) of (K or V), single layer
        return m.num_heads * m.head_dim  # e.g. 12×64 = 768 bytes per group per layer

    def _banks_for_group(self, group: str):
        mapping = {
            'key_msb': self.banks[0:8],
            'key_lsb': self.banks[8:16],
            'val_msb': self.banks[16:24],
            'val_lsb': self.banks[24:32],
        }
        return mapping[group]

    def store_kv(self, token_id: int, layer: int) -> AccessResult:
        """Store one token's KV pair for a single layer across the 4 bank groups."""
        m = self.model
        per_group = self._group_bytes(layer)
        total_cycles, total_energy, total_bytes = 0, 0.0, 0

        for group in ('key_msb', 'key_lsb', 'val_msb', 'val_lsb'):
            # Stripe across the 8 banks in this group
            bank_group = self._banks_for_group(group)
            bank_idx = token_id % len(bank_group)
            r = bank_group[bank_idx].store_token(token_id, per_group)
            total_cycles = max(total_cycles, r.cycles)  # parallel banks
            total_energy += r.energy_pj
            total_bytes  += r.bytes_transferred

        self._token_layers.setdefault(token_id, set()).add(layer)
        return AccessResult(total_cycles, total_energy, total_bytes)

    def load_kv(self, token_id: int, layer: int) -> AccessResult:
        """Load one token's KV pair for a single layer from the 4 bank groups."""
        per_group = self._group_bytes(layer)
        total_cycles, total_energy, total_bytes = 0, 0.0, 0

        for group in ('key_msb', 'key_lsb', 'val_msb', 'val_lsb'):
            bank_group = self._banks_for_group(group)
            bank_idx = token_id % len(bank_group)
            r = bank_group[bank_idx].load_token(token_id)
            total_cycles = max(total_cycles, r.cycles)
            total_energy += r.energy_pj
            total_bytes  += r.bytes_transferred

        return AccessResult(total_cycles, total_energy, total_bytes)

    def evict_token(self, token_id: int) -> None:
        for group in ('key_msb', 'key_lsb', 'val_msb', 'val_lsb'):
            bank_group = self._banks_for_group(group)
            bank_idx = token_id % len(bank_group)
            bank_group[bank_idx].evict_token(token_id)
        self._token_layers.pop(token_id, None)

    @property
    def num_tokens_cached(self) -> int:
        return len(self._token_layers)

    @property
    def used_bytes(self) -> int:
        return sum(b.used_bytes for b in self.banks)

    @property
    def utilisation(self) -> float:
        return self.used_bytes / self.hw.edram_kvcache_bytes


# ─────────────────────────────────────────────────────────────────────────────
# Activation eDRAM (256 KB)
# ─────────────────────────────────────────────────────────────────────────────

class ActivationEDRAM:
    """256 KB scratchpad for intermediate activations during compute."""

    def __init__(self, hw: HardwareConfig):
        self.hw = hw

    def read(self, num_bytes: int) -> AccessResult:
        return _access(num_bytes,
                       self.hw.edram_bytes_per_cycle,
                       self.hw.edram_latency_cycles,
                       self.hw.edram_energy_pj_per_byte)

    def write(self, num_bytes: int) -> AccessResult:
        return _access(num_bytes,
                       self.hw.edram_bytes_per_cycle,
                       self.hw.edram_latency_cycles,
                       self.hw.edram_energy_pj_per_byte)


# ─────────────────────────────────────────────────────────────────────────────
# Off-chip DRAM
# ─────────────────────────────────────────────────────────────────────────────

class OffChipDRAM:
    """16 GB LPDDR at 64 GB/s — used for model layers that overflow SRAM."""

    def __init__(self, hw: HardwareConfig):
        self.hw = hw
        self._total_bytes_transferred: int = 0

    def read(self, num_bytes: int) -> AccessResult:
        self._total_bytes_transferred += num_bytes
        return _access(num_bytes,
                       self.hw.dram_bytes_per_cycle,
                       self.hw.dram_latency_cycles,
                       self.hw.dram_energy_pj_per_byte)

    def write(self, num_bytes: int) -> AccessResult:
        self._total_bytes_transferred += num_bytes
        return _access(num_bytes,
                       self.hw.dram_bytes_per_cycle,
                       self.hw.dram_latency_cycles,
                       self.hw.dram_energy_pj_per_byte)

    @property
    def total_bytes_transferred(self) -> int:
        return self._total_bytes_transferred


# ─────────────────────────────────────────────────────────────────────────────
# Unified memory hierarchy
# ─────────────────────────────────────────────────────────────────────────────

class MemoryHierarchy:
    def __init__(self, model: ModelConfig, hw: HardwareConfig):
        self.model = model
        self.hw = hw
        self.weight_sram      = WeightSRAM(hw, model)
        self.kv_edram         = KVCacheEDRAM(hw, model)
        self.activation_edram = ActivationEDRAM(hw)
        self.dram             = OffChipDRAM(hw)

        # Determine which layers fit in SRAM
        self._sram_resident_layers: set[int] = set()
        self._init_weight_placement()

    def _init_weight_placement(self):
        """Greedily pack transformer layers into SRAM; overflow goes to DRAM."""
        m = self.model
        bytes_per_layer = (
            3 * m.d_model * m.d_model +   # QKV proj
            m.d_model * m.d_model +         # O proj
            m.d_model * m.d_ffn +           # FFN up
            m.d_ffn * m.d_model             # FFN down
        ) * (m.weight_bits // 8)

        budget = self.hw.sram_size_bytes
        for layer in range(m.num_layers):
            if bytes_per_layer <= budget:
                self._sram_resident_layers.add(layer)
                budget -= bytes_per_layer

    def load_weights(self, layer: int, num_bytes: int) -> AccessResult:
        if layer in self._sram_resident_layers:
            return self.weight_sram.read(num_bytes)
        return self.dram.read(num_bytes)

    @property
    def sram_resident_layers(self) -> set[int]:
        return self._sram_resident_layers

    @property
    def num_dram_layers(self) -> int:
        return self.model.num_layers - len(self._sram_resident_layers)
