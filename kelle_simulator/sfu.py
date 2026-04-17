"""
Specialized Function Units (SFU) — Section 3.3 of the Kelle paper.

Handles all non-linear operations that cannot be mapped to the RSA:
  • Softmax   — online max-then-exp computation (Softermax algorithm)
  • LayerNorm — mean + variance + normalise
  • Activation functions — GELU/ReLU via lookup table
  • Positional embeddings — rotary (RoPE) or learned

All operations are fully pipelined.  Cycle costs scale with the number of
elements processed; the SFU shares the 0.848 W power budget (13% of on-chip).

Timing model:
  • Softmax(N): 3 N cycles (max-pass + exp-pass + normalise-pass), pipelined to N+overhead
  • LayerNorm(N): 2 N cycles (mean/variance + normalise), pipelined
  • Activation(N): 1 cycle per element (lookup table)
  • RoPE(N): 2 cycles per element (two rotations per position)
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from .config import HardwareConfig


@dataclass
class SFUResult:
    cycles: int
    energy_pj: float
    elements: int


class SFU:
    def __init__(self, hw: HardwareConfig):
        self.hw = hw
        self.total_cycles: int = 0
        self.total_energy_pj: float = 0.0

    # ── Energy helper ─────────────────────────────────────────────────────────

    def _energy(self, cycles: int) -> float:
        return cycles * self.hw.sfu_power_w / self.hw.clock_freq_hz * 1e12

    def _record(self, cycles: int, elements: int) -> SFUResult:
        energy = self._energy(cycles)
        self.total_cycles    += cycles
        self.total_energy_pj += energy
        return SFUResult(cycles, energy, elements)

    # ── Softmax ───────────────────────────────────────────────────────────────

    def softmax(self, num_elements: int) -> SFUResult:
        """
        Online softmax (Softermax): single-pass max-tracking + normalisation.
        Pipelined: effectively N + PIPE_DEPTH cycles.
        """
        PIPE_DEPTH = 8
        cycles = num_elements + PIPE_DEPTH
        return self._record(cycles, num_elements)

    def softmax_prefill(self, seq_len: int, num_heads: int) -> SFUResult:
        """Softmax over all heads and query positions for prefill."""
        total = seq_len * seq_len * num_heads
        cycles = seq_len + 8  # each row pipelined; rows issued sequentially
        cycles *= num_heads
        return self._record(cycles, total)

    def softmax_decode(self, kv_len: int, num_heads: int) -> SFUResult:
        """Softmax over the single-query attention logits (kv_len elements per head)."""
        cycles = (kv_len + 8) * num_heads
        return self._record(cycles, kv_len * num_heads)

    # ── Layer Norm ────────────────────────────────────────────────────────────

    def layer_norm(self, num_elements: int) -> SFUResult:
        """
        Two-pass LayerNorm: mean + variance + normalise.
        Pipelined to approximately 2 cycles per element.
        """
        cycles = 2 * num_elements + 16  # 16-cycle pipeline fill
        return self._record(cycles, num_elements)

    # ── Activation functions ──────────────────────────────────────────────────

    def gelu(self, num_elements: int) -> SFUResult:
        """GELU via LUT: 1 cycle per element."""
        return self._record(num_elements, num_elements)

    def relu(self, num_elements: int) -> SFUResult:
        return self._record(math.ceil(num_elements / 4), num_elements)

    # ── Positional embeddings ─────────────────────────────────────────────────

    def rotary_embedding(self, seq_len: int, head_dim: int, num_heads: int) -> SFUResult:
        """RoPE: 2 multiply-adds per (position, dimension) pair."""
        elements = seq_len * head_dim * num_heads
        cycles = 2 * elements
        return self._record(cycles, elements)

    def learned_embedding(self, seq_len: int, d_model: int) -> SFUResult:
        """Lookup-table positional embedding: 1 cycle per element."""
        elements = seq_len * d_model
        return self._record(elements, elements)

    # ── Residual add ─────────────────────────────────────────────────────────

    def residual_add(self, num_elements: int) -> SFUResult:
        """Element-wise add: 1 cycle per 4 elements (vectorised)."""
        cycles = math.ceil(num_elements / 4)
        return self._record(cycles, num_elements)
