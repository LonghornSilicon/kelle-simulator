"""
Reconfigurable Systolic Array (RSA) — Section 3.2 of the Kelle paper.

Specs:
  • 32 × 32 PE array, each PE performs 8-bit MAC
  • Weight-stationary dataflow
  • 1 GHz clock → 1024 MACs/cycle → 2.048 TOPS (×2 for FMAC = 4.096 ≈ 4.13 TOPS)
  • Interfaces with SRAM (weights) and activation eDRAM (activations)

Cycle-accurate timing model:
  For matmul C = A × B  where A∈[M,K], B∈[K,N]:
    • Tile size: rsa_rows × rsa_cols output elements computed in K cycles
    • Num tiles: ceil(M/rsa_rows) × ceil(N/rsa_cols)
    • Total compute cycles = num_tiles × K  (weight-stationary: reload every tile)
    • Pipeline fill/drain overhead: rsa_rows + rsa_cols - 2 per tile (amortised for
      large matrices, kept explicit for small decode-phase vectors)
    • Memory stall cycles: weight load from SRAM/DRAM + activation load from eDRAM

All methods return ComputeResult(cycles, energy_pJ, ops).
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple
from .config import HardwareConfig, ModelConfig
from .memory import MemoryHierarchy, AccessResult


@dataclass
class ComputeResult:
    cycles: int
    rsa_energy_pj: float       # RSA array compute energy only
    weight_energy_pj: float    # energy to load weight matrix (SRAM or DRAM)
    act_energy_pj: float       # activation eDRAM access energy
    ops: int                   # number of MAC operations
    memory_stall_cycles: int = 0
    weight_bytes: int = 0
    activation_bytes: int = 0
    weight_from_dram: bool = False

    @property
    def energy_pj(self) -> float:
        return self.rsa_energy_pj + self.weight_energy_pj + self.act_energy_pj

    @property
    def total_cycles(self) -> int:
        return self.cycles + self.memory_stall_cycles

    @property
    def compute_utilisation(self) -> float:
        peak_ops = self.cycles * (32 * 32)
        return self.ops / peak_ops if peak_ops > 0 else 0.0


class RSA:
    """
    Models the 32×32 reconfigurable systolic array.

    The array supports two modes:
      PREFILL  — process an entire sequence in one shot (large M)
      DECODE   — process a single new token (M=1, but N and K are full dim)
    """

    def __init__(self, hw: HardwareConfig, model: ModelConfig,
                 memory: MemoryHierarchy):
        self.hw = hw
        self.model = model
        self.memory = memory

        self.total_compute_cycles: int = 0
        self.total_energy_pj: float = 0.0
        self.total_ops: int = 0
        self.total_memory_stall_cycles: int = 0

    # ── Core timing model ─────────────────────────────────────────────────────

    def _compute_cycles(self, M: int, K: int, N: int) -> int:
        """
        Weight-stationary systolic array timing.

        For each output tile [rsa_rows × rsa_cols]:
          • K accumulation cycles (pipeline depth = K)
          • Plus rsa_rows-1 fill cycles (first input arrives after rsa_rows-1 hops)
          • Amortised drain: overlap with next tile
        Simplified to: ceil(M/R) × ceil(N/C) × (K + R - 1)
        where R=rsa_rows, C=rsa_cols.
        """
        R, C = self.hw.rsa_rows, self.hw.rsa_cols
        tile_m = math.ceil(M / R)
        tile_n = math.ceil(N / C)
        cycles_per_tile = K + R - 1
        return tile_m * tile_n * cycles_per_tile

    def _energy_pj(self, ops: int) -> float:
        """RSA energy = ops × energy_per_op.  Derived from power / throughput."""
        # peak_ops_per_sec = rsa_rows × rsa_cols × 2 (MAC=2 ops) × clock_hz
        peak_ops_per_s = self.hw.rsa_rows * self.hw.rsa_cols * 2 * self.hw.clock_freq_hz
        energy_per_op_pj = (self.hw.rsa_power_w / peak_ops_per_s) * 1e12
        return ops * energy_per_op_pj

    # ── Matrix-multiply interface ─────────────────────────────────────────────

    def matmul(self, M: int, K: int, N: int,
               layer: int,
               weight_bytes: int,
               activation_bytes: int) -> ComputeResult:
        """
        Simulate one matrix multiplication [M,K] × [K,N].

        Args:
          layer:            transformer layer index (determines weight location)
          weight_bytes:     bytes to load for the weight matrix
          activation_bytes: bytes to load/store for activations
        """
        ops = M * K * N * 2  # multiply + accumulate per element

        # Compute cycles
        compute_c = self._compute_cycles(M, K, N)

        # Memory stall: weight fetch (SRAM or DRAM) overlapped with compute
        from_dram = layer not in self.memory.sram_resident_layers
        w_result  = self.memory.load_weights(layer, weight_bytes)
        act_result = self.memory.activation_edram.read(activation_bytes)

        weight_stall = max(0, w_result.cycles - compute_c)

        rsa_e = self._energy_pj(ops)
        result = ComputeResult(
            cycles=compute_c,
            rsa_energy_pj=rsa_e,
            weight_energy_pj=w_result.energy_pj,
            act_energy_pj=act_result.energy_pj,
            ops=ops,
            memory_stall_cycles=weight_stall,
            weight_bytes=weight_bytes,
            activation_bytes=activation_bytes,
            weight_from_dram=from_dram,
        )

        self.total_compute_cycles    += result.total_cycles
        self.total_energy_pj         += result.energy_pj
        self.total_ops               += ops
        self.total_memory_stall_cycles += weight_stall

        return result

    # ── Prefill-phase operations ───────────────────────────────────────────────

    def qkv_projection_prefill(self, seq_len: int, layer: int) -> ComputeResult:
        """Q, K, V projections for the full prompt (fused into one matmul pass)."""
        m = self.model
        M, K, N = seq_len, m.d_model, 3 * m.d_model
        w_bytes  = m.weight_bytes(m.d_model, 3 * m.d_model)
        act_bytes = seq_len * m.d_model * 2  # FP16 input activations
        return self.matmul(M, K, N, layer, w_bytes, act_bytes)

    def attention_score_prefill(self, seq_len: int, layer: int) -> ComputeResult:
        """QK^T → attention logits: [S, head_dim] × [head_dim, S] per head, all heads."""
        m = self.model
        # Batched over all heads: effective M=S, K=head_dim, N=S, repeated num_heads times
        M, K, N = seq_len, m.head_dim, seq_len
        ops_total = M * K * N * 2 * m.num_heads
        compute_c = self._compute_cycles(M, K, N) * m.num_heads
        act_bytes = seq_len * m.head_dim * 2 * m.num_heads * 2  # Q and K
        act_e = self.memory.activation_edram.read(act_bytes).energy_pj
        rsa_e = self._energy_pj(ops_total)

        result = ComputeResult(compute_c, rsa_energy_pj=rsa_e,
                               weight_energy_pj=0.0, act_energy_pj=act_e,
                               ops=ops_total, activation_bytes=act_bytes)
        self.total_compute_cycles += compute_c
        self.total_energy_pj      += result.energy_pj
        self.total_ops            += ops_total
        return result

    def attention_output_prefill(self, seq_len: int, layer: int) -> ComputeResult:
        """scores × V: [S, S] × [S, head_dim] per head."""
        m = self.model
        M, K, N = seq_len, seq_len, m.head_dim
        ops_total = M * K * N * 2 * m.num_heads
        compute_c = self._compute_cycles(M, K, N) * m.num_heads
        act_bytes = seq_len * seq_len * 2 * m.num_heads  # attention weights
        act_e = self.memory.activation_edram.read(act_bytes).energy_pj
        rsa_e = self._energy_pj(ops_total)

        result = ComputeResult(compute_c, rsa_energy_pj=rsa_e,
                               weight_energy_pj=0.0, act_energy_pj=act_e,
                               ops=ops_total, activation_bytes=act_bytes)
        self.total_compute_cycles += compute_c
        self.total_energy_pj      += result.energy_pj
        self.total_ops            += ops_total
        return result

    def output_projection_prefill(self, seq_len: int, layer: int) -> ComputeResult:
        m = self.model
        w_bytes  = m.weight_bytes(m.d_model, m.d_model)
        act_bytes = seq_len * m.d_model * 2
        return self.matmul(seq_len, m.d_model, m.d_model, layer, w_bytes, act_bytes)

    def ffn_prefill(self, seq_len: int, layer: int) -> Tuple[ComputeResult, ComputeResult]:
        """FFN up-projection and down-projection."""
        m = self.model
        up_w   = m.weight_bytes(m.d_model, m.d_ffn)
        down_w = m.weight_bytes(m.d_ffn, m.d_model)
        up   = self.matmul(seq_len, m.d_model, m.d_ffn, layer, up_w,   seq_len * m.d_model * 2)
        down = self.matmul(seq_len, m.d_ffn,   m.d_model, layer, down_w, seq_len * m.d_ffn * 2)
        return up, down

    # ── Decode-phase operations ────────────────────────────────────────────────

    def qkv_projection_decode(self, layer: int) -> ComputeResult:
        """QKV for a single new token: [1, d_model] × [d_model, 3×d_model]."""
        m = self.model
        w_bytes  = m.weight_bytes(m.d_model, 3 * m.d_model)
        act_bytes = m.d_model * 2
        return self.matmul(1, m.d_model, 3 * m.d_model, layer, w_bytes, act_bytes)

    def attention_score_decode(self, kv_len: int, layer: int) -> ComputeResult:
        """q × K^T: [1, head_dim] × [head_dim, kv_len] per head."""
        m = self.model
        M, K, N = 1, m.head_dim, kv_len
        ops_total = M * K * N * 2 * m.num_heads
        compute_c = self._compute_cycles(M, K, N) * m.num_heads
        act_bytes = m.head_dim * 2 + kv_len * m.head_dim * 2  # q + K cache
        act_e = self.memory.activation_edram.read(act_bytes).energy_pj
        rsa_e = self._energy_pj(ops_total)

        result = ComputeResult(compute_c, rsa_energy_pj=rsa_e,
                               weight_energy_pj=0.0, act_energy_pj=act_e,
                               ops=ops_total, activation_bytes=act_bytes)
        self.total_compute_cycles += compute_c
        self.total_energy_pj      += result.energy_pj
        self.total_ops            += ops_total
        return result

    def attention_output_decode(self, kv_len: int, layer: int) -> ComputeResult:
        """scores × V: [1, kv_len] × [kv_len, head_dim] per head."""
        m = self.model
        M, K, N = 1, kv_len, m.head_dim
        ops_total = M * K * N * 2 * m.num_heads
        compute_c = self._compute_cycles(M, K, N) * m.num_heads
        act_bytes = kv_len * 2 * m.num_heads + kv_len * m.head_dim * 2  # weights + V
        act_e = self.memory.activation_edram.read(act_bytes).energy_pj
        rsa_e = self._energy_pj(ops_total)

        result = ComputeResult(compute_c, rsa_energy_pj=rsa_e,
                               weight_energy_pj=0.0, act_energy_pj=act_e,
                               ops=ops_total, activation_bytes=act_bytes)
        self.total_compute_cycles += compute_c
        self.total_energy_pj      += result.energy_pj
        self.total_ops            += ops_total
        return result

    def output_projection_decode(self, layer: int) -> ComputeResult:
        m = self.model
        w_bytes  = m.weight_bytes(m.d_model, m.d_model)
        act_bytes = m.d_model * 2
        return self.matmul(1, m.d_model, m.d_model, layer, w_bytes, act_bytes)

    def ffn_decode(self, layer: int) -> Tuple[ComputeResult, ComputeResult]:
        m = self.model
        up_w   = m.weight_bytes(m.d_model, m.d_ffn)
        down_w = m.weight_bytes(m.d_ffn, m.d_model)
        up   = self.matmul(1, m.d_model, m.d_ffn,   layer, up_w,   m.d_model * 2)
        down = self.matmul(1, m.d_ffn,   m.d_model, layer, down_w, m.d_ffn * 2)
        return up, down

    def recompute_kv(self, layer: int) -> ComputeResult:
        """Recompute K and V for a single evicted-to-recompute token."""
        m = self.model
        w_bytes  = m.weight_bytes(m.d_model, 2 * m.d_model)  # K and V projections
        act_bytes = m.d_model * 2                              # stored input vector
        return self.matmul(1, m.d_model, 2 * m.d_model, layer, w_bytes, act_bytes)

    # ── Utilisation stats ─────────────────────────────────────────────────────

    @property
    def peak_tops(self) -> float:
        return (self.hw.rsa_rows * self.hw.rsa_cols * 2 *
                self.hw.clock_freq_hz) / 1e12

    @property
    def effective_tops(self) -> float:
        if self.total_compute_cycles == 0:
            return 0.0
        return self.total_ops / (self.total_compute_cycles /
                                  self.hw.clock_freq_hz) / 1e12
