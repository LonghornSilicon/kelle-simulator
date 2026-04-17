"""Hardware and model configurations derived from the Kelle MICRO 2025 paper."""

from dataclasses import dataclass, field
from typing import Dict
import math


@dataclass
class ModelConfig:
    name: str
    num_layers: int
    num_heads: int
    d_model: int
    d_ffn: int
    head_dim: int
    vocab_size: int
    max_seq_len: int
    weight_bits: int = 8    # INT8 weights on-chip
    kv_bits: int = 16       # KV stored as 16-bit (split into MSB+LSB halves)

    @property
    def kv_bytes_per_token_per_layer(self) -> int:
        """Bytes to store one token's K+V for a single layer (INT8 after quantisation)."""
        bytes_per_bit = self.kv_bits // 8
        return 2 * self.num_heads * self.head_dim * bytes_per_bit

    @property
    def kv_bytes_per_token(self) -> int:
        """Total KV bytes across all layers for one token."""
        return self.kv_bytes_per_token_per_layer * self.num_layers

    @property
    def input_bytes_per_token(self) -> int:
        """Input vector (d_model × 2 bytes for FP16 storage)."""
        return self.d_model * 2

    def weight_bytes(self, in_dim: int, out_dim: int) -> int:
        return in_dim * out_dim * (self.weight_bits // 8)


@dataclass
class HardwareConfig:
    # ── RSA ──────────────────────────────────────────────────────────────────
    rsa_rows: int = 32
    rsa_cols: int = 32
    clock_freq_hz: int = 1_000_000_000   # 1 GHz

    # ── Memory sizes ─────────────────────────────────────────────────────────
    sram_size_bytes: int        = 2 * 1024 * 1024    # 2 MB weight SRAM
    edram_kvcache_bytes: int    = 4 * 1024 * 1024    # 4 MB KV-cache eDRAM
    edram_activation_bytes: int = 256 * 1024          # 256 KB activation eDRAM
    dram_size_gb: float         = 16.0

    # ── Bandwidths (GB/s → bytes/cycle at 1 GHz) ─────────────────────────────
    sram_bandwidth_gbps: float  = 128.0   # → 128 B/cycle
    edram_bandwidth_gbps: float = 256.0   # → 256 B/cycle
    dram_bandwidth_gbps: float  = 64.0    # → 64 B/cycle

    # ── Latencies (cycles) ───────────────────────────────────────────────────
    sram_latency_cycles: int  = 1
    edram_latency_cycles: int = 4
    dram_latency_cycles: int  = 100

    # ── Energy per byte (pJ) ─────────────────────────────────────────────────
    sram_energy_pj_per_byte: float  = 185.9   # Table 1, Kelle paper (65nm, 4MB)
    edram_energy_pj_per_byte: float = 84.8    # Table 1
    dram_energy_pj_per_byte: float  = 640.0   # typical LPDDR off-chip

    # ── 2DRP refresh intervals (seconds) ─────────────────────────────────────
    # Section 7.1: chosen to maintain avg retention-failure rate ≤ 2×10⁻³
    refresh_hst_msb_s: float = 360e-6    # High-Score Token, bits [15:8]
    refresh_hst_lsb_s: float = 5400e-6   # High-Score Token, bits  [7:0]
    refresh_lst_msb_s: float = 1440e-6   # Low-Score  Token, bits [15:8]
    refresh_lst_lsb_s: float = 7200e-6   # Low-Score  Token, bits  [7:0]

    # eDRAM retention model: P_flip(t) ≈ t / tau  (linearised for small t)
    # Calibrated so P_flip(hst_msb_interval) ≈ target_failure_rate
    retention_tau_s: float       = 0.18   # 360µs / 2×10⁻³ = 180 ms
    target_failure_rate: float   = 2e-3

    # ── eDRAM bank structure ─────────────────────────────────────────────────
    num_edram_banks: int = 32   # 8 each for Key-MSB / Key-LSB / Val-MSB / Val-LSB

    # ── Power (W) — from Kelle Table 3 ───────────────────────────────────────
    rsa_power_w: float              = 1.108   # 17% of 6.52 W
    edram_power_w: float            = 1.891   # 29%
    sram_power_w: float             = 2.673   # 41%
    sfu_power_w: float              = 0.848   # 13%
    systolic_evictor_power_w: float = 0.028   # Section 6
    dram_power_w: float             = 11.74

    # ── Area (mm²) ────────────────────────────────────────────────────────────
    rsa_area_mm2: float              = 2.185   # 23% of 9.5 mm²
    edram_area_mm2: float            = 3.135   # 33%
    sram_area_mm2: float             = 3.515   # 37%
    sfu_area_mm2: float              = 0.665   # 7%
    systolic_evictor_area_mm2: float = 0.060   # Section 6
    total_on_chip_area_mm2: float    = 9.50

    # ── AERP parameters ───────────────────────────────────────────────────────
    kv_cache_capacity_tokens: int    = 128
    recent_window_tokens: int        = 64
    initial_tokens_preserved: int    = 10
    recompute_head_threshold: float  = 0.5   # >50% heads → store input for recompute

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def ns_per_cycle(self) -> float:
        return 1e9 / self.clock_freq_hz

    @property
    def sram_bytes_per_cycle(self) -> float:
        return self.sram_bandwidth_gbps  # GB/s == B/cycle at 1 GHz

    @property
    def edram_bytes_per_cycle(self) -> float:
        return self.edram_bandwidth_gbps

    @property
    def dram_bytes_per_cycle(self) -> float:
        return self.dram_bandwidth_gbps

    def refresh_interval_cycles(self, group: str) -> int:
        table = {
            'hst_msb': self.refresh_hst_msb_s,
            'hst_lsb': self.refresh_hst_lsb_s,
            'lst_msb': self.refresh_lst_msb_s,
            'lst_lsb': self.refresh_lst_lsb_s,
        }
        return int(table[group] * self.clock_freq_hz)

    def bit_flip_prob(self, elapsed_s: float) -> float:
        """Linearised retention-failure probability for elapsed seconds without refresh."""
        return min(1.0, elapsed_s / self.retention_tau_s)

    @property
    def total_power_w(self) -> float:
        return (self.rsa_power_w + self.edram_power_w + self.sram_power_w +
                self.sfu_power_w + self.systolic_evictor_power_w)


# ─────────────────────────────────────────────────────────────────────────────
# Pre-defined model configurations
# ─────────────────────────────────────────────────────────────────────────────

MODELS: Dict[str, ModelConfig] = {
    "opt-125m": ModelConfig(
        name="OPT-125M",
        num_layers=12,
        num_heads=12,
        d_model=768,
        d_ffn=3072,
        head_dim=64,
        vocab_size=50272,
        max_seq_len=2048,
    ),
    "opt-1.3b": ModelConfig(
        name="OPT-1.3B",
        num_layers=24,
        num_heads=32,
        d_model=2048,
        d_ffn=8192,
        head_dim=64,
        vocab_size=50272,
        max_seq_len=2048,
    ),
    "opt-6.7b": ModelConfig(
        name="OPT-6.7B",
        num_layers=32,
        num_heads=32,
        d_model=4096,
        d_ffn=16384,
        head_dim=128,
        vocab_size=50272,
        max_seq_len=2048,
    ),
}

DEFAULT_HW_CONFIG = HardwareConfig()
