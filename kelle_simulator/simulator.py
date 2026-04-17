"""
KelleSimulator — top-level cycle-accurate simulation engine.

Simulates OPT-style transformer inference on the Kelle accelerator:
  1. Prefill phase  — process the entire input prompt
  2. Decode phase   — auto-regressively generate tokens one at a time

Each phase simulates all transformer layers with:
  • RSA compute (QKV projections, attention scores/output, FFN)
  • SFU operations  (softmax, layer norm, GELU, positional embeddings)
  • Memory accesses (weight loads from SRAM/DRAM, KV reads/writes to eDRAM)
  • AERP eviction/recomputation decisions
  • 2DRP refresh scheduling (interleaved with compute idle cycles)
  • Systolic evictor attention-score accumulation

All timing is in cycles (1 cycle = 1 ns at 1 GHz).
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from .config import HardwareConfig, ModelConfig
from .memory import MemoryHierarchy
from .edram_controller import EDRAMController
from .aerp import AERP, TokenStatus
from .systolic_evictor import SystolicEvictor
from .rsa import RSA
from .sfu import SFU


# ---------------------------------------------------------------------------
# Realistic attention weight model
# ---------------------------------------------------------------------------
# Based on empirical observations from H2O, SnapKV, and StreamingLLM:
#   - Attention sink: first N tokens capture disproportionate mass
#   - Recency bias: last R tokens decay exponentially toward present
#   - Content routing: head-specific focus on semantically relevant tokens
#
# Three head archetypes (cycled by head index):
#   Sink heads   (head % 3 == 0): strong initial-token focus
#   Recency heads(head % 3 == 1): strong local/recent focus
#   Content heads(head % 3 == 2): focus on pseudo-content tokens that vary
#                                   by (layer, head) -- models sparse routing

def _head_attention_weights(
    sorted_token_ids: List[int],
    layer: int,
    head: int,
    initial_preserved: int,
    recent_window: int,
) -> Dict[int, float]:
    """
    Generate a realistic (non-uniform) attention distribution for one head.

    Returns a normalised dict {token_id: weight}.
    """
    n = len(sorted_token_ids)
    if n == 0:
        return {}

    effective_recent = min(recent_window, max(1, n // 2))
    head_type = head % 3

    raw: Dict[int, float] = {}
    for i, tid in enumerate(sorted_token_ids):
        pos_from_end = n - 1 - i

        if head_type == 0:
            # Sink head: heavy on initial tokens, light recency, near-zero middle
            sink    = 4.0 if i < initial_preserved else 0.0
            recency = math.exp(-pos_from_end * 6.0 / max(effective_recent, 1)) \
                      if pos_from_end < effective_recent else 0.0
            content = 0.02
            w = sink + recency + content

        elif head_type == 1:
            # Recency head: strong exponential decay from present
            sink    = 0.5 if i < initial_preserved else 0.0
            recency = math.exp(-pos_from_end * 3.0 / max(effective_recent, 1)) \
                      if pos_from_end < effective_recent else 0.0
            content = 0.02 + 0.08 * max(0.0, math.sin(tid * 0.31 + layer * 0.9))
            w = sink + recency + content

        else:
            # Content head: sparse focus on pseudo-content tokens.
            # Uses a deterministic hash so the same token is repeatedly
            # attended to across decode steps, simulating real content routing.
            sink      = 0.3 if i < initial_preserved else 0.0
            recency   = math.exp(-pos_from_end * 2.0 / max(effective_recent, 1)) \
                        if pos_from_end < effective_recent else 0.0
            # Content signal: high for tokens whose id hashes with (layer, head)
            content_signal = (tid * 6364136223846793005 + layer * 1442695040888963407
                              + head * 2862933555777941757) & 0xFFFFFFFF
            content = 0.05 + 1.2 * (content_signal / 0xFFFFFFFF) ** 6
            w = sink + recency + content

        raw[tid] = max(w, 1e-9)

    total = sum(raw.values())
    return {tid: v / total for tid, v in raw.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Statistics container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PhaseStats:
    name: str
    total_cycles: int = 0
    compute_cycles: int = 0
    memory_stall_cycles: int = 0
    refresh_stall_cycles: int = 0
    sfu_cycles: int = 0
    evictor_cycles: int = 0

    compute_energy_pj: float = 0.0
    sram_energy_pj: float = 0.0
    edram_energy_pj: float = 0.0
    dram_energy_pj: float = 0.0
    edram_refresh_energy_pj: float = 0.0
    sfu_energy_pj: float = 0.0
    evictor_energy_pj: float = 0.0

    ops: int = 0
    tokens_processed: int = 0

    @property
    def total_energy_pj(self) -> float:
        return (self.compute_energy_pj + self.sram_energy_pj + self.edram_energy_pj +
                self.dram_energy_pj + self.edram_refresh_energy_pj +
                self.sfu_energy_pj + self.evictor_energy_pj)

    @property
    def total_energy_uj(self) -> float:
        return self.total_energy_pj / 1e6

    @property
    def total_energy_mj(self) -> float:
        return self.total_energy_pj / 1e9

    @property
    def average_power_w(self) -> float:
        if self.total_cycles == 0:
            return 0.0
        elapsed_s = self.total_cycles / 1e9
        return self.total_energy_pj / 1e12 / elapsed_s


@dataclass
class SimStats:
    prefill: PhaseStats = field(default_factory=lambda: PhaseStats("prefill"))
    decode:  PhaseStats = field(default_factory=lambda: PhaseStats("decode"))

    # Per-decode-step latency (cycles)
    decode_latency_per_step: List[int] = field(default_factory=list)

    # KV-cache occupancy snapshots (one per decode step)
    kv_occupancy_log: List[Dict] = field(default_factory=list)

    # AERP event log
    eviction_events: int = 0
    recompute_events: int = 0

    # eDRAM integrity
    total_bit_flips_msb: int = 0
    total_bit_flips_lsb: int = 0

    # Timing
    wall_clock_start: float = 0.0
    wall_clock_end: float = 0.0

    @property
    def total_cycles(self) -> int:
        return self.prefill.total_cycles + self.decode.total_cycles

    @property
    def total_energy_pj(self) -> float:
        return self.prefill.total_energy_pj + self.decode.total_energy_pj

    @property
    def prefill_latency_ms(self) -> float:
        return self.prefill.total_cycles / 1e6  # cycles / (1e9 cycles/s) × 1e3

    @property
    def decode_latency_ms(self) -> float:
        return self.decode.total_cycles / 1e6

    @property
    def total_latency_ms(self) -> float:
        return self.total_cycles / 1e6

    @property
    def tokens_per_second(self) -> float:
        if self.decode.total_cycles == 0:
            return 0.0
        return (self.decode.tokens_processed /
                (self.decode.total_cycles / self.decode.total_cycles)
                * self.decode.tokens_processed
                / (self.decode.total_cycles / 1e9))

    def throughput_tokens_per_sec(self) -> float:
        if self.decode.total_cycles == 0:
            return 0.0
        return self.decode.tokens_processed / (self.decode.total_cycles / 1e9)

    def effective_tops(self, rsa: RSA) -> float:
        return rsa.effective_tops


# ─────────────────────────────────────────────────────────────────────────────
# Main simulator
# ─────────────────────────────────────────────────────────────────────────────

class KelleSimulator:
    def __init__(self, model_cfg: ModelConfig, hw_cfg: HardwareConfig):
        self.model = model_cfg
        self.hw    = hw_cfg

        # Build subsystems
        self.memory   = MemoryHierarchy(model_cfg, hw_cfg)
        self.evictor  = SystolicEvictor(hw_cfg, model_cfg)
        self.aerp     = AERP(hw_cfg, model_cfg, self.evictor)
        self.rsa      = RSA(hw_cfg, model_cfg, self.memory)
        self.sfu      = SFU(hw_cfg)
        self.edram_ctrl = EDRAMController(hw_cfg, self.memory.kv_edram)

        self.stats = SimStats()
        self._cycle: int = 0  # global cycle counter

    # ── Cycle accounting helper ───────────────────────────────────────────────

    def _tick(self, cycles: int, phase: PhaseStats) -> None:
        """Advance the global cycle counter and trigger refresh checks."""
        self._cycle += cycles
        stall, energy = self.edram_ctrl.advance_to_cycle(self._cycle)
        phase.total_cycles     += cycles + stall
        phase.refresh_stall_cycles += stall
        phase.edram_refresh_energy_pj += energy

    def _add_compute(self, result, phase: PhaseStats) -> None:
        phase.compute_cycles      += result.total_cycles
        phase.memory_stall_cycles += result.memory_stall_cycles
        phase.compute_energy_pj   += result.rsa_energy_pj
        phase.edram_energy_pj     += result.act_energy_pj
        if result.weight_from_dram:
            phase.dram_energy_pj  += result.weight_energy_pj
        else:
            phase.sram_energy_pj  += result.weight_energy_pj
        phase.ops                 += result.ops
        self._tick(result.total_cycles, phase)

    def _add_sfu(self, result, phase: PhaseStats) -> None:
        phase.sfu_cycles     += result.cycles
        phase.sfu_energy_pj  += result.energy_pj
        self._tick(result.cycles, phase)

    # ── Prefill ───────────────────────────────────────────────────────────────

    def prefill(self, prompt_len: int) -> PhaseStats:
        phase = self.stats.prefill
        phase.tokens_processed = prompt_len
        m = self.model

        # One-time weight prefetch: load all weights from DRAM into on-chip buffer.
        # This one-time cost replaces per-step-per-layer DRAM weight loads during decode.
        if self.memory._weight_prefetch_active:
            pf_result = self.memory.prefetch_weights()
            phase.dram_energy_pj += pf_result.energy_pj
            self._tick(pf_result.cycles, phase)

        for layer in range(m.num_layers):
            # Layer norm (pre-attention)
            self._add_sfu(self.sfu.layer_norm(prompt_len * m.d_model), phase)

            # QKV projections
            self._add_compute(self.rsa.qkv_projection_prefill(prompt_len, layer), phase)

            # Positional embeddings (RoPE for rotary, or learned)
            self._add_sfu(self.sfu.rotary_embedding(prompt_len, m.head_dim, m.num_heads), phase)

            # Attention scores: Q × K^T
            self._add_compute(self.rsa.attention_score_prefill(prompt_len, layer), phase)

            # Softmax
            self._add_sfu(self.sfu.softmax_prefill(prompt_len, m.num_heads), phase)

            # Attention output: scores × V
            self._add_compute(self.rsa.attention_output_prefill(prompt_len, layer), phase)

            # Output projection
            self._add_compute(self.rsa.output_projection_prefill(prompt_len, layer), phase)

            # Residual + LayerNorm (pre-FFN)
            self._add_sfu(self.sfu.residual_add(prompt_len * m.d_model), phase)
            self._add_sfu(self.sfu.layer_norm(prompt_len * m.d_model), phase)

            # FFN
            up, down = self.rsa.ffn_prefill(prompt_len, layer)
            self._add_compute(up, phase)
            self._add_sfu(self.sfu.gelu(prompt_len * m.d_ffn), phase)
            self._add_compute(down, phase)
            self._add_sfu(self.sfu.residual_add(prompt_len * m.d_model), phase)

            # Store KV pairs into eDRAM for all prompt tokens
            kv_store_cycles = 0
            kv_store_energy = 0.0
            for tok_id in range(prompt_len):
                r = self.memory.kv_edram.store_kv(tok_id, layer)
                kv_store_cycles = max(kv_store_cycles, r.cycles)
                kv_store_energy += r.energy_pj
            phase.edram_energy_pj += kv_store_energy
            self._tick(kv_store_cycles, phase)

        # Insert tokens into AERP (up to capacity) and register with evictor/2DRP.
        # The evictor only tracks tokens actually present in the KV cache so that
        # find_eviction_candidate() never returns a token that isn't in AERP.
        num_cached = min(prompt_len, self.hw.kv_cache_capacity_tokens)
        for tok_id in range(num_cached):
            _, _, _ = self.aerp.insert_token(tok_id, tok_id, step=0)
            self.evictor.register_token(
                tok_id, tok_id,
                num_cached=num_cached,
                recent_window=self.hw.recent_window_tokens,
                initial_preserved=self.hw.initial_tokens_preserved,
            )
            is_hst = self.evictor.is_hst(tok_id)
            self.edram_ctrl.register_token(tok_id, is_hst, self._cycle)

        return phase

    # ── Single decode step (one new token) ────────────────────────────────────

    def _decode_step(self, step: int, new_token_id: int) -> int:
        """Simulate one decode step.  Returns cycles consumed for this step."""
        phase = self.stats.decode
        m = self.model
        step_start_cycle = self._cycle

        # Insert new token into AERP (may trigger eviction)
        token_position = new_token_id  # absolute sequence position
        evicted_id, action, search_cyc = self.aerp.insert_token(
            new_token_id, token_position, step)
        if action == "evicted":
            self.stats.eviction_events += 1
            self.edram_ctrl.evict_token(evicted_id)
        elif action == "recompute":
            self.stats.recompute_events += 1

        phase.evictor_cycles    += search_cyc
        phase.evictor_energy_pj += self.evictor.total_evictor_energy_pj
        self._tick(search_cyc, phase)

        # Register new token with evictor using correct total count
        kv_len = self.aerp.total_tokens
        self.evictor.register_token(
            new_token_id, token_position,
            num_cached=kv_len,
            recent_window=self.hw.recent_window_tokens,
            initial_preserved=self.hw.initial_tokens_preserved,
        )

        for layer in range(m.num_layers):
            # ── Pre-attention LayerNorm ───────────────────────────────────────
            self._add_sfu(self.sfu.layer_norm(m.d_model), phase)

            # ── QKV projection for the new token ─────────────────────────────
            self._add_compute(self.rsa.qkv_projection_decode(layer), phase)
            self._add_sfu(self.sfu.rotary_embedding(1, m.head_dim, m.num_heads), phase)

            # ── Recompute KV for any RECOMPUTE-status tokens ──────────────────
            recompute_tokens = [
                tid for tid, tok in self.aerp._tokens.items()
                if tok.status == TokenStatus.RECOMPUTE
            ]
            for tid in recompute_tokens:
                rc = self.rsa.recompute_kv(layer)
                self._add_compute(rc, phase)
                # Write freshly computed KV back to eDRAM
                wr = self.memory.kv_edram.store_kv(tid, layer)
                phase.edram_energy_pj += wr.energy_pj
                self._tick(wr.cycles, phase)
            if recompute_tokens:
                self.stats.recompute_events += len(recompute_tokens)

            # ── Load KV cache from eDRAM ──────────────────────────────────────
            load_cycles, load_energy = 0, 0.0
            for tid in self.aerp._tokens:
                r = self.memory.kv_edram.load_kv(tid, layer)
                load_cycles = max(load_cycles, r.cycles)  # parallel banks
                load_energy += r.energy_pj
            phase.edram_energy_pj += load_energy
            self._tick(load_cycles, phase)

            # ── Attention scores: q × K^T ─────────────────────────────────────
            self._add_compute(self.rsa.attention_score_decode(kv_len, layer), phase)

            # ── Softmax + evictor attention-score update ──────────────────────
            self._add_sfu(self.sfu.softmax_decode(kv_len, m.num_heads), phase)

            # Realistic per-head attention weights (sink / recency / content).
            # Called once per head per layer so head_counts in the evictor
            # reflects which tokens are consistently above-mean across heads,
            # enabling AERP to distinguish full-evict vs recompute candidates.
            sorted_tids = sorted(self.aerp._tokens.keys())
            total_evc = 0
            for head_idx in range(m.num_heads):
                attn_weights = _head_attention_weights(
                    sorted_tids, layer, head_idx,
                    self.hw.initial_tokens_preserved,
                    self.hw.recent_window_tokens,
                )
                total_evc += self.evictor.accumulate_scores(
                    attn_weights, layer, head_idx)
            # Evictor runs in parallel with RSA; charge only 1 cycle per layer
            self._tick(1, phase)

            # ── Attention output: scores × V ──────────────────────────────────
            self._add_compute(self.rsa.attention_output_decode(kv_len, layer), phase)

            # ── Output projection ─────────────────────────────────────────────
            self._add_compute(self.rsa.output_projection_decode(layer), phase)

            # ── Residual + LayerNorm + FFN ────────────────────────────────────
            self._add_sfu(self.sfu.residual_add(m.d_model), phase)
            self._add_sfu(self.sfu.layer_norm(m.d_model), phase)
            up, down = self.rsa.ffn_decode(layer)
            self._add_compute(up, phase)
            self._add_sfu(self.sfu.gelu(m.d_ffn), phase)
            self._add_compute(down, phase)
            self._add_sfu(self.sfu.residual_add(m.d_model), phase)

            # ── Store new token's KV into eDRAM ────────────────────────────────
            wr = self.memory.kv_edram.store_kv(new_token_id, layer)
            phase.edram_energy_pj += wr.energy_pj
            self._tick(wr.cycles, phase)

        # ── Update 2DRP token classifications ─────────────────────────────────
        importance_map = self.evictor.get_importance_map()
        threshold = self.evictor.get_hst_threshold()
        self.edram_ctrl.reclassify_tokens(importance_map, threshold)
        self.edram_ctrl.register_token(new_token_id,
                                        self.evictor.is_hst(new_token_id),
                                        self._cycle)

        step_cycles = self._cycle - step_start_cycle
        return step_cycles

    # ── Full simulation run ────────────────────────────────────────────────────

    def simulate(self, prompt_len: int, num_decode_steps: int,
                 verbose: bool = True) -> SimStats:
        self.stats.wall_clock_start = time.perf_counter()

        if verbose:
            print(f"\n{'='*60}")
            print(f"  Kelle Cycle-Accurate Simulator")
            print(f"  Model : {self.model.name}")
            print(f"  Prompt: {prompt_len} tokens  |  Generate: {num_decode_steps} tokens")
            print(f"  KV-cache capacity: {self.hw.kv_cache_capacity_tokens} tokens")
            print(f"  SRAM-resident layers: {len(self.memory.sram_resident_layers)}/{self.model.num_layers}")
            print(f"{'='*60}")

        # ── Prefill ────────────────────────────────────────────────────────────
        if verbose:
            print(f"\n[Prefill] Processing {prompt_len} tokens...")
        self.prefill(prompt_len)
        if verbose:
            ps = self.stats.prefill
            print(f"  Prefill complete: {ps.total_cycles:,} cycles  "
                  f"({self.stats.prefill_latency_ms:.3f} ms)  "
                  f"{ps.total_energy_uj:.2f} uJ")

        # ── Decode ─────────────────────────────────────────────────────────────
        if verbose:
            print(f"\n[Decode] Generating {num_decode_steps} tokens...")

        self.stats.decode.tokens_processed = num_decode_steps
        for step in range(num_decode_steps):
            new_tok_id = prompt_len + step
            step_cyc = self._decode_step(step, new_tok_id)
            self.stats.decode_latency_per_step.append(step_cyc)
            self.stats.kv_occupancy_log.append(self.aerp.occupancy_snapshot())

            if verbose and (step % max(1, num_decode_steps // 10) == 0 or
                            step == num_decode_steps - 1):
                occ = self.aerp.occupancy_snapshot()
                print(f"  step {step+1:>4}/{num_decode_steps}  "
                      f"cycle={self._cycle:>12,}  "
                      f"kv={occ['cached']:>4}C/{occ['recompute']:>3}R  "
                      f"evictions={self.stats.eviction_events}  "
                      f"recomputes={self.stats.recompute_events}")

        # ── Aggregate decode energy from memory subsystems ────────────────────
        # (SRAM and DRAM energy captured at weight-load time)
        # Summarise refresh energy
        self.stats.decode.edram_refresh_energy_pj = \
            self.edram_ctrl.total_refresh_energy_pj
        self.stats.total_bit_flips_msb = self.edram_ctrl.total_bit_flips_msb
        self.stats.total_bit_flips_lsb = self.edram_ctrl.total_bit_flips_lsb

        self.stats.wall_clock_end = time.perf_counter()
        return self.stats

    # ── Results summary ────────────────────────────────────────────────────────

    def print_summary(self) -> None:
        s = self.stats
        hw = self.hw
        m = self.model
        wall = s.wall_clock_end - s.wall_clock_start

        SEP = "=" * 60
        DIV = "-" * 56

        print(f"\n{SEP}")
        print(f"  KELLE SIMULATION RESULTS -- {m.name}")
        print(f"{SEP}")

        print(f"\n-- Latency {DIV[:46]}")
        print(f"  Prefill latency   : {s.prefill_latency_ms:>10.3f} ms"
              f"  ({s.prefill.total_cycles:,} cycles)")
        print(f"  Decode latency    : {s.decode_latency_ms:>10.3f} ms"
              f"  ({s.decode.total_cycles:,} cycles)")
        print(f"  Total latency     : {s.total_latency_ms:>10.3f} ms")
        if s.decode_latency_per_step:
            avg_step = sum(s.decode_latency_per_step) / len(s.decode_latency_per_step)
            print(f"  Avg decode step   : {avg_step/1e6:>10.3f} ms/token "
                  f"({avg_step:,.0f} cycles/token)")
        print(f"  Throughput        : {s.throughput_tokens_per_sec():>10.2f} tokens/sec")

        print(f"\n-- Energy {DIV[:47]}")
        total_uj = s.total_energy_pj / 1e6
        print(f"  Total energy      : {total_uj:>10.3f} uJ")
        pf = s.prefill
        dc = s.decode
        print(f"  Prefill energy    : {pf.total_energy_uj:>10.3f} uJ")
        print(f"  Decode energy     : {dc.total_energy_uj:>10.3f} uJ")
        print(f"  -- Breakdown (decode) {DIV[:35]}")
        print(f"    Compute (RSA)   : {dc.compute_energy_pj/1e6:>10.3f} uJ  "
              f"({100*dc.compute_energy_pj/max(dc.total_energy_pj,1):.1f}%)")
        print(f"    SRAM (weights)  : {dc.sram_energy_pj/1e6:>10.3f} uJ  "
              f"({100*dc.sram_energy_pj/max(dc.total_energy_pj,1):.1f}%)")
        print(f"    eDRAM (KV I/O)  : {dc.edram_energy_pj/1e6:>10.3f} uJ  "
              f"({100*dc.edram_energy_pj/max(dc.total_energy_pj,1):.1f}%)")
        print(f"    eDRAM (refresh) : {dc.edram_refresh_energy_pj/1e6:>10.3f} uJ  "
              f"({100*dc.edram_refresh_energy_pj/max(dc.total_energy_pj,1):.1f}%)")
        print(f"    Off-chip DRAM   : {dc.dram_energy_pj/1e6:>10.3f} uJ  "
              f"({100*dc.dram_energy_pj/max(dc.total_energy_pj,1):.1f}%)")
        print(f"    SFU             : {dc.sfu_energy_pj/1e6:>10.3f} uJ  "
              f"({100*dc.sfu_energy_pj/max(dc.total_energy_pj,1):.1f}%)")
        print(f"    Evictor         : {dc.evictor_energy_pj/1e6:>10.3f} uJ  "
              f"({100*dc.evictor_energy_pj/max(dc.total_energy_pj,1):.1f}%)")

        print(f"\n-- Compute {DIV[:46]}")
        print(f"  Peak TOPs         : {self.rsa.peak_tops:>10.3f}")
        print(f"  Effective TOPs    : {self.rsa.effective_tops:>10.3f}")
        print(f"  Utilisation       : {100*self.rsa.effective_tops/self.rsa.peak_tops:>9.1f}%")
        print(f"  Total MACs        : {(pf.ops+dc.ops)//2:>10,}")

        print(f"\n-- KV Cache (AERP) {DIV[:38]}")
        print(f"  Capacity (N')     : {hw.kv_cache_capacity_tokens:>10} tokens")
        print(f"  Eviction events   : {s.eviction_events:>10}")
        print(f"  Recompute events  : {s.recompute_events:>10}")
        aerp_stats = self.aerp.stats
        print(f"  Cache hit rate    : {100*self.aerp.cache_hit_rate:>9.2f}%")
        print(f"  KV bytes saved    : {aerp_stats.kv_bytes_saved_by_recompute/1024:>10.1f} KB "
              "(via recompute path)")

        print(f"\n-- 2DRP Refresh {DIV[:41]}")
        print(f"  Refresh intervals :")
        print(f"    HST-MSB         :     {hw.refresh_hst_msb_s*1e3:.3f} ms")
        print(f"    HST-LSB         :     {hw.refresh_hst_lsb_s*1e3:.3f} ms")
        print(f"    LST-MSB         :     {hw.refresh_lst_msb_s*1e3:.3f} ms")
        print(f"    LST-LSB         :     {hw.refresh_lst_lsb_s*1e3:.3f} ms")
        print(f"  Total refresh ops : {len(self.edram_ctrl.refresh_log):>10,}")
        print(f"  Refresh energy    : {self.edram_ctrl.total_refresh_energy_pj/1e6:>10.3f} uJ")
        avg_ri = self.edram_ctrl.average_refresh_interval_ms()
        print(f"  Avg interval MSB  : {avg_ri['msb']:>10.3f} ms")
        print(f"  Avg interval LSB  : {avg_ri['lsb']:>10.3f} ms")
        print(f"  Bit-flip MSB      : {s.total_bit_flips_msb:>10,} (expected)")
        print(f"  Bit-flip LSB      : {s.total_bit_flips_lsb:>10,} (expected)")

        print(f"\n-- Memory Subsystem {DIV[:37]}")
        sram_layers = len(self.memory.sram_resident_layers)
        print(f"  SRAM layers       : {sram_layers}/{m.num_layers} "
              f"({100*sram_layers/m.num_layers:.0f}% on-chip)")
        print(f"  DRAM layers       : {self.memory.num_dram_layers}")
        print(f"  DRAM transfers    : {self.memory.dram.total_bytes_transferred/1024:.1f} KB")
        edram_util = self.memory.kv_edram.utilisation
        print(f"  eDRAM utilisation : {100*edram_util:.1f}%  "
              f"({self.memory.kv_edram.used_bytes/1024:.1f} KB / "
              f"{hw.edram_kvcache_bytes/1024:.0f} KB)")
        if self.memory._weight_prefetch_active:
            buf_mb = hw.weight_prefetch_buffer_bytes / 1024**2
            model_mb = m.total_weight_bytes / 1024**2
            print(f"  Weight prefetch   : ENABLED ({model_mb:.1f} MB / {buf_mb:.1f} MB buffer)")
            print(f"  One-time load     : {self.memory.prefetch_load_bytes/1024:.1f} KB from DRAM")
        else:
            print(f"  Weight prefetch   : disabled (weights reload from DRAM each step)")

        print(f"\n-- Hardware (reference) {DIV[:33]}")
        print(f"  On-chip area      : {hw.total_on_chip_area_mm2:.1f} mm2")
        print(f"  On-chip power     : {hw.total_power_w:.2f} W")
        print(f"  Off-chip DRAM pwr : {hw.dram_power_w:.2f} W")
        print(f"  Systolic evictor  : {hw.systolic_evictor_area_mm2:.2f} mm2  "
              f"{hw.systolic_evictor_power_w*1000:.0f} mW")
        if hw.weight_prefetch_buffer_bytes > 0:
            print(f"  Weight buf (FPGA) : {hw.weight_prefetch_buffer_bytes/1024**2:.0f} MB on-chip")

        print(f"\n-- Simulation overhead {DIV[:34]}")
        print(f"  Wall-clock time   : {wall:.3f} s")
        print(f"{SEP}\n")
