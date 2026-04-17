# Kelle Accelerator — Architecture Specification

Source: "Kelle: Co-design KV Caching and eDRAM for Efficient LLM Serving in Edge Computing"  
MICRO 2025 (58th IEEE/ACM International Symposium on Microarchitecture)  
Authors: Tianhua Xia, Sai Qian Zhang — arXiv 2510.16040

---

## 1. System Overview

Kelle is a software–hardware co-design for LLM inference on edge devices.  The core
insight is that KV caches (which grow linearly with sequence length) are the dominant
memory consumer in LLM serving, and that eDRAM is a far denser on-chip memory than
SRAM — but requires periodic refresh operations that normally hurt power and latency.
Kelle eliminates this drawback by making refresh policy aware of which tokens are
important (and therefore which bits matter more).

```
 ┌─────────────────────────────────────────────────────────────────┐
 │                        Kelle On-chip (9.5 mm²)                  │
 │                                                                  │
 │   ┌───────────────┐   ┌───────────────────────────────────────┐  │
 │   │  Weight SRAM  │   │          KV-Cache eDRAM (4 MB)        │  │
 │   │    (2 MB)     │   │  32 banks: 8×Key-MSB, 8×Key-LSB,      │  │
 │   │  128 GB/s     │   │           8×Val-MSB, 8×Val-LSB        │  │
 │   │  185.9 pJ/B   │   │  256 GB/s  |  84.8 pJ/B               │  │
 │   └──────┬────────┘   └────────────────┬──────────────────────┘  │
 │          │                             │   ┌──────────────────┐  │
 │          ▼                             │   │ eDRAM Controller │  │
 │   ┌──────────────┐                     │   │   (2DRP)         │  │
 │   │     RSA      │◄────────────────────┘   └──────────────────┘  │
 │   │  32×32 PEs   │                                               │
 │   │  8-bit MAC   │◄────── Activation eDRAM (256 KB)              │
 │   │  4.13 TOPS   │                                               │
 │   │   1 GHz      │──────► SFU (softmax, layernorm, GELU, RoPE)   │
 │   └──────┬───────┘                                               │
 │          │            ┌──────────────────────────────┐           │
 │          └───────────►│   Systolic Evictor (0.06 mm²)│           │
 │                       └──────────────────────────────┘           │
 └─────────────────────────────────────────────────────────────────┘
                │
                ▼  PCIe / AXI  
         Off-chip DRAM (16 GB, 64 GB/s)
```

---

## 2. Memory Subsystem

### 2.1 Weight SRAM (2 MB, 37% of on-chip area)

- Stores INT8-quantised transformer weights for layers that fit within budget.
- Bandwidth: 128 GB/s = 128 bytes/cycle at 1 GHz.
- Read latency: 1 cycle.
- Energy: 185.9 pJ/byte (measured at 65 nm, 2 MB array).
- Layer placement: layers are greedily packed from layer 0 upwards; overflow
  layers are paged from off-chip DRAM on demand.

### 2.2 KV-Cache eDRAM (4 MB, 33% of on-chip area)

- Stores Key and Value tensors for all cached tokens across all transformer layers.
- Bandwidth: 256 GB/s = 256 bytes/cycle.
- Read latency: 4 cycles.
- Energy: 84.8 pJ/byte (54% cheaper than SRAM of the same capacity at 65 nm).
- **Bank structure** (Section 5.1):
  - 32 banks, 128 KB each.
  - Banks 0–7:   Key MSBs (bits [15:8])
  - Banks 8–15:  Key LSBs (bits  [7:0])
  - Banks 16–23: Value MSBs
  - Banks 24–31: Value LSBs
  - Tokens are striped across the 8 banks within each group (token_id % 8).
  - Bank structure enables the 2DRP controller to apply independent refresh
    frequencies to each byte-plane independently.

### 2.3 Activation eDRAM (256 KB)

- Scratchpad for intermediate activations (attention scores, FFN intermediates).
- Same bandwidth and latency as KV-cache eDRAM.

### 2.4 Off-chip DRAM (16 GB, 64 GB/s)

- Stores weight layers that overflow the 2 MB SRAM budget.
- Latency: 100 cycles.  Energy: ~640 pJ/byte (LPDDR).
- For OPT-125M (12 layers, ~38 MB INT8): all layers spill to DRAM.
  For smaller KV-only portions, the eDRAM is sufficient.

---

## 3. eDRAM Controller — 2DRP

### 3.1 Motivation

Standard eDRAM uses a single uniform refresh interval across the entire array,
forcing a conservative short interval to protect the most sensitive cells.  This
wastes power refreshing cells that could tolerate a much longer interval.

### 3.2 Two Dimensions of Adaptivity

**Dimension 1 — Token importance (attention-score based)**  
Tokens are classified as:
- **HST** (High-Score Token): accumulated attention score ≥ median across all cached tokens.
- **LST** (Low-Score Token): below median.

HST data is more likely to be read again soon → needs to be accurate → shorter refresh.

**Dimension 2 — Bit significance (MSB vs LSB)**  
A bit-flip in the MSB (bits [15:8]) of a KV value produces a ~256× larger numerical
error than a flip in the LSB (bits [7:0]).  Therefore MSBs need more frequent refresh.

### 3.3 Refresh Intervals (Section 7.1)

| Group    | Tokens | Bits     | Interval  | Relative frequency |
|----------|--------|----------|-----------|--------------------|
| HST-MSB  | High   | [15:8]   | 0.36 ms   | most frequent      |
| LST-MSB  | Low    | [15:8]   | 1.44 ms   | 4× slower          |
| HST-LSB  | High   | [7:0]    | 5.40 ms   | 15× slower         |
| LST-LSB  | Low    | [7:0]    | 7.20 ms   | 20× slower         |

Target average retention-failure rate: **2 × 10⁻³**.

### 3.4 Refresh Scheduling

The controller is co-located with the eDRAM banks.  Refresh operations are
interleaved into otherwise-idle eDRAM cycles (between normal read/write traffic).
A scheduler tracks next-refresh deadlines per (token, bit-group) and issues
refresh bursts opportunistically.  Stall rate ≈ 10% of refresh cycles in the
model (conservative estimate; actual depends on access patterns).

---

## 4. AERP — Attention-based Eviction and Recomputation Policy

### 4.1 Problem

When the KV cache fills up (capacity = N' tokens), standard eviction policies
either drop the token entirely (losing information) or keep everything (impossible
when N' < sequence length).

### 4.2 Kelle's Insight

The input vector that was used to compute a token's KV pair is **much smaller**
than the KV pair itself:

| Quantity | Size (OPT-125M, INT8) |
|----------|-----------------------|
| KV pair (all layers) | 12 layers × 2 × 12 heads × 64 dim × 1 B = 18,432 B |
| Input vector | 768 B (d_model × 1 byte) |
| Compression ratio | 24× |

By storing the input vector instead of the KV pair, Kelle can retain 24× more
"virtual tokens" in the same memory, recomputing KV on demand when the token
is attended to again.

### 4.3 Decision Rule

After computing attention scores for a given decode step:
1. Identify the minimum-importance token via the systolic evictor.
2. If `importance_ratio ≥ 0.5` (token is important in ≥50% of attention heads):  
   → **Recompute**: evict KV pair, retain input vector.
3. Otherwise:  
   → **Full eviction**: discard token entirely.

### 4.4 Protected Tokens

- **Initial tokens** (positions 0–9): always retained in full (critical for context).
- **Recent tokens** (last 64 positions): always retained in full (high recency bias).

### 4.5 Recomputation Cost

When a RECOMPUTE token is accessed, the RSA runs a QKV projection for that single
token across all layers:  
`cost = num_layers × cycles([1, d_model] × [d_model, 2×d_model])`

For OPT-125M: 12 × ceil(768 / (32×32)) × (768 + 31) ≈ small relative to full
decode step.

---

## 5. Systolic Evictor

### 5.1 Hardware Specification (Section 6)

- Area: **0.06 mm²** (0.6% of on-chip)
- Power: **0.028 W** (0.4% of on-chip)
- Latency contribution: 5–7% of decode step latency

### 5.2 Structure

```
 Attention output (one score per head, per token)
           │
    ┌──────▼──────┐
    │  Score      │  ← column of registers, one per cached token
    │  accumulator│    updated each attention head output
    └──────┬──────┘
           │
    ┌──────▼──────────────────┐
    │  Register chain         │  ← propagates running minimum top→bottom
    │  (N' stages)            │    updated in 1 cycle (pipelined)
    └──────┬──────────────────┘
           │
    ┌──────▼──────┐
    │  Comparator │  ← find min in ceil(log₂ N') cycles
    │  tree       │
    └─────────────┘
```

### 5.3 Timing

- Score update: **1 cycle** per attention head (pipelined alongside RSA softmax output)
- Min-search: **ceil(log₂ N')** cycles (e.g., 7 cycles for N'=128)

---

## 6. Reconfigurable Systolic Array (RSA)

### 6.1 Specification

| Parameter | Value |
|-----------|-------|
| Array dimensions | 32 × 32 PEs |
| PE operation | 8-bit MAC (INT8) |
| Dataflow | Weight-stationary |
| Clock frequency | 1 GHz |
| Peak throughput | 4.13 INT8 TOPs |
| Area | 2.185 mm² (23% of on-chip) |
| Power | 1.108 W (17% of on-chip) |

### 6.2 Timing Model

Weight-stationary systolic array for matmul C = A × B, A∈[M,K], B∈[K,N]:

```
tile_m = ceil(M / 32)
tile_n = ceil(N / 32)
cycles_per_tile = K + 31     # K accumulations + 31-cycle pipeline fill
total_cycles = tile_m × tile_n × cycles_per_tile
```

Memory stall: weight fetch from SRAM is pipelined with compute; stall only
occurs when the fetch takes longer than the compute (e.g., DRAM layers).

### 6.3 Operations per Decode Step (per layer, OPT-125M)

| Operation | Shape | MACs |
|-----------|-------|------|
| Q projection | [1,768]×[768,768] | 589,824 |
| K projection | [1,768]×[768,768] | 589,824 |
| V projection | [1,768]×[768,768] | 589,824 |
| Attn scores  | [1,64]×[64,N'] per head × 12 | 12×64×N' |
| Attn output  | [1,N']×[N',64] per head × 12 | 12×N'×64 |
| O projection | [1,768]×[768,768] | 589,824 |
| FFN up       | [1,768]×[768,3072] | 2,359,296 |
| FFN down     | [1,3072]×[3072,768] | 2,359,296 |

---

## 7. Specialized Function Units (SFU)

| Operation | Algorithm | Cycles |
|-----------|-----------|--------|
| Softmax | Online max + exp (Softermax) | N + 8 pipeline |
| LayerNorm | Two-pass mean+variance | 2N + 16 |
| GELU | Lookup table | N |
| ReLU | Vectorised compare | N/4 |
| RoPE | 2 multiply-adds per dim | 2N |
| Residual add | Vectorised add | N/4 |

SFU area: 0.665 mm² (7% of on-chip).  Power: 0.848 W (13% of on-chip).

---

## 8. Hardware Area and Power Summary

| Component | Area (mm²) | Area % | Power (W) | Power % |
|-----------|-----------|--------|-----------|---------|
| RSA | 2.185 | 23% | 1.108 | 17% |
| eDRAM (total) | 3.135 | 33% | 1.891 | 29% |
| SRAM | 3.515 | 37% | 2.673 | 41% |
| SFU | 0.665 | 7% | 0.848 | 13% |
| Systolic Evictor | 0.060 | 0.6% | 0.028 | 0.4% |
| **On-chip total** | **9.50** | 100% | **6.52** | 100% |
| Off-chip DRAM | 16 mm² (PCB) | — | 11.74 | — |

---

## 9. Simulator Implementation Notes

### 9.1 Cycle-accurate model

The simulator advances a single global cycle counter.  Each operation contributes
cycles from one of three sources:
- **Compute cycles**: RSA timing model (tile-based, weight-stationary)
- **Memory stall cycles**: weight fetch time minus compute time when fetch is bottleneck
- **Refresh stall cycles**: 10% of refresh duration when bank is concurrently accessed

### 9.2 Energy model

Energy is computed per-operation, not per-cycle:
- RSA compute: derived from power / (peak_ops/s) × ops
- Memory accesses: bytes × energy_per_byte from Table 1
- eDRAM refresh: 2 × bank_bytes × edram_energy_per_byte (read + rewrite)

### 9.3 Simulation vs. real cycle-accuracy

This is a **functional + timing model**, not a gate-level simulation.  Differences
from real silicon:
- Memory access scheduling is idealised (no bank conflicts modelled explicitly)
- RSA pipeline fill/drain is approximated with the +31 term
- Refresh scheduling uses a simplified 10% stall model
- Attention weights during decode are modelled as uniform (1/N') rather than real
  transformer attention — swap in real weights for accuracy evaluation

---

## 10. Comparison Methodology (vs Titanus)

To compare Kelle against your Titanus simulator:

1. Run both simulators with the same `--model opt-125m --prompt 128 --generate 64` settings.
2. Fill in `TITANUS_REFERENCE["opt-125m"]` in `run_simulation.py`.
3. Re-run with `--compare-titanus` to see the side-by-side table.

Key metrics to compare:
- Decode throughput (tokens/sec) — Kelle claims 3.9× speedup vs SRAM baseline
- Total energy per token (µJ/token) — Kelle claims 4.5× improvement
- KV cache utilisation (tokens cached with fixed memory budget)
- Eviction rate (how often tokens must be dropped or recomputed)
