# Kelle Cycle-Accurate Simulator

Cycle-accurate simulator for the **Kelle** accelerator (MICRO 2025).  
Models KV-cache-optimised LLM inference on an edge ASIC using eDRAM with  
adaptive refresh, attention-based eviction, and a reconfigurable systolic array.

> **Research context**: Part of the Longhorn Silicon FPGA prototype project comparing
> Kelle's eDRAM+AERP approach against the Titanus sparse-attention accelerator, to
> identify which architectural ideas to combine in a new KV-cache compression
> accelerator.

---

## Quick Start

```bash
# Default: OPT-125M, 128-token prompt, 64 decode steps
python -m kelle_simulator.run_simulation

# Larger run with JSON export
python -m kelle_simulator.run_simulation \
    --model opt-125m \
    --prompt 256 \
    --generate 128 \
    --kv-capacity 128 \
    --json-out results/opt125m_p256_g128.json

# Compare against Titanus (fill in TITANUS_REFERENCE first)
python -m kelle_simulator.run_simulation --compare-titanus

# Sweep KV capacity to find the accuracy/latency/energy trade-off
python -m kelle_simulator.run_simulation --sweep-kv-capacity

# OPT-6.7B (paper's primary evaluation model)
python -m kelle_simulator.run_simulation \
    --model opt-6.7b \
    --prompt 512 \
    --generate 64 \
    --kv-capacity 512
```

### No dependencies required

The simulator uses only Python standard library (`math`, `dataclasses`, `json`,
`argparse`, `time`).  No NumPy, PyTorch, or CUDA needed.

```bash
python --version   # requires Python 3.8+
```

---

## Command-Line Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `opt-125m` | Model: `opt-125m`, `opt-1.3b`, `opt-6.7b` |
| `--prompt` | `128` | Prefill (prompt) token length |
| `--generate` | `64` | Number of decode steps to simulate |
| `--kv-capacity` | `128` | KV cache token budget N' |
| `--recent-window` | `64` | Recent-token window (always retained) |
| `--initial-tokens` | `10` | Initial tokens always retained |
| `--json-out` | `None` | Path for JSON results export |
| `--compare-titanus` | `False` | Print Kelle vs Titanus table |
| `--sweep-kv-capacity` | `False` | Sweep N'∈{32,64,128,256,512} |
| `--quiet` | `False` | Suppress step progress output |

---

## Simulator Architecture

```
kelle_simulator/
├── config.py           # ModelConfig + HardwareConfig (all paper numbers)
├── memory.py           # WeightSRAM, KVCacheEDRAM (32 banks), ActivationEDRAM, OffChipDRAM
├── edram_controller.py # 2DRP controller: HST/LST × MSB/LSB refresh scheduling
├── aerp.py             # Attention-based Eviction and Recomputation Policy
├── systolic_evictor.py # Importance-score accumulator + min-search register chain
├── rsa.py              # 32×32 systolic array timing model (prefill + decode ops)
├── sfu.py              # Softmax, LayerNorm, GELU, RoPE function units
├── simulator.py        # Top-level: prefill + decode loop, statistics collection
└── run_simulation.py   # CLI entry point + JSON export + Titanus comparison
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the complete hardware specification.

---

## Key Parameters (from the paper)

### Hardware (Kelle ASIC)

| Parameter | Value |
|-----------|-------|
| On-chip area | 9.5 mm² |
| On-chip power | 6.52 W |
| RSA | 32×32 PEs, 8-bit MAC, 4.13 INT8 TOPs |
| Clock | 1 GHz |
| Weight SRAM | 2 MB, 128 GB/s, 185.9 pJ/B |
| KV-cache eDRAM | 4 MB, 256 GB/s, 84.8 pJ/B |
| Activation eDRAM | 256 KB |
| Off-chip DRAM | 16 GB, 64 GB/s |

### 2DRP Refresh Intervals

| Group | Interval |
|-------|----------|
| HST-MSB (bits [15:8], high importance) | 0.36 ms |
| LST-MSB (bits [15:8], low importance) | 1.44 ms |
| HST-LSB (bits [7:0], high importance) | 5.40 ms |
| LST-LSB (bits [7:0], low importance) | 7.20 ms |

Target retention failure rate: 2×10⁻³

### AERP

| Parameter | Value |
|-----------|-------|
| KV-cache capacity N' | 128 (std) / 512 (WikiText) / 2048 (PG19) |
| Recent token window | 64 |
| Initial tokens preserved | 10 |
| Recompute threshold | ≥50% of heads vote important |

---

## Output Format

### Console

```
============================================================
  Kelle Cycle-Accurate Simulator
  Model : OPT-125M
  Prompt: 128 tokens  |  Generate: 64 tokens
  KV-cache capacity: 128 tokens
  SRAM-resident layers: 0/12
============================================================

[Prefill] Processing 128 tokens...
  Prefill complete: 2,847,392 cycles  (2.847 ms)  152.33 µJ

[Decode] Generating 64 tokens...
  step    1/64  cycle=     3,194,881  kv=128C/  0R  evictions=0  recomputes=0
  ...

== KELLE SIMULATION RESULTS — OPT-125M ==
── Latency ──────────────────────────────
  Prefill latency   :      2.847 ms   (2,847,392 cycles)
  Decode latency    :     22.361 ms  (22,361,040 cycles)
  Throughput        :   2,861.43 tokens/sec
── Energy ───────────────────────────────
  Total energy      :    947.28 µJ
  ── eDRAM (refresh): 2DRP vs uniform ──
  ...
```

### JSON (`--json-out results.json`)

```json
{
  "model": "OPT-125M",
  "latency": { "prefill_ms": 2.847, "decode_ms": 22.361, ... },
  "throughput_tokens_per_sec": 2861.43,
  "energy": { "total_uj": 947.28, "decode_breakdown": { ... } },
  "aerp": { "eviction_events": 12, "recompute_events": 8, ... },
  "edram_2drp": { "refresh_energy_uj": 18.4, ... },
  "kv_occupancy_log": [ ... ],
  "decode_latency_per_step_cycles": [ ... ]
}
```

---

## Comparing to Titanus

1. Run your Titanus simulator with matching parameters.
2. Open `kelle_simulator/run_simulation.py` and fill in `TITANUS_REFERENCE`:

```python
TITANUS_REFERENCE = {
    "opt-125m": {
        "prefill_latency_ms":    <value from Titanus>,
        "decode_latency_ms":     <value from Titanus>,
        "throughput_toks_per_s": <value from Titanus>,
        "total_energy_uj":       <value from Titanus>,
    }
}
```

3. Re-run with `--compare-titanus`:

```
── Kelle vs Titanus comparison ─────────────────────────
  Metric                          Kelle      Titanus    Ratio
  Prefill latency (ms)            2.847      X.XXX      X.XXx
  Decode latency (ms)            22.361      X.XXX      X.XXx
  Throughput (tok/s)           2861.43      XXXX.X      X.XXx
  Total energy (µJ)              947.28      XXXX.X      X.XXx
```

---

## Customising the Hardware

```python
from kelle_simulator.config import HardwareConfig, MODELS
from kelle_simulator.simulator import KelleSimulator

hw = HardwareConfig(
    kv_cache_capacity_tokens=256,
    recent_window_tokens=32,
    # Override 2DRP intervals (seconds)
    refresh_hst_msb_s=200e-6,
    refresh_lst_lsb_s=10000e-6,
    # Swap to larger RSA
    rsa_rows=64,
    rsa_cols=64,
)

sim = KelleSimulator(MODELS["opt-125m"], hw)
stats = sim.simulate(prompt_len=256, num_decode_steps=128)
sim.print_summary()
```

---

## Extending the Simulator

| What to add | Where |
|-------------|-------|
| Real attention weights (not uniform) | `simulator.py:_decode_step()` — replace `attn_weights = {tid: 1/kv_len}` with real scores |
| Sliding window attention | `aerp.py:insert_token()` — change eviction logic |
| Different quantisation (INT4) | `config.py:ModelConfig.weight_bits` / `kv_bits` |
| New model (LLaMA, Mistral) | Add entry to `config.py:MODELS` |
| Gate-level refresh energy | `edram_controller.py:EDRAMBank.refresh()` |
| Bank conflict modelling | `memory.py:KVCacheEDRAM.store_kv()` — add contention logic |

---

## Paper Reference

```bibtex
@inproceedings{xia2025kelle,
  title     = {Kelle: Co-design KV Caching and eDRAM for Efficient LLM Serving
               in Edge Computing},
  author    = {Tianhua Xia and Sai Qian Zhang},
  booktitle = {Proceedings of the 58th IEEE/ACM International Symposium on
               Microarchitecture (MICRO)},
  year      = {2025},
  month     = {October},
  address   = {Seoul, Korea},
}
```
