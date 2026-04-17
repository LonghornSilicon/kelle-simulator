# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the simulator

```bash
# Default run (OPT-125M, 128-token prompt, 64 decode steps)
python -m kelle_simulator.run_simulation

# Full options
python -m kelle_simulator.run_simulation \
    --model opt-125m \
    --prompt 128 \
    --generate 64 \
    --kv-capacity 128 \
    --json-out results.json \
    --quiet

# Sweep KV cache capacity trade-off
python -m kelle_simulator.run_simulation --sweep-kv-capacity

# Compare against Titanus baseline
python -m kelle_simulator.run_simulation --compare-titanus
```

No dependencies — stdlib only (Python 3.8+). No install step required.

## Architecture

This is a functional+timing simulator (not gate-level). A single global cycle counter (`KelleSimulator._cycle`) advances through every operation. Each subsystem returns a result object; the simulator accumulates cycles and energy into `PhaseStats` (prefill and decode tracked separately).

**Critical data flow during decode:**
1. `simulator._decode_step()` calls `aerp.insert_token()` — may trigger eviction via `evictor.find_eviction_candidate()`
2. For each layer: weight load (SRAM or DRAM) → QKV via RSA → attention scores via RSA → softmax via SFU → evictor score update → attention output via RSA → FFN via RSA → KV write to eDRAM
3. After all layers: `edram_ctrl.reclassify_tokens()` updates HST/LST status for 2DRP refresh intervals
4. `edram_ctrl.advance_to_cycle()` is called on every `_tick()` — issues any overdue eDRAM refreshes

**Evictor and AERP coupling:** The `SystolicEvictor` only tracks tokens currently in the AERP cache. `find_eviction_candidate()` recomputes is_initial/is_recent dynamically from dict insertion order (Python 3.7+ ordering = token insertion sequence). The recent window is capped at `num_cached // 2` to guarantee at least half the cache is always evictable.

**Energy attribution:** `ComputeResult` separates `rsa_energy_pj`, `weight_energy_pj` (routes to `sram_energy_pj` or `dram_energy_pj` depending on `weight_from_dram`), and `act_energy_pj`. For OPT-125M (81 MB > 2 MB SRAM), DRAM weight streaming dominates (~95% of decode energy).

**Attention weights are uniform** (`1/kv_len`) in the current simulator. This means recompute events never trigger (all tokens get equal importance). Replace `attn_weights` in `simulator._decode_step()` to use real scores from a model for AERP recomputation to activate.

## Key extension points

| Goal | Location |
|------|----------|
| Real attention weights | `simulator.py:_decode_step()` — replace the `attn_weights = {tid: 1/kv_len}` dict |
| New model (LLaMA, Mistral) | Add `ModelConfig` entry to `config.py:MODELS` |
| INT4 quantisation | `config.py:ModelConfig.weight_bits` and `kv_bits` |
| Titanus comparison numbers | `run_simulation.py:TITANUS_REFERENCE` dict |
| Bank conflict modelling | `memory.py:KVCacheEDRAM.store_kv()` |
| Gate-level refresh energy | `edram_controller.py:EDRAMBank.refresh()` |

## Research context

This is part of the Longhorn Silicon FPGA prototype project comparing Kelle (eDRAM + AERP + 2DRP) against the Titanus sparse-attention accelerator. The goal is to identify architectural ideas to combine in a new KV-cache compression accelerator. See `ARCHITECTURE.md` for the full hardware spec derived from the MICRO 2025 paper (arXiv 2510.16040).
