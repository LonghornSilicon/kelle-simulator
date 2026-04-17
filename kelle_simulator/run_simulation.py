"""
run_simulation.py -- CLI entry point for the Kelle cycle-accurate simulator.

Usage examples:
  python -m kelle_simulator.run_simulation
  python -m kelle_simulator.run_simulation --model opt-125m --prompt 128 --generate 64
  python -m kelle_simulator.run_simulation --model opt-6.7b --prompt 512 --generate 128 --kv-capacity 512
  python -m kelle_simulator.run_simulation --model opt-125m --compare-titanus
  python -m kelle_simulator.run_simulation --sweep-kv-capacity
"""

from __future__ import annotations
import argparse
import dataclasses
import json
import sys
import os

# Allow `python run_simulation.py` to work from inside the package dir
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kelle_simulator.config import MODELS, HardwareConfig, DEFAULT_HW_CONFIG
from kelle_simulator.simulator import KelleSimulator


# -----------------------------------------------------------------------------
# Titanus reference numbers (from the Titanus simulator / paper)
# These are the values we compare against.  Update when you have real Titanus
# results from your Titanus simulator.
# -----------------------------------------------------------------------------

TITANUS_REFERENCE = {
    "opt-125m": {
        # From Titanus simulator run: OPT-125M, 32 prefill + 32 decode tokens
        # Run: python run_simulator.py --config_path config/config_opt125m_32_32_TT.yaml
        #       --constants_path constant/constants.yaml
        # Titanus uses HBM (256 GB/s), DCIM macros, CPQ compression, 12 parallel cores
        "prefill_latency_ms":    None,   # Titanus does not separate prefill/decode latency
        "decode_latency_ms":     None,
        "total_latency_ms":      3.404,  # intra-layer pipeline + inter-layer parallelism
        "throughput_toks_per_s": 18801.47,
        "total_energy_uj":       40642.69,
        "energy_per_tok_uj":     84.19,  # per Titanus internal accounting
        "area_mm2_per_core":     83.32,
        "area_mm2_total":        999.84,  # 12 cores
        "power_mw":              31950.15,
        "clock_mhz":             200,
        "memory_type":           "HBM 256 GB/s",
        "seq_config":            "32 prefill + 32 decode",
        "notes": (
            "Titanus: DCIM-based accelerator (GLSVLSI 2025). "
            "Latency is post-pipeline-optimization (intra+inter-layer). "
            "Energy metric denominator differs from Kelle (see ARCHITECTURE.md). "
            "Kelle comparison run: 32 prefill + 32 decode for consistency."
        ),
    }
}


def run(args: argparse.Namespace) -> KelleSimulator:
    if args.fpga:
        args.weight_bits = 4
        if args.weight_prefetch_mb == 0:
            args.weight_prefetch_mb = 45

    model_cfg = MODELS[args.model]
    if args.weight_bits != model_cfg.weight_bits:
        model_cfg = dataclasses.replace(model_cfg, weight_bits=args.weight_bits)

    hw_cfg    = HardwareConfig(
        kv_cache_capacity_tokens=args.kv_capacity,
        recent_window_tokens=args.recent_window,
        initial_tokens_preserved=args.initial_tokens,
        weight_prefetch_buffer_bytes=args.weight_prefetch_mb * 1024 * 1024,
    )

    sim = KelleSimulator(model_cfg, hw_cfg)
    sim.simulate(args.prompt, args.generate, verbose=not args.quiet)
    sim.print_summary()

    if args.json_out:
        _export_json(sim, args.json_out)
        print(f"Results written to {args.json_out}")

    if args.compare_titanus:
        _compare_titanus(sim, args.model)

    return sim


def _export_json(sim: KelleSimulator, path: str) -> None:
    s = sim.stats
    out = {
        "model": sim.model.name,
        "hardware": {
            "rsa_rows": sim.hw.rsa_rows,
            "rsa_cols": sim.hw.rsa_cols,
            "clock_ghz": sim.hw.clock_freq_hz / 1e9,
            "sram_mb": sim.hw.sram_size_bytes / 1024**2,
            "edram_kvcache_mb": sim.hw.edram_kvcache_bytes / 1024**2,
            "edram_activation_kb": sim.hw.edram_activation_bytes / 1024,
            "dram_gb": sim.hw.dram_size_gb,
            "kv_capacity_tokens": sim.hw.kv_cache_capacity_tokens,
        },
        "latency": {
            "prefill_ms": s.prefill_latency_ms,
            "decode_ms": s.decode_latency_ms,
            "total_ms": s.total_latency_ms,
            "avg_decode_step_ms": (
                sum(s.decode_latency_per_step) /
                max(len(s.decode_latency_per_step), 1) / 1e6
            ),
        },
        "throughput_tokens_per_sec": s.throughput_tokens_per_sec(),
        "energy": {
            "total_uj": s.total_energy_pj / 1e6,
            "prefill_uj": s.prefill.total_energy_uj,
            "decode_uj": s.decode.total_energy_uj,
            "decode_breakdown": {
                "compute_uj":          s.decode.compute_energy_pj / 1e6,
                "sram_uj":             s.decode.sram_energy_pj / 1e6,
                "edram_kv_uj":         s.decode.edram_energy_pj / 1e6,
                "edram_refresh_uj":    s.decode.edram_refresh_energy_pj / 1e6,
                "dram_uj":             s.decode.dram_energy_pj / 1e6,
                "sfu_uj":              s.decode.sfu_energy_pj / 1e6,
                "evictor_uj":          s.decode.evictor_energy_pj / 1e6,
            },
        },
        "aerp": {
            "eviction_events":      s.eviction_events,
            "recompute_events":     s.recompute_events,
            "cache_hit_rate":       sim.aerp.cache_hit_rate,
            "kv_bytes_saved_kb":    sim.aerp.stats.kv_bytes_saved_by_recompute / 1024,
        },
        "edram_2drp": {
            "total_refresh_ops":    len(sim.edram_ctrl.refresh_log),
            "refresh_energy_uj":    sim.edram_ctrl.total_refresh_energy_pj / 1e6,
            "expected_bit_flips_msb": s.total_bit_flips_msb,
            "expected_bit_flips_lsb": s.total_bit_flips_lsb,
            "avg_refresh_interval_msb_ms": sim.edram_ctrl.average_refresh_interval_ms()['msb'],
            "avg_refresh_interval_lsb_ms": sim.edram_ctrl.average_refresh_interval_ms()['lsb'],
        },
        "compute": {
            "peak_tops":          sim.rsa.peak_tops,
            "effective_tops":     sim.rsa.effective_tops,
            "utilisation_pct":    100 * sim.rsa.effective_tops / sim.rsa.peak_tops,
            "total_macs":         (s.prefill.ops + s.decode.ops) // 2,
        },
        "memory": {
            "sram_resident_layers": len(sim.memory.sram_resident_layers),
            "dram_layers":          sim.memory.num_dram_layers,
            "dram_bytes_transferred": sim.memory.dram.total_bytes_transferred,
            "edram_utilisation_pct":  100 * sim.memory.kv_edram.utilisation,
        },
        "kv_occupancy_log": s.kv_occupancy_log,
        "decode_latency_per_step_cycles": s.decode_latency_per_step,
    }
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)


def _compare_titanus(sim: KelleSimulator, model_key: str) -> None:
    ref = TITANUS_REFERENCE.get(model_key, {})
    s   = sim.stats

    seq_note = ref.get('seq_config', 'see TITANUS_REFERENCE')
    print(f"\n-- Kelle vs Titanus comparison ({model_key}, {seq_note}) --")
    print(f"  {'Metric':<34}  {'Kelle':>14}  {'Titanus':>14}  {'Ratio (K/T)':>12}")
    print(f"  {'-'*34}  {'-'*14}  {'-'*14}  {'-'*12}")

    def row(name, kelle_val_str, titanus_val, lower_is_better=True):
        if titanus_val is not None:
            try:
                ratio = float(kelle_val_str) / float(titanus_val)
                direction = "worse" if (ratio > 1) == lower_is_better else "better"
                ratio_str = f"{ratio:.2f}x ({direction})"
            except (ValueError, ZeroDivisionError):
                ratio_str = "N/A"
            titanus_str = f"{titanus_val:>14.3f}"
        else:
            ratio_str = "N/A (pending)"
            titanus_str = f"{'(pending)':>14}"
        print(f"  {name:<34}  {kelle_val_str:>14}  {titanus_str}  {ratio_str:>12}")

    total_lat = s.total_latency_ms
    ref_total = ref.get('total_latency_ms') or (
        (ref.get('prefill_latency_ms') or 0) + (ref.get('decode_latency_ms') or 0)
        or None
    )

    n_decode = max(len(s.decode_latency_per_step), 1)
    kelle_uj_per_tok = s.total_energy_pj / 1e6 / n_decode

    row("Total latency (ms)",         f"{total_lat:.3f}",                ref_total)
    row("Prefill latency (ms)",        f"{s.prefill_latency_ms:.3f}",    ref.get('prefill_latency_ms'))
    row("Decode latency (ms)",         f"{s.decode_latency_ms:.3f}",     ref.get('decode_latency_ms'))
    row("Throughput (tok/s)",          f"{s.throughput_tokens_per_sec():.1f}", ref.get('throughput_toks_per_s'), lower_is_better=False)
    row("Total energy (uJ)",           f"{s.total_energy_pj/1e6:.2f}",  ref.get('total_energy_uj'))
    row("Energy/decode-token (uJ)",    f"{kelle_uj_per_tok:.2f}",        ref.get('energy_per_tok_uj'))
    row("On-chip area (mm2)",          f"{sim.hw.total_on_chip_area_mm2:.1f}",
        ref.get('area_mm2_per_core'))
    row("On-chip power (W)",            f"{sim.hw.total_power_w:.2f}",    ref.get('power_mw') and ref['power_mw']/1000)

    print(f"\n  Kelle memory:   LPDDR-style DRAM 64 GB/s, 4 MB eDRAM KV cache, AERP eviction")
    print(f"  Titanus memory: HBM 256 GB/s, 4 MB global SRAM buffer, CPQ KV compression")
    print(f"  Target context: Kelle=edge/IoT (9.5 mm2), Titanus=datacenter (999 mm2 x12 cores)")
    print(f"\n  Notes: {ref.get('notes', 'N/A')}")
    print()


def sweep_kv_capacity(model_key: str = "opt-125m",
                      prompt: int = 128,
                      generate: int = 64,
                      capacities: list = None) -> None:
    """Sweep over KV cache capacities and print a comparison table."""
    if capacities is None:
        capacities = [32, 64, 128, 256, 512]

    model_cfg = MODELS[model_key]
    print(f"\nKV-cache capacity sweep -- {model_cfg.name}")
    print(f"{'Capacity':>10}  {'Prefill ms':>10}  {'Decode ms':>10}  "
          f"{'Toks/s':>8}  {'Energy uJ':>10}  {'Evictions':>10}  {'Recomputes':>10}")
    print("-" * 80)

    for cap in capacities:
        hw = HardwareConfig(kv_cache_capacity_tokens=cap)
        sim = KelleSimulator(model_cfg, hw)
        sim.simulate(prompt, generate, verbose=False)
        s = sim.stats
        print(f"{cap:>10}  {s.prefill_latency_ms:>10.3f}  {s.decode_latency_ms:>10.3f}  "
              f"{s.throughput_tokens_per_sec():>8.1f}  "
              f"{s.total_energy_pj/1e6:>10.2f}  "
              f"{s.eviction_events:>10}  {s.recompute_events:>10}")


# -----------------------------------------------------------------------------
# CLI argument parser
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Kelle accelerator cycle-accurate simulator (MICRO 2025)")
    p.add_argument("--model",           default="opt-125m",
                   choices=list(MODELS.keys()),
                   help="Model to simulate (default: opt-125m)")
    p.add_argument("--prompt",          type=int, default=128,
                   help="Prompt (prefill) length in tokens (default: 128)")
    p.add_argument("--generate",        type=int, default=64,
                   help="Number of decode steps to simulate (default: 64)")
    p.add_argument("--kv-capacity",     type=int, default=128,
                   help="KV cache token capacity N' (default: 128)")
    p.add_argument("--recent-window",   type=int, default=64,
                   help="Recent-token window size (default: 64)")
    p.add_argument("--initial-tokens",  type=int, default=10,
                   help="Number of initial tokens always preserved (default: 10)")
    p.add_argument("--json-out",        type=str, default=None,
                   help="Write results to a JSON file")
    p.add_argument("--compare-titanus", action="store_true",
                   help="Print comparison table vs. Titanus baseline")
    p.add_argument("--sweep-kv-capacity", action="store_true",
                   help="Sweep KV cache capacity [32,64,128,256,512] and tabulate")
    p.add_argument("--quiet",           action="store_true",
                   help="Suppress step-by-step progress output")
    p.add_argument("--weight-bits",        type=int, default=8, choices=[4, 8, 16],
                   help="Weight quantization bits (4=INT4, 8=INT8, 16=FP16) (default: 8)")
    p.add_argument("--weight-prefetch-mb", type=int, default=0,
                   help="On-chip weight prefetch buffer size in MB. "
                        "When >= model weight size, weights load once from DRAM at prefill start. "
                        "Set to 45 for OPT-125M INT4 on FPGA. (default: 0 = disabled)")
    p.add_argument("--fpga",               action="store_true",
                   help="FPGA mode: INT4 weights + 45 MB prefetch buffer (shortcut for "
                        "--weight-bits 4 --weight-prefetch-mb 45)")
    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.sweep_kv_capacity:
        sweep_kv_capacity(args.model, args.prompt, args.generate)
        return

    run(args)


if __name__ == "__main__":
    main()
