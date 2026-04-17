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
        "prefill_latency_ms":    None,   # fill in from your Titanus simulator
        "decode_latency_ms":     None,
        "throughput_toks_per_s": None,
        "total_energy_uj":       None,
        "notes": "placeholder -- run Titanus simulator to obtain reference values",
    }
}


def run(args: argparse.Namespace) -> KelleSimulator:
    model_cfg = MODELS[args.model]
    hw_cfg    = HardwareConfig(
        kv_cache_capacity_tokens=args.kv_capacity,
        recent_window_tokens=args.recent_window,
        initial_tokens_preserved=args.initial_tokens,
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

    print("\n-- Kelle vs Titanus comparison --------------------------")
    print(f"  {'Metric':<30}  {'Kelle':>12}  {'Titanus':>12}  {'Ratio':>8}")
    print(f"  {'-'*30}  {'-'*12}  {'-'*12}  {'-'*8}")

    metrics = [
        ("Prefill latency (ms)",   f"{s.prefill_latency_ms:.3f}",
         ref.get('prefill_latency_ms')),
        ("Decode latency (ms)",    f"{s.decode_latency_ms:.3f}",
         ref.get('decode_latency_ms')),
        ("Throughput (tok/s)",     f"{s.throughput_tokens_per_sec():.1f}",
         ref.get('throughput_toks_per_s')),
        ("Total energy (uJ)",      f"{s.total_energy_pj/1e6:.2f}",
         ref.get('total_energy_uj')),
    ]

    for name, kelle_val, titanus_val in metrics:
        if titanus_val is not None:
            try:
                ratio = float(kelle_val) / float(titanus_val)
                ratio_str = f"{ratio:.2f}x"
            except (ValueError, ZeroDivisionError):
                ratio_str = "N/A"
        else:
            ratio_str = "N/A"
        titanus_str = f"{titanus_val:.3f}" if titanus_val is not None else "(pending)"
        print(f"  {name:<30}  {kelle_val:>12}  {titanus_str:>12}  {ratio_str:>8}")

    print(f"\n  Notes: {ref.get('notes', 'N/A')}")
    print(f"  To add Titanus numbers, edit TITANUS_REFERENCE in run_simulation.py")
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
