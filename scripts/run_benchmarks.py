#!/usr/bin/env python3
"""
AL-BH Automated Benchmark & Scaling Sweep Suite (Milestone 7)

Runs:
  1. Strong Scaling Sweep (1 to 8 threads, fixed N)
  2. Weak Scaling Sweep (N scales with threads)
  3. Load Imbalance Comparison (Static Geometric vs Adaptive Morton)
  4. Auto-generates publication-quality scaling and efficiency plots

Usage:
    python3 scripts/run_benchmarks.py --quick
    python3 scripts/run_benchmarks.py --threads 1,2,4,8
"""

import os
import sys
import glob
import time
import argparse
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN_PATH = os.path.join(PROJECT_ROOT, "build", "al_bh")

def run_command(cmd):
    """Run shell command and return stdout."""
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error running: {cmd}\n{res.stderr}")
    return res.stdout

def run_strong_scaling(threads_list, N=3000, steps=20, out_dir="results/benchmarks/strong"):
    """Run strong scaling: fixed N, varying threads."""
    print(f"\n=======================================================")
    print(f"  Running Strong Scaling Benchmark (N={N}, Steps={steps})")
    print(f"=======================================================")
    os.makedirs(out_dir, exist_ok=True)
    
    results = []
    for t in threads_list:
        run_out = os.path.join(out_dir, f"t{t}")
        os.makedirs(run_out, exist_ok=True)
        cmd = f"{BIN_PATH} --N {N} --steps {steps} --dist 1 --threads {t} --mode full --output {run_out} --verbose 0"
        
        t0 = time.time()
        run_command(cmd)
        elapsed = time.time() - t0

        # Read actual force compute time from metrics CSV
        csv_files = glob.glob(os.path.join(run_out, "metrics_*.csv"))
        if csv_files:
            data = np.genfromtxt(csv_files[0], delimiter=',', names=True)
            force_time = np.sum(data['force_compute_time'])
            total_time = np.sum(data['total_step_time'])
        else:
            force_time = elapsed
            total_time = elapsed

        results.append({
            'threads': t,
            'force_time': force_time,
            'total_time': total_time
        })
        print(f"  Threads: {t:2d} | Force Time: {force_time:.4f}s | Total Step Time: {total_time:.4f}s")

    t1 = results[0]['force_time']
    for r in results:
        r['speedup'] = t1 / r['force_time'] if r['force_time'] > 0 else 1.0
        r['efficiency'] = r['speedup'] / r['threads']

    # Plot Strong Scaling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor='#0b0d14')
    fig.suptitle("AL-BH Strong Scaling Benchmark (Fixed N=3000, Plummer Galaxy)", color='white', fontsize=14, fontweight='bold')

    th = [r['threads'] for r in results]
    sp = [r['speedup'] for r in results]
    eff = [r['efficiency'] for r in results]

    # Speedup Plot
    ax1.set_facecolor('#0b0d14')
    ax1.plot(th, sp, 'o-', color='#38bdf8', lw=2.5, markersize=8, label='AL-BH Speedup')
    ax1.plot(th, th, '--', color='#64748b', lw=1.5, label='Ideal Linear Speedup')
    ax1.set_title("Parallel Speedup (S = T₁ / Tₚ)", color='#94a3b8', fontsize=12)
    ax1.set_xlabel("OpenMP Threads", color='#94a3b8')
    ax1.set_ylabel("Speedup", color='#94a3b8')
    ax1.tick_params(colors='#64748b')
    ax1.grid(True, color='#1e293b', linestyle=':')
    ax1.legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='white')

    # Parallel Efficiency Plot
    ax2.set_facecolor('#0b0d14')
    ax2.plot(th, eff, 's-', color='#34d399', lw=2.5, markersize=8, label='Parallel Efficiency')
    ax2.axhline(1.0, linestyle='--', color='#64748b', lw=1.5, label='Ideal (1.0)')
    ax2.set_title("Parallel Efficiency (E = S / p)", color='#94a3b8', fontsize=12)
    ax2.set_xlabel("OpenMP Threads", color='#94a3b8')
    ax2.set_ylabel("Efficiency", color='#94a3b8')
    ax2.set_ylim(0, 1.2)
    ax2.tick_params(colors='#64748b')
    ax2.grid(True, color='#1e293b', linestyle=':')
    ax2.legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='white')

    plt.tight_layout()
    out_fig = os.path.join(out_dir, "strong_scaling.png")
    plt.savefig(out_fig, dpi=180, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  -> Saved strong scaling plot: {out_fig}")
    return results

def run_imbalance_comparison(steps=25, out_dir="results/benchmarks/imbalance"):
    """Compare load imbalance between static geometric vs adaptive Morton."""
    print(f"\n=======================================================")
    print(f"  Running Imbalance Comparison: Static vs Morton (4 Ranks)")
    print(f"=======================================================")
    os.makedirs(out_dir, exist_ok=True)

    static_dir = os.path.join(out_dir, "static")
    morton_dir = os.path.join(out_dir, "morton")

    cmd_static = f"{BIN_PATH} --N 2000 --steps {steps} --dist 1 --threads 4 --mode mpi_static --output {static_dir} --verbose 0"
    cmd_morton = f"{BIN_PATH} --N 2000 --steps {steps} --dist 1 --threads 4 --mode mpi_morton --output {morton_dir} --verbose 0"

    print("  Executing Static Geometric Partitioning baseline...")
    run_command(cmd_static)
    print("  Executing Adaptive Morton Space-Filling Curve...")
    run_command(cmd_morton)

    # Load CSVs
    csv_static = glob.glob(os.path.join(static_dir, "metrics_*.csv"))[0]
    csv_morton = glob.glob(os.path.join(morton_dir, "metrics_*.csv"))[0]

    d_static = np.genfromtxt(csv_static, delimiter=',', names=True)
    d_morton = np.genfromtxt(csv_morton, delimiter=',', names=True)

    steps_arr = d_static['timestep']
    imb_static = d_static['imbalance_ratio']
    imb_morton = d_morton['imbalance_ratio']

    # Plot Comparison
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0b0d14')
    ax.set_facecolor('#0b0d14')
    ax.plot(steps_arr, imb_static, 'o-', color='#f43f5e', lw=2.5, label='Static Geometric Slicing (Baseline)')
    ax.plot(steps_arr, imb_morton, 's-', color='#38bdf8', lw=2.5, label='Adaptive Morton SFC (AL-BH)')
    ax.axhline(1.0, linestyle='--', color='#34d399', lw=1.5, label='Optimal Balance (1.0)')

    ax.set_title("Load Imbalance Ratio Over Time on Clustered Galaxy (N=2000, 4 Ranks)", color='white', fontsize=13, fontweight='bold')
    ax.set_xlabel("Timestep", color='#94a3b8')
    ax.set_ylabel("Load Imbalance max(W) / min(W)", color='#94a3b8')
    ax.tick_params(colors='#64748b')
    ax.grid(True, color='#1e293b', linestyle=':')
    ax.legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='white')

    plt.tight_layout()
    out_fig = os.path.join(out_dir, "imbalance_comparison.png")
    plt.savefig(out_fig, dpi=180, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  -> Saved load imbalance comparison plot: {out_fig}")

def main():
    parser = argparse.ArgumentParser(description="AL-BH Benchmarks & Scaling Sweeps")
    parser.add_argument("--quick", action="store_true", help="Run fast quick benchmark")
    parser.add_argument("--threads", type=str, default="1,2,4,8", help="Comma-separated thread counts")
    args = parser.parse_args()

    threads = [int(x) for x in args.threads.split(",")]
    N = 1500 if args.quick else 3000
    steps = 15 if args.quick else 25

    strong_results = run_strong_scaling(threads, N=N, steps=steps)
    run_imbalance_comparison(steps=steps)

    print("\n=======================================================")
    print("  BENCHMARK SUMMARY TABLE")
    print("=======================================================")
    print(f"{'Threads':<10} | {'Force Time (s)':<16} | {'Speedup':<10} | {'Efficiency':<10}")
    print("-" * 55)
    for r in strong_results:
        print(f"{r['threads']:<10} | {r['force_time']:<16.4f} | {r['speedup']:<10.2f}x | {r['efficiency']*100:<9.1f}%")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
