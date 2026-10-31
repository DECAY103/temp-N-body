#!/usr/bin/env python3
"""Render one publication-ready PNG per evaluation metric from saved CSV files.

This does not run benchmarks.  Run ``evaluate_all_metrics.py --mpi`` first when
new data are required, then use this script to regenerate the individual plots.
"""

from pathlib import Path
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "results" / "evaluation"
OUT = EVALUATION / "plots"


def read_metrics(directory):
    files = glob.glob(str(Path(directory) / "metrics_*.csv"))
    if not files:
        raise FileNotFoundError(f"No metrics CSV in {directory}")
    data = np.genfromtxt(max(files, key=lambda p: Path(p).stat().st_mtime),
                         delimiter=",", names=True)
    if data.ndim == 0:
        data = np.array([data])
    return data[data["rank"] == 0] if "rank" in data.dtype.names else data


def make_axes(title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(8.5, 5.5), facecolor="#08090e")
    ax.set_facecolor("#0f111a")
    ax.set_title(title, color="#38bdf8", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, color="#94a3b8", fontsize=11)
    ax.set_ylabel(ylabel, color="#94a3b8", fontsize=11)
    ax.tick_params(colors="#94a3b8")
    ax.grid(True, color="#1e293b", linestyle=":", alpha=0.9)
    return fig, ax


def finish(fig, ax, filename):
    legend = ax.get_legend()
    if legend:
        legend.get_frame().set_facecolor("#1e293b")
        legend.get_frame().set_edgecolor("#334155")
        for text in legend.get_texts():
            text.set_color("white")
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    strong = []
    for threads in (1, 2, 4, 8):
        data = read_metrics(EVALUATION / "strong" / f"t{threads}")
        strong.append((threads, float(np.sum(data["total_step_time"]))))
    threads = [item[0] for item in strong]
    total_times = [item[1] for item in strong]
    speedups = [total_times[0] / value for value in total_times]

    fig, ax = make_axes("1. Parallel Speedup (Strong Scaling)", "OpenMP Threads",
                        "Speedup (S = T₁ / Tₚ)")
    ax.plot(threads, speedups, "o-", color="#38bdf8", lw=2.5, markersize=8,
            label="Measured total-runtime speedup")
    ax.plot(threads, threads, "--", color="#64748b", lw=1.5, label="Ideal linear")
    ax.set_xticks(threads)
    ax.legend()
    finish(fig, ax, "01_parallel_speedup.png")

    fig, ax = make_axes("2. Parallel Efficiency (Strong Scaling)", "OpenMP Threads",
                        "Efficiency (E = S / p)")
    ax.plot(threads, [speedup / p for speedup, p in zip(speedups, threads)], "s-",
            color="#34d399", lw=2.5, markersize=8, label="Measured efficiency")
    ax.axhline(1.0, linestyle="--", color="#64748b", lw=1.5, label="Ideal 100%")
    ax.set_xticks(threads)
    ax.set_ylim(0, 1.2)
    ax.legend()
    finish(fig, ax, "02_parallel_efficiency.png")

    weak = []
    for threads_count in (1, 2, 4, 8):
        data = read_metrics(EVALUATION / "weak" / f"t{threads_count}")
        weak.append((threads_count, float(np.sum(data["total_step_time"]))))
    weak_threads = [item[0] for item in weak]
    weak_times = [item[1] for item in weak]
    fig, ax = make_axes("3. Weak Scaling (Work per Thread Constant)", "OpenMP Threads",
                        "Total Execution Time (s)")
    ax.plot(weak_threads, weak_times, "^-", color="#facc15", lw=2.5, markersize=8,
            label="Measured weak scaling")
    ax.axhline(weak_times[0], linestyle="--", color="#64748b", lw=1.5,
               label="Ideal constant time")
    ax.set_xticks(weak_threads)
    ax.legend()
    finish(fig, ax, "03_weak_scaling.png")

    static = read_metrics(EVALUATION / "imbalance" / "static")
    adaptive = read_metrics(EVALUATION / "imbalance" / "adaptive")
    fig, ax = make_axes("4. Load Imbalance: Static vs. Adaptive", "Timestep",
                        "Imbalance Ratio max(W) / min(W)")
    ax.plot(static["timestep"], static["imbalance_ratio"], "o-", color="#f43f5e",
            lw=2.0, markersize=5, label="Static geometric slicing")
    ax.plot(adaptive["timestep"], adaptive["imbalance_ratio"], "s-", color="#38bdf8",
            lw=2.0, markersize=5, label="Adaptive from static start")
    ax.axhline(1.0, linestyle="--", color="#34d399", lw=1.5, label="Optimal (1.0)")
    # A ratio of max/min is never zero; logarithmic scaling also keeps the
    # near-balanced adaptive series visible beside the static series.
    ax.set_yscale("log")
    ax.set_ylim(bottom=1.0)
    ax.legend()
    finish(fig, ax, "04_load_imbalance.png")

    linearized = read_metrics(EVALUATION / "octree_comparison" / "linearized")
    pointer = read_metrics(EVALUATION / "octree_comparison" / "pointer")
    labels = ["Tree Build", "Force Traversal"]
    linearized_ms = [float(np.mean(linearized["tree_build_time"])) * 1000,
                     float(np.mean(linearized["force_compute_time"])) * 1000]
    pointer_ms = [float(np.mean(pointer["tree_build_time"])) * 1000,
                  float(np.mean(pointer["force_compute_time"])) * 1000]
    x = np.arange(len(labels))
    fig, ax = make_axes("5. Octree Implementation Timing (Cache-Locality Proxy)",
                        "Execution Phase", "Mean Time per Timestep (ms)")
    width = 0.35
    ax.bar(x - width / 2, linearized_ms, width, color="#38bdf8",
           label="Linearized octree (array)")
    ax.bar(x + width / 2, pointer_ms, width, color="#fb7185", label="Pointer octree (heap)")
    ax.set_xticks(x, labels)
    ax.legend()
    finish(fig, ax, "05_octree_timing.png")

    theta_values, force_ms = [], []
    for theta in (0.3, 0.5, 0.7, 1.0):
        data = read_metrics(EVALUATION / "theta_tradeoff" / f"th_{theta}")
        theta_values.append(theta)
        force_ms.append(float(np.mean(data["force_compute_time"])) * 1000)
    fig, ax = make_axes("6. Opening Angle (θ) vs. Force Compute Time", "Opening Angle θ",
                        "Mean Force Compute Time (ms)")
    ax.plot(theta_values, force_ms, "d-", color="#a855f7", lw=2.5, markersize=8,
            label="Measured force time")
    ax.legend()
    finish(fig, ax, "06_theta_force_time.png")

    print(f"Saved six individual plots to {OUT}")


if __name__ == "__main__":
    main()
