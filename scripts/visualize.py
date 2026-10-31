#!/usr/bin/env python3
"""
AL-BH Simulation Visualization & Analysis Suite

Usage:
    python3 scripts/visualize.py --results results/demo
    python3 scripts/visualize.py --results results/demo --gif
    python3 scripts/visualize.py --results results/demo --metrics-only
"""

import os
import sys
import glob
import struct
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless rendering
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def load_particle_snapshot(filepath):
    """
    Read binary particle snapshot produced by dump_particle_state().
    Format:
      - int32: N
      - N * [double x, y, z, vx, vy, vz, mass, int32 rank]
    """
    with open(filepath, 'rb') as f:
        data = f.read()
    
    N = struct.unpack('<i', data[:4])[0]
    expected_size = 4 + N * 60
    if len(data) != expected_size:
        raise ValueError(f"Corrupt binary file {filepath}: size {len(data)} != {expected_size}")
    
    # 7 doubles (56 bytes) + 1 int32 (4 bytes) per particle = 60 bytes
    dtype = np.dtype([
        ('x', '<f8'), ('y', '<f8'), ('z', '<f8'),
        ('vx', '<f8'), ('vy', '<f8'), ('vz', '<f8'),
        ('mass', '<f8'), ('rank', '<i4')
    ])
    particles = np.frombuffer(data[4:], dtype=dtype, count=N)
    return particles

def plot_metrics(csv_path, output_dir):
    """Plot energy conservation and time breakdown from metrics CSV."""
    if not os.path.isfile(csv_path):
        print(f"[Viz] Metrics file not found: {csv_path}")
        return

    data = np.genfromtxt(csv_path, delimiter=',', names=True)
    if data.ndim == 0:
        data = np.array([data])
    if 'rank' in data.dtype.names:
        data = data[data['rank'] == 0]
    steps = data['timestep']
    ke = data['kinetic_energy']
    pe = data['potential_energy']
    te = ke + pe
    t_force = data['force_compute_time']
    t_tree = data['tree_build_time']
    t_total = data['total_step_time']

    # Initial energy for relative drift
    e0 = te[0] if te[0] != 0 else 1.0
    rel_error = np.abs((te - e0) / e0)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("AL-BH Simulation Metrics & Physics Diagnostics", fontsize=16, fontweight='bold')

    # 1. Energy evolution
    axs[0, 0].plot(steps, ke, label='Kinetic Energy (KE)', color='#e74c3c', lw=2)
    axs[0, 0].plot(steps, pe, label='Potential Energy (PE)', color='#3498db', lw=2)
    axs[0, 0].plot(steps, te, label='Total Energy (E)', color='#2ecc71', lw=2, linestyle='--')
    axs[0, 0].set_title("Energy Components vs. Timestep")
    axs[0, 0].set_xlabel("Timestep")
    axs[0, 0].set_ylabel("Energy")
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    # 2. Energy drift (log scale)
    axs[0, 1].plot(steps, rel_error, color='#9b59b6', lw=2)
    axs[0, 1].set_title(r"Relative Energy Drift $|(E - E_0) / E_0|$")
    axs[0, 1].set_xlabel("Timestep")
    axs[0, 1].set_ylabel("Relative Error")
    axs[0, 1].set_yscale('log')
    axs[0, 1].grid(True, which="both", alpha=0.3)

    # 3. Stage Runtime Breakdown
    axs[1, 0].plot(steps, t_force * 1000, label='Force Computation', color='#e67e22', lw=2)
    axs[1, 0].plot(steps, t_tree * 1000, label='Octree Construction', color='#1abc9c', lw=2)
    axs[1, 0].plot(steps, t_total * 1000, label='Total Step Time', color='#34495e', lw=2, linestyle=':')
    axs[1, 0].set_title("Runtime Breakdown per Timestep")
    axs[1, 0].set_xlabel("Timestep")
    axs[1, 0].set_ylabel("Time (ms)")
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)

    # 4. Force vs Tree Time Ratio
    ratio = (t_force / np.maximum(t_tree, 1e-9))
    axs[1, 1].plot(steps, ratio, color='#d35400', lw=2)
    axs[1, 1].set_title("Compute vs Tree Build Ratio (Force / Tree)")
    axs[1, 1].set_xlabel("Timestep")
    axs[1, 1].set_ylabel("Ratio")
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_file = os.path.join(output_dir, "simulation_metrics.png")
    plt.savefig(out_file, dpi=180)
    plt.close()
    print(f"[Viz] Saved metrics plot: {out_file}")

def plot_snapshots_2d(particle_files, output_dir):
    """Plot 2D projection comparison of initial vs mid vs final state."""
    if not particle_files:
        print("[Viz] No particle snapshots found to plot.")
        return

    n_files = len(particle_files)
    indices = [0, n_files // 2, n_files - 1]
    chosen = sorted(list(set(indices)))

    fig, axs = plt.subplots(1, len(chosen), figsize=(5 * len(chosen), 5))
    if len(chosen) == 1:
        axs = [axs]

    fig.suptitle("N-Body Gravitational Evolution (X-Y Projection)", fontsize=14, fontweight='bold')

    for ax, idx in zip(axs, chosen):
        pfile = particle_files[idx]
        step_num = int(os.path.basename(pfile).replace('step_', '').replace('.bin', ''))
        particles = load_particle_snapshot(pfile)
        
        # Color by velocity magnitude
        speed = np.sqrt(particles['vx']**2 + particles['vy']**2 + particles['vz']**2)
        scatter = ax.scatter(particles['x'], particles['y'], c=speed, cmap='plasma', s=8, alpha=0.7)
        ax.set_title(f"Timestep {step_num} (N={len(particles)})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect('equal', 'box')
        ax.grid(True, alpha=0.2)
        fig.colorbar(scatter, ax=ax, label='Speed |v|', fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_file = os.path.join(output_dir, "particle_evolution_2d.png")
    plt.savefig(out_file, dpi=180)
    plt.close()
    print(f"[Viz] Saved 2D particle evolution plot: {out_file}")

def plot_3d_state(particle_files, output_dir):
    """
    Generate 3D perspective visualization showing:
      1. 3D particle positions colored by speed/rank.
      2. Morton Space-Filling Curve (Z-curve) threading through particles.
      3. 3D orbital trajectory trails of fastest core particles.
    """
    if not particle_files:
        return

    p_final = load_particle_snapshot(particle_files[-1])
    p_init = load_particle_snapshot(particle_files[0])

    fig = plt.figure(figsize=(16, 7), facecolor='#0b0d14')
    fig.suptitle("AL-BH 3D Spatial Hierarchy & Space-Filling Curve (Morton)", fontsize=15, color='white', fontweight='bold')

    # Panel 1: 3D Morton Space-Filling Curve
    ax1 = fig.add_subplot(1, 2, 1, projection='3d', facecolor='#0b0d14')
    ax1.set_title("Morton Space-Filling Curve (Z-Order)", color='#94a3b8', fontsize=12)
    
    # Take a representative subset for clean curve visualization
    sub_n = min(250, len(p_init))
    sub_p = p_init[:sub_n]
    ax1.plot(sub_p['x'], sub_p['y'], sub_p['z'], color='#38bdf8', lw=0.8, alpha=0.6, label='Morton Curve (Z-Order)')
    ax1.scatter(sub_p['x'], sub_p['y'], sub_p['z'], c='#f43f5e', s=12, alpha=0.8)

    ax1.set_xlabel("X", color='#64748b')
    ax1.set_ylabel("Y", color='#64748b')
    ax1.set_zlabel("Z", color='#64748b')
    ax1.tick_params(colors='#64748b')
    ax1.grid(True, color='#1e293b', linestyle=':')
    ax1.legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='white')

    # Panel 2: 3D Final Gravitational Core & Trajectories
    ax2 = fig.add_subplot(1, 2, 2, projection='3d', facecolor='#0b0d14')
    ax2.set_title(f"3D Gravitational Core at Final Step (N={len(p_final)})", color='#94a3b8', fontsize=12)

    speed = np.sqrt(p_final['vx']**2 + p_final['vy']**2 + p_final['vz']**2)
    sc = ax2.scatter(p_final['x'], p_final['y'], p_final['z'], c=speed, cmap='plasma', s=10, alpha=0.85)

    # Add trajectory trails for the top 15 fastest particles across historical steps
    if len(particle_files) > 1:
        fast_indices = np.argsort(speed)[-15:]
        history_steps = particle_files[::max(1, len(particle_files) // 10)]
        for fidx in fast_indices:
            tx, ty, tz = [], [], []
            for hfile in history_steps:
                hp = load_particle_snapshot(hfile)
                tx.append(hp['x'][fidx])
                ty.append(hp['y'][fidx])
                tz.append(hp['z'][fidx])
            ax2.plot(tx, ty, tz, color='#facc15', lw=1.2, alpha=0.7)

    ax2.set_xlabel("X", color='#64748b')
    ax2.set_ylabel("Y", color='#64748b')
    ax2.set_zlabel("Z", color='#64748b')
    ax2.tick_params(colors='#64748b')
    ax2.grid(True, color='#1e293b', linestyle=':')
    cbar = fig.colorbar(sc, ax=ax2, label='Velocity |v|', pad=0.1, fraction=0.04)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    cbar.set_label('Velocity |v|', color='white')

    plt.tight_layout()
    out_file = os.path.join(output_dir, "particle_evolution_3d.png")
    plt.savefig(out_file, dpi=180, facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Viz] Saved 3D particle evolution plot: {out_file}")

def create_animation(particle_files, output_dir):
    """Create an animated GIF showing dynamic particle evolution."""
    if not particle_files:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    particles0 = load_particle_snapshot(particle_files[0])
    
    # Determine bounds
    all_x = particles0['x']
    all_y = particles0['y']
    pad = 0.1
    ax.set_xlim(min(all_x.min() - pad, -0.2), max(all_x.max() + pad, 1.2))
    ax.set_ylim(min(all_y.min() - pad, -0.2), max(all_y.max() + pad, 1.2))
    ax.set_aspect('equal')
    ax.set_facecolor('#111116')
    fig.patch.set_facecolor('#111116')
    ax.grid(True, color='#2c2c36', linestyle='--', alpha=0.5)

    scatter = ax.scatter([], [], s=10, c=[], cmap='inferno', vmin=0, vmax=1.0)
    title = ax.set_title("", color='white', fontsize=12)

    def init():
        scatter.set_offsets(np.empty((0, 2)))
        return scatter, title

    def update(frame_file):
        step_num = int(os.path.basename(frame_file).replace('step_', '').replace('.bin', ''))
        p = load_particle_snapshot(frame_file)
        coords = np.column_stack((p['x'], p['y']))
        speed = np.sqrt(p['vx']**2 + p['vy']**2 + p['vz']**2)
        scatter.set_offsets(coords)
        scatter.set_array(speed)
        title.set_text(f"AL-BH Simulation — Timestep {step_num}")
        return scatter, title

    ani = FuncAnimation(fig, update, frames=particle_files, init_func=init, blit=False, interval=100)
    out_file = os.path.join(output_dir, "simulation_orbit.gif")
    ani.save(out_file, writer='pillow', fps=10)
    plt.close()
    print(f"[Viz] Saved animation GIF: {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Visualize AL-BH simulation metrics and trajectories.")
    parser.add_argument("--results", type=str, default="results/demo", help="Directory containing metrics and particles/")
    parser.add_argument("--gif", action="store_true", help="Generate animated GIF of particle motion")
    parser.add_argument("--metrics-only", action="store_true", help="Only plot metrics CSV")
    args = parser.parse_args()

    results_dir = args.results
    if not os.path.isdir(results_dir):
        print(f"Directory not found: {results_dir}")
        sys.exit(1)

    # 1. Find metrics CSV
    csv_candidates = glob.glob(os.path.join(results_dir, "metrics_*.csv"))
    if csv_candidates:
        plot_metrics(csv_candidates[0], results_dir)
    else:
        print("[Viz] Note: No metrics CSV found in results directory.")

    if args.metrics_only:
        return

    # 2. Find particle files
    particles_dir = os.path.join(results_dir, "particles")
    p_files = sorted(glob.glob(os.path.join(particles_dir, "step_*.bin")))
    if p_files:
        plot_snapshots_2d(p_files, results_dir)
        plot_3d_state(p_files, results_dir)
        if args.gif:
            create_animation(p_files, results_dir)
    else:
        print("[Viz] Note: No particle binary dumps found in 'particles/' subdirectory.")

if __name__ == "__main__":
    main()
