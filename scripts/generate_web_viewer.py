#!/usr/bin/env python3
"""
AL-BH 3D Interactive Web Viewer Generator
Generates a standalone, GPU-accelerated Three.js 3D web dashboard for N-body simulations.

Usage:
    python3 scripts/generate_web_viewer.py --results results/plummer_demo --open
"""

import os
import sys
import glob
import json
import struct
import argparse
import numpy as np

def load_particle_snapshot(filepath):
    """Read binary particle snapshot produced by dump_particle_state()."""
    with open(filepath, 'rb') as f:
        data = f.read()
    
    N = struct.unpack('<i', data[:4])[0]
    expected_size = 4 + N * 60
    if len(data) != expected_size:
        raise ValueError(f"Corrupt binary file {filepath}: size {len(data)} != {expected_size}")
    
    dtype = np.dtype([
        ('x', '<f8'), ('y', '<f8'), ('z', '<f8'),
        ('vx', '<f8'), ('vy', '<f8'), ('vz', '<f8'),
        ('mass', '<f8'), ('rank', '<i4')
    ])
    return np.frombuffer(data[4:], dtype=dtype, count=N)

def load_metrics(csv_path):
    """Read metrics CSV if available."""
    if not os.path.isfile(csv_path):
        return []
    try:
        data = np.genfromtxt(csv_path, delimiter=',', names=True)
        metrics = []
        # If single row, np.genfromtxt may produce a 0-d array
        if data.ndim == 0:
            data = np.array([data])
        for i in range(len(data)):
            row = data[i]
            ke = float(row['kinetic_energy'])
            pe = float(row['potential_energy'])
            metrics.append({
                'step': int(row['timestep']),
                'ke': ke,
                'pe': pe,
                'te': ke + pe,
                'tree_time': float(row['tree_build_time']),
                'force_time': float(row['force_compute_time']),
                'step_time': float(row['total_step_time']),
                'imbalance': float(row['imbalance_ratio']) if 'imbalance_ratio' in data.dtype.names else 1.0,
                'comm_time': float(row['comm_time']) if 'comm_time' in data.dtype.names else 0.0,
                'rebalance_time': float(row['rebalance_time']) if 'rebalance_time' in data.dtype.names else 0.0,
                'particles_migrated': int(row['particles_migrated']) if 'particles_migrated' in data.dtype.names else 0,
            })
        return metrics
    except Exception as e:
        print(f"[Viewer] Note: could not parse metrics: {e}")
        return []

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AL-BH 3D Interactive Telemetry & Cosmic Simulation Dashboard</title>
    <style>
        :root {
            --bg-space: #07090e;
            --panel-bg: rgba(13, 17, 26, 0.82);
            --panel-border: rgba(255, 255, 255, 0.08);
            --panel-hover: rgba(255, 255, 255, 0.14);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #8e9cae;
            --accent-cyan: #38bdf8;
            --accent-blue: #3b82f6;
            --accent-emerald: #10b981;
            --accent-amber: #f59e0b;
            --accent-rose: #f43f5e;
            --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
            --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            --shadow-hud: 0 12px 32px -4px rgba(0, 0, 0, 0.5), 0 4px 12px -2px rgba(0, 0, 0, 0.4);
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        ::selection {
            background: rgba(56, 189, 248, 0.3);
            color: #ffffff;
        }

        /* Custom Scrollbars */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: rgba(7, 9, 14, 0.8);
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.16);
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(56, 189, 248, 0.4);
        }

        body {
            background-color: var(--bg-space);
            color: var(--text-primary);
            font-family: var(--font-sans);
            overflow: hidden;
            width: 100vw;
            height: 100vh;
            user-select: none;
            -webkit-font-smoothing: antialiased;
        }

        #canvas-container {
            width: 100%;
            height: 100%;
            position: absolute;
            top: 0;
            left: 0;
            z-index: 1;
        }

        /* Common HUD Glass Container */
        .hud-glass {
            background: var(--panel-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--panel-border);
            border-radius: 12px;
            box-shadow: var(--shadow-hud);
            z-index: 10;
            pointer-events: auto;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }
        .hud-glass:hover {
            border-color: rgba(255, 255, 255, 0.12);
        }

        /* ==========================================================================
           Top-Left Header HUD
           ========================================================================== */
        .header-hud {
            position: absolute;
            top: 16px;
            left: 20px;
            padding: 14px 20px;
            max-width: 480px;
        }
        .header-top-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
        }
        .header-title {
            font-size: 18px;
            font-weight: 700;
            color: var(--text-primary);
            letter-spacing: 0.3px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .badge {
            font-size: 11px;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 4px;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }
        .badge-primary {
            background: rgba(59, 130, 246, 0.2);
            border: 1px solid rgba(59, 130, 246, 0.4);
            color: #93c5fd;
        }
        .badge-subtle {
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: var(--text-secondary);
        }
        .header-subtitle {
            font-size: 13px;
            color: var(--text-secondary);
            margin-top: 5px;
            line-height: 1.4;
        }
        .header-meta-tags {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-top: 8px;
            font-size: 12px;
            color: var(--text-muted);
            font-family: var(--font-mono);
        }
        .meta-dot {
            width: 4px;
            height: 4px;
            background: var(--text-muted);
            border-radius: 50%;
        }

        /* Header action icon buttons */
        .header-actions {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .icon-btn {
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid var(--panel-border);
            color: var(--text-secondary);
            width: 28px;
            height: 28px;
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.15s ease;
        }
        .icon-btn:hover {
            background: rgba(255, 255, 255, 0.12);
            color: var(--text-primary);
            border-color: rgba(255, 255, 255, 0.2);
        }
        .icon-btn svg {
            width: 14px;
            height: 14px;
            stroke-width: 2;
        }

        /* ==========================================================================
           Top-Right Real-Time Telemetry & Diagnostics HUD
           ========================================================================== */
        .telemetry-hud {
            position: absolute;
            top: 16px;
            right: 20px;
            padding: 16px 20px;
            width: 330px;
            max-height: calc(100vh - 120px);
            overflow-y: auto;
        }
        .hud-section-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 10px;
            padding-bottom: 6px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        }
        .hud-section-title {
            font-size: 13px;
            font-weight: 700;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.8px;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .telemetry-group {
            margin-bottom: 14px;
        }
        .telemetry-group:last-child {
            margin-bottom: 0;
        }
        .group-label {
            font-size: 11px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.6px;
            margin-bottom: 6px;
        }
        .stat-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 4px 0;
            font-size: 12px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.03);
        }
        .stat-label-wrap {
            display: flex;
            align-items: center;
            gap: 4px;
            color: var(--text-secondary);
        }
        .stat-value {
            font-family: var(--font-mono);
            font-size: 12px;
            font-weight: 600;
            font-variant-numeric: tabular-nums;
            color: var(--text-primary);
        }
        .stat-value.cyan { color: var(--accent-cyan); }
        .stat-value.emerald { color: var(--accent-emerald); }
        .stat-value.amber { color: var(--accent-amber); }
        .stat-value.rose { color: var(--accent-rose); }

        /* Conservation & Imbalance Quality Badges */
        .status-pill {
            font-size: 11px;
            font-weight: 700;
            padding: 2px 7px;
            border-radius: 4px;
            text-transform: uppercase;
            letter-spacing: 0.4px;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }
        .status-pill.good {
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        .status-pill.warn {
            background: rgba(245, 158, 11, 0.15);
            color: #fbbf24;
            border: 1px solid rgba(245, 158, 11, 0.3);
        }
        .status-pill.alert {
            background: rgba(244, 63, 94, 0.15);
            color: #fb7185;
            border: 1px solid rgba(244, 63, 94, 0.3);
        }

        /* Timing Breakdown Ratio Bar */
        .timing-bar-container {
            margin-top: 6px;
            padding: 2px 0;
        }
        .timing-bar {
            height: 6px;
            width: 100%;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 3px;
            overflow: hidden;
            display: flex;
        }
        .timing-segment-force {
            background: #38bdf8;
            height: 100%;
            transition: width 0.2s ease;
        }
        .timing-segment-tree {
            background: #a855f7;
            height: 100%;
            transition: width 0.2s ease;
        }
        .timing-legend {
            display: flex;
            justify-content: space-between;
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 5px;
            font-family: var(--font-mono);
        }

        /* ==========================================================================
           Side Controls & Color Legend HUD (Left Bottom)
           ========================================================================== */
        .sidebar-hud {
            position: absolute;
            left: 20px;
            bottom: 24px;
            width: 260px;
            padding: 14px 16px;
        }
        .sidebar-hud h2 {
            font-size: 13px;
            font-weight: 700;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.8px;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .setting-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            font-size: 12px;
            color: var(--text-secondary);
        }
        .setting-row:last-child {
            margin-bottom: 0;
        }
        select.styled-select {
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid var(--panel-border);
            color: var(--text-primary);
            padding: 5px 8px;
            border-radius: 6px;
            font-size: 12px;
            outline: none;
            cursor: pointer;
            width: 140px;
            transition: border-color 0.15s;
        }
        select.styled-select:focus {
            border-color: var(--accent-cyan);
        }
        select.styled-select option {
            background: #0f172a;
            color: #f8fafc;
        }

        /* Custom Toggle Switch */
        .toggle-switch {
            position: relative;
            display: inline-block;
            width: 32px;
            height: 18px;
        }
        .toggle-switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }
        .slider-round {
            position: absolute;
            cursor: pointer;
            top: 0; left: 0; right: 0; bottom: 0;
            background-color: rgba(255, 255, 255, 0.12);
            transition: .2s;
            border-radius: 18px;
        }
        .slider-round:before {
            position: absolute;
            content: "";
            height: 12px;
            width: 12px;
            left: 3px;
            bottom: 3px;
            background-color: white;
            transition: .2s;
            border-radius: 50%;
        }
        input:checked + .slider-round {
            background-color: var(--accent-blue);
        }
        input:checked + .slider-round:before {
            transform: translateX(14px);
        }

        /* Interactive Dynamic Color Legend */
        .legend-card {
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid rgba(255, 255, 255, 0.06);
        }
        .legend-title {
            font-size: 11px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.6px;
            margin-bottom: 6px;
            display: flex;
            justify-content: space-between;
        }
        .gradient-bar {
            height: 8px;
            width: 100%;
            border-radius: 4px;
            margin-bottom: 4px;
        }
        .gradient-plasma {
            background: linear-gradient(to right, #0d0887, #6a00a8, #b12a90, #e16462, #fca636, #f0f921);
        }
        .gradient-fire {
            background: linear-gradient(to right, #1a0000, #7a0000, #cc3300, #ff9900, #ffff99);
        }
        .gradient-cyan {
            background: linear-gradient(to right, #082f49, #0284c7, #38bdf8, #bae6fd);
        }
        .gradient-labels {
            display: flex;
            justify-content: space-between;
            font-size: 11px;
            color: var(--text-muted);
            font-family: var(--font-mono);
        }

        /* MPI Rank Grid Legend */
        .rank-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 6px;
            margin-top: 6px;
        }
        .rank-pill {
            display: flex;
            align-items: center;
            gap: 6px;
            background: rgba(255, 255, 255, 0.04);
            padding: 3px 6px;
            border-radius: 4px;
            font-size: 11px;
            font-family: var(--font-mono);
            color: var(--text-secondary);
        }
        .rank-swatch {
            width: 8px;
            height: 8px;
            border-radius: 2px;
            flex-shrink: 0;
        }

        /* Camera View Presets */
        .camera-presets {
            display: flex;
            gap: 4px;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid rgba(255, 255, 255, 0.06);
        }
        .preset-btn {
            flex: 1;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--panel-border);
            color: var(--text-secondary);
            font-size: 11px;
            font-weight: 600;
            padding: 5px 0;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.15s;
            text-align: center;
        }
        .preset-btn:hover {
            background: rgba(255, 255, 255, 0.12);
            color: var(--text-primary);
            border-color: rgba(255, 255, 255, 0.2);
        }

        /* ==========================================================================
           Bottom Floating Playback Control HUD
           ========================================================================== */
        .controls-hud {
            position: absolute;
            bottom: 24px;
            left: 50%;
            transform: translateX(-50%);
            padding: 10px 20px;
            display: flex;
            align-items: center;
            gap: 14px;
            border-radius: 16px;
        }
        .control-btns {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .btn {
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid var(--panel-border);
            color: var(--text-primary);
            padding: 6px 12px;
            border-radius: 8px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s cubic-bezier(0.4, 0, 0.2, 1);
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .btn:hover {
            background: rgba(255, 255, 255, 0.14);
            border-color: rgba(255, 255, 255, 0.25);
            transform: translateY(-1px);
        }
        .btn:active {
            transform: translateY(0);
        }
        .btn-primary {
            background: #2563eb;
            border-color: #3b82f6;
            color: #ffffff;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35);
        }
        .btn-primary:hover {
            background: #1d4ed8;
            border-color: #60a5fa;
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.45);
        }
        .btn-icon-only {
            padding: 6px 8px;
        }
        .btn svg {
            width: 13px;
            height: 13px;
            stroke-width: 2.2;
        }

        /* Scrubber */
        .scrubber-wrap {
            display: flex;
            align-items: center;
            gap: 12px;
            min-width: 240px;
        }
        input[type=range] {
            -webkit-appearance: none;
            width: 100%;
            height: 6px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
            outline: none;
            cursor: pointer;
            transition: background 0.15s;
        }
        input[type=range]:hover {
            background: rgba(255, 255, 255, 0.16);
        }
        input[type=range]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 14px;
            height: 14px;
            border-radius: 50%;
            background: var(--accent-cyan);
            cursor: pointer;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.4);
            transition: transform 0.1s ease;
        }
        input[type=range]::-webkit-slider-thumb:hover {
            transform: scale(1.25);
        }
        .time-badge {
            font-family: var(--font-mono);
            font-size: 12px;
            font-weight: 600;
            font-variant-numeric: tabular-nums;
            color: var(--accent-cyan);
            min-width: 70px;
            text-align: right;
        }

        /* Speed pills */
        .speed-group {
            display: flex;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid var(--panel-border);
            border-radius: 6px;
            overflow: hidden;
        }
        .speed-pill {
            background: transparent;
            border: none;
            color: var(--text-muted);
            font-size: 11px;
            font-weight: 600;
            font-family: var(--font-mono);
            padding: 4px 8px;
            cursor: pointer;
            transition: all 0.15s;
        }
        .speed-pill:hover {
            color: var(--text-primary);
        }
        .speed-pill.active {
            background: rgba(56, 189, 248, 0.2);
            color: var(--accent-cyan);
        }

        /* ==========================================================================
           Physics & Telemetry Explainer Slide-over Drawer
           ========================================================================== */
        .drawer-overlay {
            position: fixed;
            top: 0; left: 0; width: 100vw; height: 100vh;
            background: rgba(0, 0, 0, 0.6);
            backdrop-filter: blur(4px);
            z-index: 100;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.25s ease;
        }
        .drawer-overlay.open {
            opacity: 1;
            pointer-events: auto;
        }
        .drawer {
            position: fixed;
            top: 0;
            right: 0;
            width: 440px;
            height: 100vh;
            background: #0b0f19;
            border-left: 1px solid var(--panel-border);
            box-shadow: -8px 0 32px rgba(0, 0, 0, 0.8);
            z-index: 101;
            transform: translateX(100%);
            transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .drawer.open {
            transform: translateX(0);
        }
        .drawer-header {
            padding: 20px 24px;
            border-bottom: 1px solid var(--panel-border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .drawer-title {
            font-size: 16px;
            font-weight: 700;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .drawer-body {
            padding: 24px;
            overflow-y: auto;
            font-size: 13px;
            line-height: 1.6;
            color: #cbd5e1;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .guide-topic {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 8px;
            padding: 14px 16px;
        }
        .guide-topic h3 {
            font-size: 14px;
            font-weight: 700;
            color: var(--accent-cyan);
            margin-bottom: 6px;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .guide-topic p {
            margin-bottom: 8px;
        }
        .guide-topic p:last-child {
            margin-bottom: 0;
        }
        .guide-formula {
            background: rgba(0, 0, 0, 0.4);
            padding: 6px 10px;
            border-radius: 4px;
            font-family: var(--font-mono);
            font-size: 12px;
            color: #93c5fd;
            margin: 6px 0;
            display: inline-block;
        }
        .guide-kbd {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 4px;
            padding: 2px 6px;
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--text-primary);
        }

        /* Responsive Tweaks */
        @media (max-width: 1024px) {
            .telemetry-hud { width: 280px; }
            .sidebar-hud { width: 220px; }
            .controls-hud { padding: 8px 14px; gap: 10px; }
            .scrubber-wrap { min-width: 160px; }
        }
    </style>

    <!-- Three.js and OrbitControls -->
    <script src="assets/three.min.js"></script>
    <script src="assets/OrbitControls.js"></script>
    <!-- Fallback CDN in case assets are moved -->
    <script>
        if (typeof THREE === 'undefined') {
            document.write('<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"><\\/script>');
            document.write('<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"><\\/script>');
        }
    </script>
</head>
<body>
    <div id="canvas-container"></div>

    <!-- ========================================================================
         Top-Left Header HUD
         ======================================================================== -->
    <header class="header-hud hud-glass">
        <div class="header-top-row">
            <h1 class="header-title">
                <span>AL-BH 3D Cosmic N-Body</span>
                <span class="badge badge-primary">BARNES-HUT</span>
            </h1>
            <div class="header-actions">
                <button class="icon-btn" id="btn-open-guide" title="Telemetry & Physics Guide (?)" aria-label="Open Telemetry Guide">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                        <circle cx="12" cy="12" r="10"></circle>
                        <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path>
                        <line x1="12" y1="17" x2="12.01" y2="17"></line>
                    </svg>
                </button>
            </div>
        </div>
        <p class="header-subtitle" id="header-description">
            Adaptive Linearized Octree with Morton Space-Filling Curve Partitioning
        </p>
        <div class="header-meta-tags">
            <span id="meta-particles-label">N = ...</span>
            <span class="meta-dot"></span>
            <span id="meta-integrator-label">Leapfrog (Symplectic)</span>
            <span class="meta-dot"></span>
            <span id="meta-theta-label">&theta; = 0.5</span>
        </div>
    </header>

    <!-- ========================================================================
         Top-Right Real-Time Telemetry & Diagnostics HUD
         ======================================================================== -->
    <aside class="telemetry-hud hud-glass" aria-label="Simulation Diagnostics">
        <div class="hud-section-header">
            <div class="hud-section-title">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
                </svg>
                <span>Live Telemetry</span>
            </div>
            <span class="status-pill good" id="hud-status-badge">● STABLE</span>
        </div>

        <!-- Section 1: Physical Conservation -->
        <div class="telemetry-group">
            <div class="group-label">Physical Conservation</div>
            
            <div class="stat-row">
                <div class="stat-label-wrap">
                    <span>Total Energy (E):</span>
                </div>
                <span class="stat-value cyan" id="hud-energy">-</span>
            </div>

            <div class="stat-row">
                <div class="stat-label-wrap">
                    <span>Kinetic (K) / Potential (U):</span>
                </div>
                <span class="stat-value" id="hud-ke-pe">- / -</span>
            </div>

            <div class="stat-row">
                <div class="stat-label-wrap">
                    <span>Virial Ratio (2K/|U|):</span>
                </div>
                <span class="stat-value" id="hud-virial">-</span>
            </div>

            <div class="stat-row">
                <div class="stat-label-wrap">
                    <span>Energy Drift |&Delta;E/E₀|:</span>
                </div>
                <span class="stat-value emerald" id="hud-drift">0.000%</span>
            </div>
        </div>

        <!-- Section 2: HPC Octree & Compute Runtime -->
        <div class="telemetry-group">
            <div class="group-label">HPC Engine & Octree Overhead</div>

            <div class="stat-row">
                <div class="stat-label-wrap">
                    <span>Step Compute Time:</span>
                </div>
                <span class="stat-value" id="hud-step-time">-</span>
            </div>

            <div class="stat-row">
                <div class="stat-label-wrap">
                    <span>Force vs Tree Build:</span>
                </div>
                <span class="stat-value" id="hud-times-breakdown">- / -</span>
            </div>

            <!-- Visual Timing Proportion Bar -->
            <div class="timing-bar-container">
                <div class="timing-bar">
                    <div class="timing-segment-force" id="bar-force" style="width: 85%;"></div>
                    <div class="timing-segment-tree" id="bar-tree" style="width: 15%;"></div>
                </div>
                <div class="timing-legend">
                    <span style="color: #38bdf8;">■ Force (<span id="pct-force">85%</span>)</span>
                    <span style="color: #a855f7;">■ Tree (<span id="pct-tree">15%</span>)</span>
                </div>
            </div>

            <div class="stat-row" style="margin-top: 6px;">
                <div class="stat-label-wrap">
                    <span>MPI Imbalance Ratio:</span>
                </div>
                <span class="stat-value" id="hud-imbalance">1.00x</span>
            </div>

            <div class="stat-row">
                <div class="stat-label-wrap">
                    <span>Max Velocity |v_max|:</span>
                </div>
                <span class="stat-value" id="hud-speed">-</span>
            </div>
        </div>
    </aside>

    <!-- ========================================================================
         Side Controls & Legend Panel (Bottom-Left)
         ======================================================================== -->
    <aside class="sidebar-hud hud-glass" aria-label="Visualizer Settings">
        <h2>
            <span>View & Color Mode</span>
        </h2>

        <div class="setting-row">
            <span>Coloring:</span>
            <select id="color-mode" class="styled-select">
                <option value="plasma">Velocity Heatmap (Plasma)</option>
                <option value="rank">MPI Rank Domains</option>
                <option value="fire">Solar Flare (Fire)</option>
                <option value="cyan">Deep Space Cyan</option>
            </select>
        </div>

        <div class="setting-row">
            <span>Star Size:</span>
            <input type="range" id="point-size" min="1" max="15" value="4" style="width: 100px;">
        </div>

        <div class="setting-row">
            <span>Bounding Box:</span>
            <label class="toggle-switch">
                <input type="checkbox" id="toggle-box" checked>
                <span class="slider-round"></span>
            </label>
        </div>

        <div class="setting-row">
            <span>Spatial Axes (XYZ):</span>
            <label class="toggle-switch">
                <input type="checkbox" id="toggle-axes" checked>
                <span class="slider-round"></span>
            </label>
        </div>

        <div class="setting-row">
            <span>Auto Rotate:</span>
            <label class="toggle-switch">
                <input type="checkbox" id="toggle-rotate">
                <span class="slider-round"></span>
            </label>
        </div>

        <!-- Dynamic Color / MPI Legend -->
        <div class="legend-card" id="legend-container">
            <div class="legend-title">
                <span id="legend-label">Velocity Scale</span>
                <span id="legend-metric" style="font-family: var(--font-mono);">0.0 &rarr; 1.0</span>
            </div>
            <div id="legend-content">
                <!-- Rendered dynamically via JS based on color mode -->
                <div class="gradient-bar gradient-plasma"></div>
                <div class="gradient-labels">
                    <span>0.0 (Cold)</span>
                    <span>v_max (Fast)</span>
                </div>
            </div>
        </div>

        <!-- Camera View Angles -->
        <div class="camera-presets">
            <button class="preset-btn" id="cam-iso" title="Perspective Isometric">3D</button>
            <button class="preset-btn" id="cam-top" title="Top View (XY)">Top (XY)</button>
            <button class="preset-btn" id="cam-side" title="Side View (XZ)">Side (XZ)</button>
            <button class="preset-btn" id="cam-front" title="Front View (YZ)">Front (YZ)</button>
            <button class="preset-btn" id="cam-reset" title="Reset View">Center</button>
        </div>
    </aside>

    <!-- ========================================================================
         Bottom Floating Playback Control Bar
         ======================================================================== -->
    <nav class="controls-hud hud-glass" aria-label="Playback Navigation">
        <div class="control-btns">
            <button class="btn btn-primary" id="btn-play" aria-label="Play or Pause">
                <svg id="icon-play-state" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <rect x="6" y="4" width="4" height="16" fill="currentColor"></rect>
                    <rect x="14" y="4" width="4" height="16" fill="currentColor"></rect>
                </svg>
                <span id="text-play-state">Pause</span>
            </button>
            <button class="btn btn-icon-only" id="btn-prev" title="Step Backward (Left Arrow)" aria-label="Step Backward">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <polygon points="19 20 9 12 19 4 19 20"></polygon>
                    <line x1="5" y1="19" x2="5" y2="5"></line>
                </svg>
            </button>
            <button class="btn btn-icon-only" id="btn-next" title="Step Forward (Right Arrow)" aria-label="Step Forward">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <polygon points="5 4 15 12 5 20 5 4"></polygon>
                    <line x1="19" y1="5" x2="19" y2="19"></line>
                </svg>
            </button>
            <button class="btn btn-icon-only" id="btn-reset" title="Rewind to Step 0 (R)" aria-label="Reset to First Step">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <polyline points="1 4 1 10 7 10"></polyline>
                    <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"></path>
                </svg>
            </button>
        </div>

        <div class="scrubber-wrap">
            <input type="range" id="time-slider" min="0" max="0" value="0" aria-label="Timestep Scrubber">
            <span class="time-badge" id="time-display">0 / 0</span>
        </div>

        <div class="speed-group" aria-label="Playback Speed">
            <button class="speed-pill" data-speed="0.5">0.5x</button>
            <button class="speed-pill active" data-speed="1.0">1.0x</button>
            <button class="speed-pill" data-speed="2.0">2.0x</button>
            <button class="speed-pill" data-speed="5.0">5.0x</button>
        </div>
    </nav>

    <!-- ========================================================================
         Slide-over Telemetry & Physics Explainer Drawer
         ======================================================================== -->
    <div class="drawer-overlay" id="drawer-overlay"></div>
    <aside class="drawer" id="telemetry-drawer" aria-label="Telemetry Explainer">
        <div class="drawer-header">
            <h2 class="drawer-title">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="12" y1="16" x2="12" y2="12"></line>
                    <line x1="12" y1="8" x2="12.01" y2="8"></line>
                </svg>
                <span>Telemetry & Physics Guide</span>
            </h2>
            <button class="icon-btn" id="btn-close-guide" aria-label="Close Guide">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <line x1="18" y1="6" x2="6" y2="18"></line>
                    <line x1="6" y1="6" x2="18" y2="18"></line>
                </svg>
            </button>
        </div>

        <div class="drawer-body">
            <div class="guide-topic">
                <h3>Energy Conservation |&Delta;E/E₀|</h3>
                <p>
                    In gravitational N-body simulations, total energy <span class="guide-formula">E = K + U</span> must remain invariant over time.
                </p>
                <p>
                    AL-BH employs the <strong>Kick-Drift-Kick (KDK) Leapfrog</strong> scheme, a symplectic integrator that preserves phase space volume:
                </p>
                <div class="guide-formula">&Delta;E / E₀ &lt; 0.1% &rarr; Excellent Symplectic Bounded Drift</div>
            </div>

            <div class="guide-topic">
                <h3>Virial Theorem (2K/|U|)</h3>
                <p>
                    For a self-gravitating spherical stellar system (e.g. Plummer model) in dynamic equilibrium:
                </p>
                <div class="guide-formula">2 &lang;K&rang; + &lang;U&rang; = 0 &implies; 2K / |U| &approx; 1.00</div>
                <p>
                    If started cold (<span class="guide-formula">2K/|U| &lt; 1</span>), the cluster undergoes gravitational collapse before virializing through phase mixing.
                </p>
            </div>

            <div class="guide-topic">
                <h3>Barnes–Hut Force vs. Tree Build</h3>
                <p>
                    Traditional direct N-body requires <span class="guide-formula">O(N²)</span> pairwise interactions. Barnes–Hut reduces this to <span class="guide-formula">O(N log N)</span> by clustering distant nodes with opening criterion:
                </p>
                <div class="guide-formula">s / d &lt; &theta; (where &theta; = 0.5)</div>
                <p>
                    The linearized array-based Octree ensures tree construction takes only <span class="guide-formula">&lt; 15%</span> of step time, while force traversal dominates without cache thrashing.
                </p>
            </div>

            <div class="guide-topic">
                <h3>MPI Domain Imbalance Ratio</h3>
                <p>
                    Defined as <span class="guide-formula">I = T_max / T_avg</span> across all MPI ranks.
                </p>
                <p>
                    Without dynamic load balancing, dense cluster centers cause extreme imbalance (<span class="guide-formula">I &gt; 20x</span>). AL-BH's Morton space-filling curve partitions and adaptive boundary migration restore <span class="guide-formula">I &approx; 1.00x</span>.
                </p>
            </div>

            <div class="guide-topic">
                <h3>Keyboard Navigation</h3>
                <p><span class="guide-kbd">Space</span> — Play / Pause simulation</p>
                <p><span class="guide-kbd">&larr;</span> / <span class="guide-kbd">&rarr;</span> — Step backward / forward</p>
                <p><span class="guide-kbd">R</span> — Reset to Step 0</p>
                <p><span class="guide-kbd">?</span> — Toggle this Telemetry Guide</p>
                <p><span class="guide-kbd">Mouse Drag</span> — Orbit 3D Camera</p>
                <p><span class="guide-kbd">Scroll</span> — Zoom in / out</p>
            </div>
        </div>
    </aside>

    <!-- Simulation Trajectory Data Injected Here -->
    <script>
        const SIM_DATA = %DATA_PLACEHOLDER%;
    </script>

    <script>
        // ============================================================================
        // 3D Three.js Visualization Scene Initialization
        // ============================================================================
        const container = document.getElementById('canvas-container');
        const scene = new THREE.Scene();
        scene.fog = new THREE.FogExp2(0x07090e, 0.12);

        const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.01, 100);
        camera.position.set(1.6, 1.3, 1.9);

        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        renderer.setClearColor(0x07090e, 1);
        container.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.06;
        controls.target.set(0.5, 0.5, 0.5);

        // Simulation Bounding Box
        const boxGeometry = new THREE.BoxGeometry(1.0, 1.0, 1.0);
        const boxEdges = new THREE.EdgesGeometry(boxGeometry);
        const boxLine = new THREE.LineSegments(boxEdges, new THREE.LineBasicMaterial({ 
            color: 0x334155, 
            transparent: true, 
            opacity: 0.6 
        }));
        boxLine.position.set(0.5, 0.5, 0.5);
        scene.add(boxLine);

        // Coordinate Axes Helper (X=Red, Y=Green, Z=Blue)
        const axesHelper = new THREE.AxesHelper(0.3);
        axesHelper.position.set(0.0, 0.0, 0.0);
        scene.add(axesHelper);

        // Soft ambient illumination
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.9);
        scene.add(ambientLight);

        // Particle Glow Texture Generator
        function createStarTexture() {
            const canvas = document.createElement('canvas');
            canvas.width = 64;
            canvas.height = 64;
            const ctx = canvas.getContext('2d');
            const gradient = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
            gradient.addColorStop(0.0, 'rgba(255, 255, 255, 1.0)');
            gradient.addColorStop(0.2, 'rgba(235, 245, 255, 0.9)');
            gradient.addColorStop(0.5, 'rgba(70, 160, 255, 0.35)');
            gradient.addColorStop(1.0, 'rgba(0, 0, 0, 0.0)');
            ctx.fillStyle = gradient;
            ctx.fillRect(0, 0, 64, 64);
            const texture = new THREE.Texture(canvas);
            texture.needsUpdate = true;
            return texture;
        }

        // ============================================================================
        // Point Cloud Particle System Setup
        // ============================================================================
        const N = SIM_DATA.particles_count;
        const totalSteps = SIM_DATA.steps.length;
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(N * 3);
        const colors = new Float32Array(N * 3);

        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({
            size: 0.024,
            vertexColors: true,
            map: createStarTexture(),
            transparent: true,
            blending: THREE.AdditiveBlending,
            depthWrite: false,
        });

        const particlePoints = new THREE.Points(geometry, material);
        scene.add(particlePoints);

        // Cohesive Distinct Palette for MPI Ranks
        const rankColors = [
            { color: new THREE.Color(0x38bdf8), hex: '#38bdf8', name: 'Rank 0 (Cyan)' },
            { color: new THREE.Color(0xf43f5e), hex: '#f43f5e', name: 'Rank 1 (Rose)' },
            { color: new THREE.Color(0xfacc15), hex: '#facc15', name: 'Rank 2 (Amber)' },
            { color: new THREE.Color(0x4ade80), hex: '#4ade80', name: 'Rank 3 (Emerald)' },
            { color: new THREE.Color(0xa855f7), hex: '#a855f7', name: 'Rank 4 (Purple)' },
            { color: new THREE.Color(0xf97316), hex: '#f97316', name: 'Rank 5 (Orange)' },
            { color: new THREE.Color(0x06b6d4), hex: '#06b6d4', name: 'Rank 6 (Teal)' },
            { color: new THREE.Color(0xec4899), hex: '#ec4899', name: 'Rank 7 (Pink)' },
        ];

        // Plasma Color Transfer Function
        function getPlasmaColor(t) {
            t = Math.max(0, Math.min(1, t));
            const r = Math.min(1.0, Math.max(0.05, 1.55 * t));
            const g = Math.min(1.0, Math.max(0.05, Math.pow(t, 2) * 1.25));
            const b = Math.min(1.0, Math.max(0.12, 1.0 - t * 0.8 + Math.pow(t, 4) * 0.8));
            return [r, g, b];
        }

        // Solar Flare (Fire) Transfer Function
        function getFireColor(t) {
            t = Math.max(0, Math.min(1, t));
            const r = Math.min(1.0, 0.4 + t * 0.8);
            const g = Math.min(1.0, Math.pow(t, 1.7));
            const b = Math.min(1.0, Math.pow(t, 3.4));
            return [r, g, b];
        }

        // ============================================================================
        // Simulation Playback State Management
        // ============================================================================
        let currentStepIndex = 0;
        let isPlaying = true;
        let playbackSpeed = 1.0;
        let lastFrameTime = performance.now();
        let frameInterval = 60; // ms per step at 1.0x
        let colorMode = 'plasma';

        // Cache DOM elements
        const hudEnergy = document.getElementById('hud-energy');
        const hudKePe = document.getElementById('hud-ke-pe');
        const hudVirial = document.getElementById('hud-virial');
        const hudDrift = document.getElementById('hud-drift');
        const hudStatusBadge = document.getElementById('hud-status-badge');
        const hudStepTime = document.getElementById('hud-step-time');
        const hudTimesBreakdown = document.getElementById('hud-times-breakdown');
        const hudImbalance = document.getElementById('hud-imbalance');
        const hudSpeed = document.getElementById('hud-speed');
        const barForce = document.getElementById('bar-force');
        const barTree = document.getElementById('bar-tree');
        const pctForce = document.getElementById('pct-force');
        const pctTree = document.getElementById('pct-tree');

        const slider = document.getElementById('time-slider');
        const timeDisplay = document.getElementById('time-display');
        const btnPlay = document.getElementById('btn-play');
        const iconPlayState = document.getElementById('icon-play-state');
        const textPlayState = document.getElementById('text-play-state');
        const legendContainer = document.getElementById('legend-container');
        const legendLabel = document.getElementById('legend-label');
        const legendMetric = document.getElementById('legend-metric');
        const legendContent = document.getElementById('legend-content');

        slider.max = totalSteps - 1;

        // Populate Header Info
        const totalSim = SIM_DATA.particles_total_simulated || N;
        document.getElementById('meta-particles-label').textContent = 
            `N = ${totalSim.toLocaleString()} particles (${N === totalSim ? '100% full fidelity' : `${N.toLocaleString()} display sample`})`;

        const initialEnergy = (SIM_DATA.metrics && SIM_DATA.metrics.length > 0) ? SIM_DATA.metrics[0].te : null;

        // Render Legend Based on Current Mode
        function updateLegendUI(maxSpeed, rankCounts) {
            if (colorMode === 'rank') {
                legendLabel.textContent = "MPI Rank Domains";
                legendMetric.textContent = `${Object.keys(rankCounts).length} Ranks`;
                let html = '<div class="rank-grid">';
                for (const rank in rankCounts) {
                    const rIdx = parseInt(rank) % rankColors.length;
                    const rData = rankColors[rIdx];
                    const count = rankCounts[rank];
                    const pct = ((count / N) * 100).toFixed(1);
                    html += `
                        <div class="rank-pill" title="Rank ${rank}: ${count} particles (${pct}%)">
                            <span class="rank-swatch" style="background: ${rData.hex};"></span>
                            <span>R${rank}: ${pct}%</span>
                        </div>
                    `;
                }
                html += '</div>';
                legendContent.innerHTML = html;
            } else if (colorMode === 'plasma') {
                legendLabel.textContent = "Velocity Heatmap (Plasma)";
                legendMetric.textContent = `|v_max| = ${maxSpeed.toFixed(3)}`;
                legendContent.innerHTML = `
                    <div class="gradient-bar gradient-plasma"></div>
                    <div class="gradient-labels">
                        <span>0.0 (Cold)</span>
                        <span>${(maxSpeed * 0.5).toFixed(2)}</span>
                        <span>${maxSpeed.toFixed(2)}</span>
                    </div>
                `;
            } else if (colorMode === 'fire') {
                legendLabel.textContent = "Solar Flare Velocity";
                legendMetric.textContent = `|v_max| = ${maxSpeed.toFixed(3)}`;
                legendContent.innerHTML = `
                    <div class="gradient-bar gradient-fire"></div>
                    <div class="gradient-labels">
                        <span>Low speed</span>
                        <span>Core shock</span>
                    </div>
                `;
            } else {
                legendLabel.textContent = "Deep Space Cyan";
                legendMetric.textContent = "Uniform Field";
                legendContent.innerHTML = `
                    <div class="gradient-bar gradient-cyan"></div>
                    <div class="gradient-labels">
                        <span>Ambient</span>
                        <span>Luminous</span>
                    </div>
                `;
            }
        }

        // Main Frame Update Function
        function updateStep(stepIdx) {
            currentStepIndex = Math.max(0, Math.min(totalSteps - 1, stepIdx));
            const stepData = SIM_DATA.steps[currentStepIndex];
            const pcoords = stepData.coords; // flat array [x0, y0, z0, speed0, rank0, ...]
            
            let maxSpeed = 0.0;
            const rankCounts = {};

            // Calculate max velocity and rank distribution
            for (let i = 0; i < N; ++i) {
                const s = pcoords[i * 5 + 3];
                const r = pcoords[i * 5 + 4];
                if (s > maxSpeed) maxSpeed = s;
                rankCounts[r] = (rankCounts[r] || 0) + 1;
            }
            if (maxSpeed < 1e-6) maxSpeed = 1.0;

            const posAttr = geometry.attributes.position;
            const colAttr = geometry.attributes.color;

            for (let i = 0; i < N; ++i) {
                const idx = i * 5;
                const px = pcoords[idx];
                const py = pcoords[idx + 1];
                const pz = pcoords[idx + 2];
                const speed = pcoords[idx + 3];
                const rank = pcoords[idx + 4];

                posAttr.setXYZ(i, px, py, pz);

                const normSpeed = speed / maxSpeed;
                if (colorMode === 'plasma') {
                    const [cr, cg, cb] = getPlasmaColor(normSpeed);
                    colAttr.setXYZ(i, cr, cg, cb);
                } else if (colorMode === 'fire') {
                    const [cr, cg, cb] = getFireColor(normSpeed);
                    colAttr.setXYZ(i, cr, cg, cb);
                } else if (colorMode === 'rank') {
                    const rc = rankColors[rank % rankColors.length].color;
                    colAttr.setXYZ(i, rc.r, rc.g, rc.b);
                } else {
                    // Deep Space Cyan
                    colAttr.setXYZ(i, 0.15 + normSpeed * 0.85, 0.75, 1.0);
                }
            }

            posAttr.needsUpdate = true;
            colAttr.needsUpdate = true;

            // Update Playback UI
            slider.value = currentStepIndex;
            timeDisplay.textContent = `${currentStepIndex} / ${totalSteps - 1}`;
            hudSpeed.textContent = `${maxSpeed.toFixed(3)} u/s`;

            // Update Telemetry Metrics
            if (SIM_DATA.metrics && SIM_DATA.metrics[currentStepIndex]) {
                const m = SIM_DATA.metrics[currentStepIndex];
                hudEnergy.textContent = m.te.toFixed(6);
                hudKePe.textContent = `${m.ke.toFixed(4)} / ${m.pe.toFixed(4)}`;
                
                // Virial Ratio: 2K / |U|
                const peAbs = Math.abs(m.pe);
                if (peAbs > 1e-8) {
                    const virialRatio = (2 * m.ke) / peAbs;
                    hudVirial.textContent = virialRatio.toFixed(4);
                } else {
                    hudVirial.textContent = "N/A";
                }

                // Energy Drift
                if (initialEnergy !== null && Math.abs(initialEnergy) > 1e-8) {
                    const drift = Math.abs((m.te - initialEnergy) / initialEnergy) * 100;
                    hudDrift.textContent = `${drift.toFixed(4)}%`;
                    
                    if (drift < 0.1) {
                        hudDrift.className = "stat-value emerald";
                        hudStatusBadge.className = "status-pill good";
                        hudStatusBadge.textContent = "● SYMPLECTIC";
                    } else if (drift < 1.0) {
                        hudDrift.className = "stat-value amber";
                        hudStatusBadge.className = "status-pill warn";
                        hudStatusBadge.textContent = "▲ BOUNDED DRIFT";
                    } else {
                        hudDrift.className = "stat-value rose";
                        hudStatusBadge.className = "status-pill alert";
                        hudStatusBadge.textContent = "■ ENERGY DRIFT";
                    }
                }

                // Compute Times & Ratio Breakdown
                const fTimeMs = m.force_time * 1000;
                const tTimeMs = m.tree_time * 1000;
                const sTimeMs = (m.step_time || (m.force_time + m.tree_time)) * 1000;
                
                hudStepTime.textContent = `${sTimeMs.toFixed(2)} ms`;
                hudTimesBreakdown.textContent = `${fTimeMs.toFixed(2)} ms / ${tTimeMs.toFixed(2)} ms`;

                const totalMeasured = fTimeMs + tTimeMs;
                if (totalMeasured > 0) {
                    const fPct = Math.round((fTimeMs / totalMeasured) * 100);
                    const tPct = 100 - fPct;
                    barForce.style.width = `${fPct}%`;
                    barTree.style.width = `${tPct}%`;
                    pctForce.textContent = `${fPct}%`;
                    pctTree.textContent = `${tPct}%`;
                }

                // Imbalance ratio
                const imb = m.imbalance || 1.0;
                hudImbalance.textContent = `${imb.toFixed(2)}x`;
                if (imb <= 1.15) {
                    hudImbalance.className = "stat-value emerald";
                } else if (imb <= 2.0) {
                    hudImbalance.className = "stat-value amber";
                } else {
                    hudImbalance.className = "stat-value rose";
                }
            }

            updateLegendUI(maxSpeed, rankCounts);
        }

        // ============================================================================
        // User Controls & Event Handlers
        // ============================================================================
        function setPlayingState(playing) {
            isPlaying = playing;
            if (isPlaying) {
                btnPlay.classList.add('btn-primary');
                textPlayState.textContent = "Pause";
                iconPlayState.innerHTML = `
                    <rect x="6" y="4" width="4" height="16" fill="currentColor"></rect>
                    <rect x="14" y="4" width="4" height="16" fill="currentColor"></rect>
                `;
            } else {
                btnPlay.classList.remove('btn-primary');
                textPlayState.textContent = "Play";
                iconPlayState.innerHTML = `
                    <polygon points="5 3 19 12 5 21 5 3" fill="currentColor"></polygon>
                `;
            }
        }

        btnPlay.addEventListener('click', () => {
            setPlayingState(!isPlaying);
        });

        document.getElementById('btn-prev').addEventListener('click', () => {
            setPlayingState(false);
            updateStep(currentStepIndex - 1);
        });

        document.getElementById('btn-next').addEventListener('click', () => {
            setPlayingState(false);
            updateStep(currentStepIndex + 1);
        });

        document.getElementById('btn-reset').addEventListener('click', () => {
            updateStep(0);
        });

        slider.addEventListener('input', (e) => {
            setPlayingState(false);
            updateStep(parseInt(e.target.value));
        });

        // Speed Pills
        document.querySelectorAll('.speed-pill').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.speed-pill').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                playbackSpeed = parseFloat(btn.dataset.speed);
            });
        });

        document.getElementById('color-mode').addEventListener('change', (e) => {
            colorMode = e.target.value;
            updateStep(currentStepIndex);
        });

        document.getElementById('point-size').addEventListener('input', (e) => {
            material.size = parseFloat(e.target.value) * 0.006;
        });

        document.getElementById('toggle-box').addEventListener('change', (e) => {
            boxLine.visible = e.target.checked;
        });

        document.getElementById('toggle-axes').addEventListener('change', (e) => {
            axesHelper.visible = e.target.checked;
        });

        // Camera Preset Angles
        document.getElementById('cam-iso').addEventListener('click', () => {
            camera.position.set(1.6, 1.3, 1.9);
            controls.target.set(0.5, 0.5, 0.5);
        });
        document.getElementById('cam-top').addEventListener('click', () => {
            camera.position.set(0.5, 2.6, 0.5);
            controls.target.set(0.5, 0.5, 0.5);
        });
        document.getElementById('cam-side').addEventListener('click', () => {
            camera.position.set(2.6, 0.5, 0.5);
            controls.target.set(0.5, 0.5, 0.5);
        });
        document.getElementById('cam-front').addEventListener('click', () => {
            camera.position.set(0.5, 0.5, 2.6);
            controls.target.set(0.5, 0.5, 0.5);
        });
        document.getElementById('cam-reset').addEventListener('click', () => {
            camera.position.set(1.6, 1.3, 1.9);
            controls.target.set(0.5, 0.5, 0.5);
            controls.reset();
        });

        // Telemetry Guide Drawer Toggle
        const drawer = document.getElementById('telemetry-drawer');
        const overlay = document.getElementById('drawer-overlay');
        function toggleGuide(open) {
            drawer.classList.toggle('open', open);
            overlay.classList.toggle('open', open);
        }
        document.getElementById('btn-open-guide').addEventListener('click', () => toggleGuide(true));
        document.getElementById('btn-close-guide').addEventListener('click', () => toggleGuide(false));
        overlay.addEventListener('click', () => toggleGuide(false));

        // Keyboard Shortcuts
        window.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

            if (e.code === 'Space') {
                e.preventDefault();
                btnPlay.click();
            } else if (e.code === 'ArrowLeft') {
                e.preventDefault();
                document.getElementById('btn-prev').click();
            } else if (e.code === 'ArrowRight') {
                e.preventDefault();
                document.getElementById('btn-next').click();
            } else if (e.code === 'KeyR') {
                e.preventDefault();
                document.getElementById('btn-reset').click();
            } else if (e.key === '?' || (e.shiftKey && e.code === 'Slash')) {
                e.preventDefault();
                toggleGuide(!drawer.classList.contains('open'));
            } else if (e.code === 'Escape') {
                toggleGuide(false);
            }
        });

        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });

        // Initialize First Frame
        updateStep(0);

        // ============================================================================
        // Main Animation Loop
        // ============================================================================
        function animate(timestamp) {
            requestAnimationFrame(animate);

            if (isPlaying && totalSteps > 1) {
                const elapsed = timestamp - lastFrameTime;
                const effectiveInterval = frameInterval / playbackSpeed;
                if (elapsed >= effectiveInterval) {
                    lastFrameTime = timestamp;
                    let next = currentStepIndex + 1;
                    if (next >= totalSteps) next = 0; // loop
                    updateStep(next);
                }
            }

            if (document.getElementById('toggle-rotate').checked) {
                boxLine.rotation.y += 0.0018;
                particlePoints.rotation.y += 0.0018;
                axesHelper.rotation.y += 0.0018;
            }

            controls.update();
            renderer.render(scene, camera);
        }

        requestAnimationFrame(animate);
    </script>
</body>
</html>
"""

def generate_web_viewer(results_dir, open_browser=False, max_particles=8000):
    """Scan results directory and generate interactive Three.js 3D viewer."""
    if not os.path.isdir(results_dir):
        print(f"Directory not found: {results_dir}")
        return

    particles_dir = os.path.join(results_dir, "particles")
    bin_files = sorted(glob.glob(os.path.join(particles_dir, "step_*.bin")))
    if not bin_files:
        print(f"No binary particle files found in {particles_dir}")
        return

    csv_candidates = glob.glob(os.path.join(results_dir, "metrics_*.csv"))
    metrics = load_metrics(csv_candidates[0]) if csv_candidates else []

    print(f"[Viewer] Found {len(bin_files)} snapshots and {len(metrics)} metrics entries.")

    # Package trajectory data
    steps_data = []
    p0 = load_particle_snapshot(bin_files[0])
    N_total = len(p0)
    
    # If N is large, sample representative particles (e.g. up to max_particles)
    sample_indices = None
    if max_particles is not None and N_total > max_particles:
        step_sample = max(1, N_total // max_particles)
        sample_indices = np.arange(0, N_total, step_sample)[:max_particles]
        N_display = len(sample_indices)
        print(f"[Viewer] Large scale run ({N_total} particles): Downsampled to {N_display} stars for smooth 60fps rendering.")
    else:
        N_display = N_total

    for bfile in bin_files:
        step_num = int(os.path.basename(bfile).replace('step_', '').replace('.bin', ''))
        p = load_particle_snapshot(bfile)
        if sample_indices is not None:
            p = p[sample_indices]

        vx, vy, vz = p['vx'], p['vy'], p['vz']
        speed = np.sqrt(vx**2 + vy**2 + vz**2)
        
        flat = []
        for i in range(len(p)):
            flat.extend([
                round(float(p['x'][i]), 5),
                round(float(p['y'][i]), 5),
                round(float(p['z'][i]), 5),
                round(float(speed[i]), 5),
                int(p['rank'][i])
            ])
        steps_data.append({
            'step_num': step_num,
            'coords': flat
        })

    payload = {
        'particles_count': N_display,
        'particles_total_simulated': N_total,
        'total_steps': len(bin_files),
        'metrics': metrics,
        'steps': steps_data
    }

    # Inject into HTML
    json_str = json.dumps(payload)
    html_content = HTML_TEMPLATE.replace('%DATA_PLACEHOLDER%', json_str)

    # Save viewer inside the results directory
    out_html = os.path.join(results_dir, "viewer.html")
    with open(out_html, 'w') as f:
        f.write(html_content)

    print(f"\n[Viewer] SUCCESS: Interactive 3D Web Viewer generated at:")
    print(f"         {os.path.abspath(out_html)}\n")

    # Also link local assets into results_dir/assets if needed
    assets_src = os.path.abspath("viewer/assets")
    assets_dst = os.path.join(results_dir, "assets")
    if os.path.isdir(assets_src) and not os.path.exists(assets_dst):
        try:
            os.symlink(assets_src, assets_dst)
        except Exception:
            pass

    return os.path.abspath(out_html)

def main():
    parser = argparse.ArgumentParser(description="Generate interactive 3D Web Viewer for AL-BH simulations.")
    parser.add_argument("--results", type=str, default="results/plummer_demo", help="Directory containing particle snapshots")
    parser.add_argument("--open", action="store_true", help="Open viewer in default web browser")
    parser.add_argument("--max-particles", type=int, default=8000, help="Max particles to embed in 3D viewer (default: 8000)")
    args = parser.parse_args()

    out_file = generate_web_viewer(args.results, args.open, args.max_particles)
    if args.open and out_file:
        import webbrowser
        webbrowser.open(f"file://{out_file}")

if __name__ == "__main__":
    main()
