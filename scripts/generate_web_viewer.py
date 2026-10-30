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
    <title>AL-BH Interactive 3D Cosmic N-Body Simulation</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; user-select: none; }
        body {
            background-color: #06070c;
            color: #d1d5db;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            overflow: hidden;
            width: 100vw;
            height: 100vh;
        }
        #canvas-container {
            width: 100%;
            height: 100%;
            position: absolute;
            top: 0;
            left: 0;
            z-index: 1;
        }
        /* Top Navigation / Title HUD */
        .header-hud {
            position: absolute;
            top: 16px;
            left: 20px;
            z-index: 10;
            background: rgba(14, 18, 28, 0.75);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 12px 20px;
            pointer-events: auto;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        }
        .header-title {
            font-size: 16px;
            font-weight: 700;
            color: #ffffff;
            letter-spacing: 0.5px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .badge {
            background: linear-gradient(135deg, #3b82f6, #8b5cf6);
            color: white;
            font-size: 10px;
            font-weight: 800;
            padding: 2px 8px;
            border-radius: 999px;
            letter-spacing: 0.5px;
        }
        .header-subtitle {
            font-size: 12px;
            color: #94a3b8;
            margin-top: 4px;
        }

        /* Top Right Diagnostics HUD */
        .stats-hud {
            position: absolute;
            top: 16px;
            right: 20px;
            z-index: 10;
            background: rgba(14, 18, 28, 0.75);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 14px 20px;
            font-size: 12px;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            min-width: 280px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        }
        .stat-row {
            display: flex;
            justify-content: space-between;
            padding: 3px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        }
        .stat-label { color: #64748b; }
        .stat-value { color: #f1f5f9; font-weight: 600; }
        .stat-value.highlight { color: #38bdf8; }
        .stat-value.green { color: #34d399; }
        .stat-value.amber { color: #fbbf24; }

        /* Bottom Floating Control Bar */
        .controls-hud {
            position: absolute;
            bottom: 24px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 10;
            background: rgba(14, 18, 28, 0.85);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 16px;
            padding: 12px 24px;
            display: flex;
            align-items: center;
            gap: 16px;
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.6);
        }
        .btn {
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.12);
            color: #f1f5f9;
            padding: 8px 14px;
            border-radius: 8px;
            font-size: 13px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .btn:hover {
            background: rgba(255, 255, 255, 0.15);
            border-color: rgba(255, 255, 255, 0.25);
            transform: translateY(-1px);
        }
        .btn.active {
            background: #3b82f6;
            border-color: #60a5fa;
            color: white;
            box-shadow: 0 0 16px rgba(59, 130, 246, 0.4);
        }
        .scrubber-container {
            display: flex;
            align-items: center;
            gap: 10px;
            min-width: 220px;
        }
        input[type=range] {
            -webkit-appearance: none;
            width: 100%;
            height: 6px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
            outline: none;
        }
        input[type=range]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: #38bdf8;
            cursor: pointer;
            box-shadow: 0 0 8px #38bdf8;
            transition: transform 0.1s;
        }
        input[type=range]::-webkit-slider-thumb:hover {
            transform: scale(1.2);
        }
        .time-label {
            font-family: ui-monospace, monospace;
            font-size: 13px;
            color: #94a3b8;
            min-width: 65px;
        }

        /* Side Settings Panel */
        .settings-hud {
            position: absolute;
            left: 20px;
            bottom: 24px;
            z-index: 10;
            background: rgba(14, 18, 28, 0.75);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 12px 16px;
            display: flex;
            flex-direction: column;
            gap: 10px;
            font-size: 12px;
        }
        .setting-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 14px;
        }
        select {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.15);
            color: #f1f5f9;
            padding: 4px 8px;
            border-radius: 6px;
            outline: none;
            font-size: 12px;
            cursor: pointer;
        }

        /* Keyboard Shortcuts helper */
        .shortcuts {
            font-size: 10px;
            color: #64748b;
            text-align: center;
            margin-top: 4px;
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

    <!-- Top Left Title -->
    <div class="header-hud">
        <div class="header-title">
            <span>AL-BH 3D N-Body Simulator</span>
            <span class="badge">LEAPFROG · BARNES-HUT</span>
        </div>
        <div class="header-subtitle">Adaptive Linearized Octree Spatial Gravitational Engine</div>
    </div>

    <!-- Top Right Real-Time Diagnostics HUD -->
    <div class="stats-hud">
        <div class="stat-row">
            <span class="stat-label">Particles (N):</span>
            <span class="stat-value highlight" id="hud-particles">-</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Current Step:</span>
            <span class="stat-value" id="hud-step">0</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Total Energy (E):</span>
            <span class="stat-value highlight" id="hud-energy">-</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Kinetic (KE) / Potential:</span>
            <span class="stat-value" id="hud-ke-pe">- / -</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Energy Drift |ΔE/E₀|:</span>
            <span class="stat-value green" id="hud-drift">0.00%</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Force Compute Time:</span>
            <span class="stat-value amber" id="hud-force-time">-</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Octree Build Time:</span>
            <span class="stat-value" id="hud-tree-time">-</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Max Speed |v_max|:</span>
            <span class="stat-value" id="hud-speed">-</span>
        </div>
    </div>

    <!-- Side Settings HUD -->
    <div class="settings-hud">
        <div class="setting-item">
            <span>Coloring Mode:</span>
            <select id="color-mode">
                <option value="plasma">Velocity Heatmap (Plasma)</option>
                <option value="rank">MPI Rank Domains</option>
                <option value="cyan">Deep Space Cyan</option>
                <option value="fire">Solar Flare (Fire)</option>
            </select>
        </div>
        <div class="setting-item">
            <span>Star Size:</span>
            <input type="range" id="point-size" min="1" max="15" value="4" style="width: 80px;">
        </div>
        <div class="setting-item">
            <span>Bounding Box:</span>
            <input type="checkbox" id="toggle-box" checked>
        </div>
        <div class="setting-item">
            <span>Auto Rotate:</span>
            <input type="checkbox" id="toggle-rotate">
        </div>
    </div>

    <!-- Bottom Floating Playback HUD -->
    <div class="controls-hud">
        <button class="btn active" id="btn-play">❚❚ Pause</button>
        <button class="btn" id="btn-prev">◀ Step</button>
        <button class="btn" id="btn-next">Step ▶</button>
        <button class="btn" id="btn-reset">↺ Reset</button>

        <div class="scrubber-container">
            <input type="range" id="time-slider" min="0" max="0" value="0">
            <span class="time-label" id="time-display">0 / 0</span>
        </div>

        <button class="btn" id="btn-speed">1.0x</button>
        <div class="shortcuts">[Space: Play/Pause | ← / →: Step | Drag: 3D Rotate]</div>
    </div>

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
        scene.fog = new THREE.FogExp2(0x06070c, 0.15);

        const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.01, 100);
        camera.position.set(1.5, 1.2, 1.8);

        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio || 1);
        renderer.setClearColor(0x06070c, 1);
        container.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.target.set(0.5, 0.5, 0.5);

        // Grid & Box Helper
        const boxGeometry = new THREE.BoxGeometry(1.0, 1.0, 1.0);
        const boxEdges = new THREE.EdgesGeometry(boxGeometry);
        const boxLine = new THREE.LineSegments(boxEdges, new THREE.LineBasicMaterial({ color: 0x223344, transparent: true, opacity: 0.5 }));
        boxLine.position.set(0.5, 0.5, 0.5);
        scene.add(boxLine);

        // Soft ambient illumination
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
        scene.add(ambientLight);

        // Particle Glow Texture Generator
        function createStarTexture() {
            const canvas = document.createElement('canvas');
            canvas.width = 64;
            canvas.height = 64;
            const ctx = canvas.getContext('2d');
            const gradient = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
            gradient.addColorStop(0.0, 'rgba(255, 255, 255, 1.0)');
            gradient.addColorStop(0.2, 'rgba(230, 240, 255, 0.85)');
            gradient.addColorStop(0.5, 'rgba(80, 150, 255, 0.3)');
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
            size: 0.025,
            vertexColors: true,
            map: createStarTexture(),
            transparent: true,
            blending: THREE.AdditiveBlending,
            depthWrite: false,
        });

        const particlePoints = new THREE.Points(geometry, material);
        scene.add(particlePoints);

        // Colors Palette for MPI Ranks
        const rankColors = [
            new THREE.Color(0x38bdf8), // Cyan
            new THREE.Color(0xf43f5e), // Rose
            new THREE.Color(0xfacc15), // Amber
            new THREE.Color(0x4ade80), // Lime
            new THREE.Color(0xa855f7), // Purple
            new THREE.Color(0xf97316), // Orange
            new THREE.Color(0x06b6d4), // Teal
            new THREE.Color(0xec4899), // Pink
        ];

        // Plasma Color Mapping
        function getPlasmaColor(t) {
            t = Math.max(0, Math.min(1, t));
            // smooth plasma curve: blue -> violet -> orange -> yellow -> white
            const r = Math.min(1.0, Math.max(0.05, 1.5 * t));
            const g = Math.min(1.0, Math.max(0.05, Math.pow(t, 2) * 1.2));
            const b = Math.min(1.0, Math.max(0.1, 1.0 - t * 0.8 + Math.pow(t, 4) * 0.8));
            return [r, g, b];
        }

        function getFireColor(t) {
            t = Math.max(0, Math.min(1, t));
            const r = Math.min(1.0, 0.4 + t * 0.8);
            const g = Math.min(1.0, Math.pow(t, 1.8));
            const b = Math.min(1.0, Math.pow(t, 3.5));
            return [r, g, b];
        }

        // ============================================================================
        // Simulation Playback State Management
        // ============================================================================
        let currentStepIndex = 0;
        let isPlaying = true;
        let playbackSpeed = 1.0;
        let lastFrameTime = performance.now();
        let frameInterval = 60; // ms per step at 1x
        let colorMode = 'plasma';

        // HUD Elements
        const hudParticles = document.getElementById('hud-particles');
        const hudStep = document.getElementById('hud-step');
        const hudEnergy = document.getElementById('hud-energy');
        const hudKePe = document.getElementById('hud-ke-pe');
        const hudDrift = document.getElementById('hud-drift');
        const hudForce = document.getElementById('hud-force-time');
        const hudTree = document.getElementById('hud-tree-time');
        const hudSpeed = document.getElementById('hud-speed');

        const slider = document.getElementById('time-slider');
        const timeDisplay = document.getElementById('time-display');
        const btnPlay = document.getElementById('btn-play');
        const btnSpeed = document.getElementById('btn-speed');

        slider.max = totalSteps - 1;
        hudParticles.textContent = N.toLocaleString();

        const initialEnergy = (SIM_DATA.metrics && SIM_DATA.metrics.length > 0) ? SIM_DATA.metrics[0].te : null;

        function updateStep(stepIdx) {
            currentStepIndex = Math.max(0, Math.min(totalSteps - 1, stepIdx));
            const stepData = SIM_DATA.steps[currentStepIndex];
            const pcoords = stepData.coords; // flat array [x0, y0, z0, speed0, rank0, ...]
            
            let maxSpeed = 0.0;
            // First pass for max speed
            for (let i = 0; i < N; ++i) {
                const s = pcoords[i * 5 + 3];
                if (s > maxSpeed) maxSpeed = s;
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
                    const rc = rankColors[rank % rankColors.length];
                    colAttr.setXYZ(i, rc.r, rc.g, rc.b);
                } else {
                    // Deep Space Cyan
                    colAttr.setXYZ(i, 0.2 + normSpeed * 0.8, 0.8, 1.0);
                }
            }

            posAttr.needsUpdate = true;
            colAttr.needsUpdate = true;

            // Update UI HUD
            slider.value = currentStepIndex;
            timeDisplay.textContent = `${currentStepIndex} / ${totalSteps - 1}`;
            hudStep.textContent = `${stepData.step_num} / ${SIM_DATA.total_steps}`;
            hudSpeed.textContent = maxSpeed.toFixed(3);

            if (SIM_DATA.metrics && SIM_DATA.metrics[currentStepIndex]) {
                const m = SIM_DATA.metrics[currentStepIndex];
                hudEnergy.textContent = m.te.toFixed(6);
                hudKePe.textContent = `${m.ke.toFixed(4)} / ${m.pe.toFixed(4)}`;
                if (initialEnergy !== null && initialEnergy !== 0) {
                    const drift = Math.abs((m.te - initialEnergy) / initialEnergy) * 100;
                    hudDrift.textContent = `${drift.toFixed(4)}%`;
                }
                hudForce.textContent = `${(m.force_time * 1000).toFixed(2)} ms`;
                hudTree.textContent = `${(m.tree_time * 1000).toFixed(2)} ms`;
            }
        }

        // ============================================================================
        // Event Listeners and Controls
        // ============================================================================
        btnPlay.addEventListener('click', () => {
            isPlaying = !isPlaying;
            btnPlay.textContent = isPlaying ? "❚❚ Pause" : "▶ Play";
            btnPlay.classList.toggle('active', isPlaying);
        });

        document.getElementById('btn-prev').addEventListener('click', () => {
            isPlaying = false;
            btnPlay.textContent = "▶ Play";
            btnPlay.classList.remove('active');
            updateStep(currentStepIndex - 1);
        });

        document.getElementById('btn-next').addEventListener('click', () => {
            isPlaying = false;
            btnPlay.textContent = "▶ Play";
            btnPlay.classList.remove('active');
            updateStep(currentStepIndex + 1);
        });

        document.getElementById('btn-reset').addEventListener('click', () => {
            updateStep(0);
        });

        slider.addEventListener('input', (e) => {
            isPlaying = false;
            btnPlay.textContent = "▶ Play";
            btnPlay.classList.remove('active');
            updateStep(parseInt(e.target.value));
        });

        const speeds = [0.5, 1.0, 2.0, 5.0];
        let speedIdx = 1;
        btnSpeed.addEventListener('click', () => {
            speedIdx = (speedIdx + 1) % speeds.length;
            playbackSpeed = speeds[speedIdx];
            btnSpeed.textContent = `${playbackSpeed.toFixed(1)}x`;
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

        window.addEventListener('keydown', (e) => {
            if (e.code === 'Space') {
                e.preventDefault();
                btnPlay.click();
            } else if (e.code === 'ArrowLeft') {
                e.preventDefault();
                document.getElementById('btn-prev').click();
            } else if (e.code === 'ArrowRight') {
                e.preventDefault();
                document.getElementById('btn-next').click();
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
        // Animation Loop
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
                boxLine.rotation.y += 0.002;
                particlePoints.rotation.y += 0.002;
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
