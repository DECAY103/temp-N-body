# AL-BH: Architecture & Module Plan

## Environment Constraints
| Item | Value |
|---|---|
| CPU | AMD Ryzen 7 7840HS — 8 cores / 16 threads |
| OS | Ubuntu Linux |
| MPI | **Not installed** → will install `mpich` via apt |
| Python | 3.13 with numpy, matplotlib, scipy |
| Viz | Python (matplotlib 3D + optional vispy) |
| Purpose | Demo / course presentation |

## Adapted Execution Plan

Since this is on a personal laptop with 8 physical cores:
- **MPI ranks**: test with 1, 2, 4, 8 (skip 16)
- **OpenMP threads/rank**: test with 1, 2, 4, 8
- **Particle counts**: 10⁴, 5×10⁴, 10⁵ (skip 10⁶ for time)
- **Timesteps**: 50–100 for demos, 200 for benchmark runs

---

## Project Structure

```
AL-BH/
├── ARCHITECTURE.md              ← this file
├── AL_BH_Project_Context.md     ← project spec
├── project-deliverables-plan.md ← deliverables spec
│
├── include/                     ← C++ headers
│   ├── particle.hpp             ← Particle struct, constants
│   ├── config.hpp               ← SimulationConfig (N, dt, theta, softening…)
│   ├── morton.hpp               ← Morton key encoding/decoding
│   ├── octree.hpp               ← Linearized array-based Octree
│   ├── octree_pointer.hpp       ← Pointer-based Octree (baseline comparison)
│   ├── force.hpp                ← Barnes-Hut force computation
│   ├── partition.hpp            ← Morton-based domain decomposition
│   ├── communication.hpp        ← MPI ghost/LET exchange
│   ├── load_balancer.hpp        ← Adaptive rebalancing
│   ├── integrator.hpp           ← Leapfrog time integration
│   └── metrics.hpp              ← Timing, counters, CSV logging
│
├── src/                         ← C++ implementation
│   ├── main.cpp                 ← Entry point, CLI parsing
│   ├── particle_generator.cpp   ← Uniform + Plummer distributions
│   ├── morton.cpp               ← Bit-interleaving, quantization
│   ├── octree.cpp               ← Linearized Octree build + traversal
│   ├── octree_pointer.cpp       ← Pointer-based Octree (baseline)
│   ├── force.cpp                ← BH force engine
│   ├── partition.cpp            ← Morton partitioning across ranks
│   ├── communication.cpp        ← MPI send/recv (blocking + non-blocking)
│   ├── load_balancer.cpp        ← Imbalance detection + redistribution
│   ├── integrator.cpp           ← Leapfrog kick-drift-kick
│   ├── simulation.cpp           ← Timestep loop orchestrator
│   └── metrics.cpp              ← CSV/JSON output
│
├── baselines/                   ← Comparison implementations
│   ├── serial_bh.cpp            ← Single-process Barnes-Hut
│   ├── static_partition_bh.cpp  ← MPI with fixed geometric partitions
│   └── no_rebalance_bh.cpp      ← MPI + Morton but no adaptive rebalancing
│
├── tests/                       ← Correctness tests
│   ├── test_morton.cpp
│   ├── test_octree.cpp
│   ├── test_force.cpp
│   ├── test_partition.cpp
│   └── test_load_balance.cpp
│
├── viz/                         ← Python visualization
│   ├── plot_performance.py      ← Generate all 8 plot types from CSVs
│   ├── live_sim.py              ← Real-time 3D particle viewer
│   └── utils.py                 ← Shared plotting helpers
│
├── scripts/                     ← Automation
│   ├── setup.sh                 ← Install MPI + build deps
│   ├── build.sh                 ← CMake build script
│   ├── run_experiments.sh       ← Parameter sweep runner
│   └── run_all.sh               ← One-command: build + test + run + plot
│
├── results/                     ← Experiment output
│   ├── raw/                     ← CSV/JSON logs
│   └── plots/                   ← Generated PNG/SVG plots
│
├── CMakeLists.txt               ← Build system
└── README.md                    ← Build & run instructions
```

---

## Module Responsibilities

### Module 1: `particle.hpp / particle_generator.cpp`
**Purpose**: Define particle data layout and generate initial conditions.

```cpp
struct Particle {
    double x, y, z;       // position
    double vx, vy, vz;    // velocity
    double ax, ay, az;    // acceleration
    double mass;
    uint64_t morton_key;  // computed Morton key
    int owner_rank;       // MPI rank that owns this particle
};
```

- **Uniform distribution**: Random positions in [0, box_size]³, unit mass
- **Plummer model**: Density ∝ (1 + r²/a²)^(-5/2), creates dense core + sparse halo
- Deterministic via configurable seed

### Module 2: `config.hpp`
**Purpose**: Centralized simulation parameters.

```cpp
struct SimConfig {
    int N;                    // particle count
    int timesteps;            // number of steps
    double dt;                // time step size
    double theta;             // BH opening angle
    double softening;         // gravitational softening
    double G;                 // gravitational constant
    double box_size;          // simulation domain size
    int distribution;         // 0=uniform, 1=plummer
    int seed;                 // random seed
    double rebalance_threshold; // imbalance trigger
    int num_threads;          // OpenMP threads per rank
    // ... output paths, flags
};
```

### Module 3: `morton.hpp / morton.cpp`
**Purpose**: Map 3D coordinates → 1D Morton (Z-order) keys.

- Quantize `(x,y,z)` to 21-bit integers (fits in 63-bit key)
- Bit-interleave using magic-number method or lookup table
- Provide `encode(x,y,z) → uint64_t` and `decode(key) → (x,y,z)`
- Sort particles by Morton key for spatial locality

### Module 4: `octree.hpp / octree.cpp` (Linearized)
**Purpose**: Cache-friendly array-based Octree for Barnes-Hut.

```cpp
struct OctreeNode {
    double cx, cy, cz;       // center of mass
    double total_mass;        // aggregate mass
    double x_min, y_min, z_min; // bounding box
    double size;              // node side length
    int first_child;          // index of first child in array (-1 if leaf)
    int particle_index;       // index if leaf node (-1 if internal)
    int particle_count;       // particles in this subtree
};
```

- Nodes stored in contiguous `std::vector<OctreeNode>`
- Children at indices `[first_child, first_child+7]`
- BFS or DFS order for cache-friendly traversal
- `build(particles)` and `compute_aggregates()` methods

### Module 5: `octree_pointer.hpp / octree_pointer.cpp` (Baseline)
**Purpose**: Traditional pointer-based Octree for cache-miss comparison.

- Same logical structure but `OctreeNode*` children[8]
- Heap-allocated nodes → pointer chasing → more cache misses
- Used ONLY for the linearized-vs-pointer comparison experiment

### Module 6: `force.hpp / force.cpp`
**Purpose**: Barnes-Hut force computation engine.

- `compute_force(particle, tree, theta)`: Walk tree with opening criterion
- Opening test: `node.size / distance(particle, node.com) < theta`
- If passes → use node's center-of-mass as approximation
- If fails → recurse into children
- Softened gravity: `F = G * m1 * m2 / (r² + ε²)^(3/2) * r_vec`
- OpenMP parallelization: `#pragma omp parallel for` over target particles

### Module 7: `partition.hpp / partition.cpp`
**Purpose**: Morton-based domain decomposition across MPI ranks.

- Sort particles by Morton key
- Split into `num_ranks` contiguous ranges with ~equal particle counts
- Track partition boundaries as Morton key ranges
- Assign ownership: `particle.owner_rank = rank`

### Module 8: `communication.hpp / communication.cpp`
**Purpose**: MPI ghost/LET data exchange.

- **LET (Locally Essential Tree)**: Each rank builds a lightweight tree skeleton; applies opening criterion to determine which remote data is needed
- **Blocking version**: `MPI_Send / MPI_Recv` — wait for data before computing
- **Non-blocking version**: `MPI_Isend / MPI_Irecv` — initiate exchange, do local force computation, then `MPI_Waitall`
- Both versions for A/B comparison

### Module 9: `load_balancer.hpp / load_balancer.cpp`
**Purpose**: Monitor workload imbalance and redistribute particles.

- Each rank reports: particle count, force-computation time
- Compute `imbalance = max_work / min_work`
- If `imbalance > threshold`: binary search on Morton curve for new boundaries
- Redistribute particles via MPI
- Log: timestep, imbalance before/after, particles moved, rebalance time

### Module 10: `integrator.hpp / integrator.cpp`
**Purpose**: Leapfrog (kick-drift-kick) time integration.

```
v(t + dt/2) = v(t) + a(t) * dt/2       // half-kick
x(t + dt)   = x(t) + v(t + dt/2) * dt  // drift
compute a(t + dt)                        // new forces
v(t + dt)   = v(t + dt/2) + a(t+dt) * dt/2  // half-kick
```

- Symplectic integrator — good energy conservation
- Simple, well-understood, standard for N-body

### Module 11: `metrics.hpp / metrics.cpp`
**Purpose**: Performance measurement and logging.

- `MPI_Wtime()` for wall-clock timing
- Per-timestep: force time, comm time, rebalance time, total time
- Per-rank: particle count, workload
- Imbalance ratio every timestep
- Output to CSV with full config metadata header
- Optional: `perf stat` wrapper for cache-miss measurement

### Module 12: `viz/` (Python)
**Purpose**: All visualization, plots, and live demo.

- `plot_performance.py`: Read CSVs → generate all 8 plot types
- `live_sim.py`: Real-time 3D viewer using matplotlib 3D or vispy
  - Reads particle state files dumped by simulation
  - Color-codes by rank ownership
  - Shows Octree grid overlay (toggleable)
  - Live stats HUD

---

## Design Choices (Not Specified by Project Docs)

| Choice | Decision | Rationale |
|---|---|---|
| Time integrator | Leapfrog (KDK) | Symplectic, simple, standard for N-body |
| Gravitational constant G | 1.0 | Simplifies; we're testing parallel perf not physics |
| Softening ε | 0.01 × box_size | Prevents singularities without over-smoothing |
| Unit system | Dimensionless (G=1, box=[0,1]³) | Standard N-body convention |
| Boundary conditions | Open (no periodic wrap) | Simpler; particles can escape |
| Morton bits per dim | 21 bits | 63-bit key, ~2M resolution per axis |
| Octree node layout | BFS order in array | Cache-friendly breadth-first |
| LET protocol | Broadcast tree skeleton → request needed nodes | Simple, correct |
| Output format | CSV with JSON metadata sidecar | Easy to parse in Python |

---

## Milestone Execution Order

Each milestone is self-contained. If tokens run out, the next model can pick up from where we left off by reading this document and checking which files exist.

### Milestone 1: Serial Barnes-Hut Baseline
**Files**: `particle.hpp`, `config.hpp`, `octree.hpp`, `force.hpp`, `integrator.hpp`, `main.cpp`, `particle_generator.cpp`, `octree.cpp`, `force.cpp`, `integrator.cpp`, `simulation.cpp`
**Test**: Run with N=1000, 10 timesteps, verify energy conservation
**Status**: 🔲 TODO

### Milestone 2: Morton Keys + Linearized Octree
**Files**: `morton.hpp`, `morton.cpp`, updated `octree.cpp`
**Test**: Verify Morton encoding/decoding round-trip, sorted order
**Status**: 🔲 TODO

### Milestone 3: MPI Spatial Decomposition
**Files**: `partition.hpp`, `partition.cpp`, `communication.hpp`, `communication.cpp`, updated `main.cpp`
**Deps**: Install `mpich` via `sudo apt install mpich`
**Test**: Run with 2,4,8 ranks; verify all particles accounted for
**Status**: 🔲 TODO

### Milestone 4: OpenMP Parallelism
**Files**: Updated `force.cpp`, `octree.cpp` with `#pragma omp` directives
**Test**: Run with 1,2,4,8 threads; verify same results, measure speedup
**Status**: 🔲 TODO

### Milestone 5: LET + Non-blocking MPI
**Files**: Updated `communication.cpp` with `MPI_Isend/Irecv`
**Test**: Compare blocking vs non-blocking timing
**Status**: 🔲 TODO

### Milestone 6: Adaptive Load Balancing
**Files**: `load_balancer.hpp`, `load_balancer.cpp`
**Test**: Run on Plummer distribution, verify rebalancing triggers
**Status**: 🔲 TODO

### Milestone 7: Experiments + Visualization
**Files**: `scripts/run_experiments.sh`, `viz/plot_performance.py`, `viz/live_sim.py`
**Test**: Generate all 8 plot types, run live demo
**Status**: 🔲 TODO

---

## Build Commands (after setup)

```bash
# One-time setup
sudo apt install mpich libmpich-dev cmake
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8

# Run serial baseline
./al_bh --mode serial --N 10000 --steps 100 --theta 0.5

# Run MPI
mpirun -np 4 ./al_bh --mode full --N 50000 --steps 100 --theta 0.5 --threads 4

# Run experiments
bash scripts/run_experiments.sh

# Generate plots
python3 viz/plot_performance.py results/raw/ results/plots/

# Live visualization
python3 viz/live_sim.py results/raw/particles/
```

---

## Handoff Notes for Next Model

If tokens run out mid-implementation:
1. **Read this file first** — it's the source of truth for architecture
2. **Check which files exist** in `include/`, `src/`, `baselines/`, `viz/`
3. **Check milestone status** above — TODO items are next
4. **The CMakeLists.txt** defines what's buildable
5. **All design choices** are documented in the table above — don't change them without reason
6. **Test each milestone** before moving to the next
