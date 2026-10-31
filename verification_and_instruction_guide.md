# AL-BH: Implementation Updates & Verification Instruction Guide

This document details all implemented fixes from the **Must Fix**, **Should Fix**, and **Nice-to-Have** categories, and provides an end-to-end, empirical verification guide with exact commands and expected outputs.

---

## Part 1 — Summary of Changes Implemented

### 🔴 Must Fix Items
1. **Isolated No-Rebalance Baseline (`mpi_morton`)**:
   - In [`src/simulation.cpp`](file:///home/res/Documents/PC/Project/src/simulation.cpp#L157), `mpi_morton` is strictly treated as a static Morton baseline without rebalancing.
   - Dynamic rebalancing is now exclusively enabled in `full` mode (`enable_rebalance = (config.mode == "full")`).

2. **Genuine Non-Blocking MPI Overlap**:
   - In [`include/communication.hpp`](file:///home/res/Documents/PC/Project/include/communication.hpp) and [`src/communication.cpp`](file:///home/res/Documents/PC/Project/src/communication.cpp), implemented `start_exchange_all_particles()` and `wait_exchange_all_particles()`.
   - In [`src/simulation.cpp`](file:///home/res/Documents/PC/Project/src/simulation.cpp#L264), the simulation issues asynchronous `MPI_Iallgatherv`, performs independent local tree construction while network transfer is in flight, and only synchronizes when remote data is needed for the global tree.

3. **Plummer Scale Radius Passthrough (`plummer_a`)**:
   - Added `plummer_a` parameter to `generate_particles()` in [`include/particle.hpp`](file:///home/res/Documents/PC/Project/include/particle.hpp#L54) and [`src/particle_generator.cpp`](file:///home/res/Documents/PC/Project/src/particle_generator.cpp#L106).
   - In [`src/main.cpp`](file:///home/res/Documents/PC/Project/src/main.cpp#L40), forwarded `config.plummer_a` to the generator.

4. **Force Vector Error Verification**:
   - In [`tests/test_force.cpp`](file:///home/res/Documents/PC/Project/tests/test_force.cpp#L67), updated the accuracy test to evaluate 3D vector differences $\frac{\|\vec a_{BH} - \vec a_{direct}\|}{\|\vec a_{direct}\|}$ instead of scalar magnitudes, ensuring directional errors cannot be masked.

5. **Static Geometric Slab Sorting Before Scatter**:
   - Added `sort_by_geometric_slabs()` in [`include/partition.hpp`](file:///home/res/Documents/PC/Project/include/partition.hpp#L36) and [`src/partition.cpp`](file:///home/res/Documents/PC/Project/src/partition.cpp#L49).
   - In [`src/simulation.cpp`](file:///home/res/Documents/PC/Project/src/simulation.cpp#L166), particles are sorted by x-coordinate slab before `MPI_Scatterv`, guaranteeing that each MPI rank receives particles belonging to its spatial slab.
   - Added spatial ownership assertion tests in [`tests/test_partition.cpp`](file:///home/res/Documents/PC/Project/tests/test_partition.cpp#L121).

---

### 🟡 Should Fix Items
6. **Coincident Particles at Octree `MAX_DEPTH` (Mass Conservation)**:
   - In [`include/octree.hpp`](file:///home/res/Documents/PC/Project/include/octree.hpp), [`src/octree.cpp`](file:///home/res/Documents/PC/Project/src/octree.cpp), [`include/octree_pointer.hpp`](file:///home/res/Documents/PC/Project/include/octree_pointer.hpp), and [`src/octree_pointer.cpp`](file:///home/res/Documents/PC/Project/src/octree_pointer.cpp), coincident and near-coincident particles at `MAX_DEPTH` (cell size $\approx 4.7 \times 10^{-7}$) are aggregated into leaf overflow buckets without loss of mass or particle identity.
   - Added coincident mass aggregation and acceleration verification tests in [`tests/test_octree.cpp`](file:///home/res/Documents/PC/Project/tests/test_octree.cpp#L134) and [`tests/test_force.cpp`](file:///home/res/Documents/PC/Project/tests/test_force.cpp#L170).

7. **Target Containment & Stable Particle ID Self-Interaction Exclusion**:
   - Added `int id` to [`struct Particle`](file:///home/res/Documents/PC/Project/include/particle.hpp#L22), initialized monotonically by generators.
   - In [`src/force.cpp`](file:///home/res/Documents/PC/Project/src/force.cpp#L70), Barnes-Hut opening criterion checks spatial containment (`contains_target`), strictly preventing internal nodes containing the target particle from being approximated as monopoles.
   - Self-interaction exclusion now matches against stable `particle_id` rather than rank-local array indices, preventing incorrect exclusion in distributed MPI mode.

8. **Per-Rank Telemetry Collection**:
   - In [`src/simulation.cpp`](file:///home/res/Documents/PC/Project/src/simulation.cpp#L355), collected per-rank particle counts, compute times, and migration counts using `MPI_Gather`.
   - When `--verbose 1` is specified, rank 0 logs per-rank breakdown to the console (e.g. `Per-rank: [R0: n=500 t=0.0011s] [R1: n=500 t=0.0011s]`).

9. **CTest Integration in CMake**:
   - Added `enable_testing()` and registered all 6 test suites (`test_morton`, `test_octree`, `test_force`, `test_partition`, `test_load_balancer`, and `test_mpi_comm`) via `add_test()` in [`CMakeLists.txt`](file:///home/res/Documents/PC/Project/CMakeLists.txt#L111).
   - All tests can now be executed via `ctest --test-dir build --output-on-failure`.

10. **Global Gravitational Potential Energy**:
    - In [`src/simulation.cpp`](file:///home/res/Documents/PC/Project/src/simulation.cpp#L347), computed true global potential energy across cross-rank particle pairs for $N \le 5000$ using `global_particles`, and set to `NaN` (instead of misleading `0.0`) for $N > 5000$.

11. **Accurate Visualization Snapshot Dumps**:
    - In [`src/simulation.cpp`](file:///home/res/Documents/PC/Project/src/simulation.cpp#L404), re-gathered updated post-kick particles before dumping state to ensure rank ownership and coordinates are accurate.

12. **CLI Argument Validation**:
    - Added `--mass <float>` support and strict validation for all CLI parameters in [`src/config.cpp`](file:///home/res/Documents/PC/Project/src/config.cpp#L56). Unknown options are rejected with clear error messages.

---

### 🟢 Nice-to-Have Items
13. **Unification of `LoadBalancer`**:
    - Connected `LoadBalancer::check_and_rebalance()` in [`src/simulation.cpp`](file:///home/res/Documents/PC/Project/src/simulation.cpp#L338) for single-rank fallback runs.
14. **Automated Verification Suite**:
    - Created [`scripts/verify_all.sh`](file:///home/res/Documents/PC/Project/scripts/verify_all.sh) to execute end-to-end verification automatically.

---

## Part 2 — Step-by-Step Command Guide to Verify Everything

Run these commands in your terminal from the project root (`/home/res/Documents/PC/Project`).

### 1. Automated All-In-One Verification
```bash
./scripts/verify_all.sh
```
Runs the full suite: builds the code, checks library linkage, runs CTest, verifies OpenMP scaling, executes multi-rank MPI runs, and tests non-blocking overlap.

---

### 2. Verify Library Linkage (Proves MPI & OpenMP are Compiled In)
```bash
# Check MPI dynamic library:
ldd build/al_bh | grep mpi
# Expected: libmpi.so.XX => /path/to/libmpi.so

# Check OpenMP dynamic library:
ldd build/al_bh | grep gomp
# Expected: libgomp.so.1 => /path/to/libgomp.so.1
```

---

### 3. Run the Automated CTest Suite
```bash
ctest --test-dir build --output-on-failure
```
**Expected output:**
```
100% tests passed out of 6
Total Test time = ~2.8 sec
```

---

### 4. Verify OpenMP Multi-Threading & CPU Scaling
```bash
# 1. Print OpenMP runtime environment at startup:
OMP_DISPLAY_ENV=TRUE ./build/al_bh --N 1000 --steps 1 --threads 4 --mode serial

# 2. Compare 1 thread vs 4 threads (observe speedup in Force computation time):
./build/al_bh --N 10000 --steps 10 --threads 1 --mode serial --verbose 0
./build/al_bh --N 10000 --steps 10 --threads 4 --mode serial --verbose 0

# 3. Check CPU core activity during a larger run:
OMP_NUM_THREADS=4 ./build/al_bh --N 20000 --steps 20 --threads 4 --mode serial
# In a second terminal, run `htop` to observe 4 CPU cores at 100% load.
```

---

### 5. Verify Real Multi-Process MPI Execution
```bash
# 1. Run 2 MPI processes and observe per-rank breakdown:
mpirun -np 2 ./build/al_bh --N 2000 --steps 5 --mode full --verbose 1

# Expected per-rank lines in console:
# [Parallel Simulation (full)] Starting 5 timesteps across 2 real MPI processes
# Per-rank: [R0: n=1000 t=0.0021s] [R1: n=1000 t=0.0021s]

# 2. Run 4 MPI processes:
mpirun -np 4 ./build/al_bh --N 4000 --steps 5 --mode full --verbose 1
# Expected: Per-rank output showing R0, R1, R2, R3.

# 3. Verify real OS processes with ps:
# Run in background and check process table:
mpirun -np 4 ./build/al_bh --N 20000 --steps 50 --mode full > /dev/null &
ps aux | grep al_bh | grep -v grep
# Expected: 4 distinct process IDs (PIDs) running concurrently.
```

---

### 6. Verify Non-Blocking Communication Overlap
```bash
# Run the dedicated MPI communication and overlap test:
mpirun -np 2 ./build/test_mpi_comm

# Compare blocking vs non-blocking communication times:
mpirun -np 2 ./build/al_bh --N 5000 --steps 10 --mode full --nonblocking 0 --verbose 1
mpirun -np 2 ./build/al_bh --N 5000 --steps 10 --mode full --nonblocking 1 --verbose 1
```

---

### 7. Verify Domain Decomposition & Load Balancing Isolation
```bash
# 1. Static Geometric Partitioning (notice imbalance on clustered Plummer galaxy):
mpirun -np 2 ./build/al_bh --N 2000 --steps 5 --dist 1 --mode mpi_static --verbose 1

# 2. Morton Partitioning without rebalancing (mpi_morton baseline):
# Observe: Migrated count remains 0 throughout the simulation.
mpirun -np 2 ./build/al_bh --N 2000 --steps 5 --dist 1 --mode mpi_morton --verbose 1

# 3. Full AL-BH (adaptive rebalancing):
# Observe: Migrated count updates when imbalance exceeds threshold (1.2).
mpirun -np 2 ./build/al_bh --N 2000 --steps 5 --dist 1 --mode full --threshold 1.1 --verbose 1
```

---

### 8. Verify Hardware Cache Performance with Linux `perf`
```bash
# Measure actual hardware cache-miss counters for Linearized (array-based) tree:
perf stat -e cache-references,cache-misses,L1-dcache-load-misses,LLC-load-misses \
  ./build/al_bh --N 10000 --steps 10 --linearized 1 --threads 1 --verbose 0

# Measure actual hardware cache-miss counters for Pointer-based tree:
perf stat -e cache-references,cache-misses,L1-dcache-load-misses,LLC-load-misses \
  ./build/al_bh --N 10000 --steps 10 --linearized 0 --threads 1 --verbose 0
```
This directly measures the hardware L1 data cache and LLC (Last Level Cache) miss rates, replacing indirect timing estimates with empirical hardware counter evidence.
