# AL-BH: Adaptive Linearized Barnes–Hut

## Project context and implementation brief

**Project subtitle:** A Cache-Aware, Dynamic Space-Filling Curve Framework for Distributed N-Body Simulations  
**Project domain:** High-performance computing (HPC), parallel computing, distributed systems, computational physics  
**Primary implementation model:** Hybrid MPI + OpenMP  
**Simulation type:** Three-dimensional gravitational N-body simulation

---

## 1. Project in one paragraph

AL-BH is a distributed 3D gravitational N-body simulator designed to make large simulations faster and more scalable by improving not only the force-calculation algorithm, but also memory locality, workload distribution, and communication between processes. It uses the Barnes–Hut algorithm and an Octree to approximate the gravitational effect of distant groups of particles, reducing the typical computation cost from O(N²) to approximately O(N log N). It then improves the implementation with four complementary optimizations: (1) a **linearized, array-based Octree** to improve cache locality, (2) **Morton/Z-order keys** to arrange particles by spatial proximity and partition them across MPI ranks, (3) **adaptive load balancing** to move partition boundaries as particles move and work becomes uneven, and (4) **non-blocking MPI communication** to exchange remote/ghost particle data while useful local computation continues. OpenMP threads parallelize force computation within each MPI rank. The project evaluates these techniques against simpler Barnes–Hut implementations using execution time, scaling, load imbalance, communication overhead, and cache-miss measurements.

## 2. Problem being solved

An N-body simulation models the gravitational interaction among a large number of particles, such as stars or other celestial bodies. For every simulation timestep, the force on each particle depends on the positions and masses of the other particles.

A direct implementation evaluates every particle against every other particle. This requires approximately **O(N²)** pairwise interactions per timestep. For example, 50,000 particles imply roughly 2.5 billion pairwise evaluations for one timestep. This becomes impractical as the particle count and number of timesteps increase.

The Barnes–Hut algorithm reduces the amount of force computation by grouping distant particles in a hierarchical spatial tree. However, using Barnes–Hut alone does not guarantee good performance on multiple cores or machines:

- **Memory locality:** A conventional pointer-based Octree can allocate nodes separately across the heap. Traversal follows pointers through scattered memory locations, which can cause cache misses and memory stalls.
- **Uneven spatial density:** Equal-sized geometric regions do not necessarily contain equal numbers of particles. A rank assigned a dense cluster can do much more work than a rank assigned mostly empty space.
- **Changing particle positions:** Even a partition that begins balanced can become imbalanced as particles move over successive timesteps.
- **Communication overhead:** Distributed ranks need information about relevant remote particles. Blocking communication can leave ranks waiting instead of computing.
- **Hybrid parallelism:** Large simulations need work distributed across machines/processes and across CPU threads within each process.

**Central project idea:** Combine algorithmic approximation with hardware-aware data layout, spatially coherent distribution, runtime rebalancing, and communication/computation overlap in one distributed pipeline.

## 3. Main objectives

1. Reduce gravitational force computation using Barnes–Hut rather than evaluating every pair directly.
2. Improve Octree traversal locality using a contiguous array-based representation.
3. Use Morton ordering to keep spatially nearby particles close in the data order and to create spatially coherent MPI partitions.
4. Keep the amount of work assigned to MPI ranks reasonably balanced as particles move.
5. Hide some MPI communication latency by using non-blocking communication and overlapping data exchange with local force computation.
6. Use OpenMP for parallel force computation within each MPI rank.
7. Measure each optimization against appropriate baseline configurations rather than reporting only one overall runtime.

## 4. Core concepts and algorithms

### 4.1 Gravitational N-body simulation

Each particle has at least:

- 3D position: `(x, y, z)`
- Scalar mass: `m`
- 3D velocity: `(vx, vy, vz)`

The simulation repeatedly computes gravitational forces/accelerations and updates particle state over a configured number of timesteps. A **softening parameter** is fixed globally for a run to avoid force singularities when two particles are extremely close.

The supplied project documents do not specify a particular time-integration method (for example, leapfrog or velocity-Verlet), exact force equation convention, boundary conditions, or unit system. The implementation must choose and document these consistently; do not present such choices as already specified by the project.

### 4.2 Barnes–Hut and the Octree

Barnes–Hut is the force-approximation algorithm at the centre of the project.

- The simulation space is three-dimensional.
- An Octree recursively subdivides a region into up to eight child regions.
- Each tree node represents a spatial region and stores information needed for traversal and approximation, including its spatial extent and aggregate mass/centre of mass.
- For a target particle, nearby regions are opened and their particles/subregions are examined more closely.
- Distant regions can be approximated as one aggregate mass located at the region's centre of mass.

The **opening criterion** is:

` s / d < θ `

where:

- `s` is the size/width of the Octree region,
- `d` is the distance from the target particle to the region (or its centre of mass, according to the implementation's consistent definition),
- `θ` (theta) is the approximation threshold.

If `s/d < θ`, approximate the region as a single centre-of-mass contribution. Otherwise, subdivide/traverse its children. The theta value controls the accuracy/performance trade-off: a smaller theta generally causes more nodes to be opened and more detailed force evaluation.

The methodology proposes testing `θ = 0.3, 0.5, 0.7`.

### 4.3 Linearized, array-based Octree

A conventional pointer-based tree follows a structure such as:

`Node → Pointer → Node → Pointer → ...`

AL-BH proposes storing tree nodes in a contiguous array, conceptually:

`[Node 0][Node 1][Node 2][Node 3]...`

The purpose is to reduce pointer chasing and improve cache/memory locality during tree construction and traversal. The implementation should define an explicit node layout and child-index representation instead of using separately allocated child pointers as the main representation.

The linearized tree must retain the information Barnes–Hut needs: region bounds/size, aggregate mass, centre of mass, leaf/particle information, and child references or index ranges. The exact array indexing scheme is an implementation decision and should be documented.

### 4.4 Morton (Z-order) keys

Morton ordering converts each particle's 3D spatial coordinates into a one-dimensional key:

1. Normalize/map the particle's `(x, y, z)` coordinates into a bounded integer coordinate grid.
2. Encode the integer coordinates in binary.
3. Interleave the bits from the x, y, and z coordinates.
4. The resulting bit sequence is the particle's Morton key.
5. Sort particles by Morton key.

Particles that are close in space tend to have nearby Morton keys. This creates a one-dimensional ordering that approximately preserves spatial locality.

The sorted sequence is divided into contiguous ranges and assigned to MPI ranks. The methodology specifies balancing by **particle count along the Morton curve**, rather than assigning equal geometric volume to every rank. This is intended to handle non-uniform particle density more effectively than equal-volume decomposition.

### 4.5 Adaptive load balancing

Particle movement changes the work distribution over time. AL-BH monitors work assigned to each MPI rank and can recompute partition boundaries along the Morton ordering.

The methodology defines an imbalance ratio as:

`imbalance = maximum rank workload / minimum rank workload`

Workload can be represented by particle count or measured computation time. Both are to be tracked and compared as workload proxies.

- Check the imbalance against a configurable threshold.
- A threshold of `1.2` is given as an example: the most-loaded rank has 20% more work than the least-loaded rank.
- The threshold is a tunable experimental parameter, not a fixed universal constant.
- When the threshold is exceeded, calculate new partition boundaries using a binary search along the Morton curve to re-equalize the load.
- Redistribute particles according to the new boundaries before the next timestep.

The implementation should record when rebalancing happens, the imbalance before and after it, the particles transferred, and the cost/time spent rebalancing. These records make it possible to evaluate whether rebalancing itself is worthwhile.

### 4.6 Non-blocking MPI and ghost-particle exchange

Each MPI rank owns a spatial partition, but force computation for its local particles may require data from remote regions. The methodology describes a **Locally Essential Tree (LET)** approach for identifying which remote information is needed:

1. Each rank has access to a lightweight global tree skeleton containing remote-node bounding boxes and centres of mass.
2. For local target particles, the rank applies the same Barnes–Hut opening test (`s/d < θ`) to remote nodes.
3. A remote node that cannot safely be represented by its aggregate approximation is flagged; the actual particle data behind that node is requested from its owning rank.
4. The necessary boundary/remote particle data is exchanged as ghost data.
5. Use non-blocking MPI primitives such as `MPI_Isend` and `MPI_Irecv`.
6. Start communication as soon as the relevant data is ready, then perform useful local computation while communication is in progress.
7. Complete outstanding communications before using received data and before the relevant timestep can finish.

The objective is to hide communication latency behind computation and reduce synchronization/waiting time. Do not assume non-blocking calls automatically create overlap: the implementation must schedule independent local work between communication initiation and completion, and the benchmark must compare against a blocking exchange baseline.

### 4.7 Hybrid MPI + OpenMP execution

- **MPI** distributes spatial partitions across processes/ranks, potentially across multiple machines.
- **OpenMP** parallelizes work inside each MPI rank, particularly local Octree traversal and force evaluation.
- The methodology proposes testing multiple MPI-rank counts and OpenMP threads per rank.

The intended structure is therefore distributed-memory parallelism between ranks plus shared-memory parallelism within a rank.

## 5. End-to-end execution pipeline

The following five phases execute for every simulation timestep and repeat for the configured number of timesteps.

### Phase 1 — Simulation setup

- Generate initial particles with position, mass, and velocity.
- Support at least two distributions:
  - **Uniform random:** baseline distribution where equal-volume partitions can perform reasonably.
  - **Clustered / Plummer-model-like:** adversarial density distribution intended to expose workload imbalance.
- Configure the gravitational model and a fixed softening parameter for the run.
- Initialize simulation parameters, random seed, MPI configuration, and OpenMP configuration.

**Output:** Initial particle state and run configuration.

### Phase 2 — Distributed data-structure design

- Convert particle coordinates to Morton keys.
- Sort particles by Morton key.
- Build the linearized array-based Octree over the ordered particles.
- Divide the Morton-ordered particles into contiguous spatially coherent ranges.
- Assign one range/partition to each MPI rank, initially targeting approximately equal particle counts.

**Output:** Morton-ordered particles, local tree(s), and rank ownership/partition metadata.

### Phase 3 — Parallel force computation

- Within each rank, use OpenMP threads to process local target particles.
- Traverse the local linearized Octree.
- Apply the Barnes–Hut opening criterion for each target particle.
- Approximate sufficiently distant nodes using their aggregate mass and centre of mass; descend into nodes that do not satisfy the criterion.
- Compute the force/acceleration contributions needed for the local particles.

**Output:** Force/acceleration results for locally owned particles, using available local and remote information.

### Phase 4 — Inter-process communication

- Use the LET/global tree skeleton to determine which remote nodes need to be opened for local force calculations.
- Request/exchange the corresponding remote or ghost particle data from owning ranks.
- Use non-blocking MPI (`MPI_Isend`/`MPI_Irecv`) and overlap communication with independent local computation where possible.
- Wait for required requests to complete before consuming the received data or completing the timestep.

**Output:** Required remote data is available, with communication and computation overlap where possible.

### Phase 5 — Adaptive load balancing

- Measure each rank's workload (particle count and/or measured compute time).
- Compute and record the max/min imbalance ratio.
- If the configured threshold is exceeded, determine new Morton-curve partition boundaries using binary search/repartitioning.
- Redistribute particles between ranks before the next timestep.
- Continue to the next timestep using the updated distribution.

**Output:** Updated particle ownership and partition boundaries for the next timestep.

**Important implementation detail:** The supplied methodology presents the five phases in this order, with communication and force work interacting through the LET and non-blocking exchange. A practical implementation may pipeline portions of phases 3 and 4 to achieve overlap, but it must preserve the documented dependency that required remote data is received before it is used.

## 6. Data and workload specification

AL-BH does **not** use a labelled machine-learning dataset. It generates synthetic particle data because the goal is to evaluate system and algorithm performance: cache behaviour, load balance, communication overhead, runtime, and scalability.

The same generated particle instances should be used for the proposed implementation and the baseline configurations so that measured differences are attributable to implementation choices rather than different input workloads.

| Parameter | Values/range in the methodology |
|---|---|
| Particle count (`N`) | `10^4` to `10^6+` (scaling sweep) |
| Spatial distributions | Uniform random; clustered (Plummer-like) |
| Per-particle attributes | Position `(x,y,z)`, mass, velocity `(vx,vy,vz)` |
| Softening parameter | Fixed constant per run |
| Simulation timesteps | 100–500 per run |
| Barnes–Hut opening angle (`θ`) | 0.3, 0.5, 0.7 (sweep) |
| MPI ranks | 1, 2, 4, 8, 16 |
| OpenMP threads per rank | 1, 2, 4, 8 |
| Load-imbalance threshold | Tunable; 1.2 is an example, and threshold values should be swept |

Use deterministic/random seeds that are recorded. For repeated runs, use multiple independent seeds and report the mean and standard deviation. The precise softening value, coordinate bounds, mass distribution, random seed policy, and integrator are not fixed by the source documents and must be selected and documented by the implementation.

## 7. System architecture / suggested software modules

The following is a practical code organization for implementing the documented design. It is a suggested decomposition, not a folder structure mandated by the source documents.

```text
AL-BH/
├── README.md
├── CMakeLists.txt                # or another documented build system
├── include/
│   ├── particle.hpp              # particle state and data types
│   ├── simulation_config.hpp     # N, timesteps, theta, softening, etc.
│   ├── octree.hpp                # linearized Octree node/tree interfaces
│   ├── morton.hpp                # coordinate encoding and Morton keys
│   ├── partition.hpp             # Morton ranges and rank ownership
│   ├── force.hpp                 # Barnes-Hut traversal and force calculation
│   ├── communication.hpp         # LET and MPI ghost exchange interfaces
│   ├── load_balancer.hpp         # workload monitoring and repartitioning
│   └── metrics.hpp               # timing and performance counters/logging
├── src/
│   ├── main.cpp
│   ├── particle_generator.cpp
│   ├── morton.cpp
│   ├── octree.cpp
│   ├── force.cpp
│   ├── communication.cpp
│   ├── load_balancer.cpp
│   ├── simulation.cpp
│   └── metrics.cpp
├── baselines/
│   ├── serial_barnes_hut.cpp
│   ├── static_partition_barnes_hut.cpp
│   └── no_rebalance_barnes_hut.cpp
├── tests/
│   ├── test_morton.cpp
│   ├── test_octree.cpp
│   ├── test_force.cpp
│   ├── test_partition.cpp
│   └── test_load_balance.cpp
├── scripts/
│   ├── run_experiments.sh
│   └── analyze_results.py
└── results/
    ├── raw/
    └── plots/
```

A reasonable module responsibility map:

| Module | Responsibility |
|---|---|
| Particle generator | Create uniform and clustered synthetic particle distributions; seed and save run parameters |
| Morton encoder | Quantize coordinates, interleave coordinate bits, generate sortable Morton keys |
| Linearized Octree | Build contiguous node storage and provide traversal/aggregate information |
| Force engine | Apply the Barnes–Hut opening criterion and calculate force/acceleration contributions |
| Domain partitioner | Split sorted Morton particles into contiguous MPI-rank ranges |
| LET / communication | Identify required remote nodes and exchange remote/ghost data with MPI |
| Load balancer | Measure per-rank workload, evaluate threshold, compute new Morton boundaries, redistribute particles |
| Simulation driver | Orchestrate each timestep and particle-state updates |
| Metrics/benchmarking | Record timings, imbalance, communication, cache events, speedup, and efficiency |
| Baseline implementations | Provide controlled comparison configurations using the same particle inputs |

## 8. Required comparison configurations

Evaluate the proposed AL-BH implementation against the following reference configurations from the methodology:

1. **Conventional, non-distributed Barnes–Hut:** A single-process Barnes–Hut implementation. This provides a conventional algorithmic reference.
2. **Distributed Barnes–Hut with static equal-volume partitioning:** Work is distributed using static geometric regions, without the adaptive Morton-curve redistribution proposed by AL-BH.
3. **Distributed execution with spatial decomposition but no adaptive rebalancing:** Use the spatially coherent distribution, but keep the initial partition boundaries fixed so the effect of adaptive rebalancing can be isolated.

In addition to the three overall reference configurations, perform targeted comparisons to isolate the individual optimizations:

- Linearized array-based Octree versus pointer-based Octree for cache-miss behaviour.
- Non-blocking ghost exchange versus blocking ghost exchange for communication/synchronization overhead.
- Adaptive rebalancing enabled versus disabled for per-rank imbalance over timesteps.

The exact feature combinations used inside each reference implementation should be stated in the experiment configuration so comparisons are fair and interpretable.

## 9. Experimental procedure

For each combination of particle count, spatial distribution, MPI-rank count, OpenMP thread count, and relevant algorithm parameters:

1. Generate or load a particle instance with a recorded seed.
2. Run every comparison configuration on the same initial particle data and run settings.
3. Execute a fixed number of timesteps (the methodology specifies 100–500 per run).
4. Repeat runs with multiple independent initialization seeds to account for run-to-run variation.
5. Record the mean and standard deviation of measured values.
6. Run **strong scaling** experiments: hold total particle count fixed while increasing the number of MPI ranks (and report the OpenMP configuration).
7. Run **weak scaling** experiments: increase particle count along with rank count so that the amount of work per rank stays approximately constant.
8. Preserve raw measurements and configuration metadata so results can be reproduced.

## 10. Evaluation metrics

| Metric | What it measures | Relevant comparison |
|---|---|---|
| Total execution time | End-to-end runtime for a simulation run | AL-BH versus the reference configurations |
| Per-timestep execution time | Runtime variation and cost per timestep | Across timesteps and configurations |
| Parallel speedup | Serial/reference runtime divided by parallel runtime, using a clearly stated baseline | Increasing MPI ranks and OpenMP threads |
| Parallel efficiency | Speedup relative to ideal linear scaling for the resources used | Strong- and weak-scaling experiments |
| Load imbalance | Maximum rank workload divided by minimum rank workload | Adaptive rebalancing enabled versus disabled; record every timestep |
| Communication/synchronization overhead | Time spent communicating or waiting for communication | Blocking versus non-blocking ghost exchange |
| Cache-miss rate | Cache behaviour during tree construction/traversal | Linearized array-based Octree versus pointer-based Octree |
| Performance across timesteps | Whether performance remains stable as the particle distribution evolves | Static partitioning versus adaptive rebalancing |

Where possible, use hardware performance counters to measure cache misses. Record the exact counter/tool and hardware environment. The proposal specifically identifies cache-miss measurement via hardware performance counters.

For load balance, record both particle count and measured compute time per rank. The methodology emphasizes recording imbalance every timestep, not only when rebalancing occurs, so the trigger and its effect can be seen directly.

## 11. Expected outcomes and contributions

The project expects to demonstrate the following, and these are hypotheses to be tested rather than results to assume:

1. **Better cache locality:** The linearized Octree should reduce cache misses compared with a pointer-based Octree.
2. **Better load balance over time:** Adaptive Morton-curve repartitioning should keep the max/min rank-work ratio lower and more stable than fixed partitioning as particles move.
3. **Reduced communication waiting:** Non-blocking MPI should reduce idle/waiting time when communication can overlap with useful local computation.
4. **Improved scalability:** The integrated hybrid approach should improve speedup/efficiency as MPI ranks and OpenMP threads increase, relative to the chosen baselines.
5. **A cohesive HPC implementation:** The deliverable should demonstrate the four optimizations operating together in one five-phase distributed N-body simulation pipeline.

An optimization may provide limited benefit under some conditions. For example, communication overlap may show little advantage at low rank counts or when there is not enough independent computation to hide latency. Such cases should be measured and discussed as operating limits, not hidden.

## 12. What makes this project distinctive

The project is not proposing a new gravitational law or replacing Barnes–Hut. The contribution is the **integration and systematic evaluation** of four optimizations in a distributed Barnes–Hut pipeline:

- Linearized Octree for cache-aware tree storage/traversal.
- Morton-curve decomposition for spatially coherent rank ownership.
- Runtime-adaptive load balancing for changing particle distributions.
- Non-blocking MPI communication to overlap remote-data exchange with computation.

MPI + OpenMP provides the hybrid execution model that ties the distributed and within-rank computation together.

## 13. Implementation priorities

Build and validate the project incrementally. Do not begin with the full distributed pipeline before the simulation and tree data structures are correct.

### Milestone 1 — Correct simulation baseline

- Define particle structures, coordinate bounds, mass generation, softening, force convention, and integration method.
- Generate deterministic uniform and clustered initial distributions.
- Implement a basic serial Barnes–Hut simulator.
- Validate the simulation on small particle counts and short runs.

### Milestone 2 — Linearized Octree and Morton ordering

- Implement coordinate quantization and Morton key generation.
- Test that Morton keys are deterministic and sortable.
- Implement the linearized Octree and verify node aggregates and traversal.
- Compare its force results against a trusted small-N reference and/or a pointer-based Barnes–Hut implementation.

### Milestone 3 — MPI spatial decomposition

- Sort particles by Morton key and distribute contiguous ranges across ranks.
- Verify that every particle has exactly one owning rank.
- Implement distributed force evaluation and the required remote data model.
- Compare results and runtime against the serial baseline.

### Milestone 4 — OpenMP parallelism

- Parallelize local target-particle force evaluation.
- Check for race conditions and ensure each particle's force/acceleration is written safely.
- Measure scaling across thread counts.

### Milestone 5 — LET and non-blocking exchange

- Build the remote tree skeleton/LET selection logic.
- Implement ghost-data request and exchange with `MPI_Isend`/`MPI_Irecv`.
- Ensure requests complete before received data is consumed.
- Create a blocking-exchange version for controlled comparison.

### Milestone 6 — Adaptive rebalancing

- Measure per-rank workload every timestep.
- Implement threshold checks and Morton-boundary recomputation.
- Redistribute particles safely between ranks.
- Compare enabled/disabled configurations on uniform and clustered distributions.

### Milestone 7 — Experiments and reporting

- Automate the parameter sweeps.
- Capture runtime, per-timestep measurements, load imbalance, communication time, cache misses, and resource settings.
- Run repeated seeds and strong/weak-scaling tests.
- Generate plots/tables and describe both successful and unsuccessful optimization cases.

## 14. Correctness and engineering requirements

- Every particle must have exactly one owning MPI rank after each partition/rebalance operation.
- No particle should be lost or duplicated during redistribution.
- Morton sorting and partitioning must be deterministic for a fixed input and configuration.
- Octree aggregate mass and centre of mass must be consistent with the particles represented by each node.
- The opening criterion must be applied consistently for local tree traversal and remote LET decisions.
- A target particle must not incorrectly include itself in a force contribution; handle self-interaction according to the chosen force implementation.
- MPI requests must be completed before their buffers are reused or freed and before received data is read.
- OpenMP parallel loops must avoid data races and unsafe concurrent writes.
- All comparison configurations must use identical initial particle data, parameter values, timestep counts, and comparable hardware/resource allocations.
- Benchmark output must include enough configuration metadata to reproduce each run.
- Report correctness and performance separately. A faster run is not valid if it produces incorrect particle state or omits required interactions.

## 15. Important design choices not fully specified by the supplied documents

The project presentation, introductory paper, and methodology define the overall architecture and experiment plan, but they do not fully prescribe every low-level implementation detail. The implementation agent must explicitly decide, document, and keep these choices consistent:

1. **Time integrator:** The exact numerical integration scheme is not specified.
2. **Force equation and units:** The gravitational constant, unit system, and exact softening formula are not specified.
3. **Boundary conditions:** Periodic versus isolated/open boundaries are not specified.
4. **Coordinate quantization:** The coordinate bounds, number of bits per dimension, and handling of out-of-range positions for Morton keys are not specified.
5. **Linearized Octree indexing:** The exact node array layout, child indexing method, and leaf representation are not specified.
6. **LET representation:** The methodology specifies a lightweight global skeleton and selective remote-node requests, but not a wire format or exact request protocol.
7. **Rebalancing details:** The threshold is tunable and 1.2 is only an example; the precise timing/cost model and migration protocol must be selected.
8. **Hardware measurement tooling:** The specific performance-counter tool and cache events are not specified.
9. **Output format:** The source documents do not mandate a result-file format or visualization. A reproducible CSV/JSON log and plots are sensible implementation choices.

Do not silently invent these as facts from the proposal. Choose simple, defensible defaults, expose important values as configuration parameters, and document them in the README and experiment output.

## 16. Instructions for an AI coding agent

Use this document as the project source of truth. Your task is to help implement **AL-BH: Adaptive Linearized Barnes–Hut**, a hybrid MPI + OpenMP distributed 3D gravitational N-body simulator.

Follow these rules:

1. Preserve the architecture and objectives described here. Do not turn this into an ML project, a visualization-only project, or a generic particle simulator.
2. Implement the four key optimizations: linearized array-based Octree, Morton-order spatial decomposition, adaptive load balancing, and non-blocking MPI ghost/remote-data exchange.
3. Use Barnes–Hut with the opening criterion `s/d < θ`; support configurable theta.
4. Use synthetic uniform and clustered/Plummer-like particle distributions. Do not require an external labelled dataset.
5. Use MPI across ranks and OpenMP within ranks.
6. Keep the three baseline configurations and the targeted micro-comparisons available so the contribution of each optimization can be measured.
7. Make the project buildable and runnable from documented commands. Include configuration options for particle count, timesteps, distribution, theta, softening, MPI ranks, OpenMP threads, and rebalancing threshold.
8. Add correctness tests for Morton encoding, Octree aggregates/traversal, force computation, partition ownership, ghost exchange, and particle redistribution.
9. Log raw metrics and run configuration. Do not fabricate performance improvements or hard-code expected results into output.
10. Implement incrementally. At each milestone, provide the files changed, build/run commands, tests run, observed output, and any limitations.
11. Where the source documents leave a design choice open, select a reasonable implementation, state the choice, and make it configurable when practical. Do not claim that the original project specification mandated it.
12. Do not omit MPI correctness, buffer/request lifecycle, race safety, or reproducibility in pursuit of speed.

**First task for the coding agent:** Inspect the target environment and propose a concrete implementation plan, build system, module structure, and milestone sequence. Then implement the smallest correct serial Barnes–Hut baseline before adding Morton ordering, the linearized tree, MPI, OpenMP, communication overlap, and adaptive rebalancing.

---

