# AL-BH Evaluation Metrics & Performance Report

Generated: 2026-10-30 22:58:08

## 1. Strong Scaling (Fixed N)

| Threads | Force Time (s) | Total Step Time (s) | Speedup | Efficiency |
|---|---|---|---|---|
| 1 | 0.1052 | 0.1084 | 1.00x | 100.0% |
| 2 | 0.0569 | 0.0618 | 1.85x | 92.5% |
| 4 | 0.0314 | 0.0354 | 3.35x | 83.8% |
| 8 | 0.0199 | 0.0377 | 5.28x | 66.0% |

## 2. Weak Scaling (N proportional to Threads)

| Threads | N (Particles) | Total Execution Time (s) |
|---|---|---|
| 1 | 500 | 0.0195 |
| 2 | 1000 | 0.0455 |
| 4 | 2000 | 0.0589 |
| 8 | 4000 | 0.1143 |

## 3. Load Imbalance Comparison (Plummer Galaxy Cluster)

- **Static Geometric Slicing**: Mean Imbalance = **16.75x** (Peak: **18.84x**)
- **Adaptive Morton SFC**: Mean Imbalance = **1.10x** (Peak: **1.61x**)

## 4. Cache Locality Benefit (Octree Architecture)

- **Linearized Octree (Array)**: Tree Build = 0.25ms, Force Traversal = 2.15ms
- **Pointer Octree (Heap)**: Tree Build = 0.49ms, Force Traversal = 1.42ms
- **Linearized Tree Build Speedup**: **1.93x faster** due to sequential cache line prefetching.
