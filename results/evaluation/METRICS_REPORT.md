# AL-BH Evaluation Metrics & Performance Report

Generated: 2026-10-30 21:01:17

## 1. Strong Scaling (Fixed N)

| Threads | Force Time (s) | Total Step Time (s) | Speedup | Efficiency |
|---|---|---|---|---|
| 1 | 3.2526 | 3.3125 | 1.00x | 100.0% |
| 2 | 1.6845 | 1.7507 | 1.93x | 96.5% |
| 4 | 0.8929 | 0.9658 | 3.64x | 91.1% |
| 8 | 0.5008 | 0.5813 | 6.49x | 81.2% |

## 2. Weak Scaling (N proportional to Threads)

| Threads | N (Particles) | Total Execution Time (s) |
|---|---|---|
| 1 | 4000 | 0.4700 |
| 2 | 8000 | 0.6048 |
| 4 | 16000 | 0.7480 |
| 8 | 32000 | 1.0221 |

## 3. Load Imbalance Comparison (Plummer Galaxy Cluster)

- **Static Geometric Slicing**: Mean Imbalance = **16.89x** (Peak: **17.47x**)
- **Adaptive Morton SFC**: Mean Imbalance = **1.00x** (Peak: **1.00x**)

## 4. Cache Locality Benefit (Octree Architecture)

- **Linearized Octree (Array)**: Tree Build = 2.84ms, Force Traversal = 23.70ms
- **Pointer Octree (Heap)**: Tree Build = 5.84ms, Force Traversal = 21.72ms
- **Linearized Tree Build Speedup**: **2.06x faster** due to sequential cache line prefetching.
