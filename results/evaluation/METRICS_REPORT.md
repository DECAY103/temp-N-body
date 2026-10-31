# AL-BH Evaluation Metrics & Performance Report

Generated: 2026-10-30 23:18:24

## 1. Strong Scaling (Fixed N)

| Threads | Force Time (s) | Total Step Time (s) | Speedup | Efficiency |
|---|---|---|---|---|
| 1 | 0.6443 | 0.6587 | 1.00x | 100.0% |
| 2 | 0.3655 | 0.3807 | 1.76x | 88.1% |
| 4 | 0.1839 | 0.2004 | 3.50x | 87.6% |
| 8 | 0.1028 | 0.1337 | 6.27x | 78.3% |

## 2. Weak Scaling (N proportional to Threads)

| Threads | N (Particles) | Total Execution Time (s) |
|---|---|---|
| 1 | 1250 | 0.1022 |
| 2 | 2500 | 0.1410 |
| 4 | 5000 | 0.1994 |
| 8 | 10000 | 0.3551 |

## 3. Load Imbalance Comparison (Plummer Galaxy Cluster)

- **Static Geometric Slicing**: Mean Imbalance = **15.49x** (Peak: **16.94x**)
- **Adaptive Morton SFC**: Mean Imbalance = **1.06x** (Peak: **1.15x**)

## 4. Cache Locality Benefit (Octree Architecture)

- **Linearized Octree (Array)**: Tree Build = 0.70ms, Force Traversal = 4.45ms
- **Pointer Octree (Heap)**: Tree Build = 1.42ms, Force Traversal = 4.27ms
- **Linearized Tree Build Speedup**: **2.04x faster** due to sequential cache line prefetching.
