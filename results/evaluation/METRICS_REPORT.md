# AL-BH Evaluation Metrics & Performance Report

Generated: 2026-10-31 10:35:24

## 1. Strong Scaling (Fixed N)

| Threads | Force Time (s) | Total Step Time (s) | Speedup | Efficiency |
|---|---|---|---|---|
| 1 | 0.7431 | 0.7549 | 1.00x | 100.0% |
| 2 | 0.3880 | 0.4231 | 1.92x | 95.8% |
| 4 | 0.2201 | 0.2447 | 3.38x | 84.4% |
| 8 | 0.4754 | 0.7003 | 1.56x | 19.5% |

## 2. Weak Scaling (N proportional to Threads)

| Threads | N (Particles) | Total Execution Time (s) |
|---|---|---|
| 1 | 1250 | 0.1289 |
| 2 | 2500 | 0.1750 |
| 4 | 5000 | 0.2358 |
| 8 | 10000 | 1.2880 |

## 3. Load Imbalance Comparison (Plummer Galaxy Cluster)

- **Static Geometric Slicing**: Mean Imbalance = **50.75x** (Peak: **65.25x**)
- **Adaptive Morton SFC**: Mean Imbalance = **1.04x** (Peak: **1.07x**)

## 4. Cache Locality Benefit (Octree Architecture)

- **Linearized Octree (Array)**: Tree Build = 0.63ms, Force Traversal = 4.93ms
- **Pointer Octree (Heap)**: Tree Build = 1.26ms, Force Traversal = 4.64ms
- **Linearized Tree Build Speedup**: **1.99x faster** due to sequential cache line prefetching.
