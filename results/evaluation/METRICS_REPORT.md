# AL-BH Evaluation Metrics & Performance Report

Generated: 2026-10-31 17:14:14

## 1. Strong Scaling (Fixed N)

| Threads | Force Time (s) | Total Step Time (s) | Speedup | Efficiency |
|---|---|---|---|---|
| 1 | 0.7642 | 0.7772 | 1.00x | 100.0% |
| 2 | 0.4218 | 0.4508 | 1.72x | 86.2% |
| 4 | 0.2212 | 0.2380 | 3.27x | 81.7% |
| 8 | 0.1118 | 0.1606 | 4.84x | 60.5% |

## 2. Weak Scaling (N proportional to Threads)

| Threads | N (Particles) | Total Execution Time (s) |
|---|---|---|
| 1 | 1250 | 0.1271 |
| 2 | 2500 | 0.1773 |
| 4 | 5000 | 0.2281 |
| 8 | 10000 | 0.3291 |

## 3. Load Imbalance Comparison (Plummer Galaxy Cluster)

- **Static Geometric Slicing**: Mean Imbalance = **50.21x** (Peak: **56.30x**)
- **Adaptive Morton SFC**: Mean Imbalance = **1.05x** (Peak: **1.12x**)

## 4. Cache Locality Benefit (Octree Architecture)

- **Linearized Octree (Array)**: Tree Build = 0.65ms, Force Traversal = 4.92ms
- **Pointer Octree (Heap)**: Tree Build = 1.27ms, Force Traversal = 4.68ms
- **Linearized Tree Build Speedup**: **1.96x faster** due to sequential cache line prefetching.
