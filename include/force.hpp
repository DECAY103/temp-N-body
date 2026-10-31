#ifndef ALBH_FORCE_HPP
#define ALBH_FORCE_HPP

#include <vector>

namespace albh {

struct Particle;
class Octree;
class PointerOctree;
struct PointerOctreeNode;
struct OctreeNode;

// ============================================================================
// Barnes-Hut Force Computation
//
// Computes gravitational acceleration for particles using tree traversal.
// Opening criterion: s / d < theta
//   s = node side length
//   d = distance from particle to node's center of mass
//   theta = configurable accuracy parameter (0.3 = accurate, 0.7 = fast)
//
// Softened gravity: a = -G * M * r_vec / (|r|^2 + eps^2)^(3/2)
// ============================================================================

// Compute accelerations for ALL particles using the linearized Octree
// Resets ax, ay, az before computation.
// OpenMP-parallelized over particles.
void compute_forces_linearized(std::vector<Particle>& particles,
                                const Octree& tree,
                                double G, double theta, double softening,
                                int num_threads = 1);

// Compute accelerations using the pointer-based Octree (baseline)
void compute_forces_pointer(std::vector<Particle>& particles,
                             const PointerOctree& tree,
                             double G, double theta, double softening,
                             int num_threads = 1);

// Direct O(N^2) force computation for validation on small N
void compute_forces_direct(std::vector<Particle>& particles,
                            double G, double softening);

// --- Internal helpers (exposed for testing) ---

// Compute acceleration on particle p from a single mass M at position (mx,my,mz)
// Returns (ax, ay, az) contribution
void pairwise_accel(double px, double py, double pz,
                    double mx, double my, double mz, double M,
                    double G, double softening,
                    double& ax, double& ay, double& az);

// Recursive tree walk for linearized tree (target_id is unique particle ID for self-exclusion)
void tree_walk_linearized(int node_idx, const Octree& tree,
                           double px, double py, double pz, int target_id,
                           double G, double theta, double softening,
                           double& ax, double& ay, double& az);

// Recursive tree walk for pointer-based tree
void tree_walk_pointer(const PointerOctreeNode* node,
                        double px, double py, double pz, int target_id,
                        double G, double theta, double softening,
                        double& ax, double& ay, double& az);

} // namespace albh

#endif // ALBH_FORCE_HPP
