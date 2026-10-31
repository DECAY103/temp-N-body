#include "force.hpp"
#include "particle.hpp"
#include "octree.hpp"
#include "octree_pointer.hpp"
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace albh {

// ============================================================================
// Pairwise gravitational acceleration contribution
//
// Softened gravity: a_vec = -G * M * r_vec / (|r|^2 + eps^2)^(3/2)
// where r_vec = (px-mx, py-my, pz-mz) pointing from mass M to particle p
//
// The softening ε prevents the force from diverging when particles are close.
// ============================================================================
void pairwise_accel(double px, double py, double pz,
                    double mx, double my, double mz, double M,
                    double G, double softening,
                    double& ax, double& ay, double& az) {
    double dx = px - mx;
    double dy = py - my;
    double dz = pz - mz;
    double r2 = dx * dx + dy * dy + dz * dz + softening * softening;
    double inv_r3 = 1.0 / (r2 * std::sqrt(r2));  // = 1 / (r^2 + eps^2)^(3/2)
    double factor = -G * M * inv_r3;
    ax += factor * dx;
    ay += factor * dy;
    az += factor * dz;
}

// ============================================================================
// Linearized tree walk: recursive Barnes-Hut traversal
//
// Opening criterion: node.size / distance < theta
//   → If satisfied: use node's center-of-mass as aggregate approximation
//   → If not: descend into children
//
// self_idx prevents a particle from computing force on itself
// ============================================================================
void tree_walk_linearized(int node_idx, const Octree& tree,
                           double px, double py, double pz, int target_id,
                           double G, double theta, double softening,
                           double& ax, double& ay, double& az) {
    const OctreeNode& node = tree.nodes[node_idx];

    // Skip empty nodes
    if (node.particle_count == 0) return;

    // Leaf node
    if (node.first_child == -1 && node.particle_index >= 0) {
        if (node.particle_id == target_id) {
            // Target particle is in this leaf. If there are coincident particles at MAX_DEPTH,
            // compute force from the other particles in this leaf
            if (node.particle_count > 1 && node.total_mass > 0.0) {
                double other_mass = node.total_mass * (static_cast<double>(node.particle_count - 1) / node.particle_count);
                pairwise_accel(px, py, pz,
                               node.com_x, node.com_y, node.com_z, other_mass,
                               G, softening, ax, ay, az);
            }
            return;
        }

        pairwise_accel(px, py, pz,
                       node.com_x, node.com_y, node.com_z, node.total_mass,
                       G, softening, ax, ay, az);
        return;
    }

    // Internal node: check opening criterion
    double dx = px - node.com_x;
    double dy = py - node.com_y;
    double dz = pz - node.com_z;
    double d2 = dx * dx + dy * dy + dz * dz;
    double d = std::sqrt(d2);

    // FIX: Never approximate an internal node that spatially contains the target particle!
    // Approximating a node containing the target would include the target in its own aggregate force.
    bool contains_target = (px >= node.origin_x && px <= (node.origin_x + node.size) &&
                            py >= node.origin_y && py <= (node.origin_y + node.size) &&
                            pz >= node.origin_z && pz <= (node.origin_z + node.size));

    // Opening criterion: s/d < theta AND node does not contain target particle
    if (!contains_target && d > 0.0 && (node.size / d) < theta) {
        pairwise_accel(px, py, pz,
                       node.com_x, node.com_y, node.com_z, node.total_mass,
                       G, softening, ax, ay, az);
        return;
    }

    // Open the node: recurse into children
    if (node.first_child >= 0) {
        for (int c = 0; c < 8; ++c) {
            tree_walk_linearized(node.first_child + c, tree,
                                  px, py, pz, target_id,
                                  G, theta, softening, ax, ay, az);
        }
    }
}

// ============================================================================
// Pointer-based tree walk (baseline — same logic, different data structure)
// ============================================================================
void tree_walk_pointer(const PointerOctreeNode* node,
                        double px, double py, double pz, int target_id,
                        double G, double theta, double softening,
                        double& ax, double& ay, double& az) {
    if (!node || node->particle_count == 0) return;

    // Leaf
    if (node->particle_index >= 0 && !node->children[0]) {
        if (node->particle_id == target_id) {
            if (node->particle_count > 1 && node->total_mass > 0.0) {
                double other_mass = node->total_mass * (static_cast<double>(node->particle_count - 1) / node->particle_count);
                pairwise_accel(px, py, pz,
                               node->com_x, node->com_y, node->com_z, other_mass,
                               G, softening, ax, ay, az);
            }
            return;
        }

        pairwise_accel(px, py, pz,
                       node->com_x, node->com_y, node->com_z, node->total_mass,
                       G, softening, ax, ay, az);
        return;
    }

    // Internal node: check opening criterion
    double dx = px - node->com_x;
    double dy = py - node->com_y;
    double dz = pz - node->com_z;
    double d = std::sqrt(dx * dx + dy * dy + dz * dz);

    bool contains_target = (px >= node->origin_x && px <= (node->origin_x + node->size) &&
                            py >= node->origin_y && py <= (node->origin_y + node->size) &&
                            pz >= node->origin_z && pz <= (node->origin_z + node->size));

    if (!contains_target && d > 0.0 && (node->size / d) < theta) {
        pairwise_accel(px, py, pz,
                       node->com_x, node->com_y, node->com_z, node->total_mass,
                       G, softening, ax, ay, az);
        return;
    }

    for (int c = 0; c < 8; ++c) {
        if (node->children[c]) {
            tree_walk_pointer(node->children[c].get(),
                               px, py, pz, target_id,
                               G, theta, softening, ax, ay, az);
        }
    }
}

// ============================================================================
// Compute forces for all particles using linearized Octree
// OpenMP parallelization: each thread processes a subset of particles.
// No data races because each particle writes only to its own ax/ay/az.
// ============================================================================
void compute_forces_linearized(std::vector<Particle>& particles,
                                const Octree& tree,
                                double G, double theta, double softening,
                                int num_threads) {
    int N = static_cast<int>(particles.size());

    #ifdef _OPENMP
    omp_set_num_threads(num_threads);
    #endif

    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < N; ++i) {
        particles[i].ax = 0.0;
        particles[i].ay = 0.0;
        particles[i].az = 0.0;

        tree_walk_linearized(0, tree,
                              particles[i].x, particles[i].y, particles[i].z, particles[i].id,
                              G, theta, softening,
                              particles[i].ax, particles[i].ay, particles[i].az);
    }
}

// Compute forces using pointer-based Octree (baseline)
void compute_forces_pointer(std::vector<Particle>& particles,
                             const PointerOctree& tree,
                             double G, double theta, double softening,
                             int num_threads) {
    int N = static_cast<int>(particles.size());

    #ifdef _OPENMP
    omp_set_num_threads(num_threads);
    #endif

    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < N; ++i) {
        particles[i].ax = 0.0;
        particles[i].ay = 0.0;
        particles[i].az = 0.0;

        tree_walk_pointer(tree.root_node.get(),
                           particles[i].x, particles[i].y, particles[i].z, particles[i].id,
                           G, theta, softening,
                           particles[i].ax, particles[i].ay, particles[i].az);
    }
}

// ============================================================================
// Direct O(N²) force computation — for validation on small N only
// Computes exact forces by summing all pairwise interactions.
// ============================================================================
void compute_forces_direct(std::vector<Particle>& particles,
                            double G, double softening) {
    int N = static_cast<int>(particles.size());

    // Reset accelerations
    for (int i = 0; i < N; ++i) {
        particles[i].ax = 0.0;
        particles[i].ay = 0.0;
        particles[i].az = 0.0;
    }

    // All pairs
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            double dx = particles[i].x - particles[j].x;
            double dy = particles[i].y - particles[j].y;
            double dz = particles[i].z - particles[j].z;
            double r2 = dx * dx + dy * dy + dz * dz + softening * softening;
            double inv_r3 = 1.0 / (r2 * std::sqrt(r2));

            double fx = -G * dx * inv_r3;
            double fy = -G * dy * inv_r3;
            double fz = -G * dz * inv_r3;

            // Newton's third law: equal and opposite
            particles[i].ax += fx * particles[j].mass;
            particles[i].ay += fy * particles[j].mass;
            particles[i].az += fz * particles[j].mass;
            particles[j].ax -= fx * particles[i].mass;
            particles[j].ay -= fy * particles[i].mass;
            particles[j].az -= fz * particles[i].mass;
        }
    }
}

} // namespace albh
