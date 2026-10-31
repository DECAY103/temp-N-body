#ifndef ALBH_OCTREE_POINTER_HPP
#define ALBH_OCTREE_POINTER_HPP

#include <vector>
#include <memory>

namespace albh {

struct Particle;

// ============================================================================
// Pointer-Based Octree (Baseline for Cache Comparison)
//
// Traditional heap-allocated tree where each node has 8 child pointers.
// This deliberately creates scattered memory access patterns to serve
// as a baseline for measuring cache-miss improvements of the linearized tree.
// ============================================================================

struct PointerOctreeNode {
    // --- Center of mass (aggregate) ---
    double com_x, com_y, com_z;
    double total_mass;

    // --- Spatial bounds ---
    double origin_x, origin_y, origin_z;
    double size;

    // --- Tree structure (heap-allocated children) ---
    std::unique_ptr<PointerOctreeNode> children[8];
    int particle_index;   // index into particles[] if leaf (-1 if internal)
    int particle_id;      // unique global ID of particle if leaf (-1 if internal)
    int particle_count;
    std::vector<int> overflow_indices; // coincident particles at MAX_DEPTH

    PointerOctreeNode()
        : com_x(0), com_y(0), com_z(0), total_mass(0),
          origin_x(0), origin_y(0), origin_z(0), size(0),
          particle_index(-1), particle_id(-1), particle_count(0) {}
};

class PointerOctree {
public:
    std::unique_ptr<PointerOctreeNode> root_node;
    static constexpr int MAX_DEPTH = 21;

    // Build the pointer-based tree from particles
    void build(const std::vector<Particle>& particles,
               double box_origin_x, double box_origin_y, double box_origin_z,
               double box_size);

    void clear();

private:
    void insert(PointerOctreeNode* node, int particle_idx,
                const std::vector<Particle>& particles, int depth);

    static int get_octant(double px, double py, double pz,
                          double cx, double cy, double cz);

    void compute_aggregates(PointerOctreeNode* node,
                            const std::vector<Particle>& particles);
};

} // namespace albh

#endif // ALBH_OCTREE_POINTER_HPP
