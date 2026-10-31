#ifndef ALBH_OCTREE_HPP
#define ALBH_OCTREE_HPP

#include <vector>
#include <cstdint>
#include <unordered_map>

namespace albh {

struct Particle;

// ============================================================================
// Linearized Array-Based Octree
//
// All nodes stored in a contiguous std::vector<OctreeNode>.
// Children of node i are at indices [first_child, first_child+7].
// This layout eliminates pointer chasing and improves cache locality
// during tree construction and traversal.
//
// Node types:
//   - Internal: first_child >= 0, particle_index == -1
//   - Leaf (single particle): first_child == -1, particle_index >= 0
//   - Empty: first_child == -1, particle_index == -1, particle_count == 0
// ============================================================================

struct OctreeNode {
    // --- Center of mass (aggregate) ---
    double com_x, com_y, com_z;     // center of mass position
    double total_mass;               // total mass of subtree

    // --- Spatial bounds ---
    double origin_x, origin_y, origin_z; // lower-left corner of this node's region
    double size;                          // side length of this cubic region

    // --- Tree structure ---
    int first_child;      // index into nodes[] of first child (-1 if leaf/empty)
    int particle_index;   // index into particles[] if leaf (-1 if internal/empty)
    int particle_id;      // unique global ID of particle if leaf (-1 if internal/empty)
    int particle_count;   // total particles in this subtree

    OctreeNode()
        : com_x(0), com_y(0), com_z(0), total_mass(0),
          origin_x(0), origin_y(0), origin_z(0), size(0),
          first_child(-1), particle_index(-1), particle_id(-1), particle_count(0) {}
};

class Octree {
public:
    // All nodes in contiguous storage
    std::vector<OctreeNode> nodes;

    // Extra particles when MAX_DEPTH is reached for coincident/near-coincident particles
    std::unordered_map<int, std::vector<int>> overflow_particles;

    // Maximum depth to prevent infinite recursion (particles at same position)
    static constexpr int MAX_DEPTH = 21;

    // Maximum particles per leaf before subdivision
    static constexpr int MAX_LEAF_PARTICLES = 1;

    // Build octree from particles within the given bounding box
    // box_origin: lower corner, box_size: side length of root cube
    void build(const std::vector<Particle>& particles,
               double box_origin_x, double box_origin_y, double box_origin_z,
               double box_size);

    // Clear all nodes
    void clear();

    // Get root node (index 0)
    const OctreeNode& root() const { return nodes[0]; }

    // Number of nodes in tree
    size_t num_nodes() const { return nodes.size(); }

private:
    // Recursive helper: insert particle into the subtree rooted at node_idx
    void insert(int node_idx, int particle_idx,
                const std::vector<Particle>& particles, int depth);

    // Determine which octant (0-7) a point falls in relative to node center
    static int get_octant(double px, double py, double pz,
                          double cx, double cy, double cz);

    // Allocate 8 children for node at node_idx, return index of first child
    int allocate_children(int node_idx);

    // Recompute center-of-mass and total_mass from leaves up to root
    void compute_aggregates(int node_idx, const std::vector<Particle>& particles);
};

} // namespace albh

#endif // ALBH_OCTREE_HPP
