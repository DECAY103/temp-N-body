#include "octree.hpp"
#include "particle.hpp"
#include <stdexcept>

namespace albh {

// ============================================================================
// Build the linearized Octree from a set of particles.
//
// Strategy:
//   1. Create root node spanning the entire bounding box
//   2. Insert particles one by one into the tree
//   3. When a leaf receives a second particle, subdivide into 8 children
//   4. After all insertions, compute center-of-mass aggregates bottom-up
//
// All nodes live in a contiguous std::vector for cache friendliness.
// ============================================================================

void Octree::build(const std::vector<Particle>& particles,
                    double box_origin_x, double box_origin_y, double box_origin_z,
                    double box_size) {
    clear();

    if (particles.empty()) return;

    // Reserve generous estimate to minimize reallocation during insertion.
    // Typical octree has ~2-4x nodes relative to particle count.
    nodes.reserve(std::max(static_cast<size_t>(particles.size() * 4), static_cast<size_t>(1024)));

    // Create root node covering entire simulation domain
    OctreeNode root;
    root.origin_x = box_origin_x;
    root.origin_y = box_origin_y;
    root.origin_z = box_origin_z;
    root.size = box_size;
    root.first_child = -1;
    root.particle_index = -1;
    root.particle_count = 0;
    nodes.push_back(root);

    // Insert each particle into the tree
    for (int i = 0; i < static_cast<int>(particles.size()); ++i) {
        insert(0, i, particles, 0);
    }

    // Compute center-of-mass aggregates from leaves to root
    compute_aggregates(0, particles);
}

void Octree::clear() {
    nodes.clear();
}

// ============================================================================
// Determine which of 8 octants a point falls in.
// Octant index = 4*(z>=cz) + 2*(y>=cy) + 1*(x>=cx)
// This gives a consistent 0-7 mapping.
// ============================================================================
int Octree::get_octant(double px, double py, double pz,
                        double cx, double cy, double cz) {
    int octant = 0;
    if (px >= cx) octant |= 1;
    if (py >= cy) octant |= 2;
    if (pz >= cz) octant |= 4;
    return octant;
}

// ============================================================================
// Allocate 8 child nodes for the given parent node.
// Children are appended contiguously to the nodes vector.
// Returns the index of the first child.
//
// WARNING: This may cause vector reallocation! All references and pointers
// into nodes[] become invalid after this call. Use indices only.
// ============================================================================
int Octree::allocate_children(int node_idx) {
    int first_child = static_cast<int>(nodes.size());

    double parent_size = nodes[node_idx].size;
    double child_size = parent_size / 2.0;
    double ox = nodes[node_idx].origin_x;
    double oy = nodes[node_idx].origin_y;
    double oz = nodes[node_idx].origin_z;

    // Create 8 children with appropriate spatial bounds
    for (int i = 0; i < 8; ++i) {
        OctreeNode child;
        child.size = child_size;
        child.origin_x = ox + ((i & 1) ? child_size : 0.0);
        child.origin_y = oy + ((i & 2) ? child_size : 0.0);
        child.origin_z = oz + ((i & 4) ? child_size : 0.0);
        child.first_child = -1;
        child.particle_index = -1;
        child.particle_count = 0;
        nodes.push_back(child);
    }

    // Update parent's first_child AFTER push_back
    nodes[node_idx].first_child = first_child;

    return first_child;
}

// ============================================================================
// Insert a particle into the subtree rooted at node_idx.
//
// Cases:
//   1. Empty node -> place particle here as leaf
//   2. Leaf node (has one particle) -> subdivide, re-insert existing + new
//   3. Internal node -> recurse into appropriate child octant
//
// CRITICAL: Never hold OctreeNode& references across allocate_children()
// because vector reallocation invalidates all references. Always use
// nodes[node_idx].field directly.
// ============================================================================
void Octree::insert(int node_idx, int particle_idx,
                     const std::vector<Particle>& particles, int depth) {
    // Safety: prevent infinite recursion from coincident particles
    if (depth >= MAX_DEPTH) {
        nodes[node_idx].particle_count++;
        return;
    }

    if (nodes[node_idx].particle_count == 0 && nodes[node_idx].first_child == -1) {
        // Case 1: Empty leaf — place particle here
        nodes[node_idx].particle_index = particle_idx;
        nodes[node_idx].particle_count = 1;
        return;
    }

    if (nodes[node_idx].particle_count == 1 && nodes[node_idx].first_child == -1) {
        // Case 2: Occupied leaf — need to subdivide
        int existing_particle = nodes[node_idx].particle_index;
        nodes[node_idx].particle_index = -1;  // no longer a leaf

        // Save spatial info BEFORE allocate_children (which may realloc the vector)
        double origin_x = nodes[node_idx].origin_x;
        double origin_y = nodes[node_idx].origin_y;
        double origin_z = nodes[node_idx].origin_z;
        double size = nodes[node_idx].size;

        // Create 8 children — WARNING: may reallocate nodes[]
        allocate_children(node_idx);

        // Compute center using saved values (references are now invalid)
        double cx = origin_x + size / 2.0;
        double cy = origin_y + size / 2.0;
        double cz = origin_z + size / 2.0;

        int oct_existing = get_octant(
            particles[existing_particle].x,
            particles[existing_particle].y,
            particles[existing_particle].z,
            cx, cy, cz);

        // Reset count before re-inserting (will be recomputed)
        nodes[node_idx].particle_count = 0;
        insert(nodes[node_idx].first_child + oct_existing,
               existing_particle, particles, depth + 1);
        nodes[node_idx].particle_count++;

        // Insert new particle
        int oct_new = get_octant(
            particles[particle_idx].x,
            particles[particle_idx].y,
            particles[particle_idx].z,
            cx, cy, cz);
        insert(nodes[node_idx].first_child + oct_new,
               particle_idx, particles, depth + 1);
        nodes[node_idx].particle_count++;
        return;
    }

    // Case 3: Internal node — recurse into correct child
    double cx = nodes[node_idx].origin_x + nodes[node_idx].size / 2.0;
    double cy = nodes[node_idx].origin_y + nodes[node_idx].size / 2.0;
    double cz = nodes[node_idx].origin_z + nodes[node_idx].size / 2.0;

    int octant = get_octant(
        particles[particle_idx].x,
        particles[particle_idx].y,
        particles[particle_idx].z,
        cx, cy, cz);

    insert(nodes[node_idx].first_child + octant,
           particle_idx, particles, depth + 1);
    nodes[node_idx].particle_count++;
}

// ============================================================================
// Bottom-up aggregation: compute center of mass and total mass for each node.
//
// For leaf nodes: COM = particle position, mass = particle mass
// For internal nodes: COM = weighted average of children COMs
// ============================================================================
void Octree::compute_aggregates(int node_idx, const std::vector<Particle>& particles) {
    OctreeNode& node = nodes[node_idx];

    if (node.particle_count == 0) {
        node.total_mass = 0.0;
        node.com_x = node.com_y = node.com_z = 0.0;
        return;
    }

    // Leaf with single particle
    if (node.first_child == -1 && node.particle_index >= 0) {
        const Particle& p = particles[node.particle_index];
        node.com_x = p.x;
        node.com_y = p.y;
        node.com_z = p.z;
        node.total_mass = p.mass;
        return;
    }

    // Internal node: aggregate from children
    if (node.first_child >= 0) {
        double total_mass = 0.0;
        double weighted_x = 0.0, weighted_y = 0.0, weighted_z = 0.0;

        for (int c = 0; c < 8; ++c) {
            int child_idx = node.first_child + c;
            compute_aggregates(child_idx, particles);

            double cm = nodes[child_idx].total_mass;
            total_mass += cm;
            weighted_x += cm * nodes[child_idx].com_x;
            weighted_y += cm * nodes[child_idx].com_y;
            weighted_z += cm * nodes[child_idx].com_z;
        }

        node.total_mass = total_mass;
        if (total_mass > 0.0) {
            node.com_x = weighted_x / total_mass;
            node.com_y = weighted_y / total_mass;
            node.com_z = weighted_z / total_mass;
        }
    }
}

} // namespace albh
