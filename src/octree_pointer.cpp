#include "octree_pointer.hpp"
#include "particle.hpp"

namespace albh {

// ============================================================================
// Pointer-Based Octree (Baseline for Cache Comparison)
//
// Same logical structure as linearized Octree, but each node is individually
// heap-allocated. This causes pointer chasing during traversal, leading to
// more cache misses — which is exactly what we want to measure.
// ============================================================================

void PointerOctree::build(const std::vector<Particle>& particles,
                           double box_origin_x, double box_origin_y, double box_origin_z,
                           double box_size) {
    clear();

    if (particles.empty()) return;

    // Create root node
    root_node = std::make_unique<PointerOctreeNode>();
    root_node->origin_x = box_origin_x;
    root_node->origin_y = box_origin_y;
    root_node->origin_z = box_origin_z;
    root_node->size = box_size;

    // Insert each particle
    for (int i = 0; i < static_cast<int>(particles.size()); ++i) {
        insert(root_node.get(), i, particles, 0);
    }

    // Compute aggregates
    compute_aggregates(root_node.get(), particles);
}

void PointerOctree::clear() {
    root_node.reset();
}

int PointerOctree::get_octant(double px, double py, double pz,
                               double cx, double cy, double cz) {
    int octant = 0;
    if (px >= cx) octant |= 1;
    if (py >= cy) octant |= 2;
    if (pz >= cz) octant |= 4;
    return octant;
}

void PointerOctree::insert(PointerOctreeNode* node, int particle_idx,
                             const std::vector<Particle>& particles, int depth) {
    if (depth >= MAX_DEPTH) {
        node->particle_count++;
        if (node->particle_index == -1) {
            node->particle_index = particle_idx;
            node->particle_id = particles[particle_idx].id;
        } else {
            node->overflow_indices.push_back(particle_idx);
        }
        return;
    }

    if (node->particle_count == 0) {
        // Empty node → place particle as leaf
        node->particle_index = particle_idx;
        node->particle_id = particles[particle_idx].id;
        node->particle_count = 1;
        return;
    }

    double cx = node->origin_x + node->size / 2.0;
    double cy = node->origin_y + node->size / 2.0;
    double cz = node->origin_z + node->size / 2.0;

    if (node->particle_count == 1 && !node->children[0]) {
        // Occupied leaf → subdivide
        int existing = node->particle_index;
        node->particle_index = -1;
        node->particle_id = -1;

        double child_size = node->size / 2.0;

        // Allocate 8 children (each individually heap-allocated — deliberate)
        for (int i = 0; i < 8; ++i) {
            node->children[i] = std::make_unique<PointerOctreeNode>();
            node->children[i]->size = child_size;
            node->children[i]->origin_x = node->origin_x + ((i & 1) ? child_size : 0.0);
            node->children[i]->origin_y = node->origin_y + ((i & 2) ? child_size : 0.0);
            node->children[i]->origin_z = node->origin_z + ((i & 4) ? child_size : 0.0);
        }

        // Re-insert existing particle
        int oct_existing = get_octant(particles[existing].x, particles[existing].y,
                                       particles[existing].z, cx, cy, cz);
        node->particle_count = 0;
        insert(node->children[oct_existing].get(), existing, particles, depth + 1);
        node->particle_count++;

        // Insert new particle
        int oct_new = get_octant(particles[particle_idx].x, particles[particle_idx].y,
                                  particles[particle_idx].z, cx, cy, cz);
        insert(node->children[oct_new].get(), particle_idx, particles, depth + 1);
        node->particle_count++;
        return;
    }

    // Internal node → recurse
    int octant = get_octant(particles[particle_idx].x, particles[particle_idx].y,
                             particles[particle_idx].z, cx, cy, cz);
    insert(node->children[octant].get(), particle_idx, particles, depth + 1);
    node->particle_count++;
}

void PointerOctree::compute_aggregates(PointerOctreeNode* node,
                                        const std::vector<Particle>& particles) {
    if (!node || node->particle_count == 0) {
        if (node) {
            node->total_mass = 0.0;
            node->com_x = node->com_y = node->com_z = 0.0;
        }
        return;
    }

    // Leaf node (with primary particle and optional overflow coincident particles)
    if (node->particle_index >= 0 && !node->children[0]) {
        const Particle& p = particles[node->particle_index];
        double total_mass = p.mass;
        double wx = p.x * p.mass;
        double wy = p.y * p.mass;
        double wz = p.z * p.mass;

        for (int extra_idx : node->overflow_indices) {
            const Particle& ep = particles[extra_idx];
            total_mass += ep.mass;
            wx += ep.x * ep.mass;
            wy += ep.y * ep.mass;
            wz += ep.z * ep.mass;
        }

        node->total_mass = total_mass;
        if (total_mass > 0.0) {
            node->com_x = wx / total_mass;
            node->com_y = wy / total_mass;
            node->com_z = wz / total_mass;
        } else {
            node->com_x = p.x;
            node->com_y = p.y;
            node->com_z = p.z;
        }
        return;
    }

    // Internal node
    double total_mass = 0.0;
    double wx = 0.0, wy = 0.0, wz = 0.0;

    for (int c = 0; c < 8; ++c) {
        if (node->children[c]) {
            compute_aggregates(node->children[c].get(), particles);
            double cm = node->children[c]->total_mass;
            total_mass += cm;
            wx += cm * node->children[c]->com_x;
            wy += cm * node->children[c]->com_y;
            wz += cm * node->children[c]->com_z;
        }
    }

    node->total_mass = total_mass;
    if (total_mass > 0.0) {
        node->com_x = wx / total_mass;
        node->com_y = wy / total_mass;
        node->com_z = wz / total_mass;
    } else {
        node->com_x = node->origin_x + node->size / 2.0;
        node->com_y = node->origin_y + node->size / 2.0;
        node->com_z = node->origin_z + node->size / 2.0;
    }
}

} // namespace albh
