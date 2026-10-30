#include "partition.hpp"
#include "particle.hpp"
#include <algorithm>
#include <cmath>

namespace albh {

// ============================================================================
// Equal-Count Partitioning along the 1D Morton Space-Filling Curve
//
// Because particles are already sorted by their 64-bit Morton key,
// contiguous 1D index ranges correspond to spatially localized 3D clusters.
// Each rank receives roughly N / num_ranks particles.
// ============================================================================
std::vector<PartitionInfo> partition_equal_count(
    const std::vector<Particle>& particles, int num_ranks) {
    
    std::vector<PartitionInfo> partitions(num_ranks);
    int N = static_cast<int>(particles.size());
    if (num_ranks <= 0 || N == 0) return partitions;

    int base_count = N / num_ranks;
    int remainder = N % num_ranks;

    int current_idx = 0;
    for (int r = 0; r < num_ranks; ++r) {
        int count = base_count + (r < remainder ? 1 : 0);
        partitions[r].rank = r;
        partitions[r].start_idx = current_idx;
        partitions[r].end_idx = current_idx + count;
        partitions[r].particle_count = count;

        if (count > 0) {
            partitions[r].morton_start = particles[partitions[r].start_idx].morton_key;
            partitions[r].morton_end = particles[partitions[r].end_idx - 1].morton_key;
        } else {
            partitions[r].morton_start = 0;
            partitions[r].morton_end = 0;
        }

        current_idx += count;
    }

    return partitions;
}

// ============================================================================
// Static Geometric Partitioning (Equal-Volume Baseline)
//
// Divides the bounding box along the X-axis into equal-width slabs.
// For clustered distributions (e.g. Plummer galaxies), this causes massive
// load imbalance because the central ranks receive the vast majority of particles.
// ============================================================================
std::vector<PartitionInfo> partition_static_geometric(
    const std::vector<Particle>& particles, int num_ranks, double box_size) {
    
    std::vector<PartitionInfo> partitions(num_ranks);
    int N = static_cast<int>(particles.size());
    if (num_ranks <= 0 || N == 0) return partitions;

    double slab_width = box_size / static_cast<double>(num_ranks);

    // Count particles per slab
    std::vector<int> counts(num_ranks, 0);
    for (const auto& p : particles) {
        int r = static_cast<int>(p.x / slab_width);
        r = std::max(0, std::min(num_ranks - 1, r));
        counts[r]++;
    }

    int current_idx = 0;
    for (int r = 0; r < num_ranks; ++r) {
        partitions[r].rank = r;
        partitions[r].start_idx = current_idx;
        partitions[r].end_idx = current_idx + counts[r];
        partitions[r].particle_count = counts[r];
        partitions[r].morton_start = 0;
        partitions[r].morton_end = 0;
        current_idx += counts[r];
    }

    return partitions;
}

// ============================================================================
// Assign owner_rank to each particle according to the partition ranges
// ============================================================================
void assign_ownership(std::vector<Particle>& particles,
                       const std::vector<PartitionInfo>& partitions) {
    for (const auto& part : partitions) {
        for (int i = part.start_idx; i < part.end_idx && i < static_cast<int>(particles.size()); ++i) {
            particles[i].owner_rank = part.rank;
        }
    }
}

// ============================================================================
// Find which rank owns a given Morton key (binary search over partition ranges)
// ============================================================================
int find_owner_rank(uint64_t morton_key,
                     const std::vector<PartitionInfo>& partitions) {
    if (partitions.empty()) return 0;

    int left = 0;
    int right = static_cast<int>(partitions.size()) - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (morton_key <= partitions[mid].morton_end) {
            if (mid == 0 || morton_key > partitions[mid - 1].morton_end) {
                return partitions[mid].rank;
            }
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }

    return partitions.back().rank;
}

} // namespace albh
