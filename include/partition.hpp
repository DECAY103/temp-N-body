#ifndef ALBH_PARTITION_HPP
#define ALBH_PARTITION_HPP

#include <vector>
#include <cstdint>

namespace albh {

struct Particle;

// ============================================================================
// Morton-Based Domain Decomposition
//
// Partitions Morton-sorted particles into contiguous ranges for MPI ranks.
// Two strategies:
//   1. Equal-count: each rank gets ~N/num_ranks particles (default)
//   2. Static geometric: equal-volume spatial regions (baseline comparison)
//
// Partition boundaries are stored as Morton key ranges.
// ============================================================================

struct PartitionInfo {
    int rank;                    // MPI rank
    int start_idx;               // first particle index in sorted array
    int end_idx;                 // last particle index (exclusive)
    uint64_t morton_start;       // Morton key lower bound
    uint64_t morton_end;         // Morton key upper bound
    int particle_count;          // number of particles in this partition
};

// Compute equal-count partitions along the Morton curve
// Requires particles already sorted by Morton key
std::vector<PartitionInfo> partition_equal_count(
    const std::vector<Particle>& particles, int num_ranks);

// Compute static geometric partitions (equal-volume, baseline)
// Divides box into slabs along x-axis
std::vector<PartitionInfo> partition_static_geometric(
    const std::vector<Particle>& particles, int num_ranks, double box_size);

// Assign owner_rank to each particle based on partition info
void assign_ownership(std::vector<Particle>& particles,
                       const std::vector<PartitionInfo>& partitions);

// Find which partition (rank) a Morton key belongs to
int find_owner_rank(uint64_t morton_key,
                     const std::vector<PartitionInfo>& partitions);

} // namespace albh

#endif // ALBH_PARTITION_HPP
