#include "load_balancer.hpp"
#include "particle.hpp"
#include "morton.hpp"
#include <chrono>
#include <algorithm>
#include <cmath>

namespace albh {

LoadBalancer::LoadBalancer(int num_ranks, double threshold)
    : num_ranks_(num_ranks),
      threshold_(threshold),
      total_rebalances_(0),
      total_migrated_(0),
      total_rebalance_time_(0.0) {}

double LoadBalancer::compute_imbalance(const std::vector<double>& workloads) {
    if (workloads.empty()) return 1.0;

    double max_w = workloads[0];
    double min_w = workloads[0];

    for (double w : workloads) {
        if (w > max_w) max_w = w;
        if (w < min_w) min_w = w;
    }

    if (min_w <= 1e-9) {
        return (max_w > 0.0) ? 999.0 : 1.0;
    }

    return max_w / min_w;
}

RebalanceResult LoadBalancer::check_and_rebalance(
    std::vector<Particle>& particles,
    std::vector<PartitionInfo>& partitions,
    const std::vector<double>& rank_workloads,
    double box_size) {

    RebalanceResult result;
    result.imbalance_before = compute_imbalance(rank_workloads);
    result.imbalance_after = result.imbalance_before;

    // Check if rebalancing threshold is exceeded
    if (result.imbalance_before <= threshold_ || num_ranks_ <= 1) {
        return result;
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    // Rebalancing triggered!
    result.triggered = true;

    // 1. Sort particles along Morton curve
    compute_and_sort_by_morton(particles, box_size);

    // 2. Recompute optimal partition splitters along Morton curve
    auto new_partitions = partition_equal_count(particles, num_ranks_);

    // 3. Count migrated particles (particles whose owner_rank changed)
    int migrated = 0;
    for (const auto& part : new_partitions) {
        for (int i = part.start_idx; i < part.end_idx && i < static_cast<int>(particles.size()); ++i) {
            if (particles[i].owner_rank != part.rank) {
                migrated++;
            }
        }
    }

    // 4. Update ownership
    assign_ownership(particles, new_partitions);
    partitions = new_partitions;

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    // 5. Update metrics
    result.particles_migrated = migrated;
    result.rebalance_time = elapsed;

    // Evaluate new workload (particle count based)
    std::vector<double> new_workloads(num_ranks_);
    for (int r = 0; r < num_ranks_; ++r) {
        new_workloads[r] = static_cast<double>(partitions[r].particle_count);
    }
    result.imbalance_after = compute_imbalance(new_workloads);

    total_rebalances_++;
    total_migrated_ += migrated;
    total_rebalance_time_ += elapsed;

    return result;
}

} // namespace albh
