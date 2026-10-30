#ifndef ALBH_LOAD_BALANCER_HPP
#define ALBH_LOAD_BALANCER_HPP

#include "partition.hpp"
#include <vector>

namespace albh {

struct Particle;

// ============================================================================
// Dynamic Load Balancer for AL-BH
//
// Evaluates computational workload across ranks per timestep.
// If load imbalance I = max(W_i) / min(W_i) > threshold (default: 1.2),
// triggers repartitioning along the Morton space-filling curve.
// ============================================================================

struct RebalanceResult {
    bool triggered = false;
    double imbalance_before = 1.0;
    double imbalance_after = 1.0;
    int particles_migrated = 0;
    double rebalance_time = 0.0;
};

class LoadBalancer {
public:
    LoadBalancer(int num_ranks = 1, double threshold = 1.2);

    // Evaluate workload and rebalance if imbalance exceeds threshold.
    // Work vector can be per-rank particle counts or measured compute times.
    RebalanceResult check_and_rebalance(
        std::vector<Particle>& particles,
        std::vector<PartitionInfo>& partitions,
        const std::vector<double>& rank_workloads,
        double box_size);

    // Compute imbalance ratio max(W) / min(W)
    static double compute_imbalance(const std::vector<double>& workloads);

    // Accessors
    double get_threshold() const { return threshold_; }
    int get_total_rebalances() const { return total_rebalances_; }
    int get_total_particles_migrated() const { return total_migrated_; }
    double get_total_rebalance_time() const { return total_rebalance_time_; }

private:
    int num_ranks_;
    double threshold_;
    int total_rebalances_;
    int total_migrated_;
    double total_rebalance_time_;
};

} // namespace albh

#endif // ALBH_LOAD_BALANCER_HPP
