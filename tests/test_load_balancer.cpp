#include "load_balancer.hpp"
#include "particle.hpp"
#include "morton.hpp"
#include <iostream>
#include <cassert>
#include <vector>

using namespace albh;

// ============================================================================
// Test Suite: Dynamic Load Balancer (Milestone 6)
// ============================================================================

void test_imbalance_calculation() {
    std::cout << "[TEST] Imbalance ratio calculation... ";

    std::vector<double> balanced = {100.0, 100.0, 100.0, 100.0};
    assert(std::abs(LoadBalancer::compute_imbalance(balanced) - 1.0) < 1e-6);

    std::vector<double> imbalanced = {200.0, 100.0, 50.0, 100.0};
    // max = 200, min = 50 -> ratio = 4.0
    assert(std::abs(LoadBalancer::compute_imbalance(imbalanced) - 4.0) < 1e-6);

    std::cout << "PASSED\n";
}

void test_rebalance_trigger_and_migration() {
    std::cout << "[TEST] Dynamic Rebalancing Trigger & Particle Migration... ";

    int N = 1000;
    int P = 4;
    double box_size = 1.0;
    double threshold = 1.2;

    auto particles = generate_uniform(N, box_size, 1.0, 42);
    compute_and_sort_by_morton(particles, box_size);

    auto partitions = partition_equal_count(particles, P);
    assign_ownership(particles, partitions);

    LoadBalancer lb(P, threshold);

    // Case 1: Balanced workload (ratio = 1.0 < 1.2) -> should NOT trigger
    std::vector<double> balanced_workloads = {250.0, 250.0, 250.0, 250.0};
    auto res1 = lb.check_and_rebalance(particles, partitions, balanced_workloads, box_size);
    assert(!res1.triggered);
    assert(lb.get_total_rebalances() == 0);

    // Case 2: Imbalanced workload (ratio = 400/100 = 4.0 > 1.2) -> SHOULD trigger
    std::vector<double> imbalanced_workloads = {400.0, 300.0, 200.0, 100.0};
    auto res2 = lb.check_and_rebalance(particles, partitions, imbalanced_workloads, box_size);
    assert(res2.triggered);
    assert(res2.imbalance_before == 4.0);
    assert(res2.imbalance_after <= 1.01);
    assert(lb.get_total_rebalances() == 1);

    std::cout << "Triggered=YES, Imbalance: " << res2.imbalance_before
              << " -> " << res2.imbalance_after << " PASSED\n";
}

int main() {
    std::cout << "\n=== Dynamic Load Balancer Test Suite (Milestone 6) ===\n\n";

    test_imbalance_calculation();
    test_rebalance_trigger_and_migration();

    std::cout << "\nAll Load Balancer tests PASSED ✓\n\n";
    return 0;
}
