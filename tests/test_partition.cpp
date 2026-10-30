#include "partition.hpp"
#include "particle.hpp"
#include "morton.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

using namespace albh;

// ============================================================================
// Test Suite: Domain Decomposition & Morton Partitioning (Milestone 3)
// ============================================================================

void test_equal_count_balance() {
    std::cout << "[TEST] Morton Equal-Count Partitioning (N=1003, P=4)... ";

    int N = 1003;
    int P = 4;
    double box_size = 1.0;
    auto particles = generate_uniform(N, box_size, 1.0, 42);
    compute_and_sort_by_morton(particles, box_size);

    auto partitions = partition_equal_count(particles, P);

    assert(static_cast<int>(partitions.size()) == P);

    // Sum of particles across partitions must equal N
    int total = 0;
    for (int r = 0; r < P; ++r) {
        total += partitions[r].particle_count;
        // In 1003 / 4, counts should be 251, 251, 251, 250
        assert(partitions[r].particle_count == 250 || partitions[r].particle_count == 251);
        assert(partitions[r].end_idx - partitions[r].start_idx == partitions[r].particle_count);
    }
    assert(total == N);

    std::cout << "PASSED\n";
}

void test_morton_range_monotonicity() {
    std::cout << "[TEST] Morton Key Range Monotonicity... ";

    int N = 500;
    int P = 8;
    double box_size = 1.0;
    auto particles = generate_uniform(N, box_size, 1.0, 42);
    compute_and_sort_by_morton(particles, box_size);

    auto partitions = partition_equal_count(particles, P);

    for (int r = 0; r < P - 1; ++r) {
        assert(partitions[r].morton_end <= partitions[r + 1].morton_start &&
               "Morton ranges between adjacent ranks must be monotonically non-overlapping");
    }

    std::cout << "PASSED\n";
}

void test_ownership_and_lookup() {
    std::cout << "[TEST] Particle Ownership Assignment and Binary Search Lookup... ";

    int N = 300;
    int P = 4;
    double box_size = 1.0;
    auto particles = generate_uniform(N, box_size, 1.0, 42);
    compute_and_sort_by_morton(particles, box_size);

    auto partitions = partition_equal_count(particles, P);
    assign_ownership(particles, partitions);

    for (int r = 0; r < P; ++r) {
        for (int i = partitions[r].start_idx; i < partitions[r].end_idx; ++i) {
            assert(particles[i].owner_rank == r);
            // Verify binary search lookup
            int found_rank = find_owner_rank(particles[i].morton_key, partitions);
            assert(found_rank == r);
        }
    }

    std::cout << "PASSED\n";
}

void test_morton_vs_static_imbalance() {
    std::cout << "[TEST] Morton vs Static Geometric Imbalance on Clustered Galaxy... ";

    int N = 2000;
    int P = 8;
    double box_size = 1.0;
    // Generate Plummer distribution (clustered at center)
    auto particles = generate_plummer(N, box_size, 1.0, 0.1, 42);

    // 1. Static Geometric
    auto static_parts = partition_static_geometric(particles, P, box_size);
    int min_static = N, max_static = 0;
    for (const auto& p : static_parts) {
        min_static = std::min(min_static, p.particle_count);
        max_static = std::max(max_static, p.particle_count);
    }
    double static_imbalance = (min_static > 0) ? (static_cast<double>(max_static) / min_static) : 999.0;

    // 2. Morton Equal-Count
    compute_and_sort_by_morton(particles, box_size);
    auto morton_parts = partition_equal_count(particles, P);
    int min_morton = N, max_morton = 0;
    for (const auto& p : morton_parts) {
        min_morton = std::min(min_morton, p.particle_count);
        max_morton = std::max(max_morton, p.particle_count);
    }
    double morton_imbalance = static_cast<double>(max_morton) / min_morton;

    // Static geometric on clustered distribution should have severe imbalance
    // while Morton equal-count has perfect balance (imbalance ~ 1.0)
    assert(static_imbalance > 2.0 && "Static geometric must suffer high imbalance on clustered data");
    assert(morton_imbalance <= 1.01 && "Morton equal-count must achieve near-perfect load balance");

    std::cout << "Static Imbalance=" << static_imbalance
              << " vs Morton Imbalance=" << morton_imbalance << " PASSED\n";
}

int main() {
    std::cout << "\n=== Domain Decomposition & Partitioning Test Suite (Milestone 3) ===\n\n";

    test_equal_count_balance();
    test_morton_range_monotonicity();
    test_ownership_and_lookup();
    test_morton_vs_static_imbalance();

    std::cout << "\nAll Partition tests PASSED ✓\n\n";
    return 0;
}
