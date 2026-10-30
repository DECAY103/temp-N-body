#include "morton.hpp"
#include "particle.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <algorithm>

// ============================================================================
// Test Suite: Morton Z-order Key Encoding/Decoding
//
// Tests:
//   1. Round-trip: encode → decode recovers original coordinates
//   2. Spatial locality: nearby points have nearby keys
//   3. Sorting: Morton-sorted particles preserve spatial locality
//   4. Edge cases: boundary values, zero, max
// ============================================================================

using namespace albh;

void test_round_trip() {
    std::cout << "[TEST] Morton round-trip encoding... ";

    // Test a range of coordinate values
    uint32_t test_coords[][3] = {
        {0, 0, 0},
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1},
        {MORTON_MAX_COORD, MORTON_MAX_COORD, MORTON_MAX_COORD},
        {12345, 67890, 11111},
        {MORTON_MAX_COORD/2, MORTON_MAX_COORD/2, MORTON_MAX_COORD/2},
        {1, 2, 3},
        {100, 200, 300},
    };

    for (auto& tc : test_coords) {
        uint64_t key = encode_morton(tc[0], tc[1], tc[2]);
        uint32_t dx, dy, dz;
        decode_morton(key, dx, dy, dz);

        assert(dx == tc[0] && "X coordinate mismatch after round-trip");
        assert(dy == tc[1] && "Y coordinate mismatch after round-trip");
        assert(dz == tc[2] && "Z coordinate mismatch after round-trip");
    }

    std::cout << "PASSED\n";
}

void test_quantization() {
    std::cout << "[TEST] Coordinate quantization... ";

    double box_size = 1.0;

    // Zero maps to 0
    assert(quantize_coord(0.0, box_size) == 0);

    // Mid-point maps to ~half of max
    uint32_t mid = quantize_coord(0.5, box_size);
    assert(mid > MORTON_MAX_COORD / 2 - 100 && mid < MORTON_MAX_COORD / 2 + 100);

    // Near-max maps close to MORTON_MAX_COORD
    uint32_t near_max = quantize_coord(0.999, box_size);
    assert(near_max > MORTON_MAX_COORD - 5000);

    // Negative clamps to 0
    assert(quantize_coord(-1.0, box_size) == 0);

    // Over box_size clamps
    uint32_t over = quantize_coord(2.0, box_size);
    assert(over == MORTON_MAX_COORD);

    // De-quantize round-trip (approximate)
    for (double val : {0.0, 0.25, 0.5, 0.75, 0.999}) {
        uint32_t q = quantize_coord(val, box_size);
        double recovered = dequantize_coord(q, box_size);
        assert(std::abs(recovered - val) < 0.001);
    }

    std::cout << "PASSED\n";
}

void test_spatial_locality() {
    std::cout << "[TEST] Morton spatial locality... ";

    double box_size = 1.0;

    // Two nearby points should have nearby Morton keys
    uint64_t k1 = particle_to_morton(0.5, 0.5, 0.5, box_size);
    uint64_t k2 = particle_to_morton(0.501, 0.501, 0.501, box_size);
    uint64_t k3 = particle_to_morton(0.1, 0.9, 0.1, box_size);  // far away

    // Near points should have closer keys than far points
    uint64_t diff_near = (k1 > k2) ? (k1 - k2) : (k2 - k1);
    uint64_t diff_far  = (k1 > k3) ? (k1 - k3) : (k3 - k1);

    // This is a statistical property, not guaranteed for all points,
    // but for these specific points it should hold
    assert(diff_near < diff_far && "Nearby points should have closer Morton keys");

    std::cout << "PASSED\n";
}

void test_sort_particles() {
    std::cout << "[TEST] Morton sort of particles... ";

    double box_size = 1.0;
    auto particles = generate_uniform(100, box_size, 1.0, 42);

    compute_and_sort_by_morton(particles, box_size);

    // Verify sorted order
    for (size_t i = 1; i < particles.size(); ++i) {
        assert(particles[i].morton_key >= particles[i-1].morton_key
               && "Particles should be sorted by Morton key");
    }

    // Verify all keys are non-zero (since all positions are in (0, box_size))
    for (const auto& p : particles) {
        assert(p.morton_key > 0 || (p.x < 1e-10 && p.y < 1e-10 && p.z < 1e-10));
    }

    std::cout << "PASSED\n";
}

void test_determinism() {
    std::cout << "[TEST] Morton key determinism... ";

    double box_size = 1.0;
    auto p1 = generate_uniform(50, box_size, 1.0, 123);
    auto p2 = generate_uniform(50, box_size, 1.0, 123);

    compute_and_sort_by_morton(p1, box_size);
    compute_and_sort_by_morton(p2, box_size);

    for (size_t i = 0; i < p1.size(); ++i) {
        assert(p1[i].morton_key == p2[i].morton_key
               && "Same seed should produce identical Morton keys");
    }

    std::cout << "PASSED\n";
}

int main() {
    std::cout << "\n=== Morton Key Test Suite ===\n\n";

    test_round_trip();
    test_quantization();
    test_spatial_locality();
    test_sort_particles();
    test_determinism();

    std::cout << "\nAll Morton tests PASSED ✓\n\n";
    return 0;
}
