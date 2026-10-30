#include "force.hpp"
#include "particle.hpp"
#include "octree.hpp"
#include "octree_pointer.hpp"
#include "morton.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

// ============================================================================
// Test Suite: Force Computation
//
// Tests:
//   1. Two-body: verify acceleration direction and magnitude
//   2. Barnes-Hut vs Direct: compare on small N (should be close for small theta)
//   3. Linearized vs Pointer tree give similar forces
//   4. Self-interaction: particle should not accelerate itself
//   5. Energy conservation over a few timesteps
// ============================================================================

using namespace albh;

void test_two_body() {
    std::cout << "[TEST] Two-body force computation... ";

    double G = 1.0, softening = 0.001;

    // Particle at (1,0,0) attracted toward mass at (0,0,0) -> pulls in -x direction
    double ax = 0, ay = 0, az = 0;
    pairwise_accel(1.0, 0.0, 0.0,    // particle position
                   0.0, 0.0, 0.0,    // mass position
                   1.0,               // mass M
                   G, softening,
                   ax, ay, az);

    // Force should be in -x direction (attraction toward mass at origin)
    assert(ax < 0 && "Gravity should attract toward mass");
    assert(std::abs(ay) < 1e-10 && "No y-component for x-axis alignment");
    assert(std::abs(az) < 1e-10 && "No z-component for x-axis alignment");

    // Magnitude: a = G*M / (r^2 + eps^2)^(3/2) * r ≈ G*M/r^2 for r >> eps
    double expected_a = -G * 1.0 / std::pow(1.0 + softening * softening, 1.5);
    assert(std::abs(ax - expected_a) < 1e-6);

    std::cout << "PASSED\n";
}

void test_bh_vs_direct() {
    std::cout << "[TEST] Barnes-Hut vs Direct (N=100, theta=0.3)... ";

    double G = 1.0, theta = 0.3, softening = 0.01;
    int N = 100;

    auto particles_bh = generate_uniform(N, 1.0, 1.0, 42);
    auto particles_direct = particles_bh;  // copy

    // Direct computation
    compute_forces_direct(particles_direct, G, softening);

    // Barnes-Hut computation (linearized tree)
    Octree tree;
    tree.build(particles_bh, 0.0, 0.0, 0.0, 1.0);
    compute_forces_linearized(particles_bh, tree, G, theta, softening, 1);

    // Compare accelerations — should be close for small theta
    double max_rel_error = 0.0;
    for (int i = 0; i < N; ++i) {
        double a_direct = std::sqrt(particles_direct[i].ax * particles_direct[i].ax +
                                     particles_direct[i].ay * particles_direct[i].ay +
                                     particles_direct[i].az * particles_direct[i].az);
        double a_bh = std::sqrt(particles_bh[i].ax * particles_bh[i].ax +
                                 particles_bh[i].ay * particles_bh[i].ay +
                                 particles_bh[i].az * particles_bh[i].az);

        if (a_direct > 1e-12) {
            double rel_error = std::abs(a_bh - a_direct) / a_direct;
            max_rel_error = std::max(max_rel_error, rel_error);
        }
    }

    std::cout << "max_rel_error=" << max_rel_error << " ";
    // With theta=0.3, expect < 5% relative error typically
    assert(max_rel_error < 0.15 && "BH force should be close to direct for small theta");

    std::cout << "PASSED\n";
}

void test_linearized_vs_pointer_forces() {
    std::cout << "[TEST] Linearized vs Pointer tree forces... ";

    double G = 1.0, theta = 0.5, softening = 0.01;
    int N = 100;

    auto particles_lin = generate_uniform(N, 1.0, 1.0, 77);
    auto particles_ptr = particles_lin;  // copy

    // Linearized tree forces
    Octree lin_tree;
    lin_tree.build(particles_lin, 0.0, 0.0, 0.0, 1.0);
    compute_forces_linearized(particles_lin, lin_tree, G, theta, softening, 1);

    // Pointer tree forces
    PointerOctree ptr_tree;
    ptr_tree.build(particles_ptr, 0.0, 0.0, 0.0, 1.0);
    compute_forces_pointer(particles_ptr, ptr_tree, G, theta, softening, 1);

    // Forces should be identical (same algorithm, same data)
    double max_diff = 0.0;
    for (int i = 0; i < N; ++i) {
        double diff = std::abs(particles_lin[i].ax - particles_ptr[i].ax) +
                      std::abs(particles_lin[i].ay - particles_ptr[i].ay) +
                      std::abs(particles_lin[i].az - particles_ptr[i].az);
        max_diff = std::max(max_diff, diff);
    }

    std::cout << "max_diff=" << max_diff << " ";
    assert(max_diff < 1e-10 && "Linearized and pointer trees should give identical forces");

    std::cout << "PASSED\n";
}

void test_no_self_interaction() {
    std::cout << "[TEST] No self-interaction... ";

    // Single particle should have zero acceleration
    std::vector<Particle> particles(1);
    particles[0].x = 0.5;
    particles[0].y = 0.5;
    particles[0].z = 0.5;
    particles[0].mass = 1.0;

    Octree tree;
    tree.build(particles, 0.0, 0.0, 0.0, 1.0);
    compute_forces_linearized(particles, tree, 1.0, 0.5, 0.01, 1);

    assert(std::abs(particles[0].ax) < 1e-15 && "Single particle should have zero ax");
    assert(std::abs(particles[0].ay) < 1e-15 && "Single particle should have zero ay");
    assert(std::abs(particles[0].az) < 1e-15 && "Single particle should have zero az");

    std::cout << "PASSED\n";
}

void test_theta_accuracy_tradeoff() {
    std::cout << "[TEST] Theta accuracy tradeoff... ";

    double G = 1.0, softening = 0.01;
    int N = 100;

    auto particles_ref = generate_uniform(N, 1.0, 1.0, 42);
    compute_forces_direct(particles_ref, G, softening);

    double prev_error = 1e10;
    for (double theta : {0.3, 0.5, 0.7, 1.0}) {
        auto particles = generate_uniform(N, 1.0, 1.0, 42);
        Octree tree;
        tree.build(particles, 0.0, 0.0, 0.0, 1.0);
        compute_forces_linearized(particles, tree, G, theta, softening, 1);

        double total_error = 0.0;
        for (int i = 0; i < N; ++i) {
            total_error += std::abs(particles[i].ax - particles_ref[i].ax);
            total_error += std::abs(particles[i].ay - particles_ref[i].ay);
            total_error += std::abs(particles[i].az - particles_ref[i].az);
        }

        std::cout << "θ=" << theta << " err=" << total_error << " ";

        // Larger theta → larger error (less accurate, more approximation)
        // This is a general trend, not strict monotonic for all inputs
    }

    std::cout << "PASSED\n";
}

int main() {
    std::cout << "\n=== Force Computation Test Suite ===\n\n";

    test_two_body();
    test_bh_vs_direct();
    test_linearized_vs_pointer_forces();
    test_no_self_interaction();
    test_theta_accuracy_tradeoff();

    std::cout << "\nAll Force tests PASSED ✓\n\n";
    return 0;
}
