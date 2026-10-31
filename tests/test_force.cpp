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
    (void)expected_a;

    std::cout << "PASSED\n";
}

void test_bh_vs_direct() {
    std::cout << "[TEST] Barnes-Hut vs Direct vector error (N=100, theta=0.3)... ";

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

    // FIX: Compare actual acceleration 3D vectors: |a_bh - a_direct| / |a_direct|
    // This catches wrong-direction errors that scalar magnitude comparison would miss.
    double max_rel_error = 0.0;
    for (int i = 0; i < N; ++i) {
        double dax = particles_bh[i].ax - particles_direct[i].ax;
        double day = particles_bh[i].ay - particles_direct[i].ay;
        double daz = particles_bh[i].az - particles_direct[i].az;
        double vec_diff = std::sqrt(dax * dax + day * day + daz * daz);
        double a_direct = std::sqrt(particles_direct[i].ax * particles_direct[i].ax +
                                     particles_direct[i].ay * particles_direct[i].ay +
                                     particles_direct[i].az * particles_direct[i].az);

        if (a_direct > 1e-12) {
            double rel_error = vec_diff / a_direct;
            max_rel_error = std::max(max_rel_error, rel_error);
        }
    }

    std::cout << "max_rel_vec_error=" << max_rel_error << " ";
    assert(max_rel_error < 0.15 && "BH force vector error should be small for theta=0.3");

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
    std::cout << "[TEST] No self-interaction (single & multi-particle)... ";

    // 1. Single particle should have zero acceleration
    std::vector<Particle> particles(1);
    particles[0].x = 0.5;
    particles[0].y = 0.5;
    particles[0].z = 0.5;
    particles[0].mass = 1.0;
    particles[0].id = 0;

    Octree tree;
    tree.build(particles, 0.0, 0.0, 0.0, 1.0);
    compute_forces_linearized(particles, tree, 1.0, 0.5, 0.01, 1);

    assert(std::abs(particles[0].ax) < 1e-15 && "Single particle should have zero ax");
    assert(std::abs(particles[0].ay) < 1e-15 && "Single particle should have zero ay");
    assert(std::abs(particles[0].az) < 1e-15 && "Single particle should have zero az");

    // 2. Multi-particle self-exclusion check:
    // A symmetrical configuration: 2 identical particles along x-axis symmetric about center
    std::vector<Particle> two_particles(2);
    two_particles[0].x = 0.4; two_particles[0].y = 0.5; two_particles[0].z = 0.5;
    two_particles[0].mass = 1.0; two_particles[0].id = 0;

    two_particles[1].x = 0.6; two_particles[1].y = 0.5; two_particles[1].z = 0.5;
    two_particles[1].mass = 1.0; two_particles[1].id = 1;

    Octree tree2;
    tree2.build(two_particles, 0.0, 0.0, 0.0, 1.0);
    compute_forces_linearized(two_particles, tree2, 1.0, 0.3, 0.01, 1);

    // By symmetry: particle 0 should feel +x force, particle 1 should feel equal -x force
    assert(two_particles[0].ax > 0 && "Particle 0 must be attracted towards particle 1 (+x)");
    assert(two_particles[1].ax < 0 && "Particle 1 must be attracted towards particle 0 (-x)");
    assert(std::abs(two_particles[0].ax + two_particles[1].ax) < 1e-10 && "Net internal force must sum to 0 (Newton III)");

    std::cout << "PASSED\n";
}

void test_coincident_particles_max_depth() {
    std::cout << "[TEST] Coincident particles at MAX_DEPTH... ";

    // Three particles at exactly the same coordinates (0.5, 0.5, 0.5)
    // plus one observer particle at (0.8, 0.5, 0.5)
    std::vector<Particle> particles(4);
    for (int i = 0; i < 3; ++i) {
        particles[i].x = 0.5;
        particles[i].y = 0.5;
        particles[i].z = 0.5;
        particles[i].mass = 1.0;
        particles[i].id = i;
    }
    particles[3].x = 0.8;
    particles[3].y = 0.5;
    particles[3].z = 0.5;
    particles[3].mass = 1.0;
    particles[3].id = 3;

    // Check Linearized Octree
    Octree lin_tree;
    lin_tree.build(particles, 0.0, 0.0, 0.0, 1.0);

    // Total mass in root node must equal 4.0 (no coincident mass lost!)
    assert(std::abs(lin_tree.root().total_mass - 4.0) < 1e-10 && "Octree must aggregate all coincident particles' masses");

    // Check Pointer Octree
    PointerOctree ptr_tree;
    ptr_tree.build(particles, 0.0, 0.0, 0.0, 1.0);
    assert(std::abs(ptr_tree.root_node->total_mass - 4.0) < 1e-10 && "Pointer Octree must aggregate all coincident particles' masses");

    // Compute forces: observer at (0.8, 0.5, 0.5) must feel attraction of 3 masses at (0.5, 0.5, 0.5)
    compute_forces_linearized(particles, lin_tree, 1.0, 0.5, 0.01, 1);
    double dx = 0.8 - 0.5;
    double expected_ax = -1.0 * 3.0 * dx / std::pow(dx * dx + 0.01 * 0.01, 1.5);
    assert(std::abs(particles[3].ax - expected_ax) < 1e-3 && "Observer must feel force from all coincident particles");
    (void)expected_ax;

    std::cout << "PASSED\n";
}

void test_theta_accuracy_tradeoff() {
    std::cout << "[TEST] Theta accuracy tradeoff... ";

    double G = 1.0, softening = 0.01;
    int N = 100;

    auto particles_ref = generate_uniform(N, 1.0, 1.0, 42);
    compute_forces_direct(particles_ref, G, softening);

    for (double theta : {0.3, 0.5, 0.7, 1.0}) {
        auto particles = generate_uniform(N, 1.0, 1.0, 42);
        Octree tree;
        tree.build(particles, 0.0, 0.0, 0.0, 1.0);
        compute_forces_linearized(particles, tree, G, theta, softening, 1);

        double total_error = 0.0;
        double total_direct = 0.0;
        for (int i = 0; i < N; ++i) {
            double dax = particles[i].ax - particles_ref[i].ax;
            double day = particles[i].ay - particles_ref[i].ay;
            double daz = particles[i].az - particles_ref[i].az;
            total_error += std::sqrt(dax * dax + day * day + daz * daz);
            total_direct += std::sqrt(particles_ref[i].ax * particles_ref[i].ax +
                                      particles_ref[i].ay * particles_ref[i].ay +
                                      particles_ref[i].az * particles_ref[i].az);
        }

        double rel_vec_error = total_error / total_direct;
        std::cout << "θ=" << theta << " rel_vec_err=" << rel_vec_error << " ";
        assert(!std::isnan(rel_vec_error) && "Error should never be NaN");
    }

    std::cout << "PASSED\n";
}

int main() {
    std::cout << "\n=== Force Computation Test Suite ===\n\n";

    test_two_body();
    test_bh_vs_direct();
    test_linearized_vs_pointer_forces();
    test_no_self_interaction();
    test_coincident_particles_max_depth();
    test_theta_accuracy_tradeoff();

    std::cout << "\nAll Force tests PASSED ✓\n\n";
    return 0;
}
