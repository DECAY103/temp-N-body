#include "octree.hpp"
#include "octree_pointer.hpp"
#include "particle.hpp"
#include "morton.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

// ============================================================================
// Test Suite: Octree Construction and Aggregation
//
// Tests:
//   1. Single particle tree
//   2. Multi-particle tree — correct aggregates
//   3. Linearized vs pointer-based tree give same aggregates
//   4. All particles accounted for (no loss during insertion)
//   5. Center-of-mass correctness
// ============================================================================

using namespace albh;

void test_single_particle() {
    std::cout << "[TEST] Single particle Octree... ";

    std::vector<Particle> particles(1);
    particles[0].x = 0.5;
    particles[0].y = 0.5;
    particles[0].z = 0.5;
    particles[0].mass = 1.0;

    Octree tree;
    tree.build(particles, 0.0, 0.0, 0.0, 1.0);

    assert(tree.num_nodes() >= 1);
    assert(std::abs(tree.root().total_mass - 1.0) < 1e-10);
    assert(std::abs(tree.root().com_x - 0.5) < 1e-10);
    assert(std::abs(tree.root().com_y - 0.5) < 1e-10);
    assert(std::abs(tree.root().com_z - 0.5) < 1e-10);

    std::cout << "PASSED\n";
}

void test_two_particles() {
    std::cout << "[TEST] Two particle Octree aggregates... ";

    std::vector<Particle> particles(2);
    particles[0].x = 0.25; particles[0].y = 0.25; particles[0].z = 0.25;
    particles[0].mass = 1.0;
    particles[1].x = 0.75; particles[1].y = 0.75; particles[1].z = 0.75;
    particles[1].mass = 3.0;

    Octree tree;
    tree.build(particles, 0.0, 0.0, 0.0, 1.0);

    // Root should have total mass = 4.0
    assert(std::abs(tree.root().total_mass - 4.0) < 1e-10);

    // Center of mass = (1*0.25 + 3*0.75) / 4 = 2.5/4 = 0.625
    double expected_com = (1.0 * 0.25 + 3.0 * 0.75) / 4.0;
    assert(std::abs(tree.root().com_x - expected_com) < 1e-10);
    assert(std::abs(tree.root().com_y - expected_com) < 1e-10);
    assert(std::abs(tree.root().com_z - expected_com) < 1e-10);
    (void)expected_com;

    // Root should have particle_count = 2
    assert(tree.root().particle_count == 2);

    std::cout << "PASSED\n";
}

void test_many_particles() {
    std::cout << "[TEST] Many particle Octree (N=1000)... ";

    auto particles = generate_uniform(1000, 1.0, 1.0, 42);

    Octree tree;
    tree.build(particles, 0.0, 0.0, 0.0, 1.0);

    // Root should contain all particles
    assert(tree.root().particle_count == 1000);

    // Total mass should be 1.0
    assert(std::abs(tree.root().total_mass - 1.0) < 1e-8);

    // Center of mass should be near (0.5, 0.5, 0.5) for uniform distribution
    assert(std::abs(tree.root().com_x - 0.5) < 0.1);
    assert(std::abs(tree.root().com_y - 0.5) < 0.1);
    assert(std::abs(tree.root().com_z - 0.5) < 0.1);

    std::cout << "PASSED (nodes=" << tree.num_nodes() << ")\n";
}

void test_plummer_particles() {
    std::cout << "[TEST] Plummer distribution Octree... ";

    auto particles = generate_plummer(500, 1.0, 1.0, 0.1, 42);

    Octree tree;
    tree.build(particles, 0.0, 0.0, 0.0, 1.0);

    assert(tree.root().particle_count == 500);
    assert(std::abs(tree.root().total_mass - 1.0) < 1e-8);

    // Plummer should be centered near (0.5, 0.5, 0.5)
    assert(std::abs(tree.root().com_x - 0.5) < 0.15);
    assert(std::abs(tree.root().com_y - 0.5) < 0.15);
    assert(std::abs(tree.root().com_z - 0.5) < 0.15);

    std::cout << "PASSED (nodes=" << tree.num_nodes() << ")\n";
}

void test_linearized_vs_pointer() {
    std::cout << "[TEST] Linearized vs Pointer Octree... ";

    auto particles = generate_uniform(200, 1.0, 1.0, 99);

    Octree lin_tree;
    lin_tree.build(particles, 0.0, 0.0, 0.0, 1.0);

    PointerOctree ptr_tree;
    ptr_tree.build(particles, 0.0, 0.0, 0.0, 1.0);

    // Both should have same root aggregates
    assert(std::abs(lin_tree.root().total_mass - ptr_tree.root_node->total_mass) < 1e-10);
    assert(std::abs(lin_tree.root().com_x - ptr_tree.root_node->com_x) < 1e-10);
    assert(std::abs(lin_tree.root().com_y - ptr_tree.root_node->com_y) < 1e-10);
    assert(std::abs(lin_tree.root().com_z - ptr_tree.root_node->com_z) < 1e-10);
    assert(lin_tree.root().particle_count == ptr_tree.root_node->particle_count);

    std::cout << "PASSED\n";
}

void test_coincident_particles() {
    std::cout << "[TEST] Coincident particles at MAX_DEPTH... ";

    std::vector<Particle> particles(5);
    for (int i = 0; i < 5; ++i) {
        particles[i].x = 0.5;
        particles[i].y = 0.5;
        particles[i].z = 0.5;
        particles[i].mass = 2.0;
        particles[i].id = i;
    }

    Octree lin_tree;
    lin_tree.build(particles, 0.0, 0.0, 0.0, 1.0);

    assert(lin_tree.root().particle_count == 5);
    assert(std::abs(lin_tree.root().total_mass - 10.0) < 1e-10);
    assert(std::abs(lin_tree.root().com_x - 0.5) < 1e-10);

    PointerOctree ptr_tree;
    ptr_tree.build(particles, 0.0, 0.0, 0.0, 1.0);

    assert(ptr_tree.root_node->particle_count == 5);
    assert(std::abs(ptr_tree.root_node->total_mass - 10.0) < 1e-10);
    assert(std::abs(ptr_tree.root_node->com_x - 0.5) < 1e-10);

    std::cout << "PASSED\n";
}

int main() {
    std::cout << "\n=== Octree Test Suite ===\n\n";

    test_single_particle();
    test_two_particles();
    test_many_particles();
    test_plummer_particles();
    test_linearized_vs_pointer();
    test_coincident_particles();

    std::cout << "\nAll Octree tests PASSED ✓\n\n";
    return 0;
}
