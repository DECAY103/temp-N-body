#ifndef ALBH_PARTICLE_HPP
#define ALBH_PARTICLE_HPP

#include <cstdint>
#include <vector>
#include <string>

namespace albh {

// ============================================================================
// Particle: Core data structure for N-body simulation
// Layout: SoA-friendly struct with position, velocity, acceleration, mass,
//         Morton key for spatial ordering, and MPI rank ownership.
// ============================================================================
struct Particle {
    double x, y, z;       // position in [0, box_size]^3
    double vx, vy, vz;    // velocity
    double ax, ay, az;    // acceleration (computed each timestep)
    double mass;          // particle mass (typically 1.0 for uniform)
    uint64_t morton_key;  // Z-order key for spatial sorting
    int owner_rank;       // MPI rank that owns this particle
    int id;               // unique global particle ID for self-exclusion and tracking

    Particle()
        : x(0), y(0), z(0),
          vx(0), vy(0), vz(0),
          ax(0), ay(0), az(0),
          mass(1.0), morton_key(0), owner_rank(0), id(0) {}
};

// Distribution types for particle generation
enum class Distribution {
    UNIFORM = 0,   // Random uniform in [0, box_size]^3
    PLUMMER = 1    // Plummer model: dense core + sparse halo
};

// ----------------------------------------------------------------------------
// Generate initial particle distributions
// All generators are deterministic given a seed.
// ----------------------------------------------------------------------------

// Generate N particles with uniform random distribution in [0, box_size]^3
// Mass = total_mass / N for each particle
// Velocity = 0 (cold start)
std::vector<Particle> generate_uniform(int N, double box_size, double total_mass, int seed);

// Generate N particles following Plummer model
// Dense core near center, sparse halo extending outward
// Plummer scale radius 'a' controls concentration (default: box_size * 0.1)
std::vector<Particle> generate_plummer(int N, double box_size, double total_mass,
                                        double plummer_a, int seed);

// Convenience wrapper that dispatches based on distribution type
std::vector<Particle> generate_particles(Distribution dist, int N, double box_size,
                                          double total_mass, double plummer_a, int seed);

// Compute total kinetic energy: sum(0.5 * m * v^2)
double compute_kinetic_energy(const std::vector<Particle>& particles);

// Compute total gravitational potential energy: sum(-G*mi*mj / |ri-rj|)
// Uses softening to avoid singularities. O(N^2) — use only for validation.
double compute_potential_energy(const std::vector<Particle>& particles,
                                double G, double softening);

} // namespace albh

#endif // ALBH_PARTICLE_HPP
