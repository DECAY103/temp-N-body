#include "particle.hpp"
#include <random>
#include <cmath>
#include <algorithm>

namespace albh {

// ============================================================================
// Uniform distribution: particles randomly placed in [0, box_size]^3
// Cold start (zero velocity), equal mass
// ============================================================================
std::vector<Particle> generate_uniform(int N, double box_size, double total_mass, int seed) {
    std::vector<Particle> particles(N);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, box_size);

    double mass_per_particle = total_mass / N;

    for (int i = 0; i < N; ++i) {
        particles[i].x = dist(rng);
        particles[i].y = dist(rng);
        particles[i].z = dist(rng);
        particles[i].vx = 0.0;
        particles[i].vy = 0.0;
        particles[i].vz = 0.0;
        particles[i].ax = 0.0;
        particles[i].ay = 0.0;
        particles[i].az = 0.0;
        particles[i].mass = mass_per_particle;
        particles[i].morton_key = 0;
        particles[i].owner_rank = 0;
        particles[i].id = i;
    }

    return particles;
}

// ============================================================================
// Plummer model: density ∝ (1 + r²/a²)^(-5/2)
// Creates dense core near center with sparse halo — adversarial for load balance.
//
// Sampling uses the analytic inverse CDF for the Plummer sphere:
//   r = a / sqrt( U^(-2/3) - 1 )  where U ~ Uniform(0,1)
//
// Velocities are set using the Plummer escape velocity (simplified):
//   v_esc = sqrt(2 * G * M / sqrt(r^2 + a^2))
// Particles get a random fraction of v_esc for stability.
// ============================================================================
std::vector<Particle> generate_plummer(int N, double box_size, double total_mass,
                                        double plummer_a, int seed) {
    std::vector<Particle> particles(N);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uniform01(0.0, 1.0);

    double mass_per_particle = total_mass / N;
    double center = box_size / 2.0;  // place cluster at box center
    double a = plummer_a * box_size;  // scale radius in box units

    for (int i = 0; i < N; ++i) {
        // Sample radius from Plummer profile using inverse CDF
        double u = uniform01(rng);
        // Clamp to avoid numerical issues at extremes
        u = std::max(u, 1e-10);
        u = std::min(u, 1.0 - 1e-10);
        double r = a / std::sqrt(std::pow(u, -2.0/3.0) - 1.0);

        // Clamp radius to keep particles within box
        r = std::min(r, box_size * 0.45);

        // Random direction on unit sphere
        double costheta = 2.0 * uniform01(rng) - 1.0;
        double sintheta = std::sqrt(1.0 - costheta * costheta);
        double phi = 2.0 * M_PI * uniform01(rng);

        particles[i].x = center + r * sintheta * std::cos(phi);
        particles[i].y = center + r * sintheta * std::sin(phi);
        particles[i].z = center + r * costheta;

        // Clamp to box bounds
        particles[i].x = std::max(0.0, std::min(box_size, particles[i].x));
        particles[i].y = std::max(0.0, std::min(box_size, particles[i].y));
        particles[i].z = std::max(0.0, std::min(box_size, particles[i].z));

        // Small random velocity (fraction of "escape" velocity for stability)
        double v_scale = 0.1 * std::sqrt(total_mass / std::sqrt(r * r + a * a));
        double v_costheta = 2.0 * uniform01(rng) - 1.0;
        double v_sintheta = std::sqrt(1.0 - v_costheta * v_costheta);
        double v_phi = 2.0 * M_PI * uniform01(rng);

        particles[i].vx = v_scale * v_sintheta * std::cos(v_phi);
        particles[i].vy = v_scale * v_sintheta * std::sin(v_phi);
        particles[i].vz = v_scale * v_costheta;

        particles[i].ax = 0.0;
        particles[i].ay = 0.0;
        particles[i].az = 0.0;
        particles[i].mass = mass_per_particle;
        particles[i].morton_key = 0;
        particles[i].owner_rank = 0;
        particles[i].id = i;
    }

    return particles;
}

// Dispatch to the right generator based on distribution type
std::vector<Particle> generate_particles(Distribution dist, int N, double box_size,
                                          double total_mass, double plummer_a, int seed) {
    switch (dist) {
        case Distribution::PLUMMER:
            return generate_plummer(N, box_size, total_mass, plummer_a, seed);
        default:
            return generate_uniform(N, box_size, total_mass, seed);
    }
}

// Total kinetic energy: KE = sum(0.5 * m * |v|^2)
double compute_kinetic_energy(const std::vector<Particle>& particles) {
    double KE = 0.0;
    for (const auto& p : particles) {
        double v2 = p.vx * p.vx + p.vy * p.vy + p.vz * p.vz;
        KE += 0.5 * p.mass * v2;
    }
    return KE;
}

// Total gravitational potential energy: PE = sum_{i<j} -G*mi*mj / sqrt(rij^2 + eps^2)
// O(N^2) — for validation only
double compute_potential_energy(const std::vector<Particle>& particles,
                                double G, double softening) {
    double PE = 0.0;
    double eps2 = softening * softening;
    int N = static_cast<int>(particles.size());

    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            double dx = particles[i].x - particles[j].x;
            double dy = particles[i].y - particles[j].y;
            double dz = particles[i].z - particles[j].z;
            double r2 = dx * dx + dy * dy + dz * dz + eps2;
            double r = std::sqrt(r2);
            PE -= G * particles[i].mass * particles[j].mass / r;
        }
    }
    return PE;
}

} // namespace albh
