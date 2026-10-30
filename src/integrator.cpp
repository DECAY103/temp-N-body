#include "integrator.hpp"
#include "particle.hpp"

namespace albh {

// ============================================================================
// Leapfrog Kick-Drift-Kick Integration
//
// This is a symplectic integrator that preserves the Hamiltonian structure,
// giving much better long-term energy conservation than simple Euler.
//
// Usage pattern per timestep:
//   1. leapfrog_kick_half(particles, dt)   — v += a * dt/2
//   2. leapfrog_drift(particles, dt)       — x += v * dt
//   3. [recompute forces → updates a]
//   4. leapfrog_kick_final(particles, dt)  — v += a * dt/2
// ============================================================================

// Half-kick: update velocities using current accelerations
void leapfrog_kick_half(std::vector<Particle>& particles, double dt) {
    double half_dt = 0.5 * dt;
    for (auto& p : particles) {
        p.vx += p.ax * half_dt;
        p.vy += p.ay * half_dt;
        p.vz += p.az * half_dt;
    }
}

// Drift: update positions using current velocities
void leapfrog_drift(std::vector<Particle>& particles, double dt) {
    for (auto& p : particles) {
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        p.z += p.vz * dt;
    }
}

// Final half-kick: same operation as kick_half, called after force recomputation
void leapfrog_kick_final(std::vector<Particle>& particles, double dt) {
    leapfrog_kick_half(particles, dt);
}

} // namespace albh
