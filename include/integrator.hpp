#ifndef ALBH_INTEGRATOR_HPP
#define ALBH_INTEGRATOR_HPP

#include <vector>

namespace albh {

struct Particle;

// ============================================================================
// Leapfrog (Kick-Drift-Kick) Time Integrator
//
// Symplectic integrator with good energy conservation for Hamiltonian systems.
//
// Algorithm per timestep:
//   1. Half-kick:  v(t + dt/2) = v(t) + a(t) * dt/2
//   2. Drift:      x(t + dt)   = x(t) + v(t + dt/2) * dt
//   3. Compute new accelerations a(t + dt)  [done externally]
//   4. Half-kick:  v(t + dt)   = v(t + dt/2) + a(t + dt) * dt/2
//
// The force computation (step 3) happens between kick_half and kick_final.
// ============================================================================

// Step 1: Half-kick — update velocities by half timestep using current accelerations
void leapfrog_kick_half(std::vector<Particle>& particles, double dt);

// Step 2: Drift — update positions using current velocities
void leapfrog_drift(std::vector<Particle>& particles, double dt);

// Step 4: Final half-kick — complete velocity update with new accelerations
// (Same function as kick_half, called after force recomputation)
void leapfrog_kick_final(std::vector<Particle>& particles, double dt);

} // namespace albh

#endif // ALBH_INTEGRATOR_HPP
