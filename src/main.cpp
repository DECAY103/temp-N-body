#include "particle.hpp"
#include "config.hpp"
#include <iostream>
#include <vector>

// Forward declaration of simulation drivers
namespace albh {
    void run_serial_simulation(std::vector<Particle>& particles, const SimConfig& config);
    void run_parallel_simulation(std::vector<Particle>& particles, const SimConfig& config);
}

// ============================================================================
// Main Entry Point for AL-BH
//
// Parses CLI arguments, generates initial particle distribution,
// and dispatches to the appropriate simulation mode.
// ============================================================================
int main(int argc, char** argv) {
    albh::SimConfig config;

    // Parse command-line arguments
    if (!albh::parse_args(config, argc, argv)) {
        return 0;  // --help was requested
    }

    // Print configuration
    albh::print_config(config);

    // Generate initial particles
    albh::Distribution dist = (config.distribution == 1)
        ? albh::Distribution::PLUMMER
        : albh::Distribution::UNIFORM;

    std::vector<albh::Particle> particles = albh::generate_particles(
        dist, config.N, config.box_size, config.total_mass, config.seed);

    std::cout << "Generated " << particles.size() << " particles ("
              << (config.distribution == 0 ? "uniform" : "plummer") << ")\n";

    // Dispatch to appropriate simulation mode
    if (config.mode == "serial") {
        albh::run_serial_simulation(particles, config);
    } else if (config.mode == "mpi_static" || config.mode == "mpi_morton" || config.mode == "full") {
        albh::run_parallel_simulation(particles, config);
    } else {
        std::cerr << "ERROR: Unknown mode '" << config.mode << "'\n"
                  << "Available: serial, mpi_static, mpi_morton, full\n";
        return 1;
    }

    return 0;
}
