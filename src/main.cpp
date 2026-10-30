#include "particle.hpp"
#include "config.hpp"
#include "communication.hpp"
#include <iostream>
#include <vector>

// Forward declaration of simulation drivers
namespace albh {
    void run_serial_simulation(std::vector<Particle>& particles, const SimConfig& config);
    void run_parallel_simulation(std::vector<Particle>& particles, const SimConfig& config, const MPIContext& ctx);
}

// ============================================================================
// Main Entry Point for AL-BH (Hybrid MPI + OpenMP)
// ============================================================================
int main(int argc, char** argv) {
    albh::MPIContext ctx;
    albh::init_mpi(&argc, &argv, ctx);

    albh::SimConfig config;

    // Parse command-line arguments
    if (!albh::parse_args(config, argc, argv)) {
        albh::finalize_mpi();
        return 0;  // --help was requested
    }

    // Only root rank prints banner and config
    if (ctx.is_root()) {
        albh::print_config(config);
    }

    // Generate initial particles on Root Rank
    std::vector<albh::Particle> particles;
    if (ctx.is_root()) {
        albh::Distribution dist = (config.distribution == 1)
            ? albh::Distribution::PLUMMER
            : albh::Distribution::UNIFORM;

        particles = albh::generate_particles(
            dist, config.N, config.box_size, config.total_mass, config.seed);

        std::cout << "Generated " << particles.size() << " particles ("
                  << (config.distribution == 0 ? "uniform" : "plummer") << ") on Root Process\n";
    }

    // Dispatch to appropriate simulation mode
    if (config.mode == "serial") {
        if (ctx.is_root()) {
            albh::run_serial_simulation(particles, config);
        }
    } else if (config.mode == "mpi_static" || config.mode == "mpi_morton" || config.mode == "full") {
        albh::run_parallel_simulation(particles, config, ctx);
    } else {
        if (ctx.is_root()) {
            std::cerr << "ERROR: Unknown mode '" << config.mode << "'\n"
                      << "Available: serial, mpi_static, mpi_morton, full\n";
        }
        albh::finalize_mpi();
        return 1;
    }

    albh::finalize_mpi();
    return 0;
}
