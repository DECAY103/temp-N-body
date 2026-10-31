#include "config.hpp"
#include <iostream>
#include <cstring>
#include <cstdlib>

namespace albh {

// ============================================================================
// CLI Argument Parsing
//
// Simple hand-rolled parser. Format: --key value
// Example: ./al_bh --N 10000 --theta 0.5 --steps 100 --mode serial
// ============================================================================
bool parse_args(SimConfig& config, int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            std::cout << "AL-BH: Adaptive Linearized Barnes-Hut N-Body Simulator\n\n"
                      << "Usage: ./al_bh [options]\n\n"
                      << "Options:\n"
                      << "  --N <int>          Number of particles (default: 10000)\n"
                      << "  --steps <int>      Number of timesteps (default: 100)\n"
                      << "  --dt <float>       Timestep size (default: 0.001)\n"
                      << "  --theta <float>    BH opening angle (default: 0.5)\n"
                      << "  --softening <float> Gravitational softening (default: 0.01)\n"
                      << "  --G <float>        Gravitational constant (default: 1.0)\n"
                      << "  --box <float>      Simulation box size (default: 1.0)\n"
                      << "  --dist <0|1>       Distribution: 0=uniform, 1=plummer (default: 0)\n"
                      << "  --seed <int>       Random seed (default: 42)\n"
                      << "  --mode <string>    Execution mode: serial|mpi_static|mpi_morton|full (default: serial)\n"
                      << "  --threads <int>    OpenMP threads per rank (default: 1)\n"
                      << "  --threshold <float> Rebalance threshold (default: 1.2)\n"
                      << "  --nonblocking <0|1> Use non-blocking MPI (default: 1)\n"
                      << "  --output <dir>     Output directory (default: results/raw)\n"
                      << "  --dump <0|1>       Dump particle state for viz (default: 0)\n"
                      << "  --dump-interval <int> Dump every N steps (default: 1)\n"
                      << "  --verbose <0|1>    Print per-timestep output (default: 1)\n"
                      << "  --linearized <0|1> Use linearized octree (default: 1)\n"
                      << "  --mass <float>     Total mass of particles (default: 1.0)\n"
                      << "  --plummer-a <float> Plummer scale radius (default: 0.1)\n"
                      << std::endl;
            return false;
        }

        // Need a value after each flag
        if (i + 1 >= argc) {
            std::cerr << "ERROR: Missing value for " << arg << std::endl;
            return false;
        }

        std::string val = argv[++i];

        if (arg == "--N")              config.N = std::atoi(val.c_str());
        else if (arg == "--steps")     config.timesteps = std::atoi(val.c_str());
        else if (arg == "--dt")        config.dt = std::atof(val.c_str());
        else if (arg == "--theta")     config.theta = std::atof(val.c_str());
        else if (arg == "--softening") config.softening = std::atof(val.c_str());
        else if (arg == "--G")         config.G = std::atof(val.c_str());
        else if (arg == "--box")       config.box_size = std::atof(val.c_str());
        else if (arg == "--mass")      config.total_mass = std::atof(val.c_str());
        else if (arg == "--dist")      config.distribution = std::atoi(val.c_str());
        else if (arg == "--seed")      config.seed = std::atoi(val.c_str());
        else if (arg == "--mode")      config.mode = val;
        else if (arg == "--threads")   config.num_threads = std::atoi(val.c_str());
        else if (arg == "--threshold") config.rebalance_threshold = std::atof(val.c_str());
        else if (arg == "--nonblocking") config.use_nonblocking = (std::atoi(val.c_str()) != 0);
        else if (arg == "--output")    config.output_dir = val;
        else if (arg == "--dump")      config.dump_particles = (std::atoi(val.c_str()) != 0);
        else if (arg == "--dump-interval") config.dump_interval = std::atoi(val.c_str());
        else if (arg == "--verbose")   config.verbose = (std::atoi(val.c_str()) != 0);
        else if (arg == "--linearized") config.use_linearized_tree = (std::atoi(val.c_str()) != 0);
        else if (arg == "--plummer-a") config.plummer_a = std::atof(val.c_str());
        else {
            std::cerr << "ERROR: Unknown command-line argument: " << arg << std::endl;
            return false;
        }
    }

    // --- Validation ---
    if (config.N <= 0) {
        std::cerr << "ERROR: Invalid N (" << config.N << "). Must be > 0.\n";
        return false;
    }
    if (config.timesteps <= 0) {
        std::cerr << "ERROR: Invalid timesteps (" << config.timesteps << "). Must be > 0.\n";
        return false;
    }
    if (config.dt <= 0.0) {
        std::cerr << "ERROR: Invalid dt (" << config.dt << "). Must be > 0.\n";
        return false;
    }
    if (config.theta < 0.0 || config.theta > 2.0) {
        std::cerr << "ERROR: Invalid theta (" << config.theta << "). Must be in [0.0, 2.0].\n";
        return false;
    }
    if (config.softening < 0.0) {
        std::cerr << "ERROR: Invalid softening (" << config.softening << "). Must be >= 0.\n";
        return false;
    }
    if (config.box_size <= 0.0) {
        std::cerr << "ERROR: Invalid box_size (" << config.box_size << "). Must be > 0.\n";
        return false;
    }
    if (config.total_mass <= 0.0) {
        std::cerr << "ERROR: Invalid total_mass (" << config.total_mass << "). Must be > 0.\n";
        return false;
    }
    if (config.num_threads <= 0) {
        std::cerr << "ERROR: Invalid num_threads (" << config.num_threads << "). Must be >= 1.\n";
        return false;
    }
    if (config.mode != "serial" && config.mode != "mpi_static" &&
        config.mode != "mpi_morton" && config.mode != "full") {
        std::cerr << "ERROR: Invalid mode '" << config.mode
                  << "'. Must be serial, mpi_static, mpi_morton, or full.\n";
        return false;
    }
    if (config.distribution != 0 && config.distribution != 1) {
        std::cerr << "ERROR: Invalid distribution (" << config.distribution
                  << "). Must be 0 (uniform) or 1 (plummer).\n";
        return false;
    }
    if (config.plummer_a <= 0.0) {
        std::cerr << "ERROR: Invalid plummer_a (" << config.plummer_a << "). Must be > 0.\n";
        return false;
    }

    return true;
}

// Print configuration summary
void print_config(const SimConfig& config) {
    std::cout << "=== AL-BH Configuration ===\n"
              << "  Particles:    " << config.N << "\n"
              << "  Total Mass:   " << config.total_mass << "\n"
              << "  Distribution: " << (config.distribution == 0 ? "uniform" : "plummer") << "\n"
              << "  Plummer a:    " << config.plummer_a << "\n"
              << "  Timesteps:    " << config.timesteps << "\n"
              << "  dt:           " << config.dt << "\n"
              << "  theta:        " << config.theta << "\n"
              << "  softening:    " << config.softening << "\n"
              << "  G:            " << config.G << "\n"
              << "  Box size:     " << config.box_size << "\n"
              << "  Seed:         " << config.seed << "\n"
              << "  Mode:         " << config.mode << "\n"
              << "  Threads:      " << config.num_threads << "\n"
              << "  Rebalance:    " << config.rebalance_threshold << "\n"
              << "  Non-blocking: " << (config.use_nonblocking ? "yes" : "no") << "\n"
              << "  Linearized:   " << (config.use_linearized_tree ? "yes" : "no") << "\n"
              << "  Output:       " << config.output_dir << "\n"
              << "  Dump particles: " << (config.dump_particles ? "yes" : "no") << "\n"
              << "===========================\n";
}

} // namespace albh
