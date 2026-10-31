#ifndef ALBH_CONFIG_HPP
#define ALBH_CONFIG_HPP

#include <string>

namespace albh {

// ============================================================================
// SimConfig: Centralized simulation parameters
// All tunable values live here. Defaults are documented.
// ============================================================================
struct SimConfig {
    // --- Particle setup ---
    int N              = 10000;       // number of particles
    int distribution   = 0;          // 0 = uniform, 1 = plummer
    int seed           = 42;         // random seed for reproducibility
    double total_mass  = 1.0;        // total mass of all particles combined

    // --- Simulation ---
    int timesteps      = 100;        // number of simulation steps
    double dt          = 0.001;      // time step size
    double theta       = 0.5;        // Barnes-Hut opening angle
    double softening   = 0.01;       // gravitational softening (ε)
    double G           = 1.0;        // gravitational constant
    double box_size    = 1.0;        // simulation domain [0, box_size]^3

    // --- Plummer model ---
    double plummer_a   = 0.1;        // Plummer scale radius (fraction of box_size)

    // --- MPI / Parallelism ---
    int num_threads    = 1;          // OpenMP threads per rank

    // --- Load balancing ---
    double rebalance_threshold = 1.2; // max/min work ratio before rebalance

    // --- Execution mode ---
    // "serial"         = single-process serial BH
    // "mpi_static"     = MPI with static equal-volume partitions
    // "mpi_morton"      = MPI with Morton decomposition, no rebalance
    // "full"           = full AL-BH (Morton + rebalance + non-blocking)
    // "full_static"    = static initial partition + adaptive rebalancing
    //                    (benchmark mode for demonstrating rebalancing)
    std::string mode   = "serial";

    // --- Communication ---
    bool use_nonblocking = true;     // true = MPI_Isend/Irecv, false = blocking

    // --- Output ---
    std::string output_dir   = "results/raw";
    bool dump_particles      = false; // dump particle state each timestep (for viz)
    int dump_interval        = 1;     // dump every N timesteps
    bool verbose             = true;  // print per-timestep console output

    // --- Octree type (for cache comparison) ---
    bool use_linearized_tree = true;  // true = array-based, false = pointer-based
};

// Parse command-line arguments into SimConfig
// Returns true on success, false if --help was requested
bool parse_args(SimConfig& config, int argc, char** argv);

// Print configuration summary to stdout
void print_config(const SimConfig& config);

} // namespace albh

#endif // ALBH_CONFIG_HPP
