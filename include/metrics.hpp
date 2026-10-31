#ifndef ALBH_METRICS_HPP
#define ALBH_METRICS_HPP

#include <string>
#include <vector>
#include <fstream>

namespace albh {

struct SimConfig;

// ============================================================================
// Metrics: Per-timestep performance measurement and CSV logging
//
// Records timing for each phase of the simulation pipeline:
//   - Tree build time
//   - Force computation time
//   - Communication time (MPI)
//   - Rebalancing time
//   - Total timestep time
//   - Load imbalance ratio
//   - Particles migrated (if rebalancing occurred)
// ============================================================================

struct TimestepMetrics {
    int timestep;
    double tree_build_time;
    double force_compute_time;
    double comm_time;
    double rebalance_time;
    double total_step_time;
    double imbalance_ratio;     // max_work / min_work (1.0 = perfect)
    int particles_migrated;     // particles moved between ranks
    double kinetic_energy;      // for energy conservation tracking
    double potential_energy;
    int rank;                   // MPI rank that recorded this
    int local_particle_count;   // particles on this rank
};

class MetricsLogger {
public:
    // Initialize logger with output directory and config
    void init(const std::string& output_dir, const SimConfig& config);

    // Record a timestep's metrics
    void log_timestep(const TimestepMetrics& m);

    // Write all accumulated metrics to CSV file
    void flush();

    // Print console summary for a timestep (if verbose)
    static void print_timestep_summary(const TimestepMetrics& m,
                                        int total_steps, int total_N,
                                        double theta = 0.5);

    // Finalize: flush and close files
    void finalize();

private:
    std::string output_path_;
    std::vector<TimestepMetrics> buffer_;
    bool initialized_ = false;
};

// Dump particle state to binary file (for visualization)
// Format: N (int32), then N * (x, y, z, vx, vy, vz, mass, owner_rank)
void dump_particle_state(const std::string& output_dir, int timestep,
                          const std::vector<struct Particle>& particles);

} // namespace albh

#endif // ALBH_METRICS_HPP
