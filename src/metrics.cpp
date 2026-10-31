#include "metrics.hpp"
#include "config.hpp"
#include "particle.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <sys/stat.h>

namespace albh {

// ============================================================================
// Initialize the metrics logger — sets output path and writes CSV header
// ============================================================================
void MetricsLogger::init(const std::string& output_dir, const SimConfig& config) {
    // Create output directory if it doesn't exist
    mkdir(output_dir.c_str(), 0755);

    // Build filename from config
    std::ostringstream fname;
    fname << output_dir << "/metrics"
          << "_N" << config.N
          << "_" << config.mode
          << "_theta" << config.theta
          << "_t" << config.num_threads
          << "_seed" << config.seed
          << ".csv";
    output_path_ = fname.str();
    initialized_ = true;
    buffer_.clear();
}

// Record a timestep's metrics into the buffer
void MetricsLogger::log_timestep(const TimestepMetrics& m) {
    buffer_.push_back(m);
}

// Write all buffered metrics to CSV
void MetricsLogger::flush() {
    if (!initialized_ || buffer_.empty()) return;

    std::ofstream out(output_path_);
    if (!out.is_open()) {
        std::cerr << "ERROR: Cannot open metrics file: " << output_path_ << std::endl;
        return;
    }

    // CSV header
    out << "timestep,rank,local_particles,tree_build_time,force_compute_time,"
        << "comm_time,rebalance_time,total_step_time,imbalance_ratio,"
        << "particles_migrated,kinetic_energy,potential_energy\n";

    // Data rows
    for (const auto& m : buffer_) {
        out << m.timestep << ","
            << m.rank << ","
            << m.local_particle_count << ","
            << std::fixed << std::setprecision(6)
            << m.tree_build_time << ","
            << m.force_compute_time << ","
            << m.comm_time << ","
            << m.rebalance_time << ","
            << m.total_step_time << ","
            << std::setprecision(4)
            << m.imbalance_ratio << ","
            << m.particles_migrated << ","
            << std::setprecision(10)
            << m.kinetic_energy << ","
            << m.potential_energy << "\n";
    }

    out.close();
    std::cout << "[Metrics] Written to " << output_path_ << " (" << buffer_.size() << " entries)\n";
}

// Print a console summary for one timestep
void MetricsLogger::print_timestep_summary(const TimestepMetrics& m,
                                            int total_steps, int total_N,
                                            double theta) {
    std::cout << "[Timestep " << std::setw(4) << std::setfill('0') << m.timestep
              << "/" << total_steps << "]"
              << " N=" << total_N
              << " | θ=" << std::fixed << std::setprecision(2) << theta
              << "\n  Force: " << std::setprecision(4) << m.force_compute_time << "s"
              << "  Tree: " << m.tree_build_time << "s"
              << "  Comm: " << m.comm_time << "s"
              << "  Rebal: " << m.rebalance_time << "s"
              << "\n  Imbalance: " << std::setprecision(2) << m.imbalance_ratio
              << "  Migrated: " << m.particles_migrated
              << "  Total: " << std::setprecision(4) << m.total_step_time << "s"
              << "\n  KE: " << std::setprecision(8) << m.kinetic_energy
              << "  PE: " << m.potential_energy
              << "  E: " << (m.kinetic_energy + m.potential_energy)
              << "\n";
}

void MetricsLogger::finalize() {
    flush();
    buffer_.clear();
    initialized_ = false;
}

// ============================================================================
// Dump particle state to binary file for visualization
// Format: [int32 N] [N * Particle data as doubles + int]
// ============================================================================
void dump_particle_state(const std::string& output_dir, int timestep,
                          const std::vector<Particle>& particles) {
    std::string particles_dir = output_dir + "/particles";
    mkdir(particles_dir.c_str(), 0755);

    std::ostringstream fname;
    fname << particles_dir << "/step_" << std::setw(5) << std::setfill('0')
          << timestep << ".bin";

    std::ofstream out(fname.str(), std::ios::binary);
    if (!out.is_open()) return;

    int32_t N = static_cast<int32_t>(particles.size());
    out.write(reinterpret_cast<const char*>(&N), sizeof(N));

    for (const auto& p : particles) {
        // Write: x, y, z, vx, vy, vz, mass, owner_rank
        out.write(reinterpret_cast<const char*>(&p.x), sizeof(double));
        out.write(reinterpret_cast<const char*>(&p.y), sizeof(double));
        out.write(reinterpret_cast<const char*>(&p.z), sizeof(double));
        out.write(reinterpret_cast<const char*>(&p.vx), sizeof(double));
        out.write(reinterpret_cast<const char*>(&p.vy), sizeof(double));
        out.write(reinterpret_cast<const char*>(&p.vz), sizeof(double));
        out.write(reinterpret_cast<const char*>(&p.mass), sizeof(double));
        int32_t rank = p.owner_rank;
        out.write(reinterpret_cast<const char*>(&rank), sizeof(int32_t));
    }
}

} // namespace albh
