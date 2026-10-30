#ifndef ALBH_COMMUNICATION_HPP
#define ALBH_COMMUNICATION_HPP

#include "particle.hpp"
#include <vector>

#ifdef ENABLE_MPI
#include <mpi.h>
#endif

namespace albh {

// ============================================================================
// MPI Execution Context & Distributed Communication Engine (Milestone 3, 5, 6)
// ============================================================================

struct MPIContext {
    int rank = 0;
    int size = 1;
    bool is_root() const { return rank == 0; }
};

// Initialize and finalize MPI environment
void init_mpi(int* argc, char*** argv, MPIContext& ctx);
void finalize_mpi();

// Exchange particles across all ranks using real MPI_Allgatherv (or nonblocking MPI_Iallgatherv).
// Returns the real communication time measured via MPI_Wtime().
double exchange_all_particles(const std::vector<Particle>& local_particles,
                              std::vector<Particle>& global_particles,
                              const MPIContext& ctx,
                              bool use_nonblocking = true);

// Reduce local kinetic & potential energies to Rank 0 using MPI_Reduce
void reduce_energy(double local_ke, double local_pe,
                   double& global_ke, double& global_pe,
                   const MPIContext& ctx);

// Exchange per-rank workloads (timings or counts) to compute load imbalance globally
double gather_workloads(double local_workload,
                        std::vector<double>& all_workloads,
                        const MPIContext& ctx);

// Migrate particles across ranks when dynamic rebalancing triggers.
// Uses MPI_Alltoallv to distribute migrating particles to their new owners.
double migrate_particles_mpi(std::vector<Particle>& local_particles,
                             const std::vector<int>& target_ranks,
                             const MPIContext& ctx);

} // namespace albh

#endif // ALBH_COMMUNICATION_HPP
