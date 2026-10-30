#include "communication.hpp"
#include "particle.hpp"
#include <iostream>
#include <cassert>
#include <vector>

using namespace albh;

// ============================================================================
// Test Suite: True Distributed MPI Communication (Milestone 5 & 6)
// ============================================================================

int main(int argc, char** argv) {
    MPIContext ctx;
    init_mpi(&argc, &argv, ctx);

    if (ctx.is_root()) {
        std::cout << "\n=== True Distributed MPI Communication Test Suite (Milestone 5) ===\n";
        std::cout << "Running across " << ctx.size << " MPI ranks\n\n";
    }

    // 1. Test Allgatherv Particle Exchange
    {
        if (ctx.is_root()) std::cout << "[TEST] Real MPI_Allgatherv Particle Exchange... " << std::flush;

        int local_count = 100 + ctx.rank * 20;
        std::vector<Particle> local_particles(local_count);
        for (int i = 0; i < local_count; ++i) {
            local_particles[i].x = ctx.rank * 10.0 + i;
            local_particles[i].owner_rank = ctx.rank;
        }

        std::vector<Particle> global_particles;
        double comm_time = exchange_all_particles(local_particles, global_particles, ctx, true);

        // Expected total count = sum_{r=0}^{P-1} (100 + r * 20)
        int expected_total = 0;
        for (int r = 0; r < ctx.size; ++r) {
            expected_total += (100 + r * 20);
        }

        assert(static_cast<int>(global_particles.size()) == expected_total);

        if (ctx.is_root()) {
            std::cout << "PASSED (Total particles: " << expected_total 
                      << ", Real Comm Time: " << comm_time * 1000 << " ms)\n";
        }
    }

    // 2. Test Energy Reduction
    {
        if (ctx.is_root()) std::cout << "[TEST] Real MPI_Reduce Energy Reduction... " << std::flush;

        double local_ke = 10.0 * (ctx.rank + 1);
        double local_pe = -5.0 * (ctx.rank + 1);
        double global_ke = 0.0, global_pe = 0.0;

        reduce_energy(local_ke, local_pe, global_ke, global_pe, ctx);

        if (ctx.is_root()) {
            double expected_ke = 0.0, expected_pe = 0.0;
            for (int r = 0; r < ctx.size; ++r) {
                expected_ke += 10.0 * (r + 1);
                expected_pe += -5.0 * (r + 1);
            }
            assert(std::abs(global_ke - expected_ke) < 1e-6);
            assert(std::abs(global_pe - expected_pe) < 1e-6);
            std::cout << "PASSED (Global KE: " << global_ke << ", PE: " << global_pe << ")\n";
        }
    }

    // 3. Test Particle Migration across ranks
    {
        if (ctx.is_root()) std::cout << "[TEST] Real MPI_Alltoallv Particle Migration... " << std::flush;

        // Rank r creates 10 particles targeting rank (r + 1) % size
        int n = 10;
        std::vector<Particle> local(n);
        std::vector<int> target_ranks(n);
        int target = (ctx.rank + 1) % ctx.size;

        for (int i = 0; i < n; ++i) {
            local[i].x = ctx.rank;
            local[i].owner_rank = ctx.rank;
            target_ranks[i] = target;
        }

        double mig_time = migrate_particles_mpi(local, target_ranks, ctx);

        // Every rank should now have received 10 particles from (rank - 1 + size) % size
        assert(static_cast<int>(local.size()) == 10);
        for (const auto& p : local) {
            assert(p.owner_rank == ctx.rank);
        }

        if (ctx.is_root()) {
            std::cout << "PASSED (Migration Time: " << mig_time * 1000 << " ms)\n";
            std::cout << "\nAll MPI Communication tests PASSED ✓\n\n";
        }
    }

    finalize_mpi();
    return 0;
}
