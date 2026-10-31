#include "particle.hpp"
#include "config.hpp"
#include "morton.hpp"
#include "octree.hpp"
#include "octree_pointer.hpp"
#include "force.hpp"
#include "integrator.hpp"
#include "metrics.hpp"
#include "partition.hpp"
#include "load_balancer.hpp"
#include "communication.hpp"
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <iomanip>

namespace albh {

// ============================================================================
// Serial Simulation Driver
// ============================================================================
void run_serial_simulation(std::vector<Particle>& particles, const SimConfig& config) {
    auto wall_clock = []() {
        return std::chrono::high_resolution_clock::now();
    };
    auto elapsed = [](auto start, auto end) {
        return std::chrono::duration<double>(end - start).count();
    };

    MetricsLogger logger;
    logger.init(config.output_dir, config);

    int N = static_cast<int>(particles.size());
    Octree lin_tree;
    PointerOctree ptr_tree;

    std::cout << "\n[Serial Simulation] Starting " << config.timesteps
              << " timesteps with " << N << " particles\n\n";

    // Initial force computation
    compute_and_sort_by_morton(particles, config.box_size);
    if (config.use_linearized_tree) {
        lin_tree.build(particles, 0.0, 0.0, 0.0, config.box_size);
        compute_forces_linearized(particles, lin_tree,
                                   config.G, config.theta, config.softening,
                                   config.num_threads);
    } else {
        ptr_tree.build(particles, 0.0, 0.0, 0.0, config.box_size);
        compute_forces_pointer(particles, ptr_tree,
                                config.G, config.theta, config.softening,
                                config.num_threads);
    }

    if (config.dump_particles) {
        dump_particle_state(config.output_dir, 0, particles);
    }

    for (int step = 1; step <= config.timesteps; ++step) {
        auto t_step_start = wall_clock();

        leapfrog_kick_half(particles, config.dt);
        leapfrog_drift(particles, config.dt);

        auto t_sort_start = wall_clock();
        compute_and_sort_by_morton(particles, config.box_size);
        auto t_sort_end = wall_clock();

        auto t_tree_start = wall_clock();
        if (config.use_linearized_tree) {
            lin_tree.build(particles, 0.0, 0.0, 0.0, config.box_size);
        } else {
            ptr_tree.build(particles, 0.0, 0.0, 0.0, config.box_size);
        }
        auto t_tree_end = wall_clock();

        auto t_force_start = wall_clock();
        if (config.use_linearized_tree) {
            compute_forces_linearized(particles, lin_tree,
                                       config.G, config.theta, config.softening,
                                       config.num_threads);
        } else {
            compute_forces_pointer(particles, ptr_tree,
                                    config.G, config.theta, config.softening,
                                    config.num_threads);
        }
        auto t_force_end = wall_clock();

        leapfrog_kick_half(particles, config.dt);
        auto t_step_end = wall_clock();

        TimestepMetrics m;
        m.timestep = step;
        m.rank = 0;
        m.local_particle_count = N;
        m.tree_build_time = elapsed(t_tree_start, t_tree_end) + elapsed(t_sort_start, t_sort_end);
        m.force_compute_time = elapsed(t_force_start, t_force_end);
        m.comm_time = 0.0;
        m.rebalance_time = 0.0;
        m.total_step_time = elapsed(t_step_start, t_step_end);
        m.imbalance_ratio = 1.0;
        m.particles_migrated = 0;

        m.kinetic_energy = compute_kinetic_energy(particles);
        if (N <= 5000) {
            m.potential_energy = compute_potential_energy(particles, config.G, config.softening);
        } else {
            // Mark as NaN instead of misleading 0.0
            m.potential_energy = std::numeric_limits<double>::quiet_NaN();
        }

        logger.log_timestep(m);
        if (config.verbose) {
            MetricsLogger::print_timestep_summary(m, config.timesteps, N, config.theta);
        }
        if (config.dump_particles && (step % config.dump_interval == 0)) {
            dump_particle_state(config.output_dir, step, particles);
        }
    }

    logger.finalize();
    std::cout << "\n[Serial Simulation] Complete.\n";
}

// ============================================================================
// True Distributed Parallel Simulation Driver (MPI + OpenMP Hybrid)
//
// Modes:
//   mpi_static  — static geometric partitioning (baseline, no rebalance)
//   mpi_morton  — Morton-curve partitioning (no rebalance baseline)
//   full        — Morton + adaptive rebalancing + non-blocking overlap
// ============================================================================
void run_parallel_simulation(std::vector<Particle>& initial_particles,
                            const SimConfig& config,
                            const MPIContext& ctx) {
    auto wall_clock = []() {
        return std::chrono::high_resolution_clock::now();
    };
    auto elapsed = [](auto start, auto end) {
        return std::chrono::duration<double>(end - start).count();
    };

    MetricsLogger logger;
    if (ctx.is_root()) {
        logger.init(config.output_dir, config);
        std::cout << "\n[Parallel Simulation (" << config.mode << ")] Starting "
                  << config.timesteps << " timesteps across " << ctx.size
                  << " real MPI processes (threads/rank: " << config.num_threads << ")\n\n";
    }

    int N_total = config.N;
    bool is_static = (config.mode == "mpi_static");
    // ONLY "full" mode enables adaptive rebalancing
    // mpi_morton is now a true no-rebalance Morton baseline
    bool enable_rebalance = (config.mode == "full");

    // 1. Initial Domain Decomposition & Scattering
    std::vector<Particle> local_particles;
    std::vector<PartitionInfo> partitions;

    if (ctx.size > 1) {
        // Multi-process distributed MPI mode
        if (ctx.is_root()) {
            if (is_static) {
                // FIX: Sort particles by x-slab BEFORE partitioning
                // so that MPI_Scatterv sends correct spatial regions
                sort_by_geometric_slabs(initial_particles, ctx.size, config.box_size);
                partitions = partition_static_geometric(initial_particles, ctx.size, config.box_size);
                assign_ownership(initial_particles, partitions);
            } else {
                compute_and_sort_by_morton(initial_particles, config.box_size);
                partitions = partition_equal_count(initial_particles, ctx.size);
                assign_ownership(initial_particles, partitions);
            }
        }

        // Broadcast initial partitions count
        std::vector<int> counts(ctx.size);
        if (ctx.is_root()) {
            for (int r = 0; r < ctx.size; ++r) counts[r] = partitions[r].particle_count;
        }
        #ifdef ENABLE_MPI
        MPI_Bcast(counts.data(), ctx.size, MPI_INT, 0, MPI_COMM_WORLD);
        #endif

        int my_count = counts[ctx.rank];
        local_particles.resize(my_count);

        #ifdef ENABLE_MPI
        std::vector<int> byte_counts(ctx.size);
        std::vector<int> displs(ctx.size);
        int offset = 0;
        for (int r = 0; r < ctx.size; ++r) {
            byte_counts[r] = counts[r] * sizeof(Particle);
            displs[r] = offset * sizeof(Particle);
            offset += counts[r];
        }

        const void* sendbuf = ctx.is_root() ? initial_particles.data() : nullptr;
        MPI_Scatterv(sendbuf, byte_counts.data(), displs.data(), MPI_BYTE,
                     local_particles.data(), my_count * sizeof(Particle), MPI_BYTE,
                     0, MPI_COMM_WORLD);
        #endif
    } else {
        // Single process fallback with simulated ranks
        local_particles = std::move(initial_particles);
        int sim_ranks = config.num_threads > 1 ? config.num_threads : 4;
        if (is_static) {
            sort_by_geometric_slabs(local_particles, sim_ranks, config.box_size);
            partitions = partition_static_geometric(local_particles, sim_ranks, config.box_size);
        } else {
            compute_and_sort_by_morton(local_particles, config.box_size);
            partitions = partition_equal_count(local_particles, sim_ranks);
        }
        assign_ownership(local_particles, partitions);
    }

    LoadBalancer lb(ctx.size > 1 ? ctx.size : 4, config.rebalance_threshold);
    Octree lin_tree;
    PointerOctree ptr_tree;

    // Initial exchange and force computation (blocking for initial setup)
    std::vector<Particle> global_particles;
    exchange_all_particles(local_particles, global_particles, ctx, false);

    if (config.use_linearized_tree) {
        lin_tree.build(global_particles, 0.0, 0.0, 0.0, config.box_size);
        compute_forces_linearized(local_particles, lin_tree,
                                   config.G, config.theta, config.softening,
                                   config.num_threads);
    } else {
        ptr_tree.build(global_particles, 0.0, 0.0, 0.0, config.box_size);
        compute_forces_pointer(local_particles, ptr_tree,
                                config.G, config.theta, config.softening,
                                config.num_threads);
    }

    if (ctx.is_root() && config.dump_particles) {
        dump_particle_state(config.output_dir, 0, global_particles);
    }

    // --- Main Simulation Loop ---
    for (int step = 1; step <= config.timesteps; ++step) {
        auto t_step_start = wall_clock();

        // 1. Half-kick local particles
        leapfrog_kick_half(local_particles, config.dt);

        // 2. Drift local particles
        leapfrog_drift(local_particles, config.dt);

        // 3. Morton sort local particles
        auto t_sort_start = wall_clock();
        compute_and_sort_by_morton(local_particles, config.box_size);
        auto t_sort_end = wall_clock();

        // 4. MPI Global/Ghost Exchange with TRUE non-blocking overlap
        auto t_comm_start = wall_clock();
        double real_comm_time = 0.0;

        if (config.use_nonblocking && ctx.size > 1) {
            #ifdef ENABLE_MPI
            // ========================================
            // TRUE NON-BLOCKING OVERLAP:
            //   1. Start MPI_Iallgatherv asynchronously
            //   2. Build local-only tree while comm is in flight
            //   3. Complete comm before building global tree
            // ========================================
            auto handle = start_exchange_all_particles(local_particles, global_particles, ctx);

            // --- OVERLAP: Build local-only tree while communication is in flight ---
            if (config.use_linearized_tree) {
                lin_tree.build(local_particles, 0.0, 0.0, 0.0, config.box_size);
            } else {
                ptr_tree.build(local_particles, 0.0, 0.0, 0.0, config.box_size);
            }

            real_comm_time = wait_exchange_all_particles(handle);
            #endif
        } else {
            // Blocking exchange
            real_comm_time = exchange_all_particles(local_particles, global_particles, ctx, false);
        }
        auto t_comm_end = wall_clock();
        double comm_time_total = real_comm_time > 0.0 ? real_comm_time : elapsed(t_comm_start, t_comm_end);

        // 5. Build Octree from global particles (needed for full-accuracy forces)
        auto t_tree_start = wall_clock();
        if (config.use_linearized_tree) {
            lin_tree.build(global_particles, 0.0, 0.0, 0.0, config.box_size);
        } else {
            ptr_tree.build(global_particles, 0.0, 0.0, 0.0, config.box_size);
        }
        auto t_tree_end = wall_clock();

        // 6. Compute forces for local particles using global tree
        auto t_force_start = wall_clock();
        if (config.use_linearized_tree) {
            compute_forces_linearized(local_particles, lin_tree,
                                       config.G, config.theta, config.softening,
                                       config.num_threads);
        } else {
            compute_forces_pointer(local_particles, ptr_tree,
                                    config.G, config.theta, config.softening,
                                    config.num_threads);
        }
        auto t_force_end = wall_clock();

        // 7. Final half-kick
        leapfrog_kick_half(local_particles, config.dt);

        // 8. Dynamic Load Balancing Evaluation
        double rebalance_time = 0.0;
        int local_migrated_in = 0;
        int local_migrated_out = 0;
        double current_imbalance = 1.0;
        double local_time = elapsed(t_force_start, t_force_end);

        std::vector<double> all_workloads;
        comm_time_total += gather_workloads(local_time, all_workloads, ctx);
        current_imbalance = LoadBalancer::compute_imbalance(all_workloads);

        // ONLY "full" mode triggers rebalancing — mpi_morton is a static Morton baseline
        if (enable_rebalance && current_imbalance > config.rebalance_threshold) {
            if (ctx.size > 1) {
                std::vector<int> target_ranks(local_particles.size(), ctx.rank);

                // Sort global to get accurate Morton splitters
                compute_and_sort_by_morton(global_particles, config.box_size);
                std::vector<PartitionInfo> new_parts = partition_equal_count(global_particles, ctx.size);

                for (size_t i = 0; i < local_particles.size(); ++i) {
                    target_ranks[i] = find_owner_rank(local_particles[i].morton_key, new_parts);
                }

                rebalance_time = migrate_particles_mpi(local_particles, target_ranks, ctx,
                                                       &local_migrated_in, &local_migrated_out);
            } else {
                auto res = lb.check_and_rebalance(local_particles, partitions, all_workloads, config.box_size);
                local_migrated_in = res.particles_migrated;
                local_migrated_out = res.particles_migrated;
                rebalance_time = res.rebalance_time;
            }
        }

        // Ensure owner_rank on local particles matches this rank
        for (auto& p : local_particles) {
            p.owner_rank = ctx.rank;
        }

        auto t_step_end = wall_clock();

        // 9. Energy Computation & Reduction
        double local_ke = compute_kinetic_energy(local_particles);
        double global_ke = 0.0, global_pe = 0.0;
        reduce_energy(local_ke, 0.0, global_ke, global_pe, ctx);

        // Correct PE: use global_particles for cross-rank interactions
        if (ctx.is_root() && N_total <= 5000) {
            global_pe = compute_potential_energy(global_particles, config.G, config.softening);
        } else if (ctx.is_root()) {
            global_pe = std::numeric_limits<double>::quiet_NaN();
        }

        // 10. Per-rank metric collection
        #ifdef ENABLE_MPI
        if (ctx.size > 1) {
            int local_count = static_cast<int>(local_particles.size());
            double local_tree_time = elapsed(t_tree_start, t_tree_end) + elapsed(t_sort_start, t_sort_end);

            std::vector<int> all_counts(ctx.size);
            std::vector<double> all_force_times(ctx.size);
            std::vector<double> all_tree_times(ctx.size);
            std::vector<int> all_migrated_in(ctx.size);
            std::vector<int> all_migrated_out(ctx.size);

            MPI_Gather(&local_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Gather(&local_time, 1, MPI_DOUBLE, all_force_times.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gather(&local_tree_time, 1, MPI_DOUBLE, all_tree_times.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gather(&local_migrated_in, 1, MPI_INT, all_migrated_in.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Gather(&local_migrated_out, 1, MPI_INT, all_migrated_out.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

            if (ctx.is_root()) {
                int total_migrated = 0;
                for (int r = 0; r < ctx.size; ++r) total_migrated += all_migrated_in[r];

                // Log a metrics row for EVERY MPI rank
                for (int r = 0; r < ctx.size; ++r) {
                    TimestepMetrics m;
                    m.timestep = step;
                    m.rank = r;
                    m.local_particle_count = all_counts[r];
                    m.tree_build_time = all_tree_times[r];
                    m.force_compute_time = all_force_times[r];
                    m.comm_time = comm_time_total;
                    m.rebalance_time = rebalance_time;
                    m.total_step_time = elapsed(t_step_start, t_step_end);
                    m.imbalance_ratio = current_imbalance;
                    m.particles_migrated = all_migrated_in[r];
                    m.particles_migrated_in = all_migrated_in[r];
                    m.particles_migrated_out = all_migrated_out[r];
                    m.kinetic_energy = global_ke;
                    m.potential_energy = global_pe;

                    logger.log_timestep(m);
                }

                if (config.verbose) {
                    TimestepMetrics m_root;
                    m_root.timestep = step;
                    m_root.rank = 0;
                    m_root.local_particle_count = all_counts[0];
                    m_root.tree_build_time = all_tree_times[0];
                    m_root.force_compute_time = all_force_times[0];
                    m_root.comm_time = comm_time_total;
                    m_root.rebalance_time = rebalance_time;
                    m_root.total_step_time = elapsed(t_step_start, t_step_end);
                    m_root.imbalance_ratio = current_imbalance;
                    m_root.particles_migrated = total_migrated;
                    m_root.particles_migrated_in = all_migrated_in[0];
                    m_root.particles_migrated_out = all_migrated_out[0];
                    m_root.kinetic_energy = global_ke;
                    m_root.potential_energy = global_pe;

                    MetricsLogger::print_timestep_summary(m_root, config.timesteps, N_total, config.theta);

                    // Print per-rank breakdown with particle count and migration in/out
                    std::cout << "  Per-rank: ";
                    for (int r = 0; r < ctx.size; ++r) {
                        std::cout << "[R" << r << ": n=" << all_counts[r]
                                  << " mig_in=" << all_migrated_in[r]
                                  << " mig_out=" << all_migrated_out[r]
                                  << " t=" << std::fixed << std::setprecision(4)
                                  << all_force_times[r] << "s] ";
                    }
                    std::cout << "\n";
                }
            }

            // Collective particle gather for visualizer / particle dumping
            if (config.dump_particles && (step % config.dump_interval == 0)) {
                exchange_all_particles(local_particles, global_particles, ctx, false);
                if (ctx.is_root()) {
                    dump_particle_state(config.output_dir, step, global_particles);
                }
            }
        } else
        #endif
        {
            TimestepMetrics m;
            m.timestep = step;
            m.rank = 0;
            m.local_particle_count = static_cast<int>(local_particles.size());
            m.tree_build_time = elapsed(t_tree_start, t_tree_end) + elapsed(t_sort_start, t_sort_end);
            m.force_compute_time = elapsed(t_force_start, t_force_end);
            m.comm_time = comm_time_total;
            m.rebalance_time = rebalance_time;
            m.total_step_time = elapsed(t_step_start, t_step_end);
            m.imbalance_ratio = current_imbalance;
            m.particles_migrated = local_migrated_in;
            m.particles_migrated_in = local_migrated_in;
            m.particles_migrated_out = local_migrated_out;
            m.kinetic_energy = global_ke;
            m.potential_energy = global_pe;

            logger.log_timestep(m);
            if (config.verbose) {
                MetricsLogger::print_timestep_summary(m, config.timesteps, N_total, config.theta);
            }
            if (config.dump_particles && (step % config.dump_interval == 0)) {
                dump_particle_state(config.output_dir, step, global_particles);
            }
        }
    }

    if (ctx.is_root()) {
        logger.finalize();
        std::cout << "\n[" << config.mode << " MPI Simulation] Complete.\n";
    }
}

} // namespace albh
