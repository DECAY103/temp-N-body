#include "communication.hpp"
#include <iostream>
#include <numeric>
#include <algorithm>

namespace albh {

#ifdef ENABLE_MPI

void init_mpi(int* argc, char*** argv, MPIContext& ctx) {
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        MPI_Init(argc, argv);
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &ctx.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ctx.size);
}

void finalize_mpi() {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) {
        MPI_Finalize();
    }
}

NonblockingExchangeHandle start_exchange_all_particles(
    const std::vector<Particle>& local_particles,
    std::vector<Particle>& global_particles,
    const MPIContext& ctx) {
    NonblockingExchangeHandle handle;
    if (ctx.size <= 1) {
        global_particles = local_particles;
        return handle;
    }

    handle.t_start = MPI_Wtime();

    int local_n = static_cast<int>(local_particles.size());
    std::vector<int> counts(ctx.size);
    MPI_Allgather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    int total_n = 0;
    std::vector<int> byte_counts(ctx.size);
    std::vector<int> displs(ctx.size);

    for (int r = 0; r < ctx.size; ++r) {
        displs[r] = total_n * sizeof(Particle);
        byte_counts[r] = counts[r] * sizeof(Particle);
        total_n += counts[r];
    }

    global_particles.resize(total_n);

    int send_bytes = local_n * sizeof(Particle);
    const void* send_buf = local_particles.empty() ? nullptr : local_particles.data();
    void* recv_buf = global_particles.empty() ? nullptr : global_particles.data();

    MPI_Iallgatherv(send_buf, send_bytes, MPI_BYTE,
                    recv_buf, byte_counts.data(), displs.data(), MPI_BYTE,
                    MPI_COMM_WORLD, &handle.request);
    return handle;
}

double wait_exchange_all_particles(NonblockingExchangeHandle& handle) {
    if (handle.request != MPI_REQUEST_NULL) {
        MPI_Wait(&handle.request, MPI_STATUS_IGNORE);
        handle.request = MPI_REQUEST_NULL;
        return MPI_Wtime() - handle.t_start;
    }
    return 0.0;
}

double exchange_all_particles(const std::vector<Particle>& local_particles,
                              std::vector<Particle>& global_particles,
                              const MPIContext& ctx,
                              bool use_nonblocking) {
    if (ctx.size <= 1) {
        global_particles = local_particles;
        return 0.0;
    }

    if (use_nonblocking) {
        auto handle = start_exchange_all_particles(local_particles, global_particles, ctx);
        return wait_exchange_all_particles(handle);
    }

    double t_start = MPI_Wtime();

    int local_n = static_cast<int>(local_particles.size());
    std::vector<int> counts(ctx.size);
    MPI_Allgather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    int total_n = 0;
    std::vector<int> byte_counts(ctx.size);
    std::vector<int> displs(ctx.size);

    for (int r = 0; r < ctx.size; ++r) {
        displs[r] = total_n * sizeof(Particle);
        byte_counts[r] = counts[r] * sizeof(Particle);
        total_n += counts[r];
    }

    global_particles.resize(total_n);

    int send_bytes = local_n * sizeof(Particle);
    const void* send_buf = local_particles.empty() ? nullptr : local_particles.data();
    void* recv_buf = global_particles.empty() ? nullptr : global_particles.data();

    MPI_Allgatherv(send_buf, send_bytes, MPI_BYTE,
                   recv_buf, byte_counts.data(), displs.data(), MPI_BYTE,
                   MPI_COMM_WORLD);

    double t_end = MPI_Wtime();
    return t_end - t_start;
}

void reduce_energy(double local_ke, double local_pe,
                   double& global_ke, double& global_pe,
                   const MPIContext& ctx) {
    if (ctx.size <= 1) {
        global_ke = local_ke;
        global_pe = local_pe;
        return;
    }

    MPI_Reduce(&local_ke, &global_ke, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_pe, &global_pe, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
}

double gather_workloads(double local_workload,
                        std::vector<double>& all_workloads,
                        const MPIContext& ctx) {
    all_workloads.resize(ctx.size);
    if (ctx.size <= 1) {
        all_workloads[0] = local_workload;
        return 0.0;
    }

    double t_start = MPI_Wtime();
    MPI_Allgather(&local_workload, 1, MPI_DOUBLE,
                  all_workloads.data(), 1, MPI_DOUBLE, MPI_COMM_WORLD);
    return MPI_Wtime() - t_start;
}

double migrate_particles_mpi(std::vector<Particle>& local_particles,
                             const std::vector<int>& target_ranks,
                             const MPIContext& ctx) {
    if (ctx.size <= 1 || local_particles.empty()) {
        return 0.0;
    }

    double t_start = MPI_Wtime();

    // 1. Group particles by destination rank
    std::vector<int> send_counts(ctx.size, 0);
    for (size_t i = 0; i < local_particles.size(); ++i) {
        int target = target_ranks[i];
        if (target >= 0 && target < ctx.size) {
            send_counts[target]++;
        }
    }

    // 2. Exchange send counts across all ranks to determine receive counts
    std::vector<int> recv_counts(ctx.size, 0);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                 recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // 3. Compute byte displacements
    std::vector<int> s_displs(ctx.size, 0);
    std::vector<int> r_displs(ctx.size, 0);
    std::vector<int> s_bytes(ctx.size, 0);
    std::vector<int> r_bytes(ctx.size, 0);

    int total_send = 0;
    int total_recv = 0;
    for (int r = 0; r < ctx.size; ++r) {
        s_displs[r] = total_send * sizeof(Particle);
        r_displs[r] = total_recv * sizeof(Particle);
        s_bytes[r] = send_counts[r] * sizeof(Particle);
        r_bytes[r] = recv_counts[r] * sizeof(Particle);
        total_send += send_counts[r];
        total_recv += recv_counts[r];
    }

    // 4. Pack send buffer
    std::vector<Particle> send_buf(total_send);
    std::vector<int> current_s_offset = s_displs;
    for (size_t i = 0; i < local_particles.size(); ++i) {
        int target = target_ranks[i];
        int byte_off = current_s_offset[target];
        int idx = byte_off / sizeof(Particle);
        send_buf[idx] = local_particles[i];
        send_buf[idx].owner_rank = target;
        current_s_offset[target] += sizeof(Particle);
    }

    // 5. Allocate receive buffer and exchange with MPI_Alltoallv
    std::vector<Particle> recv_buf(total_recv);
    MPI_Alltoallv(send_buf.data(), s_bytes.data(), s_displs.data(), MPI_BYTE,
                  recv_buf.data(), r_bytes.data(), r_displs.data(), MPI_BYTE,
                  MPI_COMM_WORLD);

    // 6. Update local particle storage
    local_particles = std::move(recv_buf);

    double t_end = MPI_Wtime();
    return t_end - t_start;
}

#else

// Fallback implementations when compiling without MPI
void init_mpi(int*, char***, MPIContext& ctx) {
    ctx.rank = 0;
    ctx.size = 1;
}

void finalize_mpi() {}

double exchange_all_particles(const std::vector<Particle>& local_particles,
                              std::vector<Particle>& global_particles,
                              const MPIContext&,
                              bool) {
    global_particles = local_particles;
    return 0.0;
}

void reduce_energy(double local_ke, double local_pe,
                   double& global_ke, double& global_pe,
                   const MPIContext&) {
    global_ke = local_ke;
    global_pe = local_pe;
}

double gather_workloads(double local_workload,
                        std::vector<double>& all_workloads,
                        const MPIContext&) {
    all_workloads = {local_workload};
    return 0.0;
}

double migrate_particles_mpi(std::vector<Particle>&,
                             const std::vector<int>&,
                             const MPIContext&) {
    return 0.0;
}

#endif

} // namespace albh
