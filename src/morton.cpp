#include "morton.hpp"
#include "particle.hpp"
#include <algorithm>

namespace albh {

// ============================================================================
// Bit spreading: place each bit of a 21-bit integer into every 3rd position
// Uses the "magic bits" method for efficient bit interleaving.
//
// Example for 3-bit input 'abc':
//   Input:  00000abc  →  Output: 00a00b00c
//
// Extended to 21 bits using successive doubling of spacing.
// ============================================================================
uint64_t spread_bits_3d(uint32_t v) {
    uint64_t x = v & MORTON_MAX_COORD;  // mask to 21 bits
    // Stage 1: spread bits with gap of 32
    x = (x | (x << 32)) & 0x001F00000000FFFFull;
    // Stage 2: gap of 16
    x = (x | (x << 16)) & 0x001F0000FF0000FFull;
    // Stage 3: gap of 8
    x = (x | (x << 8))  & 0x100F00F00F00F00Full;
    // Stage 4: gap of 4
    x = (x | (x << 4))  & 0x10C30C30C30C30C3ull;
    // Stage 5: gap of 2
    x = (x | (x << 2))  & 0x1249249249249249ull;
    return x;
}

// ============================================================================
// Bit compacting: inverse of spread — extract every 3rd bit into contiguous value
// ============================================================================
uint32_t compact_bits_3d(uint64_t v) {
    v &= 0x1249249249249249ull;
    v = (v | (v >> 2))  & 0x10C30C30C30C30C3ull;
    v = (v | (v >> 4))  & 0x100F00F00F00F00Full;
    v = (v | (v >> 8))  & 0x001F0000FF0000FFull;
    v = (v | (v >> 16)) & 0x001F00000000FFFFull;
    v = (v | (v >> 32)) & 0x00000000001FFFFFull;
    return static_cast<uint32_t>(v);
}

// ============================================================================
// Quantization: map floating-point coordinate to integer grid
// coord in [0, box_size] → quantized in [0, MORTON_MAX_COORD]
// Clamps out-of-range values to prevent overflow
// ============================================================================
uint32_t quantize_coord(double coord, double box_size) {
    double normalized = coord / box_size;
    // Clamp to [0, 1)
    if (normalized < 0.0) normalized = 0.0;
    if (normalized >= 1.0) normalized = 1.0 - 1e-15;
    return static_cast<uint32_t>(normalized * (MORTON_MAX_COORD + 1));
}

double dequantize_coord(uint32_t q, double box_size) {
    return (static_cast<double>(q) / (MORTON_MAX_COORD + 1)) * box_size;
}

// ============================================================================
// Morton key encoding: interleave x, y, z bits
// key = z_spread << 2 | y_spread << 1 | x_spread
// This places x in bit positions 0,3,6,...; y in 1,4,7,...; z in 2,5,8,...
// ============================================================================
uint64_t encode_morton(uint32_t x, uint32_t y, uint32_t z) {
    return spread_bits_3d(x) | (spread_bits_3d(y) << 1) | (spread_bits_3d(z) << 2);
}

void decode_morton(uint64_t key, uint32_t& x, uint32_t& y, uint32_t& z) {
    x = compact_bits_3d(key);
    y = compact_bits_3d(key >> 1);
    z = compact_bits_3d(key >> 2);
}

// Convenience: particle position → Morton key
uint64_t particle_to_morton(double x, double y, double z, double box_size) {
    uint32_t qx = quantize_coord(x, box_size);
    uint32_t qy = quantize_coord(y, box_size);
    uint32_t qz = quantize_coord(z, box_size);
    return encode_morton(qx, qy, qz);
}

// ============================================================================
// Compute Morton keys for all particles and sort by key
// This establishes the spatial ordering used for domain decomposition.
// ============================================================================
void compute_and_sort_by_morton(std::vector<Particle>& particles, double box_size) {
    // Compute keys
    for (auto& p : particles) {
        p.morton_key = particle_to_morton(p.x, p.y, p.z, box_size);
    }

    // Sort by Morton key for spatial locality
    std::sort(particles.begin(), particles.end(),
              [](const Particle& a, const Particle& b) {
                  return a.morton_key < b.morton_key;
              });
}

} // namespace albh
