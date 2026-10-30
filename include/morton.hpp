#ifndef ALBH_MORTON_HPP
#define ALBH_MORTON_HPP

#include <cstdint>
#include <vector>

namespace albh {

// Forward declaration
struct Particle;

// ============================================================================
// Morton (Z-order) Key Encoding/Decoding
//
// Maps 3D spatial coordinates to a 1D key that preserves spatial locality.
// Uses 21 bits per dimension → 63-bit key → fits in uint64_t.
// Resolution: 2^21 = 2,097,152 levels per axis.
//
// The bit-interleaving pattern for 3 bits of (x, y, z) = (x2x1x0, y2y1y0, z2z1z0):
//   key = z2 y2 x2 z1 y1 x1 z0 y0 x0
// ============================================================================

// Maximum integer coordinate value (2^21 - 1)
constexpr uint32_t MORTON_BITS_PER_DIM = 21;
constexpr uint32_t MORTON_MAX_COORD = (1u << MORTON_BITS_PER_DIM) - 1;

// Spread bits of a 21-bit integer across a 63-bit value
// Input:  ...000000000000000cba987654321  (21 bits)
// Output: ...00c00b00a009008007006005004003002001  (bits at positions 0,3,6,9,...)
uint64_t spread_bits_3d(uint32_t v);

// Compact a spread value back to 21-bit integer (inverse of spread_bits_3d)
uint32_t compact_bits_3d(uint64_t v);

// Quantize a coordinate from [0, box_size] to [0, MORTON_MAX_COORD]
// Clamps out-of-range values
uint32_t quantize_coord(double coord, double box_size);

// De-quantize an integer coordinate back to floating point
double dequantize_coord(uint32_t q, double box_size);

// Encode 3D quantized integer coordinates to Morton key
// key = interleave(x, y, z) where x occupies bits 0,3,6,...; y bits 1,4,7,...; z bits 2,5,8,...
uint64_t encode_morton(uint32_t x, uint32_t y, uint32_t z);

// Decode Morton key to 3D quantized integer coordinates
void decode_morton(uint64_t key, uint32_t& x, uint32_t& y, uint32_t& z);

// Convenience: encode particle's (x,y,z) position to Morton key
uint64_t particle_to_morton(double x, double y, double z, double box_size);

// Compute Morton keys for all particles and sort by key
void compute_and_sort_by_morton(std::vector<Particle>& particles, double box_size);

} // namespace albh

#endif // ALBH_MORTON_HPP
