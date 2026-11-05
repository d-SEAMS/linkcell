/* Device weights, curve keys, and the prefix split.
 * Host twins of those three kernels live in this file so a machine
 * without a GPU can compile the translation unit and check them. */

#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>

#if defined(LINKCELL_HAS_GPULITE)
#include "gpulite_compat.hpp"
using gpulite::CUDART;
using gpulite::KernelFactory;
using gpulite::NVRTC;
#endif

namespace {

uint64_t spread21(uint32_t v) {
  uint64_t x = v & 0x1fffffu;
  x = (x | (x << 32)) & 0x001f00000000ffffull;
  x = (x | (x << 16)) & 0x001f0000ff0000ffull;
  x = (x | (x << 8)) & 0x100f00f00f00f00full;
  x = (x | (x << 4)) & 0x10c30c30c30c30c3ull;
  x = (x | (x << 2)) & 0x1249249249249249ull;
  return x;
}

uint64_t morton_key(uint32_t x, uint32_t y, uint32_t z) {
  return spread21(x) | (spread21(y) << 1) | (spread21(z) << 2);
}

uint64_t hilbert_key(uint32_t x, uint32_t y, uint32_t z, uint32_t bits) {
  if (bits == 0) {
    return 0;
  }
  uint32_t p[3] = {x, y, z};
  uint32_t q = 1u << (bits - 1);
  while (q > 1) {
    uint32_t mask = q - 1;
    for (int i = 0; i < 3; ++i) {
      if (p[i] & q) {
        p[0] ^= mask;
      } else {
        uint32_t t = (p[0] ^ p[i]) & mask;
        p[0] ^= t;
        p[i] ^= t;
      }
    }
    q >>= 1;
  }
  p[1] ^= p[0];
  p[2] ^= p[1];
  uint32_t t = 0;
  q = 1u << (bits - 1);
  while (q > 1) {
    if (p[2] & q) {
      t ^= q - 1;
    }
    q >>= 1;
  }
  for (int i = 0; i < 3; ++i) {
    p[i] ^= t;
  }
  uint64_t key = 0;
  for (int b = static_cast<int>(bits) - 1; b >= 0; --b) {
    for (int i = 0; i < 3; ++i) {
      key = (key << 1) | ((p[i] >> b) & 1u);
    }
  }
  return key;
}

uint32_t hilbert_bits(uint32_t nx, uint32_t ny, uint32_t nz) {
  uint32_t n = nx;
  if (ny > n) n = ny;
  if (nz > n) n = nz;
  if (n <= 1) return 0;
  uint32_t bits = 0;
  uint32_t v = n - 1;
  while (v > 0) {
    v >>= 1;
    bits += 1;
  }
  return bits;
}

uint32_t owner_of(uint64_t before, uint64_t weight, uint64_t total, uint32_t parts) {
  if (total == 0 || parts == 0) {
    return 0;
  }
  unsigned __int128 num =
      (static_cast<unsigned __int128>(2) * before + weight) * parts;
  unsigned __int128 den = static_cast<unsigned __int128>(2) * total;
  uint32_t owner = static_cast<uint32_t>(num / den);
  if (owner >= parts) {
    owner = parts - 1;
  }
  return owner;
}

const char *kCurve = R"CUDA(
__device__ unsigned long long spread21(unsigned int v) {
  unsigned long long x = v & 0x1fffffu;
  x = (x | (x << 32)) & 0x001f00000000ffffull;
  x = (x | (x << 16)) & 0x001f0000ff0000ffull;
  x = (x | (x << 8)) & 0x100f00f00f00f00full;
  x = (x | (x << 4)) & 0x10c30c30c30c30c3ull;
  x = (x | (x << 2)) & 0x1249249249249249ull;
  return x;
}

extern "C" __global__ void hist_cells(const unsigned int* bins,
    unsigned long long* weight, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(weight + bins[i], 1ull);
}

extern "C" __global__ void morton_cells(const unsigned int* coords,
    unsigned long long* keys, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned int x = coords[3 * i];
  unsigned int y = coords[3 * i + 1];
  unsigned int z = coords[3 * i + 2];
  keys[i] = spread21(x) | (spread21(y) << 1) | (spread21(z) << 2);
}

extern "C" __global__ void block_scan(const unsigned long long* in,
    unsigned long long* out, unsigned long long* block_sum, int n) {
  __shared__ unsigned long long s[256];
  int tid = threadIdx.x;
  int gid = blockIdx.x * 256 + tid;
  s[tid] = gid < n ? in[gid] : 0ull;
  __syncthreads();
  for (int offset = 1; offset < 256; offset <<= 1) {
    unsigned long long v = (tid >= offset) ? s[tid - offset] : 0ull;
    __syncthreads();
    if (tid >= offset) s[tid] += v;
    __syncthreads();
  }
  if (gid < n) out[gid] = (tid == 0) ? 0ull : s[tid - 1];
  if (tid == 255 && block_sum) block_sum[blockIdx.x] = s[255];
}

extern "C" __global__ void add_block(unsigned long long* out,
    const unsigned long long* block_prefix, int n) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < n && blockIdx.x > 0) out[gid] += block_prefix[blockIdx.x];
}
)CUDA";

} // namespace

extern "C" {

void lc_part_host_histogram(const uint32_t *bins, size_t n, uint64_t *weight,
                            size_t ncell) {
  std::memset(weight, 0, ncell * sizeof(uint64_t));
  for (size_t i = 0; i < n; ++i) {
    weight[bins[i]] += 1;
  }
}

void lc_part_host_morton(const uint32_t *coords, size_t n, uint64_t *keys) {
  for (size_t i = 0; i < n; ++i) {
    keys[i] = morton_key(coords[3 * i], coords[3 * i + 1], coords[3 * i + 2]);
  }
}

void lc_part_host_prefix(const uint64_t *values, size_t n, uint64_t *out) {
  out[0] = 0;
  for (size_t i = 0; i < n; ++i) {
    out[i + 1] = out[i] + values[i];
  }
}

uint64_t lc_part_host_hilbert(uint32_t x, uint32_t y, uint32_t z, uint32_t bits) {
  return hilbert_key(x, y, z, bits);
}

int lc_gpu_curve_split(const uint32_t *atom_bin, size_t natom,
                       const uint32_t *coords, size_t ncell, uint32_t parts,
                       uint32_t *assign, uint64_t *out_weight, uint8_t method) {
#if !defined(LINKCELL_HAS_GPULITE)
  (void)atom_bin;
  (void)natom;
  (void)coords;
  (void)ncell;
  (void)parts;
  (void)assign;
  (void)out_weight;
  (void)method;
  return 2;
#else
  if (!CUDART::loaded() || !NVRTC::loaded()) {
    return 2;
  }
  if (parts == 0 || assign == nullptr || out_weight == nullptr || ncell == 0) {
    return 1;
  }
  if (natom > 0 && atom_bin == nullptr) {
    return 1;
  }
  if (coords == nullptr) {
    return 1;
  }
  auto &rt = CUDART::instance();
  auto fail = [](void *a, void *b, void *c, void *d) {
    auto &rt = CUDART::instance();
    if (a) rt.cudaFree(a);
    if (b) rt.cudaFree(b);
    if (c) rt.cudaFree(c);
    if (d) rt.cudaFree(d);
    return 1;
  };
  uint32_t *d_bins = nullptr;
  uint64_t *d_weight = nullptr;
  uint32_t *d_coords = nullptr;
  uint64_t *d_keys = nullptr;
  if (rt.cudaMalloc(reinterpret_cast<void **>(&d_weight), ncell * sizeof(uint64_t)) != 0) {
    return 1;
  }
  rt.cudaMemset(d_weight, 0, ncell * sizeof(uint64_t));
  if (natom > 0) {
    if (rt.cudaMalloc(reinterpret_cast<void **>(&d_bins), natom * sizeof(uint32_t)) != 0) {
      return fail(d_weight, nullptr, nullptr, nullptr);
    }
    if (rt.cudaMemcpy(d_bins, atom_bin, natom * sizeof(uint32_t), cudaMemcpyHostToDevice) != 0) {
      return fail(d_weight, d_bins, nullptr, nullptr);
    }
  }
  if (rt.cudaMalloc(reinterpret_cast<void **>(&d_coords), ncell * 3 * sizeof(uint32_t)) != 0) {
    return fail(d_weight, d_bins, nullptr, nullptr);
  }
  if (rt.cudaMemcpy(d_coords, coords, ncell * 3 * sizeof(uint32_t), cudaMemcpyHostToDevice) != 0) {
    return fail(d_weight, d_bins, d_coords, nullptr);
  }
  if (rt.cudaMalloc(reinterpret_cast<void **>(&d_keys), ncell * sizeof(uint64_t)) != 0) {
    return fail(d_weight, d_bins, d_coords, nullptr);
  }
  auto &factory = KernelFactory::instance(0);
  const std::vector<std::string> opt{"-std=c++17"};
  const int block = 256;
  const int grid_atoms = static_cast<int>((natom + block - 1) / block);
  const int grid_cells = static_cast<int>((ncell + block - 1) / block);
  int n_atom = static_cast<int>(natom);
  int n_cell = static_cast<int>(ncell);
  if (natom > 0) {
    void *raw[] = {&d_bins, &d_weight, &n_atom};
    auto *k = factory.create("hist_cells", kCurve, "partition.cu", opt);
    k->launch(dim3(grid_atoms == 0 ? 1 : grid_atoms), dim3(block), 0, nullptr,
              std::vector<void *>(raw, raw + 3), true);
  }
  if (rt.cudaMemcpy(out_weight, d_weight, ncell * sizeof(uint64_t), cudaMemcpyDeviceToHost) != 0) {
    return fail(d_weight, d_bins, d_coords, d_keys);
  }
  /* Hilbert keys are produced on the host twin of the same recurrence.
   * Morton keys are produced on the device. Both copies are inside this call. */
  std::vector<uint64_t> keys(ncell);
  if (method == 0) {
    void *raw[] = {&d_coords, &d_keys, &n_cell};
    auto *k = factory.create("morton_cells", kCurve, "partition.cu", opt);
    k->launch(dim3(grid_cells == 0 ? 1 : grid_cells), dim3(block), 0, nullptr,
              std::vector<void *>(raw, raw + 3), true);
    if (rt.cudaMemcpy(keys.data(), d_keys, ncell * sizeof(uint64_t), cudaMemcpyDeviceToHost) != 0) {
      return fail(d_weight, d_bins, d_coords, d_keys);
    }
  } else {
    uint32_t nx = 0, ny = 0, nz = 0;
    for (size_t i = 0; i < ncell; ++i) {
      nx = std::max(nx, coords[3 * i] + 1);
      ny = std::max(ny, coords[3 * i + 1] + 1);
      nz = std::max(nz, coords[3 * i + 2] + 1);
    }
    uint32_t bits = hilbert_bits(nx, ny, nz);
    for (size_t i = 0; i < ncell; ++i) {
      keys[i] = hilbert_key(coords[3 * i], coords[3 * i + 1], coords[3 * i + 2], bits);
    }
    if (rt.cudaMemcpy(d_keys, keys.data(), ncell * sizeof(uint64_t), cudaMemcpyHostToDevice) != 0) {
      return fail(d_weight, d_bins, d_coords, d_keys);
    }
    if (rt.cudaMemcpy(keys.data(), d_keys, ncell * sizeof(uint64_t), cudaMemcpyDeviceToHost) != 0) {
      return fail(d_weight, d_bins, d_coords, d_keys);
    }
  }
  std::vector<uint32_t> order(ncell);
  for (size_t i = 0; i < ncell; ++i) order[i] = static_cast<uint32_t>(i);
  std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
    if (keys[a] != keys[b]) return keys[a] < keys[b];
    return a < b;
  });
  std::vector<uint64_t> ordered(ncell);
  uint64_t total = 0;
  for (size_t i = 0; i < ncell; ++i) {
    ordered[i] = out_weight[order[i]];
    total += ordered[i];
  }
  uint64_t *d_ordered = nullptr;
  uint64_t *d_prefix = nullptr;
  uint64_t *d_blocks = nullptr;
  int n_blocks = grid_cells == 0 ? 1 : grid_cells;
  if (rt.cudaMalloc(reinterpret_cast<void **>(&d_ordered), ncell * sizeof(uint64_t)) != 0 ||
      rt.cudaMalloc(reinterpret_cast<void **>(&d_prefix), (ncell + 1) * sizeof(uint64_t)) != 0 ||
      rt.cudaMalloc(reinterpret_cast<void **>(&d_blocks), static_cast<size_t>(n_blocks) * sizeof(uint64_t)) != 0) {
    if (d_ordered) rt.cudaFree(d_ordered);
    if (d_prefix) rt.cudaFree(d_prefix);
    if (d_blocks) rt.cudaFree(d_blocks);
    return fail(d_weight, d_bins, d_coords, d_keys);
  }
  if (rt.cudaMemcpy(d_ordered, ordered.data(), ncell * sizeof(uint64_t), cudaMemcpyHostToDevice) != 0) {
    rt.cudaFree(d_ordered);
    rt.cudaFree(d_prefix);
    rt.cudaFree(d_blocks);
    return fail(d_weight, d_bins, d_coords, d_keys);
  }
  {
    void *raw[] = {&d_ordered, &d_prefix, &d_blocks, &n_cell};
    auto *k = factory.create("block_scan", kCurve, "partition.cu", opt);
    k->launch(dim3(n_blocks), dim3(block), 0, nullptr, std::vector<void *>(raw, raw + 4), true);
  }
  std::vector<uint64_t> block_sum(static_cast<size_t>(n_blocks));
  std::vector<uint64_t> block_prefix(static_cast<size_t>(n_blocks), 0);
  if (rt.cudaMemcpy(block_sum.data(), d_blocks, block_sum.size() * sizeof(uint64_t),
                    cudaMemcpyDeviceToHost) != 0) {
    rt.cudaFree(d_ordered);
    rt.cudaFree(d_prefix);
    rt.cudaFree(d_blocks);
    return fail(d_weight, d_bins, d_coords, d_keys);
  }
  for (int b = 1; b < n_blocks; ++b) {
    block_prefix[static_cast<size_t>(b)] =
        block_prefix[static_cast<size_t>(b - 1)] + block_sum[static_cast<size_t>(b - 1)];
  }
  if (rt.cudaMemcpy(d_blocks, block_prefix.data(), block_prefix.size() * sizeof(uint64_t),
                    cudaMemcpyHostToDevice) != 0) {
    rt.cudaFree(d_ordered);
    rt.cudaFree(d_prefix);
    rt.cudaFree(d_blocks);
    return fail(d_weight, d_bins, d_coords, d_keys);
  }
  {
    void *raw[] = {&d_prefix, &d_blocks, &n_cell};
    auto *k = factory.create("add_block", kCurve, "partition.cu", opt);
    k->launch(dim3(n_blocks), dim3(block), 0, nullptr, std::vector<void *>(raw, raw + 3), true);
  }
  std::vector<uint64_t> prefix(ncell);
  if (rt.cudaMemcpy(prefix.data(), d_prefix, ncell * sizeof(uint64_t), cudaMemcpyDeviceToHost) != 0) {
    rt.cudaFree(d_ordered);
    rt.cudaFree(d_prefix);
    rt.cudaFree(d_blocks);
    return fail(d_weight, d_bins, d_coords, d_keys);
  }
  uint64_t use_total = total == 0 ? ncell : total;
  for (size_t i = 0; i < ncell; ++i) {
    uint64_t w = total == 0 ? 1 : ordered[i];
    assign[order[i]] = owner_of(prefix[i], w, use_total, parts);
  }
  rt.cudaFree(d_ordered);
  rt.cudaFree(d_prefix);
  rt.cudaFree(d_blocks);
  rt.cudaFree(d_weight);
  if (d_bins) rt.cudaFree(d_bins);
  rt.cudaFree(d_coords);
  rt.cudaFree(d_keys);
  return 0;
#endif
}

}
