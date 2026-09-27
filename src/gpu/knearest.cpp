#include "linkcell_gpu.hpp"

#ifdef LINKCELL_HAS_GPULITE

#include "gpulite_compat.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

using gpulite::CUDART;
using gpulite::KernelFactory;
using gpulite::NVRTC;

namespace linkcell {
namespace gpu {
namespace {

constexpr int kMaxK = 16;
constexpr int kMaxCells = 262144;

const char *kKernels = R"CUDA(
__device__ inline int remEuclid(int a, int n) {
  int r = a % n;
  return r < 0 ? r + n : r;
}
__device__ inline int divEuclid(int a, int n) {
  int q = a / n;
  int r = a % n;
  if (r < 0) --q;
  return q;
}

__device__ inline void heapPush(double* bestR2, int* bestJ, int* found, int k,
    double r2, int j) {
  int have = -1;
  for (int t = 0; t < *found; ++t) {
    if (bestJ[t] == j) {
      have = t;
      break;
    }
  }
  if (have >= 0) {
    if (r2 < bestR2[have]) bestR2[have] = r2;
    return;
  }
  if (*found < k) {
    bestR2[*found] = r2;
    bestJ[*found] = j;
    ++(*found);
    return;
  }
  int w = 0;
  for (int t = 1; t < k; ++t) {
    if (bestR2[t] > bestR2[w]) w = t;
  }
  if (r2 < bestR2[w]) {
    bestR2[w] = r2;
    bestJ[w] = j;
  }
}

__device__ inline void mulCols(const double* __restrict__ M, double s0,
    double s1, double s2, double& x, double& y, double& z) {
  x = M[0] * s0 + M[3] * s1 + M[6] * s2;
  y = M[1] * s0 + M[4] * s1 + M[7] * s2;
  z = M[2] * s0 + M[5] * s1 + M[8] * s2;
}

extern "C" __global__ void invert_cell(
    const double* __restrict__ cell, int cell_n, int nFrames,
    double* __restrict__ H, double* __restrict__ Hinv,
    double* __restrict__ origin, double* __restrict__ cmin,
    double cell_hint, int* __restrict__ plan) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  double ax, ay, az, bx, by, bz, cx, cy, cz, ox, oy, oz;
  ox = 0.0;
  oy = 0.0;
  oz = 0.0;
  if (cell_n == 3) {
    ax = cell[0];
    ay = 0.0;
    az = 0.0;
    bx = 0.0;
    by = cell[1];
    bz = 0.0;
    cx = 0.0;
    cy = 0.0;
    cz = cell[2];
  } else if (cell_n == 9 || cell_n == 12) {
    ax = cell[0];
    ay = cell[1];
    az = cell[2];
    bx = cell[3];
    by = cell[4];
    bz = cell[5];
    cx = cell[6];
    cy = cell[7];
    cz = cell[8];
    if (cell_n == 12) {
      ox = cell[9];
      oy = cell[10];
      oz = cell[11];
    }
  } else {
    plan[4] = 0;
    return;
  }
  const double det = ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) +
                     az * (bx * cy - by * cx);
  if (!(det == det) || fabs(det) < 1.0e-18) {
    plan[4] = 0;
    return;
  }
  const double invdet = 1.0 / det;
  double H0[9], Hi[9];
  H0[0] = ax;
  H0[1] = ay;
  H0[2] = az;
  H0[3] = bx;
  H0[4] = by;
  H0[5] = bz;
  H0[6] = cx;
  H0[7] = cy;
  H0[8] = cz;
  Hi[0] = (by * cz - bz * cy) * invdet;
  Hi[1] = (az * cy - ay * cz) * invdet;
  Hi[2] = (ay * bz - az * by) * invdet;
  Hi[3] = (bz * cx - bx * cz) * invdet;
  Hi[4] = (ax * cz - az * cx) * invdet;
  Hi[5] = (az * bx - ax * bz) * invdet;
  Hi[6] = (bx * cy - by * cx) * invdet;
  Hi[7] = (ay * cx - ax * cy) * invdet;
  Hi[8] = (ax * by - ay * bx) * invdet;
  const double bcx = by * cz - bz * cy;
  const double bcy = bz * cx - bx * cz;
  const double bcz = bx * cy - by * cx;
  const double cax = cy * az - cz * ay;
  const double cay = cz * ax - cx * az;
  const double caz = cx * ay - cy * ax;
  const double abx = ay * bz - az * by;
  const double aby = az * bx - ax * bz;
  const double abz = ax * by - ay * bx;
  const double adet = fabs(det);
  const double w0 = adet / sqrt(bcx * bcx + bcy * bcy + bcz * bcz);
  const double w1 = adet / sqrt(cax * cax + cay * cay + caz * caz);
  const double w2 = adet / sqrt(abx * abx + aby * aby + abz * abz);
  if (!(w0 > 0.0 && w1 > 0.0 && w2 > 0.0)) {
    plan[4] = 0;
    return;
  }
  double edge = cell_hint;
  if (!(edge > 0.0)) edge = 3.0;
  double wmin = w0;
  if (w1 < wmin) wmin = w1;
  if (w2 < wmin) wmin = w2;
  if (edge > wmin) edge = wmin;
  int nx = (int)floor(w0 / edge);
  int ny = (int)floor(w1 / edge);
  int nz = (int)floor(w2 / edge);
  if (nx < 1) nx = 1;
  if (ny < 1) ny = 1;
  if (nz < 1) nz = 1;
  const int nC = nx * ny * nz;
  if (nC <= 0 || nC > 262144) {
    plan[4] = 2;
    return;
  }
  const double cellMin =
      fmin(w0 / (double)nx, fmin(w1 / (double)ny, w2 / (double)nz));
  origin[0] = ox;
  origin[1] = oy;
  origin[2] = oz;
  for (int f = 0; f < nFrames; ++f) {
    for (int i = 0; i < 9; ++i) {
      H[f * 9 + i] = H0[i];
      Hinv[f * 9 + i] = Hi[i];
    }
    cmin[f] = cellMin;
  }
  plan[0] = nx;
  plan[1] = ny;
  plan[2] = nz;
  plan[3] = nC;
  plan[4] = 1;
}

extern "C" __global__ void bin_atoms(const double* __restrict__ xyz,
    const int* __restrict__ mask, int n, int nFrames,
    const double* __restrict__ H, const double* __restrict__ Hinv,
    const double* __restrict__ origin, int nx, int ny, int nz,
    int nC, int* __restrict__ cellOf, int* __restrict__ cellCount,
    double* __restrict__ folded) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n * nFrames) return;
  const int f = tid / n;
  const int i = tid % n;
  const int id = f * n + i;
  if (mask && mask[id] == 0) {
    cellOf[id] = -1;
    folded[id * 3 + 0] = 0;
    folded[id * 3 + 1] = 0;
    folded[id * 3 + 2] = 0;
    return;
  }
  const double* Hf = H + f * 9;
  const double* Hi = Hinv + f * 9;
  const double ox = origin[0];
  const double oy = origin[1];
  const double oz = origin[2];
  const double dx = xyz[id * 3 + 0] - ox;
  const double dy = xyz[id * 3 + 1] - oy;
  const double dz = xyz[id * 3 + 2] - oz;
  double s0, s1, s2;
  mulCols(Hi, dx, dy, dz, s0, s1, s2);
  s0 -= floor(s0);
  s1 -= floor(s1);
  s2 -= floor(s2);
  double fx, fy, fz;
  mulCols(Hf, s0, s1, s2, fx, fy, fz);
  folded[id * 3 + 0] = fx + ox;
  folded[id * 3 + 1] = fy + oy;
  folded[id * 3 + 2] = fz + oz;
  int cx = (int)(s0 * nx); if (cx < 0) cx = 0; if (cx >= nx) cx = nx - 1;
  int cy = (int)(s1 * ny); if (cy < 0) cy = 0; if (cy >= ny) cy = ny - 1;
  int cz = (int)(s2 * nz); if (cz < 0) cz = 0; if (cz >= nz) cz = nz - 1;
  const int cid = (cz * ny + cy) * nx + cx;
  cellOf[id] = cid;
  atomicAdd(cellCount + f * nC + cid, 1);
}

// Tiled Blelloch exclusive scan, one frame per block (HOOMD / CUB DeviceScan).
extern "C" __global__ void prefix_cells(int* __restrict__ cellCount,
    int* __restrict__ cellOff, int nC, int nFrames) {
  extern __shared__ int sh[];
  const int f = blockIdx.x;
  if (f >= nFrames) return;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  int* counts = cellCount + f * nC;
  int* offs = cellOff + f * (nC + 1);
  int carry = 0;
  for (int base = 0; base < nC; base += nthreads) {
    const int i = base + tid;
    const int val = (i < nC) ? counts[i] : 0;
    sh[tid] = val;
    __syncthreads();
    for (int d = 1; d < nthreads; d <<= 1) {
      int add = 0;
      if (tid >= d) add = sh[tid - d];
      __syncthreads();
      sh[tid] += add;
      __syncthreads();
    }
    // sh[tid] is inclusive; exclusive is sh[tid] - val
    if (i < nC) {
      offs[i] = carry + sh[tid] - val;
      counts[i] = 0;
    }
    __syncthreads();
    if (tid == nthreads - 1) {
      sh[0] = carry + sh[tid];
    }
    __syncthreads();
    carry = sh[0];
    __syncthreads();
  }
  if (tid == 0) offs[nC] = carry;
}

// HOOMD / vesin: store coordinates in cell order for coalesced stencil reads.
extern "C" __global__ void scatter_atoms(const double* __restrict__ folded,
    const int* __restrict__ cellOf, int* __restrict__ cellCount,
    const int* __restrict__ cellOff, int* __restrict__ order,
    double* __restrict__ sorted, int* __restrict__ home, int n, int nFrames,
    int nC) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n * nFrames) return;
  const int f = tid / n;
  const int i = tid % n;
  const int id = f * n + i;
  const int cid = cellOf[id];
  if (cid < 0) {
    home[id] = -1;
    return;
  }
  const int slot = atomicAdd(cellCount + f * nC + cid, 1);
  const int dest = f * n + cellOff[f * (nC + 1) + cid] + slot;
  order[dest] = i;
  home[id] = dest;
  sorted[dest * 3 + 0] = folded[id * 3 + 0];
  sorted[dest * 3 + 1] = folded[id * 3 + 1];
  sorted[dest * 3 + 2] = folded[id * 3 + 2];
}

// HOOMD NeighborListGPUBinned: TPP threads split the cell occupants.
// Cabana VerletLayout2D: packed n*k output. Host stop: worst <= (reach*cell_min)^2.
extern "C" __global__ void knearest_shells(const double* __restrict__ sorted,
    const int* __restrict__ cellOf, const int* __restrict__ cellOff,
    const int* __restrict__ order, const int* __restrict__ home,
    const int* __restrict__ sdx, const int* __restrict__ sdy,
    const int* __restrict__ sdz, const int* __restrict__ reachOff,
    int n, int nFrames, int nC, int nx, int ny, int nz,
    const double* __restrict__ H, const double* __restrict__ cellMin,
    int k, int maxReach, int tpp, int* __restrict__ out,
    double* __restrict__ out_d2) {
  const int lane = threadIdx.x % tpp;
  const int gid = (blockIdx.x * blockDim.x + threadIdx.x) / tpp;
  const int active = gid < n * nFrames;
  const int f = active ? gid / n : 0;
  const int i = active ? gid % n : 0;
  const int id = f * n + i;
  const int cid = active ? cellOf[id] : -1;
  if (active && lane == 0) {
    for (int t = 0; t < k; ++t) {
      out[id * k + t] = -1;
      if (out_d2) out_d2[id * k + t] = 0.0 / 0.0;
    }
  }
  const int nxy = nx * ny;
  const int iz = cid >= 0 ? cid / nxy : 0;
  const int iy = cid >= 0 ? (cid % nxy) / nx : 0;
  const int ix = cid >= 0 ? cid % nx : 0;
  double ixp = 0, iyp = 0, izp = 0;
  if (cid >= 0) {
    const int dest = home[id];
    ixp = sorted[dest * 3 + 0];
    iyp = sorted[dest * 3 + 1];
    izp = sorted[dest * 3 + 2];
  }
  const int K = k;
  double bestR2[16];
  int bestJ[16];
  int found = 0;
  extern __shared__ unsigned char raw[];
  const int ppb = blockDim.x / tpp;
  const int pslot = threadIdx.x / tpp;
  double* sh_d2 = (double*)raw;
  int* sh_j = (int*)(sh_d2 + ppb * tpp * K);
  int* sh_n = (int*)(sh_j + ppb * tpp * K);
  int* sh_stop = sh_n + ppb * tpp;
  int live = 1;
  for (int reach = 1; reach <= maxReach; ++reach) {
    const int s0 = reachOff[reach];
    const int s1 = reachOff[reach + 1];
    if (live && cid >= 0) {
      for (int st = s0; st < s1; ++st) {
        const int dx = sdx[st];
        const int dy = sdy[st];
        const int dz = sdz[st];
        const int jx = ix + dx;
        const int jy = iy + dy;
        const int jz = iz + dz;
        const int ncx = remEuclid(jx, nx);
        const int ncy = remEuclid(jy, ny);
        const int ncz = remEuclid(jz, nz);
        const int na = divEuclid(jx, nx);
        const int nb = divEuclid(jy, ny);
        const int nc = divEuclid(jz, nz);
        const double* Hf = H + f * 9;
        const double sx = (double)na * Hf[0] + (double)nb * Hf[3] + (double)nc * Hf[6];
        const double sy = (double)na * Hf[1] + (double)nb * Hf[4] + (double)nc * Hf[7];
        const double sz = (double)na * Hf[2] + (double)nb * Hf[5] + (double)nc * Hf[8];
        const int ncid = (ncz * ny + ncy) * nx + ncx;
        const int a0 = f * n + cellOff[f * (nC + 1) + ncid];
        const int a1 = f * n + cellOff[f * (nC + 1) + ncid + 1];
        for (int s = a0 + lane; s < a1; s += tpp) {
          const int j = order[s];
          if (j == i) continue;
          const double rx = sorted[s * 3 + 0] - ixp + sx;
          const double ry = sorted[s * 3 + 1] - iyp + sy;
          const double rz = sorted[s * 3 + 2] - izp + sz;
          const double r2 = rx * rx + ry * ry + rz * rz;
          if (r2 <= 0.0) continue;
          heapPush(bestR2, bestJ, &found, k, r2, j);
        }
      }
    }
    const int base = (pslot * tpp + lane) * K;
    sh_n[pslot * tpp + lane] = found;
    for (int t = 0; t < K; ++t) {
      sh_d2[base + t] = t < found ? bestR2[t] : 0;
      sh_j[base + t] = t < found ? bestJ[t] : -1;
    }
    __syncwarp();
    if (lane == 0) {
      found = 0;
      for (int L = 0; L < tpp; ++L) {
        const int nL = sh_n[pslot * tpp + L];
        const int bL = (pslot * tpp + L) * K;
        for (int t = 0; t < nL; ++t) {
          heapPush(bestR2, bestJ, &found, k, sh_d2[bL + t], sh_j[bL + t]);
        }
      }
      int stop = 0;
      if (found >= k) {
        double worst = bestR2[0];
        for (int t = 1; t < found; ++t) {
          if (bestR2[t] > worst) worst = bestR2[t];
        }
        const double bound = (double)reach * cellMin[f];
        if (worst <= bound * bound) stop = 1;
      }
      sh_n[pslot * tpp] = found;
      sh_stop[pslot] = stop;
      for (int t = 0; t < found; ++t) {
        sh_d2[pslot * tpp * K + t] = bestR2[t];
        sh_j[pslot * tpp * K + t] = bestJ[t];
      }
    }
    __syncwarp();
    found = sh_n[pslot * tpp];
    for (int t = 0; t < found; ++t) {
      bestR2[t] = sh_d2[pslot * tpp * K + t];
      bestJ[t] = sh_j[pslot * tpp * K + t];
    }
    if (sh_stop[pslot]) live = 0;
  }
  if (lane != 0) return;
  for (int a = 1; a < found; ++a) {
    const double key = bestR2[a];
    const int idj = bestJ[a];
    int p = a;
    while (p > 0 && (bestR2[p - 1] > key ||
                     (bestR2[p - 1] == key && bestJ[p - 1] > idj))) {
      bestR2[p] = bestR2[p - 1];
      bestJ[p] = bestJ[p - 1];
      --p;
    }
    bestR2[p] = key;
    bestJ[p] = idj;
  }
  for (int t = 0; t < found; ++t) {
    out[id * k + t] = bestJ[t];
    if (out_d2) out_d2[id * k + t] = bestR2[t];
  }
}

extern "C" __global__ void zero_i32(int* p, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = 0;
}
)CUDA";

void checkCuda(cudaError_t st, const char *what) {
  if (st != cudaSuccess) {
    throw Error(std::string(what) + ": " +
                CUDART::instance().cudaGetErrorString(st));
  }
}

struct DevPtr {
  void *p = nullptr;
  ~DevPtr() {
    if (p != nullptr && CUDART::loaded()) {
      CUDART::instance().cudaFree(p);
    }
  }
  void reset() {
    if (p != nullptr && CUDART::loaded()) {
      CUDART::instance().cudaFree(p);
    }
    p = nullptr;
  }
};

void growOne(DevPtr &d, std::size_t bytes, const char *what) {
  d.reset();
  checkCuda(CUDART::instance().cudaMalloc(&d.p, bytes), what);
}

bool cellIsOrtho(const Cell &cell) {
  constexpr double tol = 1.0e-12;
  return std::fabs(cell.a[1]) < tol && std::fabs(cell.a[2]) < tol &&
         std::fabs(cell.b[0]) < tol && std::fabs(cell.b[2]) < tol &&
         std::fabs(cell.c[0]) < tol && std::fabs(cell.c[1]) < tol &&
         cell.a[0] > 0.0 && cell.b[1] > 0.0 && cell.c[2] > 0.0;
}

// H and Hinv as columns of 3x3, matching Cell (Rust invert_columns).
bool fillH(const Cell &cell, double H[9], double Hinv[9], double widths[3]) {
  const double ax = cell.a[0], ay = cell.a[1], az = cell.a[2];
  const double bx = cell.b[0], by = cell.b[1], bz = cell.b[2];
  const double cx = cell.c[0], cy = cell.c[1], cz = cell.c[2];
  const double det = ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) +
                     az * (bx * cy - by * cx);
  if (!std::isfinite(det) || std::fabs(det) < 1.0e-18) {
    return false;
  }
  const double invdet = 1.0 / det;
  H[0] = ax;
  H[1] = ay;
  H[2] = az;
  H[3] = bx;
  H[4] = by;
  H[5] = bz;
  H[6] = cx;
  H[7] = cy;
  H[8] = cz;
  Hinv[0] = (by * cz - bz * cy) * invdet;
  Hinv[1] = (az * cy - ay * cz) * invdet;
  Hinv[2] = (ay * bz - az * by) * invdet;
  Hinv[3] = (bz * cx - bx * cz) * invdet;
  Hinv[4] = (ax * cz - az * cx) * invdet;
  Hinv[5] = (az * bx - ax * bz) * invdet;
  Hinv[6] = (bx * cy - by * cx) * invdet;
  Hinv[7] = (ay * cx - ax * cy) * invdet;
  Hinv[8] = (ax * by - ay * bx) * invdet;
  const double bcx = by * cz - bz * cy;
  const double bcy = bz * cx - bx * cz;
  const double bcz = bx * cy - by * cx;
  const double cax = cy * az - cz * ay;
  const double cay = cz * ax - cx * az;
  const double caz = cx * ay - cy * ax;
  const double abx = ay * bz - az * by;
  const double aby = az * bx - ax * bz;
  const double abz = ax * by - ay * bx;
  const double adet = std::fabs(det);
  widths[0] = adet / std::sqrt(bcx * bcx + bcy * bcy + bcz * bcz);
  widths[1] = adet / std::sqrt(cax * cax + cay * cay + caz * caz);
  widths[2] = adet / std::sqrt(abx * abx + aby * aby + abz * abz);
  return widths[0] > 0.0 && widths[1] > 0.0 && widths[2] > 0.0;
}

void fillOrthoH(double lx, double ly, double lz, double H[9], double Hinv[9]) {
  std::fill(H, H + 9, 0.0);
  std::fill(Hinv, Hinv + 9, 0.0);
  H[0] = lx;
  H[4] = ly;
  H[8] = lz;
  Hinv[0] = 1.0 / lx;
  Hinv[4] = 1.0 / ly;
  Hinv[8] = 1.0 / lz;
}

} // namespace

bool available() {
  return CUDART::loaded() && NVRTC::loaded();
}

void buildStencil(int maxReach, std::vector<int> &dx, std::vector<int> &dy,
                  std::vector<int> &dz, std::vector<int> &reachOff) {
  dx.clear();
  dy.clear();
  dz.clear();
  reachOff.assign(static_cast<std::size_t>(maxReach) + 2, 0);
  for (int r = 1; r <= maxReach; ++r) {
    reachOff[static_cast<std::size_t>(r)] = static_cast<int>(dx.size());
    for (int ix = -r; ix <= r; ++ix) {
      for (int iy = -r; iy <= r; ++iy) {
        for (int iz = -r; iz <= r; ++iz) {
          if (r > 1 && std::abs(ix) != r && std::abs(iy) != r &&
              std::abs(iz) != r) {
            continue;
          }
          dx.push_back(ix);
          dy.push_back(iy);
          dz.push_back(iz);
        }
      }
    }
  }
  reachOff[static_cast<std::size_t>(maxReach) + 1] = static_cast<int>(dx.size());
}

struct Workspace::Impl {
  int capN = 0;
  int capC = 0;
  int capSt = 0;
  int capF = 0;
  int cachedReach = -1;
  cudaStream_t stream = nullptr;
  DevPtr dcellOf, dcellCount, dcellOff, dorder, dfolded, dsorted, dhome;
  DevPtr dsdx, dsdy, dsdz, dreachOff;
  DevPtr dH, dHinv, dcmin, dorigin, dplan;
  std::vector<int> hsdx, hsdy, hsdz, hsoff;
  std::vector<double> hH, hHinv, hCmin;

  void ensureStream() {
    if (stream == nullptr) {
      checkCuda(CUDART::instance().cudaStreamCreate(&stream), "stream");
    }
  }

  void sync() {
    if (stream != nullptr) {
      checkCuda(CUDART::instance().cudaStreamSynchronize(stream), "stream wait");
    }
  }

  ~Impl() {
    if (stream != nullptr && CUDART::loaded()) {
      CUDART::instance().cudaStreamSynchronize(stream);
      CUDART::instance().cudaStreamDestroy(stream);
    }
  }

  void ensure(int nTot, int nCtot, int nOff) {
    if (nTot > capN) {
      capN = nTot;
      growOne(dcellOf, static_cast<std::size_t>(capN) * sizeof(int), "cellOf");
      growOne(dorder, static_cast<std::size_t>(capN) * sizeof(int), "order");
      growOne(dhome, static_cast<std::size_t>(capN) * sizeof(int), "home");
      growOne(dfolded, static_cast<std::size_t>(capN) * 3 * sizeof(double),
              "folded");
      growOne(dsorted, static_cast<std::size_t>(capN) * 3 * sizeof(double),
              "sorted");
    }
    if (nCtot > capC) {
      capC = nCtot;
      growOne(dcellCount, static_cast<std::size_t>(capC) * sizeof(int),
              "cellCount");
      growOne(dcellOff, static_cast<std::size_t>(nOff) * sizeof(int),
              "cellOff");
    }
  }

  void ensureBox(int nF) {
    if (nF <= capF && dH.p != nullptr) {
      return;
    }
    capF = nF;
    const std::size_t n9 = static_cast<std::size_t>(capF) * 9 * sizeof(double);
    const std::size_t n1 = static_cast<std::size_t>(capF) * sizeof(double);
    growOne(dH, n9, "H");
    growOne(dHinv, n9, "Hinv");
    growOne(dcmin, n1, "cmin");
    if (dorigin.p == nullptr) {
      growOne(dorigin, 3 * sizeof(double), "origin");
    }
    if (dplan.p == nullptr) {
      growOne(dplan, 5 * sizeof(int), "plan");
    }
  }

  void ensureStencil(int maxReach) {
    if (cachedReach == maxReach && dsdx.p != nullptr) {
      return;
    }
    buildStencil(maxReach, hsdx, hsdy, hsdz, hsoff);
    const int nSt = static_cast<int>(hsdx.size());
    if (nSt > capSt) {
      capSt = nSt;
      growOne(dsdx, static_cast<std::size_t>(capSt) * sizeof(int), "sdx");
      growOne(dsdy, static_cast<std::size_t>(capSt) * sizeof(int), "sdy");
      growOne(dsdz, static_cast<std::size_t>(capSt) * sizeof(int), "sdz");
    }
    growOne(dreachOff, hsoff.size() * sizeof(int), "reachOff");
    ensureStream();
    auto &rt = CUDART::instance();
    checkCuda(rt.cudaMemcpyAsync(dsdx.p, hsdx.data(),
                                 hsdx.size() * sizeof(int),
                                 cudaMemcpyHostToDevice, stream),
              "HtoD sdx");
    checkCuda(rt.cudaMemcpyAsync(dsdy.p, hsdy.data(),
                                 hsdy.size() * sizeof(int),
                                 cudaMemcpyHostToDevice, stream),
              "HtoD sdy");
    checkCuda(rt.cudaMemcpyAsync(dsdz.p, hsdz.data(),
                                 hsdz.size() * sizeof(int),
                                 cudaMemcpyHostToDevice, stream),
              "HtoD sdz");
    checkCuda(rt.cudaMemcpyAsync(dreachOff.p, hsoff.data(),
                                 hsoff.size() * sizeof(int),
                                 cudaMemcpyHostToDevice, stream),
              "HtoD reachOff");
    cachedReach = maxReach;
  }
};

Workspace::Workspace() : impl_(new Impl) {}
Workspace::~Workspace() { delete impl_; }
Workspace::Workspace(Workspace &&o) noexcept : impl_(o.impl_) {
  o.impl_ = nullptr;
}
Workspace &Workspace::operator=(Workspace &&o) noexcept {
  if (this != &o) {
    delete impl_;
    impl_ = o.impl_;
    o.impl_ = nullptr;
  }
  return *this;
}

void *Workspace::queue() {
  impl_->ensureStream();
  return impl_->stream;
}

void Workspace::wait() { impl_->sync(); }

void Workspace::knearest_into(const double *xyz, std::size_t n, const Cell &cell,
                              std::size_t k, int *out, std::size_t out_len,
                              const int *mask, double cell_hint,
                              double *out_d2) {
  knearest_into_many(xyz, n, 1, cell, k, out, out_len, mask, cell_hint, true,
                     nullptr, out_d2);
}

void Workspace::knearest_into_many(const double *xyz, std::size_t n,
                                   std::size_t nFrames, const Cell &cell,
                                   std::size_t k, int *out, std::size_t out_len,
                                   const int *mask, double cell_hint,
                                   bool wait, const double *frameBox,
                                   double *out_d2) {
  if (!available()) {
    throw Error("CUDA driver or nvrtc not loaded");
  }
  if (n == 0 || nFrames == 0) {
    throw Error("no points");
  }
  if (k == 0) {
    throw Error("k must be at least 1");
  }
  if (k > static_cast<std::size_t>(kMaxK)) {
    throw Error("device k-nearest supports k <= 16");
  }
  if (out_len != n * k * nFrames) {
    throw Error("out buffer length must be nFrames * n * k");
  }
  if (xyz == nullptr || out == nullptr) {
    throw Error("null pointer");
  }
  double H0[9], Hinv0[9], widths[3];
  if (!fillH(cell, H0, Hinv0, widths)) {
    throw Error("bad cell");
  }
  if (frameBox != nullptr && !cellIsOrtho(cell)) {
    throw Error("frame boxes on a sheared cell need a shared H");
  }

  double edge = cell_hint;
  if (!(edge > 0.0)) {
    edge = 3.0;
  }
  edge = (std::min)(edge, (std::min)(widths[0], (std::min)(widths[1], widths[2])));
  int nx = static_cast<int>(std::floor(widths[0] / edge));
  int ny = static_cast<int>(std::floor(widths[1] / edge));
  int nz = static_cast<int>(std::floor(widths[2] / edge));
  if (nx < 1) {
    nx = 1;
  }
  if (ny < 1) {
    ny = 1;
  }
  if (nz < 1) {
    nz = 1;
  }
  const int nC = nx * ny * nz;
  if (nC <= 0 || nC > kMaxCells) {
    throw Error("too many cells");
  }
  const double cellMin = (std::min)(
      widths[0] / static_cast<double>(nx),
      (std::min)(widths[1] / static_cast<double>(ny),
                 widths[2] / static_cast<double>(nz)));
  const int maxReach = (std::max)(nx, (std::max)(ny, nz)) / 2 + 1;
  impl_->ensureStream();
  auto &rt = CUDART::instance();
  const int nF = static_cast<int>(nFrames);
  const std::size_t fSz = static_cast<std::size_t>(nF);
  impl_->ensureBox(nF);
  impl_->hH.assign(fSz * 9, 0.0);
  impl_->hHinv.assign(fSz * 9, 0.0);
  impl_->hCmin.assign(fSz, cellMin);
  for (int f = 0; f < nF; ++f) {
    std::copy(H0, H0 + 9, impl_->hH.begin() + static_cast<std::size_t>(f) * 9);
    std::copy(Hinv0, Hinv0 + 9,
              impl_->hHinv.begin() + static_cast<std::size_t>(f) * 9);
  }
  if (frameBox != nullptr) {
    for (int f = 0; f < nF; ++f) {
      const double flx = frameBox[static_cast<std::size_t>(f) * 3 + 0];
      const double fly = frameBox[static_cast<std::size_t>(f) * 3 + 1];
      const double flz = frameBox[static_cast<std::size_t>(f) * 3 + 2];
      if (!(flx > 0.0 && fly > 0.0 && flz > 0.0)) {
        throw Error("bad frame box");
      }
      const int fnx = (std::max)(1, static_cast<int>(std::floor(flx / edge)));
      const int fny = (std::max)(1, static_cast<int>(std::floor(fly / edge)));
      const int fnz = (std::max)(1, static_cast<int>(std::floor(flz / edge)));
      if (fnx != nx || fny != ny || fnz != nz) {
        throw Error("frame boxes need the same cell grid");
      }
      fillOrthoH(flx, fly, flz,
                 impl_->hH.data() + static_cast<std::size_t>(f) * 9,
                 impl_->hHinv.data() + static_cast<std::size_t>(f) * 9);
      impl_->hCmin[static_cast<std::size_t>(f)] = (std::min)(
          flx / static_cast<double>(nx),
          (std::min)(fly / static_cast<double>(ny), flz / static_cast<double>(nz)));
    }
  }
  checkCuda(rt.cudaMemcpyAsync(impl_->dH.p, impl_->hH.data(),
                               fSz * 9 * sizeof(double), cudaMemcpyHostToDevice,
                               impl_->stream),
            "HtoD H");
  checkCuda(rt.cudaMemcpyAsync(impl_->dHinv.p, impl_->hHinv.data(),
                               fSz * 9 * sizeof(double), cudaMemcpyHostToDevice,
                               impl_->stream),
            "HtoD Hinv");
  checkCuda(rt.cudaMemcpyAsync(impl_->dcmin.p, impl_->hCmin.data(),
                               fSz * sizeof(double), cudaMemcpyHostToDevice,
                               impl_->stream),
            "HtoD cmin");
  const double origin_h[3] = {cell.origin[0], cell.origin[1], cell.origin[2]};
  checkCuda(rt.cudaMemcpyAsync(impl_->dorigin.p, origin_h, 3 * sizeof(double),
                               cudaMemcpyHostToDevice, impl_->stream),
            "HtoD origin");
  launchWalk(xyz, n, nFrames, k, out, mask, wait, out_d2, nx, ny, nz, nC,
             maxReach);
}

void Workspace::knearest_into_many_dcell(
    const double *xyz, std::size_t n, std::size_t nFrames, const double *cell,
    int cell_n, std::size_t k, int *out, std::size_t out_len, const int *mask,
    double cell_hint, bool wait, double *out_d2) {
  if (!available()) {
    throw Error("CUDA driver or nvrtc not loaded");
  }
  if (n == 0 || nFrames == 0) {
    throw Error("no points");
  }
  if (k == 0) {
    throw Error("k must be at least 1");
  }
  if (k > static_cast<std::size_t>(kMaxK)) {
    throw Error("device k-nearest supports k <= 16");
  }
  if (out_len != n * k * nFrames) {
    throw Error("out buffer length must be nFrames * n * k");
  }
  if (xyz == nullptr || out == nullptr || cell == nullptr) {
    throw Error("null pointer");
  }
  if (cell_n != 3 && cell_n != 9 && cell_n != 12) {
    throw Error("device cell must have 3, 9, or 12 doubles");
  }
  impl_->ensureStream();
  impl_->ensureBox(static_cast<int>(nFrames));
  auto &rt = CUDART::instance();
  auto &factory = KernelFactory::instance(0);
  const std::vector<std::string> opt{"-std=c++17"};
  auto *kInv = factory.create("invert_cell", kKernels, "linkcell.cu", opt);
  int nF = static_cast<int>(nFrames);
  int cellN = cell_n;
  void *cellP = const_cast<double *>(cell);
  double hint = cell_hint;
  {
    void *raw[] = {&cellP,          &cellN,         &nF,
                   &impl_->dH.p,    &impl_->dHinv.p, &impl_->dorigin.p,
                   &impl_->dcmin.p, &hint,           &impl_->dplan.p};
    std::vector<void *> a(raw, raw + sizeof(raw) / sizeof(raw[0]));
    kInv->launch(dim3(1), dim3(1), 0, impl_->stream, a, false);
  }
  int plan[5] = {0, 0, 0, 0, 0};
  checkCuda(rt.cudaMemcpyAsync(plan, impl_->dplan.p, 5 * sizeof(int),
                               cudaMemcpyDeviceToHost, impl_->stream),
            "DtoH plan");
  impl_->sync();
  if (plan[4] == 0) {
    throw Error("bad cell");
  }
  if (plan[4] != 1) {
    throw Error("too many cells");
  }
  const int nx = plan[0];
  const int ny = plan[1];
  const int nz = plan[2];
  const int nC = plan[3];
  const int maxReach = (std::max)(nx, (std::max)(ny, nz)) / 2 + 1;
  launchWalk(xyz, n, nFrames, k, out, mask, wait, out_d2, nx, ny, nz, nC,
             maxReach);
}

void Workspace::launchWalk(const double *xyz, std::size_t n,
                           std::size_t nFrames, std::size_t k, int *out,
                           const int *mask, bool wait, double *out_d2, int nx,
                           int ny, int nz, int nC, int maxReach) {
  int nI = static_cast<int>(n);
  int nF = static_cast<int>(nFrames);
  int kI = static_cast<int>(k);
  const int nTot = nI * nF;
  const int nCtot = nC * nF;
  const int nOff = (nC + 1) * nF;
  impl_->ensure(nTot, nCtot, nOff);
  impl_->ensureStencil(maxReach);
  impl_->ensureStream();

  auto &factory = KernelFactory::instance(0);
  const std::vector<std::string> opt{"-std=c++17"};
  auto *kZero = factory.create("zero_i32", kKernels, "linkcell.cu", opt);
  auto *kBin = factory.create("bin_atoms", kKernels, "linkcell.cu", opt);
  auto *kPref = factory.create("prefix_cells", kKernels, "linkcell.cu", opt);
  auto *kScat = factory.create("scatter_atoms", kKernels, "linkcell.cu", opt);
  auto *kNear = factory.create("knearest_shells", kKernels, "linkcell.cu", opt);

  const std::size_t kSlot = static_cast<std::size_t>(kI);
  auto shBytes = [&](int blk, int tpp) {
    const int ppb = blk / tpp;
    return static_cast<std::size_t>(ppb) * static_cast<std::size_t>(tpp) *
               kSlot * sizeof(double) +
           static_cast<std::size_t>(ppb) * static_cast<std::size_t>(tpp) *
               kSlot * sizeof(int) +
           static_cast<std::size_t>(ppb) * static_cast<std::size_t>(tpp) *
               sizeof(int) +
           static_cast<std::size_t>(ppb) * sizeof(int);
  };
  int maxThreads = 1024;
  try {
    const int attr = kNear->getFuncAttribute(
        CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
    if (attr >= 32) {
      maxThreads = attr;
    }
  } catch (const std::exception &) {
  }
  constexpr std::size_t kMaxSh = 48ull * 1024ull;
  // HOOMD times TPP 4/8 first; vesin hardcodes TPP=8. Prefer
  // occupant parallelism, then a 128/256 block that still fits
  // shmem. Maximising particles per block forces TPP=1.
  int tpp = 4;
  int block = 128;
  bool picked = false;
  const int tppCands[] = {8, 4, 2, 1};
  const int blkCands[] = {256, 128, 64, 512};
  for (int tp : tppCands) {
    for (int blk : blkCands) {
      if (blk > maxThreads || blk % 32 != 0 || blk % tp != 0) {
        continue;
      }
      if (shBytes(blk, tp) > kMaxSh) {
        continue;
      }
      tpp = tp;
      block = blk;
      picked = true;
      break;
    }
    if (picked) {
      break;
    }
  }
  if (const char *envTpp = std::getenv("LINKCELL_TPP")) {
    const int v = std::atoi(envTpp);
    if (v >= 1 && v <= 32 && block % v == 0) {
      tpp = v;
    }
  }
  if (const char *envBlk = std::getenv("LINKCELL_BLOCK")) {
    const int v = std::atoi(envBlk);
    if (v >= 32 && v <= maxThreads && v % 32 == 0 && v % tpp == 0 &&
        shBytes(v, tpp) <= kMaxSh) {
      block = v;
    }
  }
  const int grid = (nTot + block - 1) / block;
  const int tppGrid = (nTot * tpp + block - 1) / block;
  const std::size_t sh = shBytes(block, tpp);
  void *xyzP = const_cast<double *>(xyz);
  void *maskV = const_cast<int *>(mask);
  void *outP = out;
  void *outD2 = out_d2;
  int nCv = nC;
  int maxR = maxReach;
  auto launchArgs = [](void **raw, std::size_t n) {
    return std::vector<void *>(raw, raw + n);
  };
  {
    int nZero = nCtot;
    const int zGrid = (nZero + 255) / 256;
    void *raw[] = {&impl_->dcellCount.p, &nZero};
    auto a = launchArgs(raw, sizeof(raw) / sizeof(raw[0]));
    kZero->launch(dim3(zGrid), dim3(256), 0, impl_->stream, a, false);
  }
  {
    void *raw[] = {&xyzP,
                   &maskV,
                   &nI,
                   &nF,
                   &impl_->dH.p,
                   &impl_->dHinv.p,
                   &impl_->dorigin.p,
                   &nx,
                   &ny,
                   &nz,
                   &nCv,
                   &impl_->dcellOf.p,
                   &impl_->dcellCount.p,
                   &impl_->dfolded.p};
    auto a = launchArgs(raw, sizeof(raw) / sizeof(raw[0]));
    kBin->launch(dim3(grid), dim3(block), 0, impl_->stream, a, false);
  }
  {
    void *raw[] = {&impl_->dcellCount.p, &impl_->dcellOff.p, &nCv, &nF};
    auto a = launchArgs(raw, sizeof(raw) / sizeof(raw[0]));
    kPref->launch(dim3(nF), dim3(256), 256 * sizeof(int), impl_->stream, a,
                  false);
  }
  {
    void *raw[] = {&impl_->dfolded.p,    &impl_->dcellOf.p, &impl_->dcellCount.p,
                   &impl_->dcellOff.p,   &impl_->dorder.p,  &impl_->dsorted.p,
                   &impl_->dhome.p,      &nI,               &nF,
                   &nCv};
    auto a = launchArgs(raw, sizeof(raw) / sizeof(raw[0]));
    kScat->launch(dim3(grid), dim3(block), 0, impl_->stream, a, false);
  }
  {
    void *raw[] = {&impl_->dsorted.p,   &impl_->dcellOf.p, &impl_->dcellOff.p,
                   &impl_->dorder.p,    &impl_->dhome.p,   &impl_->dsdx.p,
                   &impl_->dsdy.p,      &impl_->dsdz.p,    &impl_->dreachOff.p,
                   &nI,                 &nF,               &nCv,
                   &nx,                 &ny,               &nz,
                   &impl_->dH.p,        &impl_->dcmin.p,   &kI,
                   &maxR,               &tpp,              &outP,
                   &outD2};
    auto a = launchArgs(raw, sizeof(raw) / sizeof(raw[0]));
    kNear->launch(dim3(tppGrid), dim3(block), sh, impl_->stream, a, false);
  }
  if (wait) {
    impl_->sync();
  }
}

} // namespace gpu
} // namespace linkcell

#endif
