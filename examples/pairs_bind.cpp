// C and C++ cutoff-list probe: the calls seams-core and rgpot make.
//
// Env: PAIRS_N (default 4096), PAIRS_REPS (default 8), PAIRS_HALF (0).
// RAYON_NUM_THREADS is read when the library starts. `cpp` is
// linkcell::pairs_within (count, then fill). `c` is one
// lc_pairs_within fill into buffers sized from the first call.
#include "linkcell.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

std::size_t env_size(const char *key, std::size_t fallback) {
  const char *raw = std::getenv(key);
  return raw ? static_cast<std::size_t>(std::strtoull(raw, nullptr, 10))
             : fallback;
}

std::vector<double> fill(std::size_t n) {
  const double boxl = 18.0;
  std::size_t m = static_cast<std::size_t>(std::ceil(std::cbrt(double(n))));
  while (m * m * m < n) {
    ++m;
  }
  std::vector<double> xyz;
  xyz.reserve(n * 3);
  std::size_t count = 0;
  for (std::size_t iz = 0; iz < m && count < n; ++iz) {
    for (std::size_t iy = 0; iy < m && count < n; ++iy) {
      for (std::size_t ix = 0; ix < m && count < n; ++ix) {
        xyz.push_back((double(ix) + 0.5) * boxl / double(m));
        xyz.push_back((double(iy) + 0.5) * boxl / double(m));
        xyz.push_back((double(iz) + 0.5) * boxl / double(m));
        ++count;
      }
    }
  }
  return xyz;
}

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - t0)
      .count();
}

} // namespace

int main() {
  const std::size_t n = env_size("PAIRS_N", 4096);
  const std::size_t reps = std::max<std::size_t>(1, env_size("PAIRS_REPS", 8));
  const bool half = env_size("PAIRS_HALF", 0) == 1;
  const char *threads_env = std::getenv("RAYON_NUM_THREADS");
  const std::string threads = threads_env ? threads_env : "default";
  const auto xyz = fill(n);
  const auto cell = linkcell::Cell::ortho(18.0, 18.0, 18.0);

  double early[3];
  std::size_t rows = 0;
  for (double &slot : early) {
    const auto t = std::chrono::steady_clock::now();
    rows = linkcell::pairs_within(xyz.data(), n, cell, 4.0, nullptr, 0.0, half)
               .size();
    slot = ms_since(t);
  }
  std::size_t acc = 0;
  auto t0 = std::chrono::steady_clock::now();
  for (std::size_t r = 0; r < reps; ++r) {
    acc += linkcell::pairs_within(xyz.data(), n, cell, 4.0, nullptr, 0.0, half)
               .size();
  }
  const double cpp_ms = ms_since(t0) / double(reps);
  std::printf("cpp n=%zu half=%d threads=%s pairs=%zu cold=%.3f call2=%.3f "
              "call3=%.3f reps=%zu ms_per=%.4f acc=%zu\n",
              n, int(half), threads.c_str(), rows, early[0], early[1], early[2],
              reps, cpp_ms, acc);

  const lc_cell raw = cell.raw();
  std::vector<int> ii(rows), jj(rows), ss(rows * 3);
  std::vector<double> dd(rows);
  std::size_t wrote = 0;
  for (int warm = 0; warm < 3; ++warm) {
    lc_pairs_within(xyz.data(), n, &raw, 4.0, nullptr, 0.0, half ? 1 : 0,
                    ii.data(), jj.data(), ss.data(), dd.data(), rows, &wrote);
  }
  acc = 0;
  t0 = std::chrono::steady_clock::now();
  for (std::size_t r = 0; r < reps; ++r) {
    const int status =
        lc_pairs_within(xyz.data(), n, &raw, 4.0, nullptr, 0.0, half ? 1 : 0,
                        ii.data(), jj.data(), ss.data(), dd.data(), rows, &wrote);
    if (status != 0) {
      std::fprintf(stderr, "lc_pairs_within: %s\n", lc_last_error());
      return 1;
    }
    acc += wrote;
  }
  const double c_ms = ms_since(t0) / double(reps);
  std::printf("c n=%zu half=%d threads=%s pairs=%zu reps=%zu ms_per=%.4f acc=%zu\n",
              n, int(half), threads.c_str(), wrote, reps, c_ms, acc);
  return 0;
}
