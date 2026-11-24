// linkcell::knearest, knearest_into, knearest_into_many and knearest_csr
// throw linkcell::Error with the NonFinite message for a NaN or infinite
// coordinate or cell origin.
#include "linkcell.hpp"

#include <array>
#include <cstdio>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace {

const std::string kWant = "a coordinate is not finite";
int failures = 0;

void expect_throw(const char *what, double bad, int axis,
                  const std::function<void()> &call) {
  try {
    call();
  } catch (const linkcell::Error &err) {
    if (err.status() != 0 && kWant == err.what()) {
      return;
    }
    std::fprintf(stderr, "%s bad=%g axis=%d: status=%d what=%s\n", what, bad,
                 axis, err.status(), err.what());
    failures++;
    return;
  }
  std::fprintf(stderr, "%s bad=%g axis=%d: no throw\n", what, bad, axis);
  failures++;
}

std::vector<std::array<double, 3>> points() {
  return {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 2.0, 0.0}};
}

} // namespace

int main() {
  const double bads[3] = {std::numeric_limits<double>::quiet_NaN(),
                          std::numeric_limits<double>::infinity(),
                          -std::numeric_limits<double>::infinity()};
  const auto box = linkcell::Cell::ortho(10.0, 10.0, 10.0);
  const std::size_t n = 3;
  const std::size_t k = 2;
  std::vector<int> offsets;
  std::vector<int> cols;

  (void)linkcell::knearest(points(), box, k);
  linkcell::knearest_csr(points().front().data(), n, box, k, false, offsets,
                         cols);

  for (double bad : bads) {
    for (int axis = 0; axis < 3; axis++) {
      auto xyz = points();
      xyz[1][axis] = bad;
      const double *p = xyz.front().data();
      std::vector<int> out(n * k);
      expect_throw("knearest", bad, axis,
                   [&] { (void)linkcell::knearest(xyz, box, k); });
      expect_throw("knearest_into", bad, axis, [&] {
        linkcell::knearest_into(p, n, box, k, out.data(), out.size());
      });
      for (bool mutual : {false, true}) {
        expect_throw("knearest_csr", bad, axis, [&] {
          linkcell::knearest_csr(p, n, box, k, mutual, offsets, cols);
        });
      }

      auto frames = points();
      auto second = points();
      second[2][axis] = bad;
      frames.insert(frames.end(), second.begin(), second.end());
      std::vector<int> many(2 * n * k);
      expect_throw("knearest_into_many", bad, axis, [&] {
        linkcell::knearest_into_many(frames.front().data(), n, 2, box, k,
                                     many.data(), many.size());
      });

      std::array<double, 3> origin{};
      origin[axis] = bad;
      const auto shifted = linkcell::Cell::from_vectors(
          {10.0, 0.0, 0.0}, {0.0, 10.0, 0.0}, {0.0, 0.0, 10.0}, origin);
      expect_throw("knearest origin", bad, axis,
                   [&] { (void)linkcell::knearest(points(), shifted, k); });
    }
  }
  if (failures != 0) {
    std::fprintf(stderr, "%d failures\n", failures);
    return 1;
  }
  std::printf("non-finite: ok\n");
  return 0;
}
