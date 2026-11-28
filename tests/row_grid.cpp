// The C++ row grid over the C ABI: every row against the pair list on
// orthorhombic, triclinic, skewed, mixed, and open cells, split ranges
// and single points against the whole walk, threads on disjoint ranges,
// early stops, exceptions through a visit, and the error text.
// Exit 0 on success; a failure prints its line and exits 1.
#include "linkcell.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#define CHECK(cond)                                                          \
  do {                                                                       \
    if (!(cond)) {                                                           \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__,  \
                   #cond);                                                   \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

namespace {

using Key = std::tuple<std::size_t, std::size_t, std::array<int, 3>,
                       std::uint64_t>;
using Points = std::vector<std::array<double, 3>>;

std::uint64_t bits(double v) {
  std::uint64_t out = 0;
  std::memcpy(&out, &v, sizeof out);
  return out;
}

Key key_of(const linkcell::Row &row) {
  return Key{row.i, row.j, row.shift, bits(row.dist2)};
}

struct Rng {
  std::uint64_t state;
  double next() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return static_cast<double>(state >> 11) / 9007199254740992.0;
  }
};

Points points(const linkcell::Cell &cell, std::size_t n, double lo, double hi,
              std::uint64_t seed) {
  Rng rng{seed};
  Points xyz(n);
  for (auto &p : xyz) {
    const double s[3] = {lo + (hi - lo) * rng.next(),
                         lo + (hi - lo) * rng.next(),
                         lo + (hi - lo) * rng.next()};
    for (std::size_t a = 0; a < 3; ++a) {
      p[a] = s[0] * cell.a[a] + s[1] * cell.b[a] + s[2] * cell.c[a];
    }
  }
  return xyz;
}

std::vector<Key> visited(const linkcell::RowGrid &grid, bool half) {
  std::vector<Key> rows;
  const bool whole = grid.for_rows(
      0, grid.size(), half,
      [&](const linkcell::Row &row) { rows.push_back(key_of(row)); });
  CHECK(whole);
  return rows;
}

// The rows of the grid, half and full, are the pair list's rows; single
// points, split ranges, and threads on disjoint ranges are the same walk.
void parity(const char *name, const linkcell::Cell &cell,
            std::array<bool, 3> periodic, std::size_t n, double lo, double hi,
            double cutoff, std::uint64_t seed) {
  const Points xyz = points(cell, n, lo, hi, seed);
  linkcell::RowQuery query;
  query.cutoff = cutoff;
  query.periodic = periodic;
  const linkcell::RowGrid grid(xyz, cell, query);
  CHECK(grid.size() == n);
  for (const bool half : {false, true}) {
    std::vector<Key> want;
    for (const auto &pair : linkcell::pairs_within_pbc(
             xyz.front().data(), n, cell, periodic, cutoff, nullptr, 0.0,
             half)) {
      want.emplace_back(static_cast<std::size_t>(pair.i),
                        static_cast<std::size_t>(pair.j),
                        std::array<int, 3>{pair.shift[0], pair.shift[1],
                                           pair.shift[2]},
                        bits(pair.dist2));
    }
    CHECK(!want.empty());
    const std::vector<Key> whole = visited(grid, half);
    CHECK(grid.bins() > 0);
    CHECK(half || grid.candidate_bound() >= whole.size());
    std::vector<Key> sorted = whole;
    std::sort(sorted.begin(), sorted.end());
    std::sort(want.begin(), want.end());
    if (sorted != want) {
      std::fprintf(stderr, "%s half=%d: rows differ from the pair list\n",
                   name, half ? 1 : 0);
      std::exit(1);
    }

    // The vector is r_j - r_i with the shift applied.
    double scale = 1.0;
    for (const auto &p : xyz) {
      for (const double v : p) {
        scale = std::max(scale, std::abs(v));
      }
    }
    grid.for_rows(0, n, half, [&](const linkcell::Row &row) {
      for (std::size_t a = 0; a < 3; ++a) {
        const double image = row.shift[0] * cell.a[a] +
                             row.shift[1] * cell.b[a] + row.shift[2] * cell.c[a];
        const double expect = xyz[row.j][a] + image - xyz[row.i][a];
        CHECK(std::abs(row.vector[a] - expect) <= 1.0e-12 * scale);
      }
      const double squared = row.vector[0] * row.vector[0] +
                             row.vector[1] * row.vector[1] +
                             row.vector[2] * row.vector[2];
      CHECK(std::abs(squared - row.dist2) <= 1.0e-12 * cutoff * cutoff);
    });

    std::vector<Key> single;
    for (std::size_t i = 0; i < n; ++i) {
      grid.for_point(i, half, [&](const linkcell::Row &row) {
        CHECK(row.i == i);
        single.push_back(key_of(row));
      });
    }
    CHECK(single == whole);

    // The same entries through a buffer, 41 at a time.
    std::vector<linkcell::Row> block;
    std::vector<Key> filled;
    linkcell::RowCursor cursor;
    while (cursor.point < n) {
      block.clear();
      const std::size_t got = grid.fill_rows(cursor, n, half, block, 41);
      CHECK(got == block.size() && got <= 41);
      CHECK(got > 0 || cursor.point == n);
      for (const auto &row : block) {
        filled.push_back(key_of(row));
      }
    }
    CHECK(filled == whole);

    for (const std::size_t threads : {2u, 5u, 8u}) {
      std::vector<std::vector<Key>> parts(threads);
      std::vector<std::thread> team;
      for (std::size_t t = 0; t < threads; ++t) {
        team.emplace_back([&, t] {
          const std::size_t begin = t * n / threads;
          const std::size_t end = (t + 1) * n / threads;
          const std::size_t mid = begin + (end - begin) / 2;
          grid.for_rows(begin, mid, half, [&](const linkcell::Row &row) {
            parts[t].push_back(key_of(row));
          });
          for (std::size_t i = mid; i < end; ++i) {
            grid.for_point(i, half, [&](const linkcell::Row &row) {
              parts[t].push_back(key_of(row));
            });
          }
        });
      }
      for (auto &thread : team) {
        thread.join();
      }
      std::vector<Key> joined;
      for (const auto &part : parts) {
        joined.insert(joined.end(), part.begin(), part.end());
      }
      CHECK(joined == whole);
    }

    // A visit that returns false stops the walk there.
    std::size_t calls = 0;
    const bool finished = grid.for_rows(0, n, half, [&](const linkcell::Row &) {
      ++calls;
      return calls < 4;
    });
    CHECK(!finished);
    CHECK(calls == 4);
    // An exception from a visit stops the walk and comes out unchanged.
    calls = 0;
    bool caught = false;
    try {
      grid.for_rows(0, n, half, [&](const linkcell::Row &) {
        if (++calls == 2) {
          throw std::runtime_error("from the visit");
        }
      });
    } catch (const std::runtime_error &err) {
      caught = std::string(err.what()) == "from the visit";
    }
    CHECK(caught);
    CHECK(calls == 2);
    std::printf("%s half=%d: %zu rows\n", name, half ? 1 : 0, whole.size());
  }
}

template <typename F> std::string error_of(F &&call) {
  try {
    call();
  } catch (const linkcell::Error &err) {
    CHECK(err.status() != 0);
    return err.what();
  }
  return {};
}

} // namespace

int main() {
  // The header this file compiled against and the library agree.
  CHECK(linkcell::abi_compatible());
  CHECK(linkcell::library_abi() == linkcell::header_abi());
  const linkcell::Capabilities caps = linkcell::capabilities();
  CHECK(caps.row_grid && caps.row_vector && caps.row_stop && caps.open_axes &&
        caps.row_fill && caps.closed_periodic && caps.row_bound);

  // A pair exactly at the cutoff, directly and through a periodic image.
  {
    const auto box = linkcell::Cell::ortho(10.0, 12.0, 14.0);
    const Points at = {{1.0, 1.0, 1.0}, {3.0, 1.0, 1.0}, {9.0, 1.0, 1.0}};
    linkcell::RowQuery strict;
    strict.cutoff = 2.0;
    linkcell::RowQuery closed = strict;
    closed.closed = true;
    CHECK(visited(linkcell::RowGrid(at, box, strict), true).empty());
    CHECK(visited(linkcell::RowGrid(at, box, closed), true).size() == 2);
  }

  const auto ortho = linkcell::Cell::ortho(15.0, 16.0, 14.0);
  const auto tiny = linkcell::Cell::ortho(6.0, 6.5, 7.0);
  const auto tilt = linkcell::Cell::from_vectors(
      {15.0, 0.0, 0.0}, {4.5, 15.0, 0.0}, {-3.0, 3.75, 15.0});
  const auto skew = linkcell::Cell::from_vectors(
      {14.0, 2.94, -1.82}, {-4.76, 13.02, 2.38}, {3.78, -5.74, 15.12});
  const std::array<bool, 3> ppp{true, true, true};
  parity("ortho", ortho, ppp, 300, 0.0, 1.0, 3.5, 11);
  parity("ortho outside", ortho, ppp, 300, -1.3, 2.4, 3.5, 12);
  parity("ortho past half", tiny, ppp, 60, 0.0, 1.0, 4.0, 13);
  parity("triclinic", tilt, ppp, 300, -0.2, 1.2, 3.5, 14);
  parity("skewed", skew, ppp, 300, -1.1, 2.1, 3.5, 15);
  parity("mixed ppf", tilt, {true, true, false}, 200, -0.5, 1.5, 3.5, 16);
  parity("mixed fpf", skew, {false, true, false}, 200, -0.5, 1.5, 3.5, 17);
  parity("open", ortho, {false, false, false}, 200, 0.0, 1.0, 3.5, 18);

  // Errors carry the library's text.
  const Points xyz = points(ortho, 50, 0.0, 1.0, 21);
  linkcell::RowQuery query;
  query.cutoff = 3.0;
  linkcell::RowGrid grid(xyz, ortho, query);
  const auto nothing = [](const linkcell::Row &) {};
  CHECK(error_of([&] { grid.for_rows(0, 51, false, nothing); }) ==
        "point index is out of range");
  CHECK(error_of([&] { grid.for_point(50, false, nothing); }) ==
        "point index is out of range");
  CHECK(grid.for_rows(7, 7, false, nothing));
  std::vector<linkcell::Row> none;
  linkcell::RowCursor past;
  CHECK(error_of([&] { grid.fill_rows(past, 51, false, none, 8); }) ==
        "point index is out of range");
  linkcell::RowQuery bad = query;
  bad.cutoff = -1.0;
  CHECK(error_of([&] { linkcell::RowGrid(xyz, ortho, bad); }) ==
        "cutoff must be finite and strictly positive");
  CHECK(error_of([&] { linkcell::RowGrid(Points{}, ortho, query); }) ==
        "no points");
  // A failed update leaves no rows; the next one fills the grid again.
  CHECK(!error_of([&] { grid.update(xyz, ortho, bad); }).empty());
  CHECK(grid.size() == 0);
  grid.update(xyz, ortho, query);
  CHECK(grid.size() == 50);
  // A moved grid keeps its rows, and the source holds none.
  const std::vector<Key> before = visited(grid, false);
  linkcell::RowGrid moved(std::move(grid));
  CHECK(visited(moved, false) == before);
  CHECK(grid.size() == 0); // NOLINT(bugprone-use-after-move)
  grid = std::move(moved);
  CHECK(visited(grid, false) == before);
  // Bytes that are no message are refused, with or without request support.
  const std::uint8_t junk[24] = {0xff, 0xff, 0xff, 0xff};
  const std::string refused = error_of([&] {
    (void)linkcell::RowGrid::from_request(xyz.front().data(), xyz.size(), junk,
                                          sizeof junk);
  });
  CHECK(!refused.empty());
  CHECK(caps.row_request ||
        refused == "this library was built without NeighborRequest support");
  std::puts("row_grid_cpp: ok");
  return 0;
}
