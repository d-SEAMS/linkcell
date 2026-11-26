#ifndef LINKCELL_HPP
#define LINKCELL_HPP

#if defined(_MSC_VER)
#if !defined(_MSVC_LANG) || _MSVC_LANG < 201703L
#error "linkcell.hpp requires C++17 or later"
#endif
#elif defined(__cplusplus) && __cplusplus < 201703L
#error "linkcell.hpp requires C++17 or later"
#endif

extern "C" {
#include "linkcell.h"
}

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace linkcell {

// The destructor in src/error.cpp is the key function and lives in the
// shared library linkcell_eh, so every image sees one typeinfo. Windows
// exports it from that library and imports it everywhere else.
// LINKCELL_EH_STATIC is the fully static link: one copy in each final image.
#if defined(LINKCELL_EH_STATIC)
#define LINKCELL_EH_API
#elif defined(_WIN32)
#if defined(LINKCELL_EH_BUILDING)
#define LINKCELL_EH_API __declspec(dllexport)
#else
#define LINKCELL_EH_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define LINKCELL_EH_API __attribute__((visibility("default")))
#else
#define LINKCELL_EH_API
#endif

/// Failure from a neighbour search. `status()` is the C ABI return
/// (0 success, nonzero failure).
/// libc++ matches a `catch` by the typeinfo address. This class has one
/// address because its destructor is the key function.
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4275) // std::runtime_error has no dll interface
#endif
class LINKCELL_EH_API Error : public std::runtime_error {
public:
  Error(int status, std::string message);
  explicit Error(std::string message);
  Error(const Error &);
  Error(Error &&) noexcept;
  Error &operator=(const Error &);
  Error &operator=(Error &&) noexcept;
  ~Error() override;

  [[nodiscard]] int status() const noexcept;

private:
  int status_;
};
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
#undef LINKCELL_EH_API

/// Periodic parallelepiped. Fields match `lc_cell` (lattice vectors
/// a, b, c and dump-cell origin).
struct Cell {
  std::array<double, 3> a{};
  std::array<double, 3> b{};
  std::array<double, 3> c{};
  std::array<double, 3> origin{};

  Cell() = default;

  explicit Cell(const lc_cell &raw) noexcept
      : a{raw.ax, raw.ay, raw.az}, b{raw.bx, raw.by, raw.bz},
        c{raw.cx, raw.cy, raw.cz}, origin{raw.ox, raw.oy, raw.oz} {}

  static Cell ortho(double lx, double ly, double lz) noexcept {
    Cell cell;
    cell.a = {lx, 0.0, 0.0};
    cell.b = {0.0, ly, 0.0};
    cell.c = {0.0, 0.0, lz};
    return cell;
  }

  static Cell from_vectors(std::array<double, 3> a, std::array<double, 3> b,
                           std::array<double, 3> c,
                           std::array<double, 3> origin = {}) noexcept {
    Cell cell;
    cell.a = a;
    cell.b = b;
    cell.c = c;
    cell.origin = origin;
    return cell;
  }

  [[nodiscard]] lc_cell raw() const noexcept {
    return lc_cell{a[0],        a[1],        a[2],        b[0],
                   b[1],        b[2],        c[0],        c[1],
                   c[2],        origin[0],   origin[1],   origin[2]};
  }

  [[nodiscard]] explicit operator lc_cell() const noexcept { return raw(); }
};

/// Owned packed n-by-k neighbour indices.
///
/// `data()[i * k() + j]` is the j-th neighbour of source i, nearest
/// first. The slot is -1 when that neighbour is missing (masked or
/// isolated).
class Neighbours {
public:
  Neighbours() = default;

  Neighbours(std::vector<int> idx, std::size_t n, std::size_t k)
      : Neighbours(std::move(idx),
                   std::vector<double>(n * k, std::numeric_limits<double>::quiet_NaN()),
                   n, k) {}

  Neighbours(std::vector<int> idx, std::vector<double> d2, std::size_t n,
             std::size_t k)
      : idx_(std::move(idx)), d2_(std::move(d2)), n_(n), k_(k) {
    if (k != 0 && n > std::numeric_limits<std::size_t>::max() / k) {
      throw Error("n * k overflows");
    }
    if (idx_.size() != n * k || d2_.size() != n * k) {
      throw Error("out buffer length must be n * k");
    }
  }

  /// j-th neighbour of source i, or -1 if missing.
  [[nodiscard]] int neighbour(std::size_t i, std::size_t j) const {
    if (i >= n_ || j >= k_) {
      throw std::out_of_range("linkcell: neighbour index out of range");
    }
    return idx_[i * k_ + j];
  }

  /// Squared distance of neighbour (i, j). Unused slots are NaN.
  [[nodiscard]] double dist2(std::size_t i, std::size_t j) const {
    if (i >= n_ || j >= k_) {
      throw std::out_of_range("linkcell: neighbour index out of range");
    }
    return d2_[i * k_ + j];
  }

  [[nodiscard]] std::size_t n() const noexcept { return n_; }
  [[nodiscard]] std::size_t k() const noexcept { return k_; }
  [[nodiscard]] std::size_t size() const noexcept { return n_ * k_; }
  [[nodiscard]] const int *data() const noexcept { return idx_.data(); }
  [[nodiscard]] const double *distances() const noexcept { return d2_.data(); }
  [[nodiscard]] const int *begin() const noexcept { return idx_.data(); }
  [[nodiscard]] const int *end() const noexcept { return idx_.data() + size(); }

private:
  std::vector<int> idx_;
  std::vector<double> d2_;
  std::size_t n_ = 0;
  std::size_t k_ = 0;
};

namespace detail {

template <typename R, typename Xyz, typename Count, typename... Rest>
Count count_arg(R (*)(Xyz, Count, Rest...));

// C ABI counts are size_t. Convert through the declared parameter
// type and refuse truncation if a future header regresses.
using lc_count = decltype(count_arg(&lc_knearest));

static_assert(std::is_integral<lc_count>::value,
              "lc_knearest count parameter must be an integral type");

inline lc_count to_c_count(std::size_t v, const char *what) {
  const auto maxv =
      static_cast<std::size_t>(std::numeric_limits<lc_count>::max());
  if (v > maxv) {
    throw Error(std::string("linkcell: ") + what +
                " exceeds the C ABI count range");
  }
  return static_cast<lc_count>(v);
}

inline void call_c(const double *xyz, std::size_t n, const Cell &cell,
                   std::size_t k, int *out, const int *mask, double cell_hint,
                   double *out_d2 = nullptr, std::size_t n_frames = 1) {
  if (n == 0 || n_frames == 0) {
    throw Error("no points");
  }
  if (k == 0) {
    throw Error("k must be at least 1");
  }
  if (xyz == nullptr || out == nullptr) {
    throw Error("null pointer");
  }
  const lc_cell raw = cell.raw();
  const int status = lc_knearest_many(
      xyz, to_c_count(n, "n"), to_c_count(n_frames, "n_frames"), &raw,
      to_c_count(k, "k"), mask, cell_hint, out, out_d2);
  if (status != 0) {
    const char *msg = lc_last_error();
    throw Error(status, msg ? std::string(msg)
                            : std::string("linkcell: knearest failed"));
  }
}

} // namespace detail

/// Write packed indices into caller-owned `out` of length `out_len`.
/// `out_len` must be `n * k`. `mask` is nullptr or n ints.
inline void knearest_into(const double *xyz, std::size_t n, const Cell &cell,
                          std::size_t k, int *out, std::size_t out_len,
                          const int *mask = nullptr, double cell_hint = 0.0,
                          double *out_d2 = nullptr) {
  if (k != 0 && n > std::numeric_limits<std::size_t>::max() / k) {
    throw Error("n * k overflows");
  }
  if (out_len != n * k) {
    throw Error("out buffer length must be n * k");
  }
  detail::call_c(xyz, n, cell, k, out, mask, cell_hint, out_d2);
}

/// Frame-major batch. `xyz` is `n_frames * n * 3`. `out` / `out_d2`
/// are `n_frames * n * k`.
inline void knearest_into_many(const double *xyz, std::size_t n,
                               std::size_t n_frames, const Cell &cell,
                               std::size_t k, int *out, std::size_t out_len,
                               const int *mask = nullptr,
                               double cell_hint = 0.0,
                               double *out_d2 = nullptr) {
  if (k != 0 && n_frames != 0 &&
      n > std::numeric_limits<std::size_t>::max() / k / n_frames) {
    throw Error("n * k * n_frames overflows");
  }
  if (out_len != n * k * n_frames) {
    throw Error("out buffer length must be n_frames * n * k");
  }
  detail::call_c(xyz, n, cell, k, out, mask, cell_hint, out_d2, n_frames);
}

/// The k-nearest graph as CSR into `offsets` (n + 1) and `cols`; see
/// `lc_knearest_csr` for the row order. `mask` is nullptr or n ints.
inline void knearest_csr(const double *xyz, std::size_t n, const Cell &cell,
                         std::size_t k, bool mutual, std::vector<int> &offsets,
                         std::vector<int> &cols, const int *mask = nullptr,
                         double cell_hint = 0.0) {
  if (n == 0) {
    throw Error("no points");
  }
  if (k == 0) {
    throw Error("k must be at least 1");
  }
  if (xyz == nullptr) {
    throw Error("null pointer");
  }
  offsets.resize(n + 1);
  struct Sink {
    std::vector<int> *cols;
    static int *grow(void *raw, std::size_t total) noexcept {
      auto *self = static_cast<Sink *>(raw);
      try {
        self->cols->resize(total);
      } catch (...) {
        return nullptr;
      }
      return self->cols->data();
    }
  } sink{&cols};
  const lc_cell raw = cell.raw();
  const int status =
      lc_knearest_csr(xyz, detail::to_c_count(n, "n"), &raw,
                      detail::to_c_count(k, "k"), mask, cell_hint, mutual ? 1 : 0,
                      offsets.data(), &Sink::grow, &sink);
  if (status != 0) {
    const char *msg = lc_last_error();
    throw Error(status, msg ? std::string(msg)
                            : std::string("linkcell: knearest_csr failed"));
  }
}

/// One allocation of n * k indices. `neighbour(i, j)` is the j-th
/// neighbour of i, or -1.
[[nodiscard]] inline Neighbours
knearest(const double *xyz, std::size_t n, const Cell &cell, std::size_t k,
         const int *mask = nullptr, double cell_hint = 0.0) {
  if (k != 0 && n > std::numeric_limits<std::size_t>::max() / k) {
    throw Error("n * k overflows");
  }
  std::vector<int> out(n * k, -1);
  std::vector<double> d2(n * k, std::numeric_limits<double>::quiet_NaN());
  detail::call_c(xyz, n, cell, k, out.data(), mask, cell_hint, d2.data());
  return Neighbours(std::move(out), std::move(d2), n, k);
}

/// `std::vector<std::array<double, 3>>` is already packed n * 3 doubles.
[[nodiscard]] inline Neighbours
knearest(const std::vector<std::array<double, 3>> &xyz, const Cell &cell,
         std::size_t k, const int *mask = nullptr, double cell_hint = 0.0) {
  const double *ptr = xyz.empty() ? nullptr : xyz.front().data();
  return knearest(ptr, xyz.size(), cell, k, mask, cell_hint);
}

/// One atom-image inside a cutoff. `shift` is vesin / tonari `S`.
/// Displacement is `r_j - r_i + S H` in the caller's basis.
struct ShiftedPair {
  int i = 0;
  int j = 0;
  std::array<int, 3> shift{{0, 0, 0}};
  double dist2 = 0.0;
};

/// Pairs with squared distance strictly below `cutoff * cutoff`.
///
/// `half` keeps one side of `(i, j, S)` and `(j, i, -S)`. This is the
/// cutoff list. k-nearest is [`knearest`].
[[nodiscard]] inline std::vector<ShiftedPair>
pairs_within(const double *xyz, std::size_t n, const Cell &cell, double cutoff,
             const int *mask = nullptr, double cell_hint = 0.0,
             bool half = false) {
  if (n == 0 || xyz == nullptr) {
    throw Error("no points");
  }
  static_assert(sizeof(ShiftedPair) == sizeof(lc_pair) &&
                    alignof(ShiftedPair) == alignof(lc_pair) &&
                    offsetof(ShiftedPair, j) == offsetof(lc_pair, j) &&
                    offsetof(ShiftedPair, shift) == offsetof(lc_pair, shift) &&
                    offsetof(ShiftedPair, dist2) == offsetof(lc_pair, dist2),
                "ShiftedPair must match lc_pair");
  const lc_cell raw = cell.raw();
  std::size_t count = 0;
  // The query keeps its search on this thread; the fill below writes
  // the rows from it.
  const int status = lc_pairs_within_rows(
      xyz, detail::to_c_count(n, "n"), &raw, cutoff, mask, cell_hint,
      half ? 1 : 0, nullptr, 0, &count);
  if (status != 0) {
    const char *msg = lc_last_error();
    throw Error(status, msg ? std::string(msg)
                            : std::string("linkcell: pairs_within failed"));
  }
  if (count == 0) {
    return {};
  }
  std::vector<ShiftedPair> out(count);
  std::size_t wrote = 0;
  const int fill = lc_pairs_within_rows(
      xyz, detail::to_c_count(n, "n"), &raw, cutoff, mask, cell_hint,
      half ? 1 : 0, reinterpret_cast<lc_pair *>(out.data()), count, &wrote);
  if (fill != 0) {
    const char *msg = lc_last_error();
    throw Error(fill, msg ? std::string(msg)
                          : std::string("linkcell: pairs_within failed"));
  }
  out.resize(wrote);
  return out;
}

/// How many partner stencils have been built in this process.
[[nodiscard]] inline std::uint64_t stencil_builds() noexcept {
  return lc_stencil_builds();
}

/// Cartesian half pair list of an open cluster.
///
/// Each row has `i < j`, shift zero, and squared distance strictly
/// below `cutoff * cutoff`. Bins are the occupied Cartesian cells. Their
/// partner stencil depends on the cutoff and `cell_hint` alone, so it is
/// reused by frames that occupy other bins.
[[nodiscard]] inline std::vector<ShiftedPair>
pairs_open(const double *xyz, std::size_t n, double cutoff,
           double cell_hint = 0.0) {
  if (n == 0 || xyz == nullptr) {
    return {};
  }
  static_assert(sizeof(ShiftedPair) == sizeof(lc_pair) &&
                    alignof(ShiftedPair) == alignof(lc_pair),
                "ShiftedPair must match lc_pair");
  std::size_t count = 0;
  const int status = lc_pairs_open(xyz, detail::to_c_count(n, "n"), cutoff,
                                   cell_hint, nullptr, 0, &count);
  if (status != 0) {
    const char *msg = lc_last_error();
    throw Error(status, msg ? std::string(msg)
                            : std::string("linkcell: pairs_open failed"));
  }
  if (count == 0) {
    return {};
  }
  std::vector<ShiftedPair> out(count);
  std::size_t wrote = 0;
  const int fill =
      lc_pairs_open(xyz, detail::to_c_count(n, "n"), cutoff, cell_hint,
                    reinterpret_cast<lc_pair *>(out.data()), count, &wrote);
  if (fill != 0) {
    const char *msg = lc_last_error();
    throw Error(fill, msg ? std::string(msg)
                          : std::string("linkcell: pairs_open failed"));
  }
  out.resize(wrote);
  return out;
}

/// [`pairs_within`] with a periodicity mask.
///
/// `periodic[k]` false is an open axis. `closed` keeps the cutoff
/// boundary on an open or mixed cell. `exclude_self_images` drops
/// `i == j`.
[[nodiscard]] inline std::vector<ShiftedPair>
pairs_within_pbc(const double *xyz, std::size_t n, const Cell &cell,
                 const std::array<bool, 3> &periodic, double cutoff,
                 const int *mask = nullptr, double cell_hint = 0.0,
                 bool half = false, bool closed = false,
                 bool exclude_self_images = false) {
  if (n == 0 || xyz == nullptr) {
    return {};
  }
  const int pbc[3] = {periodic[0] ? 1 : 0, periodic[1] ? 1 : 0,
                      periodic[2] ? 1 : 0};
  const lc_cell raw = cell.raw();
  std::size_t count = 0;
  const int status = lc_pairs_within_pbc(
      xyz, detail::to_c_count(n, "n"), &raw, pbc, cutoff, mask, cell_hint,
      half ? 1 : 0, closed ? 1 : 0, exclude_self_images ? 1 : 0, nullptr, 0,
      &count);
  if (status != 0) {
    const char *msg = lc_last_error();
    throw Error(status, msg ? std::string(msg)
                            : std::string("linkcell: pairs_within_pbc failed"));
  }
  if (count == 0) {
    return {};
  }
  std::vector<ShiftedPair> out(count);
  std::size_t wrote = 0;
  const int fill = lc_pairs_within_pbc(
      xyz, detail::to_c_count(n, "n"), &raw, pbc, cutoff, mask, cell_hint,
      half ? 1 : 0, closed ? 1 : 0, exclude_self_images ? 1 : 0,
      reinterpret_cast<lc_pair *>(out.data()), count, &wrote);
  if (fill != 0) {
    const char *msg = lc_last_error();
    throw Error(fill, msg ? std::string(msg)
                          : std::string("linkcell: pairs_within_pbc failed"));
  }
  out.resize(wrote);
  return out;
}

/// Cutoff neighbours as CSR rows; see `lc_pairs_within_csr`. Row `i` is
/// `cols[offsets[i]] .. cols[offsets[i + 1] - 1]`, ascending, each
/// neighbour once. `dist2` is filled when `with_dist2` is set.
struct PairCsr {
  std::vector<int> offsets;
  std::vector<int> cols;
  std::vector<double> dist2;
};

/// `lc_pairs_within_csr` into `out`, keeping its capacity. One call
/// writes the rows when `out.cols` already has room for them; otherwise
/// a second call copies the rows the first kept.
inline void pairs_within_csr(const double *xyz, std::size_t n,
                             const Cell &cell, double cutoff, PairCsr &out,
                             bool half = false,
                             const std::array<bool, 3> &periodic = {true, true,
                                                                    true},
                             const int *mask = nullptr, double cell_hint = 0.0,
                             bool with_dist2 = false) {
  const int pbc[3] = {periodic[0] ? 1 : 0, periodic[1] ? 1 : 0,
                      periodic[2] ? 1 : 0};
  const lc_cell raw = cell.raw();
  out.offsets.resize(n + 1);
  out.cols.resize(out.cols.capacity());
  if (with_dist2) {
    out.dist2.resize(out.cols.size());
  } else {
    out.dist2.clear();
  }
  std::size_t need_offsets = 0;
  std::size_t need_cols = 0;
  // A non-null buffer of zero length still takes part in the fill.
  int spare_col = 0;
  double spare_d2 = 0.0;
  const auto call = [&]() {
    int *cols = out.cols.empty() ? &spare_col : out.cols.data();
    double *d2 = !with_dist2 ? nullptr
                 : out.dist2.empty() ? &spare_d2
                                     : out.dist2.data();
    return lc_pairs_within_csr(xyz, n, &raw, pbc, cutoff, mask, cell_hint,
                               half ? 1 : 0, out.offsets.data(),
                               out.offsets.size(), cols, d2, out.cols.size(),
                               &need_offsets, &need_cols);
  };
  int status = call();
  if (status != 0 && need_offsets == n + 1 && need_cols > out.cols.size()) {
    out.cols.resize(need_cols);
    if (with_dist2) {
      out.dist2.resize(need_cols);
    }
    status = call();
  }
  if (status != 0) {
    const char *msg = lc_last_error();
    throw Error(status, msg ? std::string(msg)
                            : std::string("linkcell: pairs_within_csr failed"));
  }
  out.cols.resize(need_cols);
  if (with_dist2) {
    out.dist2.resize(need_cols);
  }
}

[[nodiscard]] inline const char *version() noexcept { return lc_version(); }

/// Fractional bins and a point count per bin.
struct CellGrid {
  int nx = 0;
  int ny = 0;
  int nz = 0;
  std::array<double, 3> bin_width{};
  std::vector<std::uint64_t> weight;

  [[nodiscard]] std::size_t ncell() const noexcept { return weight.size(); }
};

/// Periodic adjacency. Each undirected pair is two arcs of one weight.
struct CellGraph {
  CellGrid grid;
  std::vector<std::uint32_t> offsets;
  std::vector<std::uint32_t> neighbor;
  std::vector<std::uint64_t> edge_weight;
};

/// Domain id of each cell. `method` matches `lc_partition`.
/// `balance <= 0` selects `1 + parts * w_max / total`.
/// `w_max` is the heaviest cell weight. `device` asks for the CUDA curve split.
[[nodiscard]] inline std::vector<int>
partition(const double *xyz, std::size_t n, const Cell &cell, double cutoff,
          std::size_t parts, int method = 4, const int *mask = nullptr,
          double cell_hint = 0.0, double balance = 0.0, bool device = false) {
  if (n == 0 || xyz == nullptr) {
    throw Error("no points");
  }
  const lc_cell raw = cell.raw();
  std::size_t ncell = 0;
  const int query = lc_partition(
      xyz, detail::to_c_count(n, "n"), &raw, cutoff, mask, cell_hint, parts,
      static_cast<std::uint8_t>(method), balance, device ? 1 : 0, nullptr, 0,
      &ncell);
  if (query != 0) {
    const char *msg = lc_last_error();
    throw Error(query, msg ? std::string(msg) : std::string("linkcell: partition failed"));
  }
  std::vector<int> out(ncell);
  std::size_t wrote = 0;
  const int fill = lc_partition(
      xyz, detail::to_c_count(n, "n"), &raw, cutoff, mask, cell_hint, parts,
      static_cast<std::uint8_t>(method), balance, device ? 1 : 0, out.data(),
      ncell, &wrote);
  if (fill != 0) {
    const char *msg = lc_last_error();
    throw Error(fill, msg ? std::string(msg) : std::string("linkcell: partition failed"));
  }
  out.resize(wrote);
  return out;
}

/// Bin radii for `cutoff * depth`.
[[nodiscard]] inline std::array<std::uint32_t, 3>
halo_width(double cutoff, std::uint32_t depth,
           const std::array<double, 3> &bin_width) {
  std::array<std::uint32_t, 3> out{};
  const int status = lc_halo_width(cutoff, depth, bin_width.data(), out.data());
  if (status != 0) {
    const char *msg = lc_last_error();
    throw Error(status, msg ? std::string(msg) : std::string("linkcell: halo width failed"));
  }
  return out;
}

/// Ghost cell ids of `domain`.
[[nodiscard]] inline std::vector<int>
halo(const std::array<int, 3> &nxyz, const std::vector<int> &assign, int domain,
     const std::array<std::uint32_t, 3> &width) {
  std::size_t count = 0;
  const int query =
      lc_halo(nxyz.data(), assign.data(), assign.size(), domain, width.data(),
              nullptr, 0, &count);
  if (query != 0) {
    const char *msg = lc_last_error();
    throw Error(query, msg ? std::string(msg) : std::string("linkcell: halo failed"));
  }
  std::vector<int> out(count);
  if (count == 0) {
    return out;
  }
  std::size_t wrote = 0;
  const int fill =
      lc_halo(nxyz.data(), assign.data(), assign.size(), domain, width.data(),
              out.data(), count, &wrote);
  if (fill != 0) {
    const char *msg = lc_last_error();
    throw Error(fill, msg ? std::string(msg) : std::string("linkcell: halo failed"));
  }
  out.resize(wrote);
  return out;
}

} // namespace linkcell

#endif
