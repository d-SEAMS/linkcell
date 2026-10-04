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
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace linkcell {

/// Failure from a neighbour search. `status()` is the C ABI return
/// (0 success, nonzero failure).
class Error : public std::runtime_error {
public:
  Error(int status, std::string message)
      : std::runtime_error(message.empty() ? "linkcell: knearest failed"
                                           : std::move(message)),
        status_(status) {}

  explicit Error(std::string message) : Error(1, std::move(message)) {}

  [[nodiscard]] int status() const noexcept { return status_; }

private:
  int status_;
};

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
  const lc_cell raw = cell.raw();
  std::size_t count = 0;
  const int status = lc_pairs_within(
      xyz, detail::to_c_count(n, "n"), &raw, cutoff, mask, cell_hint,
      half ? 1 : 0, nullptr, nullptr, nullptr, nullptr, 0, &count);
  if (status != 0) {
    const char *msg = lc_last_error();
    throw Error(status, msg ? std::string(msg)
                            : std::string("linkcell: pairs_within failed"));
  }
  if (count == 0) {
    return {};
  }
  std::vector<int> ii(count);
  std::vector<int> jj(count);
  std::vector<int> shift(count * 3);
  std::vector<double> d2(count);
  std::size_t wrote = 0;
  const int fill = lc_pairs_within(
      xyz, detail::to_c_count(n, "n"), &raw, cutoff, mask, cell_hint,
      half ? 1 : 0, ii.data(), jj.data(), shift.data(), d2.data(), count,
      &wrote);
  if (fill != 0) {
    const char *msg = lc_last_error();
    throw Error(fill, msg ? std::string(msg)
                          : std::string("linkcell: pairs_within failed"));
  }
  std::vector<ShiftedPair> out(wrote);
  for (std::size_t t = 0; t < wrote; ++t) {
    out[t].i = ii[t];
    out[t].j = jj[t];
    out[t].shift = {shift[3 * t], shift[3 * t + 1], shift[3 * t + 2]};
    out[t].dist2 = d2[t];
  }
  return out;
}

[[nodiscard]] inline const char *version() noexcept { return lc_version(); }

} // namespace linkcell

#endif
