#include "knearest.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {

using L = Kokkos::LayoutRight;
using Exec = Kokkos::DefaultExecutionSpace;
using Mem = typename Exec::memory_space;
using View3 = Kokkos::View<double**, L, Mem>;
using View3i = Kokkos::View<int**, L, Mem>;
using View1i = Kokkos::View<int*, Mem>;

constexpr double kCertRel = 1.0e-8;
constexpr double kCertAbs = 1.0e-12;
constexpr int kMaxCells = 16777216;

struct Geom {
    double col[3][3];
    double hinv[3][3];
    double origin[3];
    double widths[3];
    int nx;
    int ny;
    int nz;
    int mode;
    int k;
    int max_reach;
};

KOKKOS_INLINE_FUNCTION double wrap01(double s) {
    s -= Kokkos::floor(s);
    if (s >= 1.0) {
        return 0.0;
    }
    return s;
}

KOKKOS_INLINE_FUNCTION int bin_coord(double s, int n) {
    int c = static_cast<int>(s * static_cast<double>(n));
    if (c < 0) {
        c = 0;
    }
    if (c >= n) {
        c = n - 1;
    }
    return c;
}

KOKKOS_INLINE_FUNCTION void split_axis(int i, int n, int& coord, int& image) {
    if (i >= 0 && i < n) {
        coord = i;
        image = 0;
        return;
    }
    int q = i / n;
    int r = i % n;
    if (r < 0) {
        r += n;
        q -= 1;
    }
    coord = r;
    image = q;
}

KOKKOS_INLINE_FUNCTION int cell_index(int ix, int iy, int iz, int nx, int ny, int nz) {
    int cx, cy, cz, na, nb, nc;
    split_axis(ix, nx, cx, na);
    split_axis(iy, ny, cy, nb);
    split_axis(iz, nz, cz, nc);
    return (cz * ny + cy) * nx + cx;
}

KOKKOS_INLINE_FUNCTION double axis_gap(double s, int bin, int reach, int n, double w) {
    double nf = static_cast<double>(n);
    double plus = (static_cast<double>(bin + reach + 1) / nf - s) * w;
    double minus = (s - static_cast<double>(bin - reach) / nf) * w;
    return plus < minus ? plus : minus;
}

KOKKOS_INLINE_FUNCTION double certify(double dist) {
    if (!(dist > 0.0) || !Kokkos::isfinite(dist)) {
        return 0.0;
    }
    double d = dist - dist * kCertRel - kCertAbs;
    return d > 0.0 ? d : 0.0;
}

KOKKOS_INLINE_FUNCTION double slab_dist2(double s0, double s1, double s2, int jx, int jy, int jz,
                                         int nx, int ny, int nz, double w0, double w1, double w2) {
    int unfolded[3] = {jx, jy, jz};
    int n[3] = {nx, ny, nz};
    double s[3] = {s0, s1, s2};
    double w[3] = {w0, w1, w2};
    double gap = 0.0;
    for (int a = 0; a < 3; ++a) {
        double nf = static_cast<double>(n[a]);
        double lo = static_cast<double>(unfolded[a]) / nf;
        double hi = static_cast<double>(unfolded[a] + 1) / nf;
        double axis = 0.0;
        if (s[a] < lo) {
            axis = (lo - s[a]) * w[a];
        } else if (s[a] > hi) {
            axis = (s[a] - hi) * w[a];
        }
        if (axis > gap) {
            gap = axis;
        }
    }
    double d = certify(gap);
    return d * d;
}

struct Heap {
    double d2[16];
    int idx[16];
    int n;
    int k;
    int worst_at;

    KOKKOS_INLINE_FUNCTION Heap() : n(0), k(0), worst_at(0) {
        for (int t = 0; t < 16; ++t) {
            d2[t] = 0.0;
            idx[t] = 0;
        }
    }

    KOKKOS_INLINE_FUNCTION bool worse(double dist, int j, int other) const {
        double od = d2[other];
        int oj = idx[other];
        return dist > od || (dist == od && j > oj);
    }

    KOKKOS_INLINE_FUNCTION void recompute() {
        int w = 0;
        for (int t = 1; t < n; ++t) {
            if (worse(d2[t], idx[t], w)) {
                w = t;
            }
        }
        worst_at = w;
    }

    KOKKOS_INLINE_FUNCTION void push(double dist, int j) {
        if (n == k) {
            double wd = d2[worst_at];
            int wj = idx[worst_at];
            if (dist > wd || (dist == wd && j >= wj)) {
                return;
            }
        }
        for (int t = 0; t < n; ++t) {
            if (idx[t] == j) {
                if (dist < d2[t]) {
                    d2[t] = dist;
                    if (t == worst_at) {
                        recompute();
                    }
                }
                return;
            }
        }
        if (n < k) {
            int slot = n;
            d2[slot] = dist;
            idx[slot] = j;
            if (slot == 0 || worse(dist, j, worst_at)) {
                worst_at = slot;
            }
            n += 1;
            return;
        }
        d2[worst_at] = dist;
        idx[worst_at] = j;
        recompute();
    }

    KOKKOS_INLINE_FUNCTION bool full() const { return n >= k; }
    KOKKOS_INLINE_FUNCTION double worst() const { return d2[worst_at]; }
};

KOKKOS_INLINE_FUNCTION void write_sorted(const Heap& heap, int i, View3i out_nn, View3 out_d2,
                                         bool write_d2) {
    int order[16];
    for (int t = 0; t < heap.n; ++t) {
        order[t] = t;
    }
    for (int a = 1; a < heap.n; ++a) {
        int key = order[a];
        int b = a;
        while (b > 0) {
            int prev = order[b - 1];
            bool less = heap.d2[key] < heap.d2[prev] ||
                        (heap.d2[key] == heap.d2[prev] && heap.idx[key] < heap.idx[prev]);
            if (!less) {
                break;
            }
            order[b] = prev;
            b -= 1;
        }
        order[b] = key;
    }
    for (int t = 0; t < heap.n; ++t) {
        int s = order[t];
        out_nn(i, t) = heap.idx[s];
        if (write_d2) {
            out_d2(i, t) = heap.d2[s];
        }
    }
}

struct Fill {
    View3 xyz;
    View3 frac;
    View3 folded;
    View3i bin;
    View1i counts;
    Geom g;

    KOKKOS_INLINE_FUNCTION void operator()(int i) const {
        double d0 = xyz(i, 0) - g.origin[0];
        double d1 = xyz(i, 1) - g.origin[1];
        double d2 = xyz(i, 2) - g.origin[2];
        double s0, s1, s2;
        if (g.mode == 0) {
            s0 = wrap01(d0 / g.widths[0]);
            s1 = wrap01(d1 / g.widths[1]);
            s2 = wrap01(d2 / g.widths[2]);
        } else {
            s0 = g.hinv[0][0] * d0 + g.hinv[1][0] * d1 + g.hinv[2][0] * d2;
            s1 = g.hinv[0][1] * d0 + g.hinv[1][1] * d1 + g.hinv[2][1] * d2;
            s2 = g.hinv[0][2] * d0 + g.hinv[1][2] * d1 + g.hinv[2][2] * d2;
            s0 = wrap01(s0);
            s1 = wrap01(s1);
            s2 = wrap01(s2);
        }
        double f0 = g.col[0][0] * s0 + g.col[1][0] * s1 + g.col[2][0] * s2 + g.origin[0];
        double f1 = g.col[0][1] * s0 + g.col[1][1] * s1 + g.col[2][1] * s2 + g.origin[1];
        double f2 = g.col[0][2] * s0 + g.col[1][2] * s1 + g.col[2][2] * s2 + g.origin[2];
        int ix = bin_coord(s0, g.nx);
        int iy = bin_coord(s1, g.ny);
        int iz = bin_coord(s2, g.nz);
        frac(i, 0) = s0;
        frac(i, 1) = s1;
        frac(i, 2) = s2;
        folded(i, 0) = f0;
        folded(i, 1) = f1;
        folded(i, 2) = f2;
        bin(i, 0) = ix;
        bin(i, 1) = iy;
        bin(i, 2) = iz;
        int c = cell_index(ix, iy, iz, g.nx, g.ny, g.nz);
        Kokkos::atomic_fetch_add(&counts(c), 1);
    }
};

struct ExclusiveScan {
    View1i counts;
    View1i offsets;

    KOKKOS_INLINE_FUNCTION void operator()(int i, int& update, bool final) const {
        int val = counts(i);
        if (final) {
            offsets(i) = update;
        }
        update += val;
    }
    KOKKOS_INLINE_FUNCTION void init(int& update) const { update = 0; }
    KOKKOS_INLINE_FUNCTION void join(int& dst, const int& src) const { dst += src; }
};

struct Scatter {
    View3i bin;
    View1i cursor;
    View1i occupants;
    int nx, ny, nz;

    KOKKOS_INLINE_FUNCTION void operator()(int i) const {
        int ix = bin(i, 0);
        int iy = bin(i, 1);
        int iz = bin(i, 2);
        int c = cell_index(ix, iy, iz, nx, ny, nz);
        int dest = Kokkos::atomic_fetch_add(&cursor(c), 1);
        occupants(dest) = i;
    }
};

template <int Mode>
struct Walk {
    View3 frac;
    View3 folded;
    View3i bin;
    View1i offsets;
    View1i occupants;
    View3i out_nn;
    View3 out_d2;
    Geom g;
    bool write_d2;

    KOKKOS_INLINE_FUNCTION void visit(Heap& heap, int i, int jx, int jy, int jz, bool allow_slab) const {
        int cx, cy, cz, na, nb, nc;
        split_axis(jx, g.nx, cx, na);
        split_axis(jy, g.ny, cy, nb);
        split_axis(jz, g.nz, cz, nc);
        int cell = (cz * g.ny + cy) * g.nx + cx;
        int lo = offsets(cell);
        int hi = offsets(cell + 1);
        int count = hi - lo;
        if (count == 0) {
            return;
        }
        if (allow_slab && !(Mode == 0 && count == 1) && heap.full()) {
            double lb = slab_dist2(frac(i, 0), frac(i, 1), frac(i, 2), jx, jy, jz, g.nx, g.ny,
                                    g.nz, g.widths[0], g.widths[1], g.widths[2]);
            if (heap.worst() <= lb) {
                return;
            }
        }
        double shift0 = 0.0;
        double shift1 = 0.0;
        double shift2 = 0.0;
        if ((na | nb | nc) != 0) {
            double fa = static_cast<double>(na);
            double fb = static_cast<double>(nb);
            double fc = static_cast<double>(nc);
            if (Mode == 0) {
                shift0 = fa * g.widths[0];
                shift1 = fb * g.widths[1];
                shift2 = fc * g.widths[2];
            } else if (Mode == 1) {
                shift0 = fa * g.col[0][0] + fb * g.col[1][0] + fc * g.col[2][0];
                shift1 = fb * g.col[1][1] + fc * g.col[2][1];
                shift2 = fc * g.col[2][2];
            } else {
                shift0 = fa * g.col[0][0] + fb * g.col[1][0] + fc * g.col[2][0];
                shift1 = fa * g.col[0][1] + fb * g.col[1][1] + fc * g.col[2][1];
                shift2 = fa * g.col[0][2] + fb * g.col[1][2] + fc * g.col[2][2];
            }
        }
        double pi0 = folded(i, 0);
        double pi1 = folded(i, 1);
        double pi2 = folded(i, 2);
        for (int s = lo; s < hi; ++s) {
            int ju = occupants(s);
            if (ju == i) {
                continue;
            }
            double dx = folded(ju, 0) + shift0 - pi0;
            double dy = folded(ju, 1) + shift1 - pi1;
            double dz = folded(ju, 2) + shift2 - pi2;
            heap.push(dx * dx + dy * dy + dz * dz, ju);
        }
    }

    KOKKOS_INLINE_FUNCTION void operator()(int i) const {
        Heap heap;
        heap.k = g.k;
        int ix = bin(i, 0);
        int iy = bin(i, 1);
        int iz = bin(i, 2);
        int prev0 = -1, prev1 = -1, prev2 = -1;
        int reach0 = 1, reach1 = 1, reach2 = 1;
        for (;;) {
            bool allow_slab = prev0 >= 0;
            for (int dz = -reach2; dz <= reach2; ++dz) {
                for (int dy = -reach1; dy <= reach1; ++dy) {
                    for (int dx = -reach0; dx <= reach0; ++dx) {
                        if (dx < 0 ? -dx <= prev0 : dx <= prev0) {
                            if (dy < 0 ? -dy <= prev1 : dy <= prev1) {
                                if (dz < 0 ? -dz <= prev2 : dz <= prev2) {
                                    continue;
                                }
                            }
                        }
                        visit(heap, i, ix + dx, iy + dy, iz + dz, allow_slab);
                    }
                }
            }
            double gaps[3] = {
                axis_gap(frac(i, 0), ix, reach0, g.nx, g.widths[0]),
                axis_gap(frac(i, 1), iy, reach1, g.ny, g.widths[1]),
                axis_gap(frac(i, 2), iz, reach2, g.nz, g.widths[2]),
            };
            if (heap.full()) {
                double bound = 1.0e300;
                for (int a = 0; a < 3; ++a) {
                    double gap = gaps[a];
                    double gap2 = (gap > 0.0 && Kokkos::isfinite(gap)) ? gap * gap : 0.0;
                    if (gap2 < bound) {
                        bound = gap2;
                    }
                }
                if (heap.worst() <= bound) {
                    break;
                }
            }
            prev0 = reach0;
            prev1 = reach1;
            prev2 = reach2;
            bool grew = false;
            if (heap.full()) {
                double worst = heap.worst();
                int reach[3] = {reach0, reach1, reach2};
                for (int a = 0; a < 3; ++a) {
                    double gap = gaps[a];
                    double gap2 = (gap > 0.0 && Kokkos::isfinite(gap)) ? gap * gap : 0.0;
                    if (worst > gap2 && reach[a] < g.max_reach) {
                        reach[a] += 1;
                        grew = true;
                    }
                }
                reach0 = reach[0];
                reach1 = reach[1];
                reach2 = reach[2];
            } else {
                if (reach0 < g.max_reach) {
                    reach0 += 1;
                    grew = true;
                }
                if (reach1 < g.max_reach) {
                    reach1 += 1;
                    grew = true;
                }
                if (reach2 < g.max_reach) {
                    reach2 += 1;
                    grew = true;
                }
            }
            if (!grew) {
                break;
            }
        }
        write_sorted(heap, i, out_nn, out_d2, write_d2);
    }
};

int bins_1d(double width, double edge) {
    double n = std::floor(width / edge);
    if (n < 1.0) {
        n = 1.0;
    }
    if (!std::isfinite(n) || n > 1000000.0) {
        return -1;
    }
    return static_cast<int>(n);
}

double box_diameter(const double col[3][3]) {
    double best = 0.0;
    for (int sa = -1; sa <= 1; sa += 2) {
        for (int sb = -1; sb <= 1; sb += 2) {
            for (int sc = -1; sc <= 1; sc += 2) {
                double x = sa * col[0][0] + sb * col[1][0] + sc * col[2][0];
                double y = sa * col[0][1] + sb * col[1][1] + sc * col[2][1];
                double z = sa * col[0][2] + sb * col[1][2] + sc * col[2][2];
                double d2 = x * x + y * y + z * z;
                if (d2 > best) {
                    best = d2;
                }
            }
        }
    }
    return std::sqrt(best);
}

bool invert(const double col[3][3], double hinv[3][3]) {
    const double* a = col[0];
    const double* b = col[1];
    const double* c = col[2];
    double det = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) +
                 a[2] * (b[0] * c[1] - b[1] * c[0]);
    if (!std::isfinite(det) || std::fabs(det) < 1e-18) {
        return false;
    }
    double invdet = 1.0 / det;
    hinv[0][0] = (b[1] * c[2] - b[2] * c[1]) * invdet;
    hinv[0][1] = (a[2] * c[1] - a[1] * c[2]) * invdet;
    hinv[0][2] = (a[1] * b[2] - a[2] * b[1]) * invdet;
    hinv[1][0] = (b[2] * c[0] - b[0] * c[2]) * invdet;
    hinv[1][1] = (a[0] * c[2] - a[2] * c[0]) * invdet;
    hinv[1][2] = (a[2] * b[0] - a[0] * b[2]) * invdet;
    hinv[2][0] = (b[0] * c[1] - b[1] * c[0]) * invdet;
    hinv[2][1] = (a[1] * c[0] - a[0] * c[1]) * invdet;
    hinv[2][2] = (a[0] * b[1] - a[1] * b[0]) * invdet;
    return true;
}

void widths_of(const double col[3][3], double widths[3]) {
    auto cross = [](const double* u, const double* v, double* o) {
        o[0] = u[1] * v[2] - u[2] * v[1];
        o[1] = u[2] * v[0] - u[0] * v[2];
        o[2] = u[0] * v[1] - u[1] * v[0];
    };
    auto norm = [](const double* v) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); };
    double bc[3], ca[3], ab[3];
    cross(col[1], col[2], bc);
    cross(col[2], col[0], ca);
    cross(col[0], col[1], ab);
    double det = col[0][0] * bc[0] + col[0][1] * bc[1] + col[0][2] * bc[2];
    double ad = std::fabs(det);
    widths[0] = ad / norm(bc);
    widths[1] = ad / norm(ca);
    widths[2] = ad / norm(ab);
}

template <int Mode>
void launch_walk(const Geom& g, View3 frac, View3 folded, View3i bin, View1i offsets,
                 View1i occupants, View3i out_nn, View3 out_d2, bool write_d2) {
    Walk<Mode> walk{frac, folded, bin, offsets, occupants, out_nn, out_d2, g, write_d2};
    Kokkos::parallel_for(Kokkos::RangePolicy<Exec>(0, static_cast<int>(frac.extent(0))), walk);
    Kokkos::fence();
}

}  // namespace

int lc_kokkos_knearest(const double* xyz, int n, const LcKokkosBox& box, int k, double cell_hint,
                       int* out_nn, double* out_d2) {
    if (!Kokkos::is_initialized() || xyz == nullptr || out_nn == nullptr || n <= 1 || k <= 0 ||
        k > 16 || box.mode < 0 || box.mode > 2) {
        return 1;
    }
    Geom g{};
    g.col[0][0] = box.a[0];
    g.col[0][1] = box.a[1];
    g.col[0][2] = box.a[2];
    g.col[1][0] = box.b[0];
    g.col[1][1] = box.b[1];
    g.col[1][2] = box.b[2];
    g.col[2][0] = box.c[0];
    g.col[2][1] = box.c[1];
    g.col[2][2] = box.c[2];
    g.origin[0] = box.origin[0];
    g.origin[1] = box.origin[1];
    g.origin[2] = box.origin[2];
    g.mode = box.mode;
    g.k = k;
    widths_of(g.col, g.widths);
    if (!(g.widths[0] > 0.0 && g.widths[1] > 0.0 && g.widths[2] > 0.0)) {
        return 1;
    }
    if (box.mode != 0 && !invert(g.col, g.hinv)) {
        return 1;
    }
    double edge = cell_hint;
    if (!std::isfinite(edge) || edge <= 0.0) {
        edge = 3.0;
    }
    edge = std::min(edge, std::min(g.widths[0], std::min(g.widths[1], g.widths[2])));
    g.nx = bins_1d(g.widths[0], edge);
    g.ny = bins_1d(g.widths[1], edge);
    g.nz = bins_1d(g.widths[2], edge);
    if (g.nx < 1 || g.ny < 1 || g.nz < 1) {
        return 2;
    }
    long long ncell64 = 1LL * g.nx * g.ny * g.nz;
    if (ncell64 <= 0 || ncell64 > kMaxCells) {
        return 2;
    }
    int ncell = static_cast<int>(ncell64);
    double cell_min = std::min(g.widths[0] / g.nx, std::min(g.widths[1] / g.ny, g.widths[2] / g.nz));
    double need = std::ceil(box_diameter(g.col) / cell_min);
    int image_reach = std::max(g.nx, std::max(g.ny, g.nz)) / 2 + 2;
    if (std::isfinite(need) && need < 1.0e7) {
        g.max_reach = std::max(static_cast<int>(need) + 3, image_reach);
    } else {
        g.max_reach = image_reach;
    }

    View3 xyz_d("xyz", n, 3);
    auto xyz_h = Kokkos::create_mirror_view(xyz_d);
    for (int i = 0; i < n; ++i) {
        xyz_h(i, 0) = xyz[3 * i];
        xyz_h(i, 1) = xyz[3 * i + 1];
        xyz_h(i, 2) = xyz[3 * i + 2];
    }
    Kokkos::deep_copy(xyz_d, xyz_h);

    View3 frac("frac", n, 3);
    View3 folded("folded", n, 3);
    View3i bin("bin", n, 3);
    View1i counts("counts", ncell);
    Fill fill{xyz_d, frac, folded, bin, counts, g};
    Kokkos::parallel_for(Kokkos::RangePolicy<Exec>(0, n), fill);
    Kokkos::fence();

    View1i offsets("offsets", ncell + 1);
    ExclusiveScan scan{counts, offsets};
    Kokkos::parallel_scan(Kokkos::RangePolicy<Exec>(0, ncell), scan);
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Exec>(ncell, ncell + 1), KOKKOS_LAMBDA(int) {
            offsets(ncell) = offsets(ncell - 1) + counts(ncell - 1);
        });
    Kokkos::fence();

    View1i cursor("cursor", ncell);
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Exec>(0, ncell), KOKKOS_LAMBDA(int c) { cursor(c) = offsets(c); });
    View1i occupants("occupants", n);
    Scatter scatter{bin, cursor, occupants, g.nx, g.ny, g.nz};
    Kokkos::parallel_for(Kokkos::RangePolicy<Exec>(0, n), scatter);
    Kokkos::fence();

    View3i out_d("out", n, k);
    Kokkos::deep_copy(out_d, -1);
    bool write_d2 = out_d2 != nullptr;
    View3 d2_d;
    if (write_d2) {
        d2_d = View3("d2", n, k);
        Kokkos::deep_copy(d2_d, std::numeric_limits<double>::quiet_NaN());
    }
    if (box.mode == 0) {
        launch_walk<0>(g, frac, folded, bin, offsets, occupants, out_d, d2_d, write_d2);
    } else if (box.mode == 1) {
        launch_walk<1>(g, frac, folded, bin, offsets, occupants, out_d, d2_d, write_d2);
    } else {
        launch_walk<2>(g, frac, folded, bin, offsets, occupants, out_d, d2_d, write_d2);
    }

    auto out_h = Kokkos::create_mirror_view(out_d);
    Kokkos::deep_copy(out_h, out_d);
    for (int i = 0; i < n * k; ++i) {
        out_nn[i] = out_h(i / k, i % k);
    }
    if (write_d2) {
        auto d2_h = Kokkos::create_mirror_view(d2_d);
        Kokkos::deep_copy(d2_h, d2_d);
        for (int i = 0; i < n * k; ++i) {
            out_d2[i] = d2_h(i / k, i % k);
        }
    }
    return 0;
}
