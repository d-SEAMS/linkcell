// Self-check and strong-scaling timer for the Kokkos certified walk.
//
// Kokkos::initialize reads --kokkos-threads. Optional args after that:
//   lc_kokkos_bench [n] [reps]
// Env LC_KOKKOS_N / LC_KOKKOS_REPS override the defaults (262144, 5).

#include "knearest.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <utility>
#include <vector>

namespace {

double wrap_half(double d, double length) {
    double half = 0.5 * length;
    if (d < -half) {
        d += length;
    }
    if (d >= half) {
        d -= length;
    }
    return d;
}

double shift_at(const LcKokkosBox& box, int na, int nb, int nc, int axis) {
    return na * box.a[axis] + nb * box.b[axis] + nc * box.c[axis];
}

double brute_dist2(const LcKokkosBox& box, const double* p, const double* q) {
    if (box.mode == 0) {
        double dx = wrap_half(q[0] - p[0], box.a[0]);
        double dy = wrap_half(q[1] - p[1], box.b[1]);
        double dz = wrap_half(q[2] - p[2], box.c[2]);
        return dx * dx + dy * dy + dz * dz;
    }
    double best = 1.0e300;
    for (int na = -1; na <= 1; ++na) {
        for (int nb = -1; nb <= 1; ++nb) {
            for (int nc = -1; nc <= 1; ++nc) {
                double dx = q[0] + shift_at(box, na, nb, nc, 0) - p[0];
                double dy = q[1] + shift_at(box, na, nb, nc, 1) - p[1];
                double dz = q[2] + shift_at(box, na, nb, nc, 2) - p[2];
                double d2 = dx * dx + dy * dy + dz * dz;
                if (d2 < best) {
                    best = d2;
                }
            }
        }
    }
    return best;
}

std::vector<int> brute_k(const LcKokkosBox& box, const std::vector<double>& xyz, int n, int i,
                         int k) {
    std::vector<std::pair<double, int>> best;
    best.reserve(static_cast<size_t>(n - 1));
    const double* p = xyz.data() + 3 * i;
    for (int j = 0; j < n; ++j) {
        if (j == i) {
            continue;
        }
        best.emplace_back(brute_dist2(box, p, xyz.data() + 3 * j), j);
    }
    std::sort(best.begin(), best.end());
    std::vector<int> out;
    for (int t = 0; t < k && t < static_cast<int>(best.size()); ++t) {
        out.push_back(best[t].second);
    }
    return out;
}

std::vector<double> ortho_lattice(int nside, double& boxl) {
    boxl = nside * 3.125;
    std::vector<double> xyz;
    xyz.reserve(static_cast<size_t>(nside) * nside * nside * 3);
    for (int iz = 0; iz < nside; ++iz) {
        for (int iy = 0; iy < nside; ++iy) {
            for (int ix = 0; ix < nside; ++ix) {
                xyz.push_back(ix * 3.125);
                xyz.push_back(iy * 3.125);
                xyz.push_back(iz * 3.125);
            }
        }
    }
    return xyz;
}

std::vector<double> tilt_lattice(int nside, LcKokkosBox& box) {
    double lx = nside * 3.125;
    box.a[0] = lx;
    box.a[1] = 0;
    box.a[2] = 0;
    box.b[0] = lx * 0.2;
    box.b[1] = lx * 0.9;
    box.b[2] = 0;
    box.c[0] = lx * 0.05;
    box.c[1] = lx * -0.08;
    box.c[2] = lx * 0.95;
    box.origin[0] = box.origin[1] = box.origin[2] = 0;
    box.mode = 1;
    std::vector<double> xyz;
    double nf = static_cast<double>(nside);
    for (int iz = 0; iz < nside; ++iz) {
        for (int iy = 0; iy < nside; ++iy) {
            for (int ix = 0; ix < nside; ++ix) {
                double s0 = (ix + 0.5) / nf;
                double s1 = (iy + 0.5) / nf;
                double s2 = (iz + 0.5) / nf;
                xyz.push_back(box.a[0] * s0 + box.b[0] * s1 + box.c[0] * s2);
                xyz.push_back(box.a[1] * s0 + box.b[1] * s1 + box.c[1] * s2);
                xyz.push_back(box.a[2] * s0 + box.b[2] * s1 + box.c[2] * s2);
            }
        }
    }
    return xyz;
}

int check_case(const char* name, const LcKokkosBox& box, const std::vector<double>& xyz, int k) {
    int n = static_cast<int>(xyz.size() / 3);
    std::vector<int> nn(static_cast<size_t>(n) * k, -2);
    std::vector<double> d2(static_cast<size_t>(n) * k, 0);
    int rc = lc_kokkos_knearest(xyz.data(), n, box, k, 3.0, nn.data(), d2.data());
    if (rc != 0) {
        std::cerr << "check " << name << " rc=" << rc << "\n";
        return rc;
    }
    int step = std::max(1, n / 24);
    const double* pbase = xyz.data();
    for (int i = 0; i < n; i += step) {
        std::vector<std::pair<double, int>> all;
        all.reserve(static_cast<size_t>(n - 1));
        for (int j = 0; j < n; ++j) {
            if (j == i) {
                continue;
            }
            all.emplace_back(brute_dist2(box, pbase + 3 * i, pbase + 3 * j), j);
        }
        std::sort(all.begin(), all.end());
        double kth = all[static_cast<size_t>(k - 1)].first;
        double tol = 1e-8 * std::max(1.0, kth);
        for (int t = 0; t < k; ++t) {
            int got = nn[i * k + t];
            double gd = brute_dist2(box, pbase + 3 * i, pbase + 3 * got);
            if (!(gd <= kth + tol)) {
                std::cerr << "check " << name << " i=" << i << " slot=" << t << " got=" << got
                          << " d2=" << gd << " kth=" << kth << "\n";
                return 1;
            }
        }
        for (const auto& cand : all) {
            if (cand.first < kth - tol) {
                bool found = false;
                for (int t = 0; t < k; ++t) {
                    if (nn[i * k + t] == cand.second) {
                        found = true;
                    }
                }
                if (!found) {
                    std::cerr << "check " << name << " i=" << i << " missed " << cand.second
                              << " d2=" << cand.first << " kth=" << kth << "\n";
                    return 1;
                }
            }
        }
    }
    std::cout << "check " << name << " ok n=" << n << "\n";
    return 0;
}

int env_int(const char* key, int fallback) {
    const char* v = std::getenv(key);
    if (v == nullptr || v[0] == '\0') {
        return fallback;
    }
    return std::atoi(v);
}

}  // namespace

int ExecSpaceConcurrency() { return Kokkos::DefaultExecutionSpace().concurrency(); }

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    int rc = 0;
    {
        double boxl = 0;
        auto xyz = ortho_lattice(8, boxl);
        LcKokkosBox box{};
        box.a[0] = boxl;
        box.b[1] = boxl;
        box.c[2] = boxl;
        box.mode = 0;
        rc = check_case("ortho", box, xyz, 4);
        if (rc == 0) {
            LcKokkosBox tilt{};
            auto txyz = tilt_lattice(6, tilt);
            rc = check_case("tilt", tilt, txyz, 4);
        }
        if (rc == 0) {
            int nside = 64;
            int n_req = env_int("LC_KOKKOS_N", argc > 1 ? std::atoi(argv[1]) : 262144);
            int reps = env_int("LC_KOKKOS_REPS", argc > 2 ? std::atoi(argv[2]) : 5);
            if (n_req > 0) {
                nside = static_cast<int>(std::lround(std::cbrt(static_cast<double>(n_req))));
                if (nside < 2) {
                    nside = 2;
                }
            }
            double timed_box = 0;
            auto timed = ortho_lattice(nside, timed_box);
            int n = static_cast<int>(timed.size() / 3);
            LcKokkosBox run{};
            run.a[0] = timed_box;
            run.b[1] = timed_box;
            run.c[2] = timed_box;
            run.mode = 0;
            std::vector<int> nn(static_cast<size_t>(n) * 4, -1);
            lc_kokkos_knearest(timed.data(), n, run, 4, 3.0, nn.data(), nullptr);
            uint64_t useful[64] = {};
            if (std::getenv("LC_KOKKOS_POP") != nullptr) {
                lc_kokkos_pop_bind(useful);
            }
            int acc = nn[0];
            auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < reps; ++r) {
                lc_kokkos_knearest(timed.data(), n, run, 4, 3.0, nn.data(), nullptr);
                acc += nn[0];
            }
            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            int nt = ExecSpaceConcurrency();
            std::cout << "backend=" << Kokkos::DefaultExecutionSpace::name()
                      << " threads=" << nt << " n=" << n << " k=4 reps=" << reps << " ms=" << ms
                      << " ms_per=" << (ms / reps) << " acc=" << acc << "\n";
            if (std::getenv("LC_KOKKOS_POP") != nullptr) {
                uint64_t sum = 0;
                uint64_t mx = 0;
                int nslot = nt < 64 ? nt : 64;
                for (int t = 0; t < nslot; ++t) {
                    sum += useful[t];
                    if (useful[t] > mx) {
                        mx = useful[t];
                    }
                }
                double avg = nslot > 0 ? static_cast<double>(sum) / nslot : 0;
                double lb = mx > 0 ? avg / static_cast<double>(mx) : 0;
                uint64_t wall_ns = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
                double ce = wall_ns > 0 ? static_cast<double>(mx) / static_cast<double>(wall_ns) : 0;
                std::cout << "pop backend=OpenMP threads=" << nt << " n=" << n << " reps=" << reps
                          << " wall_ns=" << wall_ns << " useful_ns=";
                for (int t = 0; t < nslot; ++t) {
                    if (t) {
                        std::cout << ",";
                    }
                    std::cout << useful[t];
                }
                std::cout << " lb=" << lb << " ce=" << ce << " pe=" << (lb * ce) << "\n";
            }
        }
    }
    Kokkos::finalize();
    return rc;
}
