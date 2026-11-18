#include "linkcell.hpp"

#include "periodic_ref.h"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

namespace {

lc_cell raw_of(const double h[3][3]) {
    lc_cell cell{};
    cell.ax = h[0][0];
    cell.ay = h[0][1];
    cell.az = h[0][2];
    cell.bx = h[1][0];
    cell.by = h[1][1];
    cell.bz = h[1][2];
    cell.cx = h[2][0];
    cell.cy = h[2][1];
    cell.cz = h[2][2];
    return cell;
}

int query(const char *label, const double *xyz, int n, const double h[3][3], int k) {
    const linkcell::Cell cell(raw_of(h));
    const linkcell::Neighbours nn = linkcell::knearest(xyz, static_cast<std::size_t>(n), cell,
                                                       static_cast<std::size_t>(k));
    std::vector<int> idx(static_cast<std::size_t>(n * k));
    std::vector<double> d2(static_cast<std::size_t>(n * k));
    for (int i = 0; i < n; i++) {
        for (int t = 0; t < k; t++) {
            idx[static_cast<std::size_t>(i * k + t)] = nn.neighbour(static_cast<std::size_t>(i),
                                                                     static_cast<std::size_t>(t));
            d2[static_cast<std::size_t>(i * k + t)] =
                nn.dist2(static_cast<std::size_t>(i), static_cast<std::size_t>(t));
        }
    }
    return ref_check(label, xyz, n, h, k, 2, idx.data(), d2.data());
}

}  // namespace

int main(int argc, char **argv) {
    const double ortho_h[3][3] = {
        {10.0, 0.0, 0.0},
        {0.0, 10.0, 0.0},
        {0.0, 0.0, 10.0},
    };
    const double quick_xyz[] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
    const double wrap_xyz[] = {0.2, 0.0, 0.0, 9.4, 0.0, 0.0};
    double shear_xyz[REF_SHEAR_N * 3];
    double tri_xyz[REF_TRI_N * 3];

    if (argc != 2) {
        std::fprintf(stderr, "usage: periodic_cpp <version>\n");
        return 2;
    }
    const std::string version = lc_version();
    std::printf("lc_version %s\n", version.c_str());
    if (version != argv[1]) {
        std::fprintf(stderr, "version %s != %s\n", version.c_str(), argv[1]);
        return 1;
    }

    const linkcell::Cell box = linkcell::Cell::ortho(10.0, 10.0, 10.0);
    const linkcell::Neighbours quick = linkcell::knearest(quick_xyz, 2, box, 1);
    std::cout << "0 -> " << quick.neighbour(0, 0) << "\n";
    std::cout << "1 -> " << quick.neighbour(1, 0) << "\n";
    if (quick.neighbour(0, 0) != 1 || quick.neighbour(1, 0) != 0) {
        std::fprintf(stderr, "quickstart mismatch\n");
        return 1;
    }

    const linkcell::Neighbours wrap = linkcell::knearest(wrap_xyz, 2, box, 1);
    std::printf("wrap %d %d d2 %.16g %.16g\n", wrap.neighbour(0, 0), wrap.neighbour(1, 0),
                wrap.dist2(0, 0), wrap.dist2(1, 0));
    if (wrap.neighbour(0, 0) != 1 || wrap.neighbour(1, 0) != 0 ||
        std::fabs(wrap.dist2(0, 0) - 0.64) > 1.0e-12 ||
        std::fabs(wrap.dist2(1, 0) - 0.64) > 1.0e-12) {
        std::fprintf(stderr, "periodic pair mismatch\n");
        return 1;
    }

    if (query("ortho", REF_ORTHO_XYZ, REF_ORTHO_N, ortho_h, REF_ORTHO_K) != 0) {
        return 1;
    }
    ref_cartesian(REF_SHEAR_FRAC, REF_SHEAR_N, REF_SHEAR_H, shear_xyz);
    if (query("sheared", shear_xyz, REF_SHEAR_N, REF_SHEAR_H, REF_SHEAR_K) != 0) {
        return 1;
    }
    ref_cartesian(REF_TRI_FRAC, REF_TRI_N, REF_TRI_H, tri_xyz);
    if (query("triclinic", tri_xyz, REF_TRI_N, REF_TRI_H, REF_TRI_K) != 0) {
        return 1;
    }
    return 0;
}
