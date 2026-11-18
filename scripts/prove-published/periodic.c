#include "linkcell.h"

#include "periodic_ref.h"

#include <stdio.h>
#include <string.h>

static int query(const char *label, const double *xyz, int n, const lc_cell *box,
                 const double h[3][3], int k, int span) {
    int *nn = (int *)malloc((size_t)n * (size_t)k * sizeof(int));
    double *d2 = (double *)malloc((size_t)n * (size_t)k * sizeof(double));
    int status;
    if (nn == NULL || d2 == NULL) {
        fprintf(stderr, "%s: out of memory\n", label);
        free(nn);
        free(d2);
        return 1;
    }
    status = lc_knearest_d2(xyz, (size_t)n, box, (size_t)k, NULL, 0.0, nn, d2);
    if (status != 0) {
        fprintf(stderr, "%s: %s\n", label, lc_last_error());
        free(nn);
        free(d2);
        return 1;
    }
    status = ref_check(label, xyz, n, h, k, span, nn, d2);
    free(nn);
    free(d2);
    return status;
}

int main(int argc, char **argv) {
    const double ortho_h[3][3] = {
        {10.0, 0.0, 0.0},
        {0.0, 10.0, 0.0},
        {0.0, 0.0, 10.0},
    };
    double wrap_xyz[] = {0.2, 0.0, 0.0, 9.4, 0.0, 0.0};
    double quick_xyz[] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
    double shear_xyz[REF_SHEAR_N * 3];
    double tri_xyz[REF_TRI_N * 3];
    lc_cell box;
    lc_cell shear;
    lc_cell tri;
    int quick[2];
    int wrap_nn[2];
    double wrap_d2[2];
    const char *version;

    if (argc != 2) {
        fprintf(stderr, "usage: periodic <version>\n");
        return 2;
    }
    version = lc_version();
    printf("lc_version %s\n", version);
    if (version == NULL || strcmp(version, argv[1]) != 0) {
        fprintf(stderr, "version %s != %s\n", version ? version : "(null)", argv[1]);
        return 1;
    }

    box = lc_cell_ortho(10.0, 10.0, 10.0);
    if (lc_knearest(quick_xyz, 2, &box, 1, NULL, 0.0, quick) != 0) {
        fprintf(stderr, "quickstart: %s\n", lc_last_error());
        return 1;
    }
    printf("0 -> %d\n1 -> %d\n", quick[0], quick[1]);
    if (quick[0] != 1 || quick[1] != 0) {
        fprintf(stderr, "quickstart neighbours %d %d\n", quick[0], quick[1]);
        return 1;
    }

    if (lc_knearest_d2(wrap_xyz, 2, &box, 1, NULL, 0.0, wrap_nn, wrap_d2) != 0) {
        fprintf(stderr, "wrap: %s\n", lc_last_error());
        return 1;
    }
    printf("wrap %d %d d2 %.16g %.16g\n", wrap_nn[0], wrap_nn[1], wrap_d2[0], wrap_d2[1]);
    if (wrap_nn[0] != 1 || wrap_nn[1] != 0 || fabs(wrap_d2[0] - 0.64) > 1.0e-12 ||
        fabs(wrap_d2[1] - 0.64) > 1.0e-12) {
        fprintf(stderr, "periodic pair mismatch\n");
        return 1;
    }

    if (query("ortho", REF_ORTHO_XYZ, REF_ORTHO_N, &box, ortho_h, REF_ORTHO_K, 2) != 0) {
        return 1;
    }

    ref_cartesian(REF_SHEAR_FRAC, REF_SHEAR_N, REF_SHEAR_H, shear_xyz);
    shear = box;
    shear.ax = REF_SHEAR_H[0][0];
    shear.ay = REF_SHEAR_H[0][1];
    shear.az = REF_SHEAR_H[0][2];
    shear.bx = REF_SHEAR_H[1][0];
    shear.by = REF_SHEAR_H[1][1];
    shear.bz = REF_SHEAR_H[1][2];
    shear.cx = REF_SHEAR_H[2][0];
    shear.cy = REF_SHEAR_H[2][1];
    shear.cz = REF_SHEAR_H[2][2];
    shear.ox = 0.0;
    shear.oy = 0.0;
    shear.oz = 0.0;
    if (query("sheared", shear_xyz, REF_SHEAR_N, &shear, REF_SHEAR_H, REF_SHEAR_K, 2) != 0) {
        return 1;
    }

    ref_cartesian(REF_TRI_FRAC, REF_TRI_N, REF_TRI_H, tri_xyz);
    tri = shear;
    tri.ax = REF_TRI_H[0][0];
    tri.ay = REF_TRI_H[0][1];
    tri.az = REF_TRI_H[0][2];
    tri.bx = REF_TRI_H[1][0];
    tri.by = REF_TRI_H[1][1];
    tri.bz = REF_TRI_H[1][2];
    tri.cx = REF_TRI_H[2][0];
    tri.cy = REF_TRI_H[2][1];
    tri.cz = REF_TRI_H[2][2];
    if (query("triclinic", tri_xyz, REF_TRI_N, &tri, REF_TRI_H, REF_TRI_K, 2) != 0) {
        return 1;
    }
    return 0;
}
