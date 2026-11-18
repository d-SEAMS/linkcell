/* Independent periodic k-nearest reference.
 * Neighbour order is increasing squared distance, then point index.
 * The search is every lattice image with each index in [-span, span].
 */
#ifndef LINKCELL_PERIODIC_REF_H
#define LINKCELL_PERIODIC_REF_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    double d;
    int j;
} RefCand;

static const double REF_ORTHO_XYZ[] = {
    0.20, 0.00, 0.00, 9.40, 0.00, 0.00, 0.20, 2.50, 0.10, 5.00, 5.00, 5.00,
    0.00, 0.10, 9.70, 3.30, 1.20, 4.80, 8.90, 8.10, 0.40, 1.40, 7.20, 2.60,
    6.60, 0.30, 8.80, 4.20, 9.50, 3.30, 2.20, 3.80, 6.10, 7.70, 4.40, 1.90,
};
enum { REF_ORTHO_N = 12, REF_ORTHO_K = 4 };

static const double REF_SHEAR_H[3][3] = {
    {10.0, 0.0, 0.0},
    {5.0, 8.66, 0.0},
    {0.0, 0.0, 10.0},
};
static const double REF_SHEAR_FRAC[] = {
    0.02, 0.10, 0.20, 0.90, 0.05, 0.15, 0.40, 0.70, 0.30,
    0.15, 0.40, 0.80, 0.60, 0.20, 0.55, 0.33, 0.88, 0.10,
};
enum { REF_SHEAR_N = 6, REF_SHEAR_K = 3 };

static const double REF_TRI_H[3][3] = {
    {10.0, 0.2, 0.1},
    {1.2, 8.5, -0.3},
    {0.4, 0.5, 9.2},
};
static const double REF_TRI_FRAC[] = {
    0.02, 0.00, 0.00, 0.94, 0.00, 0.01, 0.10, 0.20, 0.30, 0.50, 0.50, 0.50,
    0.80, 0.15, 0.90, 0.33, 0.71, 0.22, 0.61, 0.08, 0.44, 0.12, 0.88, 0.67,
};
enum { REF_TRI_N = 8, REF_TRI_K = 3 };

static void ref_cartesian(const double *frac, int n, const double h[3][3], double *xyz) {
    int i;
    for (i = 0; i < n; i++) {
        const double fa = frac[3 * i];
        const double fb = frac[3 * i + 1];
        const double fc = frac[3 * i + 2];
        xyz[3 * i] = fa * h[0][0] + fb * h[1][0] + fc * h[2][0];
        xyz[3 * i + 1] = fa * h[0][1] + fb * h[1][1] + fc * h[2][1];
        xyz[3 * i + 2] = fa * h[0][2] + fb * h[1][2] + fc * h[2][2];
    }
}

static double ref_image_dist2(const double *pi, const double *pj, const double h[3][3], int span) {
    double best = 1.0e300;
    int ia, ib, ic;
    for (ia = -span; ia <= span; ia++) {
        for (ib = -span; ib <= span; ib++) {
            for (ic = -span; ic <= span; ic++) {
                const double dx = pj[0] + (double)ia * h[0][0] + (double)ib * h[1][0] +
                                  (double)ic * h[2][0] - pi[0];
                const double dy = pj[1] + (double)ia * h[0][1] + (double)ib * h[1][1] +
                                  (double)ic * h[2][1] - pi[1];
                const double dz = pj[2] + (double)ia * h[0][2] + (double)ib * h[1][2] +
                                  (double)ic * h[2][2] - pi[2];
                const double d2 = dx * dx + dy * dy + dz * dz;
                if (d2 < best) {
                    best = d2;
                }
            }
        }
    }
    return best;
}

static void ref_sort(RefCand *a, int n) {
    int i;
    for (i = 1; i < n; i++) {
        RefCand key = a[i];
        int k = i;
        while (k > 0 && (a[k - 1].d > key.d || (a[k - 1].d == key.d && a[k - 1].j > key.j))) {
            a[k] = a[k - 1];
            k--;
        }
        a[k] = key;
    }
}

/* Returns 0 when packed nn/d2 match the reference. Prints the max |d2| error. */
static int ref_check(const char *label, const double *xyz, int n, const double h[3][3], int k,
                     int span, const int *nn, const double *d2) {
    int i, t, j, m;
    double max_abs = 0.0;
    RefCand *cands = (RefCand *)malloc((size_t)(n > 1 ? n - 1 : 1) * sizeof(RefCand));
    if (cands == NULL) {
        fprintf(stderr, "%s: out of memory\n", label);
        return 1;
    }
    for (i = 0; i < n; i++) {
        m = 0;
        for (j = 0; j < n; j++) {
            if (j == i) {
                continue;
            }
            cands[m].d = ref_image_dist2(xyz + 3 * i, xyz + 3 * j, h, span);
            cands[m].j = j;
            m++;
        }
        ref_sort(cands, m);
        if (m < k) {
            fprintf(stderr, "%s: n=%d is too small for k=%d\n", label, n, k);
            free(cands);
            return 1;
        }
        if (m > k && !(cands[k].d - cands[k - 1].d > 1.0e-6)) {
            fprintf(stderr, "%s: ambiguous neighbour gap at i=%d (%g)\n", label, i,
                    cands[k].d - cands[k - 1].d);
            free(cands);
            return 1;
        }
        for (t = 0; t < k; t++) {
            const int got = nn[i * k + t];
            const double gd = d2[i * k + t];
            const double err = fabs(gd - cands[t].d);
            if (got != cands[t].j || !(err <= 1.0e-8)) {
                fprintf(stderr,
                        "%s: i=%d rank=%d got (%d, %.16g) expected (%d, %.16g)\n",
                        label, i, t, got, gd, cands[t].j, cands[t].d);
                free(cands);
                return 1;
            }
            if (err > max_abs) {
                max_abs = err;
            }
        }
    }
    free(cands);
    printf("%s ok n=%d k=%d max_abs_d2=%.3e\n", label, n, k, max_abs);
    return 0;
}

#endif
