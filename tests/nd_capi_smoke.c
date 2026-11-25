#include <math.h>
#include <stddef.h>

#include "linkcell.h"

int main(void) {
    const double data[] = {
        0.0, 0.0,
        1.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
    };
    lc_nd_index *index = NULL;
    if (lc_nd_build(data, 4, 2, NULL, &index) != 0 || index == NULL) {
        return 1;
    }

    const double query[] = {0.0, 0.0};
    int out_idx[4];
    double out_d2[4];
    if (lc_nd_knn(index, query, 4, out_idx, out_d2) != 0) {
        lc_nd_free(index);
        return 2;
    }
    if (out_idx[0] != 0 || out_idx[1] != 1 || out_idx[2] != 2 || out_idx[3] != 3) {
        lc_nd_free(index);
        return 3;
    }
    if (out_d2[0] != 0.0 || out_d2[1] != 1.0 || out_d2[2] != 1.0 || out_d2[3] != 2.0) {
        lc_nd_free(index);
        return 4;
    }

    size_t needed = 0;
    if (lc_nd_within(index, query, 1.0, NULL, NULL, 0, &needed) != 0) {
        lc_nd_free(index);
        return 5;
    }
    if (needed != 3) {
        lc_nd_free(index);
        return 6;
    }
    int within_idx[4];
    if (lc_nd_within(index, query, 1.0, within_idx, NULL, 4, &needed) != 0) {
        lc_nd_free(index);
        return 7;
    }
    if (needed != 3 || within_idx[0] != 0 || within_idx[1] != 1 || within_idx[2] != 2) {
        lc_nd_free(index);
        return 8;
    }

    size_t pair_count = 0;
    if (lc_nd_pairs_within(index, 1.0, NULL, NULL, NULL, 0, &pair_count) != 0) {
        lc_nd_free(index);
        return 9;
    }
    if (pair_count != 4) {
        lc_nd_free(index);
        return 10;
    }
    int pair_i[4];
    int pair_j[4];
    if (lc_nd_pairs_within(index, 1.0, pair_i, pair_j, NULL, 4, &pair_count) != 0) {
        lc_nd_free(index);
        return 11;
    }
    if (pair_count != 4 || pair_i[0] != 0 || pair_j[0] != 1 || pair_i[1] != 0 ||
        pair_j[1] != 2 || pair_i[2] != 1 || pair_j[2] != 3 || pair_i[3] != 2 ||
        pair_j[3] != 3) {
        lc_nd_free(index);
        return 12;
    }

    int labels[4];
    int core[4];
    size_t clusters = 0;
    if (lc_nd_dbscan(index, 1.01, 2, labels, core, &clusters) != 0) {
        lc_nd_free(index);
        return 13;
    }
    if (clusters != 1) {
        lc_nd_free(index);
        return 14;
    }
    if (labels[0] != 0 || labels[1] != 0 || labels[2] != 0 || labels[3] != 0) {
        lc_nd_free(index);
        return 15;
    }
    if (core[0] != 1 || core[1] != 1 || core[2] != 1 || core[3] != 1) {
        lc_nd_free(index);
        return 16;
    }

    lc_nd_free(index);
    return 0;
}
