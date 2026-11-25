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
    if (out_idx[0] != 0 || out_d2[0] != 0.0) {
        lc_nd_free(index);
        return 3;
    }

    size_t needed = 0;
    if (lc_nd_within(index, query, 1.0, NULL, NULL, 0, &needed) != 0) {
        lc_nd_free(index);
        return 4;
    }
    if (needed == 0) {
        lc_nd_free(index);
        return 5;
    }
    int within_idx[4];
    if (lc_nd_within(index, query, 1.0, within_idx, NULL, 4, &needed) != 0) {
        lc_nd_free(index);
        return 6;
    }

    size_t pair_count = 0;
    if (lc_nd_pairs_within(index, 1.0, NULL, NULL, NULL, 0, &pair_count) != 0) {
        lc_nd_free(index);
        return 7;
    }

    int labels[4];
    int core[4];
    size_t clusters = 0;
    if (lc_nd_dbscan(index, 1.01, 2, labels, core, &clusters) != 0) {
        lc_nd_free(index);
        return 8;
    }
    if (clusters == 0) {
        lc_nd_free(index);
        return 9;
    }

    lc_nd_free(index);
    return 0;
}
