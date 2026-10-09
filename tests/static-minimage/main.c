#include "linkcell.h"
#include "minimage.h"

#include <math.h>
#include <stdio.h>

int main(void) {
    const double xyz[] = {0.2, 0.0, 0.0, 29.4, 0.0, 0.0};
    const lc_cell lc = lc_cell_ortho(10.0, 10.0, 10.0);
    const mi_cell mi = mi_cell_ortho(10.0, 10.0, 10.0);
    int neighbors[2] = {-1, -1};
    double distances[2] = {0.0, 0.0};
    double dr[3];
    if (lc_knearest_d2(xyz, 2, &lc, 1, NULL, 0.0, neighbors, distances) != 0 ||
        mi_displacement_euclidean(&mi, xyz, xyz + 3, dr) != 0) {
        return 1;
    }
    const double d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
    if (neighbors[0] != 1 || neighbors[1] != 0 ||
        fabs(d2 - 0.64) > 1e-12 || fabs(distances[0] - d2) > 1e-12 ||
        fabs(distances[1] - d2) > 1e-12) {
        fprintf(stderr, "inconsistent periodic distances: %g %g %g\n",
                d2, distances[0], distances[1]);
        return 1;
    }
    return 0;
}
