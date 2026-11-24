/* Every lc_knearest* entry returns nonzero, with the NonFinite message,
 * for a NaN or infinite coordinate or cell origin. */
#include "linkcell.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 3
#define K 2
#define FRAMES 2

static const char *const WANT = "a coordinate is not finite";
static int failures = 0;

static void expect(int rc, const char *what, double bad, int axis) {
  const char *msg = lc_last_error();
  if (rc == 0 || msg == NULL || strcmp(msg, WANT) != 0) {
    fprintf(stderr, "%s bad=%g axis=%d: rc=%d msg=%s\n", what, bad, axis, rc,
            msg ? msg : "(null)");
    failures++;
  }
}

static void expect_cleared(const int *nn, const double *d2, size_t len,
                           const char *what) {
  for (size_t t = 0; t < len; t++) {
    if (nn[t] != -1 || (d2 != NULL && !isnan(d2[t]))) {
      fprintf(stderr, "%s: slot %zu not cleared\n", what, t);
      failures++;
      return;
    }
  }
}

static int *grow(void *ctx, size_t total) {
  int **cols = (int **)ctx;
  int *next = realloc(*cols, (total ? total : 1) * sizeof **cols);
  if (next == NULL) {
    return NULL;
  }
  *cols = next;
  return next;
}

static void fill(double *xyz) {
  const double base[N * 3] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0};
  memcpy(xyz, base, sizeof base);
}

int main(void) {
  const double bads[3] = {NAN, INFINITY, -INFINITY};
  const lc_cell box = lc_cell_ortho(10.0, 10.0, 10.0);
  double xyz[FRAMES * N * 3];
  int nn[FRAMES * N * K];
  double d2[FRAMES * N * K];
  int offsets[N + 1];
  int *cols = NULL;

  fill(xyz);
  if (lc_knearest(xyz, N, &box, K, NULL, 0.0, nn) != 0 ||
      lc_knearest_csr(xyz, N, &box, K, NULL, 0.0, 0, offsets, grow, &cols) !=
          0) {
    fprintf(stderr, "finite input failed: %s\n", lc_last_error());
    return 1;
  }

  for (int b = 0; b < 3; b++) {
    const double bad = bads[b];
    for (int axis = 0; axis < 3; axis++) {
      fill(xyz);
      xyz[3 + axis] = bad;

      memset(nn, 7, sizeof nn);
      expect(lc_knearest(xyz, N, &box, K, NULL, 0.0, nn), "lc_knearest", bad,
             axis);
      expect_cleared(nn, NULL, N * K, "lc_knearest");

      memset(nn, 7, sizeof nn);
      for (size_t t = 0; t < N * K; t++) {
        d2[t] = 1.0;
      }
      expect(lc_knearest_d2(xyz, N, &box, K, NULL, 0.0, nn, d2),
             "lc_knearest_d2", bad, axis);
      expect_cleared(nn, d2, N * K, "lc_knearest_d2");

      for (int mutual = 0; mutual < 2; mutual++) {
        expect(lc_knearest_csr(xyz, N, &box, K, NULL, 0.0, mutual, offsets,
                               grow, &cols),
               "lc_knearest_csr", bad, axis);
      }

      fill(xyz);
      fill(xyz + N * 3);
      xyz[N * 3 + 3 + axis] = bad;
      memset(nn, 7, sizeof nn);
      for (size_t t = 0; t < FRAMES * N * K; t++) {
        d2[t] = 1.0;
      }
      expect(lc_knearest_many(xyz, N, FRAMES, &box, K, NULL, 0.0, nn, d2),
             "lc_knearest_many", bad, axis);
      expect_cleared(nn, d2, FRAMES * N * K, "lc_knearest_many");

      lc_cell shifted = box;
      double *origin[3] = {&shifted.ox, &shifted.oy, &shifted.oz};
      *origin[axis] = bad;
      fill(xyz);
      expect(lc_knearest(xyz, N, &shifted, K, NULL, 0.0, nn),
             "lc_knearest origin", bad, axis);
      expect(lc_knearest_csr(xyz, N, &shifted, K, NULL, 0.0, 0, offsets, grow,
                             &cols),
             "lc_knearest_csr origin", bad, axis);

      lc_cell stretched = box;
      double *length[3] = {&stretched.ax, &stretched.by, &stretched.cz};
      *length[axis] = bad;
      if (lc_knearest(xyz, N, &stretched, K, NULL, 0.0, nn) == 0 ||
          lc_knearest_csr(xyz, N, &stretched, K, NULL, 0.0, 0, offsets, grow,
                          &cols) == 0) {
        fprintf(stderr, "box length bad=%g axis=%d accepted\n", bad, axis);
        failures++;
      }
    }
  }
  free(cols);
  if (failures != 0) {
    fprintf(stderr, "%d failures\n", failures);
    return 1;
  }
  printf("non-finite: ok\n");
  return 0;
}
