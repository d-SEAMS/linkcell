#include "linkcell.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Cutoff rows through lc_pairs_within_csr: sizing, the rows of a small
 * frame, the same bits at two thread widths, and an error code (not an
 * abort) for a coordinate that is not finite. */

typedef struct {
  int *offsets;
  int *cols;
  double *d2;
  size_t ncols;
} rows;

static void drop(rows *r) {
  free(r->offsets);
  free(r->cols);
  free(r->d2);
  memset(r, 0, sizeof *r);
}

/* Query, then fill exactly. */
static int build(const double *xyz, size_t n, const lc_cell *box, double cutoff,
                 int half, rows *out) {
  size_t need_offsets = 0;
  size_t need_cols = 0;
  if (lc_pairs_within_csr(xyz, n, box, NULL, cutoff, NULL, 0.0, half, NULL, 0,
                          NULL, NULL, 0, &need_offsets, &need_cols) != 0) {
    fprintf(stderr, "query: %s\n", lc_last_error());
    return 1;
  }
  if (need_offsets != n + 1) {
    fprintf(stderr, "query: %zu offsets for %zu points\n", need_offsets, n);
    return 1;
  }
  out->offsets = malloc(need_offsets * sizeof *out->offsets);
  out->cols = malloc((need_cols + 1) * sizeof *out->cols);
  out->d2 = malloc((need_cols + 1) * sizeof *out->d2);
  out->ncols = need_cols;
  if (out->offsets == NULL || out->cols == NULL || out->d2 == NULL) {
    return 1;
  }
  if (lc_pairs_within_csr(xyz, n, box, NULL, cutoff, NULL, 0.0, half,
                          out->offsets, need_offsets, out->cols, out->d2,
                          need_cols, &need_offsets, &need_cols) != 0) {
    fprintf(stderr, "fill: %s\n", lc_last_error());
    return 1;
  }
  return 0;
}

static int same(const rows *a, const rows *b, size_t n) {
  return a->ncols == b->ncols &&
         memcmp(a->offsets, b->offsets, (n + 1) * sizeof *a->offsets) == 0 &&
         memcmp(a->cols, b->cols, a->ncols * sizeof *a->cols) == 0 &&
         memcmp(a->d2, b->d2, a->ncols * sizeof *a->d2) == 0;
}

int main(void) {
  const lc_cell box = lc_cell_ortho(10.0, 10.0, 10.0);
  /* 0 and 1 are 0.8 apart through the x face; 2 is alone. */
  const double three[] = {0.2, 0.0, 0.0, 9.4, 0.0, 0.0, 5.0, 5.0, 5.0};
  rows r = {0};
  if (build(three, 3, &box, 1.0, 0, &r) != 0) {
    return 1;
  }
  const int want_offsets[] = {0, 1, 2, 2};
  const int want_cols[] = {1, 0};
  if (r.ncols != 2 || memcmp(r.offsets, want_offsets, sizeof want_offsets) != 0 ||
      memcmp(r.cols, want_cols, sizeof want_cols) != 0 ||
      fabs(r.d2[0] - 0.64) > 1e-12) {
    fprintf(stderr, "three points: wrong rows\n");
    return 1;
  }
  for (size_t i = 0; i < 3; i++) {
    printf("%zu ->", i);
    for (int k = r.offsets[i]; k < r.offsets[i + 1]; k++) {
      printf(" %d (%.2f)", r.cols[k], r.d2[k]);
    }
    printf("\n");
  }
  drop(&r);

  /* 4000 points at water density, cutoff 3.5. */
  const size_t n = 4000;
  const double edge = cbrt((double)n / 0.0334);
  const lc_cell water = lc_cell_ortho(edge, edge, edge);
  double *xyz = malloc(n * 3 * sizeof *xyz);
  if (xyz == NULL) {
    return 1;
  }
  uint64_t state = 0x9e3779b97f4a7c15ull;
  for (size_t k = 0; k < n * 3; k++) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    xyz[k] = edge * (double)(state >> 11) / 9007199254740992.0;
  }
  for (int half = 0; half <= 1; half++) {
    rows one = {0};
    rows four = {0};
    lc_set_threads(1);
    int rc = build(xyz, n, &water, 3.5, half, &one);
    lc_set_threads(4);
    rc = rc || build(xyz, n, &water, 3.5, half, &four);
    if (rc != 0 || !same(&one, &four, n)) {
      fprintf(stderr, "half %d: rows differ between one thread and four\n", half);
      return 1;
    }
    for (size_t i = 0; i < n; i++) {
      for (int k = one.offsets[i] + 1; k < one.offsets[i + 1]; k++) {
        if (one.cols[k - 1] >= one.cols[k]) {
          fprintf(stderr, "row %zu is not ascending\n", i);
          return 1;
        }
      }
    }
    printf("water %s: %zu columns, the same on 1 and 4 threads\n",
           half ? "half" : "full", one.ncols);
    drop(&one);
    drop(&four);
  }
  lc_set_threads(0);

  xyz[5] = NAN;
  size_t need_offsets = 0;
  size_t need_cols = 0;
  if (lc_pairs_within_csr(xyz, n, &water, NULL, 3.5, NULL, 0.0, 0, NULL, 0,
                          NULL, NULL, 0, &need_offsets, &need_cols) == 0 ||
      lc_last_error() == NULL) {
    fprintf(stderr, "a NaN coordinate was accepted\n");
    return 1;
  }
  printf("NaN coordinate: %s\n", lc_last_error());
  free(xyz);
  return 0;
}
