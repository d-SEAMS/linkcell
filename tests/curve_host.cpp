/* Host twins of the curve kernels. The device entry returns 2 here. */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" {
void lc_part_host_histogram(const uint32_t *bins, size_t n, uint64_t *weight,
                            size_t ncell);
void lc_part_host_morton(const uint32_t *coords, size_t n, uint64_t *keys);
void lc_part_host_prefix(const uint64_t *values, size_t n, uint64_t *out);
uint64_t lc_part_host_hilbert(uint32_t x, uint32_t y, uint32_t z, uint32_t bits);
int lc_gpu_curve_split(const uint32_t *atom_bin, size_t natom,
                       const uint32_t *coords, size_t ncell, uint32_t parts,
                       uint32_t *assign, uint64_t *out_weight, uint8_t method);
}

static int fail(const char *msg) {
  fprintf(stderr, "%s\n", msg);
  return 1;
}

int main(void) {
  const uint32_t bins[] = {0, 1, 1, 2, 0};
  uint64_t weight[4];
  lc_part_host_histogram(bins, 5, weight, 4);
  if (weight[0] != 2 || weight[1] != 2 || weight[2] != 1 || weight[3] != 0) {
    return fail("histogram");
  }
  uint64_t prefix[5];
  lc_part_host_prefix(weight, 4, prefix);
  if (prefix[0] != 0 || prefix[1] != 2 || prefix[2] != 4 || prefix[3] != 5 ||
      prefix[4] != 5) {
    return fail("prefix");
  }
  const uint32_t coords[] = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
  uint64_t keys[4];
  lc_part_host_morton(coords, 4, keys);
  if (keys[0] != 0 || keys[1] != 1 || keys[2] != 2 || keys[3] != 4) {
    return fail("morton");
  }
  if (lc_part_host_hilbert(0, 0, 0, 1) != 0 ||
      lc_part_host_hilbert(1, 0, 0, 1) != 7 ||
      lc_part_host_hilbert(0, 1, 0, 2) != 1 ||
      lc_part_host_hilbert(1, 1, 1, 2) != 5) {
    return fail("hilbert key");
  }
  int seen[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (uint32_t z = 0; z < 2; ++z) {
    for (uint32_t y = 0; y < 2; ++y) {
      for (uint32_t x = 0; x < 2; ++x) {
        uint64_t key = lc_part_host_hilbert(x, y, z, 1);
        if (key >= 8 || seen[key]) {
          return fail("hilbert");
        }
        seen[key] = 1;
      }
    }
  }
  uint32_t assign[1] = {7};
  uint64_t out_w[1] = {0};
  uint32_t one[] = {0, 0, 0};
  int rc = lc_gpu_curve_split(0, 0, one, 1, 1, assign, out_w, 0);
  if (rc != 2) {
    return fail("device should be absent");
  }
  return 0;
}
