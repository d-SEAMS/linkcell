#include "linkcell.h"
#include <stdio.h>

int main(void) {
  double xyz[] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
  lc_cell box = lc_cell_ortho(10.0, 10.0, 10.0);
  int out[2];
  if (lc_knearest(xyz, 2, &box, 1, NULL, 0.0, out) != 0) {
    fprintf(stderr, "%s\n", lc_last_error());
    return 1;
  }
  printf("0 -> %d\n1 -> %d\n", out[0], out[1]);
  return 0;
}
