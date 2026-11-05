/* Weak stand-in. The gpulite translation unit supplies the strong symbol. */

#include <stddef.h>
#include <stdint.h>

__attribute__((weak)) int lc_gpu_curve_split(const uint32_t *atom_bin, size_t natom,
                                             const uint32_t *coords, size_t ncell,
                                             uint32_t parts, uint32_t *assign,
                                             uint64_t *out_weight, uint8_t method) {
  (void)atom_bin;
  (void)natom;
  (void)coords;
  (void)ncell;
  (void)parts;
  (void)assign;
  (void)out_weight;
  (void)method;
  return 2;
}
