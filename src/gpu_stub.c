/* Weak stand-in. The gpulite translation unit supplies the strong symbol. */

#include <stddef.h>
#include <stdint.h>

/* MSVC has no weak definitions. The stub takes its own name, and the
 * linker resolves lc_gpu_curve_split to it only when no other object
 * defines that symbol. */
#if defined(_MSC_VER)
#define LC_STUB_NAME lc_gpu_curve_split_stub
#if defined(_M_IX86)
#pragma comment(linker, "/alternatename:_lc_gpu_curve_split=_lc_gpu_curve_split_stub")
#else
#pragma comment(linker, "/alternatename:lc_gpu_curve_split=lc_gpu_curve_split_stub")
#endif
#else
#define LC_STUB_NAME lc_gpu_curve_split
#endif

#if !defined(_MSC_VER)
__attribute__((weak))
#endif
int LC_STUB_NAME(const uint32_t *atom_bin, size_t natom,
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
