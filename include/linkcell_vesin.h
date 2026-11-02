/* vesin-compatible entry points implemented by linkcell.
 *
 * The structs match vesin's layout. Fully periodic CPU lists are built
 * here. A non-periodic axis or a CUDA device is forwarded to libvesin.so
 * when that library is on the loader path.
 *
 * vesin_free releases a list this library filled. Do not pass it a list
 * allocated by another vesin build.
 */

#ifndef LINKCELL_VESIN_H
#define LINKCELL_VESIN_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    LC_VESIN_AUTO = 0,
    LC_VESIN_BRUTE = 1,
    LC_VESIN_CELL = 2
};

enum {
    LC_VESIN_CPU = 1,
    LC_VESIN_CUDA = 2
};

struct lc_vesin_device {
    int32_t kind;
    int32_t device_id;
};

struct lc_vesin_options {
    double cutoff;
    bool full;
    bool sorted;
    size_t padded_to;
    int32_t algorithm;
    double skin;
    int32_t n_threads;
    bool return_shifts;
    bool return_distances;
    bool return_vectors;
};

struct lc_vesin_list {
    size_t length;
    struct lc_vesin_device device;
    size_t (*pairs)[2];
    int32_t (*shifts)[3];
    double *distances;
    double (*vectors)[3];
    void *opaque;
};

int32_t vesin_neighbors(
    const double (*points)[3],
    size_t n_points,
    const double box[3][3],
    const bool periodic[3],
    struct lc_vesin_device device,
    struct lc_vesin_options options,
    struct lc_vesin_list *neighbors,
    const char **error_message
);

void vesin_free(struct lc_vesin_list *neighbors);

int32_t lc_vesin_neighbors(
    const double (*points)[3],
    size_t n_points,
    const double box[3][3],
    const bool periodic[3],
    struct lc_vesin_device device,
    struct lc_vesin_options options,
    struct lc_vesin_list *neighbors,
    const char **error_message
);

void lc_vesin_free(struct lc_vesin_list *neighbors);

#ifdef __cplusplus
}
#endif

#endif
