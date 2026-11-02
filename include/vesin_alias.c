/* Drop-in object. Link this instead of libvesin when every call is a
 * fully periodic CPU list. Mixed periodicity and CUDA stay in vesin.
 */
#include "linkcell_vesin.h"

int32_t vesin_neighbors(
    const double (*points)[3],
    size_t n_points,
    const double box[3][3],
    const bool periodic[3],
    struct lc_vesin_device device,
    struct lc_vesin_options options,
    struct lc_vesin_list *neighbors,
    const char **error_message
) {
    return lc_vesin_neighbors(
        points, n_points, box, periodic, device, options, neighbors, error_message
    );
}

void vesin_free(struct lc_vesin_list *neighbors) {
    lc_vesin_free(neighbors);
}
