/* A host that compiles against the linkcell headers alone and takes the
 * library at run time: no linkcell symbol is linked. The path of the
 * shared library is argv[1], or LINKCELL_LIBRARY. The ABI the header
 * names is checked against the library before any other call. The row
 * grid is then taken from the operations table, whose null members are
 * operations the library lacks, and every row of a kept grid is compared
 * with lc_pairs_within_pbc on orthorhombic, triclinic, mixed, and open
 * cells, by visit and by buffer.
 * Exit 0 on success; a failure prints its line and exits 1. */
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "linkcell.h"

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, \
                    #cond);                                                  \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

/* The entry points this host uses, resolved by name. */
static struct {
    lc_abi (*abi_stamp)(void);
    int (*abi_compatible)(const lc_abi *);
    int (*capabilities)(lc_caps *);
    const char *(*last_error)(void);
    const char *(*version)(void);
    int (*pairs_within_pbc)(const double *, size_t, const lc_cell *,
                            const int *, double, const int *, double, int,
                            int, int, lc_pair *, size_t, size_t *);
    int (*row_grid_build)(const double *, size_t, const lc_cell *,
                          const int *, double, const int *, double, int, int,
                          lc_row_grid **);
    int (*row_grid_update)(lc_row_grid *, const double *, size_t,
                           const lc_cell *, const int *, double, const int *,
                           double, int, int);
    size_t (*row_grid_len)(const lc_row_grid *);
    void (*row_grid_free)(lc_row_grid *);
    int (*row_grid_rows)(const lc_row_grid *, size_t, size_t, int,
                         lc_row_visit, void *);
    int (*row_grid_point)(const lc_row_grid *, size_t, int, lc_row_visit,
                          void *);
    int (*row_grid_fill)(const lc_row_grid *, lc_row_cursor *, size_t, int,
                         lc_row *, size_t, size_t *);
    int (*row_grid_bound)(const lc_row_grid *, uint64_t *);
    int (*row_ops_get)(lc_row_ops *);
} lc;

static void *symbol(void *lib, const char *name) {
    void *found = dlsym(lib, name);
    if (found == NULL) {
        fprintf(stderr, "missing symbol %s\n", name);
        exit(1);
    }
    return found;
}

/* A function pointer through an object pointer, as dlsym returns it. */
#define LOAD(member, name)                    \
    do {                                      \
        void *sym_ = symbol(lib, name);       \
        memcpy(&lc.member, &sym_, sizeof sym_); \
    } while (0)

static uint64_t rng_state = UINT64_C(0x9e3779b97f4a7c15);

static double next(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (double)(rng_state >> 11) / 9007199254740992.0;
}

typedef struct {
    lc_pair *rows;
    size_t count;
    size_t cap;
    /* Stop when this many rows are held; zero never stops. */
    size_t stop_after;
    /* Largest |vector . vector - dist2| seen. */
    double worst;
} seen_t;

static int collect(void *user, const lc_row *row) {
    seen_t *seen = (seen_t *)user;
    CHECK(seen->count < seen->cap);
    lc_pair *out = &seen->rows[seen->count++];
    out->i = row->i;
    out->j = row->j;
    memcpy(out->shift, row->shift, sizeof out->shift);
    out->dist2 = row->dist2;
    const double sq = row->vector[0] * row->vector[0] +
                      row->vector[1] * row->vector[1] +
                      row->vector[2] * row->vector[2];
    const double off = sq > row->dist2 ? sq - row->dist2 : row->dist2 - sq;
    if (off > seen->worst) {
        seen->worst = off;
    }
    return seen->count == seen->stop_after ? 7 : 0;
}

static int by_row(const void *a, const void *b) {
    const lc_pair *x = (const lc_pair *)a;
    const lc_pair *y = (const lc_pair *)b;
    if (x->i != y->i) {
        return x->i < y->i ? -1 : 1;
    }
    if (x->j != y->j) {
        return x->j < y->j ? -1 : 1;
    }
    for (int k = 0; k < 3; ++k) {
        if (x->shift[k] != y->shift[k]) {
            return x->shift[k] < y->shift[k] ? -1 : 1;
        }
    }
    return 0;
}

static int same_rows(const lc_pair *a, const lc_pair *b, size_t count) {
    for (size_t k = 0; k < count; ++k) {
        if (a[k].i != b[k].i || a[k].j != b[k].j ||
            memcmp(a[k].shift, b[k].shift, sizeof a[k].shift) != 0 ||
            memcmp(&a[k].dist2, &b[k].dist2, sizeof(double)) != 0) {
            return 0;
        }
    }
    return 1;
}

/* Rows of the grid, half and full, against the pair list; single points
 * and split ranges against the whole walk; an early stop. */
static void parity(const char *name, const lc_cell *cell, const int *pbc,
                   size_t n, double lo, double hi, double cutoff) {
    double *xyz = (double *)malloc(3 * n * sizeof *xyz);
    CHECK(xyz != NULL);
    for (size_t i = 0; i < n; ++i) {
        const double s[3] = {lo + (hi - lo) * next(), lo + (hi - lo) * next(),
                             lo + (hi - lo) * next()};
        xyz[3 * i] = s[0] * cell->ax + s[1] * cell->bx + s[2] * cell->cx;
        xyz[3 * i + 1] = s[0] * cell->ay + s[1] * cell->by + s[2] * cell->cy;
        xyz[3 * i + 2] = s[0] * cell->az + s[1] * cell->bz + s[2] * cell->cz;
    }
    lc_row_grid *grid = NULL;
    if (lc.row_grid_build(xyz, n, cell, pbc, cutoff, NULL, 0.0, 0, 0, &grid) !=
        0) {
        fprintf(stderr, "%s: build: %s\n", name, lc.last_error());
        exit(1);
    }
    CHECK(grid != NULL);
    CHECK(lc.row_grid_len(grid) == n);
    uint64_t bound = 0;
    CHECK(lc.row_grid_bound(grid, &bound) == 0);
    for (int half = 0; half < 2; ++half) {
        size_t count = 0;
        CHECK(lc.pairs_within_pbc(xyz, n, cell, pbc, cutoff, NULL, 0.0, half,
                                  0, 0, NULL, 0, &count) == 0);
        CHECK(count > 0);
        lc_pair *want = (lc_pair *)malloc(count * sizeof *want);
        lc_pair *whole = (lc_pair *)malloc(count * sizeof *whole);
        lc_pair *split = (lc_pair *)malloc(count * sizeof *split);
        CHECK(want != NULL && whole != NULL && split != NULL);
        size_t wrote = 0;
        CHECK(lc.pairs_within_pbc(xyz, n, cell, pbc, cutoff, NULL, 0.0, half,
                                  0, 0, want, count, &wrote) == 0);
        CHECK(wrote == count);

        seen_t seen = {whole, 0, count, 0, 0.0};
        CHECK(lc.row_grid_rows(grid, 0, n, half, collect, &seen) == 0);
        CHECK(lc.last_error() == NULL);
        CHECK(seen.count == count);
        /* The occupancy bound covers the full list. */
        CHECK(half != 0 || bound >= count);
        /* The vector is the difference dist2 was squared from; a compiler
         * that fuses the multiply-adds here rounds the sum differently. */
        CHECK(seen.worst <= 1.0e-12 * cutoff * cutoff);

        /* A third by range, a third point by point, the rest by range. */
        seen_t parts = {split, 0, count, 0, 0.0};
        const size_t a = n / 3;
        const size_t b = 2 * n / 3;
        CHECK(lc.row_grid_rows(grid, 0, a, half, collect, &parts) == 0);
        for (size_t i = a; i < b; ++i) {
            CHECK(lc.row_grid_point(grid, i, half, collect, &parts) == 0);
        }
        CHECK(lc.row_grid_rows(grid, b, n, half, collect, &parts) == 0);
        CHECK(parts.count == count);
        CHECK(same_rows(split, whole, count));

        /* The same rows through a buffer of 29, one call per buffer. */
        lc_row block[29];
        lc_row_cursor cursor = {0, 0};
        size_t filled = 0;
        while (cursor.point < n) {
            size_t got = 0;
            CHECK(lc.row_grid_fill(grid, &cursor, n, half, block, 29, &got) ==
                  0);
            CHECK(got <= 29);
            CHECK(got > 0 || cursor.point == n);
            for (size_t k = 0; k < got; ++k) {
                CHECK(filled < count);
                CHECK(block[k].i == whole[filled].i &&
                      block[k].j == whole[filled].j &&
                      memcmp(block[k].shift, whole[filled].shift,
                             sizeof block[k].shift) == 0 &&
                      memcmp(&block[k].dist2, &whole[filled].dist2,
                             sizeof(double)) == 0);
                ++filled;
            }
        }
        CHECK(filled == count);

        /* A nonzero return stops the walk and comes back, with no error. */
        seen_t some = {split, 0, count, 3, 0.0};
        CHECK(lc.row_grid_rows(grid, 0, n, half, collect, &some) == 7);
        CHECK(lc.last_error() == NULL);
        CHECK(some.count == 3);
        CHECK(same_rows(split, whole, 3));

        qsort(want, count, sizeof *want, by_row);
        qsort(whole, count, sizeof *whole, by_row);
        if (!same_rows(whole, want, count)) {
            fprintf(stderr, "%s half=%d: rows differ from the pair list\n",
                    name, half);
            exit(1);
        }
        printf("%s half=%d: %zu rows\n", name, half, count);
        free(want);
        free(whole);
        free(split);
    }
    /* Past the end, and without a callback. */
    seen_t none = {NULL, 0, 0, 0, 0.0};
    CHECK(lc.row_grid_rows(grid, 0, n + 1, 0, collect, &none) != 0);
    CHECK(strcmp(lc.last_error(), "point index is out of range") == 0);
    CHECK(lc.row_grid_point(grid, n, 0, collect, &none) != 0);
    CHECK(lc.row_grid_rows(grid, 0, n, 0, NULL, &none) != 0);
    CHECK(strcmp(lc.last_error(), "null pointer") == 0);
    CHECK(none.count == 0);
    /* An update that fails leaves a valid, empty grid. */
    CHECK(lc.row_grid_update(grid, xyz, n, cell, pbc, -1.0, NULL, 0.0, 0, 0) !=
          0);
    CHECK(lc.row_grid_len(grid) == 0);
    CHECK(lc.row_grid_update(grid, xyz, n / 2, cell, pbc, cutoff, NULL, 0.0, 0,
                             0) == 0);
    CHECK(lc.row_grid_len(grid) == n / 2);
    lc.row_grid_free(grid);
    free(xyz);
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : getenv("LINKCELL_LIBRARY");
    if (path == NULL) {
        fprintf(stderr, "usage: row_grid <liblinkcell>, or set LINKCELL_LIBRARY\n");
        return 2;
    }
    void *lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (lib == NULL) {
        fprintf(stderr, "dlopen %s: %s\n", path, dlerror());
        return 1;
    }
    /* The stamp first: nothing else is resolved from a library that does
     * not serve this header. */
    LOAD(abi_stamp, "lc_abi_stamp");
    LOAD(abi_compatible, "lc_abi_compatible");
    const lc_abi header = {LC_ABI_MAJOR, LC_ABI_MINOR, LC_ABI_LAYOUT_REVISION};
    const lc_abi loaded = lc.abi_stamp();
    const int ok = lc.abi_compatible(&header);
    LOAD(version, "lc_version");
    printf("abi library=linkcell header=%u.%u.%u loaded=%u.%u.%u version=%s "
           "status=%s\n",
           (unsigned)header.major, (unsigned)header.minor,
           (unsigned)header.layout_revision, (unsigned)loaded.major,
           (unsigned)loaded.minor, (unsigned)loaded.layout_revision,
           lc.version(), ok ? "verified" : "mismatch");
    CHECK(ok == 1);
    CHECK(loaded.major == header.major && loaded.minor == header.minor &&
          loaded.layout_revision == header.layout_revision);
    lc_abi newer = header;
    newer.minor = (uint16_t)(newer.minor + 1);
    CHECK(lc.abi_compatible(&newer) == 0);
    CHECK(lc.abi_compatible(NULL) == 0);

    /* One more symbol by name: the table of row-grid operations. */
    LOAD(row_ops_get, "lc_row_ops_get");
    lc_row_ops ops;
    memset(&ops, 0, sizeof ops);
    ops.size = (uint32_t)sizeof ops;
    CHECK(lc.row_ops_get(&ops) == 0);
    CHECK(ops.size == sizeof ops);
    /* A null member is an operation this library does not have. This
     * host needs these; build_request is optional and only reported. */
    CHECK(ops.build != NULL && ops.update != NULL && ops.len != NULL &&
          ops.free != NULL && ops.rows != NULL && ops.point != NULL &&
          ops.fill != NULL && ops.last_error != NULL &&
          ops.capabilities != NULL && ops.bound != NULL);
    lc.row_grid_bound = ops.bound;
    lc.row_grid_build = ops.build;
    lc.row_grid_update = ops.update;
    lc.row_grid_len = ops.len;
    lc.row_grid_free = ops.free;
    lc.row_grid_rows = ops.rows;
    lc.row_grid_point = ops.point;
    lc.row_grid_fill = ops.fill;
    lc.last_error = ops.last_error;
    lc.capabilities = ops.capabilities;
    printf("ops build_request=%s\n",
           ops.build_request != NULL ? "present" : "absent");
    lc_caps caps;
    memset(&caps, 0, sizeof caps);
    caps.size = (uint32_t)sizeof caps;
    CHECK(lc.capabilities(&caps) == 0);
    CHECK(caps.size == sizeof caps);
    CHECK(caps.row_grid == 1 && caps.row_vector == 1 && caps.row_stop == 1 &&
          caps.open_axes == 1 && caps.row_fill == 1 &&
          caps.closed_periodic == 1 && caps.row_bound == 1);
    CHECK((caps.row_request == 1) == (ops.build_request != NULL));
    printf("caps row_grid=%d row_vector=%d row_stop=%d row_request=%d "
           "open_axes=%d threads=%d\n",
           caps.row_grid, caps.row_vector, caps.row_stop, caps.row_request,
           caps.open_axes, caps.threads);

    /* The pair list this host compares the rows with. */
    LOAD(pairs_within_pbc, "lc_pairs_within_pbc");

    const int ppp[3] = {1, 1, 1};
    const int ppf[3] = {1, 1, 0};
    const int fpf[3] = {0, 1, 0};
    const int fff[3] = {0, 0, 0};
    const lc_cell box = lc_cell_ortho(15.0, 16.0, 14.0);
    const lc_cell tiny = lc_cell_ortho(6.0, 6.5, 7.0);
    const lc_cell tilt = {15.0, 0.0, 0.0, 4.5, 15.0, 0.0,
                          -3.0, 3.75, 15.0, 0.0, 0.0, 0.0};
    const lc_cell skew = {14.0, 2.94, -1.82, -4.76, 13.02, 2.38,
                          3.78, -5.74, 15.12, 0.0, 0.0, 0.0};
    parity("ortho", &box, ppp, 300, 0.0, 1.0, 3.5);
    parity("ortho outside", &box, ppp, 300, -1.3, 2.4, 3.5);
    parity("ortho past half", &tiny, ppp, 60, 0.0, 1.0, 4.0);
    parity("triclinic", &tilt, ppp, 300, -0.2, 1.2, 3.5);
    parity("skewed", &skew, ppp, 300, -1.1, 2.1, 3.5);
    parity("mixed ppf", &tilt, ppf, 200, -0.5, 1.5, 3.5);
    parity("mixed fpf", &skew, fpf, 200, -0.5, 1.5, 3.5);
    parity("open", &box, fff, 200, 0.0, 1.0, 3.5);

    lc.row_grid_free(NULL);
    CHECK(lc.row_grid_len(NULL) == 0);
    lc_row_grid *grid = NULL;
    const double one[3] = {0.0, 0.0, 0.0};
    CHECK(lc.row_grid_build(one, 0, &box, ppp, 3.5, NULL, 0.0, 0, 0, &grid) != 0);
    CHECK(grid == NULL);
    CHECK(strcmp(lc.last_error(), "no points") == 0);
    dlclose(lib);
    puts("row_grid: ok");
    return 0;
}
