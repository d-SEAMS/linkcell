#pragma once

#include <cstdint>

// Certified linked-cell k-nearest on Kokkos::DefaultExecutionSpace.
//
// The caller initializes Kokkos. linkcell is a sub-library: an MPI
// caller keeps the primary communicator and passes a sub-communicator.
// `box` is the walk cell: orthorhombic
// as stored, restricted triclinic already tilt-reduced, or a general
// cell already Minkowski-reduced. `mode` is 0, 1, or 2 in that order.
// `xyz` is row-major `n * 3`. `out_nn` is packed `n * k`, `-1` unused.
// `out_d2` may be null; otherwise it is packed `n * k`, NaN unused.
// `k` is at most 16. Returns 0 on success.

struct LcKokkosBox {
    double a[3];
    double b[3];
    double c[3];
    double origin[3];
    int mode;
};

int lc_kokkos_knearest(const double* xyz, int n, const LcKokkosBox& box, int k,
                       double cell_hint, int* out_nn, double* out_d2);

// When `slots` is non-null, each OpenMP thread adds its CPU time
// inside its RangePolicy slices into that slot. One sample covers the
// slice, not each index, and a preempted thread does not count the
// stall. Pass null to leave counters off. `slots` must hold at least
// the thread count entries.
void lc_kokkos_pop_bind(uint64_t* slots);
