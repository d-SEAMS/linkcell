# Single instruction, multiple data (SIMD) paths and dispatch

The mesh fold shares one runtime path.
The k-nearest scan uses the path.
The cutoff pair list uses the path too.
`simd_path` returns `scalar`, `avx2`, `avx512`, or `neon`.

## Selection

`select` reads a test override on the calling thread.
`select` then reads `LINKCELL_SIMD`.
The widest path this CPU can run wins after that. The environment
variable is read once and cached. Unknown names are ignored. A name
outside this CPU's instruction set selects `scalar`. Workers keep the cached path.
The calling thread captures the path before any rayon fan-out.

`LINKCELL_SIMD` accepts `scalar`, `avx2`, `avx512`, and `neon`.
`avx2fma` is `avx2`. `avx512f` is `avx512`. `0` is `scalar`.

AVX-512 requires AVX-512F, AVX-512VL, POPCNT, and fused multiply-add (FMA) together. A CPU
with AVX-512F but not VL uses AVX2. AVX2 requires FMA as well, so
Sandy Bridge stays on `scalar`. The aarch64 vector extension (NEON) is the baseline. A build
never passes `-C target-cpu` or a `target-feature` flag. Each kernel
enables its own features and is called only after `select`.

## Bit identity

The same input gives the same output bits on every path and every thread
count, down to the sign of zero. Sums that are stored are separate
multiplies and adds, in the scalar order `(dx * dx + dy * dy) + dz * dz`.

Pair screening on this path may use a fused multiply-add. The screen keeps a lane
only when the expanded distance is below the cutoff by more than
`expanded_margin`. A lane inside that band is decided by the direct
formula. The value written for every kept lane is the direct formula,
not the expanded one. The margin is the bound from Páll and Hess,
Comput. Phys. Commun. 184, 2641 (2013), doi:10.1016/j.cpc.2013.06.003,
applied to this tile: a short sum of terms no larger than the squares
of the coordinates, the shift, and the cutoff.

The fold floor is minimage `floor_fast` for this architecture.
Truncation toward zero on x86_64 keeps an input of -0. `f64::floor`
elsewhere maps that input to +0. Paths on one machine match each other.
Paths differ across that architecture split.

## What is not dispatched

`LINKCELL_SIMD=scalar` runs the portable loops. The build omits an Advanced Vector Extensions (AVX) path
without AVX2. The build omits a vector-length path (SVE), because those intrinsics are nightly-only.
Per-microarchitecture checks on real nodes are the EESSI build bot
(Dröge et al., Softw. Pract. Exp. 53, 176 (2023), doi:10.1002/spe.3075),
not an emulator in CI.
