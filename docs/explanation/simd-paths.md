# Single instruction, multiple data paths and dispatch

The mesh fold, the k-nearest scan, and the cutoff pair list share one runtime path.
`simd_path` returns its name: `scalar`, `avx2`, `avx512`, or `neon`.

## Selection

The selector is `select`.
It reads a test override on the calling thread.
It then reads `LINKCELL_SIMD`.
It then reads the widest path this CPU can run.
The environment variable is read once and cached.
Unknown names are ignored.
A name the CPU omits selects `scalar`.
Workers read that cached path.
The calling thread captures the path before any rayon fan-out.

`LINKCELL_SIMD` accepts `scalar`, `avx2`, `avx512`, and `neon`.
`avx2fma` is `avx2`.
`avx512f` is `avx512`.
`0` is `scalar`.

AVX-512 requires AVX-512F, AVX-512VL, POPCNT, and fused multiply-add (FMA) together.
A CPU with AVX-512F but not VL uses AVX2.
AVX2 requires FMA as well, so Sandy Bridge stays on `scalar`.
The Neon extension (NEON) is the aarch64 baseline.
A build passes no `-C target-cpu` flag and no `target-feature` flag.
Each kernel enables its own features and is called only after `select`.

## Bit identity

The same input gives the same output bits on every path and every thread count, down to the sign of zero.
Sums that are stored are separate multiplies and adds, in the scalar order `(dx * dx + dy * dy) + dz * dz`.

The same path may use a fused multiply-add for pair screening.
The screen keeps a lane only when the expanded distance is below the cutoff by more than `expanded_margin`.
A lane inside that band is decided by the direct formula.
The value written for every kept lane is the direct formula.
The margin is the bound from Páll and Hess, Comput. Phys. Commun. 184, 2641 (2013).
The doi is 10.1016/j.cpc.2013.06.003.
That tile margin is a short sum of terms no larger than the squares of the coordinates, the shift, and the cutoff.

The fold floor is minimage `floor_fast` for this architecture.
Truncation toward zero on x86_64 keeps an input of -0.
`f64::floor` elsewhere maps that input to +0.
Paths on one machine match each other.
Paths differ across that architecture split.

## Dispatch limits

`LINKCELL_SIMD=scalar` runs the portable loops.
The build omits an Advanced Vector Extensions (AVX) path without AVX2.
The build omits a vector length extension (SVE) path, because those intrinsics are nightly-only.
Per-microarchitecture checks on real nodes are the European Environment for Scientific Software Installations (EESSI) build bot.
Dröge et al. describe that bot in Softw. Pract. Exp. 53, 176 (2023).
The doi is 10.1002/spe.3075.
The checks use real nodes.
CI keeps an emulator out of this dispatch page.
