# Open-cell stencil keyed on the bin edge and cutoff

Serial A/B of three trees, each built with the same release profile
(`lto = true`, `codegen-units = 1`) and the same `open_band_bench.rs`:

| variant | commit | tree | what it is |
|---|---|---|---|
| base | `8874f131` (main) | `2b1649a1` | per-bin link list keyed on the bounding bins |
| v0 | `6f1ee0b1` | `dd48b3e6` | base plus the two-commit series: every offset within reach, keyed on the bin edge and cutoff |
| fix | `386daf09` | `8f9573e4` | v0 plus the clipped rows and the inlined open pair loop |

Host: `Intel(R) Xeon(R) Processor` (virtual), 8 processors, rustc
1.100.0. Every process ran pinned to CPU 5 with `RAYON_NUM_THREADS=1`
and `OMP_NUM_THREADS=1`, and nothing else was running.

## Method

`run-ab.sh raw base=TREE v0=TREE fix=TREE` ran six rounds. Each round
ran every variant once, in an order that rotates each round, so drift
on the host lands on all three. `summarize.py` pools the rounds.

- `open_band_bench`: 11 timed samples of 20000 calls per point and
  round, 66 per point. The small points are one molecule repeated in
  the same bins. The band points visit the images of one molecule in
  turn: `band8_*` is a 16-atom chain bent, tilted and moved over eight
  images, and `alt2_*` is the near and far frames of the
  `alternating_open_frames_share_one_stencil` test.
- `open_cell_bench` (the existing 27-atom, 400-move bench): the median
  of 11 repetitions per process, 6 per arm. `padded_ms` is the periodic
  padded box and does not take the open walk.
- `knn_bench` at 4096 and 110592 points, k = 4, 1 thread: 11 samples
  per process, 66 per point.
- `pairs_scale` at 256, 1024 and 4096 points, 1 thread: the mean of 50
  calls per process, two processes per round, 12 per point.

`summary.md` holds the medians, the IQRs, the after/before ratio of
the medians with a bootstrap 95% interval, and the stencil builds per
sample. A point is slower or faster only when the interval excludes 1
and the median moved by more than the larger IQR. `callgrind.md` holds
instruction counts per call, which do not move with host noise.

## Result

v0 removes the rebuilds on band frames (17500 or 20000 builds per
20000 calls down to 0) and runs those points at 0.21-0.76x of base,
but four of the six small points are slower, at 1.09-1.42x of base
time.

The callgrind profile of v0 shows the cost in `walk`. Every occupied
bin tried all 81 offsets within reach at a bin edge equal to the
cutoff, bounds-checked each against the frame, and called an outlined
`pair_bins` for each kept bin. A molecule in one bin discarded 80 of
the 81. On the 16-atom chain the walk went from 13.4k to 28.6k
instructions per call.

The fix stores the largest `|dx|` within reach on each `(dy, dz)` row
and clips the rows to the frame, so the walk only visits bins the
frame holds, in the same ascending flat order. The open pair loop is
inlined, has no shift or image work, and starts each bin at the first
partner above `i`. Against base it is 0.98x on water (within noise),
0.58-0.95x on the other small molecules, 0.12-0.44x on the band
frames, and 0.61x on `open_cell_bench`. The knn, pairs and padded-box
controls are within noise.

The handoff reported 9.37 s to 4.21 s and 340009 to 22 rebuilds on 44
gpridge NEB bands. Those frames are not in this repository. The band
points here reproduce the same mechanism on synthetic frames.

## Files

- `run-ab.sh`, `summarize.py`: the harness and the summary.
- `raw/`: every log, named `<bench>.<variant>.r<round>.log`.
- `summary.md`, `callgrind.md`: the tables above.
