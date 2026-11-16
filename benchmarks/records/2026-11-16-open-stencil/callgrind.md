# Instruction counts

`valgrind --tool=callgrind` on `open_band_bench` built with `CARGO_PROFILE_RELEASE_DEBUG=true`,
`OPEN_BAND_REPS=6 OPEN_BAND_CALLS=2000 OPEN_BAND_POINTS=<point>`, so 14000 calls of `pairs_open`
(one warmup sample and six timed). Instructions per call: the whole process divided by 14000, and
`open_cell::walk` inclusive divided by 14000. Instruction counts do not move with host noise.

| point | base total | v0 total | fix total | base walk | v0 walk | fix walk | fix/base total |
|---|---|---|---|---|---|---|---|
| water3_c2.0 | 4600 | 5671 | 4586 | 668 | 1741 | 629 | 0.997 |
| methane5_c2.0 | 7515 | 8693 | 7325 | 2023 | 3058 | 1663 | 0.975 |
| benzene12_c2.5 | 19391 | 31356 | 17341 | 11336 | 23303 | 9262 | 0.894 |
| cluster27_c2.5_h8 | 44540 | 45310 | 31144 | 32924 | 33630 | 19437 | 0.699 |
| cluster27_c1.5 | 49828 | 61377 | 36510 | 37422 | 48946 | 24052 | 0.733 |
| chain16_c2.0 | 22181 | 37624 | 18921 | 13399 | 28642 | 9912 | 0.853 |
| band8_chain16_c2.0 | 212716 | 63875 | 33341 | 21163 | 54254 | 23240 | 0.157 |
| band8_chain16_c3.0 | 54775 | 40936 | 23080 | 17509 | 31851 | 13968 | 0.421 |
| alt2_cluster27_c1.5 | 82018 | 81981 | 44793 | 44314 | 69300 | 32085 | 0.546 |
