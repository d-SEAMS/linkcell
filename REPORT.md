# Cell partition

The cell graph, the partition, and the geometric halo live in linkcell.
Ring membership, the Message Passing Interface (MPI) share, and ice clustering stay in seams-core.
eOn calls `pairs_within` only. rgpot builds pairs through vesin. readcon-core reads configuration (CON) frames.

This cell-partition report is generated from one log.
Every median and every interquartile range (IQR) is the value that log printed for that cell.
The raw sample line is the series those values came from, at type-7.

The log names the hardware as Intel Xeon Processor, 8 physical cores, one thread per core, one node, 2400 MHz.
A second node is recorded as `multi_node HOST NOT-RUN`.
Commit `457bf753ee2935d225a235a1c2003f93839e7a4b`.
Tree `ff098916a06c74faa0b141364136bf3dff9d12df`.
Exit code 0.

## Balance

The bound is `1 + parts * w_max / total`.
A weighted median gives the straddling cell to one side, so a part can exceed the ideal share by at most the heaviest cell.
The ratio is `1 + parts * heaviest_cell / total_weight`.
The same formula is used on every system.

## Cutoff and density

The mW cutoff is 4.306500 Ångströms, which is 1.8 times sigma with sigma 2.3925 Ångströms.
The TIP4P cutoff is 7.5 Ångströms, the spherical cutoff in Jorgensen, Chandrasekhar, Madura, Impey, and Klein, Journal of Chemical Physics 79, 926 (1983).
The water density used to place molecules is 0.03342796 molecules per cubic Ångström.
Basis: 1 g cm-3, molar mass 18.01528 g mol-1.
One cubic centimetre is 1e24 cubic Ångströms, so the conversion from moles per cubic centimetre to molecules per cubic Ångström uses the factor 1e-24.

## Scotch seed

Serial Scotch and parallel Scotch both call `SCOTCH_randomReset` and then `SCOTCH_randomSeed` with seed 1.
The panel log repeats `SCOTCH_randomReset 1` and `SCOTCH_randomSeed 1` for each parallel Scotch run.

## Head to head

Times are seconds.
The pair is the median and the interquartile range (IQR) of the raw samples.
The halo fraction is the largest ghost-cell count divided by the owned-cell count.
A value above 1 means the ghost set is larger than the owned set.

```text
method system cut imbalance halo median iqr eligible
morton mW-110592 20894 1.000145 0.941354 0.020751 0.001418 1
hilbert mW-110592 20694 1.000145 0.943571 0.002327 0.000241 1
rcb mW-110592 13824 1.000000 0.396092 0.002620 0.000212 1
hilbert-fm mW-110592 20594 1.000579 0.942531 0.012357 0.001241 1
rcb-fm mW-110592 13824 1.000000 0.396092 0.009003 0.000641 1
multi mW-110592 13824 1.000000 0.396092 0.028714 0.002688 1
scotch mW-110592 15547 1.000217 0.478564 0.123440 0.008815 0
zoltan-rcb mW-110592 13824 1.000000 0.396092 0.003796 0.000296 0
ptscotch mW-110592 14402 1.000000 0.415141 13.732653 0.290923 0
morton mW-884736 83857 1.000009 0.397780 0.183753 0.009531 1
hilbert mW-884736 81032 1.000018 0.418792 0.021357 0.000737 1
rcb mW-884736 55990 1.000009 0.186900 0.016481 0.002483 1
hilbert-fm mW-884736 80931 1.000072 0.418751 0.070325 0.002711 1
rcb-fm mW-884736 55802 1.000054 0.183404 0.139626 0.018501 1
multi mW-884736 55800 1.000072 0.183409 0.869452 0.050697 1
scotch mW-884736 64754 1.000054 0.232028 0.693373 0.029713 0
zoltan-rcb mW-884736 55983 1.000009 0.186892 0.029397 0.000592 0
ptscotch mW-884736 70532 1.000000 0.258921 55.432768 1.271927 0
morton TIP4P-128000 2804168 1.000000 1.687500 0.000848 0.000060 1
hilbert TIP4P-128000 2791850 1.000687 1.661710 0.000117 0.000013 1
rcb TIP4P-128000 2492438 1.000625 1.353383 0.000373 0.000059 1
hilbert-fm TIP4P-128000 2671517 1.005625 1.500000 0.001838 0.000164 1
rcb-fm TIP4P-128000 2145600 1.000000 1.000000 0.002270 0.000155 1
multi TIP4P-128000 2145600 1.000000 1.000000 0.005105 0.000205 1
scotch TIP4P-128000 2169032 1.005375 1.052632 0.016222 0.000545 0
zoltan-rcb TIP4P-128000 2485058 1.000625 1.353383 0.000268 0.000034 0
ptscotch TIP4P-128000 2130992 1.001250 1.007692 3.484962 0.231274 0
morton TIP4P-1024000 13084202 1.000320 0.812144 0.008060 0.000579 1
hilbert TIP4P-1024000 12710308 1.000195 0.848197 0.001175 0.000060 1
rcb TIP4P-1024000 9965162 1.000266 0.576619 0.001298 0.000087 1
hilbert-fm TIP4P-1024000 12192983 1.000758 0.834594 0.025455 0.002468 1
rcb-fm TIP4P-1024000 8793439 1.000766 0.466543 0.012366 0.000661 1
multi TIP4P-1024000 8800880 1.000766 0.466263 0.038720 0.002756 1
scotch TIP4P-1024000 8911827 1.000719 0.490884 0.105480 0.004425 0
zoltan-rcb TIP4P-1024000 9897435 1.000312 0.564509 0.001604 0.000156 0
ptscotch TIP4P-1024000 8702272 1.000203 0.468284 5.014207 0.242389 0
morton slab-110592 18819 1.000217 0.750051 0.039945 0.000919 1
hilbert slab-110592 18188 1.000145 0.747628 0.004784 0.000261 1
rcb slab-110592 12672 1.000000 0.500000 0.003311 0.000238 1
hilbert-fm slab-110592 18148 1.000579 0.746570 0.009772 0.000393 1
rcb-fm slab-110592 11520 1.000000 0.396092 0.010881 0.000389 1
multi slab-110592 11520 1.000000 0.396092 0.065302 0.003402 1
scotch slab-110592 12553 1.000579 0.531308 0.146309 0.002972 0
zoltan-rcb slab-110592 12672 1.000000 0.500000 0.007598 0.000241 0
ptscotch slab-110592 11895 1.000000 2.006019 24.356326 0.389076 0
```

## Halo counts

Each row is the domain with the largest ghost-to-owned ratio on the last method that printed halo counts, with bin widths in Ångströms.

```text
mW-110592 owned 5015 ghosts 2400 cells 39304 bin_widths 4.382538 4.382538 4.382538
mW-884736 owned 40771 ghosts 9460 cells 328509 bin_widths 4.319023 4.319023 4.319023
TIP4P-128000 owned 266 ghosts 280 cells 2080 bin_widths 7.760744 7.641348 7.760744
TIP4P-1024000 owned 2139 ghosts 1050 cells 17160 bin_widths 7.760744 7.641348 7.525570
slab-110592 owned 10205 ghosts 5422 cells 79764 bin_widths 4.382538 4.382538 4.319023
```

On the small TIP4P lattice the bin is about as wide as the 7.5 Ångström cutoff, so the radius is one bin.
That part owns a few hundred cells.
The one-bin shell can contain more cells than the part.
The fraction divides ghosts by owned cells.

## Cut sums

Each sum adds the edge cut of one method over the five systems in this log.

```text
sum method morton cut 16011940 median_s 0.253357 eligible 1
sum method hilbert cut 15622072 median_s 0.029759 eligible 1
sum method rcb cut 12540086 median_s 0.024082 eligible 1
sum method hilbert-fm cut 14984173 median_s 0.119748 eligible 1
sum method rcb-fm cut 11020185 median_s 0.174146 eligible 1
sum method multi cut 11027624 median_s 1.007294 eligible 1
default_method rcb-fm
```

## Raw samples

Each line is the twelve wall times, in seconds, in the order they were measured.

```text
morton mW-110592 0.020082 0.021161 0.022187 0.020876 0.020688 0.037650 0.020965 0.019536 0.020813 0.019423 0.019490 0.019617
hilbert mW-110592 0.002180 0.002797 0.002524 0.002271 0.002478 0.003283 0.002348 0.002259 0.002306 0.002353 0.002218 0.002183
rcb mW-110592 0.002830 0.002555 0.002935 0.002626 0.002732 0.002589 0.002820 0.002714 0.002614 0.002354 0.002364 0.002500
hilbert-fm mW-110592 0.011726 0.012291 0.012939 0.012424 0.013357 0.017522 0.012702 0.011693 0.013016 0.011573 0.011797 0.011427
rcb-fm mW-110592 0.009293 0.009353 0.009413 0.008734 0.009598 0.010277 0.008757 0.008854 0.009153 0.008431 0.008705 0.008588
multi mW-110592 0.027024 0.028560 0.028868 0.027481 0.034017 0.033846 0.029901 0.026603 0.029384 0.027308 0.029997 0.026095
scotch mW-110592 0.130023 0.122334 0.127879 0.122813 0.130730 0.133606 0.125753 0.117178 0.124066 0.117768 0.120211 0.114390
zoltan-rcb mW-110592 0.004174 0.003754 0.004050 0.003962 0.003996 0.003672 0.003850 0.003601 0.003542 0.003676 0.003774 0.003819
ptscotch mW-110592 13.799427 13.622922 13.653087 14.315775 13.659692 13.741357 13.812411 13.723949 13.170680 17.325602 14.308641 13.517461
morton mW-884736 0.184759 0.182766 0.183259 0.182663 0.184007 0.182440 0.188240 0.354803 0.182078 0.216660 0.183498 0.204365
hilbert mW-884736 0.021364 0.021455 0.021309 0.021235 0.021298 0.021281 0.022068 0.056310 0.021349 0.027054 0.021138 0.022018
rcb mW-884736 0.016480 0.016361 0.019389 0.016150 0.016198 0.016037 0.016713 0.089704 0.020217 0.018491 0.016244 0.016482
hilbert-fm mW-884736 0.072232 0.070349 0.069880 0.068306 0.071357 0.069703 0.073491 0.313940 0.070223 0.098444 0.068751 0.070302
rcb-fm mW-884736 0.141946 0.136748 0.165704 0.132763 0.133825 0.133460 0.162233 0.447756 0.137306 0.151946 0.148718 0.136778
multi mW-884736 0.902139 0.853249 0.898333 0.855881 0.866695 0.834330 0.865309 1.427947 0.872210 0.917261 0.832940 0.941178
scotch mW-884736 0.702560 0.685319 0.724126 0.677903 0.688683 0.666723 0.682940 0.710735 0.680975 0.735236 0.698062 0.716442
zoltan-rcb mW-884736 0.027968 0.029284 0.029596 0.029321 0.031317 0.029426 0.030487 0.029070 0.029139 0.032048 0.029367 0.029624
ptscotch mW-884736 54.649985 54.589043 56.089985 53.953640 55.887464 55.544464 55.964313 56.097124 55.578350 55.321072 53.984806 55.067316
morton TIP4P-128000 0.000836 0.000889 0.000810 0.000861 0.000867 0.000813 0.000993 0.000865 0.000833 0.000975 0.000806 0.000810
hilbert TIP4P-128000 0.000116 0.000137 0.000115 0.000123 0.000144 0.000115 0.000127 0.000114 0.000115 0.000132 0.000117 0.000115
rcb TIP4P-128000 0.000382 0.000331 0.000473 0.000350 0.000396 0.000220 0.000364 0.000602 0.000384 0.000297 0.000362 0.000430
hilbert-fm TIP4P-128000 0.001747 0.001825 0.001935 0.001851 0.001771 0.001962 0.001915 0.002068 0.001779 0.002026 0.001759 0.001791
rcb-fm TIP4P-128000 0.002077 0.002138 0.002281 0.002311 0.002214 0.002308 0.002259 0.002160 0.002282 0.002052 0.002362 0.002320
multi TIP4P-128000 0.005151 0.005080 0.005130 0.005333 0.004838 0.005021 0.005398 0.005197 0.004908 0.005065 0.004954 0.005246
scotch TIP4P-128000 0.015749 0.017056 0.016785 0.015175 0.016303 0.015946 0.016472 0.016431 0.015971 0.016141 0.016324 0.015419
zoltan-rcb TIP4P-128000 0.000253 0.000288 0.000236 0.000261 0.000389 0.000264 0.000275 0.000255 0.000272 0.000289 0.000294 0.000247
ptscotch TIP4P-128000 3.590629 3.403058 3.494592 3.485994 3.646496 3.335563 3.757427 3.362398 3.730986 3.299870 3.376963 3.483929
morton TIP4P-1024000 0.007782 0.007939 0.012604 0.008180 0.008556 0.007695 0.007769 0.012590 0.008323 0.008285 0.007866 0.007808
hilbert TIP4P-1024000 0.001187 0.001165 0.001186 0.001135 0.001217 0.001150 0.001156 0.001526 0.001265 0.001156 0.001156 0.001216
rcb TIP4P-1024000 0.001332 0.001293 0.001489 0.001351 0.001096 0.001339 0.001162 0.004662 0.001302 0.001242 0.001260 0.001276
hilbert-fm TIP4P-1024000 0.024414 0.025586 0.025325 0.025606 0.027748 0.024418 0.024430 0.048632 0.027626 0.027084 0.025102 0.024859
rcb-fm TIP4P-1024000 0.011937 0.012279 0.012511 0.012249 0.013009 0.011848 0.011847 0.023967 0.013086 0.012513 0.012452 0.011989
multi TIP4P-1024000 0.036959 0.039173 0.038334 0.037585 0.042821 0.036892 0.036606 0.082241 0.040611 0.039106 0.039394 0.036339
scotch TIP4P-1024000 0.102210 0.105594 0.108831 0.105366 0.105993 0.100667 0.100022 0.220685 0.108919 0.104842 0.106006 0.102313
zoltan-rcb TIP4P-1024000 0.001653 0.001541 0.001640 0.001565 0.001717 0.001528 0.001533 0.005489 0.001568 0.001965 0.001554 0.001703
ptscotch TIP4P-1024000 4.886127 5.061817 5.119448 5.191130 4.966597 4.798540 4.960399 5.128552 5.176104 4.887247 4.756534 5.131768
morton slab-110592 0.040063 0.039386 0.045334 0.042211 0.039920 0.039970 0.039570 0.043291 0.039651 0.040021 0.039818 0.039691
hilbert slab-110592 0.004764 0.004791 0.005838 0.005291 0.004776 0.004683 0.004739 0.005651 0.004845 0.004889 0.004697 0.004660
rcb slab-110592 0.003151 0.003422 0.003691 0.003450 0.003336 0.003255 0.003204 0.003457 0.003141 0.003290 0.003331 0.003147
hilbert-fm slab-110592 0.009480 0.009699 0.010783 0.010690 0.009754 0.009786 0.009842 0.010730 0.009759 0.009887 0.009676 0.009683
rcb-fm slab-110592 0.010935 0.010645 0.011405 0.012152 0.010923 0.010809 0.010975 0.012805 0.010709 0.010538 0.010632 0.010838
multi slab-110592 0.064952 0.064226 0.073028 0.076024 0.066046 0.065653 0.064318 0.073644 0.064775 0.065975 0.064261 0.064413
scotch slab-110592 0.145349 0.146740 0.155613 0.147656 0.150444 0.145877 0.142793 0.150689 0.144922 0.145391 0.145675 0.146983
zoltan-rcb slab-110592 0.007398 0.007521 0.007917 0.007710 0.007764 0.007686 0.007339 0.007490 0.007241 0.007708 0.007572 0.007624
ptscotch slab-110592 24.319736 24.245686 23.637196 24.849140 24.660374 24.490818 24.636140 24.502452 24.094873 24.164106 24.392916 23.986620
```

## Balance sweep

The same five ratios appear on every system in this sweep.
The main table uses the derived bound.

```text
morton 1.00 mW-110592 cut 20894 imbalance 1.000145 median 0.019213 iqr 0.000344
hilbert 1.00 mW-110592 cut 20694 imbalance 1.000145 median 0.002113 iqr 0.000092
rcb 1.00 mW-110592 cut 13824 imbalance 1.000000 median 0.002324 iqr 0.000144
hilbert-fm 1.00 mW-110592 cut 20692 imbalance 1.000000 median 0.005352 iqr 0.000122
rcb-fm 1.00 mW-110592 cut 13824 imbalance 1.000000 median 0.007907 iqr 0.000092
multi 1.00 mW-110592 cut 13824 imbalance 1.000000 median 0.025542 iqr 0.001051
scotch 1.00 mW-110592 cut 20801 imbalance 1.000000 median 0.116202 iqr 0.002582
morton 1.01 mW-110592 cut 20894 imbalance 1.000145 median 0.019100 iqr 0.000264
hilbert 1.01 mW-110592 cut 20694 imbalance 1.000145 median 0.002105 iqr 0.000030
rcb 1.01 mW-110592 cut 13824 imbalance 1.000000 median 0.002198 iqr 0.000125
hilbert-fm 1.01 mW-110592 cut 20584 imbalance 1.002604 median 0.005424 iqr 0.000130
rcb-fm 1.01 mW-110592 cut 13824 imbalance 1.000000 median 0.008363 iqr 0.000186
multi 1.01 mW-110592 cut 13824 imbalance 1.000000 median 0.026598 iqr 0.000123
scotch 1.01 mW-110592 cut 14698 imbalance 1.009910 median 0.115691 iqr 0.002055
morton 1.02 mW-110592 cut 20894 imbalance 1.000145 median 0.019082 iqr 0.000195
hilbert 1.02 mW-110592 cut 20694 imbalance 1.000145 median 0.002122 iqr 0.000116
rcb 1.02 mW-110592 cut 13824 imbalance 1.000000 median 0.002250 iqr 0.000092
hilbert-fm 1.02 mW-110592 cut 20584 imbalance 1.002604 median 0.005409 iqr 0.000197
rcb-fm 1.02 mW-110592 cut 13824 imbalance 1.000000 median 0.008504 iqr 0.000300
multi 1.02 mW-110592 cut 13824 imbalance 1.000000 median 0.026978 iqr 0.000655
scotch 1.02 mW-110592 cut 14391 imbalance 1.019676 median 0.101720 iqr 0.001097
morton 1.05 mW-110592 cut 20894 imbalance 1.000145 median 0.019016 iqr 0.000214
hilbert 1.05 mW-110592 cut 20694 imbalance 1.000145 median 0.002140 iqr 0.000094
rcb 1.05 mW-110592 cut 13824 imbalance 1.000000 median 0.002301 iqr 0.000107
hilbert-fm 1.05 mW-110592 cut 20584 imbalance 1.002604 median 0.005334 iqr 0.000146
rcb-fm 1.05 mW-110592 cut 13824 imbalance 1.000000 median 0.008330 iqr 0.000139
multi 1.05 mW-110592 cut 13824 imbalance 1.000000 median 0.027004 iqr 0.000319
scotch 1.05 mW-110592 cut 14817 imbalance 1.045139 median 0.115268 iqr 0.001437
morton 1.10 mW-110592 cut 20894 imbalance 1.000145 median 0.019084 iqr 0.000245
hilbert 1.10 mW-110592 cut 20694 imbalance 1.000145 median 0.002100 iqr 0.000058
rcb 1.10 mW-110592 cut 13824 imbalance 1.000000 median 0.002346 iqr 0.000162
hilbert-fm 1.10 mW-110592 cut 20584 imbalance 1.002604 median 0.005352 iqr 0.000178
rcb-fm 1.10 mW-110592 cut 13824 imbalance 1.000000 median 0.008431 iqr 0.000301
multi 1.10 mW-110592 cut 13824 imbalance 1.000000 median 0.027538 iqr 0.001789
scotch 1.10 mW-110592 cut 15098 imbalance 1.099826 median 0.118129 iqr 0.001898
morton 1.00 mW-884736 cut 83857 imbalance 1.000009 median 0.181369 iqr 0.000912
hilbert 1.00 mW-884736 cut 81032 imbalance 1.000018 median 0.021027 iqr 0.000210
rcb 1.00 mW-884736 cut 55990 imbalance 1.000009 median 0.016313 iqr 0.000399
hilbert-fm 1.00 mW-884736 cut 81032 imbalance 1.000000 median 0.040840 iqr 0.000790
rcb-fm 1.00 mW-884736 cut 55990 imbalance 1.000009 median 0.098337 iqr 0.004509
multi 1.00 mW-884736 cut 55990 imbalance 1.000009 median 0.279509 iqr 0.011370
scotch 1.00 mW-884736 cut 81445 imbalance 1.000000 median 0.650919 iqr 0.010032
morton 1.01 mW-884736 cut 83857 imbalance 1.000009 median 0.181381 iqr 0.001064
hilbert 1.01 mW-884736 cut 81032 imbalance 1.000018 median 0.020847 iqr 0.000140
rcb 1.01 mW-884736 cut 55990 imbalance 1.000009 median 0.015993 iqr 0.000752
hilbert-fm 1.01 mW-884736 cut 80799 imbalance 1.004792 median 0.235611 iqr 0.003665
rcb-fm 1.01 mW-884736 cut 55894 imbalance 1.000868 median 0.084283 iqr 0.001647
multi 1.01 mW-884736 cut 55837 imbalance 1.000922 median 0.465296 iqr 0.006128
scotch 1.01 mW-884736 cut 58097 imbalance 1.009865 median 0.594547 iqr 0.003199
morton 1.02 mW-884736 cut 83857 imbalance 1.000009 median 0.182134 iqr 0.004936
hilbert 1.02 mW-884736 cut 81032 imbalance 1.000018 median 0.020898 iqr 0.000840
rcb 1.02 mW-884736 cut 55990 imbalance 1.000009 median 0.015436 iqr 0.000946
hilbert-fm 1.02 mW-884736 cut 80799 imbalance 1.004792 median 0.237085 iqr 0.009760
rcb-fm 1.02 mW-884736 cut 55894 imbalance 1.000868 median 0.082375 iqr 0.002107
multi 1.02 mW-884736 cut 55837 imbalance 1.000922 median 0.464476 iqr 0.019126
scotch 1.02 mW-884736 cut 60287 imbalance 1.019938 median 0.609563 iqr 0.011940
morton 1.05 mW-884736 cut 83857 imbalance 1.000009 median 0.187129 iqr 0.007690
hilbert 1.05 mW-884736 cut 81032 imbalance 1.000018 median 0.021666 iqr 0.000927
rcb 1.05 mW-884736 cut 55990 imbalance 1.000009 median 0.016549 iqr 0.001006
hilbert-fm 1.05 mW-884736 cut 80799 imbalance 1.004792 median 0.246919 iqr 0.009701
rcb-fm 1.05 mW-884736 cut 55894 imbalance 1.000868 median 0.086078 iqr 0.004288
multi 1.05 mW-884736 cut 55837 imbalance 1.000922 median 0.487873 iqr 0.020429
scotch 1.05 mW-884736 cut 62348 imbalance 1.018971 median 0.627556 iqr 0.018801
morton 1.10 mW-884736 cut 83857 imbalance 1.000009 median 0.182872 iqr 0.004770
hilbert 1.10 mW-884736 cut 81032 imbalance 1.000018 median 0.021093 iqr 0.000678
rcb 1.10 mW-884736 cut 55990 imbalance 1.000009 median 0.015901 iqr 0.000345
hilbert-fm 1.10 mW-884736 cut 80799 imbalance 1.004792 median 0.238241 iqr 0.003281
rcb-fm 1.10 mW-884736 cut 55894 imbalance 1.000868 median 0.083322 iqr 0.001777
multi 1.10 mW-884736 cut 55837 imbalance 1.000922 median 0.472530 iqr 0.006796
scotch 1.10 mW-884736 cut 59853 imbalance 1.086842 median 0.623037 iqr 0.004061
morton 1.00 TIP4P-128000 cut 2804168 imbalance 1.000000 median 0.000786 iqr 0.000033
hilbert 1.00 TIP4P-128000 cut 2791850 imbalance 1.000687 median 0.000109 iqr 0.000004
rcb 1.00 TIP4P-128000 cut 2492438 imbalance 1.000625 median 0.000177 iqr 0.000067
hilbert-fm 1.00 TIP4P-128000 cut 2791850 imbalance 1.000687 median 0.000271 iqr 0.000016
rcb-fm 1.00 TIP4P-128000 cut 2145600 imbalance 1.000000 median 0.000696 iqr 0.000182
multi 1.00 TIP4P-128000 cut 2145600 imbalance 1.000000 median 0.003518 iqr 0.000274
scotch 1.00 TIP4P-128000 cut 2776979 imbalance 1.000687 median 0.009032 iqr 0.000768
morton 1.01 TIP4P-128000 cut 2804168 imbalance 1.000000 median 0.000808 iqr 0.000072
hilbert 1.01 TIP4P-128000 cut 2791850 imbalance 1.000687 median 0.000115 iqr 0.000014
rcb 1.01 TIP4P-128000 cut 2492438 imbalance 1.000625 median 0.000259 iqr 0.000188
hilbert-fm 1.01 TIP4P-128000 cut 2668242 imbalance 1.005250 median 0.002318 iqr 0.000166
rcb-fm 1.01 TIP4P-128000 cut 2130992 imbalance 1.001250 median 0.004811 iqr 0.000210
multi 1.01 TIP4P-128000 cut 2130992 imbalance 1.001250 median 0.007889 iqr 0.000308
scotch 1.01 TIP4P-128000 cut 2172264 imbalance 1.008563 median 0.015936 iqr 0.000668
morton 1.02 TIP4P-128000 cut 2804168 imbalance 1.000000 median 0.000774 iqr 0.000015
hilbert 1.02 TIP4P-128000 cut 2791850 imbalance 1.000687 median 0.000111 iqr 0.000001
rcb 1.02 TIP4P-128000 cut 2492438 imbalance 1.000625 median 0.000320 iqr 0.000253
hilbert-fm 1.02 TIP4P-128000 cut 2667911 imbalance 1.016563 median 0.001465 iqr 0.000068
rcb-fm 1.02 TIP4P-128000 cut 2145600 imbalance 1.000000 median 0.002633 iqr 0.000260
multi 1.02 TIP4P-128000 cut 2145600 imbalance 1.000000 median 0.005892 iqr 0.000160
scotch 1.02 TIP4P-128000 cut 2196133 imbalance 1.019062 median 0.014869 iqr 0.000729
morton 1.05 TIP4P-128000 cut 2804168 imbalance 1.000000 median 0.000793 iqr 0.000050
hilbert 1.05 TIP4P-128000 cut 2791850 imbalance 1.000687 median 0.000113 iqr 0.000008
rcb 1.05 TIP4P-128000 cut 2492438 imbalance 1.000625 median 0.000196 iqr 0.000127
hilbert-fm 1.05 TIP4P-128000 cut 2658326 imbalance 1.049063 median 0.001548 iqr 0.000083
rcb-fm 1.05 TIP4P-128000 cut 2120992 imbalance 1.050000 median 0.004717 iqr 0.000279
multi 1.05 TIP4P-128000 cut 2120512 imbalance 1.050000 median 0.016509 iqr 0.000467
scotch 1.05 TIP4P-128000 cut 2155060 imbalance 1.050000 median 0.013412 iqr 0.001187
morton 1.10 TIP4P-128000 cut 2804168 imbalance 1.000000 median 0.000820 iqr 0.000076
hilbert 1.10 TIP4P-128000 cut 2791850 imbalance 1.000687 median 0.000111 iqr 0.000005
rcb 1.10 TIP4P-128000 cut 2492438 imbalance 1.000625 median 0.000201 iqr 0.000123
hilbert-fm 1.10 TIP4P-128000 cut 2654022 imbalance 1.096750 median 0.001522 iqr 0.000069
rcb-fm 1.10 TIP4P-128000 cut 2092800 imbalance 1.062500 median 0.004959 iqr 0.000119
multi 1.10 TIP4P-128000 cut 2092800 imbalance 1.062500 median 0.008545 iqr 0.000179
scotch 1.10 TIP4P-128000 cut 2118900 imbalance 1.089063 median 0.013944 iqr 0.000560
morton 1.00 TIP4P-1024000 cut 13084202 imbalance 1.000320 median 0.007572 iqr 0.000100
hilbert 1.00 TIP4P-1024000 cut 12710308 imbalance 1.000195 median 0.001073 iqr 0.000014
rcb 1.00 TIP4P-1024000 cut 9965162 imbalance 1.000266 median 0.001133 iqr 0.000083
hilbert-fm 1.00 TIP4P-1024000 cut 12710308 imbalance 1.000195 median 0.002079 iqr 0.000037
rcb-fm 1.00 TIP4P-1024000 cut 9965162 imbalance 1.000266 median 0.005973 iqr 0.000178
multi 1.00 TIP4P-1024000 cut 9965162 imbalance 1.000266 median 0.029293 iqr 0.000641
scotch 1.00 TIP4P-1024000 cut 15041835 imbalance 1.000133 median 0.081377 iqr 0.000686
morton 1.01 TIP4P-1024000 cut 13084202 imbalance 1.000320 median 0.007689 iqr 0.000069
hilbert 1.01 TIP4P-1024000 cut 12710308 imbalance 1.000195 median 0.001080 iqr 0.000038
rcb 1.01 TIP4P-1024000 cut 9965162 imbalance 1.000266 median 0.001156 iqr 0.000064
hilbert-fm 1.01 TIP4P-1024000 cut 12126380 imbalance 1.010000 median 0.020557 iqr 0.000139
rcb-fm 1.01 TIP4P-1024000 cut 8741847 imbalance 1.009797 median 0.015049 iqr 0.000233
multi 1.01 TIP4P-1024000 cut 8762734 imbalance 1.009813 median 0.042718 iqr 0.000395
scotch 1.01 TIP4P-1024000 cut 8949479 imbalance 1.009922 median 0.108143 iqr 0.001131
morton 1.02 TIP4P-1024000 cut 13084202 imbalance 1.000320 median 0.007903 iqr 0.000449
hilbert 1.02 TIP4P-1024000 cut 12710308 imbalance 1.000195 median 0.001125 iqr 0.000101
rcb 1.02 TIP4P-1024000 cut 9965162 imbalance 1.000266 median 0.001176 iqr 0.000066
hilbert-fm 1.02 TIP4P-1024000 cut 12078105 imbalance 1.019977 median 0.008365 iqr 0.000673
rcb-fm 1.02 TIP4P-1024000 cut 8726026 imbalance 1.019969 median 0.015198 iqr 0.001002
multi 1.02 TIP4P-1024000 cut 8760713 imbalance 1.019969 median 0.045086 iqr 0.002773
scotch 1.02 TIP4P-1024000 cut 8861660 imbalance 1.020000 median 0.107095 iqr 0.006257
morton 1.05 TIP4P-1024000 cut 13084202 imbalance 1.000320 median 0.007758 iqr 0.000192
hilbert 1.05 TIP4P-1024000 cut 12710308 imbalance 1.000195 median 0.001111 iqr 0.000034
rcb 1.05 TIP4P-1024000 cut 9965162 imbalance 1.000266 median 0.001149 iqr 0.000158
hilbert-fm 1.05 TIP4P-1024000 cut 11893822 imbalance 1.050000 median 0.012413 iqr 0.000343
rcb-fm 1.05 TIP4P-1024000 cut 8743808 imbalance 1.050000 median 0.027021 iqr 0.005772
multi 1.05 TIP4P-1024000 cut 8746824 imbalance 1.050000 median 0.055130 iqr 0.005974
scotch 1.05 TIP4P-1024000 cut 8730300 imbalance 1.028125 median 0.097422 iqr 0.002069
morton 1.10 TIP4P-1024000 cut 13084202 imbalance 1.000320 median 0.007745 iqr 0.000115
hilbert 1.10 TIP4P-1024000 cut 12710308 imbalance 1.000195 median 0.001099 iqr 0.000058
rcb 1.10 TIP4P-1024000 cut 9965162 imbalance 1.000266 median 0.001193 iqr 0.000088
hilbert-fm 1.10 TIP4P-1024000 cut 11733490 imbalance 1.099828 median 0.015117 iqr 0.000301
rcb-fm 1.10 TIP4P-1024000 cut 8530800 imbalance 1.100000 median 0.014110 iqr 0.001253
multi 1.10 TIP4P-1024000 cut 8523300 imbalance 1.100000 median 0.069957 iqr 0.000884
scotch 1.10 TIP4P-1024000 cut 8812500 imbalance 1.058594 median 0.096097 iqr 0.003196
morton 1.00 slab-110592 cut 18819 imbalance 1.000217 median 0.039506 iqr 0.000173
hilbert 1.00 slab-110592 cut 18188 imbalance 1.000145 median 0.004616 iqr 0.000169
rcb 1.00 slab-110592 cut 12672 imbalance 1.000000 median 0.002837 iqr 0.000245
hilbert-fm 1.00 slab-110592 cut 18200 imbalance 1.000000 median 0.006887 iqr 0.000108
rcb-fm 1.00 slab-110592 cut 11520 imbalance 1.000000 median 0.010074 iqr 0.000416
multi 1.00 slab-110592 cut 11520 imbalance 1.000000 median 0.061870 iqr 0.001995
scotch 1.00 slab-110592 cut 16596 imbalance 1.000000 median 0.143885 iqr 0.001062
morton 1.01 slab-110592 cut 18819 imbalance 1.000217 median 0.039305 iqr 0.000350
hilbert 1.01 slab-110592 cut 18188 imbalance 1.000145 median 0.004590 iqr 0.000067
rcb 1.01 slab-110592 cut 12672 imbalance 1.000000 median 0.002837 iqr 0.000315
hilbert-fm 1.01 slab-110592 cut 18078 imbalance 1.009404 median 0.043202 iqr 0.000211
rcb-fm 1.01 slab-110592 cut 11520 imbalance 1.000000 median 0.010556 iqr 0.000200
multi 1.01 slab-110592 cut 11520 imbalance 1.000000 median 0.063673 iqr 0.001191
scotch 1.01 slab-110592 cut 11839 imbalance 1.009549 median 0.115865 iqr 0.002277
morton 1.02 slab-110592 cut 18819 imbalance 1.000217 median 0.039313 iqr 0.000439
hilbert 1.02 slab-110592 cut 18188 imbalance 1.000145 median 0.004575 iqr 0.000125
rcb 1.02 slab-110592 cut 12672 imbalance 1.000000 median 0.002936 iqr 0.000092
hilbert-fm 1.02 slab-110592 cut 18065 imbalance 1.019748 median 0.056232 iqr 0.000240
rcb-fm 1.02 slab-110592 cut 11520 imbalance 1.000000 median 0.010592 iqr 0.000459
multi 1.02 slab-110592 cut 11520 imbalance 1.000000 median 0.063623 iqr 0.001134
scotch 1.02 slab-110592 cut 12384 imbalance 1.018953 median 0.118643 iqr 0.001295
morton 1.05 slab-110592 cut 18819 imbalance 1.000217 median 0.039280 iqr 0.000182
hilbert 1.05 slab-110592 cut 18188 imbalance 1.000145 median 0.004608 iqr 0.000055
rcb 1.05 slab-110592 cut 12672 imbalance 1.000000 median 0.002960 iqr 0.000258
hilbert-fm 1.05 slab-110592 cut 18012 imbalance 1.049986 median 0.098341 iqr 0.000986
rcb-fm 1.05 slab-110592 cut 11520 imbalance 1.000000 median 0.010549 iqr 0.000241
multi 1.05 slab-110592 cut 11520 imbalance 1.000000 median 0.063966 iqr 0.001087
scotch 1.05 slab-110592 cut 11911 imbalance 1.049986 median 0.117861 iqr 0.001661
morton 1.10 slab-110592 cut 18819 imbalance 1.000217 median 0.039758 iqr 0.001029
hilbert 1.10 slab-110592 cut 18188 imbalance 1.000145 median 0.004604 iqr 0.000111
rcb 1.10 slab-110592 cut 12672 imbalance 1.000000 median 0.002910 iqr 0.000130
hilbert-fm 1.10 slab-110592 cut 18008 imbalance 1.053168 median 0.104027 iqr 0.005567
rcb-fm 1.10 slab-110592 cut 11520 imbalance 1.000000 median 0.010700 iqr 0.000260
multi 1.10 slab-110592 cut 11520 imbalance 1.000000 median 0.065149 iqr 0.000873
scotch 1.10 slab-110592 cut 11938 imbalance 1.094184 median 0.118511 iqr 0.002298
```

## Scaling

Strong scaling keeps the lattice at 1,520,875 points (115 cubed).
Weak scaling uses 190,109 points times the thread count, so eight threads is 1,520,872 points.
Median and interquartile range (IQR) are recomputed from the raw line.
`profile_ns` sums the critical-path wall time of select, boundary refinement, coarsening, packing, the balance repair, and the graph check over the repetitions, in nanoseconds.

```text
kind threads n method median iqr
strong 1 1520875 rcb 0.024760 0.001213 9f5cf54822c49335
strong 1 1520875 rcb-fm 0.160004 0.009075 29bd56d7602a754c
strong 1 1520875 multi 0.489242 0.006055 792dbf6957d1b8c5
weak 1 190109 rcb 0.005869 0.000387 a6ef469b3dc6d7e9
weak 1 190109 rcb-fm 0.022639 0.000641 a6ef469b3dc6d7e9
weak 1 190109 multi 0.071344 0.003001 a6ef469b3dc6d7e9
strong 2 1520875 rcb 0.018562 0.000214 9f5cf54822c49335
strong 2 1520875 rcb-fm 0.100907 0.002329 29bd56d7602a754c
strong 2 1520875 multi 0.415890 0.007444 792dbf6957d1b8c5
weak 2 380218 rcb 0.002613 0.000167 263ada9bf59665cd
weak 2 380218 rcb-fm 0.009062 0.000524 263ada9bf59665cd
weak 2 380218 multi 0.039105 0.000982 263ada9bf59665cd
strong 4 1520875 rcb 0.017892 0.000670 9f5cf54822c49335
strong 4 1520875 rcb-fm 0.091477 0.009419 29bd56d7602a754c
strong 4 1520875 multi 0.407587 0.019089 792dbf6957d1b8c5
weak 4 760436 rcb 0.005002 0.000248 c50aded0e71cdd51
weak 4 760436 rcb-fm 0.019047 0.001529 f6b34d1d05faa473
weak 4 760436 multi 0.074408 0.002798 f6b34d1d05faa473
strong 8 1520875 rcb 0.017751 0.000538 9f5cf54822c49335
strong 8 1520875 rcb-fm 0.086000 0.004628 29bd56d7602a754c
strong 8 1520875 multi 0.392307 0.010965 792dbf6957d1b8c5
weak 8 1520872 rcb 0.012813 0.000409 eef7430cb40dfd70
weak 8 1520872 rcb-fm 0.054243 0.002661 ac62a33a608f3394
weak 8 1520872 multi 0.290923 0.008622 ac62a33a608f3394
```

## Amdahl

The clocks are select, boundary refinement, coarsening, packing, the balance repair, and the graph check.
Each figure is wall time.
At one thread the two sides of a join are added.
Above one thread the slower side is kept.
A share is that summed wall time divided by the mean partition wall time times the repetition count.
The other share is what remains.
Karp-Flatt uses the strong rows.
With speedup s and thread count p, the serial fraction is (1/s - 1/p) / (1 - 1/p).

```text
method kind threads n select fm coarse pack repair check other
strong-rcb threads 1 n 1520875 select 0.382 fm 0.000 coarse 0.000 pack 0.537 repair 0.000 check 0.081 other 0.000
strong-rcb-fm threads 1 n 1520875 select 0.491 fm 0.390 coarse 0.000 pack 0.076 repair 0.031 check 0.012 other 0.000
strong-multi threads 1 n 1520875 select 0.159 fm 0.186 coarse 0.586 pack 0.054 repair 0.011 check 0.004 other 0.000
weak-rcb threads 1 n 190109 select 0.296 fm 0.000 coarse 0.000 pack 0.624 repair 0.000 check 0.080 other 0.000
weak-rcb-fm threads 1 n 190109 select 0.472 fm 0.312 coarse 0.000 pack 0.163 repair 0.033 check 0.020 other 0.000
weak-multi threads 1 n 190109 select 0.149 fm 0.120 coarse 0.618 pack 0.096 repair 0.011 check 0.007 other 0.000
strong-rcb threads 2 n 1520875 select 0.297 fm 0.000 coarse 0.000 pack 0.592 repair 0.000 check 0.112 other 0.000
strong-rcb-fm threads 2 n 1520875 select 0.471 fm 0.379 coarse 0.000 pack 0.082 repair 0.047 check 0.020 other 0.000
strong-multi threads 2 n 1520875 select 0.113 fm 0.138 coarse 0.680 pack 0.052 repair 0.012 check 0.005 other 0.000
weak-rcb threads 2 n 380218 select 0.233 fm 0.000 coarse 0.000 pack 0.671 repair 0.000 check 0.096 other 0.000
weak-rcb-fm threads 2 n 380218 select 0.359 fm 0.321 coarse 0.000 pack 0.253 repair 0.039 check 0.028 other 0.000
weak-multi threads 2 n 380218 select 0.085 fm 0.117 coarse 0.683 pack 0.099 repair 0.010 check 0.007 other 0.000
strong-rcb threads 4 n 1520875 select 0.312 fm 0.000 coarse 0.000 pack 0.562 repair 0.000 check 0.127 other 0.000
strong-rcb-fm threads 4 n 1520875 select 0.550 fm 0.284 coarse 0.000 pack 0.093 repair 0.050 check 0.023 other 0.000
strong-multi threads 4 n 1520875 select 0.123 fm 0.103 coarse 0.700 pack 0.056 repair 0.013 check 0.005 other 0.000
weak-rcb threads 4 n 760436 select 0.197 fm 0.000 coarse 0.000 pack 0.677 repair 0.000 check 0.126 other 0.000
weak-rcb-fm threads 4 n 760436 select 0.384 fm 0.362 coarse 0.000 pack 0.183 repair 0.040 check 0.032 other 0.000
weak-multi threads 4 n 760436 select 0.092 fm 0.131 coarse 0.664 pack 0.093 repair 0.011 check 0.008 other 0.000
strong-rcb threads 8 n 1520875 select 0.305 fm 0.000 coarse 0.000 pack 0.568 repair 0.000 check 0.127 other 0.000
strong-rcb-fm threads 8 n 1520875 select 0.573 fm 0.247 coarse 0.000 pack 0.099 repair 0.056 check 0.025 other 0.000
strong-multi threads 8 n 1520875 select 0.122 fm 0.082 coarse 0.721 pack 0.057 repair 0.013 check 0.005 other 0.000
weak-rcb threads 8 n 1520872 select 0.196 fm 0.000 coarse 0.000 pack 0.647 repair 0.000 check 0.157 other 0.000
weak-rcb-fm threads 8 n 1520872 select 0.490 fm 0.285 coarse 0.000 pack 0.137 repair 0.054 check 0.034 other 0.000
weak-multi threads 8 n 1520872 select 0.079 fm 0.068 coarse 0.777 pack 0.059 repair 0.011 check 0.006 other 0.000
```

```text
strong method threads select_e fm_e coarse_e pack_e repair_e check_e
strong multi threads 2 select_e 0.212 fm_e 0.270 coarse_e 0.983 pack_e 0.665 repair_e 0.782 check_e 0.951
strong multi threads 4 select_e 0.530 fm_e 0.287 coarse_e 0.999 pack_e 0.826 repair_e 0.962 check_e 0.982
strong multi threads 8 select_e 0.563 fm_e 0.260 coarse_e 0.987 pack_e 0.831 repair_e 0.940 check_e 0.993
strong rcb threads 2 select_e 0.129 fm_e na coarse_e na pack_e 0.606 repair_e na check_e 1.016
strong rcb threads 4 select_e 0.436 fm_e na coarse_e na pack_e 0.655 repair_e na check_e 1.154
strong rcb threads 8 select_e 0.492 fm_e na coarse_e na pack_e 0.700 repair_e na check_e 1.115
strong rcb-fm threads 2 select_e 0.159 fm_e 0.176 coarse_e na pack_e 0.312 repair_e 0.812 check_e 1.035
strong rcb-fm threads 4 select_e 0.520 fm_e 0.223 coarse_e na pack_e 0.599 repair_e 0.883 check_e 1.118
strong rcb-fm threads 8 select_e 0.553 fm_e 0.236 coarse_e na pack_e 0.638 repair_e 0.930 check_e 1.075
```

The measured limiter for rcb-fm at eight threads is the weighted median, with Karp-Flatt 0.553.
That rank uses serial time, Karp-Flatt times the one-thread phase wall.
The first cut stays on one thread.
The two halves already run in parallel after that cut.
Packing, the balance repair, and the graph check are smaller.
The graph check stays near its one-thread time, and that walk can run in parallel.

## Prose

The prose pass covers the placement note, the report, the partition page, the C API page, and the README.
Two older pages keep their earlier wording:

```text
docs/explanation/mic-and-cells.md
docs/reference/algorithm.md
```

