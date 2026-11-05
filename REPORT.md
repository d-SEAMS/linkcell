# Cell partition

The cell graph, the partition, and the geometric halo live in linkcell.
Ring membership, the Message Passing Interface (MPI) share, and ice clustering stay in seams-core.
eOn calls `pairs_within` only. rgpot builds pairs through vesin. readcon-core reads configuration (CON) frames.

This cell-partition report is generated from one log.
Every median and every interquartile range (IQR) is the value that log printed for that cell.
The raw sample line is the series those values came from, at type-7.

The log names the hardware as Intel Xeon Processor, 8 physical cores, one thread per core, one node, 2400 MHz.
A second node is recorded as `multi_node HOST NOT-RUN`.
Commit `d6dd05ed27b0f574b7705ab34ce8264193e152f1`.
Tree `f8d4b3e86a9cde8d260b56530c4eb414a249f313`.
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
Parallel Scotch uses that seed on 8 ranks.
The twelve cuts are logged because the runs still differ.

## Head to head

Times are seconds.
The pair is the median and the interquartile range (IQR) of the raw samples.
The halo fraction is the largest ghost-cell count divided by the owned-cell count.
A value above 1 means the ghost set is larger than the owned set.

```text
method system cut imbalance halo median iqr eligible
morton mW-110592 20894 1.000145 0.941354 0.018995 0.000827 1
hilbert mW-110592 20694 1.000145 0.943571 0.002140 0.000070 1
rcb mW-110592 13824 1.000000 0.396092 0.002388 0.000140 1
hilbert-fm mW-110592 20594 1.000579 0.942531 0.011277 0.000357 1
rcb-fm mW-110592 13824 1.000000 0.396092 0.008420 0.000131 1
multi mW-110592 13824 1.000000 0.396092 0.026935 0.000886 1
scotch mW-110592 15547 1.000217 0.478564 0.117244 0.004370 0
zoltan-rcb mW-110592 13824 1.000000 0.396092 0.003634 0.000144 0
ptscotch mW-110592 14272 1.000000 0.434368 13.688413 0.267505 0
morton mW-884736 83857 1.000009 0.397780 0.180414 0.000671 1
hilbert mW-884736 81032 1.000018 0.418792 0.020892 0.000384 1
rcb mW-884736 55990 1.000009 0.186900 0.017591 0.001042 1
hilbert-fm mW-884736 80931 1.000072 0.418751 0.068880 0.001303 1
rcb-fm mW-884736 55802 1.000054 0.183404 0.129499 0.007821 1
multi mW-884736 55800 1.000072 0.183409 0.843473 0.020993 1
scotch mW-884736 64754 1.000054 0.232028 0.678933 0.014212 0
zoltan-rcb mW-884736 55983 1.000009 0.186892 0.029449 0.001242 0
ptscotch mW-884736 68795 1.000009 0.251396 55.126436 0.713981 0
morton TIP4P-128000 2804168 1.000000 1.687500 0.000768 0.000020 1
hilbert TIP4P-128000 2791850 1.000687 1.661710 0.000100 0.000006 1
rcb TIP4P-128000 2492438 1.000625 1.353383 0.000395 0.000062 1
hilbert-fm TIP4P-128000 2671517 1.005625 1.500000 0.001763 0.000064 1
rcb-fm TIP4P-128000 2145600 1.000000 1.000000 0.002365 0.000153 1
multi TIP4P-128000 2145600 1.000000 1.000000 0.005152 0.000164 1
scotch TIP4P-128000 2169032 1.005375 1.052632 0.015219 0.001042 0
zoltan-rcb TIP4P-128000 2485058 1.000625 1.353383 0.000250 0.000033 0
ptscotch TIP4P-128000 2130992 1.001250 1.007692 3.460620 0.175503 0
morton TIP4P-1024000 13084202 1.000320 0.812144 0.007761 0.000665 1
hilbert TIP4P-1024000 12710308 1.000195 0.848197 0.001116 0.000143 1
rcb TIP4P-1024000 9965162 1.000266 0.576619 0.001331 0.000246 1
hilbert-fm TIP4P-1024000 12192983 1.000758 0.834594 0.025063 0.001869 1
rcb-fm TIP4P-1024000 8793439 1.000766 0.466543 0.012951 0.001100 1
multi TIP4P-1024000 8800880 1.000766 0.466263 0.039471 0.003192 1
scotch TIP4P-1024000 8911827 1.000719 0.490884 0.104243 0.008197 0
zoltan-rcb TIP4P-1024000 9897435 1.000312 0.564509 0.001721 0.000183 0
ptscotch TIP4P-1024000 8799875 1.000125 0.484201 4.989212 0.140390 0
morton slab-110592 18819 1.000217 0.750051 0.041164 0.004790 1
hilbert slab-110592 18188 1.000145 0.747628 0.004880 0.000454 1
rcb slab-110592 12672 1.000000 0.500000 0.004787 0.000786 1
hilbert-fm slab-110592 18148 1.000579 0.746570 0.010305 0.000818 1
rcb-fm slab-110592 11520 1.000000 0.396092 0.014391 0.001261 1
multi slab-110592 11520 1.000000 0.396092 0.070792 0.005193 1
scotch slab-110592 12553 1.000579 0.531308 0.150262 0.008118 0
zoltan-rcb slab-110592 12672 1.000000 0.500000 0.007780 0.000260 0
ptscotch slab-110592 11803 1.000000 2.165770 24.345488 0.544624 0
```

## Parallel Scotch by system

Seed 1 and 8 ranks.
The median cut is type-7 over the twelve logged cuts.
The minimum and the maximum are the observed range.

```text
system median min max
mW-110592 14272.0000 13924 14551
mW-884736 68794.5000 65424 72012
TIP4P-128000 2130992.0000 2130992 2165484
TIP4P-1024000 8799875.0000 8679032 8962235
slab-110592 11802.5000 11520 12138
```

```text
mW-110592 14254
mW-110592 14386
mW-110592 14319
mW-110592 14290
mW-110592 14457
mW-110592 13948
mW-110592 14035
mW-110592 13954
mW-110592 13924
mW-110592 14315
mW-110592 14016
mW-110592 14551
mW-884736 67210
mW-884736 70379
mW-884736 68339
mW-884736 70060
mW-884736 67940
mW-884736 67009
mW-884736 65424
mW-884736 69250
mW-884736 65736
mW-884736 71057
mW-884736 72012
mW-884736 70754
TIP4P-128000 2130992
TIP4P-128000 2130992
TIP4P-128000 2130992
TIP4P-128000 2130992
TIP4P-128000 2130992
TIP4P-128000 2130992
TIP4P-128000 2130992
TIP4P-128000 2130992
TIP4P-128000 2130992
TIP4P-128000 2130992
TIP4P-128000 2165484
TIP4P-128000 2130992
TIP4P-1024000 8776038
TIP4P-1024000 8962235
TIP4P-1024000 8873880
TIP4P-1024000 8679032
TIP4P-1024000 8798211
TIP4P-1024000 8827956
TIP4P-1024000 8838507
TIP4P-1024000 8724773
TIP4P-1024000 8696519
TIP4P-1024000 8801539
TIP4P-1024000 8751189
TIP4P-1024000 8864580
slab-110592 11664
slab-110592 11905
slab-110592 12138
slab-110592 11839
slab-110592 12014
slab-110592 11520
slab-110592 11616
slab-110592 11760
slab-110592 11904
slab-110592 12004
slab-110592 11766
slab-110592 11735
```

Parallel Scotch cuts fewer edges on TIP4P-128000.
The mW lattices, the slab, and TIP4P-1024000 keep a smaller rcb-fm cut.
On TIP4P-1024000 the parallel Scotch range crosses that rcb-fm cut, and the type-7 median stays above it.
Wall time favors rcb-fm on the slab, both TIP4P lattices, and both mW lattices.
The comparison stays on one system at a time.

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

Each sum adds the edge cut of one native method over the five systems in this log.
Parallel Scotch stays in the system-by-system comparison above.

```text
sum method morton cut 16011940 median_s 0.249101 eligible 1
sum method hilbert cut 15622072 median_s 0.029127 eligible 1
sum method rcb cut 12540086 median_s 0.026492 eligible 1
sum method hilbert-fm cut 14984173 median_s 0.117289 eligible 1
sum method rcb-fm cut 11020185 median_s 0.167627 eligible 1
sum method multi cut 11027624 median_s 0.985823 eligible 1
default_method rcb-fm
```

## Raw samples

Each line is the twelve wall times, in seconds, in the order they were measured.

```text
morton mW-110592 0.018835 0.028358 0.018890 0.018836 0.018722 0.018864 0.018996 0.019290 0.020352 0.019461 0.018993 0.030327
hilbert mW-110592 0.002102 0.002255 0.002120 0.002228 0.002106 0.002110 0.002119 0.002172 0.002316 0.002162 0.002134 0.002145
rcb mW-110592 0.002603 0.002415 0.002334 0.002356 0.002390 0.002296 0.002727 0.002457 0.002386 0.002320 0.002262 0.002509
hilbert-fm mW-110592 0.011083 0.012379 0.011461 0.011237 0.011180 0.011161 0.011160 0.011750 0.013830 0.011416 0.011291 0.011263
rcb-fm mW-110592 0.009084 0.009281 0.008450 0.008365 0.008343 0.008417 0.008125 0.008502 0.008261 0.008438 0.008300 0.008424
multi mW-110592 0.030791 0.029278 0.026837 0.026244 0.026107 0.026560 0.026672 0.027033 0.027292 0.027137 0.026499 0.027846
scotch mW-110592 0.119779 0.116390 0.117359 0.115366 0.115074 0.117129 0.115496 0.119997 0.122252 0.118630 0.114707 0.128482
zoltan-rcb mW-110592 0.003615 0.003930 0.003652 0.003606 0.003552 0.003455 0.003679 0.003736 0.003675 0.003750 0.003465 0.003540
ptscotch mW-110592 13.545537 13.801759 13.719692 13.593227 13.567377 13.657134 14.024420 13.799715 14.065488 13.458635 13.912411 13.517040
morton mW-884736 0.180440 0.185049 0.180407 0.179816 0.180223 0.180708 0.180963 0.180420 0.179540 0.179347 0.183554 0.180195
hilbert mW-884736 0.020960 0.020865 0.020456 0.020919 0.021692 0.020920 0.020700 0.020717 0.020579 0.020537 0.030272 0.021339
rcb mW-884736 0.016992 0.018245 0.017059 0.018010 0.018310 0.022331 0.017244 0.017289 0.017416 0.017145 0.021019 0.017766
hilbert-fm mW-884736 0.069831 0.068507 0.069373 0.069147 0.070025 0.066875 0.068527 0.067217 0.069279 0.067176 0.082502 0.068614
rcb-fm mW-884736 0.136957 0.129491 0.129110 0.129457 0.127614 0.153178 0.128628 0.128615 0.129507 0.129515 0.154310 0.136762
multi mW-884736 0.868826 0.855630 0.849114 0.831375 0.839997 0.845925 0.841021 0.840906 0.826411 0.831751 2.051820 0.871460
scotch mW-884736 0.684711 0.691216 0.669145 0.664444 0.682932 0.693967 0.669682 0.672940 0.678549 0.674685 0.692414 0.679317
zoltan-rcb mW-884736 0.026957 0.034205 0.029139 0.029179 0.029446 0.029453 0.029134 0.030129 0.029501 0.029293 0.031258 0.032043
ptscotch mW-884736 55.047683 54.579486 54.850036 54.746534 55.205189 55.494820 56.152950 55.705510 54.385623 55.520387 55.591405 55.030737
morton TIP4P-128000 0.000765 0.000799 0.000777 0.000768 0.000762 0.000759 0.000829 0.000763 0.000757 0.000777 0.000768 0.000853
hilbert TIP4P-128000 0.000099 0.000099 0.000099 0.000099 0.000109 0.000100 0.000104 0.000116 0.000100 0.000100 0.000097 0.000109
rcb TIP4P-128000 0.000393 0.000393 0.000638 0.000383 0.000246 0.000397 0.000499 0.000438 0.000380 0.000355 0.000433 0.000462
hilbert-fm TIP4P-128000 0.001699 0.001731 0.001769 0.001876 0.001797 0.001746 0.001724 0.001943 0.001767 0.001760 0.001736 0.001804
rcb-fm TIP4P-128000 0.002262 0.002157 0.002389 0.002359 0.002227 0.002370 0.002223 0.002411 0.002812 0.002503 0.002405 0.002290
multi TIP4P-128000 0.005074 0.005305 0.005106 0.006391 0.005127 0.005200 0.005050 0.005178 0.005851 0.004901 0.005039 0.005207
scotch TIP4P-128000 0.014583 0.014942 0.015891 0.016291 0.014569 0.015716 0.016123 0.014764 0.015117 0.015322 0.015839 0.014825
zoltan-rcb TIP4P-128000 0.000270 0.000296 0.000319 0.000228 0.000224 0.000243 0.000236 0.000251 0.000275 0.000266 0.000240 0.000250
ptscotch TIP4P-128000 3.392735 3.568821 3.410611 3.305808 3.422271 3.733829 3.498968 3.566814 3.366750 3.621619 3.389046 3.506609
morton TIP4P-1024000 0.007707 0.007615 0.007750 0.007726 0.007629 0.008677 0.007579 0.009970 0.008245 0.009016 0.008098 0.007772
hilbert TIP4P-1024000 0.001066 0.001027 0.001127 0.001050 0.001026 0.001231 0.001029 0.001116 0.001235 0.001255 0.001173 0.001115
rcb TIP4P-1024000 0.001259 0.001248 0.001187 0.001314 0.001216 0.001584 0.001370 0.001453 0.001806 0.001627 0.001347 0.001127
hilbert-fm TIP4P-1024000 0.023784 0.024610 0.024269 0.024680 0.024212 0.030469 0.024559 0.025446 0.025824 0.027952 0.027964 0.025762
rcb-fm TIP4P-1024000 0.012576 0.012985 0.012324 0.012331 0.012461 0.015100 0.012686 0.012918 0.013432 0.018080 0.014290 0.012990
multi TIP4P-1024000 0.037523 0.037770 0.037308 0.039940 0.037378 0.043674 0.037665 0.040303 0.040722 0.047700 0.041120 0.039003
scotch TIP4P-1024000 0.100737 0.102011 0.098604 0.101008 0.101480 0.145435 0.104760 0.108382 0.110033 0.115075 0.109401 0.103725
zoltan-rcb TIP4P-1024000 0.001753 0.001474 0.001764 0.001504 0.001585 0.001771 0.001534 0.001719 0.001866 0.001723 0.001747 0.001702
ptscotch TIP4P-1024000 4.836168 4.870440 4.986719 5.096560 5.106773 5.085419 5.000744 4.991705 5.202748 4.979189 4.973606 4.840994
morton slab-110592 0.038951 0.039052 0.044058 0.044445 0.074811 0.041973 0.044034 0.040085 0.038756 0.039316 0.040410 0.041917
hilbert slab-110592 0.004615 0.004620 0.005018 0.006253 0.007372 0.005156 0.005129 0.004677 0.004700 0.004746 0.004684 0.005013
rcb slab-110592 0.004255 0.004436 0.005165 0.004795 0.005674 0.005319 0.004945 0.004780 0.004401 0.005361 0.004422 0.004379
hilbert-fm slab-110592 0.009690 0.009649 0.010382 0.010513 0.014588 0.011525 0.010957 0.010228 0.009845 0.010026 0.009658 0.010495
rcb-fm slab-110592 0.013740 0.013730 0.015436 0.014328 0.016435 0.014454 0.014772 0.014557 0.013308 0.017432 0.013500 0.013515
multi slab-110592 0.066646 0.067136 0.072033 0.079866 0.086490 0.071646 0.075091 0.070011 0.066494 0.071573 0.067761 0.068961
scotch slab-110592 0.143742 0.143576 0.152831 0.154198 0.160242 0.149675 0.158242 0.150848 0.143699 0.151074 0.145492 0.148832
zoltan-rcb slab-110592 0.007797 0.007519 0.008223 0.007773 0.007867 0.008010 0.008295 0.007584 0.007611 0.007654 0.007663 0.007787
ptscotch slab-110592 23.996897 24.624170 24.070748 24.587822 23.909960 24.272395 25.173991 24.944617 24.074364 24.573650 23.970779 24.418582
```

## Balance sweep

The same five ratios appear on every system in this sweep.
The main table uses the derived bound.

```text
morton 1.00 mW-110592 cut 20894 imbalance 1.000145 median 0.018722 iqr 0.000179
hilbert 1.00 mW-110592 cut 20694 imbalance 1.000145 median 0.002021 iqr 0.000075
rcb 1.00 mW-110592 cut 13824 imbalance 1.000000 median 0.002231 iqr 0.000184
hilbert-fm 1.00 mW-110592 cut 20692 imbalance 1.000000 median 0.005187 iqr 0.000131
rcb-fm 1.00 mW-110592 cut 13824 imbalance 1.000000 median 0.007701 iqr 0.000237
multi 1.00 mW-110592 cut 13824 imbalance 1.000000 median 0.025291 iqr 0.000428
scotch 1.00 mW-110592 cut 20801 imbalance 1.000000 median 0.112682 iqr 0.001116
morton 1.01 mW-110592 cut 20894 imbalance 1.000145 median 0.018699 iqr 0.000313
hilbert 1.01 mW-110592 cut 20694 imbalance 1.000145 median 0.002065 iqr 0.000099
rcb 1.01 mW-110592 cut 13824 imbalance 1.000000 median 0.002245 iqr 0.000223
hilbert-fm 1.01 mW-110592 cut 20584 imbalance 1.002604 median 0.005176 iqr 0.000148
rcb-fm 1.01 mW-110592 cut 13824 imbalance 1.000000 median 0.008227 iqr 0.000185
multi 1.01 mW-110592 cut 13824 imbalance 1.000000 median 0.026533 iqr 0.000520
scotch 1.01 mW-110592 cut 14698 imbalance 1.009910 median 0.113514 iqr 0.000760
morton 1.02 mW-110592 cut 20894 imbalance 1.000145 median 0.018630 iqr 0.000090
hilbert 1.02 mW-110592 cut 20694 imbalance 1.000145 median 0.002013 iqr 0.000056
rcb 1.02 mW-110592 cut 13824 imbalance 1.000000 median 0.002250 iqr 0.000186
hilbert-fm 1.02 mW-110592 cut 20584 imbalance 1.002604 median 0.005136 iqr 0.000107
rcb-fm 1.02 mW-110592 cut 13824 imbalance 1.000000 median 0.008223 iqr 0.000148
multi 1.02 mW-110592 cut 13824 imbalance 1.000000 median 0.026595 iqr 0.000186
scotch 1.02 mW-110592 cut 14391 imbalance 1.019676 median 0.098953 iqr 0.000808
morton 1.05 mW-110592 cut 20894 imbalance 1.000145 median 0.018646 iqr 0.000078
hilbert 1.05 mW-110592 cut 20694 imbalance 1.000145 median 0.001998 iqr 0.000042
rcb 1.05 mW-110592 cut 13824 imbalance 1.000000 median 0.002234 iqr 0.000127
hilbert-fm 1.05 mW-110592 cut 20584 imbalance 1.002604 median 0.005137 iqr 0.000091
rcb-fm 1.05 mW-110592 cut 13824 imbalance 1.000000 median 0.008245 iqr 0.000209
multi 1.05 mW-110592 cut 13824 imbalance 1.000000 median 0.026902 iqr 0.000208
scotch 1.05 mW-110592 cut 14817 imbalance 1.045139 median 0.113234 iqr 0.001111
morton 1.10 mW-110592 cut 20894 imbalance 1.000145 median 0.018817 iqr 0.000341
hilbert 1.10 mW-110592 cut 20694 imbalance 1.000145 median 0.002018 iqr 0.000045
rcb 1.10 mW-110592 cut 13824 imbalance 1.000000 median 0.002134 iqr 0.000170
hilbert-fm 1.10 mW-110592 cut 20584 imbalance 1.002604 median 0.005153 iqr 0.000139
rcb-fm 1.10 mW-110592 cut 13824 imbalance 1.000000 median 0.008145 iqr 0.000117
multi 1.10 mW-110592 cut 13824 imbalance 1.000000 median 0.027050 iqr 0.000330
scotch 1.10 mW-110592 cut 15098 imbalance 1.099826 median 0.115227 iqr 0.000380
morton 1.00 mW-884736 cut 83857 imbalance 1.000009 median 0.178171 iqr 0.001005
hilbert 1.00 mW-884736 cut 81032 imbalance 1.000018 median 0.020502 iqr 0.000233
rcb 1.00 mW-884736 cut 55990 imbalance 1.000009 median 0.016645 iqr 0.000758
hilbert-fm 1.00 mW-884736 cut 81032 imbalance 1.000000 median 0.039185 iqr 0.000243
rcb-fm 1.00 mW-884736 cut 55990 imbalance 1.000009 median 0.093888 iqr 0.001200
multi 1.00 mW-884736 cut 55990 imbalance 1.000009 median 0.268425 iqr 0.005823
scotch 1.00 mW-884736 cut 81445 imbalance 1.000000 median 0.628489 iqr 0.005007
morton 1.01 mW-884736 cut 83857 imbalance 1.000009 median 0.181351 iqr 0.004880
hilbert 1.01 mW-884736 cut 81032 imbalance 1.000018 median 0.021025 iqr 0.000469
rcb 1.01 mW-884736 cut 55990 imbalance 1.000009 median 0.017379 iqr 0.001515
hilbert-fm 1.01 mW-884736 cut 80799 imbalance 1.004792 median 0.230571 iqr 0.005196
rcb-fm 1.01 mW-884736 cut 55894 imbalance 1.000868 median 0.077742 iqr 0.001381
multi 1.01 mW-884736 cut 55837 imbalance 1.000922 median 0.474814 iqr 0.034794
scotch 1.01 mW-884736 cut 58097 imbalance 1.009865 median 0.595584 iqr 0.018882
morton 1.02 mW-884736 cut 83857 imbalance 1.000009 median 0.180975 iqr 0.001414
hilbert 1.02 mW-884736 cut 81032 imbalance 1.000018 median 0.021015 iqr 0.000254
rcb 1.02 mW-884736 cut 55990 imbalance 1.000009 median 0.017780 iqr 0.001051
hilbert-fm 1.02 mW-884736 cut 80799 imbalance 1.004792 median 0.232941 iqr 0.003562
rcb-fm 1.02 mW-884736 cut 55894 imbalance 1.000868 median 0.079355 iqr 0.004143
multi 1.02 mW-884736 cut 55837 imbalance 1.000922 median 0.477482 iqr 0.016444
scotch 1.02 mW-884736 cut 60287 imbalance 1.019938 median 0.621879 iqr 0.024732
morton 1.05 mW-884736 cut 83857 imbalance 1.000009 median 0.179127 iqr 0.000586
hilbert 1.05 mW-884736 cut 81032 imbalance 1.000018 median 0.020660 iqr 0.000169
rcb 1.05 mW-884736 cut 55990 imbalance 1.000009 median 0.016473 iqr 0.000590
hilbert-fm 1.05 mW-884736 cut 80799 imbalance 1.004792 median 0.230637 iqr 0.003139
rcb-fm 1.05 mW-884736 cut 55894 imbalance 1.000868 median 0.076017 iqr 0.001124
multi 1.05 mW-884736 cut 55837 imbalance 1.000922 median 0.454466 iqr 0.005050
scotch 1.05 mW-884736 cut 62348 imbalance 1.018971 median 0.597613 iqr 0.011022
morton 1.10 mW-884736 cut 83857 imbalance 1.000009 median 0.178739 iqr 0.000628
hilbert 1.10 mW-884736 cut 81032 imbalance 1.000018 median 0.020745 iqr 0.000203
rcb 1.10 mW-884736 cut 55990 imbalance 1.000009 median 0.016918 iqr 0.000770
hilbert-fm 1.10 mW-884736 cut 80799 imbalance 1.004792 median 0.229864 iqr 0.005836
rcb-fm 1.10 mW-884736 cut 55894 imbalance 1.000868 median 0.076634 iqr 0.002564
multi 1.10 mW-884736 cut 55837 imbalance 1.000922 median 0.456081 iqr 0.003419
scotch 1.10 mW-884736 cut 59853 imbalance 1.086842 median 0.606846 iqr 0.009690
morton 1.00 TIP4P-128000 cut 2804168 imbalance 1.000000 median 0.000756 iqr 0.000016
hilbert 1.00 TIP4P-128000 cut 2791850 imbalance 1.000687 median 0.000095 iqr 0.000002
rcb 1.00 TIP4P-128000 cut 2492438 imbalance 1.000625 median 0.000346 iqr 0.000151
hilbert-fm 1.00 TIP4P-128000 cut 2791850 imbalance 1.000687 median 0.000259 iqr 0.000021
rcb-fm 1.00 TIP4P-128000 cut 2145600 imbalance 1.000000 median 0.000884 iqr 0.000091
multi 1.00 TIP4P-128000 cut 2145600 imbalance 1.000000 median 0.003517 iqr 0.000214
scotch 1.00 TIP4P-128000 cut 2776979 imbalance 1.000687 median 0.008956 iqr 0.000538
morton 1.01 TIP4P-128000 cut 2804168 imbalance 1.000000 median 0.000769 iqr 0.000032
hilbert 1.01 TIP4P-128000 cut 2791850 imbalance 1.000687 median 0.000098 iqr 0.000003
rcb 1.01 TIP4P-128000 cut 2492438 imbalance 1.000625 median 0.000365 iqr 0.000050
hilbert-fm 1.01 TIP4P-128000 cut 2668242 imbalance 1.005250 median 0.002285 iqr 0.000061
rcb-fm 1.01 TIP4P-128000 cut 2130992 imbalance 1.001250 median 0.004759 iqr 0.000103
multi 1.01 TIP4P-128000 cut 2130992 imbalance 1.001250 median 0.008004 iqr 0.000180
scotch 1.01 TIP4P-128000 cut 2172264 imbalance 1.008563 median 0.015513 iqr 0.000507
morton 1.02 TIP4P-128000 cut 2804168 imbalance 1.000000 median 0.000761 iqr 0.000021
hilbert 1.02 TIP4P-128000 cut 2791850 imbalance 1.000687 median 0.000098 iqr 0.000002
rcb 1.02 TIP4P-128000 cut 2492438 imbalance 1.000625 median 0.000455 iqr 0.000177
hilbert-fm 1.02 TIP4P-128000 cut 2667911 imbalance 1.016563 median 0.001442 iqr 0.000031
rcb-fm 1.02 TIP4P-128000 cut 2145600 imbalance 1.000000 median 0.002824 iqr 0.000094
multi 1.02 TIP4P-128000 cut 2145600 imbalance 1.000000 median 0.006042 iqr 0.000147
scotch 1.02 TIP4P-128000 cut 2196133 imbalance 1.019062 median 0.014972 iqr 0.000652
morton 1.05 TIP4P-128000 cut 2804168 imbalance 1.000000 median 0.000755 iqr 0.000023
hilbert 1.05 TIP4P-128000 cut 2791850 imbalance 1.000687 median 0.000095 iqr 0.000003
rcb 1.05 TIP4P-128000 cut 2492438 imbalance 1.000625 median 0.000325 iqr 0.000053
hilbert-fm 1.05 TIP4P-128000 cut 2658326 imbalance 1.049063 median 0.001409 iqr 0.000123
rcb-fm 1.05 TIP4P-128000 cut 2120992 imbalance 1.050000 median 0.004542 iqr 0.000086
multi 1.05 TIP4P-128000 cut 2120512 imbalance 1.050000 median 0.016408 iqr 0.000297
scotch 1.05 TIP4P-128000 cut 2155060 imbalance 1.050000 median 0.013352 iqr 0.000828
morton 1.10 TIP4P-128000 cut 2804168 imbalance 1.000000 median 0.000763 iqr 0.000023
hilbert 1.10 TIP4P-128000 cut 2791850 imbalance 1.000687 median 0.000095 iqr 0.000002
rcb 1.10 TIP4P-128000 cut 2492438 imbalance 1.000625 median 0.000330 iqr 0.000081
hilbert-fm 1.10 TIP4P-128000 cut 2654022 imbalance 1.096750 median 0.001517 iqr 0.000115
rcb-fm 1.10 TIP4P-128000 cut 2092800 imbalance 1.062500 median 0.005062 iqr 0.000131
multi 1.10 TIP4P-128000 cut 2092800 imbalance 1.062500 median 0.008713 iqr 0.000177
scotch 1.10 TIP4P-128000 cut 2118900 imbalance 1.089063 median 0.014067 iqr 0.000962
morton 1.00 TIP4P-1024000 cut 13084202 imbalance 1.000320 median 0.007431 iqr 0.000142
hilbert 1.00 TIP4P-1024000 cut 12710308 imbalance 1.000195 median 0.001005 iqr 0.000106
rcb 1.00 TIP4P-1024000 cut 9965162 imbalance 1.000266 median 0.001150 iqr 0.000147
hilbert-fm 1.00 TIP4P-1024000 cut 12710308 imbalance 1.000195 median 0.001960 iqr 0.000075
rcb-fm 1.00 TIP4P-1024000 cut 9965162 imbalance 1.000266 median 0.006551 iqr 0.000291
multi 1.00 TIP4P-1024000 cut 9965162 imbalance 1.000266 median 0.030191 iqr 0.000402
scotch 1.00 TIP4P-1024000 cut 15041835 imbalance 1.000133 median 0.081965 iqr 0.001229
morton 1.01 TIP4P-1024000 cut 13084202 imbalance 1.000320 median 0.007600 iqr 0.000098
hilbert 1.01 TIP4P-1024000 cut 12710308 imbalance 1.000195 median 0.000964 iqr 0.000059
rcb 1.01 TIP4P-1024000 cut 9965162 imbalance 1.000266 median 0.001076 iqr 0.000185
hilbert-fm 1.01 TIP4P-1024000 cut 12126380 imbalance 1.010000 median 0.020668 iqr 0.000552
rcb-fm 1.01 TIP4P-1024000 cut 8741847 imbalance 1.009797 median 0.015870 iqr 0.000471
multi 1.01 TIP4P-1024000 cut 8762734 imbalance 1.009813 median 0.044171 iqr 0.000473
scotch 1.01 TIP4P-1024000 cut 8949479 imbalance 1.009922 median 0.111612 iqr 0.002795
morton 1.02 TIP4P-1024000 cut 13084202 imbalance 1.000320 median 0.007743 iqr 0.000364
hilbert 1.02 TIP4P-1024000 cut 12710308 imbalance 1.000195 median 0.000996 iqr 0.000058
rcb 1.02 TIP4P-1024000 cut 9965162 imbalance 1.000266 median 0.001178 iqr 0.000180
hilbert-fm 1.02 TIP4P-1024000 cut 12078105 imbalance 1.019977 median 0.007898 iqr 0.000167
rcb-fm 1.02 TIP4P-1024000 cut 8726026 imbalance 1.019969 median 0.015692 iqr 0.000499
multi 1.02 TIP4P-1024000 cut 8760713 imbalance 1.019969 median 0.045638 iqr 0.001556
scotch 1.02 TIP4P-1024000 cut 8861660 imbalance 1.020000 median 0.107546 iqr 0.001489
morton 1.05 TIP4P-1024000 cut 13084202 imbalance 1.000320 median 0.007531 iqr 0.000165
hilbert 1.05 TIP4P-1024000 cut 12710308 imbalance 1.000195 median 0.000951 iqr 0.000008
rcb 1.05 TIP4P-1024000 cut 9965162 imbalance 1.000266 median 0.001235 iqr 0.000095
hilbert-fm 1.05 TIP4P-1024000 cut 11893822 imbalance 1.050000 median 0.011841 iqr 0.000190
rcb-fm 1.05 TIP4P-1024000 cut 8743808 imbalance 1.050000 median 0.026935 iqr 0.000867
multi 1.05 TIP4P-1024000 cut 8746824 imbalance 1.050000 median 0.054213 iqr 0.001745
scotch 1.05 TIP4P-1024000 cut 8730300 imbalance 1.028125 median 0.095961 iqr 0.003250
morton 1.10 TIP4P-1024000 cut 13084202 imbalance 1.000320 median 0.007457 iqr 0.000084
hilbert 1.10 TIP4P-1024000 cut 12710308 imbalance 1.000195 median 0.000956 iqr 0.000013
rcb 1.10 TIP4P-1024000 cut 9965162 imbalance 1.000266 median 0.001181 iqr 0.000066
hilbert-fm 1.10 TIP4P-1024000 cut 11733490 imbalance 1.099828 median 0.014473 iqr 0.000194
rcb-fm 1.10 TIP4P-1024000 cut 8530800 imbalance 1.100000 median 0.014317 iqr 0.000131
multi 1.10 TIP4P-1024000 cut 8523300 imbalance 1.100000 median 0.069560 iqr 0.000269
scotch 1.10 TIP4P-1024000 cut 8812500 imbalance 1.058594 median 0.093348 iqr 0.001545
morton 1.00 slab-110592 cut 18819 imbalance 1.000217 median 0.041926 iqr 0.000934
hilbert 1.00 slab-110592 cut 18188 imbalance 1.000145 median 0.004932 iqr 0.000308
rcb 1.00 slab-110592 cut 12672 imbalance 1.000000 median 0.004314 iqr 0.000218
hilbert-fm 1.00 slab-110592 cut 18200 imbalance 1.000000 median 0.007345 iqr 0.000618
rcb-fm 1.00 slab-110592 cut 11520 imbalance 1.000000 median 0.013640 iqr 0.000991
multi 1.00 slab-110592 cut 11520 imbalance 1.000000 median 0.068485 iqr 0.005426
scotch 1.00 slab-110592 cut 16596 imbalance 1.000000 median 0.154578 iqr 0.006569
morton 1.01 slab-110592 cut 18819 imbalance 1.000217 median 0.041158 iqr 0.001065
hilbert 1.01 slab-110592 cut 18188 imbalance 1.000145 median 0.004858 iqr 0.000327
rcb 1.01 slab-110592 cut 12672 imbalance 1.000000 median 0.004435 iqr 0.000443
hilbert-fm 1.01 slab-110592 cut 18078 imbalance 1.009404 median 0.047334 iqr 0.002840
rcb-fm 1.01 slab-110592 cut 11520 imbalance 1.000000 median 0.014397 iqr 0.001048
multi 1.01 slab-110592 cut 11520 imbalance 1.000000 median 0.072911 iqr 0.003886
scotch 1.01 slab-110592 cut 11839 imbalance 1.009549 median 0.122680 iqr 0.004509
morton 1.02 slab-110592 cut 18819 imbalance 1.000217 median 0.040736 iqr 0.002770
hilbert 1.02 slab-110592 cut 18188 imbalance 1.000145 median 0.004919 iqr 0.000253
rcb 1.02 slab-110592 cut 12672 imbalance 1.000000 median 0.004339 iqr 0.000275
hilbert-fm 1.02 slab-110592 cut 18065 imbalance 1.019748 median 0.059716 iqr 0.004477
rcb-fm 1.02 slab-110592 cut 11520 imbalance 1.000000 median 0.014338 iqr 0.001203
multi 1.02 slab-110592 cut 11520 imbalance 1.000000 median 0.072625 iqr 0.005501
scotch 1.02 slab-110592 cut 12384 imbalance 1.018953 median 0.125855 iqr 0.005889
morton 1.05 slab-110592 cut 18819 imbalance 1.000217 median 0.042747 iqr 0.001598
hilbert 1.05 slab-110592 cut 18188 imbalance 1.000145 median 0.005047 iqr 0.000356
rcb 1.05 slab-110592 cut 12672 imbalance 1.000000 median 0.004588 iqr 0.000443
hilbert-fm 1.05 slab-110592 cut 18012 imbalance 1.049986 median 0.113837 iqr 0.011107
rcb-fm 1.05 slab-110592 cut 11520 imbalance 1.000000 median 0.015239 iqr 0.001233
multi 1.05 slab-110592 cut 11520 imbalance 1.000000 median 0.076673 iqr 0.002796
scotch 1.05 slab-110592 cut 11911 imbalance 1.049986 median 0.133948 iqr 0.008426
morton 1.10 slab-110592 cut 18819 imbalance 1.000217 median 0.039951 iqr 0.001295
hilbert 1.10 slab-110592 cut 18188 imbalance 1.000145 median 0.004761 iqr 0.000239
rcb 1.10 slab-110592 cut 12672 imbalance 1.000000 median 0.004353 iqr 0.000169
hilbert-fm 1.10 slab-110592 cut 18008 imbalance 1.053168 median 0.109125 iqr 0.002347
rcb-fm 1.10 slab-110592 cut 11520 imbalance 1.000000 median 0.014190 iqr 0.000362
multi 1.10 slab-110592 cut 11520 imbalance 1.000000 median 0.071507 iqr 0.001472
scotch 1.10 slab-110592 cut 11938 imbalance 1.094184 median 0.124098 iqr 0.001855
```

## Scaling

Strong scaling keeps the lattice at 1,520,875 points (115 cubed).
Weak scaling uses 190,109 points times the thread count, so eight threads is 1,520,872 points.
Median and interquartile range (IQR) are recomputed from the raw line.
`profile_ns` sums the critical-path wall time of select, boundary refinement, coarsening, packing, the balance repair, the graph check, and join scheduling over the repetitions, in nanoseconds.

```text
kind threads n method median iqr
strong 1 1520875 rcb 0.029177 0.001188 9f5cf54822c49335
strong 1 1520875 rcb-fm 0.143802 0.007999 29bd56d7602a754c
strong 1 1520875 multi 0.488806 0.006992 792dbf6957d1b8c5
weak 1 190109 rcb 0.009395 0.000281 a6ef469b3dc6d7e9
weak 1 190109 rcb-fm 0.033199 0.001310 a6ef469b3dc6d7e9
weak 1 190109 multi 0.085666 0.003034 a6ef469b3dc6d7e9
strong 2 1520875 rcb 0.021792 0.001409 9f5cf54822c49335
strong 2 1520875 rcb-fm 0.098176 0.001698 29bd56d7602a754c
strong 2 1520875 multi 0.441405 0.013739 792dbf6957d1b8c5
weak 2 380218 rcb 0.004023 0.000418 263ada9bf59665cd
weak 2 380218 rcb-fm 0.013579 0.000371 263ada9bf59665cd
weak 2 380218 multi 0.047704 0.001070 263ada9bf59665cd
strong 4 1520875 rcb 0.020888 0.000678 9f5cf54822c49335
strong 4 1520875 rcb-fm 0.086724 0.007112 29bd56d7602a754c
strong 4 1520875 multi 0.465560 0.063486 792dbf6957d1b8c5
weak 4 760436 rcb 0.007003 0.000550 c50aded0e71cdd51
weak 4 760436 rcb-fm 0.026234 0.001336 f6b34d1d05faa473
weak 4 760436 multi 0.091501 0.005431 f6b34d1d05faa473
strong 8 1520875 rcb 0.019124 0.001403 9f5cf54822c49335
strong 8 1520875 rcb-fm 0.071486 0.003909 29bd56d7602a754c
strong 8 1520875 multi 0.422502 0.017838 792dbf6957d1b8c5
weak 8 1520872 rcb 0.016647 0.001109 eef7430cb40dfd70
weak 8 1520872 rcb-fm 0.057140 0.003976 ac62a33a608f3394
weak 8 1520872 multi 0.335828 0.004775 ac62a33a608f3394
```

## Amdahl

The clocks are select, boundary refinement, coarsening, packing, the balance repair, the graph check, and join scheduling.
Each figure is wall time.
At one thread the two sides of a join are added.
Above one thread the slower side is kept, and join scheduling is the join wall beyond that side.
A share is that summed wall time divided by the mean partition wall time times the repetition count.
The other share is the mean wall time minus the named phases.
Karp-Flatt uses the strong rows.
With speedup s and thread count p, the serial fraction is (1/s - 1/p) / (1 - 1/p).

```text
method kind threads n select fm coarse pack repair check schedule other
strong-rcb threads 1 n 1520875 select 0.516 fm 0.000 coarse 0.000 pack 0.429 repair 0.000 check 0.046 schedule 0.009 other 0.00017
strong-rcb-fm threads 1 n 1520875 select 0.455 fm 0.414 coarse 0.000 pack 0.082 repair 0.037 check 0.010 schedule 0.002 other 0.00004
strong-multi threads 1 n 1520875 select 0.135 fm 0.183 coarse 0.634 pack 0.030 repair 0.012 check 0.002 schedule 0.001 other 0.00279
weak-rcb threads 1 n 190109 select 0.546 fm 0.000 coarse 0.000 pack 0.400 repair 0.000 check 0.031 schedule 0.023 other 0.00037
weak-rcb-fm threads 1 n 190109 select 0.628 fm 0.220 coarse 0.000 pack 0.112 repair 0.024 check 0.009 schedule 0.007 other 0.00011
weak-multi threads 1 n 190109 select 0.243 fm 0.107 coarse 0.577 pack 0.053 repair 0.009 check 0.003 schedule 0.002 other 0.00442
strong-rcb threads 2 n 1520875 select 0.471 fm 0.000 coarse 0.000 pack 0.449 repair 0.000 check 0.060 schedule 0.020 other 0.00018
strong-rcb-fm threads 2 n 1520875 select 0.422 fm 0.415 coarse 0.000 pack 0.095 repair 0.051 check 0.014 schedule 0.002 other 0.00005
strong-multi threads 2 n 1520875 select 0.094 fm 0.144 coarse 0.717 pack 0.027 repair 0.012 check 0.002 schedule 0.001 other 0.00302
weak-rcb threads 2 n 380218 select 0.495 fm 0.000 coarse 0.000 pack 0.381 repair 0.000 check 0.070 schedule 0.053 other 0.00083
weak-rcb-fm threads 2 n 380218 select 0.562 fm 0.252 coarse 0.000 pack 0.124 repair 0.029 check 0.012 schedule 0.020 other 0.00025
weak-multi threads 2 n 380218 select 0.164 fm 0.118 coarse 0.648 pack 0.046 repair 0.008 check 0.006 schedule 0.006 other 0.00464
strong-rcb threads 4 n 1520875 select 0.438 fm 0.000 coarse 0.000 pack 0.499 repair 0.000 check 0.049 schedule 0.013 other 0.00019
strong-rcb-fm threads 4 n 1520875 select 0.474 fm 0.334 coarse 0.000 pack 0.113 repair 0.068 check 0.009 schedule 0.002 other 0.00006
strong-multi threads 4 n 1520875 select 0.079 fm 0.103 coarse 0.771 pack 0.028 repair 0.014 check 0.002 schedule 0.001 other 0.00342
weak-rcb threads 4 n 760436 select 0.467 fm 0.000 coarse 0.000 pack 0.453 repair 0.000 check 0.054 schedule 0.025 other 0.00046
weak-rcb-fm threads 4 n 760436 select 0.564 fm 0.266 coarse 0.000 pack 0.123 repair 0.032 check 0.011 schedule 0.005 other 0.00015
weak-multi threads 4 n 760436 select 0.150 fm 0.106 coarse 0.677 pack 0.048 repair 0.010 check 0.003 schedule 0.001 other 0.00491
strong-rcb threads 8 n 1520875 select 0.428 fm 0.000 coarse 0.000 pack 0.532 repair 0.000 check 0.034 schedule 0.007 other 0.00020
strong-rcb-fm threads 8 n 1520875 select 0.458 fm 0.316 coarse 0.000 pack 0.138 repair 0.077 check 0.009 schedule 0.001 other 0.00006
strong-multi threads 8 n 1520875 select 0.080 fm 0.079 coarse 0.791 pack 0.030 repair 0.014 check 0.001 schedule 0.000 other 0.00334
weak-rcb threads 8 n 1520872 select 0.427 fm 0.000 coarse 0.000 pack 0.509 repair 0.000 check 0.044 schedule 0.019 other 0.00023
weak-rcb-fm threads 8 n 1520872 select 0.507 fm 0.292 coarse 0.000 pack 0.135 repair 0.056 check 0.007 schedule 0.003 other 0.00008
weak-multi threads 8 n 1520872 select 0.087 fm 0.070 coarse 0.795 pack 0.032 repair 0.010 check 0.001 schedule 0.000 other 0.00396
```

```text
strong method threads select_e fm_e coarse_e pack_e repair_e check_e schedule_e
strong multi threads 2 select_e 0.245 fm_e 0.418 coarse_e 1.032 pack_e 0.607 repair_e 0.850 check_e 0.800 schedule_e 0.795
strong multi threads 4 select_e 0.407 fm_e 0.384 coarse_e 1.216 pack_e 0.830 repair_e 1.195 check_e 0.504 schedule_e 1.100
strong multi threads 8 select_e 0.450 fm_e 0.289 coarse_e 1.100 pack_e 0.848 repair_e 1.052 check_e 0.298 schedule_e 0.495
strong rcb threads 2 select_e 0.349 fm_e na coarse_e na pack_e 0.548 repair_e na check_e 0.935 schedule_e 2.259
strong rcb threads 4 select_e 0.463 fm_e na coarse_e na pack_e 0.758 repair_e na check_e 0.682 schedule_e 0.979
strong rcb threads 8 select_e 0.474 fm_e na coarse_e na pack_e 0.780 repair_e na check_e 0.408 schedule_e 0.422
strong rcb-fm threads 2 select_e 0.256 fm_e 0.358 coarse_e na pack_e 0.558 repair_e 0.860 check_e 0.942 schedule_e 1.032
strong rcb-fm threads 4 select_e 0.491 fm_e 0.304 coarse_e na pack_e 0.757 repair_e 1.097 check_e 0.408 schedule_e 0.768
strong rcb-fm threads 8 select_e 0.419 fm_e 0.283 coarse_e na pack_e 0.796 repair_e 1.010 check_e 0.356 schedule_e 0.328
```

The measured limiter for rcb-fm at eight threads is the weighted median, with Karp-Flatt 0.419.
That rank uses serial time, Karp-Flatt times the one-thread phase wall.
The first cut builds one weighted histogram per thread.
The bin count is the span of the cut coordinate.
A prefix sum finds the half-weight bin, and ties break on the cell id.
The two halves already run in parallel after that cut.
The graph check walks offsets and neighbors in chunks and reduces with a boolean and.
Packing, the balance repair, the graph check, and join scheduling stay smaller than the median.

## Prose

The prose pass covers the placement note, the report, the partition page, the C API page, and the README.
Two older pages keep their earlier wording:

```text
docs/explanation/mic-and-cells.md
docs/reference/algorithm.md
```

