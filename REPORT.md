# Cell partition

The cell graph, the partition, and the geometric halo live in linkcell.
Ring membership, the Message Passing Interface (MPI) share, and ice clustering stay in seams-core.
eOn calls `pairs_within` only. rgpot builds pairs through vesin. readcon-core reads configuration (CON) frames.

This cell-partition report is generated from one log.
Every median and every interquartile range (IQR) is the value that log printed for that cell.
The raw sample line is the series those values came from, at type-7.

The log names the hardware as Intel Xeon Processor, 8 physical cores, one thread per core, one node, 2400 MHz.
A second node is recorded as `multi_node HOST NOT-RUN`.
Commit `bcafe9238903e3f204f46c3a6e7a716fe3713582`.
Tree `e74e97e776154e3e700996a2270282329c8f03b9`.
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
morton mW-110592 20894 1.000145 0.941354 0.019555 0.000542 1
hilbert mW-110592 20694 1.000145 0.943571 0.002206 0.000136 1
rcb mW-110592 13824 1.000000 0.396092 0.002476 0.000275 1
hilbert-fm mW-110592 20594 1.000579 0.942531 0.011575 0.000551 1
rcb-fm mW-110592 13824 1.000000 0.396092 0.008842 0.000564 1
multi mW-110592 13824 1.000000 0.396092 0.027556 0.001638 1
scotch mW-110592 15547 1.000217 0.478564 0.116616 0.005713 0
zoltan-rcb mW-110592 13824 1.000000 0.396092 0.003627 0.000245 0
ptscotch mW-110592 14067 1.000000 0.420738 13.633960 0.319207 0
morton mW-884736 83857 1.000009 0.397780 0.184968 0.004938 1
hilbert mW-884736 81032 1.000018 0.418792 0.021264 0.000673 1
rcb mW-884736 55990 1.000009 0.186900 0.016543 0.000992 1
hilbert-fm mW-884736 80931 1.000072 0.418751 0.069491 0.002472 1
rcb-fm mW-884736 55802 1.000054 0.183404 0.142402 0.005010 1
multi mW-884736 55800 1.000072 0.183409 0.869499 0.018604 1
scotch mW-884736 64754 1.000054 0.232028 0.691284 0.018936 0
zoltan-rcb mW-884736 55983 1.000009 0.186892 0.030214 0.001291 0
ptscotch mW-884736 72836 1.000000 0.261248 55.419726 0.699787 0
morton TIP4P-128000 2804168 1.000000 1.687500 0.000846 0.000219 1
hilbert TIP4P-128000 2791850 1.000687 1.661710 0.000108 0.000030 1
rcb TIP4P-128000 2492438 1.000625 1.353383 0.000394 0.000149 1
hilbert-fm TIP4P-128000 2671517 1.005625 1.500000 0.001955 0.000448 1
rcb-fm TIP4P-128000 2145600 1.000000 1.000000 0.002343 0.000285 1
multi TIP4P-128000 2145600 1.000000 1.000000 0.005471 0.000532 1
scotch TIP4P-128000 2169032 1.005375 1.052632 0.016137 0.001073 0
zoltan-rcb TIP4P-128000 2485058 1.000625 1.353383 0.000275 0.000029 0
ptscotch TIP4P-128000 2130992 1.001250 1.007692 3.520878 0.115090 0
morton TIP4P-1024000 13084202 1.000320 0.812144 0.007948 0.000173 1
hilbert TIP4P-1024000 12710308 1.000195 0.848197 0.001121 0.000055 1
rcb TIP4P-1024000 9965162 1.000266 0.576619 0.001187 0.000136 1
hilbert-fm TIP4P-1024000 12192983 1.000758 0.834594 0.025467 0.001076 1
rcb-fm TIP4P-1024000 8793439 1.000766 0.466543 0.012692 0.000883 1
multi TIP4P-1024000 8800880 1.000766 0.466263 0.038481 0.002613 1
scotch TIP4P-1024000 8911827 1.000719 0.490884 0.103454 0.003539 0
zoltan-rcb TIP4P-1024000 9897435 1.000312 0.564509 0.001757 0.000201 0
ptscotch TIP4P-1024000 8813078 1.000203 0.472678 5.087573 0.117622 0
morton slab-110592 18819 1.000217 0.750051 0.040121 0.000874 1
hilbert slab-110592 18188 1.000145 0.747628 0.004819 0.000085 1
rcb slab-110592 12672 1.000000 0.500000 0.003232 0.000128 1
hilbert-fm slab-110592 18148 1.000579 0.746570 0.009952 0.000946 1
rcb-fm slab-110592 11520 1.000000 0.396092 0.011342 0.000479 1
multi slab-110592 11520 1.000000 0.396092 0.065583 0.001041 1
scotch slab-110592 12553 1.000579 0.531308 0.146851 0.001929 0
zoltan-rcb slab-110592 12672 1.000000 0.500000 0.007688 0.000209 0
ptscotch slab-110592 11902 1.000000 2.827651 24.382509 0.520296 0
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
sum method morton cut 16011940 median_s 0.253439 eligible 1
sum method hilbert cut 15622072 median_s 0.029518 eligible 1
sum method rcb cut 12540086 median_s 0.023832 eligible 1
sum method hilbert-fm cut 14984173 median_s 0.118440 eligible 1
sum method rcb-fm cut 11020185 median_s 0.177621 eligible 1
sum method multi cut 11027624 median_s 1.006590 eligible 1
default_method rcb-fm
```

## Raw samples

Each line is the twelve wall times, in seconds, in the order they were measured.

```text
morton mW-110592 0.022727 0.019357 0.019424 0.019142 0.019130 0.019686 0.019978 0.019718 0.020866 0.019825 0.019353 0.019225
hilbert mW-110592 0.002472 0.002146 0.002170 0.002119 0.002163 0.002244 0.002426 0.002243 0.002394 0.002247 0.002140 0.002148
rcb mW-110592 0.002417 0.002135 0.002303 0.002560 0.002523 0.002853 0.002707 0.002323 0.002577 0.002642 0.002179 0.002430
hilbert-fm mW-110592 0.011877 0.011384 0.011252 0.011321 0.011266 0.013021 0.012034 0.012025 0.011671 0.011883 0.011479 0.011440
rcb-fm mW-110592 0.010019 0.009066 0.008584 0.008614 0.008907 0.008552 0.009455 0.009075 0.008688 0.011181 0.008496 0.008777
multi mW-110592 0.032292 0.027023 0.028059 0.026448 0.026657 0.027132 0.028575 0.027179 0.027932 0.030334 0.028925 0.027026
scotch mW-110592 0.137306 0.116031 0.115251 0.117201 0.114804 0.114752 0.120690 0.120357 0.128502 0.121342 0.115735 0.114042
zoltan-rcb mW-110592 0.004510 0.003863 0.003542 0.003563 0.003632 0.003602 0.003622 0.003935 0.003782 0.003755 0.003475 0.003506
ptscotch mW-110592 13.404205 13.472840 13.620397 14.227890 13.488867 13.458015 13.623216 13.742622 14.062006 13.793934 13.644704 13.834466
morton mW-884736 0.183915 0.182922 0.206234 0.185036 0.182376 0.185184 0.187400 0.191424 0.183603 0.183066 0.184900 0.206110
hilbert mW-884736 0.022082 0.021162 0.022594 0.021437 0.021083 0.021163 0.021365 0.021722 0.021113 0.021147 0.021043 0.023667
rcb mW-884736 0.016834 0.015454 0.016939 0.016410 0.015852 0.015447 0.015927 0.016838 0.015849 0.016675 0.016862 0.017395
hilbert-fm mW-884736 0.073916 0.069675 0.071987 0.069307 0.068124 0.069163 0.073332 0.069293 0.068219 0.069868 0.069271 0.071625
rcb-fm mW-884736 0.150770 0.138822 0.146605 0.142288 0.139917 0.141077 0.148520 0.143840 0.140268 0.139237 0.142515 0.144718
multi mW-884736 0.931887 0.883639 0.844163 0.862215 0.846837 0.862913 0.874456 0.880577 0.866922 0.872076 0.863428 0.898093
scotch mW-884736 0.744667 0.714823 0.681953 0.709072 0.667782 0.689979 0.692588 0.686909 0.679949 0.697794 0.680851 0.696962
zoltan-rcb mW-884736 0.034361 0.030846 0.031546 0.030107 0.030321 0.030077 0.031181 0.030473 0.029472 0.029648 0.029610 0.029223
ptscotch mW-884736 54.840186 55.112354 56.399860 55.353543 55.485910 55.758639 56.844685 54.610576 55.671754 55.132453 55.978883 55.114433
morton TIP4P-128000 0.000796 0.000795 0.001049 0.000835 0.001109 0.000920 0.000824 0.000801 0.000797 0.001393 0.001009 0.000857
hilbert TIP4P-128000 0.000100 0.000103 0.000180 0.000108 0.000184 0.000117 0.000108 0.000106 0.000107 0.000172 0.000125 0.000109
rcb TIP4P-128000 0.000353 0.000393 0.000537 0.000218 0.000252 0.000462 0.000500 0.000579 0.000469 0.000396 0.000234 0.000385
hilbert-fm TIP4P-128000 0.001749 0.001775 0.002280 0.001878 0.002367 0.002005 0.001823 0.002029 0.001803 0.002822 0.001906 0.002262
rcb-fm TIP4P-128000 0.002076 0.002500 0.002422 0.002128 0.002121 0.002283 0.002314 0.002372 0.002435 0.002512 0.002467 0.002168
multi TIP4P-128000 0.005054 0.005354 0.006201 0.004952 0.005144 0.005632 0.005607 0.005540 0.005014 0.007440 0.005402 0.005717
scotch TIP4P-128000 0.015095 0.016919 0.016785 0.015453 0.015611 0.016397 0.016066 0.015817 0.016208 0.016700 0.017499 0.015660
zoltan-rcb TIP4P-128000 0.000244 0.000271 0.000302 0.000276 0.000301 0.000289 0.000265 0.000291 0.000299 0.000275 0.000263 0.000260
ptscotch TIP4P-128000 3.481099 3.693273 3.434691 3.513597 3.372446 3.578654 3.382846 3.641552 3.528159 3.498020 3.602386 3.534146
morton TIP4P-1024000 0.008149 0.008020 0.007784 0.007856 0.007944 0.007970 0.007953 0.007837 0.008037 0.008842 0.007860 0.007826
hilbert TIP4P-1024000 0.001136 0.001110 0.001131 0.001060 0.001097 0.001143 0.001150 0.001086 0.001154 0.001347 0.001091 0.001080
rcb TIP4P-1024000 0.001004 0.001351 0.001237 0.001139 0.001281 0.001149 0.001224 0.001069 0.001065 0.001226 0.001126 0.001320
hilbert-fm TIP4P-1024000 0.026740 0.025493 0.026595 0.025164 0.024827 0.024942 0.025018 0.025441 0.027800 0.025605 0.026074 0.025322
rcb-fm TIP4P-1024000 0.012877 0.012766 0.015986 0.012655 0.012342 0.012024 0.012252 0.014901 0.012728 0.012431 0.014210 0.012284
multi TIP4P-1024000 0.038616 0.042464 0.042050 0.038175 0.037287 0.037321 0.039308 0.040335 0.037967 0.038347 0.040668 0.037242
scotch TIP4P-1024000 0.103891 0.101928 0.103254 0.105332 0.102344 0.101763 0.108243 0.101703 0.102575 0.132320 0.107119 0.103655
zoltan-rcb TIP4P-1024000 0.001939 0.001720 0.001774 0.002123 0.002132 0.001544 0.001581 0.001689 0.001654 0.001798 0.001741 0.001862
ptscotch TIP4P-1024000 5.461829 5.098591 5.059494 5.125334 4.828378 5.182657 4.850787 5.153639 5.011857 5.045962 5.159314 5.076555
morton slab-110592 0.040215 0.041194 0.039701 0.039610 0.042306 0.040028 0.040023 0.039593 0.040514 0.040763 0.039952 0.040766
hilbert slab-110592 0.004872 0.006033 0.004814 0.004866 0.004931 0.004784 0.004710 0.004840 0.004809 0.004748 0.004779 0.004824
rcb slab-110592 0.003247 0.003283 0.003281 0.003202 0.003358 0.003141 0.003084 0.003185 0.003216 0.003477 0.003501 0.003074
hilbert-fm slab-110592 0.009816 0.012715 0.010642 0.010753 0.009706 0.009685 0.009614 0.009569 0.010087 0.010256 0.010604 0.009547
rcb-fm slab-110592 0.011597 0.012528 0.011333 0.011300 0.011351 0.010872 0.011167 0.011001 0.012059 0.011598 0.011660 0.011037
multi slab-110592 0.065209 0.066380 0.065355 0.064452 0.065698 0.065683 0.064395 0.064403 0.069520 0.065955 0.066656 0.065483
scotch slab-110592 0.143648 0.145590 0.148608 0.147615 0.147215 0.145736 0.144404 0.145911 0.146945 0.146757 0.147671 0.153650
zoltan-rcb slab-110592 0.007935 0.007500 0.007583 0.007710 0.007556 0.007619 0.007666 0.007875 0.007800 0.007767 0.007627 0.008278
ptscotch slab-110592 24.099267 24.326646 24.438373 24.999640 24.101547 24.475938 24.852825 24.544089 24.053383 24.048213 24.147804 24.914885
```

## Balance sweep

The same five ratios appear on every system in this sweep.
The main table uses the derived bound.

```text
morton 1.00 mW-110592 cut 20894 imbalance 1.000145 median 0.018941 iqr 0.000363
hilbert 1.00 mW-110592 cut 20694 imbalance 1.000145 median 0.002014 iqr 0.000043
rcb 1.00 mW-110592 cut 13824 imbalance 1.000000 median 0.002277 iqr 0.000122
hilbert-fm 1.00 mW-110592 cut 20692 imbalance 1.000000 median 0.005239 iqr 0.000082
rcb-fm 1.00 mW-110592 cut 13824 imbalance 1.000000 median 0.008014 iqr 0.000164
multi 1.00 mW-110592 cut 13824 imbalance 1.000000 median 0.025585 iqr 0.000437
scotch 1.00 mW-110592 cut 20801 imbalance 1.000000 median 0.113968 iqr 0.001750
morton 1.01 mW-110592 cut 20894 imbalance 1.000145 median 0.019015 iqr 0.000313
hilbert 1.01 mW-110592 cut 20694 imbalance 1.000145 median 0.002036 iqr 0.000085
rcb 1.01 mW-110592 cut 13824 imbalance 1.000000 median 0.002267 iqr 0.000169
hilbert-fm 1.01 mW-110592 cut 20584 imbalance 1.002604 median 0.005240 iqr 0.000186
rcb-fm 1.01 mW-110592 cut 13824 imbalance 1.000000 median 0.008520 iqr 0.000240
multi 1.01 mW-110592 cut 13824 imbalance 1.000000 median 0.026820 iqr 0.000379
scotch 1.01 mW-110592 cut 14698 imbalance 1.009910 median 0.114119 iqr 0.001692
morton 1.02 mW-110592 cut 20894 imbalance 1.000145 median 0.018958 iqr 0.000290
hilbert 1.02 mW-110592 cut 20694 imbalance 1.000145 median 0.002046 iqr 0.000095
rcb 1.02 mW-110592 cut 13824 imbalance 1.000000 median 0.002353 iqr 0.000159
hilbert-fm 1.02 mW-110592 cut 20584 imbalance 1.002604 median 0.005230 iqr 0.000224
rcb-fm 1.02 mW-110592 cut 13824 imbalance 1.000000 median 0.008510 iqr 0.000505
multi 1.02 mW-110592 cut 13824 imbalance 1.000000 median 0.027163 iqr 0.000408
scotch 1.02 mW-110592 cut 14391 imbalance 1.019676 median 0.100675 iqr 0.001061
morton 1.05 mW-110592 cut 20894 imbalance 1.000145 median 0.019315 iqr 0.000510
hilbert 1.05 mW-110592 cut 20694 imbalance 1.000145 median 0.002080 iqr 0.000104
rcb 1.05 mW-110592 cut 13824 imbalance 1.000000 median 0.002322 iqr 0.000233
hilbert-fm 1.05 mW-110592 cut 20584 imbalance 1.002604 median 0.005285 iqr 0.000124
rcb-fm 1.05 mW-110592 cut 13824 imbalance 1.000000 median 0.008739 iqr 0.000742
multi 1.05 mW-110592 cut 13824 imbalance 1.000000 median 0.027961 iqr 0.000849
scotch 1.05 mW-110592 cut 14817 imbalance 1.045139 median 0.115859 iqr 0.006026
morton 1.10 mW-110592 cut 20894 imbalance 1.000145 median 0.018954 iqr 0.000162
hilbert 1.10 mW-110592 cut 20694 imbalance 1.000145 median 0.002040 iqr 0.000083
rcb 1.10 mW-110592 cut 13824 imbalance 1.000000 median 0.002192 iqr 0.000263
hilbert-fm 1.10 mW-110592 cut 20584 imbalance 1.002604 median 0.005273 iqr 0.000221
rcb-fm 1.10 mW-110592 cut 13824 imbalance 1.000000 median 0.008584 iqr 0.000296
multi 1.10 mW-110592 cut 13824 imbalance 1.000000 median 0.027479 iqr 0.000207
scotch 1.10 mW-110592 cut 15098 imbalance 1.099826 median 0.117056 iqr 0.001703
morton 1.00 mW-884736 cut 83857 imbalance 1.000009 median 0.187041 iqr 0.008841
hilbert 1.00 mW-884736 cut 81032 imbalance 1.000018 median 0.021134 iqr 0.000553
rcb 1.00 mW-884736 cut 55990 imbalance 1.000009 median 0.016088 iqr 0.000869
hilbert-fm 1.00 mW-884736 cut 81032 imbalance 1.000000 median 0.040935 iqr 0.001205
rcb-fm 1.00 mW-884736 cut 55990 imbalance 1.000009 median 0.101003 iqr 0.003034
multi 1.00 mW-884736 cut 55990 imbalance 1.000009 median 0.286046 iqr 0.011401
scotch 1.00 mW-884736 cut 81445 imbalance 1.000000 median 0.651561 iqr 0.007770
morton 1.01 mW-884736 cut 83857 imbalance 1.000009 median 0.183269 iqr 0.003241
hilbert 1.01 mW-884736 cut 81032 imbalance 1.000018 median 0.020798 iqr 0.000688
rcb 1.01 mW-884736 cut 55990 imbalance 1.000009 median 0.015870 iqr 0.000395
hilbert-fm 1.01 mW-884736 cut 80799 imbalance 1.004792 median 0.234011 iqr 0.005909
rcb-fm 1.01 mW-884736 cut 55894 imbalance 1.000868 median 0.086701 iqr 0.002896
multi 1.01 mW-884736 cut 55837 imbalance 1.000922 median 0.473925 iqr 0.014129
scotch 1.01 mW-884736 cut 58097 imbalance 1.009865 median 0.597505 iqr 0.018048
morton 1.02 mW-884736 cut 83857 imbalance 1.000009 median 0.189828 iqr 0.021147
hilbert 1.02 mW-884736 cut 81032 imbalance 1.000018 median 0.022102 iqr 0.002486
rcb 1.02 mW-884736 cut 55990 imbalance 1.000009 median 0.017215 iqr 0.002306
hilbert-fm 1.02 mW-884736 cut 80799 imbalance 1.004792 median 0.246760 iqr 0.030353
rcb-fm 1.02 mW-884736 cut 55894 imbalance 1.000868 median 0.088397 iqr 0.010138
multi 1.02 mW-884736 cut 55837 imbalance 1.000922 median 0.510240 iqr 0.053698
scotch 1.02 mW-884736 cut 60287 imbalance 1.019938 median 0.652405 iqr 0.073182
morton 1.05 mW-884736 cut 83857 imbalance 1.000009 median 0.189448 iqr 0.013658
hilbert 1.05 mW-884736 cut 81032 imbalance 1.000018 median 0.021481 iqr 0.001526
rcb 1.05 mW-884736 cut 55990 imbalance 1.000009 median 0.015939 iqr 0.001038
hilbert-fm 1.05 mW-884736 cut 80799 imbalance 1.004792 median 0.241601 iqr 0.012855
rcb-fm 1.05 mW-884736 cut 55894 imbalance 1.000868 median 0.089690 iqr 0.004610
multi 1.05 mW-884736 cut 55837 imbalance 1.000922 median 0.496548 iqr 0.025305
scotch 1.05 mW-884736 cut 62348 imbalance 1.018971 median 0.629518 iqr 0.034671
morton 1.10 mW-884736 cut 83857 imbalance 1.000009 median 0.185313 iqr 0.006305
hilbert 1.10 mW-884736 cut 81032 imbalance 1.000018 median 0.020906 iqr 0.000588
rcb 1.10 mW-884736 cut 55990 imbalance 1.000009 median 0.016104 iqr 0.000311
hilbert-fm 1.10 mW-884736 cut 80799 imbalance 1.004792 median 0.237963 iqr 0.013824
rcb-fm 1.10 mW-884736 cut 55894 imbalance 1.000868 median 0.087613 iqr 0.003229
multi 1.10 mW-884736 cut 55837 imbalance 1.000922 median 0.481697 iqr 0.013653
scotch 1.10 mW-884736 cut 59853 imbalance 1.086842 median 0.619927 iqr 0.023787
morton 1.00 TIP4P-128000 cut 2804168 imbalance 1.000000 median 0.000813 iqr 0.000036
hilbert 1.00 TIP4P-128000 cut 2791850 imbalance 1.000687 median 0.000104 iqr 0.000007
rcb 1.00 TIP4P-128000 cut 2492438 imbalance 1.000625 median 0.000342 iqr 0.000070
hilbert-fm 1.00 TIP4P-128000 cut 2791850 imbalance 1.000687 median 0.000283 iqr 0.000011
rcb-fm 1.00 TIP4P-128000 cut 2145600 imbalance 1.000000 median 0.000961 iqr 0.000217
multi 1.00 TIP4P-128000 cut 2145600 imbalance 1.000000 median 0.003779 iqr 0.000216
scotch 1.00 TIP4P-128000 cut 2776979 imbalance 1.000687 median 0.009552 iqr 0.000491
morton 1.01 TIP4P-128000 cut 2804168 imbalance 1.000000 median 0.000829 iqr 0.000021
hilbert 1.01 TIP4P-128000 cut 2791850 imbalance 1.000687 median 0.000106 iqr 0.000004
rcb 1.01 TIP4P-128000 cut 2492438 imbalance 1.000625 median 0.000366 iqr 0.000119
hilbert-fm 1.01 TIP4P-128000 cut 2668242 imbalance 1.005250 median 0.002493 iqr 0.000190
rcb-fm 1.01 TIP4P-128000 cut 2130992 imbalance 1.001250 median 0.005145 iqr 0.000252
multi 1.01 TIP4P-128000 cut 2130992 imbalance 1.001250 median 0.008352 iqr 0.000318
scotch 1.01 TIP4P-128000 cut 2172264 imbalance 1.008563 median 0.017039 iqr 0.001238
morton 1.02 TIP4P-128000 cut 2804168 imbalance 1.000000 median 0.000794 iqr 0.000044
hilbert 1.02 TIP4P-128000 cut 2791850 imbalance 1.000687 median 0.000099 iqr 0.000010
rcb 1.02 TIP4P-128000 cut 2492438 imbalance 1.000625 median 0.000357 iqr 0.000034
hilbert-fm 1.02 TIP4P-128000 cut 2667911 imbalance 1.016563 median 0.001501 iqr 0.000129
rcb-fm 1.02 TIP4P-128000 cut 2145600 imbalance 1.000000 median 0.002849 iqr 0.000119
multi 1.02 TIP4P-128000 cut 2145600 imbalance 1.000000 median 0.006161 iqr 0.000230
scotch 1.02 TIP4P-128000 cut 2196133 imbalance 1.019062 median 0.015584 iqr 0.000472
morton 1.05 TIP4P-128000 cut 2804168 imbalance 1.000000 median 0.000765 iqr 0.000031
hilbert 1.05 TIP4P-128000 cut 2791850 imbalance 1.000687 median 0.000095 iqr 0.000010
rcb 1.05 TIP4P-128000 cut 2492438 imbalance 1.000625 median 0.000309 iqr 0.000071
hilbert-fm 1.05 TIP4P-128000 cut 2658326 imbalance 1.049063 median 0.001486 iqr 0.000168
rcb-fm 1.05 TIP4P-128000 cut 2120992 imbalance 1.050000 median 0.004761 iqr 0.000183
multi 1.05 TIP4P-128000 cut 2120512 imbalance 1.050000 median 0.017083 iqr 0.000474
scotch 1.05 TIP4P-128000 cut 2155060 imbalance 1.050000 median 0.015006 iqr 0.000577
morton 1.10 TIP4P-128000 cut 2804168 imbalance 1.000000 median 0.000780 iqr 0.000090
hilbert 1.10 TIP4P-128000 cut 2791850 imbalance 1.000687 median 0.000097 iqr 0.000011
rcb 1.10 TIP4P-128000 cut 2492438 imbalance 1.000625 median 0.000358 iqr 0.000115
hilbert-fm 1.10 TIP4P-128000 cut 2654022 imbalance 1.096750 median 0.001588 iqr 0.000178
rcb-fm 1.10 TIP4P-128000 cut 2092800 imbalance 1.062500 median 0.005288 iqr 0.000238
multi 1.10 TIP4P-128000 cut 2092800 imbalance 1.062500 median 0.009034 iqr 0.000604
scotch 1.10 TIP4P-128000 cut 2118900 imbalance 1.089063 median 0.015110 iqr 0.000441
morton 1.00 TIP4P-1024000 cut 13084202 imbalance 1.000320 median 0.007623 iqr 0.000066
hilbert 1.00 TIP4P-1024000 cut 12710308 imbalance 1.000195 median 0.000953 iqr 0.000006
rcb 1.00 TIP4P-1024000 cut 9965162 imbalance 1.000266 median 0.001056 iqr 0.000065
hilbert-fm 1.00 TIP4P-1024000 cut 12710308 imbalance 1.000195 median 0.001982 iqr 0.000049
rcb-fm 1.00 TIP4P-1024000 cut 9965162 imbalance 1.000266 median 0.006212 iqr 0.000304
multi 1.00 TIP4P-1024000 cut 9965162 imbalance 1.000266 median 0.029626 iqr 0.000173
scotch 1.00 TIP4P-1024000 cut 15041835 imbalance 1.000133 median 0.081314 iqr 0.000978
morton 1.01 TIP4P-1024000 cut 13084202 imbalance 1.000320 median 0.007658 iqr 0.000154
hilbert 1.01 TIP4P-1024000 cut 12710308 imbalance 1.000195 median 0.000962 iqr 0.000018
rcb 1.01 TIP4P-1024000 cut 9965162 imbalance 1.000266 median 0.001076 iqr 0.000076
hilbert-fm 1.01 TIP4P-1024000 cut 12126380 imbalance 1.010000 median 0.020528 iqr 0.000241
rcb-fm 1.01 TIP4P-1024000 cut 8741847 imbalance 1.009797 median 0.015353 iqr 0.000373
multi 1.01 TIP4P-1024000 cut 8762734 imbalance 1.009813 median 0.043751 iqr 0.001280
scotch 1.01 TIP4P-1024000 cut 8949479 imbalance 1.009922 median 0.108554 iqr 0.001965
morton 1.02 TIP4P-1024000 cut 13084202 imbalance 1.000320 median 0.007643 iqr 0.000159
hilbert 1.02 TIP4P-1024000 cut 12710308 imbalance 1.000195 median 0.000958 iqr 0.000013
rcb 1.02 TIP4P-1024000 cut 9965162 imbalance 1.000266 median 0.001039 iqr 0.000085
hilbert-fm 1.02 TIP4P-1024000 cut 12078105 imbalance 1.019977 median 0.007777 iqr 0.000145
rcb-fm 1.02 TIP4P-1024000 cut 8726026 imbalance 1.019969 median 0.014649 iqr 0.000298
multi 1.02 TIP4P-1024000 cut 8760713 imbalance 1.019969 median 0.043806 iqr 0.000406
scotch 1.02 TIP4P-1024000 cut 8861660 imbalance 1.020000 median 0.104233 iqr 0.001542
morton 1.05 TIP4P-1024000 cut 13084202 imbalance 1.000320 median 0.007699 iqr 0.000154
hilbert 1.05 TIP4P-1024000 cut 12710308 imbalance 1.000195 median 0.000970 iqr 0.000037
rcb 1.05 TIP4P-1024000 cut 9965162 imbalance 1.000266 median 0.001060 iqr 0.000072
hilbert-fm 1.05 TIP4P-1024000 cut 11893822 imbalance 1.050000 median 0.012454 iqr 0.000610
rcb-fm 1.05 TIP4P-1024000 cut 8743808 imbalance 1.050000 median 0.026672 iqr 0.000609
multi 1.05 TIP4P-1024000 cut 8746824 imbalance 1.050000 median 0.054694 iqr 0.001840
scotch 1.05 TIP4P-1024000 cut 8730300 imbalance 1.028125 median 0.096339 iqr 0.001758
morton 1.10 TIP4P-1024000 cut 13084202 imbalance 1.000320 median 0.007655 iqr 0.000142
hilbert 1.10 TIP4P-1024000 cut 12710308 imbalance 1.000195 median 0.000960 iqr 0.000027
rcb 1.10 TIP4P-1024000 cut 9965162 imbalance 1.000266 median 0.001090 iqr 0.000076
hilbert-fm 1.10 TIP4P-1024000 cut 11733490 imbalance 1.099828 median 0.014927 iqr 0.000350
rcb-fm 1.10 TIP4P-1024000 cut 8530800 imbalance 1.100000 median 0.014002 iqr 0.000056
multi 1.10 TIP4P-1024000 cut 8523300 imbalance 1.100000 median 0.070319 iqr 0.000641
scotch 1.10 TIP4P-1024000 cut 8812500 imbalance 1.058594 median 0.093778 iqr 0.001450
morton 1.00 slab-110592 cut 18819 imbalance 1.000217 median 0.040815 iqr 0.002121
hilbert 1.00 slab-110592 cut 18188 imbalance 1.000145 median 0.004631 iqr 0.000209
rcb 1.00 slab-110592 cut 12672 imbalance 1.000000 median 0.002863 iqr 0.000305
hilbert-fm 1.00 slab-110592 cut 18200 imbalance 1.000000 median 0.007046 iqr 0.000288
rcb-fm 1.00 slab-110592 cut 11520 imbalance 1.000000 median 0.010628 iqr 0.000573
multi 1.00 slab-110592 cut 11520 imbalance 1.000000 median 0.063834 iqr 0.001865
scotch 1.00 slab-110592 cut 16596 imbalance 1.000000 median 0.147886 iqr 0.004935
morton 1.01 slab-110592 cut 18819 imbalance 1.000217 median 0.039938 iqr 0.001705
hilbert 1.01 slab-110592 cut 18188 imbalance 1.000145 median 0.004539 iqr 0.000107
rcb 1.01 slab-110592 cut 12672 imbalance 1.000000 median 0.003079 iqr 0.000348
hilbert-fm 1.01 slab-110592 cut 18078 imbalance 1.009404 median 0.044044 iqr 0.001379
rcb-fm 1.01 slab-110592 cut 11520 imbalance 1.000000 median 0.011345 iqr 0.001091
multi 1.01 slab-110592 cut 11520 imbalance 1.000000 median 0.066658 iqr 0.002145
scotch 1.01 slab-110592 cut 11839 imbalance 1.009549 median 0.117906 iqr 0.002968
morton 1.02 slab-110592 cut 18819 imbalance 1.000217 median 0.040377 iqr 0.000964
hilbert 1.02 slab-110592 cut 18188 imbalance 1.000145 median 0.004563 iqr 0.000213
rcb 1.02 slab-110592 cut 12672 imbalance 1.000000 median 0.002827 iqr 0.000156
hilbert-fm 1.02 slab-110592 cut 18065 imbalance 1.019748 median 0.057096 iqr 0.001677
rcb-fm 1.02 slab-110592 cut 11520 imbalance 1.000000 median 0.011052 iqr 0.000473
multi 1.02 slab-110592 cut 11520 imbalance 1.000000 median 0.065152 iqr 0.002751
scotch 1.02 slab-110592 cut 12384 imbalance 1.018953 median 0.121542 iqr 0.003217
morton 1.05 slab-110592 cut 18819 imbalance 1.000217 median 0.041670 iqr 0.002890
hilbert 1.05 slab-110592 cut 18188 imbalance 1.000145 median 0.004724 iqr 0.000244
rcb 1.05 slab-110592 cut 12672 imbalance 1.000000 median 0.002982 iqr 0.000288
hilbert-fm 1.05 slab-110592 cut 18012 imbalance 1.049986 median 0.102427 iqr 0.005179
rcb-fm 1.05 slab-110592 cut 11520 imbalance 1.000000 median 0.011435 iqr 0.000589
multi 1.05 slab-110592 cut 11520 imbalance 1.000000 median 0.067503 iqr 0.002361
scotch 1.05 slab-110592 cut 11911 imbalance 1.049986 median 0.124915 iqr 0.008732
morton 1.10 slab-110592 cut 18819 imbalance 1.000217 median 0.043174 iqr 0.003505
hilbert 1.10 slab-110592 cut 18188 imbalance 1.000145 median 0.005090 iqr 0.000603
rcb 1.10 slab-110592 cut 12672 imbalance 1.000000 median 0.003219 iqr 0.000193
hilbert-fm 1.10 slab-110592 cut 18008 imbalance 1.053168 median 0.110633 iqr 0.007247
rcb-fm 1.10 slab-110592 cut 11520 imbalance 1.000000 median 0.011856 iqr 0.001322
multi 1.10 slab-110592 cut 11520 imbalance 1.000000 median 0.070391 iqr 0.008654
scotch 1.10 slab-110592 cut 11938 imbalance 1.094184 median 0.129656 iqr 0.014065
```

## Scaling

Strong scaling keeps the lattice at 1,520,875 points (115 cubed).
Weak scaling uses 190,109 points times the thread count, so eight threads is 1,520,872 points.
Median and interquartile range (IQR) are recomputed from the raw line.
`profile_ns` sums the critical-path wall time of select, boundary refinement, and coarsening over the repetitions, in nanoseconds.

```text
kind threads n method median iqr
strong 1 1520875 rcb 0.027011 0.003717
strong 1 1520875 rcb-fm 0.192400 0.032978
strong 1 1520875 multi 0.553992 0.069628
weak 1 190109 rcb 0.005982 0.000270
weak 1 190109 rcb-fm 0.023205 0.000370
weak 1 190109 multi 0.071937 0.001192
strong 2 1520875 rcb 0.019197 0.000973
strong 2 1520875 rcb-fm 0.108147 0.005536
strong 2 1520875 multi 0.447646 0.039549
weak 2 380218 rcb 0.002768 0.000251
weak 2 380218 rcb-fm 0.009871 0.000758
weak 2 380218 multi 0.041577 0.002573
strong 4 1520875 rcb 0.018555 0.000654
strong 4 1520875 rcb-fm 0.093014 0.002522
strong 4 1520875 multi 0.421269 0.016517
weak 4 760436 rcb 0.005023 0.000562
weak 4 760436 rcb-fm 0.019127 0.001163
weak 4 760436 multi 0.077521 0.003096
strong 8 1520875 rcb 0.017535 0.000961
strong 8 1520875 rcb-fm 0.088328 0.003765
strong 8 1520875 multi 0.409058 0.013415
weak 8 1520872 rcb 0.013057 0.000428
weak 8 1520872 rcb-fm 0.056997 0.002400
weak 8 1520872 multi 0.309010 0.014302
```

## Amdahl

Select, boundary refinement, and coarsening are timed on the critical path.
Each figure is wall time.
At one thread the two sides of a join are added.
Above one thread the slower side is kept.
A share is that summed wall time divided by the mean partition wall time times the repetition count.
The other share is what remains, and a negative value would mean the phases over-count the wall clock.
Karp-Flatt uses the strong rows.
With speedup s and thread count p, the serial fraction is (1/s - 1/p) / (1 - 1/p).

```text
method kind threads n select fm coarse other
strong-rcb threads 1 n 1520875 select 0.389 fm 0.000 coarse 0.000 other 0.611
strong-rcb-fm threads 1 n 1520875 select 0.477 fm 0.405 coarse 0.000 other 0.118
strong-multi threads 1 n 1520875 select 0.168 fm 0.190 coarse 0.569 other 0.073
weak-rcb threads 1 n 190109 select 0.282 fm 0.000 coarse 0.000 other 0.718
weak-rcb-fm threads 1 n 190109 select 0.474 fm 0.308 coarse 0.000 other 0.218
weak-multi threads 1 n 190109 select 0.153 fm 0.122 coarse 0.608 other 0.117
strong-rcb threads 2 n 1520875 select 0.316 fm 0.000 coarse 0.000 other 0.684
strong-rcb-fm threads 2 n 1520875 select 0.472 fm 0.370 coarse 0.000 other 0.158
strong-multi threads 2 n 1520875 select 0.116 fm 0.141 coarse 0.674 other 0.070
weak-rcb threads 2 n 380218 select 0.197 fm 0.000 coarse 0.000 other 0.803
weak-rcb-fm threads 2 n 380218 select 0.337 fm 0.326 coarse 0.000 other 0.337
weak-multi threads 2 n 380218 select 0.077 fm 0.124 coarse 0.679 other 0.120
strong-rcb threads 4 n 1520875 select 0.344 fm 0.000 coarse 0.000 other 0.656
strong-rcb-fm threads 4 n 1520875 select 0.508 fm 0.323 coarse 0.000 other 0.169
strong-multi threads 4 n 1520875 select 0.118 fm 0.104 coarse 0.703 other 0.075
weak-rcb threads 4 n 760436 select 0.213 fm 0.000 coarse 0.000 other 0.787
weak-rcb-fm threads 4 n 760436 select 0.410 fm 0.343 coarse 0.000 other 0.247
weak-multi threads 4 n 760436 select 0.097 fm 0.131 coarse 0.659 other 0.113
strong-rcb threads 8 n 1520875 select 0.332 fm 0.000 coarse 0.000 other 0.668
strong-rcb-fm threads 8 n 1520875 select 0.578 fm 0.246 coarse 0.000 other 0.176
strong-multi threads 8 n 1520875 select 0.122 fm 0.080 coarse 0.722 other 0.076
weak-rcb threads 8 n 1520872 select 0.205 fm 0.000 coarse 0.000 other 0.795
weak-rcb-fm threads 8 n 1520872 select 0.494 fm 0.300 coarse 0.000 other 0.206
weak-multi threads 8 n 1520872 select 0.081 fm 0.071 coarse 0.771 other 0.077
```

```text
strong method threads select_e fm_e coarse_e
strong multi threads 2 select_e 0.090 fm_e 0.168 coarse_e 0.870
strong multi threads 4 select_e 0.365 fm_e 0.207 coarse_e 0.892
strong multi threads 8 select_e 0.440 fm_e 0.197 coarse_e 0.877
strong rcb threads 2 select_e 0.194 fm_e na coarse_e na
strong rcb threads 4 select_e 0.484 fm_e na coarse_e na
strong rcb threads 8 select_e 0.504 fm_e na coarse_e na
strong rcb-fm threads 2 select_e 0.130 fm_e 0.045 coarse_e na
strong rcb-fm threads 4 select_e 0.404 fm_e 0.219 coarse_e na
strong rcb-fm threads 8 select_e 0.484 fm_e 0.171 coarse_e na
```

## Prose

The prose pass covers the placement note, the report, the partition page, the C API page, and the README.
Two older pages keep their earlier wording:

```text
docs/explanation/mic-and-cells.md
docs/reference/algorithm.md
```

