# Cell partition

The cell graph, the partition, and the geometric halo live in linkcell.
Ring membership, the Message Passing Interface (MPI) share, and ice clustering stay in seams-core.
eOn calls `pairs_within` only. rgpot builds pairs through vesin. readcon-core reads configuration (CON) frames.

This cell-partition report is generated from one log.
Every median and every interquartile range (IQR) is the value that log printed for that cell.
The raw sample line is the series those values came from, at type-7.

The log names the hardware as Intel Xeon Processor, 8 physical cores, one thread per core, one node, 2400 MHz.
A second node is recorded as `multi_node HOST NOT-RUN`.
Commit `2bf5324d594e2e74df7f94347a05bcc71b518bdd`.
Tree `e0a089a6804e6cfdb1b9aad2b96a2a0a205324bb`.
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
morton mW-110592 20894 1.000145 0.941354 0.018990 0.000533 1
hilbert mW-110592 20694 1.000145 0.943571 0.002260 0.000093 1
rcb mW-110592 13824 1.000000 0.396092 0.002476 0.000158 1
hilbert-fm mW-110592 20594 1.000579 0.942531 0.011428 0.000181 1
rcb-fm mW-110592 13824 1.000000 0.396092 0.008595 0.000328 1
multi mW-110592 13824 1.000000 0.396092 0.026413 0.000337 1
scotch mW-110592 15547 1.000217 0.478564 0.115872 0.001062 0
zoltan-rcb mW-110592 13824 1.000000 0.396092 0.003595 0.000211 0
ptscotch mW-110592 14496 1.000000 0.430718 13.854890 0.331555 0
morton mW-884736 83857 1.000009 0.397780 0.185085 0.004763 1
hilbert mW-884736 81032 1.000018 0.418792 0.021385 0.000607 1
rcb mW-884736 55990 1.000009 0.186900 0.016444 0.000931 1
hilbert-fm mW-884736 80931 1.000072 0.418751 0.069580 0.002718 1
rcb-fm mW-884736 55802 1.000054 0.183404 0.139333 0.004838 1
multi mW-884736 55800 1.000072 0.183409 0.871422 0.014596 1
scotch mW-884736 64754 1.000054 0.232028 0.693512 0.019693 0
zoltan-rcb mW-884736 55983 1.000009 0.186892 0.029687 0.001776 0
ptscotch mW-884736 67414 1.000000 0.241669 55.206710 0.953117 0
morton TIP4P-128000 2804168 1.000000 1.687500 0.000832 0.000084 1
hilbert TIP4P-128000 2791850 1.000687 1.661710 0.000116 0.000010 1
rcb TIP4P-128000 2492438 1.000625 1.353383 0.000366 0.000138 1
hilbert-fm TIP4P-128000 2671517 1.005625 1.500000 0.001871 0.000133 1
rcb-fm TIP4P-128000 2145600 1.000000 1.000000 0.002242 0.000054 1
multi TIP4P-128000 2145600 1.000000 1.000000 0.005377 0.000937 1
scotch TIP4P-128000 2169032 1.005375 1.052632 0.015972 0.000983 0
zoltan-rcb TIP4P-128000 2485058 1.000625 1.353383 0.000286 0.000053 0
ptscotch TIP4P-128000 2130992 1.001250 1.007692 3.462330 0.175761 0
morton TIP4P-1024000 13084202 1.000320 0.812144 0.007948 0.000489 1
hilbert TIP4P-1024000 12710308 1.000195 0.848197 0.001231 0.000176 1
rcb TIP4P-1024000 9965162 1.000266 0.576619 0.001253 0.000157 1
hilbert-fm TIP4P-1024000 12192983 1.000758 0.834594 0.026031 0.000999 1
rcb-fm TIP4P-1024000 8793439 1.000766 0.466543 0.012635 0.001236 1
multi TIP4P-1024000 8800880 1.000766 0.466263 0.038897 0.004418 1
scotch TIP4P-1024000 8911827 1.000719 0.490884 0.103809 0.002759 0
zoltan-rcb TIP4P-1024000 9897435 1.000312 0.564509 0.001674 0.000185 0
ptscotch TIP4P-1024000 8825191 1.000156 0.471275 5.095990 0.245732 0
morton slab-110592 18819 1.000217 0.750051 0.039947 0.002022 1
hilbert slab-110592 18188 1.000145 0.747628 0.004835 0.000187 1
rcb slab-110592 12672 1.000000 0.500000 0.003279 0.000165 1
hilbert-fm slab-110592 18148 1.000579 0.746570 0.010029 0.000879 1
rcb-fm slab-110592 11520 1.000000 0.396092 0.010938 0.000530 1
multi slab-110592 11520 1.000000 0.396092 0.064662 0.001500 1
scotch slab-110592 12553 1.000579 0.531308 0.144783 0.003000 0
zoltan-rcb slab-110592 12672 1.000000 0.500000 0.007598 0.000207 0
ptscotch slab-110592 11739 1.000000 2.968611 24.352664 0.903492 0
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
sum method morton cut 16011940 median_s 0.252802 eligible 1
sum method hilbert cut 15622072 median_s 0.029828 eligible 1
sum method rcb cut 12540086 median_s 0.023819 eligible 1
sum method hilbert-fm cut 14984173 median_s 0.118939 eligible 1
sum method rcb-fm cut 11020185 median_s 0.173743 eligible 1
sum method multi cut 11027624 median_s 1.006771 eligible 1
default_method rcb-fm
```

## Raw samples

Each line is the twelve wall times, in seconds, in the order they were measured.

```text
morton mW-110592 0.018833 0.019451 0.018961 0.018820 0.019575 0.018913 0.018978 0.018990 0.019271 0.020598 0.018989 0.023517
hilbert mW-110592 0.002101 0.002192 0.002332 0.002326 0.002270 0.002235 0.002278 0.002192 0.002274 0.002304 0.002161 0.002249
rcb mW-110592 0.002413 0.002447 0.002539 0.002665 0.002483 0.002470 0.002916 0.002540 0.002529 0.002288 0.002192 0.002197
hilbert-fm mW-110592 0.010934 0.011526 0.011471 0.011578 0.011375 0.011280 0.011385 0.011323 0.011708 0.011370 0.011492 0.011580
rcb-fm mW-110592 0.009087 0.008368 0.008429 0.010785 0.008527 0.008805 0.008557 0.008718 0.008602 0.008589 0.008345 0.008907
multi mW-110592 0.026022 0.026987 0.026234 0.026234 0.026418 0.026339 0.026190 0.026530 0.026538 0.026671 0.026408 0.027620
scotch mW-110592 0.119737 0.121094 0.115648 0.115784 0.115786 0.116112 0.114229 0.116769 0.116519 0.115958 0.114054 0.115135
zoltan-rcb mW-110592 0.003501 0.003744 0.003521 0.003790 0.003704 0.003516 0.003462 0.004018 0.003670 0.003504 0.003445 0.003686
ptscotch mW-110592 13.518904 13.898812 16.697302 15.089048 13.995538 13.810968 14.010710 13.656229 13.630008 13.698459 13.984599 13.671625
morton mW-884736 0.191874 0.183904 0.181904 0.184290 0.182379 0.185879 0.187838 0.193147 0.183063 0.187343 0.189742 0.183714
hilbert mW-884736 0.021214 0.021787 0.021038 0.021011 0.021151 0.021430 0.021883 0.022088 0.021231 0.021648 0.021858 0.021340
rcb mW-884736 0.016323 0.016264 0.015629 0.015753 0.016407 0.018596 0.016557 0.017313 0.016152 0.017958 0.017118 0.016482
hilbert-fm mW-884736 0.070988 0.071674 0.067807 0.068649 0.067944 0.070304 0.068580 0.073798 0.068024 0.072653 0.069825 0.069334
rcb-fm mW-884736 0.141978 0.136722 0.134774 0.138163 0.158453 0.136775 0.136584 0.154730 0.136560 0.141375 0.141132 0.140502
multi mW-884736 0.894994 0.860030 0.837305 0.869672 0.862631 0.875808 0.863759 0.941735 0.859258 0.873172 0.875228 0.878881
scotch mW-884736 0.691344 0.684143 0.695680 0.684085 0.697803 0.686060 0.816613 0.729845 0.707370 0.678876 0.702638 0.673588
zoltan-rcb mW-884736 0.027964 0.028965 0.030148 0.029786 0.030900 0.029588 0.032342 0.035504 0.029433 0.029579 0.029502 0.033225
ptscotch mW-884736 55.516821 53.438639 55.144551 55.268870 56.513684 54.575710 54.818119 54.657810 55.859080 55.713713 55.783496 54.824383
morton TIP4P-128000 0.000783 0.000804 0.000798 0.000812 0.000857 0.000853 0.000919 0.000903 0.000926 0.000881 0.000810 0.000798
hilbert TIP4P-128000 0.000114 0.000114 0.000115 0.000115 0.000124 0.000115 0.000131 0.000123 0.000134 0.000125 0.000118 0.000114
rcb TIP4P-128000 0.000238 0.000432 0.000231 0.000379 0.000234 0.000344 0.000252 0.000382 0.000502 0.000383 0.000396 0.000354
hilbert-fm TIP4P-128000 0.001865 0.001827 0.001886 0.001792 0.001874 0.001769 0.002011 0.001867 0.001976 0.001950 0.001791 0.001954
rcb-fm TIP4P-128000 0.002213 0.002246 0.002188 0.002240 0.002235 0.002250 0.003304 0.002244 0.002434 0.002293 0.002180 0.002144
multi TIP4P-128000 0.004951 0.005296 0.006826 0.006192 0.005061 0.004967 0.006945 0.005096 0.005969 0.005201 0.005459 0.005479
scotch TIP4P-128000 0.015913 0.014199 0.015200 0.015943 0.016000 0.017723 0.018697 0.016064 0.017806 0.015876 0.016346 0.015125
zoltan-rcb TIP4P-128000 0.000240 0.000344 0.000342 0.000266 0.000311 0.000426 0.000300 0.000268 0.000255 0.000266 0.000289 0.000283
ptscotch TIP4P-128000 3.433336 3.529756 3.347508 3.253304 3.619106 3.413705 3.358117 3.510107 3.535635 3.304911 3.491324 3.605398
morton TIP4P-1024000 0.008213 0.008010 0.007808 0.008155 0.008532 0.007887 0.007793 0.014045 0.007871 0.007728 0.007699 0.008635
hilbert TIP4P-1024000 0.001237 0.001152 0.001132 0.001352 0.001226 0.001167 0.001142 0.001647 0.001326 0.001145 0.001256 0.001326
rcb TIP4P-1024000 0.001356 0.001381 0.001265 0.001171 0.001081 0.001556 0.001174 0.001346 0.001237 0.001216 0.001395 0.001242
hilbert-fm TIP4P-1024000 0.025924 0.024770 0.026435 0.026638 0.026138 0.026342 0.024779 0.037252 0.025893 0.024489 0.025722 0.029819
rcb-fm TIP4P-1024000 0.012675 0.012635 0.012163 0.013100 0.012365 0.012634 0.012111 0.015440 0.012050 0.014789 0.012103 0.014244
multi TIP4P-1024000 0.037694 0.038138 0.037441 0.041536 0.037707 0.040880 0.038294 0.054328 0.039500 0.051423 0.037295 0.043879
scotch TIP4P-1024000 0.103913 0.105560 0.103049 0.109442 0.105252 0.102563 0.103510 0.106065 0.100244 0.100198 0.103705 0.117703
zoltan-rcb TIP4P-1024000 0.001638 0.001635 0.001727 0.001548 0.001971 0.001641 0.001561 0.001708 0.001806 0.001589 0.001813 0.001873
ptscotch TIP4P-1024000 5.062095 5.047774 5.097448 5.223077 5.094532 5.411930 4.964509 5.267978 4.949527 5.499996 5.286826 4.801973
morton slab-110592 0.039405 0.040136 0.041907 0.039654 0.039758 0.040390 0.039419 0.041388 0.039480 0.042512 0.039501 0.049156
hilbert slab-110592 0.004673 0.004897 0.004924 0.004693 0.004840 0.004760 0.004831 0.005233 0.004702 0.004739 0.004915 0.005955
rcb slab-110592 0.003114 0.003237 0.003330 0.003888 0.003035 0.003350 0.003208 0.003410 0.003213 0.003174 0.003614 0.003320
hilbert-fm slab-110592 0.009578 0.010530 0.010687 0.010687 0.009613 0.009803 0.009744 0.010256 0.009549 0.009716 0.010304 0.011312
rcb-fm slab-110592 0.011288 0.010962 0.010779 0.013670 0.010826 0.010626 0.010741 0.011514 0.010915 0.010865 0.011199 0.012922
multi slab-110592 0.065003 0.064158 0.065198 0.067018 0.064138 0.063981 0.064320 0.069182 0.063799 0.064283 0.065162 0.070185
scotch slab-110592 0.144099 0.143348 0.145394 0.144449 0.143329 0.145861 0.143873 0.149721 0.143684 0.145117 0.150627 0.171268
zoltan-rcb slab-110592 0.007949 0.007415 0.007740 0.007769 0.007599 0.007562 0.007313 0.007474 0.007596 0.007625 0.007575 0.008004
ptscotch slab-110592 23.695273 24.107425 24.879178 23.562598 24.849050 24.008549 24.656004 24.413364 24.291965 23.786715 24.983844 24.979955
```

## Balance sweep

The same five ratios appear on every system in this sweep.
The main table uses the derived bound.

```text
morton 1.00 mW-110592 cut 20894 imbalance 1.000145 median 0.018866 iqr 0.000108
hilbert 1.00 mW-110592 cut 20694 imbalance 1.000145 median 0.002082 iqr 0.000076
rcb 1.00 mW-110592 cut 13824 imbalance 1.000000 median 0.002313 iqr 0.000140
hilbert-fm 1.00 mW-110592 cut 20692 imbalance 1.000000 median 0.005267 iqr 0.000157
rcb-fm 1.00 mW-110592 cut 13824 imbalance 1.000000 median 0.008056 iqr 0.000496
multi 1.00 mW-110592 cut 13824 imbalance 1.000000 median 0.025352 iqr 0.000813
scotch 1.00 mW-110592 cut 20801 imbalance 1.000000 median 0.113534 iqr 0.001003
morton 1.01 mW-110592 cut 20894 imbalance 1.000145 median 0.018770 iqr 0.000069
hilbert 1.01 mW-110592 cut 20694 imbalance 1.000145 median 0.002064 iqr 0.000066
rcb 1.01 mW-110592 cut 13824 imbalance 1.000000 median 0.002257 iqr 0.000118
hilbert-fm 1.01 mW-110592 cut 20584 imbalance 1.002604 median 0.005187 iqr 0.000090
rcb-fm 1.01 mW-110592 cut 13824 imbalance 1.000000 median 0.008337 iqr 0.000153
multi 1.01 mW-110592 cut 13824 imbalance 1.000000 median 0.026245 iqr 0.000125
scotch 1.01 mW-110592 cut 14698 imbalance 1.009910 median 0.113687 iqr 0.000985
morton 1.02 mW-110592 cut 20894 imbalance 1.000145 median 0.018792 iqr 0.000116
hilbert 1.02 mW-110592 cut 20694 imbalance 1.000145 median 0.002060 iqr 0.000070
rcb 1.02 mW-110592 cut 13824 imbalance 1.000000 median 0.002195 iqr 0.000333
hilbert-fm 1.02 mW-110592 cut 20584 imbalance 1.002604 median 0.005244 iqr 0.000108
rcb-fm 1.02 mW-110592 cut 13824 imbalance 1.000000 median 0.008423 iqr 0.000105
multi 1.02 mW-110592 cut 13824 imbalance 1.000000 median 0.026552 iqr 0.000371
scotch 1.02 mW-110592 cut 14391 imbalance 1.019676 median 0.098743 iqr 0.001102
morton 1.05 mW-110592 cut 20894 imbalance 1.000145 median 0.018991 iqr 0.000228
hilbert 1.05 mW-110592 cut 20694 imbalance 1.000145 median 0.002061 iqr 0.000061
rcb 1.05 mW-110592 cut 13824 imbalance 1.000000 median 0.002243 iqr 0.000215
hilbert-fm 1.05 mW-110592 cut 20584 imbalance 1.002604 median 0.005262 iqr 0.000205
rcb-fm 1.05 mW-110592 cut 13824 imbalance 1.000000 median 0.008485 iqr 0.000223
multi 1.05 mW-110592 cut 13824 imbalance 1.000000 median 0.026697 iqr 0.000253
scotch 1.05 mW-110592 cut 14817 imbalance 1.045139 median 0.113075 iqr 0.001007
morton 1.10 mW-110592 cut 20894 imbalance 1.000145 median 0.018742 iqr 0.000147
hilbert 1.10 mW-110592 cut 20694 imbalance 1.000145 median 0.002086 iqr 0.000122
rcb 1.10 mW-110592 cut 13824 imbalance 1.000000 median 0.002297 iqr 0.000198
hilbert-fm 1.10 mW-110592 cut 20584 imbalance 1.002604 median 0.005206 iqr 0.000072
rcb-fm 1.10 mW-110592 cut 13824 imbalance 1.000000 median 0.008367 iqr 0.000234
multi 1.10 mW-110592 cut 13824 imbalance 1.000000 median 0.026769 iqr 0.000271
scotch 1.10 mW-110592 cut 15098 imbalance 1.099826 median 0.115422 iqr 0.000765
morton 1.00 mW-884736 cut 83857 imbalance 1.000009 median 0.185303 iqr 0.005003
hilbert 1.00 mW-884736 cut 81032 imbalance 1.000018 median 0.021027 iqr 0.000295
rcb 1.00 mW-884736 cut 55990 imbalance 1.000009 median 0.015884 iqr 0.000668
hilbert-fm 1.00 mW-884736 cut 81032 imbalance 1.000000 median 0.040623 iqr 0.000912
rcb-fm 1.00 mW-884736 cut 55990 imbalance 1.000009 median 0.099439 iqr 0.001120
multi 1.00 mW-884736 cut 55990 imbalance 1.000009 median 0.278099 iqr 0.005045
scotch 1.00 mW-884736 cut 81445 imbalance 1.000000 median 0.645650 iqr 0.023425
morton 1.01 mW-884736 cut 83857 imbalance 1.000009 median 0.186192 iqr 0.003419
hilbert 1.01 mW-884736 cut 81032 imbalance 1.000018 median 0.021071 iqr 0.000743
rcb 1.01 mW-884736 cut 55990 imbalance 1.000009 median 0.016174 iqr 0.000765
hilbert-fm 1.01 mW-884736 cut 80799 imbalance 1.004792 median 0.239452 iqr 0.010712
rcb-fm 1.01 mW-884736 cut 55894 imbalance 1.000868 median 0.086959 iqr 0.006283
multi 1.01 mW-884736 cut 55837 imbalance 1.000922 median 0.498067 iqr 0.030937
scotch 1.01 mW-884736 cut 58097 imbalance 1.009865 median 0.615935 iqr 0.034345
morton 1.02 mW-884736 cut 83857 imbalance 1.000009 median 0.186549 iqr 0.010046
hilbert 1.02 mW-884736 cut 81032 imbalance 1.000018 median 0.021154 iqr 0.000477
rcb 1.02 mW-884736 cut 55990 imbalance 1.000009 median 0.016362 iqr 0.001774
hilbert-fm 1.02 mW-884736 cut 80799 imbalance 1.004792 median 0.239462 iqr 0.006982
rcb-fm 1.02 mW-884736 cut 55894 imbalance 1.000868 median 0.087658 iqr 0.006518
multi 1.02 mW-884736 cut 55837 imbalance 1.000922 median 0.502358 iqr 0.030825
scotch 1.02 mW-884736 cut 60287 imbalance 1.019938 median 0.645597 iqr 0.042785
morton 1.05 mW-884736 cut 83857 imbalance 1.000009 median 0.184365 iqr 0.004183
hilbert 1.05 mW-884736 cut 81032 imbalance 1.000018 median 0.021214 iqr 0.000450
rcb 1.05 mW-884736 cut 55990 imbalance 1.000009 median 0.016049 iqr 0.000376
hilbert-fm 1.05 mW-884736 cut 80799 imbalance 1.004792 median 0.236270 iqr 0.007964
rcb-fm 1.05 mW-884736 cut 55894 imbalance 1.000868 median 0.084966 iqr 0.003119
multi 1.05 mW-884736 cut 55837 imbalance 1.000922 median 0.478606 iqr 0.019964
scotch 1.05 mW-884736 cut 62348 imbalance 1.018971 median 0.619705 iqr 0.011138
morton 1.10 mW-884736 cut 83857 imbalance 1.000009 median 0.200678 iqr 0.018327
hilbert 1.10 mW-884736 cut 81032 imbalance 1.000018 median 0.022153 iqr 0.002179
rcb 1.10 mW-884736 cut 55990 imbalance 1.000009 median 0.017089 iqr 0.001298
hilbert-fm 1.10 mW-884736 cut 80799 imbalance 1.004792 median 0.254162 iqr 0.028011
rcb-fm 1.10 mW-884736 cut 55894 imbalance 1.000868 median 0.094910 iqr 0.011493
multi 1.10 mW-884736 cut 55837 imbalance 1.000922 median 0.532290 iqr 0.072303
scotch 1.10 mW-884736 cut 59853 imbalance 1.086842 median 0.673434 iqr 0.079310
morton 1.00 TIP4P-128000 cut 2804168 imbalance 1.000000 median 0.000828 iqr 0.000054
hilbert 1.00 TIP4P-128000 cut 2791850 imbalance 1.000687 median 0.000117 iqr 0.000012
rcb 1.00 TIP4P-128000 cut 2492438 imbalance 1.000625 median 0.000340 iqr 0.000029
hilbert-fm 1.00 TIP4P-128000 cut 2791850 imbalance 1.000687 median 0.000291 iqr 0.000023
rcb-fm 1.00 TIP4P-128000 cut 2145600 imbalance 1.000000 median 0.000903 iqr 0.000235
multi 1.00 TIP4P-128000 cut 2145600 imbalance 1.000000 median 0.003676 iqr 0.000211
scotch 1.00 TIP4P-128000 cut 2776979 imbalance 1.000687 median 0.009434 iqr 0.000600
morton 1.01 TIP4P-128000 cut 2804168 imbalance 1.000000 median 0.000778 iqr 0.000042
hilbert 1.01 TIP4P-128000 cut 2791850 imbalance 1.000687 median 0.000112 iqr 0.000010
rcb 1.01 TIP4P-128000 cut 2492438 imbalance 1.000625 median 0.000364 iqr 0.000051
hilbert-fm 1.01 TIP4P-128000 cut 2668242 imbalance 1.005250 median 0.002308 iqr 0.000113
rcb-fm 1.01 TIP4P-128000 cut 2130992 imbalance 1.001250 median 0.004931 iqr 0.000222
multi 1.01 TIP4P-128000 cut 2130992 imbalance 1.001250 median 0.008030 iqr 0.000143
scotch 1.01 TIP4P-128000 cut 2172264 imbalance 1.008563 median 0.015692 iqr 0.000564
morton 1.02 TIP4P-128000 cut 2804168 imbalance 1.000000 median 0.000791 iqr 0.000035
hilbert 1.02 TIP4P-128000 cut 2791850 imbalance 1.000687 median 0.000111 iqr 0.000004
rcb 1.02 TIP4P-128000 cut 2492438 imbalance 1.000625 median 0.000350 iqr 0.000048
hilbert-fm 1.02 TIP4P-128000 cut 2667911 imbalance 1.016563 median 0.001463 iqr 0.000055
rcb-fm 1.02 TIP4P-128000 cut 2145600 imbalance 1.000000 median 0.002776 iqr 0.000182
multi 1.02 TIP4P-128000 cut 2145600 imbalance 1.000000 median 0.006115 iqr 0.000290
scotch 1.02 TIP4P-128000 cut 2196133 imbalance 1.019062 median 0.015413 iqr 0.000793
morton 1.05 TIP4P-128000 cut 2804168 imbalance 1.000000 median 0.000823 iqr 0.000090
hilbert 1.05 TIP4P-128000 cut 2791850 imbalance 1.000687 median 0.000121 iqr 0.000015
rcb 1.05 TIP4P-128000 cut 2492438 imbalance 1.000625 median 0.000372 iqr 0.000204
hilbert-fm 1.05 TIP4P-128000 cut 2658326 imbalance 1.049063 median 0.001652 iqr 0.000098
rcb-fm 1.05 TIP4P-128000 cut 2120992 imbalance 1.050000 median 0.004893 iqr 0.000238
multi 1.05 TIP4P-128000 cut 2120512 imbalance 1.050000 median 0.017763 iqr 0.000509
scotch 1.05 TIP4P-128000 cut 2155060 imbalance 1.050000 median 0.014926 iqr 0.000978
morton 1.10 TIP4P-128000 cut 2804168 imbalance 1.000000 median 0.000813 iqr 0.000095
hilbert 1.10 TIP4P-128000 cut 2791850 imbalance 1.000687 median 0.000117 iqr 0.000007
rcb 1.10 TIP4P-128000 cut 2492438 imbalance 1.000625 median 0.000237 iqr 0.000141
hilbert-fm 1.10 TIP4P-128000 cut 2654022 imbalance 1.096750 median 0.001701 iqr 0.000063
rcb-fm 1.10 TIP4P-128000 cut 2092800 imbalance 1.062500 median 0.005239 iqr 0.000477
multi 1.10 TIP4P-128000 cut 2092800 imbalance 1.062500 median 0.008998 iqr 0.000732
scotch 1.10 TIP4P-128000 cut 2118900 imbalance 1.089063 median 0.014527 iqr 0.001242
morton 1.00 TIP4P-1024000 cut 13084202 imbalance 1.000320 median 0.007873 iqr 0.000319
hilbert 1.00 TIP4P-1024000 cut 12710308 imbalance 1.000195 median 0.001113 iqr 0.000093
rcb 1.00 TIP4P-1024000 cut 9965162 imbalance 1.000266 median 0.001202 iqr 0.000067
hilbert-fm 1.00 TIP4P-1024000 cut 12710308 imbalance 1.000195 median 0.002262 iqr 0.000112
rcb-fm 1.00 TIP4P-1024000 cut 9965162 imbalance 1.000266 median 0.006482 iqr 0.000340
multi 1.00 TIP4P-1024000 cut 9965162 imbalance 1.000266 median 0.031291 iqr 0.001427
scotch 1.00 TIP4P-1024000 cut 15041835 imbalance 1.000133 median 0.084993 iqr 0.002121
morton 1.01 TIP4P-1024000 cut 13084202 imbalance 1.000320 median 0.008157 iqr 0.000907
hilbert 1.01 TIP4P-1024000 cut 12710308 imbalance 1.000195 median 0.001160 iqr 0.000241
rcb 1.01 TIP4P-1024000 cut 9965162 imbalance 1.000266 median 0.001192 iqr 0.000146
hilbert-fm 1.01 TIP4P-1024000 cut 12126380 imbalance 1.010000 median 0.021780 iqr 0.002593
rcb-fm 1.01 TIP4P-1024000 cut 8741847 imbalance 1.009797 median 0.015998 iqr 0.001111
multi 1.01 TIP4P-1024000 cut 8762734 imbalance 1.009813 median 0.045622 iqr 0.003490
scotch 1.01 TIP4P-1024000 cut 8949479 imbalance 1.009922 median 0.116393 iqr 0.008520
morton 1.02 TIP4P-1024000 cut 13084202 imbalance 1.000320 median 0.007844 iqr 0.000128
hilbert 1.02 TIP4P-1024000 cut 12710308 imbalance 1.000195 median 0.001150 iqr 0.000128
rcb 1.02 TIP4P-1024000 cut 9965162 imbalance 1.000266 median 0.001167 iqr 0.000114
hilbert-fm 1.02 TIP4P-1024000 cut 12078105 imbalance 1.019977 median 0.008554 iqr 0.000335
rcb-fm 1.02 TIP4P-1024000 cut 8726026 imbalance 1.019969 median 0.015250 iqr 0.001078
multi 1.02 TIP4P-1024000 cut 8760713 imbalance 1.019969 median 0.046088 iqr 0.000950
scotch 1.02 TIP4P-1024000 cut 8861660 imbalance 1.020000 median 0.107732 iqr 0.001184
morton 1.05 TIP4P-1024000 cut 13084202 imbalance 1.000320 median 0.007869 iqr 0.000240
hilbert 1.05 TIP4P-1024000 cut 12710308 imbalance 1.000195 median 0.001105 iqr 0.000071
rcb 1.05 TIP4P-1024000 cut 9965162 imbalance 1.000266 median 0.001164 iqr 0.000152
hilbert-fm 1.05 TIP4P-1024000 cut 11893822 imbalance 1.050000 median 0.013033 iqr 0.000649
rcb-fm 1.05 TIP4P-1024000 cut 8743808 imbalance 1.050000 median 0.028124 iqr 0.002748
multi 1.05 TIP4P-1024000 cut 8746824 imbalance 1.050000 median 0.055698 iqr 0.002332
scotch 1.05 TIP4P-1024000 cut 8730300 imbalance 1.028125 median 0.099730 iqr 0.003240
morton 1.10 TIP4P-1024000 cut 13084202 imbalance 1.000320 median 0.007711 iqr 0.000211
hilbert 1.10 TIP4P-1024000 cut 12710308 imbalance 1.000195 median 0.001128 iqr 0.000038
rcb 1.10 TIP4P-1024000 cut 9965162 imbalance 1.000266 median 0.001133 iqr 0.000125
hilbert-fm 1.10 TIP4P-1024000 cut 11733490 imbalance 1.099828 median 0.015107 iqr 0.000286
rcb-fm 1.10 TIP4P-1024000 cut 8530800 imbalance 1.100000 median 0.014742 iqr 0.000742
multi 1.10 TIP4P-1024000 cut 8523300 imbalance 1.100000 median 0.070832 iqr 0.003888
scotch 1.10 TIP4P-1024000 cut 8812500 imbalance 1.058594 median 0.094582 iqr 0.002343
morton 1.00 slab-110592 cut 18819 imbalance 1.000217 median 0.043092 iqr 0.001899
hilbert 1.00 slab-110592 cut 18188 imbalance 1.000145 median 0.005142 iqr 0.000130
rcb 1.00 slab-110592 cut 12672 imbalance 1.000000 median 0.003054 iqr 0.000155
hilbert-fm 1.00 slab-110592 cut 18200 imbalance 1.000000 median 0.007638 iqr 0.000279
rcb-fm 1.00 slab-110592 cut 11520 imbalance 1.000000 median 0.011256 iqr 0.000826
multi 1.00 slab-110592 cut 11520 imbalance 1.000000 median 0.070330 iqr 0.001000
scotch 1.00 slab-110592 cut 16596 imbalance 1.000000 median 0.160157 iqr 0.002884
morton 1.01 slab-110592 cut 18819 imbalance 1.000217 median 0.043431 iqr 0.002550
hilbert 1.01 slab-110592 cut 18188 imbalance 1.000145 median 0.005136 iqr 0.000243
rcb 1.01 slab-110592 cut 12672 imbalance 1.000000 median 0.003156 iqr 0.000246
hilbert-fm 1.01 slab-110592 cut 18078 imbalance 1.009404 median 0.047788 iqr 0.002614
rcb-fm 1.01 slab-110592 cut 11520 imbalance 1.000000 median 0.011753 iqr 0.000362
multi 1.01 slab-110592 cut 11520 imbalance 1.000000 median 0.073095 iqr 0.003359
scotch 1.01 slab-110592 cut 11839 imbalance 1.009549 median 0.129141 iqr 0.007650
morton 1.02 slab-110592 cut 18819 imbalance 1.000217 median 0.043368 iqr 0.003524
hilbert 1.02 slab-110592 cut 18188 imbalance 1.000145 median 0.005030 iqr 0.000302
rcb 1.02 slab-110592 cut 12672 imbalance 1.000000 median 0.003106 iqr 0.000141
hilbert-fm 1.02 slab-110592 cut 18065 imbalance 1.019748 median 0.062206 iqr 0.001791
rcb-fm 1.02 slab-110592 cut 11520 imbalance 1.000000 median 0.011640 iqr 0.000442
multi 1.02 slab-110592 cut 11520 imbalance 1.000000 median 0.070898 iqr 0.002072
scotch 1.02 slab-110592 cut 12384 imbalance 1.018953 median 0.139963 iqr 0.021311
morton 1.05 slab-110592 cut 18819 imbalance 1.000217 median 0.041572 iqr 0.000651
hilbert 1.05 slab-110592 cut 18188 imbalance 1.000145 median 0.004787 iqr 0.000307
rcb 1.05 slab-110592 cut 12672 imbalance 1.000000 median 0.002978 iqr 0.000232
hilbert-fm 1.05 slab-110592 cut 18012 imbalance 1.049986 median 0.104221 iqr 0.002773
rcb-fm 1.05 slab-110592 cut 11520 imbalance 1.000000 median 0.011426 iqr 0.000830
multi 1.05 slab-110592 cut 11520 imbalance 1.000000 median 0.067886 iqr 0.001489
scotch 1.05 slab-110592 cut 11911 imbalance 1.049986 median 0.123891 iqr 0.004038
morton 1.10 slab-110592 cut 18819 imbalance 1.000217 median 0.042471 iqr 0.000537
hilbert 1.10 slab-110592 cut 18188 imbalance 1.000145 median 0.004991 iqr 0.000230
rcb 1.10 slab-110592 cut 12672 imbalance 1.000000 median 0.003165 iqr 0.000187
hilbert-fm 1.10 slab-110592 cut 18008 imbalance 1.053168 median 0.113434 iqr 0.001877
rcb-fm 1.10 slab-110592 cut 11520 imbalance 1.000000 median 0.011709 iqr 0.000418
multi 1.10 slab-110592 cut 11520 imbalance 1.000000 median 0.070699 iqr 0.001920
scotch 1.10 slab-110592 cut 11938 imbalance 1.094184 median 0.128201 iqr 0.002643
```

## Scaling

Strong scaling keeps the lattice at 1,520,875 points (115 cubed).
Weak scaling uses 190,109 points times the thread count, so eight threads is 1,520,872 points.
Median and interquartile range (IQR) are recomputed from the raw line.
`profile_ns` sums the critical-path wall time of select, boundary refinement, and coarsening over the repetitions, in nanoseconds.

```text
kind threads n method median iqr
strong 1 1520875 rcb 0.027934 0.002018 9f5cf54822c49335
strong 1 1520875 rcb-fm 0.190682 0.013140 29bd56d7602a754c
strong 1 1520875 multi 0.572839 0.041307 792dbf6957d1b8c5
weak 1 190109 rcb 0.010789 0.003756 a6ef469b3dc6d7e9
weak 1 190109 rcb-fm 0.043010 0.021078 a6ef469b3dc6d7e9
weak 1 190109 multi 0.140870 0.038868 a6ef469b3dc6d7e9
strong 2 1520875 rcb 0.021450 0.000898 9f5cf54822c49335
strong 2 1520875 rcb-fm 0.124009 0.013141 29bd56d7602a754c
strong 2 1520875 multi 0.515633 0.048766 792dbf6957d1b8c5
weak 2 380218 rcb 0.002951 0.000390 263ada9bf59665cd
weak 2 380218 rcb-fm 0.010806 0.000796 263ada9bf59665cd
weak 2 380218 multi 0.045123 0.001449 263ada9bf59665cd
strong 4 1520875 rcb 0.019949 0.001825 9f5cf54822c49335
strong 4 1520875 rcb-fm 0.104158 0.005460 29bd56d7602a754c
strong 4 1520875 multi 0.451513 0.023976 792dbf6957d1b8c5
weak 4 760436 rcb 0.005650 0.000273 c50aded0e71cdd51
weak 4 760436 rcb-fm 0.022230 0.001215 f6b34d1d05faa473
weak 4 760436 multi 0.084849 0.002790 f6b34d1d05faa473
strong 8 1520875 rcb 0.020026 0.001999 9f5cf54822c49335
strong 8 1520875 rcb-fm 0.100512 0.006690 29bd56d7602a754c
strong 8 1520875 multi 0.463451 0.024893 792dbf6957d1b8c5
weak 8 1520872 rcb 0.015961 0.001439 eef7430cb40dfd70
weak 8 1520872 rcb-fm 0.071370 0.008611 ac62a33a608f3394
weak 8 1520872 multi 0.362170 0.021438 ac62a33a608f3394
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
strong-rcb threads 1 n 1520875 select 0.372 fm 0.000 coarse 0.000 other 0.628
strong-rcb-fm threads 1 n 1520875 select 0.481 fm 0.399 coarse 0.000 other 0.120
strong-multi threads 1 n 1520875 select 0.160 fm 0.194 coarse 0.570 other 0.076
weak-rcb threads 1 n 190109 select 0.241 fm 0.000 coarse 0.000 other 0.759
weak-rcb-fm threads 1 n 190109 select 0.469 fm 0.332 coarse 0.000 other 0.198
weak-multi threads 1 n 190109 select 0.149 fm 0.139 coarse 0.599 other 0.113
strong-rcb threads 2 n 1520875 select 0.288 fm 0.000 coarse 0.000 other 0.712
strong-rcb-fm threads 2 n 1520875 select 0.435 fm 0.387 coarse 0.000 other 0.179
strong-multi threads 2 n 1520875 select 0.123 fm 0.143 coarse 0.657 other 0.077
weak-rcb threads 2 n 380218 select 0.204 fm 0.000 coarse 0.000 other 0.796
weak-rcb-fm threads 2 n 380218 select 0.347 fm 0.338 coarse 0.000 other 0.315
weak-multi threads 2 n 380218 select 0.080 fm 0.127 coarse 0.679 other 0.114
strong-rcb threads 4 n 1520875 select 0.318 fm 0.000 coarse 0.000 other 0.682
strong-rcb-fm threads 4 n 1520875 select 0.521 fm 0.310 coarse 0.000 other 0.168
strong-multi threads 4 n 1520875 select 0.120 fm 0.105 coarse 0.700 other 0.074
weak-rcb threads 4 n 760436 select 0.199 fm 0.000 coarse 0.000 other 0.801
weak-rcb-fm threads 4 n 760436 select 0.400 fm 0.356 coarse 0.000 other 0.244
weak-multi threads 4 n 760436 select 0.099 fm 0.133 coarse 0.663 other 0.106
strong-rcb threads 8 n 1520875 select 0.312 fm 0.000 coarse 0.000 other 0.688
strong-rcb-fm threads 8 n 1520875 select 0.577 fm 0.241 coarse 0.000 other 0.182
strong-multi threads 8 n 1520875 select 0.121 fm 0.080 coarse 0.723 other 0.076
weak-rcb threads 8 n 1520872 select 0.218 fm 0.000 coarse 0.000 other 0.782
weak-rcb-fm threads 8 n 1520872 select 0.513 fm 0.280 coarse 0.000 other 0.207
weak-multi threads 8 n 1520872 select 0.091 fm 0.067 coarse 0.767 other 0.076
```

```text
strong method threads select_e fm_e coarse_e
strong multi threads 2 select_e 0.457 fm_e 0.404 coarse_e 1.186
strong multi threads 4 select_e 0.464 fm_e 0.243 coarse_e 0.970
strong multi threads 8 select_e 0.567 fm_e 0.244 coarse_e 1.046
strong rcb threads 2 select_e 0.187 fm_e na coarse_e na
strong rcb threads 4 select_e 0.472 fm_e na coarse_e na
strong rcb threads 8 select_e 0.553 fm_e na coarse_e na
strong rcb-fm threads 2 select_e 0.271 fm_e 0.364 coarse_e na
strong rcb-fm threads 4 select_e 0.458 fm_e 0.236 coarse_e na
strong rcb-fm threads 8 select_e 0.566 fm_e 0.215 coarse_e na
```

## Prose

The prose pass covers the placement note, the report, the partition page, the C API page, and the README.
Two older pages keep their earlier wording:

```text
docs/explanation/mic-and-cells.md
docs/reference/algorithm.md
```

