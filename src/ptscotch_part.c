/* PT-Scotch entry. The library is not vendored.
 * MPI must already be initialized. This file does not call MPI_Init.
 * The standalone driver (LC_PTSCOTCH_MAIN) does. SCOTCH_Num is int32. */

#include <ptscotch.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

_Static_assert(sizeof(SCOTCH_Num) == sizeof(int32_t), "SCOTCH_Num");
_Static_assert(sizeof(int) == sizeof(int32_t), "MPI_INT");

static void die(int rank, const char *msg) {
  fprintf(stderr, "ptscotch rank %d: %s\n", rank, msg);
  fflush(stderr);
  MPI_Abort(MPI_COMM_WORLD, 1);
}

/* Collective. Arrays are meaningful on rank 0. part is filled on rank 0. */
int lc_ptscotch_part(int n, const int32_t *offsets, const int32_t *edges,
                     const int32_t *vwgt, const int32_t *ewgt, int parts,
                     double imbalance, int32_t *part, double *seconds_max) {
  int ready = 0;
  MPI_Initialized(&ready);
  if (!ready) {
    return 2;
  }
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int32_t header[2];
  header[0] = n;
  header[1] = parts;
  MPI_Bcast(header, 2, MPI_INT, 0, MPI_COMM_WORLD);
  n = header[0];
  parts = header[1];
  if (n < 1 || parts < 1) {
    die(rank, "empty graph");
  }

  int32_t *off = NULL;
  int32_t *edg = NULL;
  int32_t *vw = NULL;
  int32_t *ew = NULL;
  int nnz = 0;
  if (rank == 0) {
    if (offsets == NULL || edges == NULL || vwgt == NULL || ewgt == NULL ||
        part == NULL) {
      die(rank, "null array");
    }
    nnz = offsets[n];
    off = (int32_t *)offsets;
    edg = (int32_t *)edges;
    vw = (int32_t *)vwgt;
    ew = (int32_t *)ewgt;
  }
  MPI_Bcast(&nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (nnz < 0) {
    die(rank, "bad edge count");
  }
  if (rank != 0) {
    off = (int32_t *)calloc((size_t)n + 1, sizeof(int32_t));
    edg = (int32_t *)calloc(nnz ? (size_t)nnz : 1, sizeof(int32_t));
    vw = (int32_t *)calloc((size_t)n, sizeof(int32_t));
    ew = (int32_t *)calloc(nnz ? (size_t)nnz : 1, sizeof(int32_t));
    if (off == NULL || edg == NULL || vw == NULL || ew == NULL) {
      die(rank, "alloc");
    }
  }
  MPI_Bcast(off, n + 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (nnz > 0) {
    MPI_Bcast(edg, nnz, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(ew, nnz, MPI_INT, 0, MPI_COMM_WORLD);
  }
  MPI_Bcast(vw, n, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&imbalance, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  int base = n / size;
  int extra = n % size;
  int loc = base + (rank < extra ? 1 : 0);
  int start = rank * base + (rank < extra ? rank : extra);
  int loc_nnz = 0;
  if (loc > 0) {
    loc_nnz = off[start + loc] - off[start];
  }

  SCOTCH_Num *vert = (SCOTCH_Num *)calloc((size_t)loc + 1, sizeof(SCOTCH_Num));
  SCOTCH_Num *edge =
      (SCOTCH_Num *)calloc(loc_nnz ? (size_t)loc_nnz : 1, sizeof(SCOTCH_Num));
  SCOTCH_Num *vel =
      (SCOTCH_Num *)calloc(loc ? (size_t)loc : 1, sizeof(SCOTCH_Num));
  SCOTCH_Num *edl =
      (SCOTCH_Num *)calloc(loc_nnz ? (size_t)loc_nnz : 1, sizeof(SCOTCH_Num));
  SCOTCH_Num *out =
      (SCOTCH_Num *)calloc(loc ? (size_t)loc : 1, sizeof(SCOTCH_Num));
  if (vert == NULL || edge == NULL || vel == NULL || edl == NULL || out == NULL) {
    die(rank, "local alloc");
  }
  vert[0] = 0;
  for (int i = 0; i < loc; ++i) {
    int g = start + i;
    int deg = off[g + 1] - off[g];
    vert[i + 1] = vert[i] + deg;
    vel[i] = vw[g];
    for (int e = 0; e < deg; ++e) {
      edge[vert[i] + e] = edg[off[g] + e];
      edl[vert[i] + e] = ew[off[g] + e];
    }
  }

  SCOTCH_Dgraph dgraph;
  SCOTCH_Strat strat;
  int d_ready = 0;
  int s_ready = 0;
  if (SCOTCH_dgraphInit(&dgraph, MPI_COMM_WORLD) != 0) {
    die(rank, "dgraphInit");
  }
  d_ready = 1;
  if (SCOTCH_dgraphBuild(&dgraph, 0, (SCOTCH_Num)loc, (SCOTCH_Num)loc, vert,
                         NULL, vel, NULL, (SCOTCH_Num)loc_nnz,
                         (SCOTCH_Num)loc_nnz, edge, NULL, edl) != 0) {
    die(rank, "dgraphBuild");
  }
  if (SCOTCH_dgraphCheck(&dgraph) != 0) {
    die(rank, "dgraphCheck");
  }
  SCOTCH_randomReset();
  SCOTCH_randomSeed(1);
  if (SCOTCH_stratInit(&strat) != 0) {
    die(rank, "stratInit");
  }
  s_ready = 1;
  if (SCOTCH_stratDgraphMapBuild(&strat, SCOTCH_STRATQUALITY | SCOTCH_STRATBALANCE,
                                 (SCOTCH_Num)size, (SCOTCH_Num)parts,
                                 imbalance) != 0) {
    die(rank, "stratDgraphMapBuild");
  }

  MPI_Barrier(MPI_COMM_WORLD);
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  int rc = SCOTCH_dgraphPart(&dgraph, (SCOTCH_Num)parts, &strat, out);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  double local = (double)(t1.tv_sec - t0.tv_sec) +
                 (double)(t1.tv_nsec - t0.tv_nsec) / 1.0e9;
  double reduced = 0.0;
  MPI_Reduce(&local, &reduced, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (rc != 0) {
    die(rank, "dgraphPart");
  }
  if (rank == 0 && seconds_max != NULL) {
    *seconds_max = reduced;
  }

  int *counts = NULL;
  int *displs = NULL;
  if (rank == 0) {
    counts = (int *)calloc((size_t)size, sizeof(int));
    displs = (int *)calloc((size_t)size, sizeof(int));
    if (counts == NULL || displs == NULL) {
      die(rank, "gatherv alloc");
    }
    for (int r = 0; r < size; ++r) {
      counts[r] = base + (r < extra ? 1 : 0);
      displs[r] = r * base + (r < extra ? r : extra);
    }
  }
  MPI_Gatherv(out, loc, MPI_INT, part, counts, displs, MPI_INT, 0,
              MPI_COMM_WORLD);

  if (s_ready) {
    SCOTCH_stratExit(&strat);
  }
  if (d_ready) {
    SCOTCH_dgraphExit(&dgraph);
  }
  free(vert);
  free(edge);
  free(vel);
  free(edl);
  free(out);
  free(counts);
  free(displs);
  if (rank != 0) {
    free(off);
    free(edg);
    free(vw);
    free(ew);
  }
  return 0;
}

#ifdef LC_PTSCOTCH_MAIN

static int32_t *slurp(FILE *in, size_t count) {
  int32_t *buf = (int32_t *)calloc(count ? count : 1, sizeof(int32_t));
  if (buf == NULL) {
    return NULL;
  }
  if (count > 0 && fread(buf, sizeof(int32_t), count, in) != count) {
    free(buf);
    return NULL;
  }
  return buf;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (argc != 4) {
    if (rank == 0) {
      fprintf(stderr, "usage: ptscotch_part graph.bin assign.txt imbalance\n");
    }
    MPI_Finalize();
    return 1;
  }
  int n = 0;
  int parts = 0;
  int nnz = 0;
  int32_t *off = NULL;
  int32_t *edg = NULL;
  int32_t *vw = NULL;
  int32_t *ew = NULL;
  int32_t *part = NULL;
  struct timespec whole0, whole1;
  clock_gettime(CLOCK_MONOTONIC, &whole0);
  if (rank == 0) {
    FILE *in = fopen(argv[1], "rb");
    if (in == NULL) {
      die(rank, "open");
    }
    int32_t head[2];
    if (fread(head, sizeof(int32_t), 2, in) != 2) {
      die(rank, "header");
    }
    n = head[0];
    parts = head[1];
    off = slurp(in, (size_t)n + 1);
    if (off == NULL) {
      die(rank, "offsets");
    }
    nnz = off[n];
    edg = slurp(in, (size_t)nnz);
    vw = slurp(in, (size_t)n);
    ew = slurp(in, (size_t)nnz);
    fclose(in);
    if (edg == NULL || vw == NULL || ew == NULL) {
      die(rank, "arrays");
    }
    part = (int32_t *)calloc((size_t)n, sizeof(int32_t));
    if (part == NULL) {
      die(rank, "part");
    }
  }
  double imbalance = 0.0;
  if (rank == 0) {
    imbalance = strtod(argv[3], NULL);
  }
  double seconds = 0.0;
  int rc = lc_ptscotch_part(n, off, edg, vw, ew, parts, imbalance, part, &seconds);
  if (rc != 0) {
    die(rank, "partition");
  }
  if (rank == 0) {
    printf("PART_SECONDS %.9f\n", seconds);
    printf("SECONDS %.9f\n", seconds);
    printf("SEED 1\n");
    printf("RANKS %d\n", size);
    printf("SCOTCH_VERSION %d.%d.%d\n", SCOTCH_VERSION, SCOTCH_RELEASE, SCOTCH_PATCHLEVEL);
    printf("STRATEGY SCOTCH_STRATQUALITY|SCOTCH_STRATBALANCE\n");
    printf("SCOTCH_randomReset 1\n");
    printf("SCOTCH_randomSeed 1\n");
    fflush(stdout);
    FILE *outf = fopen(argv[2], "w");
    if (outf == NULL) {
      die(rank, "assign");
    }
    for (int i = 0; i < n; ++i) {
      fprintf(outf, "%d\n", part[i]);
    }
    fclose(outf);
    clock_gettime(CLOCK_MONOTONIC, &whole1);
    double whole = (double)(whole1.tv_sec - whole0.tv_sec) +
                   (double)(whole1.tv_nsec - whole0.tv_nsec) / 1.0e9;
    printf("WHOLE_SECONDS %.9f\n", whole);
    fflush(stdout);
    free(off);
    free(edg);
    free(vw);
    free(ew);
    free(part);
  }
  MPI_Finalize();
  return 0;
}

#endif
