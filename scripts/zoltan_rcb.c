/* Zoltan recursive coordinate bisection on one rank.
 * The input is a text grid: count, part count, then x y z weight. */

#include <mpi.h>
#include <zoltan.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int g_n;
static int g_parts;
static double *g_x;
static double *g_y;
static double *g_z;
static float *g_w;

static int num_obj(void *data, int *ierr) {
  (void)data;
  *ierr = ZOLTAN_OK;
  return g_n;
}

static void obj_list(void *data, int num_gid, int num_lid, ZOLTAN_ID_PTR global_ids,
                     ZOLTAN_ID_PTR local_ids, int wdim, float *objwgts, int *ierr) {
  (void)data;
  (void)num_gid;
  (void)num_lid;
  for (int i = 0; i < g_n; ++i) {
    global_ids[i] = (ZOLTAN_ID_TYPE)i;
    local_ids[i] = (ZOLTAN_ID_TYPE)i;
    if (wdim > 0) {
      objwgts[i] = g_w[i];
    }
  }
  *ierr = ZOLTAN_OK;
}

static int num_geom(void *data, int *ierr) {
  (void)data;
  *ierr = ZOLTAN_OK;
  return 3;
}

static void geom_multi(void *data, int num_gid, int num_lid, int num_obj,
                       ZOLTAN_ID_PTR global_ids, ZOLTAN_ID_PTR local_ids, int num_dim,
                       double *geom_vec, int *ierr) {
  (void)data;
  (void)num_gid;
  (void)num_lid;
  (void)global_ids;
  for (int i = 0; i < num_obj; ++i) {
    int id = (int)local_ids[i];
    geom_vec[num_dim * i] = g_x[id];
    geom_vec[num_dim * i + 1] = g_y[id];
    geom_vec[num_dim * i + 2] = g_z[id];
  }
  *ierr = ZOLTAN_OK;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  if (argc != 3 && argc != 4) {
    fprintf(stderr, "usage: zoltan_rcb grid.txt assign.txt\n");
    MPI_Finalize();
    return 1;
  }
  FILE *in = fopen(argv[1], "r");
  if (in == 0) {
    MPI_Finalize();
    return 1;
  }
  if (fscanf(in, "%d %d", &g_n, &g_parts) != 2 || g_n < 1 || g_parts < 1) {
    fclose(in);
    MPI_Finalize();
    return 1;
  }
  g_x = (double *)calloc((size_t)g_n, sizeof(double));
  g_y = (double *)calloc((size_t)g_n, sizeof(double));
  g_z = (double *)calloc((size_t)g_n, sizeof(double));
  g_w = (float *)calloc((size_t)g_n, sizeof(float));
  for (int i = 0; i < g_n; ++i) {
    double w = 0.0;
    if (fscanf(in, "%lf %lf %lf %lf", &g_x[i], &g_y[i], &g_z[i], &w) != 4) {
      fclose(in);
      MPI_Finalize();
      return 1;
    }
    g_w[i] = (float)w;
  }
  fclose(in);

  float version = 0.0f;
  if (Zoltan_Initialize(argc, argv, &version) != ZOLTAN_OK) {
    MPI_Finalize();
    return 1;
  }
  struct Zoltan_Struct *zz = Zoltan_Create(MPI_COMM_WORLD);
  char parts[32];
  snprintf(parts, sizeof(parts), "%d", g_parts);
  Zoltan_Set_Param(zz, "LB_METHOD", "RCB");
  Zoltan_Set_Param(zz, "NUM_GID_ENTRIES", "1");
  Zoltan_Set_Param(zz, "NUM_LID_ENTRIES", "1");
  Zoltan_Set_Param(zz, "OBJ_WEIGHT_DIM", "1");
  Zoltan_Set_Param(zz, "NUM_GLOBAL_PARTS", parts);
  const char *tol = argc >= 4 ? argv[3] : "1.0";
  Zoltan_Set_Param(zz, "IMBALANCE_TOL", tol);
  Zoltan_Set_Param(zz, "RETURN_LISTS", "EXPORT");
  Zoltan_Set_Param(zz, "DEBUG_LEVEL", "0");
  Zoltan_Set_Fn(zz, ZOLTAN_NUM_OBJ_FN_TYPE, (void (*)())num_obj, 0);
  Zoltan_Set_Fn(zz, ZOLTAN_OBJ_LIST_FN_TYPE, (void (*)())obj_list, 0);
  Zoltan_Set_Fn(zz, ZOLTAN_NUM_GEOM_FN_TYPE, (void (*)())num_geom, 0);
  Zoltan_Set_Fn(zz, ZOLTAN_GEOM_MULTI_FN_TYPE, (void (*)())geom_multi, 0);

  int changes = 0;
  int num_gid = 0;
  int num_lid = 0;
  int num_import = 0;
  ZOLTAN_ID_PTR import_g = 0;
  ZOLTAN_ID_PTR import_l = 0;
  int *import_procs = 0;
  int *import_to = 0;
  int num_export = 0;
  ZOLTAN_ID_PTR export_g = 0;
  ZOLTAN_ID_PTR export_l = 0;
  int *export_procs = 0;
  int *export_to = 0;
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  int rc = Zoltan_LB_Partition(zz, &changes, &num_gid, &num_lid, &num_import, &import_g,
                               &import_l, &import_procs, &import_to, &num_export, &export_g,
                               &export_l, &export_procs, &export_to);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  double seconds = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) / 1.0e9;
  printf("SECONDS %.9f\n", seconds);
  if (rc != ZOLTAN_OK) {
    Zoltan_Destroy(&zz);
    MPI_Finalize();
    return 1;
  }
  int *part = (int *)calloc((size_t)g_n, sizeof(int));
  for (int i = 0; i < num_export; ++i) {
    int id = (int)export_l[i];
    if (id >= 0 && id < g_n) {
      part[id] = export_to[i];
    }
  }
  FILE *out = fopen(argv[2], "w");
  if (out == 0) {
    Zoltan_LB_Free_Part(&import_g, &import_l, &import_procs, &import_to);
    Zoltan_LB_Free_Part(&export_g, &export_l, &export_procs, &export_to);
    Zoltan_Destroy(&zz);
    MPI_Finalize();
    return 1;
  }
  for (int i = 0; i < g_n; ++i) {
    fprintf(out, "%d\n", part[i]);
  }
  fclose(out);
  free(part);
  Zoltan_LB_Free_Part(&import_g, &import_l, &import_procs, &import_to);
  Zoltan_LB_Free_Part(&export_g, &export_l, &export_procs, &export_to);
  Zoltan_Destroy(&zz);
  MPI_Finalize();
  return 0;
}
