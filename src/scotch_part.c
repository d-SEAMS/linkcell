/* Serial Scotch entry. Compiled only with `--features scotch`.
 * The library is not vendored. SCOTCH_Num is the header's integer. */

#include <scotch.h>

#include <stdint.h>

int lc_scotch_part(int n, const int32_t *offsets, const int32_t *edges,
                   const int32_t *vwgt, const int32_t *ewgt, int parts,
                   double imbalance, int32_t *part) {
  if (n < 0 || parts < 1 || offsets == 0 || part == 0) {
    return 1;
  }
  SCOTCH_Num *vert = 0;
  SCOTCH_Num *edge = 0;
  SCOTCH_Num *vwt = 0;
  SCOTCH_Num *ewt = 0;
  SCOTCH_Num *out = 0;
  int rc = 1;
  SCOTCH_Graph graph;
  SCOTCH_Graph bound;
  SCOTCH_Strat strat;
  SCOTCH_Context context;
  int graph_ready = 0;
  int bound_ready = 0;
  int strat_ready = 0;
  int context_ready = 0;
  SCOTCH_Num edgenbr = 0;

  if (sizeof(SCOTCH_Num) == sizeof(int32_t)) {
    vert = (SCOTCH_Num *)(void *)offsets;
    edge = (SCOTCH_Num *)(void *)edges;
    vwt = (SCOTCH_Num *)(void *)vwgt;
    ewt = (SCOTCH_Num *)(void *)ewgt;
    out = (SCOTCH_Num *)(void *)part;
  } else {
    return 1;
  }
  if (n > 0) {
    edgenbr = offsets[n];
  }
  if (SCOTCH_graphInit(&graph) != 0) {
    return 1;
  }
  graph_ready = 1;
  if (SCOTCH_graphBuild(&graph, 0, (SCOTCH_Num)n, vert, 0, vwt, 0, edgenbr,
                        edge, ewt) != 0) {
    goto done;
  }
  if (SCOTCH_graphCheck(&graph) != 0) {
    goto done;
  }
  if (SCOTCH_contextInit(&context) != 0) {
    goto done;
  }
  context_ready = 1;
  SCOTCH_contextRandomSeed(&context, 1);
  SCOTCH_contextOptionSetNum(&context, SCOTCH_OPTIONNUMDETERMINISTIC, 1);
  SCOTCH_contextOptionSetNum(&context, SCOTCH_OPTIONNUMRANDOMFIXEDSEED, 1);
  if (SCOTCH_contextBindGraph(&context, &graph, &bound) != 0) {
    goto done;
  }
  bound_ready = 1;
  if (SCOTCH_stratInit(&strat) != 0) {
    goto done;
  }
  strat_ready = 1;
  SCOTCH_randomReset();
  SCOTCH_randomSeed(1);
  if (SCOTCH_stratGraphMapBuild(&strat,
                                SCOTCH_STRATQUALITY | SCOTCH_STRATBALANCE,
                                (SCOTCH_Num)parts, imbalance) != 0) {
    goto done;
  }
  if (SCOTCH_graphPart(&bound, (SCOTCH_Num)parts, &strat, out) != 0) {
    goto done;
  }
  rc = 0;
done:
  if (strat_ready) {
    SCOTCH_stratExit(&strat);
  }
  if (bound_ready) {
    SCOTCH_graphExit(&bound);
  }
  if (context_ready) {
    SCOTCH_contextExit(&context);
  }
  if (graph_ready) {
    SCOTCH_graphExit(&graph);
  }
  return rc;
}
