@0xab76990b7c23e2e9;

# The neighbour declaration of the shared potential schema.
#
# Everything below is copied from potential.capnp without change. A
# struct's wire layout follows from its fields, so a message built with
# the full schema decodes with this file. A field added there is added
# here, with the same ordinal, and scripts/regen-neighbor-capnp.sh
# rewrites src/neighbor_capnp.rs.

# Values of NeighborRequest.format, which stays a UInt8 on the wire.
# They are the values of rgpot_nlist_request.shape.
# pairs: one row per pair. cluster4: four consecutive central atoms and
# one partner per row, with NeighborList.mask naming the lanes; a cluster
# list is a full shell.
const neighborFormatPairs :UInt8 = 0;
const neighborFormatCluster4 :UInt8 = 1;

# Values of NeighborRequest.cut, which stays a UInt8 on the wire.
# They are the values of rgpot_nlist_request.cut.
# strict: distance < cutoff, and skin is 0.
# loose: distance < cutoff + skin, or distance <= cutoff when skin is 0.
const neighborCutStrict :UInt8 = 0;
const neighborCutLoose :UInt8 = 1;

# NeighborRequest.mode is reserved. 0 is not a neighbour mode and is
# refused. No server gives the other values a meaning: a request is
# served the same way for every nonzero value. New requests send the
# default, 1.
const neighborModeInvalid :UInt8 = 0;
const neighborModeDefault :UInt8 = 1;

# Which side of each pair a list holds.
# half: one row per pair, (i, j, S) or (j, i, -S). full: both.
enum NeighborShell {
  half @0;
  full @1;
}

# @struct NeighborRequest
# @brief The neighbour list a consumer declares.
# @field cutoff Pair cutoff, in the length unit of the positions.
# @field skin Buffer added to the cutoff under a loose cut.
# @field format One of the neighborFormat constants.
# @field cut One of the neighborCut constants.
# @field periodic Empty for a fully periodic cell, or one flag per lattice axis.
# @field shifts Whether a stored reply carries the integer cell shifts.
# @field vectors Whether a stored reply carries the pair vectors.
# @field distances Whether a stored reply carries the pair distances.
# @field mode Reserved; see the neighborMode constants.
# @field cursor First row of the chunk a stored reply returns.
# @field limit Most rows in that chunk; 0 leaves the choice to the server.
# @field systems The systems, which do not interact with each other.
# @field shell Half or full list. A cluster format is a full shell.
# @field binEdge Edge of the search bins; 0 or less leaves it to the builder.
# @field origin Empty for a cell at the origin, or the three coordinates of
#   the corner the lattice vectors start from.
# @field excludeSelfImages Drop the rows that pair an atom with a periodic
#   image of itself. A pair of an atom with itself at shift zero is never
#   returned.
# A per-atom mask is not part of the declaration. It travels with the
# positions, as a typed buffer.
struct NeighborRequest {
  cutoff @0 :Float64;
  skin @1 :Float64 = 0;
  format @2 :UInt8 = 0;
  cut @3 :UInt8 = 0;
  periodic @4 :List(Bool);
  shifts @5 :Bool = true;
  vectors @6 :Bool = true;
  distances @7 :Bool = true;
  mode @8 :UInt8 = 1;
  cursor @9 :UInt64 = 0;
  limit @10 :UInt32 = 0;
  systems @11 :List(NeighborSystem);
  shell @12 :NeighborShell = half;
  binEdge @13 :Float64 = 0;
  origin @14 :List(Float64);
  excludeSelfImages @15 :Bool = false;
}

# @struct NeighborSystem
# @field count Number of atoms of this system in the concatenated positions.
# @field box Nine lattice entries, vectors as rows: ax ay az bx by bz cx cy cz.
struct NeighborSystem {
  count @0 :UInt32;
  box @1 :List(Float64);
}
