//! One neighbour structure for every bead of a ring polymer.
//!
//! The candidates are built on the bead centroid at
//! `cutoff + 2 R + skin`, where `R` is the largest minimum-image
//! distance from the centroid to a bead. Any pair inside `cutoff` on a
//! bead that still lies within `R` of the centroid is in that list.
//! [`ReplicaList::update_bead`] filters one bead.
//! [`ReplicaList::filter_all`] loads each candidate once and filters every
//! bead in that pass. A bead that has moved more than `skin / 2` from the
//! positions used at the last build rebuilds the shared candidates from
//! the stored bead coordinates.

use minimage::ring_centroid;

use crate::cell::Cell;
use crate::pairs::PairColumns;
use crate::skin::gather_candidates;
use crate::skin_simd::Soa;
use crate::Error;

/// Shared candidate list and one filtered frame per bead.
pub struct ReplicaList {
    cell: Cell,
    cutoff: f64,
    skin: f64,
    half: bool,
    beads: usize,
    atoms: usize,
    soa: Soa,
    d2: Vec<f64>,
    /// Bead-major positions from the last build or update.
    last: Vec<[f64; 3]>,
    frames: Vec<PairColumns>,
    /// Largest minimum-image distance from the centroid at the last build.
    spread: f64,
}

impl ReplicaList {
    /// Build the shared list and filter every bead.
    ///
    /// `xyz` is bead-major, `beads * atoms` points, `atoms = xyz.len() / beads`.
    pub fn build(
        xyz: &[[f64; 3]],
        beads: usize,
        cell: Cell,
        cutoff: f64,
        skin: f64,
        half: bool,
    ) -> Result<Self, Error> {
        if !cutoff.is_finite() || cutoff <= 0.0 || !skin.is_finite() || skin < 0.0 {
            return Err(Error::BadCutoff);
        }
        if beads == 0 || xyz.is_empty() || xyz.len() % beads != 0 {
            return Err(Error::Empty);
        }
        let atoms = xyz.len() / beads;
        let mut list = Self {
            cell,
            cutoff,
            skin,
            half,
            beads,
            atoms,
            soa: Soa::empty(),
            d2: Vec::new(),
            last: xyz.to_vec(),
            frames: (0..beads).map(|_| PairColumns::default()).collect(),
            spread: 0.0,
        };
        list.rebuild_from_last()?;
        Ok(list)
    }

    /// Number of beads.
    pub fn beads(&self) -> usize {
        self.beads
    }

    /// Atoms in one bead.
    pub fn atoms(&self) -> usize {
        self.atoms
    }

    /// Spread `R` from the last shared build.
    pub fn spread(&self) -> f64 {
        self.spread
    }

    /// Filtered rows of bead `b` from the last build or update.
    pub fn bead(&self, b: usize) -> Result<&PairColumns, Error> {
        self.frames.get(b).ok_or(Error::BufferSize)
    }

    /// Unique unordered atom pairs in the shared candidate list.
    ///
    /// Self-images (`i == j`) are omitted. This is the spatial sparsity
    /// pattern of a ring-polymer potential: every bead's cutoff pairs are
    /// a subset, and the spring terms along the ring do not add atom pairs.
    pub fn sparsity(&self) -> Vec<(u32, u32)> {
        let mut pairs = Vec::new();
        self.soa.each_end(|i, j| {
            if i != j {
                pairs.push(if i < j { (i, j) } else { (j, i) });
            }
        });
        pairs.sort_unstable();
        pairs.dedup();
        pairs
    }

    /// Filter every stored bead from the shared candidate list.
    ///
    /// One pass loads each candidate once and gathers every bead's
    /// coordinates into the same lanes. A shift that does not fit the
    /// packed code falls back to a pass per bead.
    pub fn filter_all(&mut self) -> Result<(), Error> {
        let cut2 = self.cutoff * self.cutoff;
        if self.soa.is_coded() {
            return self.soa.filter_beads(
                &self.last,
                self.beads,
                &self.cell,
                cut2,
                &mut self.frames,
            );
        }
        for b in 0..self.beads {
            let start = b * self.atoms;
            let frame = self.last[start..start + self.atoms].to_vec();
            self.filter_bead(b, &frame)?;
        }
        Ok(())
    }

    /// Replace bead `b` and return its filtered rows.
    ///
    /// A move inside the skin only filters that bead. A larger move
    /// rebuilds the shared candidates and every bead's rows.
    pub fn update_bead(&mut self, b: usize, xyz: &[[f64; 3]]) -> Result<&PairColumns, Error> {
        if b >= self.beads || xyz.len() != self.atoms {
            return Err(Error::BufferSize);
        }
        let moved = self.bead_moved(b, xyz);
        let base = b * self.atoms;
        self.last[base..base + self.atoms].copy_from_slice(xyz);
        if moved {
            self.rebuild_from_last()?;
        } else {
            self.filter_bead(b, xyz)?;
        }
        Ok(&self.frames[b])
    }

    fn bead_moved(&self, b: usize, xyz: &[[f64; 3]]) -> bool {
        let half = self.skin * 0.5;
        let room2 = half * half;
        let base = b * self.atoms;
        for (p, r) in xyz.iter().zip(self.last[base..base + self.atoms].iter()) {
            let dx = p[0] - r[0];
            let dy = p[1] - r[1];
            let dz = p[2] - r[2];
            if dx * dx + dy * dy + dz * dz > room2 {
                return true;
            }
        }
        false
    }

    fn rebuild_from_last(&mut self) -> Result<(), Error> {
        let centroid = ring_centroid(&self.cell, self.beads, &self.last)?;
        let mut spread2 = 0.0;
        for (a, centre) in centroid.iter().enumerate() {
            for b in 0..self.beads {
                let d2 = self.cell.dist2(*centre, self.last[b * self.atoms + a]);
                if d2 > spread2 {
                    spread2 = d2;
                }
            }
        }
        self.spread = spread2.sqrt();
        let reach = self.cutoff + 2.0 * self.spread + self.skin;
        self.soa = gather_candidates(&centroid, &self.cell, reach, self.half)?;
        self.filter_all()
    }

    fn filter_bead(&mut self, b: usize, xyz: &[[f64; 3]]) -> Result<(), Error> {
        let mut vectors = Vec::new();
        self.soa.filter(
            xyz,
            &self.cell,
            self.cutoff * self.cutoff,
            false,
            &mut self.d2,
            &mut self.frames[b],
            &mut vectors,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pairs::pairs_within_columns;

    fn canon(cols: &PairColumns) -> Vec<(i32, i32, [i32; 3])> {
        let mut v: Vec<_> = (0..cols.len())
            .map(|k| {
                let (i, j, s) = (cols.i[k], cols.j[k], cols.shift[k]);
                if i < j
                    || (i == j
                        && (s[0] < 0 || (s[0] == 0 && (s[1] < 0 || (s[1] == 0 && s[2] < 0)))))
                {
                    (i, j, s)
                } else {
                    (j, i, [-s[0], -s[1], -s[2]])
                }
            })
            .collect();
        v.sort_unstable();
        v
    }

    #[test]
    fn each_bead_matches_its_own_list() {
        let cell = Cell::ortho(14.0, 14.0, 14.0).unwrap();
        let atoms = 40;
        let beads = 4;
        let mut xyz = vec![[0.0; 3]; beads * atoms];
        let mut state = 7_u64;
        for a in 0..atoms {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let base = [
                (state >> 11) as f64 / (1u64 << 53) as f64 * 14.0,
                {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    (state >> 11) as f64 / (1u64 << 53) as f64 * 14.0
                },
                {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    (state >> 11) as f64 / (1u64 << 53) as f64 * 14.0
                },
            ];
            for b in 0..beads {
                xyz[b * atoms + a] = [
                    (base[0] + 0.03 * b as f64).rem_euclid(14.0),
                    base[1],
                    base[2],
                ];
            }
        }
        let rep = ReplicaList::build(&xyz, beads, cell, 3.2, 0.5, true).unwrap();
        assert!(rep.spread() < 0.2);
        for b in 0..beads {
            let mut fresh = PairColumns::default();
            pairs_within_columns(
                &xyz[b * atoms..(b + 1) * atoms],
                &cell,
                3.2,
                None,
                None,
                true,
                &mut fresh,
            )
            .unwrap();
            assert_eq!(canon(rep.bead(b).unwrap()), canon(&fresh));
        }
        let mut moved = xyz[0..atoms].to_vec();
        for p in &mut moved {
            p[1] += 0.01;
        }
        let mut owned = ReplicaList::build(&xyz, beads, cell, 3.2, 0.5, true).unwrap();
        let cols = owned.update_bead(0, &moved).unwrap();
        let mut fresh = PairColumns::default();
        pairs_within_columns(&moved, &cell, 3.2, None, None, true, &mut fresh).unwrap();
        assert_eq!(canon(cols), canon(&fresh));
        assert!(!rep.sparsity().is_empty());
    }
}
