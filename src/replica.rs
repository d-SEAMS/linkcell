//! One neighbour structure for every bead of a ring polymer.
//!
//! The candidates are built on the bead centroid at
//! `cutoff + 2 R + skin`, where `R` is the largest minimum-image
//! distance from the centroid to a bead (assuming each ring's spread
//! along any periodic axis stays below half the box width so the
//! minimum-image centroid is uniquely defined). Any pair inside `cutoff`
//! on a bead that still lies within `R` of the centroid is in that list.
//! [`ReplicaList::update_bead`] filters one bead.
//! [`ReplicaList::filter_all`] loads each candidate once and filters every
//! bead in that pass when no bead crosses a periodic boundary relative to
//! the centroid, and unwraps any boundary-straddling bead onto the
//! centroid otherwise. A bead whose minimum-image displacement from the
//! positions at the last candidate build exceeds `skin / 2`, or a cell
//! change, rebuilds the shared candidates from the stored bead coordinates.

use minimage::ring_centroid;

use crate::cell::Cell;
use crate::pairs::PairColumns;
use crate::skin::{
    filter_with_wraps, gather_candidates, lattice_wraps, moved_past_skin, same_lattice, WrapFrame,
};
use crate::skin_simd::Soa;
use crate::Error;

/// Shared candidate list and one filtered frame per bead.
pub struct ReplicaList {
    cell: Cell,
    ref_h: [[f64; 3]; 3],
    cutoff: f64,
    skin: f64,
    half: bool,
    sorted: bool,
    ref_sorted: bool,
    beads: usize,
    atoms: usize,
    soa: Soa,
    d2: Vec<f64>,
    /// Bead-major positions at the last candidate build.
    reference: Vec<[f64; 3]>,
    /// Centroid positions at the last candidate build.
    centroid: Vec<[f64; 3]>,
    /// Bead-major positions from the last build or update.
    last: Vec<[f64; 3]>,
    frames: Vec<PairColumns>,
    /// Largest minimum-image distance from the centroid at the last build.
    spread: f64,
    builds: u64,
    reuses: u64,
    last_reused: bool,
}

impl ReplicaList {
    /// Build the shared list and filter every bead.
    ///
    /// `xyz` is bead-major, `beads * atoms` points, `atoms = xyz.len() / beads`.
    /// `cutoff` and `skin` use the cell's length unit. `cutoff` must be
    /// finite and strictly positive. `skin` must be finite and at least
    /// zero. Otherwise the result is [`Error::BadCutoff`]. `beads == 0`,
    /// an empty `xyz`, or a length that is not a multiple of `beads`
    /// returns [`Error::Empty`].
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
        for p in xyz {
            if !p[0].is_finite() || !p[1].is_finite() || !p[2].is_finite() {
                return Err(Error::NonFinite);
            }
        }
        let atoms = xyz.len() / beads;
        let mut list = Self {
            cell,
            ref_h: cell.h(),
            cutoff,
            skin,
            half,
            sorted: false,
            ref_sorted: false,
            beads,
            atoms,
            soa: Soa::empty(),
            d2: Vec::new(),
            reference: Vec::new(),
            centroid: Vec::new(),
            last: xyz.to_vec(),
            frames: (0..beads).map(|_| PairColumns::default()).collect(),
            spread: 0.0,
            builds: 0,
            reuses: 0,
            last_reused: false,
        };
        list.rebuild_from(&list.last.clone())?;
        Ok(list)
    }

    /// Build the shared list from a slice of per-bead coordinate frames
    /// sharing one atom topology.
    ///
    /// An empty frame list or an empty first frame returns [`Error::Empty`].
    /// A frame whose length differs from the first frame returns
    /// [`Error::BufferSize`].
    pub fn from_frames<F: AsRef<[[f64; 3]]>>(
        frames: &[F],
        cell: Cell,
        cutoff: f64,
        skin: f64,
        half: bool,
    ) -> Result<Self, Error> {
        let Some(first) = frames.first() else {
            return Err(Error::Empty);
        };
        let atoms = first.as_ref().len();
        if atoms == 0 {
            return Err(Error::Empty);
        }
        let beads = frames.len();
        let total = beads.checked_mul(atoms).ok_or(Error::Overflow)?;
        let mut xyz = Vec::with_capacity(total);
        for frame in frames {
            let slice = frame.as_ref();
            if slice.len() != atoms {
                return Err(Error::BufferSize);
            }
            xyz.extend_from_slice(slice);
        }
        Self::build(&xyz, beads, cell, cutoff, skin, half)
    }

    /// Periodic cell currently held by the list.
    pub fn cell(&self) -> &Cell {
        &self.cell
    }

    /// Replace the periodic cell for subsequent rebuild checks and updates.
    pub fn set_cell(&mut self, cell: Cell) {
        self.cell = cell;
    }

    /// Cutoff the bead rows were filtered to.
    pub fn cutoff(&self) -> f64 {
        self.cutoff
    }

    /// Skin added on top of `cutoff + 2 * spread` when the candidates were built.
    pub fn skin(&self) -> f64 {
        self.skin
    }

    /// `true` when the list stores only the canonical half of each pair.
    pub fn half(&self) -> bool {
        self.half
    }

    /// `true` when candidate and filtered bead rows are ordered by `(i, j, shift)`.
    pub fn sorted(&self) -> bool {
        self.sorted
    }

    /// Enable or disable canonical `(i, j, shift)` row sorting on rebuild and filter.
    pub fn set_sorted(&mut self, sorted: bool) {
        self.sorted = sorted;
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

    /// Minimum-image ring centroid from the last shared build.
    pub fn centroid(&self) -> &[[f64; 3]] {
        &self.centroid
    }

    /// Shared candidate-list rebuilds performed so far.
    pub fn builds(&self) -> u64 {
        self.builds
    }

    /// Number of updates that reused the shared candidates without rebuilding.
    pub fn reuses(&self) -> u64 {
        self.reuses
    }

    /// `true` when the most recent update reused the shared candidates.
    pub fn last_reused(&self) -> bool {
        self.last_reused
    }

    /// Filtered rows of bead `b` from the last build or update.
    pub fn bead(&self, b: usize) -> Result<&PairColumns, Error> {
        self.frames.get(b).ok_or(Error::BufferSize)
    }

    /// Single-shift vectors `q + H S - p` for the filtered rows of bead `b`.
    pub fn vectors_of(&self, b: usize) -> Result<Vec<[f64; 3]>, Error> {
        let cols = self.bead(b)?;
        let base = b * self.atoms;
        let xyz = &self.last[base..base + self.atoms];
        let mut out = Vec::with_capacity(cols.len());
        for t in 0..cols.len() {
            let i = cols.i[t] as usize;
            let j = cols.j[t] as usize;
            let s = cols.shift[t];
            let shift = self.cell.lattice_shift(s[0], s[1], s[2]);
            let p = xyz[i];
            let q = xyz[j];
            out.push([
                q[0] + shift[0] - p[0],
                q[1] + shift[1] - p[1],
                q[2] + shift[2] - p[2],
            ]);
        }
        Ok(out)
    }

    /// Unique unordered atom pairs `(i, j)` with `i < j` in the shared
    /// candidate list.
    ///
    /// Self-images (`i == j`) are omitted. This is the spatial sparsity
    /// pattern of a ring-polymer potential: every bead's cutoff pairs are
    /// a subset, and the spring terms along the ring do not add atom pairs.
    pub fn sparsity(&self) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        self.soa.each_end(|i, j| {
            if i != j {
                let a = i as usize;
                let b = j as usize;
                pairs.push(if a < b { (a, b) } else { (b, a) });
            }
        });
        pairs.sort_unstable();
        pairs.dedup();
        pairs
    }

    /// `true` when bead-major `xyz` in the stored cell is not covered by
    /// the shared candidate list.
    pub fn needs_rebuild(&self, xyz: &[[f64; 3]]) -> bool {
        self.needs_rebuild_in(&self.cell, xyz)
    }

    /// `true` when bead-major `xyz` in `cell` is not covered by the shared
    /// candidate list.
    pub fn needs_rebuild_in(&self, cell: &Cell, xyz: &[[f64; 3]]) -> bool {
        if xyz.len() != self.beads * self.atoms
            || self.reference.len() != xyz.len()
            || self.sorted != self.ref_sorted
        {
            return true;
        }
        if !same_lattice(cell.h(), self.ref_h) {
            return true;
        }
        moved_past_skin(cell, &self.reference, xyz, self.skin)
    }

    /// `true` when per-bead `frames` are not covered by the shared candidate list.
    pub fn needs_rebuild_frames<F: AsRef<[[f64; 3]]>>(&self, frames: &[F]) -> bool {
        if frames.len() != self.beads
            || self.reference.len() != self.beads * self.atoms
            || self.sorted != self.ref_sorted
        {
            return true;
        }
        if !same_lattice(self.cell.h(), self.ref_h) {
            return true;
        }
        for (b, frame) in frames.iter().enumerate() {
            let slice = frame.as_ref();
            if slice.len() != self.atoms {
                return true;
            }
            let base = b * self.atoms;
            if moved_past_skin(
                &self.cell,
                &self.reference[base..base + self.atoms],
                slice,
                self.skin,
            ) {
                return true;
            }
        }
        false
    }

    /// Filter every stored bead from the shared candidate list.
    ///
    /// One pass loads each candidate once and gathers every bead's
    /// coordinates into the same lanes when no bead crosses a periodic
    /// boundary relative to the centroid. A shift outside the packed code
    /// or a bead that straddles a periodic boundary relative to the
    /// centroid uses a pass per bead.
    pub fn filter_all(&mut self) -> Result<(), Error> {
        let frames =
            self.filter_beads_from(&self.soa, &self.centroid, &self.reference, &self.last)?;
        self.frames = frames;
        Ok(())
    }

    /// Update all beads from bead-major `xyz`, rebuilding the shared
    /// candidate list only when the skin is spent or the cell has changed.
    ///
    /// If an error is returned, all stored frames, reference coordinates,
    /// and counters are left unchanged.
    pub fn update(&mut self, xyz: &[[f64; 3]]) -> Result<(), Error> {
        if xyz.len() != self.beads * self.atoms {
            return Err(Error::BufferSize);
        }
        for p in xyz {
            if !p[0].is_finite() || !p[1].is_finite() || !p[2].is_finite() {
                return Err(Error::NonFinite);
            }
        }
        if self.needs_rebuild(xyz) {
            self.rebuild_from(xyz)?;
            self.last_reused = false;
        } else {
            let frames = self.filter_beads_from(&self.soa, &self.centroid, &self.reference, xyz)?;
            self.last.copy_from_slice(xyz);
            self.frames = frames;
            self.reuses += 1;
            self.last_reused = true;
        }
        Ok(())
    }

    /// Replace the cell and update all beads from bead-major `xyz`.
    ///
    /// Restores the previous cell if the update returns an error.
    pub fn update_in(&mut self, cell: Cell, xyz: &[[f64; 3]]) -> Result<(), Error> {
        let prev = self.cell;
        self.cell = cell;
        if let Err(err) = self.update(xyz) {
            self.cell = prev;
            return Err(err);
        }
        Ok(())
    }

    /// Update all beads from a slice of per-bead coordinate frames.
    pub fn update_frames<F: AsRef<[[f64; 3]]>>(&mut self, frames: &[F]) -> Result<(), Error> {
        if frames.len() != self.beads {
            return Err(Error::BufferSize);
        }
        let mut xyz = Vec::with_capacity(self.beads * self.atoms);
        for frame in frames {
            let slice = frame.as_ref();
            if slice.len() != self.atoms {
                return Err(Error::BufferSize);
            }
            xyz.extend_from_slice(slice);
        }
        self.update(&xyz)
    }

    /// Replace bead `b` and return its filtered rows.
    ///
    /// A move inside the skin only filters that bead. A larger move
    /// or a cell change rebuilds the shared candidates and every bead's rows.
    /// If an error is returned, all stored frames, reference coordinates,
    /// and counters are left unchanged.
    pub fn update_bead(&mut self, b: usize, xyz: &[[f64; 3]]) -> Result<&PairColumns, Error> {
        if b >= self.beads || xyz.len() != self.atoms {
            return Err(Error::BufferSize);
        }
        for p in xyz {
            if !p[0].is_finite() || !p[1].is_finite() || !p[2].is_finite() {
                return Err(Error::NonFinite);
            }
        }
        let moved = self.bead_moved(b, xyz);
        let base = b * self.atoms;
        if moved {
            let mut next = self.last.clone();
            next[base..base + self.atoms].copy_from_slice(xyz);
            self.rebuild_from(&next)?;
            self.last_reused = false;
        } else {
            let mut vectors = Vec::new();
            filter_with_wraps(
                &self.soa,
                WrapFrame {
                    cell: &self.cell,
                    anchor: &self.centroid,
                    reference: &self.reference[base..base + self.atoms],
                    xyz,
                    sorted: self.sorted,
                },
                self.cutoff * self.cutoff,
                false,
                &mut self.d2,
                &mut self.frames[b],
                &mut vectors,
            )?;
            self.last[base..base + self.atoms].copy_from_slice(xyz);
            self.reuses += 1;
            self.last_reused = true;
        }
        Ok(&self.frames[b])
    }

    fn bead_moved(&self, b: usize, xyz: &[[f64; 3]]) -> bool {
        if !same_lattice(self.cell.h(), self.ref_h) || self.sorted != self.ref_sorted {
            return true;
        }
        let base = b * self.atoms;
        moved_past_skin(
            &self.cell,
            &self.reference[base..base + self.atoms],
            xyz,
            self.skin,
        )
    }

    fn rebuild_from(&mut self, xyz: &[[f64; 3]]) -> Result<(), Error> {
        let centroid = ring_centroid(&self.cell, self.beads, xyz)?;
        let mut spread2 = 0.0;
        for (a, centre) in centroid.iter().enumerate() {
            for b in 0..self.beads {
                let d2 = self.cell.dist2(*centre, xyz[b * self.atoms + a]);
                if !d2.is_finite() {
                    return Err(Error::NonFinite);
                }
                if d2 > spread2 {
                    spread2 = d2;
                }
            }
        }
        let spread = spread2.sqrt();
        let reach = self.cutoff + 2.0 * spread + self.skin;
        let soa = gather_candidates(&centroid, &self.cell, reach, self.half, self.sorted)?;
        let frames = self.filter_beads_from(&soa, &centroid, xyz, xyz)?;
        self.soa = soa;
        self.centroid = centroid;
        self.reference.clear();
        self.reference.extend_from_slice(xyz);
        self.last.clear();
        self.last.extend_from_slice(xyz);
        self.ref_h = self.cell.h();
        self.ref_sorted = self.sorted;
        self.spread = spread;
        self.frames = frames;
        self.builds += 1;
        Ok(())
    }

    fn filter_beads_from(
        &self,
        soa: &Soa,
        centroid: &[[f64; 3]],
        reference: &[[f64; 3]],
        xyz: &[[f64; 3]],
    ) -> Result<Vec<PairColumns>, Error> {
        let cut2 = self.cutoff * self.cutoff;
        let mut any_wrapped = false;
        for b in 0..self.beads {
            let base = b * self.atoms;
            let ref_bead = &reference[base..base + self.atoms];
            let now_bead = &xyz[base..base + self.atoms];
            if lattice_wraps(&self.cell, centroid, ref_bead, now_bead)?.is_some() {
                any_wrapped = true;
                break;
            }
        }
        let mut frames: Vec<PairColumns> =
            (0..self.beads).map(|_| PairColumns::default()).collect();
        if soa.is_coded() && !any_wrapped {
            soa.filter_beads(xyz, self.beads, &self.cell, cut2, &mut frames)?;
            return Ok(frames);
        }
        let mut d2 = Vec::new();
        let mut vectors = Vec::new();
        for (b, out) in frames.iter_mut().enumerate() {
            let start = b * self.atoms;
            filter_with_wraps(
                soa,
                WrapFrame {
                    cell: &self.cell,
                    anchor: centroid,
                    reference: &reference[start..start + self.atoms],
                    xyz: &xyz[start..start + self.atoms],
                    sorted: self.sorted,
                },
                cut2,
                false,
                &mut d2,
                out,
                &mut vectors,
            )?;
        }
        Ok(frames)
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
        assert_eq!(rep.builds(), 1);
        assert_eq!(rep.builds() + rep.reuses(), 1);
        assert!(!rep.last_reused());
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
        assert_eq!(owned.builds(), 1);
        assert_eq!(owned.builds() + owned.reuses(), 2);
        assert!(owned.last_reused());
        assert!(!rep.sparsity().is_empty());
    }

    #[test]
    fn from_frames_straddling_pbc_matches_fresh_bead_lists() {
        let cell = Cell::ortho(10.0, 10.0, 10.0).unwrap();
        let frame0 = [[0.1, 0.0, 0.0], [0.5, 0.0, 0.0]];
        let frame1 = [[9.9, 0.0, 0.0], [0.5, 0.0, 0.0]];
        let mut rep = ReplicaList::from_frames(&[frame0, frame1], cell, 1.0, 0.5, true).unwrap();
        assert_eq!(rep.beads(), 2);
        assert_eq!(rep.atoms(), 2);
        assert_eq!(rep.sparsity(), vec![(0, 1)]);
        let b0 = rep.bead(0).unwrap();
        assert_eq!(b0.i, vec![0]);
        assert_eq!(b0.j, vec![1]);
        assert_eq!(b0.shift, vec![[0, 0, 0]]);
        let b1 = rep.bead(1).unwrap();
        assert_eq!(b1.i, vec![0]);
        assert_eq!(b1.j, vec![1]);
        assert_eq!(b1.shift, vec![[1, 0, 0]]);
        let v1 = rep.vectors_of(1).unwrap();
        assert!((v1[0][0] - 0.6).abs() <= 16.0 * f64::EPSILON);

        let moved0 = [[9.95, 0.0, 0.0], [0.5, 0.0, 0.0]];
        let moved1 = [[0.05, 0.0, 0.0], [0.5, 0.0, 0.0]];
        assert!(!rep.needs_rebuild_frames(&[moved0, moved1]));
        rep.update_frames(&[moved0, moved1]).unwrap();
        assert_eq!(rep.builds(), 1);
        assert_eq!(rep.builds() + rep.reuses(), 2);
        assert!(rep.last_reused());
        assert_eq!(rep.bead(0).unwrap().shift, vec![[1, 0, 0]]);
        assert_eq!(rep.bead(1).unwrap().shift, vec![[0, 0, 0]]);
    }

    #[test]
    fn replica_handles_multi_box_shifts_in_both_directions() {
        // Wrapped coordinates in a 3 A box with reach 8.82 < 3 * 3.0 = 9.0:
        // every centroid candidate shift is inside [-3, 3]^3, and images
        // (2, -1, 0) (slot 262 > u8::MAX) and (3, 0, 0) (sx = 3, slot 318 > u8::MAX)
        // are inside the cutoff.
        let small = Cell::ortho(3.0, 3.0, 3.0).unwrap();
        let small0 = [[2.5, 0.1, 0.1], [0.2, 0.1, 0.1]];
        let small1 = [[2.52, 0.1, 0.1], [0.22, 0.1, 0.1]];
        let rep_small = ReplicaList::from_frames(&[small0, small1], small, 8.6, 0.2, true).unwrap();
        for (b, frame) in [small0, small1].iter().enumerate() {
            let mut fresh = PairColumns::default();
            pairs_within_columns(frame, &small, 8.6, None, None, true, &mut fresh).unwrap();
            let got = rep_small.bead(b).unwrap();
            assert_eq!(canon(got), canon(&fresh));
            assert!(got.shift.contains(&[2, -1, 0]));
            assert!(got.shift.contains(&[3, 0, 0]));
        }

        // Also test unwrapped bead coordinates in [-3, 3]^3 in a 10 A box.
        let cell = Cell::ortho(10.0, 10.0, 10.0).unwrap();
        let frame0 = [[30.2, 0.1, 0.0], [20.3, -9.8, 0.0], [0.4, 0.1, 0.0]];
        let frame1 = [[30.25, 0.15, 0.0], [20.35, -9.75, 0.0], [0.45, 0.15, 0.0]];
        let rep = ReplicaList::from_frames(&[frame0, frame1], cell, 1.2, 0.4, true).unwrap();
        assert_eq!(rep.sparsity(), vec![(0, 1), (0, 2), (1, 2)]);
        for (b, frame) in [frame0, frame1].iter().enumerate() {
            let mut fresh = PairColumns::default();
            pairs_within_columns(frame, &cell, 1.2, None, None, true, &mut fresh).unwrap();
            let got = rep.bead(b).unwrap();
            assert_eq!(canon(got), canon(&fresh));
            assert!(got.shift.contains(&[2, -1, 0]));
            assert!(got.shift.contains(&[3, 0, 0]));
        }
    }

    #[test]
    fn failed_update_leaves_replica_state_and_counters_unchanged() {
        let cell = Cell::ortho(10.0, 10.0, 10.0).unwrap();
        let frame0 = [[0.2, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let frame1 = [[0.25, 0.0, 0.0], [1.05, 0.0, 0.0]];
        let mut rep = ReplicaList::from_frames(&[frame0, frame1], cell, 1.5, 0.5, true).unwrap();
        let b0_before = rep.bead(0).unwrap().clone();
        let b1_before = rep.bead(1).unwrap().clone();
        let spread_before = rep.spread();
        let centroid_before = rep.centroid().to_vec();

        let bad_overflow = [[1.0e12, 0.0, 0.0], [1.0, 0.0, 0.0]];
        assert_eq!(
            rep.update_frames(&[bad_overflow, frame1]),
            Err(Error::Overflow)
        );
        assert_eq!(rep.builds(), 1);
        assert_eq!(rep.builds() + rep.reuses(), 1);
        assert!(!rep.last_reused());
        assert_eq!(rep.spread(), spread_before);
        assert_eq!(rep.centroid(), &centroid_before[..]);
        assert_eq!(rep.bead(0).unwrap(), &b0_before);
        assert_eq!(rep.bead(1).unwrap(), &b1_before);
        assert!(!rep.needs_rebuild_frames(&[frame0, frame1]));

        let bad_nan = [[f64::NAN, 0.0, 0.0], [1.0, 0.0, 0.0]];
        assert_eq!(rep.update_bead(0, &bad_nan), Err(Error::NonFinite));
        assert_eq!(rep.builds(), 1);
        assert_eq!(rep.builds() + rep.reuses(), 1);
        assert!(!rep.last_reused());
        assert_eq!(rep.bead(0).unwrap(), &b0_before);
        assert_eq!(rep.bead(1).unwrap(), &b1_before);

        let wrap0 = [[9.9, 0.0, 0.0], [0.1, 0.0, 0.0]];
        let wrap1 = [[9.85, 0.0, 0.0], [0.1, 0.0, 0.0]];
        let mut wrap_rep = ReplicaList::from_frames(&[wrap0, wrap1], cell, 1.0, 0.5, true).unwrap();
        let wrap_b0_before = wrap_rep.bead(0).unwrap().clone();
        let wrap_b1_before = wrap_rep.bead(1).unwrap().clone();
        assert_eq!(wrap_b0_before.shift, vec![[1, 0, 0]]);
        let wrap_moved0 = [[9.9 + 10.0 * i32::MAX as f64, 0.0, 0.0], [0.1, 0.0, 0.0]];
        assert!(!wrap_rep.needs_rebuild_frames(&[wrap_moved0, wrap1]));
        assert_eq!(
            wrap_rep.update_frames(&[wrap_moved0, wrap1]),
            Err(Error::Overflow)
        );
        assert_eq!(wrap_rep.builds(), 1);
        assert_eq!(wrap_rep.builds() + wrap_rep.reuses(), 1);
        assert!(!wrap_rep.last_reused());
        assert_eq!(wrap_rep.bead(0).unwrap(), &wrap_b0_before);
        assert_eq!(wrap_rep.bead(1).unwrap(), &wrap_b1_before);

        assert_eq!(wrap_rep.update_bead(0, &wrap_moved0), Err(Error::Overflow));
        assert_eq!(wrap_rep.builds(), 1);
        assert_eq!(wrap_rep.builds() + wrap_rep.reuses(), 1);
        assert!(!wrap_rep.last_reused());
        assert_eq!(wrap_rep.bead(0).unwrap(), &wrap_b0_before);
        assert_eq!(wrap_rep.bead(1).unwrap(), &wrap_b1_before);
    }
}
