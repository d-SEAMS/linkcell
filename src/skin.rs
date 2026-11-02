//! Verlet skin over a cutoff pair list.
//!
//! The candidate list is [`pairs_within_columns`] at `cutoff + skin`.
//! A later frame reuses it while every atom is within `skin / 2` of its
//! reference position, after the box-corner correction of Verlet
//! (Phys. Rev. **159**, 98, 1967) as used by vesin and LAMMPS. The kept
//! rows satisfy `|q + H S - p|^2` strictly below `cutoff²`. Shifts inside
//! `[-3, 3]^3` are a one-byte code; the filter gathers coordinates eight
//! at a time and compacts the hits. A wider shift keeps the three shift
//! columns and uses the same test.

use crate::cell::Cell;
use crate::pairs::{pairs_within_columns, PairColumns};
use crate::skin_simd::Soa;
use crate::Error;

/// Cutoff list with a Verlet skin.
///
/// `update` rebuilds the candidates when an atom has moved too far and
/// otherwise only filters them. The returned columns borrow the list.
pub struct SkinList {
    cell: Cell,
    cutoff: f64,
    skin: f64,
    half: bool,
    reference: Vec<[f64; 3]>,
    ref_h: [[f64; 3]; 3],
    soa: Soa,
    d2: Vec<f64>,
    cols: PairColumns,
    vectors: Vec<[f64; 3]>,
    built: bool,
}

impl SkinList {
    /// Empty list. The first [`Self::update`] builds the candidates.
    pub fn new(cell: Cell, cutoff: f64, skin: f64, half: bool) -> Result<Self, Error> {
        if !cutoff.is_finite() || cutoff <= 0.0 {
            return Err(Error::BadCutoff);
        }
        if !skin.is_finite() || skin < 0.0 {
            return Err(Error::BadCutoff);
        }
        Ok(Self {
            cell,
            cutoff,
            skin,
            half,
            reference: Vec::new(),
            ref_h: cell.h(),
            soa: Soa::empty(),
            d2: Vec::new(),
            cols: PairColumns::default(),
            vectors: Vec::new(),
            built: false,
        })
    }

    /// Cutoff the rows were filtered to.
    pub fn cutoff(&self) -> f64 {
        self.cutoff
    }

    /// Skin added on top of the cutoff when the candidates were built.
    pub fn skin(&self) -> f64 {
        self.skin
    }

    /// Rows from the last [`Self::update`].
    pub fn columns(&self) -> &PairColumns {
        &self.cols
    }

    /// Shifted vectors from the last update that asked for them.
    /// Empty when that update did not.
    pub fn vectors(&self) -> &[[f64; 3]] {
        &self.vectors
    }

    /// `true` when `xyz` is not covered by the stored candidates.
    pub fn needs_rebuild(&self, xyz: &[[f64; 3]]) -> bool {
        if !self.built || xyz.len() != self.reference.len() {
            return true;
        }
        let half = self.skin * 0.5;
        let corner = corner_span(self.cell.h(), self.ref_h) * 0.5;
        if corner > half {
            return true;
        }
        let room = half - corner;
        let room2 = room * room;
        for (p, r) in xyz.iter().zip(self.reference.iter()) {
            let dx = p[0] - r[0];
            let dy = p[1] - r[1];
            let dz = p[2] - r[2];
            if dx * dx + dy * dy + dz * dz > room2 {
                return true;
            }
        }
        false
    }

    /// Filter `xyz`, rebuilding the candidates first when the skin is spent.
    ///
    /// `want_vector` fills [`Self::vectors`] with `q + H S - p`.
    pub fn update(&mut self, xyz: &[[f64; 3]], want_vector: bool) -> Result<&PairColumns, Error> {
        if xyz.is_empty() {
            return Err(Error::Empty);
        }
        if self.needs_rebuild(xyz) {
            self.rebuild(xyz)?;
        }
        self.filter(xyz, want_vector)?;
        Ok(&self.cols)
    }

    fn rebuild(&mut self, xyz: &[[f64; 3]]) -> Result<(), Error> {
        let reach = self.cutoff + self.skin;
        self.soa = gather_candidates(xyz, &self.cell, reach, self.half)?;
        self.reference = xyz.to_vec();
        self.ref_h = self.cell.h();
        self.built = true;
        Ok(())
    }

    fn filter(&mut self, xyz: &[[f64; 3]], want_vector: bool) -> Result<(), Error> {
        self.soa.filter(
            xyz,
            &self.cell,
            self.cutoff * self.cutoff,
            want_vector,
            &mut self.d2,
            &mut self.cols,
            &mut self.vectors,
        )
    }
}

pub(crate) fn gather_candidates(
    xyz: &[[f64; 3]],
    cell: &Cell,
    reach: f64,
    half: bool,
) -> Result<Soa, Error> {
    let mut cols = PairColumns::default();
    pairs_within_columns(xyz, cell, reach, None, None, half, &mut cols)?;
    Soa::from_columns(&cols, xyz.len())
}

fn corner_span(h: [[f64; 3]; 3], href: [[f64; 3]; 3]) -> f64 {
    let mut best1 = 0.0;
    let mut best2 = 0.0;
    for bits in 0..8 {
        let b = [
            f64::from(bits & 1),
            f64::from((bits >> 1) & 1),
            f64::from((bits >> 2) & 1),
        ];
        let mut d = [0.0; 3];
        for a in 0..3 {
            let corner = b[0] * h[0][a] + b[1] * h[1][a] + b[2] * h[2][a];
            let old = b[0] * href[0][a] + b[1] * href[1][a] + b[2] * href[2][a];
            d[a] = corner - old;
        }
        let n2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        if n2 > best1 {
            best2 = best1;
            best1 = n2;
        } else if n2 > best2 {
            best2 = n2;
        }
    }
    best1.sqrt() + best2.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pairs::pairs_within_columns;

    fn canon(i: i32, j: i32, mut s: [i32; 3]) -> (i32, i32, [i32; 3]) {
        let swap = if i != j {
            i > j
        } else if s[0] != 0 {
            s[0] > 0
        } else if s[1] != 0 {
            s[1] > 0
        } else {
            s[2] > 0
        };
        if swap {
            s = [-s[0], -s[1], -s[2]];
            (j, i, s)
        } else {
            (i, j, s)
        }
    }

    fn set_of(cols: &PairColumns) -> Vec<(i32, i32, [i32; 3])> {
        let mut v: Vec<_> = (0..cols.len())
            .map(|k| canon(cols.i[k], cols.j[k], cols.shift[k]))
            .collect();
        v.sort_unstable();
        v
    }

    #[test]
    fn skin_filter_matches_a_fresh_list() {
        let cell = Cell::ortho(12.0, 11.0, 10.0).unwrap();
        let mut xyz = Vec::new();
        let mut state = 0x1234_5678_u64;
        for _ in 0..80 {
            let mut p = [0.0; 3];
            for (a, slot) in p.iter_mut().enumerate() {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                *slot = (state >> 33) as f64 / (1u64 << 31) as f64 * cell.widths()[a];
            }
            xyz.push(p);
        }
        let mut fresh = PairColumns::default();
        pairs_within_columns(&xyz, &cell, 3.5, None, None, true, &mut fresh).unwrap();
        let mut skin = SkinList::new(cell, 3.5, 0.6, true).unwrap();
        let (got, n) = {
            let cols = skin.update(&xyz, true).unwrap();
            (set_of(cols), cols.len())
        };
        assert_eq!(got, set_of(&fresh));
        assert_eq!(skin.vectors().len(), n);
        let mut moved = xyz.clone();
        for p in &mut moved {
            p[0] += 0.02;
        }
        assert!(!skin.needs_rebuild(&moved));
        let got = {
            let cols = skin.update(&moved, false).unwrap();
            set_of(cols)
        };
        let mut again = PairColumns::default();
        pairs_within_columns(&moved, &cell, 3.5, None, None, true, &mut again).unwrap();
        assert_eq!(got, set_of(&again));
        assert!(skin.vectors().is_empty());
    }
}
