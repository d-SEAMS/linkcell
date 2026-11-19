//! Verlet skin over a cutoff pair list.
//!
//! The candidate list is [`pairs_within_columns`] at `cutoff + skin`.
//! A later frame reuses it while every atom is within `skin / 2` of its
//! reference position, after the box-corner correction of Verlet
//! (Phys. Rev. **159**, 98, 1967) as used by vesin and LAMMPS. The kept
//! rows satisfy `|q + H S - p|^2` strictly below `cutoff²`. Shifts inside
//! `[-3, 3]^3` are a 16-bit code; the filter gathers coordinates eight
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
    periodic: [bool; 3],
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
    builds: u64,
    reuses: u64,
    last_reused: bool,
}

impl SkinList {
    /// Empty list. The first [`Self::update`] builds the candidates.
    ///
    /// `cutoff` and `skin` use the cell's length unit. `cutoff` must be
    /// finite and strictly positive. `skin` must be finite and at least
    /// zero. Otherwise the result is [`Error::BadCutoff`].
    pub fn new(cell: Cell, cutoff: f64, skin: f64, half: bool) -> Result<Self, Error> {
        if !cutoff.is_finite() || cutoff <= 0.0 {
            return Err(Error::BadCutoff);
        }
        if !skin.is_finite() || skin < 0.0 {
            return Err(Error::BadCutoff);
        }
        Ok(Self {
            cell,
            periodic: [true, true, true],
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
            builds: 0,
            reuses: 0,
            last_reused: false,
        })
    }

    /// Set the periodicity mask per lattice axis.
    pub fn with_periodic(mut self, periodic: [bool; 3]) -> Self {
        self.periodic = periodic;
        self.built = false;
        self
    }

    /// Periodicity mask per lattice axis.
    pub fn periodic(&self) -> [bool; 3] {
        self.periodic
    }

    /// Simulation cell used for candidate generation and filtering.
    pub fn cell(&self) -> &Cell {
        &self.cell
    }

    /// Update the simulation cell for subsequent [`Self::needs_rebuild`] and [`Self::update`] calls.
    pub fn set_cell(&mut self, cell: Cell) {
        self.cell = cell;
    }

    /// Cutoff the rows were filtered to.
    pub fn cutoff(&self) -> f64 {
        self.cutoff
    }

    /// Skin added on top of the cutoff when the candidates were built.
    pub fn skin(&self) -> f64 {
        self.skin
    }

    /// Number of candidate cell-list builds at `cutoff + skin` performed so far.
    pub fn builds(&self) -> u64 {
        self.builds
    }

    /// Number of [`Self::update`] calls that reused the cached candidate list without rebuilding.
    pub fn reuses(&self) -> u64 {
        self.reuses
    }

    /// Whether the most recent [`Self::update`] call reused the cached candidate list.
    pub fn last_reused(&self) -> bool {
        self.last_reused
    }

    /// Maximum Euclidean displacement of `xyz` from the stored reference positions,
    /// or `None` before the first build or when `xyz.len()` differs from the reference.
    pub fn max_displacement(&self, xyz: &[[f64; 3]]) -> Option<f64> {
        if !self.built || xyz.len() != self.reference.len() {
            return None;
        }
        let max_d2 = xyz
            .iter()
            .zip(self.reference.iter())
            .fold(0.0_f64, |acc, (p, r)| {
                let dx = p[0] - r[0];
                let dy = p[1] - r[1];
                let dz = p[2] - r[2];
                acc.max(dx * dx + dy * dy + dz * dz)
            });
        Some(max_d2.sqrt())
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
        if !self.built || self.skin == 0.0 || xyz.len() != self.reference.len() {
            return true;
        }
        let half = self.skin * 0.5;
        let corner = if self.periodic == [false, false, false] {
            0.0
        } else {
            corner_span(self.cell.h(), self.ref_h) * 0.5
        };
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
            self.builds += 1;
            self.last_reused = false;
        } else {
            self.reuses += 1;
            self.last_reused = true;
        }
        self.filter(xyz, want_vector)?;
        Ok(&self.cols)
    }

    fn rebuild(&mut self, xyz: &[[f64; 3]]) -> Result<(), Error> {
        let reach = self.cutoff + self.skin;
        self.soa = if self.periodic == [true, true, true] {
            gather_candidates(xyz, &self.cell, reach, self.half)?
        } else {
            gather_candidates_pbc(xyz, &self.cell, self.periodic, reach, self.half)?
        };
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

fn canonicalize_half_columns(cols: &mut PairColumns) {
    let mut rows: Vec<(i32, i32, [i32; 3], f64)> = (0..cols.len())
        .map(|k| {
            let i = cols.i[k];
            let j = cols.j[k];
            let s = cols.shift[k];
            let d2 = cols.dist2[k];
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
                (j, i, [-s[0], -s[1], -s[2]], d2)
            } else {
                (i, j, s, d2)
            }
        })
        .collect();
    rows.sort_unstable_by_key(|a| (a.0, a.1, a.2));
    for (k, (i, j, s, d2)) in rows.into_iter().enumerate() {
        cols.i[k] = i;
        cols.j[k] = j;
        cols.shift[k] = s;
        cols.dist2[k] = d2;
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
    if half {
        canonicalize_half_columns(&mut cols);
    }
    Soa::from_columns(&cols, xyz.len())
}

fn gather_candidates_pbc(
    xyz: &[[f64; 3]],
    cell: &Cell,
    periodic: [bool; 3],
    reach: f64,
    half: bool,
) -> Result<Soa, Error> {
    let query = crate::open_cell::PbcQuery::new(periodic, reach).half(half);
    let pairs = crate::open_cell::pairs_within_pbc(xyz, cell, &query)?;
    let mut cols = PairColumns {
        i: Vec::with_capacity(pairs.len()),
        j: Vec::with_capacity(pairs.len()),
        shift: Vec::with_capacity(pairs.len()),
        dist2: Vec::with_capacity(pairs.len()),
    };
    for p in pairs {
        cols.i.push(p.i as i32);
        cols.j.push(p.j as i32);
        cols.shift.push(p.shift);
        cols.dist2.push(p.dist2);
    }
    if half {
        canonicalize_half_columns(&mut cols);
    }
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
        for (a, slot) in d.iter_mut().enumerate() {
            let corner = b[0] * h[0][a] + b[1] * h[1][a] + b[2] * h[2][a];
            let old = b[0] * href[0][a] + b[1] * href[1][a] + b[2] * href[2][a];
            *slot = corner - old;
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

    #[test]
    fn cached_list_matches_fresh_build_after_small_moves_and_tracks_reuses() {
        let cell = Cell::ortho(12.0, 11.0, 10.0).unwrap();
        let mut xyz = Vec::new();
        let mut state = 0x9e37_79b9_u64;
        for _ in 0..96 {
            let mut p = [0.0; 3];
            for (a, slot) in p.iter_mut().enumerate() {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                *slot = (state >> 33) as f64 / (1u64 << 31) as f64 * cell.widths()[a];
            }
            xyz.push(p);
        }

        let cutoff = 3.5;
        let skin_width = 0.6;
        let mut cached = SkinList::new(cell, cutoff, skin_width, true).unwrap();
        assert_eq!(cached.builds(), 0);
        assert_eq!(cached.reuses(), 0);
        assert!(!cached.last_reused());
        assert_eq!(cached.max_displacement(&xyz), None);

        let initial_rows: Vec<(i32, i32, [i32; 3], u64)> = {
            let cols = cached.update(&xyz, true).unwrap();
            (0..cols.len())
                .map(|k| (cols.i[k], cols.j[k], cols.shift[k], cols.dist2[k].to_bits()))
                .collect()
        };
        assert_eq!(cached.builds(), 1);
        assert_eq!(cached.reuses(), 0);
        assert!(!cached.last_reused());
        assert_eq!(cached.max_displacement(&xyz), Some(0.0));

        let mut fresh_zero = SkinList::new(cell, cutoff, 0.0, true).unwrap();
        let fresh_initial: Vec<(i32, i32, [i32; 3], u64)> = {
            let cols = fresh_zero.update(&xyz, true).unwrap();
            (0..cols.len())
                .map(|k| (cols.i[k], cols.j[k], cols.shift[k], cols.dist2[k].to_bits()))
                .collect()
        };
        assert_eq!(initial_rows, fresh_initial);

        for step in 1_u32..=3 {
            let scale = 0.08 * f64::from(step);
            let moved: Vec<[f64; 3]> = xyz
                .iter()
                .enumerate()
                .map(|(idx, p)| {
                    let phase = idx as f64 * 0.41;
                    [
                        p[0] + scale * phase.sin(),
                        p[1] + scale * phase.cos() * 0.5,
                        p[2] - scale * (2.0 * phase).sin() * 0.5,
                    ]
                })
                .collect();
            let max_disp = cached.max_displacement(&moved).unwrap();
            assert!(max_disp > 0.0 && max_disp <= 0.5 * skin_width);
            assert!(!cached.needs_rebuild(&moved));

            let cached_rows: Vec<(i32, i32, [i32; 3], u64)> = {
                let cols = cached.update(&moved, true).unwrap();
                (0..cols.len())
                    .map(|k| (cols.i[k], cols.j[k], cols.shift[k], cols.dist2[k].to_bits()))
                    .collect()
            };
            assert_eq!(cached.builds(), 1);
            assert_eq!(cached.reuses(), u64::from(step));
            assert!(cached.last_reused());

            let mut fresh = PairColumns::default();
            pairs_within_columns(&moved, &cell, cutoff, None, None, true, &mut fresh).unwrap();
            let cached_set: Vec<_> = cached_rows
                .iter()
                .map(|&(i, j, s, _)| canon(i, j, s))
                .collect();
            assert_eq!(cached_set, set_of(&fresh));

            let fresh_rows: Vec<(i32, i32, [i32; 3], u64)> = {
                let cols = fresh_zero.update(&moved, true).unwrap();
                (0..cols.len())
                    .map(|k| (cols.i[k], cols.j[k], cols.shift[k], cols.dist2[k].to_bits()))
                    .collect()
            };
            assert_eq!(cached_rows, fresh_rows);
            assert_eq!(cached.vectors(), fresh_zero.vectors());
        }
    }

    #[test]
    fn skin_list_rebuilds_once_maximum_displacement_exceeds_half_skin() {
        let cell = Cell::ortho(12.0, 12.0, 12.0).unwrap();
        let xyz = vec![
            [1.0, 1.0, 1.0],
            [3.2, 1.0, 1.0],
            [7.0, 4.0, 2.0],
            [7.0, 6.8, 2.0],
        ];
        let cutoff = 3.0;
        let skin_width = 0.5;
        let mut skin = SkinList::new(cell, cutoff, skin_width, true).unwrap();
        skin.update(&xyz, true).unwrap();
        assert_eq!(skin.builds(), 1);
        assert_eq!(skin.reuses(), 0);
        assert!(!skin.last_reused());

        let mut inside = xyz.clone();
        inside[0][0] += 0.24;
        assert!(!skin.needs_rebuild(&inside));
        skin.update(&inside, true).unwrap();
        assert_eq!(skin.builds(), 1);
        assert_eq!(skin.reuses(), 1);
        assert!(skin.last_reused());

        let mut past_half_skin = xyz.clone();
        past_half_skin[0][0] += 0.26;
        assert!(skin.max_displacement(&past_half_skin).unwrap() > 0.5 * skin_width);
        assert!(skin.needs_rebuild(&past_half_skin));
        let rebuilt = {
            let cols = skin.update(&past_half_skin, true).unwrap();
            set_of(cols)
        };
        assert_eq!(skin.builds(), 2);
        assert_eq!(skin.reuses(), 1);
        assert!(!skin.last_reused());

        let mut fresh = PairColumns::default();
        pairs_within_columns(&past_half_skin, &cell, cutoff, None, None, true, &mut fresh).unwrap();
        assert_eq!(rebuilt, set_of(&fresh));
    }

    #[test]
    fn skin_list_supports_open_boundaries_and_cell_updates() {
        let cell = Cell::ortho(6.0, 6.0, 6.0).unwrap();
        let xyz = [[0.2, 0.0, 0.0], [5.8, 0.0, 0.0], [1.5, 0.0, 0.0]];
        let mut open_skin = SkinList::new(cell, 2.0, 0.4, true)
            .unwrap()
            .with_periodic([false, false, false]);
        assert_eq!(open_skin.periodic(), [false, false, false]);
        let cols = open_skin.update(&xyz, true).unwrap();
        assert_eq!(cols.len(), 1);
        assert_eq!((cols.i[0], cols.j[0], cols.shift[0]), (0, 2, [0, 0, 0]));
        assert_eq!(open_skin.builds(), 1);

        let stretched = Cell::ortho(8.0, 6.0, 6.0).unwrap();
        let mut pbc_skin = SkinList::new(cell, 2.0, 0.4, true).unwrap();
        pbc_skin.update(&xyz, true).unwrap();
        pbc_skin.set_cell(stretched);
        assert_eq!(pbc_skin.cell().widths(), stretched.widths());
        assert!(pbc_skin.needs_rebuild(&xyz));
        pbc_skin.update(&xyz, true).unwrap();
        assert_eq!(pbc_skin.builds(), 2);
        assert_eq!(pbc_skin.reuses(), 0);
        assert!(!pbc_skin.last_reused());
    }

    #[test]
    fn skin_list_matches_pairs_within_across_multi_image_shifts() {
        let mut state = 0x243f_6a88_85a3_08d3_u64;
        let mut next = || -> f64 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut saw_shift_2 = false;
        let mut saw_shift_3 = false;
        for trial in 0..200 {
            let lx = 1.25 + 0.45 * next();
            let ly = 1.35 + 0.55 * next();
            let lz = 1.45 + 0.55 * next();
            let cell = if trial % 2 == 0 {
                Cell::ortho(lx, ly, lz).unwrap()
            } else {
                let xy = (next() - 0.5) * 0.7 * lx;
                let xz = (next() - 0.5) * 0.7 * lx;
                let yz = (next() - 0.5) * 0.7 * ly;
                Cell::from_vectors([lx, 0.0, 0.0], [xy, ly, 0.0], [xz, yz, lz], [0.0; 3]).unwrap()
            };
            let min_w = cell.widths().into_iter().fold(f64::INFINITY, f64::min);
            let cutoff = min_w * (1.85 + 0.35 * next());
            let skin_width = 0.25 * min_w;
            let n = 16;
            let xyz: Vec<[f64; 3]> = (0..n)
                .map(|_| cell.cartesian([next(), next(), next()]))
                .collect();
            let step = 0.2 * skin_width;
            let moved: Vec<[f64; 3]> = xyz
                .iter()
                .enumerate()
                .map(|(idx, p)| {
                    let phase = idx as f64 * 0.73;
                    [
                        p[0] + step * phase.sin(),
                        p[1] + step * phase.cos() * 0.5,
                        p[2] - step * (2.0 * phase).sin() * 0.5,
                    ]
                })
                .collect();
            let half = trial % 3 != 0;

            let mut fresh0 = PairColumns::default();
            pairs_within_columns(&xyz, &cell, cutoff, None, None, half, &mut fresh0).unwrap();
            for s in &fresh0.shift {
                for &c in s {
                    if c.abs() == 2 {
                        saw_shift_2 = true;
                    }
                    if c.abs() == 3 {
                        saw_shift_3 = true;
                    }
                }
            }
            let mut want0_keys: Vec<_> = (0..fresh0.len())
                .map(|k| {
                    if half {
                        canon(fresh0.i[k], fresh0.j[k], fresh0.shift[k])
                    } else {
                        (fresh0.i[k], fresh0.j[k], fresh0.shift[k])
                    }
                })
                .collect();
            want0_keys.sort_unstable();

            let mut fresh1 = PairColumns::default();
            pairs_within_columns(&moved, &cell, cutoff, None, None, half, &mut fresh1).unwrap();
            let mut want1_keys: Vec<_> = (0..fresh1.len())
                .map(|k| {
                    if half {
                        canon(fresh1.i[k], fresh1.j[k], fresh1.shift[k])
                    } else {
                        (fresh1.i[k], fresh1.j[k], fresh1.shift[k])
                    }
                })
                .collect();
            want1_keys.sort_unstable();

            let mut ref_rows0 = None;
            let mut ref_rows1 = None;
            for path in crate::simd::paths_to_compare() {
                let _guard = crate::simd::force(path);
                let mut skin = SkinList::new(cell, cutoff, skin_width, half).unwrap();
                let (got0_keys, got0_rows): (Vec<_>, Vec<_>) = {
                    let cols = skin.update(&xyz, false).unwrap();
                    let keys = (0..cols.len())
                        .map(|k| (cols.i[k], cols.j[k], cols.shift[k]))
                        .collect();
                    let rows = (0..cols.len())
                        .map(|k| (cols.i[k], cols.j[k], cols.shift[k], cols.dist2[k].to_bits()))
                        .collect();
                    (keys, rows)
                };
                assert!(skin.soa.is_coded());
                assert_eq!(skin.builds(), 1);
                assert_eq!(skin.reuses(), 0);
                assert!(!skin.last_reused());
                let mut sorted0_keys = got0_keys;
                sorted0_keys.sort_unstable();
                assert_eq!(
                    sorted0_keys, want0_keys,
                    "initial build trial={trial} path={path:?}"
                );
                if let Some(ref want_bits) = ref_rows0 {
                    assert_eq!(
                        &got0_rows, want_bits,
                        "simd bits0 trial={trial} path={path:?}"
                    );
                } else {
                    ref_rows0 = Some(got0_rows);
                }

                let (got1_keys, got1_rows): (Vec<_>, Vec<_>) = {
                    let cols = skin.update(&moved, false).unwrap();
                    let keys = (0..cols.len())
                        .map(|k| (cols.i[k], cols.j[k], cols.shift[k]))
                        .collect();
                    let rows = (0..cols.len())
                        .map(|k| (cols.i[k], cols.j[k], cols.shift[k], cols.dist2[k].to_bits()))
                        .collect();
                    (keys, rows)
                };
                assert_eq!(skin.builds(), 1);
                assert_eq!(skin.reuses(), 1);
                assert!(skin.last_reused());
                let mut sorted1_keys = got1_keys;
                sorted1_keys.sort_unstable();
                assert_eq!(
                    sorted1_keys, want1_keys,
                    "reused filter trial={trial} path={path:?}"
                );
                if let Some(ref want_bits) = ref_rows1 {
                    assert_eq!(
                        &got1_rows, want_bits,
                        "simd bits1 trial={trial} path={path:?}"
                    );
                } else {
                    ref_rows1 = Some(got1_rows);
                }
            }
        }
        assert!(saw_shift_2 && saw_shift_3);
    }

    #[test]
    fn skin_keeps_shifts_whose_table_slot_exceeds_u8_max() {
        let cell = Cell::ortho(10.0, 10.0, 10.0).unwrap();
        // Every candidate shift among these three atoms lies in [-3, 3]^3:
        // (0, 1) has shift [1, 1, 0] (slot 227), (0, 2) has shift [3, 0, 0]
        // (sx = 3, slot 318 > u8::MAX), and (1, 2) has shift [2, -1, 0]
        // (slot 262 > u8::MAX).
        let explicit = [[30.2, 0.1, 0.0], [20.3, -9.8, 0.0], [0.4, 0.1, 0.0]];
        let mut fresh = PairColumns::default();
        pairs_within_columns(&explicit, &cell, 1.0, None, None, true, &mut fresh).unwrap();
        let mut skin = SkinList::new(cell, 1.0, 0.4, true).unwrap();
        let cols = skin.update(&explicit, false).unwrap();
        assert_eq!(set_of(cols), set_of(&fresh));
        assert!(cols.shift.contains(&[2, -1, 0]));
        assert!(cols.shift.contains(&[3, 0, 0]));

        // Also test wrapped coordinates in a 3 A box with reach 8.8 < 3 * 3.0,
        // where every candidate shift lies in [-3, 3]^3 and images (2, -1, 0)
        // and (3, 0, 0) are inside the cutoff.
        let small = Cell::ortho(3.0, 3.0, 3.0).unwrap();
        let wrapped = [[2.5, 0.1, 0.1], [0.2, 0.1, 0.1]];
        let mut fresh_small = PairColumns::default();
        pairs_within_columns(&wrapped, &small, 8.6, None, None, true, &mut fresh_small).unwrap();
        let mut skin_small = SkinList::new(small, 8.6, 0.2, true).unwrap();
        let cols_small = skin_small.update(&wrapped, false).unwrap();
        assert_eq!(set_of(cols_small), set_of(&fresh_small));
        assert!(cols_small.shift.contains(&[2, -1, 0]));
        assert!(cols_small.shift.contains(&[3, 0, 0]));

        let mut s = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = || {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            (s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..200 {
            let xyz: Vec<[f64; 3]> = (0..40)
                .map(|k| {
                    let mut p = [next() * 10.0, next() * 10.0, next() * 10.0];
                    if k % 2 == 1 {
                        for slot in &mut p {
                            *slot += 10.0 * (next() * 3.0).floor();
                        }
                    }
                    p
                })
                .collect();
            let mut want = PairColumns::default();
            pairs_within_columns(&xyz, &cell, 3.0, None, None, true, &mut want).unwrap();
            let mut list = SkinList::new(cell, 3.0, 0.5, true).unwrap();
            let got = list.update(&xyz, false).unwrap();
            assert_eq!(set_of(got), set_of(&want));
        }
    }
}
