//! The neighbour declaration of the shared potential schema, read for a
//! row grid.
//!
//! `NeighborRequest` and `NeighborSystem` are copied field for field from
//! the potential schema into `schema/neighbor.capnp`, and
//! `neighbor_capnp.rs` is the code `capnpc-rust` writes for that file.
//! A struct's wire layout follows from its fields alone, so a message
//! built with the full schema decodes here.

use capnp::message::ReaderOptions;
use capnp::serialize;

use crate::neighbor_capnp::neighbor_request;

/// What a row grid takes from a `NeighborRequest`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Decoded {
    /// Radius the rows are tested against: `cutoff`, or `cutoff + skin`
    /// for a loose cut.
    pub(crate) cutoff: f64,
    /// Keep `dist2 == cutoff²`: a loose cut with no skin.
    pub(crate) closed: bool,
    pub(crate) periodic: [bool; 3],
    /// Lattice rows a, b, c. Zero when every axis is open and the
    /// request has no box.
    pub(crate) lattice: [f64; 9],
    /// Corner the lattice vectors start from.
    pub(crate) origin: [f64; 3],
    /// The declared shell: `true` for a half list.
    pub(crate) half: bool,
    /// Bin edge, or `None` for the builder's.
    pub(crate) cell_hint: Option<f64>,
    pub(crate) exclude_self: bool,
}

/// Decode one stream-framed message whose root is a `NeighborRequest`
/// for `n` points. The error is the sentence the C entry reports.
pub(crate) fn decode(bytes: &[u8], n: usize) -> Result<Decoded, String> {
    let bad = |err: capnp::Error| format!("NeighborRequest does not decode: {err}");
    let mut slice = bytes;
    let message =
        serialize::read_message_from_flat_slice(&mut slice, ReaderOptions::new()).map_err(bad)?;
    let request = message
        .get_root::<neighbor_request::Reader<'_>>()
        .map_err(bad)?;
    if request.get_format() != crate::neighbor_capnp::NEIGHBOR_FORMAT_PAIRS {
        return Err("NeighborRequest.format must be 0 (pairs) for a row grid".to_owned());
    }
    if request.get_mode() == crate::neighbor_capnp::NEIGHBOR_MODE_INVALID {
        return Err("NeighborRequest.mode 0 is not a neighbour mode".to_owned());
    }
    let (cutoff, skin) = (request.get_cutoff(), request.get_skin());
    if !skin.is_finite() || skin < 0.0 {
        return Err("NeighborRequest.skin must be finite and not negative".to_owned());
    }
    let (cutoff, closed) = match request.get_cut() {
        crate::neighbor_capnp::NEIGHBOR_CUT_STRICT if skin == 0.0 => (cutoff, false),
        crate::neighbor_capnp::NEIGHBOR_CUT_STRICT => {
            return Err("NeighborRequest.skin needs cut = 1 (loose)".to_owned())
        }
        crate::neighbor_capnp::NEIGHBOR_CUT_LOOSE => (cutoff + skin, skin == 0.0),
        _ => return Err("NeighborRequest.cut must be 0 (strict) or 1 (loose)".to_owned()),
    };
    let half = match request.get_shell() {
        Ok(crate::neighbor_capnp::NeighborShell::Half) => true,
        Ok(crate::neighbor_capnp::NeighborShell::Full) => false,
        Err(_) => return Err("NeighborRequest.shell must be half or full".to_owned()),
    };
    let edge = request.get_bin_edge();
    let cell_hint = (edge.is_finite() && edge > 0.0).then_some(edge);
    let corner = request.get_origin().map_err(bad)?;
    let origin = match corner.len() {
        0 => [0.0; 3],
        3 => [corner.get(0), corner.get(1), corner.get(2)],
        _ => return Err("NeighborRequest.origin must be empty or three coordinates".to_owned()),
    };
    let flags = request.get_periodic().map_err(bad)?;
    let periodic = match flags.len() {
        0 => [true, true, true],
        3 => [flags.get(0), flags.get(1), flags.get(2)],
        _ => return Err("NeighborRequest.periodic must be empty or three flags".to_owned()),
    };
    let systems = request.get_systems().map_err(bad)?;
    if systems.len() != 1 {
        return Err(
            "NeighborRequest.systems must hold exactly one system for a row grid".to_owned(),
        );
    }
    let system = systems.get(0);
    if system.get_count() as usize != n {
        return Err("NeighborRequest.systems[0].count is not the number of points".to_owned());
    }
    let cell = system.get_box().map_err(bad)?;
    let mut lattice = [0.0; 9];
    match cell.len() {
        9 => {
            for (k, entry) in lattice.iter_mut().enumerate() {
                *entry = cell.get(k as u32);
            }
        }
        0 if periodic == [false, false, false] => {}
        _ => {
            return Err("NeighborRequest.systems[0].box must hold nine lattice entries".to_owned())
        }
    }
    Ok(Decoded {
        cutoff,
        closed,
        periodic,
        lattice,
        origin,
        half,
        cell_hint,
        exclude_self: request.get_exclude_self_images(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A serialized request, with the fields a test sets.
    struct Wire {
        cutoff: f64,
        skin: f64,
        format: u8,
        cut: u8,
        mode: u8,
        periodic: Vec<bool>,
        systems: Vec<(u32, Vec<f64>)>,
        full: bool,
        bin_edge: f64,
        origin: Vec<f64>,
        exclude_self: bool,
    }

    impl Wire {
        fn new(cutoff: f64, n: u32, lattice: [f64; 9]) -> Self {
            Wire {
                cutoff,
                skin: 0.0,
                format: 0,
                cut: 0,
                mode: 1,
                periodic: Vec::new(),
                systems: vec![(n, lattice.to_vec())],
                full: false,
                bin_edge: 0.0,
                origin: Vec::new(),
                exclude_self: false,
            }
        }

        fn bytes(&self) -> Vec<u8> {
            let mut message = capnp::message::Builder::new_default();
            {
                let mut request = message.init_root::<neighbor_request::Builder<'_>>();
                request.set_cutoff(self.cutoff);
                request.set_skin(self.skin);
                request.set_format(self.format);
                request.set_cut(self.cut);
                request.set_mode(self.mode);
                request.set_shell(if self.full {
                    crate::neighbor_capnp::NeighborShell::Full
                } else {
                    crate::neighbor_capnp::NeighborShell::Half
                });
                request.set_bin_edge(self.bin_edge);
                request.set_exclude_self_images(self.exclude_self);
                let mut corner = request.reborrow().init_origin(self.origin.len() as u32);
                for (k, &v) in self.origin.iter().enumerate() {
                    corner.set(k as u32, v);
                }
                let mut flags = request.reborrow().init_periodic(self.periodic.len() as u32);
                for (k, &on) in self.periodic.iter().enumerate() {
                    flags.set(k as u32, on);
                }
                let mut systems = request.init_systems(self.systems.len() as u32);
                for (k, (count, lattice)) in self.systems.iter().enumerate() {
                    let mut system = systems.reborrow().get(k as u32);
                    system.set_count(*count);
                    let mut entries = system.init_box(lattice.len() as u32);
                    for (a, &v) in lattice.iter().enumerate() {
                        entries.set(a as u32, v);
                    }
                }
            }
            serialize::write_message_to_words(&message)
        }
    }

    const BOX: [f64; 9] = [9.0, 0.0, 0.0, 1.5, 8.0, 0.0, -1.0, 0.5, 10.0];

    #[test]
    fn a_request_decodes_to_the_grid_arguments() {
        let strict = decode(&Wire::new(3.0, 5, BOX).bytes(), 5).unwrap();
        assert_eq!(
            strict,
            Decoded {
                cutoff: 3.0,
                closed: false,
                periodic: [true, true, true],
                lattice: BOX,
                origin: [0.0; 3],
                half: true,
                cell_hint: None,
                exclude_self: false,
            }
        );
        let mut declared = Wire::new(3.0, 5, BOX);
        declared.full = true;
        declared.bin_edge = 1.5;
        declared.origin = vec![-1.0, 2.0, 0.5];
        declared.exclude_self = true;
        let declared = decode(&declared.bytes(), 5).unwrap();
        assert!(!declared.half);
        assert_eq!(declared.cell_hint, Some(1.5));
        assert_eq!(declared.origin, [-1.0, 2.0, 0.5]);
        assert!(declared.exclude_self);
        let mut loose = Wire::new(3.0, 5, BOX);
        loose.cut = 1;
        loose.skin = 0.5;
        loose.periodic = vec![true, false, true];
        let loose = decode(&loose.bytes(), 5).unwrap();
        assert_eq!(loose.cutoff, 3.5);
        assert!(!loose.closed);
        assert_eq!(loose.periodic, [true, false, true]);
        // A loose cut with no skin keeps the boundary, on any cell.
        let mut closed = Wire::new(3.0, 5, BOX);
        closed.cut = 1;
        let periodic = decode(&closed.bytes(), 5).unwrap();
        assert!(periodic.closed);
        assert_eq!(periodic.cutoff, 3.0);
        closed.periodic = vec![false, false, false];
        closed.systems = vec![(5, Vec::new())];
        let closed = decode(&closed.bytes(), 5).unwrap();
        assert!(closed.closed);
        assert_eq!(closed.lattice, [0.0; 9]);
    }

    #[test]
    fn a_request_the_grid_cannot_serve_is_refused() {
        let refused = |edit: &dyn Fn(&mut Wire), n: usize| {
            let mut wire = Wire::new(3.0, 5, BOX);
            edit(&mut wire);
            decode(&wire.bytes(), n).unwrap_err()
        };
        assert!(refused(&|w| w.format = 1, 5).contains("format"));
        assert!(refused(&|w| w.mode = 0, 5).contains("mode"));
        assert!(refused(&|w| w.skin = 0.2, 5).contains("skin"));
        assert!(refused(&|w| w.skin = -0.2, 5).contains("skin"));
        assert!(refused(&|w| w.cut = 2, 5).contains("cut"));
        assert!(refused(&|w| w.periodic = vec![true, true], 5).contains("periodic"));
        assert!(refused(&|w| w.systems.push((5, BOX.to_vec())), 5).contains("exactly one"));
        assert!(refused(&|w| w.systems.clear(), 5).contains("exactly one"));
        assert!(refused(&|_| (), 6).contains("count"));
        assert!(refused(&|w| w.systems[0].1.truncate(6), 5).contains("nine"));
        assert!(refused(&|w| w.origin = vec![1.0, 2.0], 5).contains("origin"));
        assert!(decode(&[1, 2, 3], 5)
            .unwrap_err()
            .starts_with("NeighborRequest does not decode"));
    }

    /// The C entry builds the grid the plain build does for the same
    /// arguments.
    #[cfg(feature = "capi")]
    #[test]
    fn the_c_entry_builds_the_same_grid() {
        use crate::capi::{
            lc_row, lc_row_grid_build_request, lc_row_grid_free, lc_row_grid_len, lc_row_grid_rows,
        };
        use std::ffi::{c_int, c_void};

        unsafe extern "C" fn collect(user: *mut c_void, row: *const lc_row) -> c_int {
            let rows = unsafe { &mut *(user as *mut Vec<(i32, i32, [i32; 3], u64)>) };
            let row = unsafe { *row };
            rows.push((row.i, row.j, row.shift, row.dist2.to_bits()));
            0
        }

        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let xyz: Vec<[f64; 3]> = (0..120)
            .map(|_| {
                [
                    12.0 * next() - 1.0,
                    11.0 * next() - 1.0,
                    12.0 * next() - 1.0,
                ]
            })
            .collect();
        for periodic in [[true, true, true], [true, true, false]] {
            let mut wire = Wire::new(2.5, 120, BOX);
            wire.cut = 1;
            wire.skin = 0.5;
            wire.periodic = periodic.to_vec();
            wire.full = true;
            wire.bin_edge = 1.75;
            wire.origin = vec![0.5, -0.25, 1.0];
            wire.exclude_self = true;
            let bytes = wire.bytes();
            let keep: Vec<bool> = (0..120).map(|i| i % 7 != 3).collect();
            let mask: Vec<c_int> = keep.iter().map(|&on| c_int::from(on)).collect();
            let mut grid = std::ptr::null_mut();
            let mut half: c_int = -1;
            let code = unsafe {
                lc_row_grid_build_request(
                    xyz.as_ptr().cast(),
                    xyz.len(),
                    mask.as_ptr(),
                    bytes.as_ptr(),
                    bytes.len(),
                    &mut half,
                    &mut grid,
                )
            };
            assert_eq!(code, 0);
            assert_eq!(half, 0);
            assert_eq!(unsafe { lc_row_grid_len(grid) }, 120);
            let mut got: Vec<(i32, i32, [i32; 3], u64)> = Vec::new();
            let user = (&mut got as *mut Vec<(i32, i32, [i32; 3], u64)>).cast();
            assert_eq!(
                unsafe { lc_row_grid_rows(grid, 0, 120, 0, Some(collect), user) },
                0
            );
            unsafe { lc_row_grid_free(grid) };
            let cell = crate::cell_for_pbc(
                [
                    [BOX[0], BOX[1], BOX[2]],
                    [BOX[3], BOX[4], BOX[5]],
                    [BOX[6], BOX[7], BOX[8]],
                ],
                [0.5, -0.25, 1.0],
                periodic,
            )
            .unwrap();
            let query = crate::PbcQuery::new(periodic, 3.0)
                .mask(Some(&keep))
                .cell_hint(Some(1.75))
                .exclude_self_images(true);
            let plain = crate::RowGrid::build(&xyz, &cell, &query).unwrap();
            let mut want = Vec::new();
            plain
                .for_rows(0..120, false, |row| {
                    want.push((row.i as i32, row.j as i32, row.shift, row.dist2.to_bits()))
                })
                .unwrap();
            assert!(!want.is_empty());
            assert_eq!(got, want);
        }
    }
}
