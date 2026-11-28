//! Rows read from a kept grid: every row against the pair list, on
//! orthorhombic, triclinic, skewed, open, and mixed cells, over split
//! ranges, and from several threads.

use linkcell::{pairs_within, pairs_within_pbc, Cell, Error, Pair, PbcQuery, RowGrid};

fn rng(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed;
    move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// `n` points with fractional coordinates drawn from `[lo, hi)` of the
/// lattice `h`, so a range past `[0, 1)` leaves points outside the cell.
fn points(h: [[f64; 3]; 3], n: usize, lo: f64, hi: f64, seed: u64) -> Vec<[f64; 3]> {
    let mut next = rng(seed);
    (0..n)
        .map(|_| {
            let s = [
                lo + (hi - lo) * next(),
                lo + (hi - lo) * next(),
                lo + (hi - lo) * next(),
            ];
            std::array::from_fn(|a| s[0] * h[0][a] + s[1] * h[1][a] + s[2] * h[2][a])
        })
        .collect()
}

fn ortho(lx: f64, ly: f64, lz: f64) -> [[f64; 3]; 3] {
    [[lx, 0.0, 0.0], [0.0, ly, 0.0], [0.0, 0.0, lz]]
}

/// A lower-triangular cell of edge `edge` with tilts `tilt * edge`.
fn tilted(edge: f64, tilt: [f64; 3]) -> [[f64; 3]; 3] {
    [
        [edge, 0.0, 0.0],
        [tilt[0] * edge, edge, 0.0],
        [tilt[1] * edge, tilt[2] * edge, edge],
    ]
}

/// A cell with no zero entry and no axis along a Cartesian one.
fn skewed(edge: f64) -> [[f64; 3]; 3] {
    [
        [edge, 0.21 * edge, -0.13 * edge],
        [-0.34 * edge, 0.93 * edge, 0.17 * edge],
        [0.27 * edge, -0.41 * edge, 1.08 * edge],
    ]
}

fn cell_of(h: [[f64; 3]; 3]) -> Cell {
    Cell::from_vectors(h[0], h[1], h[2], [0.0; 3]).unwrap()
}

type Row = (usize, usize, [i32; 3], u64);

/// Name, lattice, point count, fractional range, cutoff, and bin edge.
type PeriodicCase = (
    &'static str,
    [[f64; 3]; 3],
    usize,
    (f64, f64),
    f64,
    Option<f64>,
);

/// Name, lattice, and periodic axes.
type ThreadCase = (&'static str, [[f64; 3]; 3], [bool; 3]);

fn key(pair: &Pair) -> Row {
    (pair.i, pair.j, pair.shift, pair.dist2.to_bits())
}

/// The rows of `0..n` in visit order.
fn visited(grid: &RowGrid, half: bool) -> Vec<Row> {
    let mut rows = Vec::new();
    grid.for_rows(0..grid.points(), half, |i, j, shift, dist2| {
        rows.push((i, j, shift, dist2.to_bits()))
    })
    .unwrap();
    rows
}

fn query_of<'a>(
    periodic: [bool; 3],
    cutoff: f64,
    mask: Option<&'a [bool]>,
    hint: Option<f64>,
    closed: bool,
    no_self: bool,
) -> PbcQuery<'a> {
    PbcQuery::new(periodic, cutoff)
        .mask(mask)
        .cell_hint(hint)
        .closed(closed)
        .exclude_self_images(no_self)
}

/// The grid's rows, half and full, are the pair list's rows: the same
/// sources, targets, and shifts, and the same bits in every distance.
/// Split ranges, single points, and the whole range visit each row with
/// the same entries in the same order. Returns the full row count.
#[allow(clippy::too_many_arguments)]
fn assert_rows(
    name: &str,
    xyz: &[[f64; 3]],
    cell: &Cell,
    periodic: [bool; 3],
    cutoff: f64,
    mask: Option<&[bool]>,
    hint: Option<f64>,
    closed: bool,
    no_self: bool,
) -> usize {
    let query = query_of(periodic, cutoff, mask, hint, closed, no_self);
    let grid = RowGrid::build(xyz, cell, &query).unwrap();
    assert_eq!(grid.points(), xyz.len(), "{name}");
    assert_eq!(grid.cutoff(), cutoff, "{name}");
    assert_eq!(grid.periodic(), periodic, "{name}");
    let mut full_rows = 0;
    for half in [false, true] {
        let listed = query_of(periodic, cutoff, mask, hint, closed, no_self).half(half);
        let mut want: Vec<Row> = pairs_within_pbc(xyz, cell, &listed)
            .unwrap()
            .iter()
            .map(key)
            .collect();
        want.sort_unstable();
        let whole = visited(&grid, half);
        let mut got = whole.clone();
        got.sort_unstable();
        assert_eq!(got.len(), want.len(), "{name} half={half}: row count");
        assert_eq!(got, want, "{name} half={half}");
        if !half {
            full_rows = got.len();
        }
        // Rows come in ascending point order.
        assert!(whole.windows(2).all(|w| w[0].0 <= w[1].0), "{name}");
        // Disjoint ranges, in order, are the whole walk.
        let n = xyz.len();
        for parts in [2usize, 3, 7] {
            let mut joined = Vec::new();
            for part in 0..parts {
                let (lo, hi) = (part * n / parts, (part + 1) * n / parts);
                grid.for_rows(lo..hi, half, |i, j, shift, dist2| {
                    assert!((lo..hi).contains(&i));
                    joined.push((i, j, shift, dist2.to_bits()));
                })
                .unwrap();
            }
            assert_eq!(joined, whole, "{name} half={half} parts={parts}");
        }
        // One point at a time is the same walk.
        let mut single = Vec::new();
        for i in 0..n {
            grid.for_point(i, half, |j, shift, dist2| {
                single.push((i, j, shift, dist2.to_bits()))
            })
            .unwrap();
        }
        assert_eq!(single, whole, "{name} half={half}: for_point");
    }
    full_rows
}

const PPP: [bool; 3] = [true, true, true];

#[test]
fn periodic_rows_match_the_pair_list() {
    let cases: Vec<PeriodicCase> = vec![
        ("ortho", ortho(21.0, 23.0, 19.0), 600, (0.0, 1.0), 3.5, None),
        (
            "ortho outside",
            ortho(21.0, 23.0, 19.0),
            600,
            (-1.3, 2.4),
            3.5,
            None,
        ),
        (
            "ortho fine bins",
            ortho(21.0, 23.0, 19.0),
            600,
            (0.0, 1.0),
            3.5,
            Some(1.2),
        ),
        (
            "ortho two bins",
            ortho(8.0, 8.5, 9.0),
            150,
            (-0.2, 1.2),
            3.5,
            None,
        ),
        (
            "ortho past half",
            ortho(6.0, 6.5, 7.0),
            60,
            (0.0, 1.0),
            4.0,
            None,
        ),
        (
            "ortho several images",
            ortho(3.0, 3.5, 4.0),
            12,
            (-0.5, 1.5),
            7.5,
            None,
        ),
        (
            "slab-like",
            ortho(30.0, 4.0, 30.0),
            300,
            (0.0, 1.0),
            3.0,
            None,
        ),
        (
            "triclinic",
            tilted(20.0, [0.3, -0.2, 0.25]),
            600,
            (0.0, 1.0),
            3.5,
            None,
        ),
        (
            "triclinic outside",
            tilted(20.0, [0.3, -0.2, 0.25]),
            600,
            (-1.1, 2.1),
            3.5,
            None,
        ),
        (
            "triclinic past half",
            tilted(6.5, [0.4, 0.3, -0.35]),
            70,
            (-0.1, 1.1),
            4.5,
            None,
        ),
        ("skewed", skewed(18.0), 500, (0.0, 1.0), 3.5, None),
        (
            "skewed outside",
            skewed(18.0),
            500,
            (-2.2, 3.1),
            3.5,
            Some(2.0),
        ),
        ("skewed small", skewed(5.0), 40, (-0.3, 1.3), 4.0, None),
    ];
    for (k, (name, h, n, (lo, hi), cutoff, hint)) in cases.into_iter().enumerate() {
        let cell = cell_of(h);
        let xyz = points(h, n, lo, hi, 0x9e37_79b9 + k as u64);
        let rows = assert_rows(name, &xyz, &cell, PPP, cutoff, None, hint, false, false);
        assert!(rows > 0, "{name} has no rows");
        // The same rows through the fully periodic entry.
        let grid =
            RowGrid::build(&xyz, &cell, &PbcQuery::periodic(cutoff).cell_hint(hint)).unwrap();
        for half in [false, true] {
            let mut want: Vec<Row> = pairs_within(&xyz, &cell, cutoff, None, hint, half)
                .unwrap()
                .iter()
                .map(key)
                .collect();
            want.sort_unstable();
            let mut got = visited(&grid, half);
            got.sort_unstable();
            assert_eq!(got, want, "{name} half={half}: pairs_within");
        }
        assert_rows(name, &xyz, &cell, PPP, cutoff, None, hint, false, true);
        let mask: Vec<bool> = (0..n).map(|i| i % 3 != 1).collect();
        assert_rows(
            name,
            &xyz,
            &cell,
            PPP,
            cutoff,
            Some(&mask),
            hint,
            false,
            false,
        );
    }
}

#[test]
fn open_and_mixed_rows_match_the_pair_list() {
    let lattices: Vec<(&str, [[f64; 3]; 3])> = vec![
        ("ortho", ortho(14.0, 15.0, 13.0)),
        ("triclinic", tilted(14.0, [0.3, -0.2, 0.25])),
        ("skewed", skewed(12.0)),
        ("narrow", ortho(5.0, 5.5, 6.0)),
    ];
    let masks = [
        [false, false, false],
        [true, false, false],
        [false, true, false],
        [false, false, true],
        [true, true, false],
        [true, false, true],
        [false, true, true],
    ];
    let mut seed = 0x5151_u64;
    for (lattice, h) in &lattices {
        let cell = cell_of(*h);
        for periodic in masks {
            for (lo, hi) in [(0.0, 1.0), (-0.7, 1.6)] {
                seed += 1;
                let n = if *lattice == "narrow" { 50 } else { 160 };
                let xyz = points(*h, n, lo, hi, seed);
                let name = format!("{lattice} {periodic:?} [{lo}, {hi})");
                let cutoff = 3.5;
                let rows = assert_rows(
                    &name, &xyz, &cell, periodic, cutoff, None, None, false, false,
                );
                assert!(rows > 0, "{name} has no rows");
                if *lattice == "triclinic" {
                    let fine = Some(2.0);
                    assert_rows(
                        &name, &xyz, &cell, periodic, cutoff, None, fine, false, false,
                    );
                }
                assert_rows(&name, &xyz, &cell, periodic, cutoff, None, None, true, true);
                let mask: Vec<bool> = (0..n).map(|i| i % 4 != 2).collect();
                assert_rows(
                    &name,
                    &xyz,
                    &cell,
                    periodic,
                    cutoff,
                    Some(&mask),
                    None,
                    false,
                    false,
                );
            }
        }
    }
}

/// A pair exactly at the cutoff is kept only by a closed query on an open
/// cell, as in the pair list.
#[test]
fn closed_keeps_the_boundary_on_an_open_cell() {
    let xyz = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let cell = Cell::ortho(1.0, 1.0, 1.0).unwrap();
    let open = RowGrid::build(&xyz, &cell, &PbcQuery::open(2.0)).unwrap();
    let closed = RowGrid::build(&xyz, &cell, &PbcQuery::open(2.0).closed(true)).unwrap();
    assert_eq!(visited(&open, true).len(), 1);
    assert_eq!(visited(&closed, true).len(), 2);
    assert_rows(
        "boundary", &xyz, &cell, [false; 3], 2.0, None, None, true, false,
    );
}

/// A point sees its own periodic images, on the half list from one side.
#[test]
fn self_images_are_rows() {
    let xyz = [[0.5, 0.5, 0.5]];
    let cell = Cell::ortho(2.0, 2.0, 2.0).unwrap();
    let grid = RowGrid::build(&xyz, &cell, &PbcQuery::periodic(3.0)).unwrap();
    let full = visited(&grid, false);
    // Six faces at 2 and twelve edges at 2 sqrt 2 are below 3; the corners
    // at 2 sqrt 3 are not.
    assert_eq!(full.len(), 18);
    assert!(full
        .iter()
        .all(|row| row.0 == 0 && row.1 == 0 && row.2 != [0, 0, 0]));
    assert_eq!(visited(&grid, true).len(), 9);
    let none = RowGrid::build(
        &xyz,
        &cell,
        &PbcQuery::periodic(3.0).exclude_self_images(true),
    )
    .unwrap();
    assert!(visited(&none, false).is_empty());
    assert_rows("self", &xyz, &cell, PPP, 3.0, None, None, false, false);
}

/// Threads walking disjoint ranges of one grid visit what one thread
/// visits, row for row.
#[test]
fn disjoint_ranges_on_several_threads_are_the_whole_walk() {
    let cases: Vec<ThreadCase> = vec![
        ("periodic", tilted(26.0, [0.2, 0.1, -0.3]), PPP),
        ("periodic small", ortho(7.0, 7.5, 8.0), PPP),
        ("mixed", ortho(26.0, 24.0, 22.0), [true, true, false]),
        ("open", ortho(26.0, 24.0, 22.0), [false; 3]),
    ];
    for (name, h, periodic) in cases {
        let cell = cell_of(h);
        let n = if name == "periodic small" { 200 } else { 2500 };
        let xyz = points(h, n, -0.2, 1.2, 0xabcdef);
        let grid = RowGrid::build(&xyz, &cell, &PbcQuery::new(periodic, 3.5)).unwrap();
        for half in [false, true] {
            let whole = visited(&grid, half);
            assert!(!whole.is_empty(), "{name}");
            for threads in [2usize, 5, 8] {
                let grid = &grid;
                let parts: Vec<Vec<Row>> = std::thread::scope(|scope| {
                    let handles: Vec<_> = (0..threads)
                        .map(|t| {
                            scope.spawn(move || {
                                let mut rows = Vec::new();
                                // Each thread also reads single points of
                                // its range while the others walk theirs.
                                let (lo, hi) = (t * n / threads, (t + 1) * n / threads);
                                let mid = lo + (hi - lo) / 2;
                                grid.for_rows(lo..mid, half, |i, j, shift, dist2| {
                                    rows.push((i, j, shift, dist2.to_bits()))
                                })
                                .unwrap();
                                for i in mid..hi {
                                    grid.for_point(i, half, |j, shift, dist2| {
                                        rows.push((i, j, shift, dist2.to_bits()))
                                    })
                                    .unwrap();
                                }
                                rows
                            })
                        })
                        .collect();
                    handles.into_iter().map(|h| h.join().unwrap()).collect()
                });
                let joined: Vec<Row> = parts.into_iter().flatten().collect();
                assert_eq!(joined, whole, "{name} half={half} threads={threads}");
            }
        }
    }
}

/// An update bins new points in place and reads the rows a new grid
/// would. A failed update leaves no rows.
#[test]
fn update_replaces_the_bins() {
    let h = ortho(15.0, 16.0, 17.0);
    let cell = cell_of(h);
    let first = points(h, 300, 0.0, 1.0, 11);
    let second = points(h, 420, -0.5, 1.5, 12);
    let mut grid = RowGrid::build(&first, &cell, &PbcQuery::periodic(3.0)).unwrap();
    let bins = grid.bins();
    assert!(bins > 0);
    grid.update(&second, &cell, &PbcQuery::periodic(3.0))
        .unwrap();
    assert_eq!(grid.points(), 420);
    let fresh = RowGrid::build(&second, &cell, &PbcQuery::periodic(3.0)).unwrap();
    assert_eq!(visited(&grid, false), visited(&fresh, false));
    // The same grid, now with an open axis and another cutoff.
    let mixed = PbcQuery::new([true, false, true], 2.5);
    grid.update(&first, &cell, &mixed).unwrap();
    let fresh = RowGrid::build(&first, &cell, &mixed).unwrap();
    assert_eq!(grid.points(), 300);
    assert_eq!(grid.cutoff(), 2.5);
    assert_eq!(visited(&grid, true), visited(&fresh, true));
    assert_eq!(
        grid.update(&second, &cell, &PbcQuery::periodic(-1.0)),
        Err(Error::BadCutoff)
    );
    assert_eq!(grid.points(), 0);
    assert_eq!(grid.bins(), 0);
    assert_eq!(
        grid.for_rows(0..1, false, |_, _, _, _| ()),
        Err(Error::IndexOutOfRange)
    );
    grid.update(&second, &cell, &PbcQuery::periodic(3.0))
        .unwrap();
    grid.clear();
    assert_eq!(grid.points(), 0);
}

#[test]
fn bad_arguments_are_errors() {
    let cell = Cell::ortho(10.0, 10.0, 10.0).unwrap();
    let xyz = [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]];
    let built = |xyz: &[[f64; 3]], query: &PbcQuery<'_>| RowGrid::build(xyz, &cell, query).err();
    assert_eq!(
        built(&xyz, &PbcQuery::periodic(0.0)),
        Some(Error::BadCutoff)
    );
    assert_eq!(
        built(&xyz, &PbcQuery::periodic(f64::NAN)),
        Some(Error::BadCutoff)
    );
    assert_eq!(built(&[], &PbcQuery::periodic(1.0)), Some(Error::Empty));
    assert_eq!(built(&[], &PbcQuery::open(1.0)), Some(Error::Empty));
    let short = [true];
    assert_eq!(
        built(&xyz, &PbcQuery::periodic(1.0).mask(Some(&short))),
        Some(Error::MaskLen)
    );
    let bad = [[1.0, f64::INFINITY, 1.0], [2.0, 1.0, 1.0]];
    assert_eq!(
        built(&bad, &PbcQuery::periodic(1.0)),
        Some(Error::NonFinite)
    );
    assert_eq!(built(&bad, &PbcQuery::open(1.0)), Some(Error::NonFinite));

    let grid = RowGrid::build(&xyz, &cell, &PbcQuery::periodic(1.5)).unwrap();
    let mut calls = 0;
    assert_eq!(
        grid.for_rows(0..3, false, |_, _, _, _| calls += 1),
        Err(Error::IndexOutOfRange)
    );
    assert_eq!(
        grid.for_point(2, false, |_, _, _| calls += 1),
        Err(Error::IndexOutOfRange)
    );
    assert_eq!(calls, 0);
    // An empty or reversed range visits nothing, wherever it starts.
    #[allow(clippy::reversed_empty_ranges)]
    {
        grid.for_rows(2..2, false, |_, _, _, _| calls += 1).unwrap();
        grid.for_rows(9..3, false, |_, _, _, _| calls += 1).unwrap();
    }
    assert_eq!(calls, 0);
    // No active point: every row is empty.
    let off = [false, false];
    let masked = RowGrid::build(&xyz, &cell, &PbcQuery::periodic(1.5).mask(Some(&off))).unwrap();
    assert_eq!(masked.points(), 2);
    assert!(visited(&masked, false).is_empty());
    let masked = RowGrid::build(&xyz, &cell, &PbcQuery::open(1.5).mask(Some(&off))).unwrap();
    assert!(visited(&masked, false).is_empty());
}
