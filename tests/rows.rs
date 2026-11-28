//! Rows read from a kept grid: every row against the pair list, on
//! orthorhombic, triclinic, skewed, open, and mixed cells, over split
//! ranges, from several threads, and through the C ABI.

use std::ops::ControlFlow;

use linkcell::{
    pairs_within, pairs_within_pbc, Cell, Error, Pair, PbcQuery, Row as Entry, RowCursor, RowGrid,
};

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

fn entry_key(row: &Entry) -> Row {
    (row.i, row.j, row.shift, row.dist2.to_bits())
}

fn key(pair: &Pair) -> Row {
    (pair.i, pair.j, pair.shift, pair.dist2.to_bits())
}

/// The rows of `0..n` in visit order.
fn visited(grid: &RowGrid, half: bool) -> Vec<Row> {
    let mut rows = Vec::new();
    grid.for_rows(0..grid.points(), half, |row| rows.push(entry_key(row)))
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
        assert_vectors(name, xyz, cell, periodic, &grid, half);
        // Rows come in ascending point order.
        assert!(whole.windows(2).all(|w| w[0].0 <= w[1].0), "{name}");
        // Disjoint ranges, in order, are the whole walk.
        let n = xyz.len();
        for parts in [2usize, 3, 7] {
            let mut joined = Vec::new();
            for part in 0..parts {
                let (lo, hi) = (part * n / parts, (part + 1) * n / parts);
                grid.for_rows(lo..hi, half, |row| {
                    assert!((lo..hi).contains(&row.i));
                    joined.push(entry_key(row));
                })
                .unwrap();
            }
            assert_eq!(joined, whole, "{name} half={half} parts={parts}");
        }
        // One point at a time is the same walk.
        let mut single = Vec::new();
        for i in 0..n {
            grid.for_point(i, half, |row| {
                assert_eq!(row.i, i);
                single.push(entry_key(row))
            })
            .unwrap();
        }
        assert_eq!(single, whole, "{name} half={half}: for_point");
        // Buffers of any length, also shorter than a row, read the same
        // entries in the same order.
        for cap in [1usize, 3, 64, whole.len() + 5] {
            let blank = Entry {
                i: 0,
                j: 0,
                shift: [0; 3],
                vector: [0.0; 3],
                dist2: 0.0,
            };
            let mut buffer = vec![blank; cap];
            let mut cursor = RowCursor::at(0);
            let mut filled = Vec::new();
            let mut calls = 0;
            while cursor.point < n {
                let got = grid.fill_rows(&mut cursor, n, half, &mut buffer).unwrap();
                assert!(got <= cap);
                assert!(
                    got > 0 || cursor.point == n,
                    "{name}: an empty read before the end"
                );
                filled.extend(buffer[..got].iter().map(entry_key));
                calls += 1;
                assert!(
                    calls <= whole.len() + n + 2,
                    "{name}: the read does not advance"
                );
            }
            assert_eq!(cursor, RowCursor::at(n));
            assert_eq!(filled, whole, "{name} half={half} cap={cap}: fill_rows");
        }
    }
    full_rows
}

/// Every vector squares to its `dist2` bit for bit and is `r_j - r_i`
/// with the shift applied. On a full list the two entries of a pair hold
/// opposite shifts, opposite vectors, and the same distance.
fn assert_vectors(
    name: &str,
    xyz: &[[f64; 3]],
    cell: &Cell,
    periodic: [bool; 3],
    grid: &RowGrid,
    half: bool,
) {
    let mut seen = std::collections::BTreeMap::new();
    // The fold of a point far outside the cell rounds at the scale of
    // its coordinates.
    let scale = xyz
        .iter()
        .flatten()
        .fold(1.0f64, |most, v| most.max(v.abs()));
    grid.for_rows(0..grid.points(), half, |row| {
        let v = row.vector;
        let squared = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        assert_eq!(squared.to_bits(), row.dist2.to_bits(), "{name}: |vector|^2");
        let lattice = if periodic == [false; 3] {
            [0.0; 3]
        } else {
            cell.lattice_shift(row.shift[0], row.shift[1], row.shift[2])
        };
        for a in 0..3 {
            let want = xyz[row.j][a] + lattice[a] - xyz[row.i][a];
            assert!(
                (v[a] - want).abs() <= 1.0e-12 * scale,
                "{name}: vector {v:?} of row {row:?}, expected component {a} = {want}"
            );
        }
        let bits = v.map(|x| (x + 0.0).to_bits());
        assert!(
            seen.insert((row.i, row.j, row.shift), bits).is_none(),
            "{name}: row {row:?} twice"
        );
    })
    .unwrap();
    if half {
        return;
    }
    for (&(i, j, shift), bits) in &seen {
        let back = seen
            .get(&(j, i, [-shift[0], -shift[1], -shift[2]]))
            .unwrap_or_else(|| panic!("{name}: no reverse of ({i}, {j}, {shift:?})"));
        let negated = bits.map(|b| (0.0 - f64::from_bits(b)).to_bits());
        assert_eq!(
            *back, negated,
            "{name}: reverse vector of ({i}, {j}, {shift:?})"
        );
    }
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
        assert_rows(name, &xyz, &cell, PPP, cutoff, None, hint, true, false);
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

/// The same on a periodic cell: a pair at the cutoff, directly and through
/// an image, is kept by a closed query and dropped by a strict one, as in
/// the pair list.
#[test]
fn closed_keeps_the_boundary_on_a_periodic_cell() {
    // 0-1 are 2 apart inside the cell, 0-2 are 2 apart through the x face,
    // and 0-3 are 2.5 apart.
    let xyz = [
        [1.0, 1.0, 1.0],
        [3.0, 1.0, 1.0],
        [9.0, 1.0, 1.0],
        [1.0, 3.5, 1.0],
    ];
    for h in [ortho(10.0, 12.0, 14.0), tilted(10.0, [0.0, 0.3, 0.2])] {
        let cell = cell_of(h);
        let strict = RowGrid::build(&xyz, &cell, &PbcQuery::periodic(2.0)).unwrap();
        let closed = RowGrid::build(&xyz, &cell, &PbcQuery::periodic(2.0).closed(true)).unwrap();
        assert!(visited(&strict, true).is_empty());
        // Entries of a row come in bin order, so sort them by target.
        let mut kept = visited(&closed, true);
        kept.sort_unstable();
        assert_eq!(kept.len(), 2, "{kept:?}");
        assert_eq!((kept[0].0, kept[0].1, kept[0].2), (0, 1, [0, 0, 0]));
        assert_eq!((kept[1].0, kept[1].1, kept[1].2), (0, 2, [-1, 0, 0]));
        assert!(kept.iter().all(|row| f64::from_bits(row.3) == 4.0));
        assert_eq!(visited(&closed, false).len(), 4);
        assert_eq!(strict.cutoff(), closed.cutoff());
        assert_rows("boundary", &xyz, &cell, PPP, 2.0, None, None, true, false);
        // One double below the distance, a closed cutoff drops them again.
        let below = f64::from_bits(2.0f64.to_bits() - 1);
        let short = RowGrid::build(&xyz, &cell, &PbcQuery::periodic(below).closed(true)).unwrap();
        assert!(visited(&short, true).is_empty());
    }
}

/// The candidate bound counts, for each point, the points its row reads:
/// it is never below the number of full-row entries, it needs no distance,
/// and it is zero without points.
#[test]
fn the_candidate_bound_covers_every_row() {
    let cases: Vec<ThreadCase> = vec![
        ("periodic", tilted(20.0, [0.2, 0.1, -0.3]), PPP),
        ("periodic small", ortho(7.0, 7.5, 8.0), PPP),
        ("mixed", ortho(18.0, 16.0, 17.0), [true, true, false]),
        ("open", ortho(18.0, 16.0, 17.0), [false; 3]),
    ];
    for (name, h, periodic) in cases {
        let cell = cell_of(h);
        let n = if name == "periodic small" { 150 } else { 600 };
        let xyz = points(h, n, 0.0, 1.0, 0x51ed);
        let grid = RowGrid::build(&xyz, &cell, &PbcQuery::new(periodic, 3.5)).unwrap();
        let rows = visited(&grid, false).len() as u64;
        let bound = grid.candidate_bound().unwrap();
        assert!(rows > 0, "{name}");
        assert!(bound >= rows, "{name}: bound {bound} below {rows} rows");
        // A stencil of bins no finer than the cutoff holds a bounded
        // multiple of the cutoff sphere.
        assert!(
            bound <= 64 * rows + (n * n) as u64,
            "{name}: bound {bound} for {rows} rows"
        );
        assert!(grid.bins() > 0);
    }
    let cell = Cell::ortho(10.0, 10.0, 10.0).unwrap();
    let xyz = [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]];
    let off = [false, false];
    let masked = RowGrid::build(&xyz, &cell, &PbcQuery::periodic(1.5).mask(Some(&off))).unwrap();
    assert_eq!(masked.candidate_bound(), Ok(0));
    let pair = RowGrid::build(&xyz, &cell, &PbcQuery::open(1.5)).unwrap();
    assert_eq!(pair.candidate_bound(), Ok(2));
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
                                grid.for_rows(lo..mid, half, |row| rows.push(entry_key(row)))
                                    .unwrap();
                                for i in mid..hi {
                                    grid.for_point(i, half, |row| rows.push(entry_key(row)))
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
        grid.for_rows(0..1, false, |_| ()),
        Err(Error::IndexOutOfRange)
    );
    grid.update(&second, &cell, &PbcQuery::periodic(3.0))
        .unwrap();
    grid.clear();
    assert_eq!(grid.points(), 0);
}

/// A visit that breaks ends the walk at that entry and hands its value
/// back.
#[test]
fn a_visit_can_stop_the_walk() {
    for periodic in [PPP, [true, false, true], [false; 3]] {
        let h = ortho(12.0, 13.0, 14.0);
        let cell = cell_of(h);
        let xyz = points(h, 200, 0.0, 1.0, 99);
        let grid = RowGrid::build(&xyz, &cell, &PbcQuery::new(periodic, 3.5)).unwrap();
        let whole = visited(&grid, false);
        assert!(whole.len() > 20);
        for stop_at in [0usize, 1, 7, whole.len() - 1] {
            let mut seen = Vec::new();
            let walked = grid
                .try_for_rows(0..grid.points(), false, |row| {
                    seen.push(entry_key(row));
                    if seen.len() == stop_at + 1 {
                        ControlFlow::Break(seen.len())
                    } else {
                        ControlFlow::Continue(())
                    }
                })
                .unwrap();
            assert_eq!(walked, ControlFlow::Break(stop_at + 1));
            assert_eq!(seen, whole[..=stop_at]);
        }
        let walked = grid
            .try_for_rows(0..grid.points(), false, |_| ControlFlow::<()>::Continue(()))
            .unwrap();
        assert_eq!(walked, ControlFlow::Continue(()));
        // One point: stop at its first entry.
        let first = whole[0].0;
        let mut calls = 0;
        let walked = grid
            .try_for_point(first, false, |row| {
                calls += 1;
                ControlFlow::Break(row.j)
            })
            .unwrap();
        assert_eq!(walked, ControlFlow::Break(whole[0].1));
        assert_eq!(calls, 1);
    }
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
        grid.for_rows(0..3, false, |_| calls += 1),
        Err(Error::IndexOutOfRange)
    );
    assert_eq!(
        grid.for_point(2, false, |_| calls += 1),
        Err(Error::IndexOutOfRange)
    );
    assert_eq!(calls, 0);
    let mut cursor = RowCursor::at(0);
    assert_eq!(
        grid.fill_rows(&mut cursor, 3, false, &mut []),
        Err(Error::IndexOutOfRange)
    );
    let mut cursor = RowCursor::at(2);
    assert_eq!(
        grid.fill_rows(&mut cursor, 1, false, &mut []),
        Err(Error::IndexOutOfRange)
    );
    assert_eq!(
        grid.fill_rows(&mut RowCursor::at(0), 2, false, &mut []),
        Ok(0)
    );
    // An empty or reversed range visits nothing, wherever it starts.
    #[allow(clippy::reversed_empty_ranges)]
    {
        grid.for_rows(2..2, false, |_| calls += 1).unwrap();
        grid.for_rows(9..3, false, |_| calls += 1).unwrap();
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

#[cfg(feature = "capi")]
mod c_abi {
    use super::*;
    use std::ffi::{c_char, c_int, c_void, CStr};

    #[repr(C)]
    struct LcCell {
        h: [f64; 9],
        origin: [f64; 3],
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct LcRow {
        i: c_int,
        j: c_int,
        shift: [c_int; 3],
        vector: [f64; 3],
        dist2: f64,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct LcAbi {
        major: u16,
        minor: u16,
        layout_revision: u16,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    struct LcCaps {
        size: u32,
        row_grid: c_int,
        row_vector: c_int,
        row_stop: c_int,
        row_request: c_int,
        open_axes: c_int,
        threads: c_int,
        row_fill: c_int,
        closed_periodic: c_int,
        row_bound: c_int,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    struct LcRowCursor {
        point: usize,
        entry: usize,
    }

    type BuildFn = unsafe extern "C" fn(
        *const f64,
        usize,
        *const LcCell,
        *const c_int,
        f64,
        *const c_int,
        f64,
        c_int,
        c_int,
        *mut *mut LcRowGrid,
    ) -> c_int;
    type RequestFn = unsafe extern "C" fn(
        *const f64,
        usize,
        *const c_int,
        *const u8,
        usize,
        *mut c_int,
        *mut *mut LcRowGrid,
    ) -> c_int;
    type UpdateFn = unsafe extern "C" fn(
        *mut LcRowGrid,
        *const f64,
        usize,
        *const LcCell,
        *const c_int,
        f64,
        *const c_int,
        f64,
        c_int,
        c_int,
    ) -> c_int;
    type FillFn = unsafe extern "C" fn(
        *const LcRowGrid,
        *mut LcRowCursor,
        usize,
        c_int,
        *mut LcRow,
        usize,
        *mut usize,
    ) -> c_int;

    /// `lc_row_ops` as a host compiled it.
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct LcRowOps {
        size: u32,
        build: Option<BuildFn>,
        build_request: Option<RequestFn>,
        update: Option<UpdateFn>,
        len: Option<unsafe extern "C" fn(*const LcRowGrid) -> usize>,
        free: Option<unsafe extern "C" fn(*mut LcRowGrid)>,
        rows: Option<
            unsafe extern "C" fn(
                *const LcRowGrid,
                usize,
                usize,
                c_int,
                Visit,
                *mut c_void,
            ) -> c_int,
        >,
        point: Option<
            unsafe extern "C" fn(*const LcRowGrid, usize, c_int, Visit, *mut c_void) -> c_int,
        >,
        fill: Option<FillFn>,
        last_error: Option<unsafe extern "C" fn() -> *const c_char>,
        capabilities: Option<unsafe extern "C" fn(*mut LcCaps) -> c_int>,
        bins: Option<unsafe extern "C" fn(*const LcRowGrid) -> usize>,
        bound: Option<unsafe extern "C" fn(*const LcRowGrid, *mut u64) -> c_int>,
    }

    #[repr(C)]
    struct LcRowGrid {
        _private: u8,
    }

    type Visit = Option<unsafe extern "C" fn(*mut c_void, *const LcRow) -> c_int>;

    extern "C" {
        fn lc_last_error() -> *const c_char;
        fn lc_abi_stamp() -> LcAbi;
        fn lc_abi_compatible(host: *const LcAbi) -> c_int;
        fn lc_capabilities(out: *mut LcCaps) -> c_int;
        fn lc_row_grid_build_request(
            xyz: *const f64,
            n: usize,
            mask: *const c_int,
            request: *const u8,
            request_len: usize,
            out_half: *mut c_int,
            out: *mut *mut LcRowGrid,
        ) -> c_int;
        fn lc_row_ops_get(out: *mut LcRowOps) -> c_int;
        fn lc_row_grid_fill(
            grid: *const LcRowGrid,
            cursor: *mut LcRowCursor,
            end: usize,
            half: c_int,
            out: *mut LcRow,
            cap: usize,
            count: *mut usize,
        ) -> c_int;
        fn lc_row_grid_build(
            xyz: *const f64,
            n: usize,
            simbox: *const LcCell,
            pbc: *const c_int,
            cutoff: f64,
            mask: *const c_int,
            cell_hint: f64,
            closed: c_int,
            exclude_self_images: c_int,
            out: *mut *mut LcRowGrid,
        ) -> c_int;
        fn lc_row_grid_update(
            grid: *mut LcRowGrid,
            xyz: *const f64,
            n: usize,
            simbox: *const LcCell,
            pbc: *const c_int,
            cutoff: f64,
            mask: *const c_int,
            cell_hint: f64,
            closed: c_int,
            exclude_self_images: c_int,
        ) -> c_int;
        fn lc_row_grid_len(grid: *const LcRowGrid) -> usize;
        fn lc_row_grid_free(grid: *mut LcRowGrid);
        fn lc_row_grid_rows(
            grid: *const LcRowGrid,
            begin: usize,
            end: usize,
            half: c_int,
            visit: Visit,
            user: *mut c_void,
        ) -> c_int;
        fn lc_row_grid_point(
            grid: *const LcRowGrid,
            i: usize,
            half: c_int,
            visit: Visit,
            user: *mut c_void,
        ) -> c_int;
    }

    /// Rows and the vectors that came with them.
    #[derive(Default)]
    struct Seen {
        rows: Vec<Row>,
        vectors: Vec<[u64; 3]>,
        /// Stop once this many rows are held; zero never stops.
        stop_after: usize,
    }

    unsafe extern "C" fn collect(user: *mut c_void, row: *const LcRow) -> c_int {
        let seen = unsafe { &mut *(user as *mut Seen) };
        let row = unsafe { *row };
        seen.rows.push((
            row.i as usize,
            row.j as usize,
            row.shift,
            row.dist2.to_bits(),
        ));
        seen.vectors.push(row.vector.map(f64::to_bits));
        if seen.rows.len() == seen.stop_after {
            7
        } else {
            0
        }
    }

    fn no_error() -> bool {
        unsafe { lc_last_error() }.is_null()
    }

    fn last_error() -> String {
        let msg = unsafe { lc_last_error() };
        assert!(!msg.is_null());
        unsafe { CStr::from_ptr(msg) }.to_str().unwrap().to_owned()
    }

    fn raw(h: [[f64; 3]; 3]) -> LcCell {
        LcCell {
            h: [
                h[0][0], h[0][1], h[0][2], h[1][0], h[1][1], h[1][2], h[2][0], h[2][1], h[2][2],
            ],
            origin: [0.0; 3],
        }
    }

    #[test]
    fn c_rows_are_the_rust_rows() {
        let h = tilted(16.0, [0.25, -0.15, 0.2]);
        let cell = cell_of(h);
        let c_cell = raw(h);
        let xyz = points(h, 400, -0.6, 1.6, 77);
        let n = xyz.len();
        for periodic in [PPP, [true, false, true], [false; 3]] {
            let pbc = periodic.map(c_int::from);
            let mut grid = std::ptr::null_mut();
            let code = unsafe {
                lc_row_grid_build(
                    xyz.as_ptr().cast(),
                    n,
                    &c_cell,
                    pbc.as_ptr(),
                    3.5,
                    std::ptr::null(),
                    0.0,
                    0,
                    0,
                    &mut grid,
                )
            };
            assert_eq!(code, 0, "{}", last_error());
            assert_eq!(unsafe { lc_row_grid_len(grid) }, n);
            let rust = RowGrid::build(&xyz, &cell, &PbcQuery::new(periodic, 3.5)).unwrap();
            for half in [false, true] {
                let want = visited(&rust, half);
                assert!(!want.is_empty());
                let mut got = Seen::default();
                let user = (&mut got as *mut Seen).cast();
                let mid = n / 3;
                assert_eq!(
                    unsafe {
                        lc_row_grid_rows(grid, 0, mid, c_int::from(half), Some(collect), user)
                    },
                    0
                );
                for i in mid..n {
                    assert_eq!(
                        unsafe {
                            lc_row_grid_point(grid, i, c_int::from(half), Some(collect), user)
                        },
                        0
                    );
                }
                assert_eq!(got.rows, want, "{periodic:?} half={half}");
                let mut vectors = Vec::new();
                rust.for_rows(0..n, half, |row| vectors.push(row.vector.map(f64::to_bits)))
                    .unwrap();
                assert_eq!(got.vectors, vectors, "{periodic:?} half={half}: vectors");
                // A nonzero return stops the walk and comes back, with no error.
                let mut some = Seen {
                    stop_after: 5,
                    ..Seen::default()
                };
                let user = (&mut some as *mut Seen).cast();
                assert_eq!(
                    unsafe { lc_row_grid_rows(grid, 0, n, c_int::from(half), Some(collect), user) },
                    7
                );
                assert!(no_error());
                assert_eq!(some.rows, want[..5]);
            }
            // Out of range, a null callback, and a null grid are errors.
            let mut got = Seen::default();
            let user = (&mut got as *mut Seen).cast();
            assert_ne!(
                unsafe { lc_row_grid_rows(grid, 0, n + 1, 0, Some(collect), user) },
                0
            );
            assert_eq!(last_error(), "point index is out of range");
            assert_ne!(
                unsafe { lc_row_grid_point(grid, n, 0, Some(collect), user) },
                0
            );
            assert_ne!(unsafe { lc_row_grid_rows(grid, 0, n, 0, None, user) }, 0);
            assert_eq!(last_error(), "null pointer");
            assert_ne!(
                unsafe { lc_row_grid_rows(std::ptr::null(), 0, 0, 0, Some(collect), user) },
                0
            );
            assert!(got.rows.is_empty());
            // Update in place with fewer points and a mask.
            let fewer = &xyz[..150];
            let mask: Vec<c_int> = (0..150).map(|i| c_int::from(i % 5 != 0)).collect();
            let code = unsafe {
                lc_row_grid_update(
                    grid,
                    fewer.as_ptr().cast(),
                    fewer.len(),
                    &c_cell,
                    pbc.as_ptr(),
                    4.0,
                    mask.as_ptr(),
                    2.0,
                    0,
                    1,
                )
            };
            assert_eq!(code, 0, "{}", last_error());
            assert_eq!(unsafe { lc_row_grid_len(grid) }, 150);
            let keep: Vec<bool> = mask.iter().map(|&v| v != 0).collect();
            let query = PbcQuery::new(periodic, 4.0)
                .mask(Some(&keep))
                .cell_hint(Some(2.0))
                .exclude_self_images(true);
            let want = visited(&RowGrid::build(fewer, &cell, &query).unwrap(), false);
            let mut got = Seen::default();
            let user = (&mut got as *mut Seen).cast();
            assert_eq!(
                unsafe { lc_row_grid_rows(grid, 0, 150, 0, Some(collect), user) },
                0
            );
            assert_eq!(got.rows, want);
            // A failed update empties the grid and keeps the handle.
            let code = unsafe {
                lc_row_grid_update(
                    grid,
                    fewer.as_ptr().cast(),
                    fewer.len(),
                    &c_cell,
                    pbc.as_ptr(),
                    -1.0,
                    std::ptr::null(),
                    0.0,
                    0,
                    0,
                )
            };
            assert_ne!(code, 0);
            assert_eq!(last_error(), "cutoff must be finite and strictly positive");
            assert_eq!(unsafe { lc_row_grid_len(grid) }, 0);
            unsafe { lc_row_grid_free(grid) };
        }
        unsafe { lc_row_grid_free(std::ptr::null_mut()) };
        assert_eq!(unsafe { lc_row_grid_len(std::ptr::null()) }, 0);
        let mut grid = std::ptr::null_mut();
        let code = unsafe {
            lc_row_grid_build(
                xyz.as_ptr().cast(),
                0,
                &c_cell,
                std::ptr::null(),
                3.5,
                std::ptr::null(),
                0.0,
                0,
                0,
                &mut grid,
            )
        };
        assert_ne!(code, 0);
        assert!(grid.is_null());
        assert_eq!(last_error(), "no points");
    }

    /// The header constants a host compiles in, here as this test holds
    /// them, are what the library reports, and the library accepts them.
    #[test]
    fn abi_stamp_and_capabilities() {
        let stamp = unsafe { lc_abi_stamp() };
        assert_eq!(
            stamp,
            LcAbi {
                major: linkcell::LC_ABI_MAJOR,
                minor: linkcell::LC_ABI_MINOR,
                layout_revision: linkcell::LC_ABI_LAYOUT_REVISION,
            }
        );
        assert_eq!(unsafe { lc_abi_compatible(&stamp) }, 1);
        assert_eq!(unsafe { lc_abi_compatible(std::ptr::null()) }, 0);
        let newer_minor = LcAbi {
            minor: stamp.minor + 1,
            ..stamp
        };
        assert_eq!(unsafe { lc_abi_compatible(&newer_minor) }, 0);
        let other_major = LcAbi {
            major: stamp.major + 1,
            ..stamp
        };
        assert_eq!(unsafe { lc_abi_compatible(&other_major) }, 0);
        let other_layout = LcAbi {
            layout_revision: stamp.layout_revision + 1,
            ..stamp
        };
        assert_eq!(unsafe { lc_abi_compatible(&other_layout) }, 0);

        let mut caps = LcCaps {
            size: std::mem::size_of::<LcCaps>() as u32,
            ..LcCaps::default()
        };
        assert_eq!(unsafe { lc_capabilities(&mut caps) }, 0);
        assert_eq!(caps.size as usize, std::mem::size_of::<LcCaps>());
        assert_eq!(
            (
                caps.row_grid,
                caps.row_vector,
                caps.row_stop,
                caps.open_axes,
                caps.row_fill,
                caps.closed_periodic,
                caps.row_bound
            ),
            (1, 1, 1, 1, 1, 1, 1)
        );
        assert_eq!(caps.row_request, c_int::from(cfg!(feature = "request")));
        assert_eq!(caps.threads, c_int::from(cfg!(feature = "parallel")));
        // A host compiled with a shorter struct gets only what fits.
        let mut short = LcCaps {
            size: 8,
            row_vector: -5,
            ..LcCaps::default()
        };
        assert_eq!(unsafe { lc_capabilities(&mut short) }, 0);
        assert_eq!((short.size, short.row_grid, short.row_vector), (8, 1, -5));
        let mut tiny = LcCaps {
            size: 2,
            ..LcCaps::default()
        };
        assert_ne!(unsafe { lc_capabilities(&mut tiny) }, 0);
        assert_ne!(unsafe { lc_capabilities(std::ptr::null_mut()) }, 0);
        assert_eq!(last_error(), "null pointer");

        // Bytes that are no message: a decode error with request support,
        // and the sentence that names its absence without.
        let xyz = [[0.0f64; 3]; 2];
        let junk = [0xffu8; 24];
        let mut grid = std::ptr::null_mut();
        let code = unsafe {
            lc_row_grid_build_request(
                xyz.as_ptr().cast(),
                2,
                std::ptr::null(),
                junk.as_ptr(),
                junk.len(),
                std::ptr::null_mut(),
                &mut grid,
            )
        };
        assert_ne!(code, 0);
        assert!(grid.is_null());
        if caps.row_request == 1 {
            assert!(last_error().starts_with("NeighborRequest does not decode"));
        } else {
            assert_eq!(
                last_error(),
                "this library was built without NeighborRequest support"
            );
        }
    }

    /// The operations table holds the entry points a host would resolve
    /// by name, a shorter host struct gets the members that fit, and a
    /// grid driven through the table alone reads the rows, by visit and
    /// by buffer.
    #[test]
    fn the_operations_table_drives_a_grid() {
        let mut ops = LcRowOps {
            size: std::mem::size_of::<LcRowOps>() as u32,
            ..LcRowOps::default()
        };
        assert_eq!(unsafe { lc_row_ops_get(&mut ops) }, 0);
        assert_eq!(ops.size as usize, std::mem::size_of::<LcRowOps>());
        assert_eq!(
            ops.build.map(|f| f as usize),
            Some(lc_row_grid_build as *const () as usize)
        );
        assert_eq!(
            ops.rows.map(|f| f as usize),
            Some(lc_row_grid_rows as *const () as usize)
        );
        assert_eq!(
            ops.fill.map(|f| f as usize),
            Some(lc_row_grid_fill as *const () as usize)
        );
        assert_eq!(
            ops.build_request.is_some(),
            cfg!(feature = "request"),
            "build_request is null exactly when the library cannot decode a request"
        );
        // A host that knows only the first two operations.
        let known = std::mem::offset_of!(LcRowOps, update);
        let mut short = LcRowOps {
            size: known as u32,
            ..LcRowOps::default()
        };
        assert_eq!(unsafe { lc_row_ops_get(&mut short) }, 0);
        assert_eq!(short.size as usize, known);
        assert!(short.build.is_some());
        assert!(short.update.is_none() && short.fill.is_none());
        let mut tiny = LcRowOps {
            size: 3,
            ..LcRowOps::default()
        };
        assert_ne!(unsafe { lc_row_ops_get(&mut tiny) }, 0);
        assert_ne!(unsafe { lc_row_ops_get(std::ptr::null_mut()) }, 0);

        let h = tilted(14.0, [0.2, 0.1, -0.15]);
        let cell = cell_of(h);
        let c_cell = raw(h);
        let xyz = points(h, 250, -0.4, 1.4, 5);
        let n = xyz.len();
        let pbc = [1, 1, 0];
        let mut grid = std::ptr::null_mut();
        let build = ops.build.unwrap();
        let code = unsafe {
            build(
                xyz.as_ptr().cast(),
                n,
                &c_cell,
                pbc.as_ptr(),
                3.5,
                std::ptr::null(),
                0.0,
                0,
                0,
                &mut grid,
            )
        };
        assert_eq!(code, 0);
        assert_eq!(unsafe { ops.len.unwrap()(grid) }, n);
        assert_eq!(
            unsafe { ops.bins.unwrap()(grid) },
            RowGrid::build(&xyz, &cell, &PbcQuery::new([true, true, false], 3.5))
                .unwrap()
                .bins()
        );
        let mut bound = 0u64;
        assert_eq!(unsafe { ops.bound.unwrap()(grid, &mut bound) }, 0);
        assert_eq!(
            Ok(bound),
            RowGrid::build(&xyz, &cell, &PbcQuery::new([true, true, false], 3.5))
                .unwrap()
                .candidate_bound()
        );
        assert_ne!(unsafe { ops.bound.unwrap()(grid, std::ptr::null_mut()) }, 0);
        let rust = RowGrid::build(&xyz, &cell, &PbcQuery::new([true, true, false], 3.5)).unwrap();
        for half in [false, true] {
            let want = visited(&rust, half);
            let mut got = Seen::default();
            let user = (&mut got as *mut Seen).cast();
            assert_eq!(
                unsafe { ops.rows.unwrap()(grid, 0, n, c_int::from(half), Some(collect), user) },
                0
            );
            assert_eq!(got.rows, want);
            // The same rows through the buffer, 37 at a time.
            let blank = LcRow {
                i: -1,
                j: -1,
                shift: [0; 3],
                vector: [0.0; 3],
                dist2: 0.0,
            };
            let mut buffer = [blank; 37];
            let mut cursor = LcRowCursor::default();
            let mut filled = Vec::new();
            let mut vectors = Vec::new();
            while cursor.point < n {
                let mut count = usize::MAX;
                let code = unsafe {
                    ops.fill.unwrap()(
                        grid,
                        &mut cursor,
                        n,
                        c_int::from(half),
                        buffer.as_mut_ptr(),
                        buffer.len(),
                        &mut count,
                    )
                };
                assert_eq!(code, 0);
                assert!(count <= buffer.len());
                assert!(count > 0 || cursor.point == n);
                for row in &buffer[..count] {
                    filled.push((
                        row.i as usize,
                        row.j as usize,
                        row.shift,
                        row.dist2.to_bits(),
                    ));
                    vectors.push(row.vector.map(f64::to_bits));
                }
            }
            assert_eq!(filled, want);
            assert_eq!(vectors, got.vectors);
        }
        // Past the end, and a cursor past the range.
        let mut count = 0usize;
        let mut cursor = LcRowCursor::default();
        let fill = ops.fill.unwrap();
        assert_ne!(
            unsafe {
                fill(
                    grid,
                    &mut cursor,
                    n + 1,
                    0,
                    std::ptr::null_mut(),
                    0,
                    &mut count,
                )
            },
            0
        );
        assert_eq!(last_error(), "point index is out of range");
        assert_eq!(
            unsafe { fill(grid, &mut cursor, n, 0, std::ptr::null_mut(), 0, &mut count) },
            0
        );
        assert_eq!((count, cursor), (0, LcRowCursor::default()));
        assert_ne!(
            unsafe {
                fill(
                    grid,
                    std::ptr::null_mut(),
                    n,
                    0,
                    std::ptr::null_mut(),
                    0,
                    &mut count,
                )
            },
            0
        );
        let message = unsafe { ops.last_error.unwrap()() };
        assert_eq!(
            unsafe { CStr::from_ptr(message) }.to_str().unwrap(),
            "null pointer"
        );
        unsafe { ops.free.unwrap()(grid) };
    }
}
