use linkcell::{pairs_within, pairs_within_columns, Cell, Error, PairColumns};
use std::collections::BTreeMap;

type Key = (usize, usize, [i32; 3]);

fn scan(xyz: &[[f64; 3]], lattice: [[f64; 3]; 3], cutoff: f64, half: bool) -> BTreeMap<Key, f64> {
    let mut found = BTreeMap::new();
    for (i, p) in xyz.iter().enumerate() {
        for (j, q) in xyz.iter().enumerate() {
            for a in -8..=8 {
                for b in -8..=8 {
                    for c in -8..=8 {
                        let shift = [a, b, c];
                        if i == j && shift == [0; 3] {
                            continue;
                        }
                        if half && (i > j || (i == j && shift > [0; 3])) {
                            continue;
                        }
                        let d2: f64 = (0..3)
                            .map(|k| {
                                let d = q[k] - p[k]
                                    + a as f64 * lattice[0][k]
                                    + b as f64 * lattice[1][k]
                                    + c as f64 * lattice[2][k];
                                d * d
                            })
                            .sum();
                        if d2 < cutoff * cutoff {
                            found.insert((i, j, shift), d2);
                        }
                    }
                }
            }
        }
    }
    found
}

#[test]
fn unwrapped_rows_and_columns_match_raw_lattice_scan() {
    for lattice in [
        [[2.0, 0.0, 0.0], [0.0, 2.5, 0.0], [0.0, 0.0, 3.0]],
        [[2.0, 0.0, 0.0], [0.7, 2.5, 0.0], [0.3, 0.4, 3.0]],
    ] {
        let cell =
            Cell::from_vectors(lattice[0], lattice[1], lattice[2], [0.3, -0.2, 0.1]).unwrap();
        for images in [-2, 0, 2] {
            let mut xyz = [[0.4, 0.2, 0.3], [1.4, 1.1, 1.7], [0.5, 2.0, 1.0]];
            for k in 0..3 {
                xyz[1][k] += images as f64 * (lattice[0][k] - lattice[1][k] + lattice[2][k]);
            }
            for half in [false, true] {
                let want = scan(&xyz, lattice, 3.1, half);
                assert!(!want.is_empty());
                let rows = pairs_within(&xyz, &cell, 3.1, None, None, half).unwrap();
                let mut columns = PairColumns::default();
                pairs_within_columns(&xyz, &cell, 3.1, None, None, half, &mut columns).unwrap();
                assert_eq!(rows.len(), want.len());
                assert_eq!(columns.len(), want.len());
                let got: BTreeMap<_, _> = rows
                    .iter()
                    .map(|p| ((p.i, p.j, p.shift), p.dist2))
                    .collect();
                let cols: BTreeMap<_, _> = (0..columns.len())
                    .map(|t| {
                        (
                            (
                                columns.i[t] as usize,
                                columns.j[t] as usize,
                                columns.shift[t],
                            ),
                            columns.dist2[t],
                        )
                    })
                    .collect();
                assert_eq!(got, cols);
                assert_eq!(
                    got.keys().collect::<Vec<_>>(),
                    want.keys().collect::<Vec<_>>()
                );
                for (key, d2) in &want {
                    assert!(
                        (got[key] - d2).abs() < 1e-11,
                        "{key:?}: {} vs {d2}",
                        got[key]
                    );
                }
            }
        }
    }
}

#[test]
fn unrepresentable_relative_image_span_is_an_error() {
    let cell = Cell::ortho(1.0, 1.0, 1.0).unwrap();
    let xyz = [[0.0; 3], [i32::MAX as f64 + 1.0, 0.0, 0.0]];
    assert!(matches!(
        pairs_within(&xyz, &cell, 0.4, None, None, false),
        Err(Error::Overflow)
    ));
    let rows = pairs_within(&xyz, &cell, 0.4, Some(&[true, false]), None, false).unwrap();
    assert!(rows.is_empty());
}
