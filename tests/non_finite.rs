//! Every k-nearest entry rejects a NaN or infinite coordinate, and a
//! non-finite cell origin, as [`Error::NonFinite`] before any search.

use linkcell::{
    knearest, knearest_brute, knearest_csr, knearest_into, knearest_into_d2, knearest_into_many,
    Cell, Error,
};

const BAD: [f64; 3] = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY];

fn ortho() -> Cell {
    Cell::ortho(10.0, 10.0, 10.0).unwrap()
}

fn sheared() -> Cell {
    Cell::from_vectors(
        [10.0, 0.0, 0.0],
        [3.0, 9.0, 0.0],
        [1.0, 2.0, 8.0],
        [0.0, 0.0, 0.0],
    )
    .unwrap()
}

fn general() -> Cell {
    Cell::from_vectors(
        [9.0, 1.0, 0.5],
        [2.0, 8.5, 1.0],
        [0.5, 1.5, 9.5],
        [0.0, 0.0, 0.0],
    )
    .unwrap()
}

/// `n` points on a jittered grid inside a 10-unit box.
fn grid(n: usize) -> Vec<[f64; 3]> {
    let side = (n as f64).cbrt().ceil() as usize;
    let step = 10.0 / side as f64;
    (0..n)
        .map(|i| {
            let (x, y, z) = (i % side, (i / side) % side, i / (side * side));
            let jitter = 0.01 * (i % 7) as f64;
            [
                x as f64 * step + jitter,
                y as f64 * step,
                z as f64 * step + jitter,
            ]
        })
        .collect()
}

/// Every placement of every non-finite value: each axis of the first,
/// a middle, and the last point.
fn poisoned(n: usize) -> impl Iterator<Item = Vec<[f64; 3]>> {
    let base = grid(n);
    BAD.into_iter().flat_map(move |bad| {
        let base = base.clone();
        [0, n / 2, n - 1].into_iter().flat_map(move |i| {
            let base = base.clone();
            (0..3).map(move |axis| {
                let mut xyz = base.clone();
                xyz[i][axis] = bad;
                xyz
            })
        })
    })
}

#[test]
fn knearest_rejects_non_finite_coordinates() {
    for cell in [ortho(), sheared(), general()] {
        for xyz in poisoned(40) {
            assert_eq!(knearest(&xyz, &cell, 3, None, None), Err(Error::NonFinite));
        }
    }
}

#[test]
fn knearest_rejects_non_finite_on_the_parallel_mesh() {
    let n = 9_000;
    for bad in BAD {
        let mut xyz = grid(n);
        xyz[n / 3][1] = bad;
        assert_eq!(
            knearest(&xyz, &ortho(), 4, None, None),
            Err(Error::NonFinite)
        );
        assert_eq!(
            knearest(&xyz, &general(), 4, None, None),
            Err(Error::NonFinite)
        );
    }
}

#[test]
fn knearest_into_rejects_and_clears_the_rows() {
    let k = 3;
    for xyz in poisoned(40) {
        let mut out = vec![7i32; xyz.len() * k];
        assert_eq!(
            knearest_into(&xyz, &ortho(), k, None, None, &mut out),
            Err(Error::NonFinite)
        );
        assert!(out.iter().all(|&j| j == -1));
    }
}

#[test]
fn knearest_into_d2_rejects_and_clears_the_rows() {
    let k = 3;
    for xyz in poisoned(40) {
        let mut nn = vec![7i32; xyz.len() * k];
        let mut d2 = vec![1.0f64; xyz.len() * k];
        assert_eq!(
            knearest_into_d2(&xyz, &sheared(), k, None, None, &mut nn, Some(&mut d2)),
            Err(Error::NonFinite)
        );
        assert!(nn.iter().all(|&j| j == -1));
        assert!(d2.iter().all(|v| v.is_nan()));
    }
}

#[test]
fn knearest_into_many_rejects_a_bad_later_frame_before_any_frame() {
    let (n, k, frames) = (40, 3, 3);
    for bad in BAD {
        let mut xyz = grid(n);
        xyz.extend(grid(n));
        let mut last = grid(n);
        last[n - 1][2] = bad;
        xyz.extend(last);
        let mut nn = vec![7i32; frames * n * k];
        let mut d2 = vec![1.0f64; frames * n * k];
        assert_eq!(
            knearest_into_many(
                &xyz,
                n,
                frames,
                &ortho(),
                k,
                None,
                None,
                &mut nn,
                Some(&mut d2)
            ),
            Err(Error::NonFinite)
        );
        assert!(nn.iter().all(|&j| j == -1));
        assert!(d2.iter().all(|v| v.is_nan()));
    }
}

#[test]
fn knearest_csr_rejects_non_finite_coordinates() {
    for cell in [ortho(), sheared(), general()] {
        for mutual in [false, true] {
            for xyz in poisoned(40) {
                assert_eq!(
                    knearest_csr(&xyz, &cell, 3, None, None, mutual),
                    Err(Error::NonFinite)
                );
            }
        }
    }
}

#[test]
fn knearest_brute_rejects_non_finite_coordinates() {
    for cell in [ortho(), general()] {
        for xyz in poisoned(12) {
            assert_eq!(knearest_brute(&xyz, &cell, 3, None), Err(Error::NonFinite));
        }
    }
}

#[test]
fn a_masked_non_finite_point_is_still_rejected() {
    let n = 40;
    let mut xyz = grid(n);
    xyz[5][0] = f64::NAN;
    let mut mask = vec![true; n];
    mask[5] = false;
    let k = 3;
    assert_eq!(
        knearest(&xyz, &ortho(), k, Some(&mask), None),
        Err(Error::NonFinite)
    );
    let mut out = vec![0i32; n * k];
    assert_eq!(
        knearest_into(&xyz, &ortho(), k, Some(&mask), None, &mut out),
        Err(Error::NonFinite)
    );
    assert_eq!(
        knearest_csr(&xyz, &ortho(), k, Some(&mask), None, false),
        Err(Error::NonFinite)
    );
}

#[test]
fn a_single_active_point_does_not_skip_the_check() {
    let xyz = [[f64::NAN, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let mask = [false, true];
    assert_eq!(
        knearest(&xyz, &ortho(), 1, Some(&mask), None),
        Err(Error::NonFinite)
    );
    assert_eq!(
        knearest(&xyz[..1], &ortho(), 1, None, None),
        Err(Error::NonFinite)
    );
}

#[test]
fn a_non_finite_origin_is_rejected() {
    let xyz = grid(40);
    for bad in BAD {
        for axis in 0..3 {
            let mut origin = [0.0; 3];
            origin[axis] = bad;
            let cell =
                Cell::from_vectors([10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0], origin)
                    .unwrap();
            assert_eq!(knearest(&xyz, &cell, 3, None, None), Err(Error::NonFinite));
            let mut out = vec![0i32; xyz.len() * 3];
            assert_eq!(
                knearest_into(&xyz, &cell, 3, None, None, &mut out),
                Err(Error::NonFinite)
            );
            assert_eq!(
                knearest_csr(&xyz, &cell, 3, None, None, false),
                Err(Error::NonFinite)
            );
            assert_eq!(knearest_brute(&xyz, &cell, 3, None), Err(Error::NonFinite));
        }
    }
}

#[test]
fn non_finite_box_lengths_cannot_build_a_cell() {
    for bad in BAD {
        for axis in 0..3 {
            let mut lengths = [10.0; 3];
            lengths[axis] = bad;
            assert_eq!(
                Cell::ortho(lengths[0], lengths[1], lengths[2])
                    .map_err(Error::from)
                    .err(),
                Some(Error::BadBox)
            );
            for row in 0..3 {
                let mut h = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]];
                h[row][axis] = bad;
                assert_eq!(
                    Cell::from_vectors(h[0], h[1], h[2], [0.0; 3])
                        .map_err(Error::from)
                        .err(),
                    Some(Error::BadBox)
                );
            }
        }
    }
}

#[test]
fn argument_errors_keep_their_precedence() {
    let xyz = [[f64::NAN, 0.0, 0.0], [1.0, 0.0, 0.0]];
    assert_eq!(knearest(&xyz, &ortho(), 0, None, None), Err(Error::ZeroK));
    assert_eq!(
        knearest(&xyz, &ortho(), 1, Some(&[true]), None),
        Err(Error::MaskLen)
    );
    let mut short = [0i32; 1];
    assert_eq!(
        knearest_into(&xyz, &ortho(), 1, None, None, &mut short),
        Err(Error::BufferSize)
    );
}

#[test]
fn finite_extremes_still_search() {
    let mut xyz = grid(40);
    xyz[0] = [f64::MAX, -f64::MAX, 1.0e300];
    xyz[1] = [-1.0e-300, f64::MIN_POSITIVE, 0.0];
    let rows = knearest(&xyz, &ortho(), 3, None, None).unwrap();
    assert_eq!(rows.len(), xyz.len());
    assert!(rows.iter().all(|r| r.indices.len() == 3));
    assert!(rows.iter().flat_map(|r| &r.dist2).all(|d| d.is_finite()));
}
