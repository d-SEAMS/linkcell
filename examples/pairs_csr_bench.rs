//! Cutoff CSR rows through the C ABI against the pair rows they replace.
//!
//! A random box at water density (0.0334 per cubic angstrom), cutoff
//! 3.5, orthorhombic and with an xy tilt of 0.2, full and half lists, at
//! each width in `CSR_THREADS` (default `1,8`, set by `lc_set_threads`).
//! Cases, each timed on its own after `CSR_WARM` (default 2) calls:
//!
//! - `csr`: one `lc_pairs_within_csr` call into buffers already large
//!   enough, the rows straight from the search.
//! - `csr_query_fill`: a query, then the fill that copies the kept rows.
//! - `rows`: `lc_pairs_within_rows` into a large enough buffer, no rows.
//! - `rows_serial_csr`: `rows`, then the rows grouped by source on the
//!   calling thread (count, place, sort each row), the step `csr`
//!   replaces.
//!
//! Env: `CSR_N` (default 110592), `CSR_REPS` (default 7). Each timed call
//! prints one `rep` line; each case then prints its median.

use std::os::raw::c_int;
use std::ptr::{null, null_mut};
use std::time::Instant;

use linkcell::{lc_cell, lc_pair, lc_pairs_within_csr, lc_pairs_within_rows, lc_set_threads};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn frame(n: usize, tilt: f64) -> (Vec<[f64; 3]>, lc_cell) {
    let edge = (n as f64 / 0.0334).cbrt();
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let xyz = (0..n)
        .map(|_| {
            let (a, b, c) = (next(), next(), next());
            [edge * (a + tilt * b), edge * b, edge * c]
        })
        .collect();
    let cell = lc_cell {
        ax: edge,
        ay: 0.0,
        az: 0.0,
        bx: tilt * edge,
        by: edge,
        bz: 0.0,
        cx: 0.0,
        cy: 0.0,
        cz: edge,
        ox: 0.0,
        oy: 0.0,
        oz: 0.0,
    };
    (xyz, cell)
}

struct Bufs {
    offsets: Vec<c_int>,
    cols: Vec<c_int>,
    rows: Vec<lc_pair>,
    first: Vec<usize>,
    partner: Vec<c_int>,
    next: Vec<usize>,
}

fn csr(xyz: &[[f64; 3]], cell: &lc_cell, half: bool, b: &mut Bufs, query: bool) -> usize {
    let n = xyz.len();
    let mut need = (0usize, 0usize);
    let call = |offsets: *mut c_int, cols: *mut c_int, cap: usize, need: &mut (usize, usize)| unsafe {
        lc_pairs_within_csr(
            xyz.as_ptr().cast(),
            n,
            cell,
            null(),
            3.5,
            null(),
            0.0,
            c_int::from(half),
            offsets,
            n + 1,
            cols,
            null_mut(),
            cap,
            &mut need.0,
            &mut need.1,
        )
    };
    if query {
        assert_eq!(call(null_mut(), null_mut(), 0, &mut need), 0);
        assert!(b.cols.len() >= need.1);
    }
    let rc = call(
        b.offsets.as_mut_ptr(),
        b.cols.as_mut_ptr(),
        b.cols.len(),
        &mut need,
    );
    assert_eq!(rc, 0, "buffer of {} for {} columns", b.cols.len(), need.1);
    need.1
}

fn rows(xyz: &[[f64; 3]], cell: &lc_cell, half: bool, b: &mut Bufs) -> usize {
    let mut count = 0usize;
    let rc = unsafe {
        lc_pairs_within_rows(
            xyz.as_ptr().cast(),
            xyz.len(),
            cell,
            3.5,
            null(),
            0.0,
            c_int::from(half),
            b.rows.as_mut_ptr(),
            b.rows.len(),
            &mut count,
        )
    };
    assert_eq!(rc, 0);
    count
}

/// Group `rows` by source on this thread: no self image, one entry per
/// neighbour, ascending. Below half the box a neighbour has one image,
/// so this is count, place, and sort each row.
fn serial_csr(n: usize, count: usize, b: &mut Bufs) {
    b.first.clear();
    b.first.resize(n + 1, 0);
    for r in &b.rows[..count] {
        if r.i != r.j {
            b.first[r.i as usize + 1] += 1;
        }
    }
    for i in 0..n {
        b.first[i + 1] += b.first[i];
    }
    b.next.clear();
    b.next.extend_from_slice(&b.first[..n]);
    b.partner.resize(b.first[n], 0);
    for r in &b.rows[..count] {
        if r.i != r.j {
            let at = &mut b.next[r.i as usize];
            b.partner[*at] = r.j;
            *at += 1;
        }
    }
    for i in 0..n {
        b.partner[b.first[i]..b.first[i + 1]].sort_unstable();
    }
}

/// One timed case: fills the buffers and returns the column count.
type Case<'a> = dyn Fn(&mut Bufs) -> usize + 'a;

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    let m = v.len() / 2;
    if v.len() % 2 == 1 {
        v[m]
    } else {
        0.5 * (v[m - 1] + v[m])
    }
}

fn main() {
    let n = env_usize("CSR_N", 110_592);
    let reps = env_usize("CSR_REPS", 7).max(1);
    let warm = env_usize("CSR_WARM", 2);
    let widths: Vec<i32> = std::env::var("CSR_THREADS")
        .unwrap_or_else(|_| "1,8".to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    println!(
        "# n={n} reps={reps} warm={warm} simd={:?} version={}",
        linkcell::simd_path(),
        env!("CARGO_PKG_VERSION")
    );
    for (geom, tilt) in [("ortho", 0.0), ("tilt0.2", 0.2)] {
        let (xyz, cell) = frame(n, tilt);
        let cap = 64 * n;
        let mut b = Bufs {
            offsets: vec![0; n + 1],
            cols: vec![0; cap],
            rows: vec![lc_pair::default(); cap],
            first: Vec::new(),
            partner: Vec::new(),
            next: Vec::new(),
        };
        for half in [false, true] {
            for &t in &widths {
                lc_set_threads(t);
                let cases: [(&str, &Case); 4] = [
                    ("csr", &|b| csr(&xyz, &cell, half, b, false)),
                    ("csr_query_fill", &|b| csr(&xyz, &cell, half, b, true)),
                    ("rows", &|b| rows(&xyz, &cell, half, b)),
                    ("rows_serial_csr", &|b| {
                        let count = rows(&xyz, &cell, half, b);
                        serial_csr(n, count, b);
                        b.first[n]
                    }),
                ];
                for (case, run) in cases {
                    for _ in 0..warm {
                        run(&mut b);
                    }
                    let mut ms = Vec::with_capacity(reps);
                    let mut size = 0;
                    for rep in 0..reps {
                        let start = Instant::now();
                        size = run(&mut b);
                        let took = start.elapsed().as_secs_f64() * 1e3;
                        println!(
                            "rep geom={geom} half={} threads={t} case={case} rep={rep} ms={took:.3} size={size}",
                            u8::from(half)
                        );
                        ms.push(took);
                    }
                    let lo = ms.iter().copied().fold(f64::INFINITY, f64::min);
                    let hi = ms.iter().copied().fold(0.0, f64::max);
                    println!(
                        "median geom={geom} half={} threads={t} case={case} ms={:.3} min={lo:.3} max={hi:.3} size={size}",
                        u8::from(half),
                        median(&mut ms)
                    );
                }
            }
        }
    }
    lc_set_threads(0);
}
