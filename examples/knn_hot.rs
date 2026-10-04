//! Tight k-nearest loop for perf / cachegrind / roofline.
//!
//! Env: `KNN_HOT_N` (default 4096), `KNN_HOT_K` (default 4),
//! `KNN_HOT_REPS` (default 40), `KNN_HOT_SHAPE`:
//! `cubic` (default), `slab` (eOn / readcon orthorhombic film),
//! `column` (seams water column), `hex` (rgsaddle hexagonal cell),
//! `tilt` (seams LAMMPS dump with an xy tilt).

use std::time::Instant;

use linkcell::{knearest_into, Cell};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn lattice(n: usize, boxl: f64) -> Vec<[f64; 3]> {
    let nside = ((n as f64).cbrt().ceil() as usize).max(1);
    let a = boxl / nside as f64;
    let mut xyz = Vec::with_capacity(n);
    for i in 0..n {
        let ix = i % nside;
        let iy = (i / nside) % nside;
        let iz = i / (nside * nside);
        xyz.push([ix as f64 * a, iy as f64 * a, iz as f64 * a]);
    }
    xyz
}

fn frac_grid(cell: &Cell, n: usize) -> Vec<[f64; 3]> {
    let nside = ((n as f64).cbrt().ceil() as usize).max(1);
    let mut xyz = Vec::with_capacity(n);
    for i in 0..n {
        let ix = i % nside;
        let iy = (i / nside) % nside;
        let iz = i / (nside * nside);
        let s = [
            (ix as f64 + 0.5) / nside as f64,
            (iy as f64 + 0.5) / nside as f64,
            (iz as f64 + 0.5) / nside as f64,
        ];
        xyz.push(cell.cartesian(s));
    }
    xyz
}

fn nside_of(n: usize) -> usize {
    ((n as f64).cbrt().round() as usize).max(1)
}

/// Orthorhombic film at the cubic spacing, with an equal vacuum along z.
/// This is an eOn slab or a readcon CON, scaled up.
fn slab(n: usize) -> (Vec<[f64; 3]>, Cell) {
    let nside = nside_of(n);
    let film = nside as f64 * 3.125;
    let cell = Cell::ortho(film, film, film * 2.0).expect("slab");
    let mut xyz = Vec::with_capacity(n);
    for i in 0..n {
        let ix = i % nside;
        let iy = (i / nside) % nside;
        let iz = i / (nside * nside);
        xyz.push([
            (ix as f64 + 0.5) / nside as f64 * film,
            (iy as f64 + 0.5) / nside as f64 * film,
            (iz as f64 + 0.5) / nside as f64 * film,
        ]);
    }
    (xyz, cell)
}

/// Tall orthorhombic column at the cubic spacing. Points occupy a band
/// and the rest of z is vacuum, as in a seams nanotube trajectory.
fn column(n: usize) -> (Vec<[f64; 3]>, Cell) {
    let nside = nside_of(n);
    let side = nside as f64 * 3.125;
    let cell = Cell::ortho(side, side, side * 4.0).expect("column");
    let mut xyz = Vec::with_capacity(n);
    for i in 0..n {
        let ix = i % nside;
        let iy = (i / nside) % nside;
        let iz = i / (nside * nside);
        xyz.push([
            (ix as f64 + 0.5) / nside as f64 * side,
            (iy as f64 + 0.5) / nside as f64 * side,
            (iz as f64 + 0.5) / nside as f64 * side + side * 1.5,
        ]);
    }
    (xyz, cell)
}

fn system(shape: &str, n: usize) -> (Vec<[f64; 3]>, Cell) {
    match shape {
        "cubic" => {
            let boxl = 50.0;
            (
                lattice(n, boxl),
                Cell::ortho(boxl, boxl, boxl).expect("box"),
            )
        }
        "slab" => slab(n),
        "column" => column(n),
        "hex" => {
            // rgsaddle hexagonal cell, scaled to the cubic spacing.
            let cell = Cell::from_vectors(
                [50.0, 0.0, 0.0],
                [25.0, 43.30127018922193, 0.0],
                [0.0, 0.0, 55.55555555555556],
                [0.0, 0.0, 0.0],
            )
            .expect("hex");
            (frac_grid(&cell, n), cell)
        }
        "tilt" => {
            // genice sH dump (BOX BOUNDS xy xz yz), scaled to the cubic spacing.
            let s = 2.85;
            let cell = Cell::from_lammps(
                -6.2106057 * s,
                24.8424228 * s,
                0.0,
                10.7570846 * s,
                0.0,
                20.131291 * s,
                -6.2106057 * s,
                0.0,
                0.0,
            )
            .expect("tilt");
            (frac_grid(&cell, n), cell)
        }
        other => panic!("unknown KNN_HOT_SHAPE {other}"),
    }
}

fn main() {
    let n = env_usize("KNN_HOT_N", 4096);
    let k = env_usize("KNN_HOT_K", 4);
    let reps = env_usize("KNN_HOT_REPS", 40);
    let shape = std::env::var("KNN_HOT_SHAPE").unwrap_or_else(|_| "cubic".to_string());
    let (xyz, cell) = system(&shape, n);
    let mut out = vec![-1i32; xyz.len() * k];
    let mut acc = 0i32;
    let started = Instant::now();
    for _ in 0..reps {
        knearest_into(&xyz, &cell, k, None, Some(3.0), &mut out).expect("knearest");
        acc = acc.wrapping_add(out[0]);
    }
    let ms = started.elapsed().as_secs_f64() * 1000.0;
    std::hint::black_box(acc);
    eprintln!(
        "shape={shape} n={} k={} reps={} ms={ms:.1}",
        xyz.len(),
        k,
        reps
    );
}
