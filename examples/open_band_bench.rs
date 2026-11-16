//! Open-cell contact queries on small molecules and on band frames.
//!
//! Each point times `OPEN_BAND_CALLS` (default 20000) calls of
//! [`pairs_open`] per sample. The small points query one molecule that
//! sits inside one set of bins. The band points visit the images of one
//! molecule in turn, as the contact check of a band step does, and those
//! images sit in different bins and span different bin counts.
//! `OPEN_BAND_REPS` (default 11, at least 6) is the number of timed
//! samples after one warmup sample. `OPEN_BAND_POINTS` is an optional
//! comma list of point names.
//!
//! One line per point: the median and the interquartile range in
//! microseconds per call, the stencil builds per sample, the pair rows
//! per sample, and every raw sample. No host name.

use std::time::Instant;

use linkcell::{pairs_open, stencil_builds};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(default)
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    let last = sorted.len() - 1;
    let pos = q * last as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let weight = pos - lo as f64;
        sorted[lo] * (1.0 - weight) + sorted[hi] * weight
    }
}

fn water() -> Vec<[f64; 3]> {
    vec![
        [5.000, 5.000, 5.000],
        [5.757, 5.586, 5.000],
        [4.243, 5.586, 5.000],
    ]
}

fn methane() -> Vec<[f64; 3]> {
    let c = [5.0, 5.0, 5.0];
    let d = 0.629;
    vec![
        c,
        [c[0] + d, c[1] + d, c[2] + d],
        [c[0] - d, c[1] - d, c[2] + d],
        [c[0] - d, c[1] + d, c[2] - d],
        [c[0] + d, c[1] - d, c[2] - d],
    ]
}

fn benzene() -> Vec<[f64; 3]> {
    let mut xyz = Vec::with_capacity(12);
    for (radius, phase) in [(1.39, 0.0), (2.47, 0.0)] {
        for k in 0..6 {
            let angle = phase + k as f64 * std::f64::consts::PI / 3.0;
            xyz.push([6.0 + radius * angle.cos(), 6.0 + radius * angle.sin(), 6.0]);
        }
    }
    xyz
}

fn cluster() -> Vec<[f64; 3]> {
    let mut xyz = Vec::with_capacity(27);
    for iz in 0..3 {
        for iy in 0..3 {
            for ix in 0..3 {
                xyz.push([(ix + 2) as f64, (iy + 2) as f64, (iz + 2) as f64]);
            }
        }
    }
    xyz
}

/// A 16-atom zigzag chain with one hydrogen per carbon.
fn chain() -> Vec<[f64; 3]> {
    let mut xyz = Vec::with_capacity(16);
    for k in 0..8 {
        let x = 1.26 * k as f64;
        let y = if k % 2 == 0 { 0.0 } else { 0.89 };
        xyz.push([x, y, 0.0]);
        xyz.push([x, if k % 2 == 0 { -1.09 } else { 1.98 }, 0.31]);
    }
    xyz
}

/// Images of the chain along a band: each image bends the chain, tilts
/// it, and moves it, so consecutive images sit in other bins.
fn band(images: usize) -> Vec<Vec<[f64; 3]>> {
    let base = chain();
    (0..images)
        .map(|image| {
            let t = image as f64 / (images - 1) as f64;
            let tilt = 0.9 * t;
            let (sin, cos) = tilt.sin_cos();
            let shift = [3.7 * t + 11.0, -2.9 * t + 7.0, 4.3 * t + 5.0];
            base.iter()
                .enumerate()
                .map(|(atom, p)| {
                    let bend = 0.6 * t * (atom as f64 / 15.0).powi(2);
                    let x = cos * p[0] - sin * p[2];
                    let z = sin * p[0] + cos * p[2] + bend;
                    [x + shift[0], p[1] + shift[1], z + shift[2]]
                })
                .collect()
        })
        .collect()
}

/// The near and far frames of the `alternating_open_frames_share_one_stencil`
/// test.
fn alternating() -> Vec<Vec<[f64; 3]>> {
    let near = cluster();
    let mut far: Vec<[f64; 3]> = near
        .iter()
        .map(|p| [1.3 * p[0] + 7.1, 0.9 * p[1] - 3.4, 1.1 * p[2] + 11.6])
        .collect();
    far[5][1] -= 2.2;
    vec![near, far]
}

struct Point {
    name: &'static str,
    frames: Vec<Vec<[f64; 3]>>,
    cutoff: f64,
    hint: Option<f64>,
}

fn points() -> Vec<Point> {
    vec![
        Point {
            name: "water3_c2.0",
            frames: vec![water()],
            cutoff: 2.0,
            hint: None,
        },
        Point {
            name: "methane5_c2.0",
            frames: vec![methane()],
            cutoff: 2.0,
            hint: None,
        },
        Point {
            name: "benzene12_c2.5",
            frames: vec![benzene()],
            cutoff: 2.5,
            hint: None,
        },
        Point {
            name: "cluster27_c2.5_h8",
            frames: vec![cluster()],
            cutoff: 2.5,
            hint: Some(8.0),
        },
        Point {
            name: "cluster27_c1.5",
            frames: vec![cluster()],
            cutoff: 1.5,
            hint: None,
        },
        Point {
            name: "chain16_c2.0",
            frames: vec![chain()],
            cutoff: 2.0,
            hint: None,
        },
        Point {
            name: "band8_chain16_c2.0",
            frames: band(8),
            cutoff: 2.0,
            hint: None,
        },
        Point {
            name: "band8_chain16_c3.0",
            frames: band(8),
            cutoff: 3.0,
            hint: None,
        },
        Point {
            name: "alt2_cluster27_c1.5",
            frames: alternating(),
            cutoff: 1.5,
            hint: None,
        },
    ]
}

fn sample(point: &Point, calls: usize) -> (f64, usize) {
    let mut rows = 0usize;
    let frames = point.frames.len();
    let start = Instant::now();
    for call in 0..calls {
        let xyz = &point.frames[call % frames];
        rows += pairs_open(std::hint::black_box(xyz), point.cutoff, point.hint)
            .expect("open")
            .len();
    }
    let us = start.elapsed().as_secs_f64() * 1.0e6 / calls as f64;
    (us, std::hint::black_box(rows))
}

fn main() {
    let reps = env_usize("OPEN_BAND_REPS", 11).max(6);
    let calls = env_usize("OPEN_BAND_CALLS", 20_000).max(1);
    let only = std::env::var("OPEN_BAND_POINTS").ok();
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    println!("cores={cores} reps={reps} calls={calls}");
    for point in points() {
        if let Some(only) = &only {
            if !only.split(',').any(|name| name == point.name) {
                continue;
            }
        }
        let atoms = point.frames[0].len();
        let _ = sample(&point, calls);
        let mut samples = Vec::with_capacity(reps);
        let mut builds = 0u64;
        let mut rows = 0usize;
        for _ in 0..reps {
            let before = stencil_builds();
            let (us, got) = sample(&point, calls);
            builds += stencil_builds() - before;
            rows = got;
            samples.push(us);
        }
        let raw = samples
            .iter()
            .map(|us| format!("{us:.5}"))
            .collect::<Vec<_>>()
            .join(",");
        samples.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        let median = percentile(&samples, 0.5);
        let iqr = percentile(&samples, 0.75) - percentile(&samples, 0.25);
        println!(
            "point={} atoms={atoms} frames={} median_us={median:.5} iqr_us={iqr:.5} builds_per_sample={} rows_per_sample={rows} samples_us={raw}",
            point.name,
            point.frames.len(),
            builds / reps as u64,
        );
    }
}
