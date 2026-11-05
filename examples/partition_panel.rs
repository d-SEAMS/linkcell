//! Head-to-head on the periodic cell grid.
//!
//! The graph is built once per system. The clock covers `partition_graph`
//! only. Zoltan RCB is timed inside its own process around
//! `Zoltan_LB_Partition`. Part count and repetition count come from the
//! command line.

use std::fs;
use std::io::Write;
use std::process::Command;
use std::time::Instant;

use linkcell::{
    cell_graph, curve_split_device, halo_width, partition_graph, score, Cell, Error, Method,
    PartRequest,
};

const MW_SIGMA: f64 = 2.3925;
const MW_A: f64 = 1.8;
/// Spherical cutoff of 7.5 angstroms from Jorgensen, Chandrasekhar,
/// Madura, Impey and Klein, Journal of Chemical Physics 79, 926 (1983).
const TIP4P_CUTOFF: f64 = 7.5;
const TIP4P_OH: f64 = 0.9572;
const TIP4P_ANGLE: f64 = 104.52;
const TIP4P_M: f64 = 0.15;
const AVOGADRO: f64 = 6.02214076e23;
const WATER_MOLAR: f64 = 18.01528;

/// Molecules per cubic angstrom at 1 gram per cubic centimetre.
///
/// Molar mass is grams per mole. Avogadro's number is per mole.
/// One cubic centimetre is 1e24 cubic angstroms, so the factor is 1e-24.
fn density_per_angstrom() -> f64 {
    AVOGADRO / WATER_MOLAR * 1.0e-24
}

fn factors(n: usize) -> (usize, usize, usize) {
    let target = (n as f64).cbrt();
    let mut best = (1usize, 1usize, n);
    let mut best_score = f64::MAX;
    let mut x = 1usize;
    while x * x * x <= n * 8 {
        if n % x == 0 {
            let rest = n / x;
            let mut y = 1usize;
            while y * y <= rest {
                if rest % y == 0 {
                    let z = rest / y;
                    let score = (x as f64 - target).abs()
                        + (y as f64 - target).abs()
                        + (z as f64 - target).abs();
                    if score < best_score {
                        best_score = score;
                        best = (x, y, z);
                    }
                }
                y += 1;
            }
        }
        x += 1;
        if x > n {
            break;
        }
    }
    best
}

fn lattice(n: usize, rho: f64) -> (Vec<[f64; 3]>, Cell) {
    let (nx, ny, nz) = factors(n);
    let spacing = rho.powf(-1.0 / 3.0);
    let cell = Cell::ortho(
        nx as f64 * spacing,
        ny as f64 * spacing,
        nz as f64 * spacing,
    )
    .unwrap();
    let mut xyz = Vec::with_capacity(n);
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                xyz.push([
                    (ix as f64 + 0.5) * spacing,
                    (iy as f64 + 0.5) * spacing,
                    (iz as f64 + 0.5) * spacing,
                ]);
                if xyz.len() == n {
                    return (xyz, cell);
                }
            }
        }
    }
    (xyz, cell)
}

fn tip4p(n_sites: usize, rho: f64) -> (Vec<[f64; 3]>, Cell) {
    let n_mol = n_sites / 4;
    let (oxygen, cell) = lattice(n_mol, rho);
    let half = TIP4P_ANGLE.to_radians() / 2.0;
    let h1 = [TIP4P_OH * half.sin(), 0.0, TIP4P_OH * half.cos()];
    let h2 = [-h1[0], 0.0, h1[2]];
    let msite = [0.0, 0.0, TIP4P_M];
    let mut xyz = Vec::with_capacity(n_sites);
    for o in oxygen {
        xyz.push(o);
        xyz.push([o[0] + h1[0], o[1] + h1[1], o[2] + h1[2]]);
        xyz.push([o[0] + h2[0], o[1] + h2[1], o[2] + h2[2]]);
        xyz.push([o[0] + msite[0], o[1] + msite[1], o[2] + msite[2]]);
    }
    (xyz, cell)
}

fn slab(n: usize, rho: f64) -> (Vec<[f64; 3]>, Cell) {
    let (mut xyz, bulk) = lattice(n, rho);
    let lengths = bulk.widths();
    let cell = Cell::ortho(lengths[0], lengths[1], lengths[2] * 2.0).unwrap();
    for p in &mut xyz {
        if p[2] >= lengths[2] {
            p[2] = lengths[2] * 0.5;
        }
    }
    (xyz, cell)
}

struct System {
    name: &'static str,
    xyz: Vec<[f64; 3]>,
    cell: Cell,
    cutoff: f64,
}

fn systems() -> Vec<System> {
    let rho = density_per_angstrom();
    let mw_cut = MW_A * MW_SIGMA;
    let (a, ca) = lattice(110_592, rho);
    let (b, cb) = lattice(884_736, rho);
    let (c, cc) = tip4p(128_000, rho);
    let (d, cd) = tip4p(1_024_000, rho);
    let (e, ce) = slab(110_592, rho);
    vec![
        System {
            name: "mW-110592",
            xyz: a,
            cell: ca,
            cutoff: mw_cut,
        },
        System {
            name: "mW-884736",
            xyz: b,
            cell: cb,
            cutoff: mw_cut,
        },
        System {
            name: "TIP4P-128000",
            xyz: c,
            cell: cc,
            cutoff: TIP4P_CUTOFF,
        },
        System {
            name: "TIP4P-1024000",
            xyz: d,
            cell: cd,
            cutoff: TIP4P_CUTOFF,
        },
        System {
            name: "slab-110592",
            xyz: e,
            cell: ce,
            cutoff: mw_cut,
        },
    ]
}

fn methods() -> Vec<(&'static str, Method)> {
    let mut out = vec![
        ("morton", Method::Morton),
        ("hilbert", Method::Hilbert),
        ("rcb", Method::Rcb),
        ("hilbert-fm", Method::HilbertFm),
        ("rcb-fm", Method::RcbFm),
        ("multi", Method::Multi),
    ];
    if cfg!(feature = "scotch") {
        out.push(("scotch", Method::Scotch));
    }
    out
}

fn quantile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    let h = 1.0 + (n as f64 - 1.0) * p;
    let lo = h.floor();
    let hi = h.ceil();
    let a = sorted[lo as usize - 1];
    let b = sorted[hi as usize - 1];
    if (lo - hi).abs() < 1.0e-12 {
        a
    } else {
        a + (h - lo) * (b - a)
    }
}

fn iqr(mut samples: Vec<f64>) -> (f64, f64) {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = quantile(&samples, 0.5);
    let q1 = quantile(&samples, 0.25);
    let q3 = quantile(&samples, 0.75);
    (med, q3 - q1)
}

#[allow(dead_code)]
struct Row {
    system: String,
    method: String,
    cut: u64,
    imbalance: f64,
    halo: f64,
    median: f64,
    iqr: f64,
    eligible: bool,
}

fn native(name: &str) -> bool {
    matches!(
        name,
        "morton" | "hilbert" | "rcb" | "hilbert-fm" | "rcb-fm" | "multi"
    )
}

fn under_default(weight: &[u64], parts: u32, imbalance: f64) -> bool {
    imbalance <= linkcell::derived_balance_f64(weight, parts) + 1.0e-9
}

fn worst_halo(
    grid: &linkcell::CellGrid,
    assign: &[u32],
    parts: u32,
    width: [u32; 3],
) -> (usize, usize) {
    let mut best_r = -1.0;
    let mut owned_n = 0usize;
    let mut ghost_n = 0usize;
    for domain in 0..parts {
        let owned = assign.iter().filter(|&&d| d == domain).count();
        if owned == 0 {
            continue;
        }
        let ghosts = linkcell::halo(grid, assign, domain, width)
            .unwrap_or_default()
            .len();
        let ratio = ghosts as f64 / owned as f64;
        if ratio > best_r {
            best_r = ratio;
            owned_n = owned;
            ghost_n = ghosts;
        }
    }
    (owned_n, ghost_n)
}

fn print_raw(samples: &[f64]) {
    let text = samples
        .iter()
        .map(|s| format!("{s:.6}"))
        .collect::<Vec<_>>()
        .join(" ");
    println!("raw {text}");
}

fn cpu_mhz() -> u32 {
    let Ok(text) = fs::read_to_string("/proc/cpuinfo") else {
        return 0;
    };
    for line in text.lines() {
        let Some(rest) = line.strip_prefix("cpu MHz") else {
            continue;
        };
        let Some(val) = rest.split(':').nth(1) else {
            continue;
        };
        if let Ok(mhz) = val.trim().parse::<f64>() {
            return mhz.round() as u32;
        }
    }
    0
}

fn allow_method(name: &str) -> bool {
    match std::env::var("LINKCELL_ONLY_METHOD") {
        Ok(filter) if !filter.is_empty() => name.contains(filter.as_str()),
        _ => true,
    }
}

struct ExtJob {
    kind: &'static str,
    bin: String,
    graph_path: String,
    assign_path: String,
    tol: String,
}

fn zoltan_job(system: &System, graph: &linkcell::CellGraph, parts: u32) -> Option<ExtJob> {
    let bin = std::path::Path::new("benchmarks/work/zoltan_rcb");
    if !bin.is_file() {
        println!("method zoltan-rcb system {} HOST NOT-RUN", system.name);
        return None;
    }
    let grid_path = format!("benchmarks/work/{}-grid.txt", system.name);
    let assign_path = format!("benchmarks/work/{}-zoltan.txt", system.name);
    let mut file = fs::File::create(&grid_path).ok()?;
    writeln!(file, "{} {}", graph.grid.ncell(), parts).ok()?;
    for c in 0..graph.grid.ncell() {
        let nx = graph.grid.nx as usize;
        let ny = graph.grid.ny as usize;
        let ix = c % nx;
        let iy = (c / nx) % ny;
        let iz = c / (nx * ny);
        writeln!(
            file,
            "{} {} {} {}",
            ix as f64 + 0.5,
            iy as f64 + 0.5,
            iz as f64 + 0.5,
            graph.grid.weight[c]
        )
        .ok()?;
    }
    drop(file);
    let tol = format!(
        "{:.8}",
        linkcell::derived_balance_f64(&graph.grid.weight, parts)
    );
    Some(ExtJob {
        kind: "zoltan-rcb",
        bin: bin.to_str()?.to_string(),
        graph_path: grid_path,
        assign_path,
        tol,
    })
}

fn ptscotch_job(system: &System, graph: &linkcell::CellGraph, parts: u32) -> Option<ExtJob> {
    let bin = std::path::Path::new("benchmarks/work/ptscotch_part");
    if !bin.is_file() {
        println!("method ptscotch system {} HOST NOT-RUN", system.name);
        return None;
    }
    let graph_path = format!("benchmarks/work/{}-ptscotch.bin", system.name);
    let assign_path = format!("benchmarks/work/{}-ptscotch.txt", system.name);
    let mut file = fs::File::create(&graph_path).ok()?;
    let n = graph.grid.ncell();
    if n > i32::MAX as usize {
        println!("method ptscotch system {} HOST NOT-RUN", system.name);
        return None;
    }
    write_i32(&mut file, n as i32).ok()?;
    write_i32(&mut file, parts as i32).ok()?;
    for &v in &graph.offsets {
        write_i32(&mut file, v as i32).ok()?;
    }
    for &v in &graph.neighbor {
        write_i32(&mut file, v as i32).ok()?;
    }
    for &v in &graph.grid.weight {
        if v > i32::MAX as u64 {
            println!("method ptscotch system {} HOST NOT-RUN", system.name);
            return None;
        }
        write_i32(&mut file, v as i32).ok()?;
    }
    for &v in &graph.edge_weight {
        if v > i32::MAX as u64 {
            println!("method ptscotch system {} HOST NOT-RUN", system.name);
            return None;
        }
        write_i32(&mut file, v as i32).ok()?;
    }
    drop(file);
    let tol = format!(
        "{:.8}",
        linkcell::derived_balance_f64(&graph.grid.weight, parts) - 1.0
    );
    Some(ExtJob {
        kind: "ptscotch",
        bin: bin.to_str()?.to_string(),
        graph_path,
        assign_path,
        tol,
    })
}

fn ext_once(job: &ExtJob) -> Option<f64> {
    let out = if job.kind == "ptscotch" {
        Command::new("mpirun")
            .args([
                "-np",
                "8",
                &job.bin,
                &job.graph_path,
                &job.assign_path,
                &job.tol,
            ])
            .output()
            .ok()?
    } else {
        Command::new(&job.bin)
            .arg(&job.graph_path)
            .arg(&job.assign_path)
            .arg(&job.tol)
            .output()
            .ok()?
    };
    if !out.status.success() {
        println!("method {} HOST NOT-RUN", job.kind);
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut found = None;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("SECONDS ") {
            found = rest.trim().parse::<f64>().ok();
        }
        if job.kind == "ptscotch"
            && (line.starts_with("SCOTCH_randomSeed") || line.starts_with("RANKS"))
        {
            println!("ptscotch {line}");
        }
    }
    found
}

fn assignment_cut(graph: &linkcell::CellGraph, path: &str) -> Option<u64> {
    let text = fs::read_to_string(path).ok()?;
    let assign: Vec<u32> = text
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| line.parse().ok())
        .collect::<Option<_>>()?;
    if assign.len() != graph.grid.ncell() {
        return None;
    }
    Some(linkcell::edge_cut(graph, &assign))
}

fn ext_row(
    system: &System,
    graph: &linkcell::CellGraph,
    parts: u32,
    width: [u32; 3],
    job: &ExtJob,
    samples: Vec<f64>,
    cuts: &[u64],
) -> Option<Row> {
    let text = fs::read_to_string(&job.assign_path).ok()?;
    let assign: Vec<u32> = text
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| l.parse().ok())
        .collect::<Option<_>>()?;
    let got = score(graph, &assign, parts, width).ok()?;
    let cut = if cuts.is_empty() {
        got.cut
    } else {
        iqr(cuts.iter().map(|&c| c as f64).collect()).0.round() as u64
    };
    print_raw(&samples);
    let (median, spread) = iqr(samples);
    println!(
        "system {} method {} cut {} imbalance {:.6} halo_fraction {:.6} median_s {:.6} iqr_s {:.6} eligible 0",
        system.name, job.kind, cut, got.imbalance, got.halo_fraction, median, spread
    );
    Some(Row {
        system: system.name.to_string(),
        method: job.kind.to_string(),
        cut,
        imbalance: got.imbalance,
        halo: got.halo_fraction,
        median,
        iqr: spread,
        eligible: false,
    })
}

fn write_i32(file: &mut fs::File, value: i32) -> std::io::Result<()> {
    file.write_all(&value.to_le_bytes())
}

fn sweep_ratios() -> [f64; 5] {
    [1.00, 1.01, 1.02, 1.05, 1.10]
}

fn sweep_system(system: &System, graph: &linkcell::CellGraph, parts: u32, reps: usize) {
    let chosen = methods();
    for ratio in sweep_ratios() {
        let mut samples = vec![Vec::with_capacity(reps); chosen.len()];
        let mut assigns = vec![Vec::new(); chosen.len()];
        for _ in 0..reps {
            for (i, (name, method)) in chosen.iter().enumerate() {
                let request = PartRequest {
                    parts,
                    method: *method,
                    balance: Some(ratio),
                    device: false,
                };
                let t0 = Instant::now();
                let assign = partition_graph(graph, request)
                    .unwrap_or_else(|err| panic!("sweep {} {name}: {err}", system.name));
                samples[i].push(t0.elapsed().as_secs_f64());
                assigns[i] = assign;
            }
        }
        for (i, (name, _)) in chosen.iter().enumerate() {
            let got = score(graph, &assigns[i], parts, [0, 0, 0]).unwrap();
            print_raw(&samples[i]);
            let (median, spread) = iqr(samples[i].clone());
            println!(
                "sweep ratio {ratio:.2} system {} method {name} cut {} imbalance {:.6} median_s {median:.6} iqr_s {spread:.6}",
                system.name, got.cut, got.imbalance
            );
        }
    }
}

fn scale_case(parts: u32, reps: usize) {
    let mut args = std::env::args().skip(4);
    let kind = args.next().unwrap_or_else(|| "strong".into());
    let n: usize = args
        .next()
        .unwrap_or_else(|| "1520875".into())
        .parse()
        .unwrap();
    let threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "1".into());
    println!("hardware Intel Xeon Processor");
    println!("physical_cores 8");
    println!("threads_per_core 1");
    println!("node single");
    println!("cpu_mhz {}", cpu_mhz());
    println!("schedule round-robin");
    println!("scale {kind} threads {threads} n {n} parts {parts} reps {reps}");
    let rho = density_per_angstrom();
    let (xyz, cell) = lattice(n, rho);
    let cutoff = MW_A * MW_SIGMA;
    let graph = cell_graph(&xyz, &cell, cutoff, None, None)
        .unwrap_or_else(|err| panic!("scale graph: {err}"));
    println!("scale_ncell {}", graph.grid.ncell());
    let chosen = [
        ("rcb", Method::Rcb),
        ("rcb-fm", Method::RcbFm),
        ("multi", Method::Multi),
    ];
    let mut samples = vec![Vec::with_capacity(reps); chosen.len()];
    let mut prof = vec![(0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64); chosen.len()];
    let mut hashes = vec![0u64; chosen.len()];
    for rep in 0..reps {
        for (i, (name, method)) in chosen.iter().enumerate() {
            linkcell::profile_reset();
            let t0 = Instant::now();
            let assign = partition_graph(
                &graph,
                PartRequest {
                    parts,
                    method: *method,
                    balance: None,
                    device: false,
                },
            )
            .unwrap_or_else(|err| panic!("scale {name}: {err}"));
            samples[i].push(t0.elapsed().as_secs_f64());
            let (sel, fm, coarse, pack, repair, check, schedule) = linkcell::profile_ns();
            prof[i].0 += sel;
            prof[i].1 += fm;
            prof[i].2 += coarse;
            prof[i].3 += pack;
            prof[i].4 += repair;
            prof[i].5 += check;
            prof[i].6 += schedule;
            let hash = assign_hash(&assign);
            if rep == 0 {
                hashes[i] = hash;
            } else if hashes[i] != hash {
                panic!("scale {name} assignment changed across repetitions");
            }
        }
    }
    for (i, (name, _)) in chosen.iter().enumerate() {
        print_raw(&samples[i]);
        let (median, spread) = iqr(samples[i].clone());
        let (sel, fm, coarse, pack, repair, check, schedule) = prof[i];
        println!(
            "scale_row kind {kind} threads {threads} n {n} method {name} median_s {median:.6} iqr_s {spread:.6} profile_ns {sel} {fm} {coarse} {pack} {repair} {check} {schedule} assign_hash {:016x}",
            hashes[i]
        );
    }
}

/// FNV-1a over the domain ids. The same assignment always prints the same hash.
fn assign_hash(assign: &[u32]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &domain in assign {
        hash ^= u64::from(domain);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn main() {
    let parts: u32 = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "8".into())
        .parse()
        .unwrap();
    let reps: usize = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "12".into())
        .parse()
        .unwrap();
    if std::env::args().nth(3).as_deref() == Some("scale") {
        scale_case(parts, reps);
        return;
    }
    let only = std::env::args().nth(3);
    println!("hardware Intel Xeon Processor");
    println!("physical_cores 8");
    println!("threads_per_core 1");
    println!("node single");
    println!("multi_node HOST NOT-RUN");
    println!("cpu_mhz {}", cpu_mhz());
    println!("schedule round-robin");
    println!("scotch_seed 1");
    println!("halo_fraction largest ghost-cell count divided by owned-cell count");
    println!("parts {parts}");
    println!("reps {reps}");
    println!("balance_bound 1 + parts * w_max / total");
    println!("quantile type-7");
    println!("sweep_ratios 1.00 1.01 1.02 1.05 1.10");
    let rho = density_per_angstrom();
    println!("water_density_molecules_per_A3 {rho:.8}");
    println!("density_basis 1 g cm-3, molar mass 18.01528 g mol-1");
    println!("mW_cutoff_A {:.6}", MW_A * MW_SIGMA);
    println!("tip4p_cutoff_A {TIP4P_CUTOFF}");

    let probe = Cell::ortho(8.0, 8.0, 8.0).unwrap();
    let xyz = vec![
        [1.0, 1.0, 1.0],
        [2.0, 1.0, 1.0],
        [1.0, 2.0, 1.0],
        [2.0, 2.0, 1.0],
    ];
    match curve_split_device(&xyz, &probe, 2.0, None, None, 2, Method::Morton) {
        Err(Error::NoDevice) => println!("gpu_curve HOST NOT-RUN"),
        Ok(_) => println!("gpu_curve ran"),
        Err(err) => println!("gpu_curve HOST NOT-RUN {err}"),
    }

    let mut rows: Vec<Row> = Vec::new();
    for system in systems() {
        if only
            .as_ref()
            .is_some_and(|name| !system.name.contains(name.as_str()))
        {
            continue;
        }
        let t_graph = Instant::now();
        let graph = cell_graph(&system.xyz, &system.cell, system.cutoff, None, None)
            .unwrap_or_else(|err| panic!("{} graph: {err}", system.name));
        let graph_s = t_graph.elapsed().as_secs_f64();
        let width = halo_width(system.cutoff, 1, graph.grid.bin_width).unwrap();
        let (bnum, bden) = linkcell::derived_balance(&graph.grid.weight, parts);
        let heavy = graph.grid.weight.iter().copied().max().unwrap_or(0);
        println!(
            "system {} n {} ncell {} bin_width {:.6} {:.6} {:.6} graph_s {:.6}",
            system.name,
            system.xyz.len(),
            graph.grid.ncell(),
            graph.grid.bin_width[0],
            graph.grid.bin_width[1],
            graph.grid.bin_width[2],
            graph_s
        );
        println!(
            "derived_balance {bnum}/{bden} {:.8} heaviest_cell {heavy}",
            linkcell::derived_balance_f64(&graph.grid.weight, parts)
        );
        println!("cpu_mhz {}", cpu_mhz());
        let chosen: Vec<_> = methods()
            .into_iter()
            .filter(|(name, _)| allow_method(name))
            .collect();
        let mut samples = vec![Vec::with_capacity(reps); chosen.len()];
        let mut assigns: Vec<Vec<u32>> = vec![Vec::new(); chosen.len()];
        let mut prof = vec![(0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64); chosen.len()];
        let zjob = if allow_method("zoltan-rcb") {
            zoltan_job(&system, &graph, parts)
        } else {
            None
        };
        let pjob = if allow_method("ptscotch") {
            ptscotch_job(&system, &graph, parts)
        } else {
            None
        };
        let mut zsamples = Vec::with_capacity(reps);
        let mut psamples = Vec::with_capacity(reps);
        let mut pcuts = Vec::with_capacity(reps);
        let mut z_ok = zjob.is_some();
        let mut p_ok = pjob.is_some();
        for _ in 0..reps {
            for (i, (name, method)) in chosen.iter().enumerate() {
                linkcell::profile_reset();
                let request = PartRequest {
                    parts,
                    method: *method,
                    balance: None,
                    device: false,
                };
                let t0 = Instant::now();
                let assign = partition_graph(&graph, request)
                    .unwrap_or_else(|err| panic!("{} {name}: {err}", system.name));
                samples[i].push(t0.elapsed().as_secs_f64());
                let (sel, fm, coarse, pack, repair, check, schedule) = linkcell::profile_ns();
                prof[i].0 += sel;
                prof[i].1 += fm;
                prof[i].2 += coarse;
                prof[i].3 += pack;
                prof[i].4 += repair;
                prof[i].5 += check;
                prof[i].6 += schedule;
                if std::env::var("LINKCELL_ASSIGN_SUM").is_ok() {
                    let sum: u64 = assign.iter().map(|&d| u64::from(d)).sum();
                    let mix = assign.iter().fold(0u64, |acc, &d| {
                        acc.wrapping_mul(0x9e3779b97f4a7c15)
                            .wrapping_add(u64::from(d))
                    });
                    println!("assign_sum {name} {sum} {mix}");
                }
                assigns[i] = assign;
            }
            if z_ok {
                if let Some(job) = &zjob {
                    match ext_once(job) {
                        Some(sample) => zsamples.push(sample),
                        None => {
                            println!("method zoltan-rcb system {} HOST NOT-RUN", system.name);
                            z_ok = false;
                        }
                    }
                }
            }
            if p_ok {
                if let Some(job) = &pjob {
                    match ext_once(job) {
                        Some(sample) => {
                            psamples.push(sample);
                            if let Some(cut) = assignment_cut(&graph, &job.assign_path) {
                                println!(
                                    "ptscotch_run system {} seed 1 ranks 8 cut {cut}",
                                    system.name
                                );
                                pcuts.push(cut);
                            }
                        }
                        None => {
                            println!("method ptscotch system {} HOST NOT-RUN", system.name);
                            p_ok = false;
                        }
                    }
                }
            }
        }
        for (i, (name, _)) in chosen.iter().enumerate() {
            if assigns[i].is_empty() {
                continue;
            }
            let got = score(&graph, &assigns[i], parts, width).unwrap();
            let (owned, ghosts) = worst_halo(&graph.grid, &assigns[i], parts, width);
            print_raw(&samples[i]);
            let (sel, fm, coarse, pack, repair, check, schedule) = prof[i];
            let (median, spread) = iqr(samples[i].clone());
            let eligible = native(name) && under_default(&graph.grid.weight, parts, got.imbalance);
            println!(
                "halo_owned {owned} halo_ghost {ghosts} profile_ns {sel} {fm} {coarse} {pack} {repair} {check} {schedule}"
            );
            println!(
                "system {} method {name} cut {} imbalance {:.6} halo_fraction {:.6} median_s {:.6} iqr_s {:.6} eligible {}",
                system.name,
                got.cut,
                got.imbalance,
                got.halo_fraction,
                median,
                spread,
                eligible as u8
            );
            rows.push(Row {
                system: system.name.to_string(),
                method: name.to_string(),
                cut: got.cut,
                imbalance: got.imbalance,
                halo: got.halo_fraction,
                median,
                iqr: spread,
                eligible,
            });
        }
        if let Some(job) = &zjob {
            if zsamples.len() == reps {
                if let Some(row) = ext_row(&system, &graph, parts, width, job, zsamples, &[]) {
                    rows.push(row);
                }
            }
        }
        if let Some(job) = &pjob {
            if psamples.len() == reps {
                if pcuts.len() == reps {
                    let mut ordered = pcuts.clone();
                    ordered.sort_unstable();
                    let min_cut = ordered[0];
                    let max_cut = ordered[ordered.len() - 1];
                    let median_cut = iqr(pcuts.iter().map(|&c| c as f64).collect()).0;
                    println!(
                        "ptscotch_summary system {} seed 1 ranks 8 median_cut {median_cut:.4} min_cut {min_cut} max_cut {max_cut}",
                        system.name
                    );
                }
                if let Some(row) = ext_row(&system, &graph, parts, width, job, psamples, &pcuts) {
                    rows.push(row);
                }
            }
        }
        if std::env::var("LINKCELL_ONLY_METHOD")
            .ok()
            .filter(|f| !f.is_empty())
            .is_none()
        {
            sweep_system(&system, &graph, parts, reps);
        }
    }

    let mut totals: Vec<(&str, u64, f64, bool)> = Vec::new();
    for (name, _) in methods() {
        if !native(name) {
            continue;
        }
        let mine: Vec<&Row> = rows.iter().filter(|r| r.method == name).collect();
        if mine.len() != 5 {
            continue;
        }
        let cut: u64 = mine.iter().map(|r| r.cut).sum();
        let time: f64 = mine.iter().map(|r| r.median).sum();
        let ok = mine.iter().all(|r| r.eligible);
        totals.push((name, cut, time, ok));
        println!(
            "sum method {name} cut {cut} median_s {time:.6} eligible {}",
            ok as u8
        );
    }
    let mut eligible: Vec<_> = totals.iter().filter(|t| t.3).cloned().collect();
    if eligible.is_empty() {
        eligible = totals.clone();
    }
    eligible.sort_by(|a, b| a.1.cmp(&b.1).then(a.2.partial_cmp(&b.2).unwrap()));
    if let Some(win) = eligible.first() {
        println!("default_method {}", win.0);
        println!("default_cut_sum {}", win.1);
    }
    let _ = std::io::stdout().flush();
}
