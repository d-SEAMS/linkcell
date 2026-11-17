use linkcell::{knearest, Cell};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    readme()?;
    let ortho = Cell::ortho(10.0, 10.0, 10.0)?;
    let ortho_h = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]];
    let ortho_xyz = [
        [0.20, 0.00, 0.00],
        [9.40, 0.00, 0.00],
        [0.20, 2.50, 0.10],
        [5.00, 5.00, 5.00],
        [0.00, 0.10, 9.70],
        [3.30, 1.20, 4.80],
        [8.90, 8.10, 0.40],
        [1.40, 7.20, 2.60],
        [6.60, 0.30, 8.80],
        [4.20, 9.50, 3.30],
        [2.20, 3.80, 6.10],
        [7.70, 4.40, 1.90],
    ];
    check("ortho", &ortho_xyz, &ortho, &ortho_h, 4)?;

    let shear_h = [[10.0, 0.0, 0.0], [5.0, 8.66, 0.0], [0.0, 0.0, 10.0]];
    let shear = Cell::from_vectors(shear_h[0], shear_h[1], shear_h[2], [0.0, 0.0, 0.0])?;
    let shear_frac = [
        [0.02, 0.10, 0.20],
        [0.90, 0.05, 0.15],
        [0.40, 0.70, 0.30],
        [0.15, 0.40, 0.80],
        [0.60, 0.20, 0.55],
        [0.33, 0.88, 0.10],
    ];
    let shear_xyz = cartesian(&shear_frac, &shear_h);
    check("sheared", &shear_xyz, &shear, &shear_h, 3)?;

    let tri_h = [[10.0, 0.2, 0.1], [1.2, 8.5, -0.3], [0.4, 0.5, 9.2]];
    let tri = Cell::from_vectors(tri_h[0], tri_h[1], tri_h[2], [0.0, 0.0, 0.0])?;
    let tri_frac = [
        [0.02, 0.00, 0.00],
        [0.94, 0.00, 0.01],
        [0.10, 0.20, 0.30],
        [0.50, 0.50, 0.50],
        [0.80, 0.15, 0.90],
        [0.33, 0.71, 0.22],
        [0.61, 0.08, 0.44],
        [0.12, 0.88, 0.67],
    ];
    let tri_xyz = cartesian(&tri_frac, &tri_h);
    check("triclinic", &tri_xyz, &tri, &tri_h, 3)?;
    Ok(())
}

fn readme() -> Result<(), linkcell::Error> {
    let sim = Cell::ortho(10.0, 10.0, 10.0)?;
    let sheared = Cell::from_vectors(
        [10.0, 0.0, 0.0],
        [5.0, 8.66, 0.0],
        [0.0, 0.0, 10.0],
        [0.0, 0.0, 0.0],
    )?;
    let xyz = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let rows = knearest(&xyz, &sim, 1, None, None)?;
    assert_eq!(rows[0].indices, vec![1]);
    assert_eq!(rows[1].indices, vec![0]);
    println!("quickstart 0 -> {:?}", rows[0].indices);
    let sheared_rows = knearest(&xyz, &sheared, 1, None, None)?;
    assert_eq!(sheared_rows[0].indices, vec![1]);
    assert_eq!(sheared_rows[1].indices, vec![0]);
    println!("quickstart sheared 0 -> {:?}", sheared_rows[0].indices);

    let wrap = [[0.2, 0.0, 0.0], [9.4, 0.0, 0.0]];
    let wrap_rows = knearest(&wrap, &sim, 1, None, None)?;
    assert_eq!(wrap_rows[0].indices, vec![1]);
    assert_eq!(wrap_rows[1].indices, vec![0]);
    assert!((wrap_rows[0].dist2[0] - 0.64).abs() < 1e-12);
    assert!((wrap_rows[1].dist2[0] - 0.64).abs() < 1e-12);
    println!(
        "wrap d2 {:.16} {:.16}",
        wrap_rows[0].dist2[0], wrap_rows[1].dist2[0]
    );
    Ok(())
}

fn cartesian(frac: &[[f64; 3]], h: &[[f64; 3]; 3]) -> Vec<[f64; 3]> {
    frac.iter()
        .map(|f| {
            [
                f[0] * h[0][0] + f[1] * h[1][0] + f[2] * h[2][0],
                f[0] * h[0][1] + f[1] * h[1][1] + f[2] * h[2][1],
                f[0] * h[0][2] + f[1] * h[1][2] + f[2] * h[2][2],
            ]
        })
        .collect()
}

fn image_dist2(pi: [f64; 3], pj: [f64; 3], h: &[[f64; 3]; 3]) -> f64 {
    let mut best = f64::INFINITY;
    for ia in -2..=2 {
        for ib in -2..=2 {
            for ic in -2..=2 {
                let shift = [
                    ia as f64 * h[0][0] + ib as f64 * h[1][0] + ic as f64 * h[2][0],
                    ia as f64 * h[0][1] + ib as f64 * h[1][1] + ic as f64 * h[2][1],
                    ia as f64 * h[0][2] + ib as f64 * h[1][2] + ic as f64 * h[2][2],
                ];
                let d = [
                    pj[0] + shift[0] - pi[0],
                    pj[1] + shift[1] - pi[1],
                    pj[2] + shift[2] - pi[2],
                ];
                let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                if d2 < best {
                    best = d2;
                }
            }
        }
    }
    best
}

fn check(
    label: &str,
    xyz: &[[f64; 3]],
    cell: &Cell,
    h: &[[f64; 3]; 3],
    k: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let got = knearest(xyz, cell, k, None, None)?;
    let mut max_abs = 0.0_f64;
    for (i, row) in got.iter().enumerate() {
        let mut cands: Vec<(f64, usize)> = (0..xyz.len())
            .filter(|&j| j != i)
            .map(|j| (image_dist2(xyz[i], xyz[j], h), j))
            .collect();
        cands.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        if cands.len() <= k || !(cands[k].0 - cands[k - 1].0 > 1.0e-6) {
            return Err(format!("{label}: ambiguous neighbour gap at {i}").into());
        }
        if row.indices.len() != k {
            return Err(format!(
                "{label}: i={i} returned {} neighbours, expected {k}",
                row.indices.len()
            )
            .into());
        }
        for t in 0..k {
            let err = (row.dist2[t] - cands[t].0).abs();
            if row.indices[t] != cands[t].1 || err > 1.0e-8 {
                return Err(format!(
                    "{label}: i={i} rank={t} got ({}, {:.16}) expected ({}, {:.16})",
                    row.indices[t], row.dist2[t], cands[t].1, cands[t].0
                )
                .into());
            }
            max_abs = max_abs.max(err);
        }
    }
    println!("{label} ok n={} k={k} max_abs_d2={max_abs:.3e}", xyz.len());
    Ok(())
}
