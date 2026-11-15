fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/knn_region.c");
    let (major, minor) = rustc_version();
    // `unexpected_cfgs` arrived in 1.80. AVX-512 intrinsics arrived in 1.89.
    if major > 1 || minor >= 80 {
        println!("cargo:rustc-check-cfg=cfg(linkcell_avx512)");
        println!("cargo:rustc-check-cfg=cfg(linkcell_openmp)");
    }
    if major > 1 || minor >= 89 {
        println!("cargo:rustc-cfg=linkcell_avx512");
    }
    // Linux with `parallel` links the fused region. A failed probe is a
    // build error there. Other targets keep the rayon walk.
    if std::env::var_os("CARGO_FEATURE_PARALLEL").is_some() && compile_knn_region() {
        println!("cargo:rustc-cfg=linkcell_openmp");
    }
}

fn linux_target() -> bool {
    std::env::var("CARGO_CFG_TARGET_OS").ok().as_deref() == Some("linux")
}

fn require_linux(step: &str) -> ! {
    panic!("OpenMP k-nearest region: {step} failed on Linux");
}

fn compile_knn_region() -> bool {
    if !linux_target() {
        return false;
    }
    let out_dir = match std::env::var("OUT_DIR") {
        Ok(dir) => std::path::PathBuf::from(dir),
        Err(_) => require_linux("OUT_DIR"),
    };
    let cc = std::env::var("CC").unwrap_or_else(|_| "gcc".to_string());
    let obj = out_dir.join("knn_region.o");
    let archive = out_dir.join("libknnregion.a");
    let compiled = std::process::Command::new(&cc)
        .args([
            "-c",
            "-O2",
            "-fPIC",
            "-fopenmp",
            "-pthread",
            "-Wall",
            "-Wextra",
            "src/knn_region.c",
            "-o",
        ])
        .arg(&obj)
        .status()
        .map(|status| status.success())
        .unwrap_or(false);
    if !compiled {
        require_linux("compiling src/knn_region.c");
    }
    let archived = std::process::Command::new("ar")
        .arg("crs")
        .arg(&archive)
        .arg(&obj)
        .status()
        .map(|status| status.success())
        .unwrap_or(false);
    if !archived {
        require_linux("archiving libknnregion.a");
    }
    let Some(gomp_dir) = gomp_dir(&cc) else {
        require_linux("finding libgomp");
    };
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-search=native={}", gomp_dir.display());
    println!("cargo:rustc-link-lib=static=knnregion");
    println!("cargo:rustc-link-lib=gomp");
    true
}

fn gomp_dir(cc: &str) -> Option<std::path::PathBuf> {
    let out = std::process::Command::new(cc)
        .arg("-print-file-name=libgomp.so")
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8(out.stdout).ok()?;
    let path = std::path::PathBuf::from(text.trim());
    path.parent()
        .filter(|_| path.is_file())
        .map(|dir| dir.to_path_buf())
}

fn rustc_version() -> (u32, u32) {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let out = std::process::Command::new(rustc)
        .arg("-vV")
        .output()
        .expect("rustc -vV");
    let text = String::from_utf8_lossy(&out.stdout);
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("release: ") {
            let ver = rest.split('-').next().unwrap_or(rest);
            let mut parts = ver.split('.');
            let major = parts.next().unwrap_or("1").parse().unwrap_or(1);
            let minor = parts.next().unwrap_or("70").parse().unwrap_or(70);
            return (major, minor);
        }
    }
    (1, 70)
}
