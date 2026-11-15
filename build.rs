fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/knn_region.c");
    println!("cargo:rerun-if-changed=src/gpu_stub.c");
    cc::Build::new()
        .file("src/gpu_stub.c")
        .warnings(true)
        .compile("linkcell_gpu_stub");
    #[cfg(feature = "scotch")]
    build_scotch();
    #[cfg(feature = "ptscotch")]
    build_ptscotch();
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

fn target_is(key: &str, value: &str) -> bool {
    std::env::var(key).ok().as_deref() == Some(value)
}

/// Compile `src/knn_region.c` with OpenMP for the target, through the `cc`
/// crate so `CC`, the target triple and cross compilers are honoured.
/// On glibc Linux a failure is a build error. musl and other targets keep
/// the rayon walk when the region does not build.
fn compile_knn_region() -> bool {
    if !target_is("CARGO_CFG_TARGET_OS", "linux") {
        return false;
    }
    let required = target_is("CARGO_CFG_TARGET_ENV", "gnu");
    let mut build = cc::Build::new();
    build
        .file("src/knn_region.c")
        .flag("-fopenmp")
        .flag("-pthread")
        .opt_level(2)
        .warnings(true)
        .extra_warnings(true);
    if let Err(err) = build.try_compile("knnregion") {
        if required {
            panic!("OpenMP k-nearest region: compiling src/knn_region.c failed: {err}");
        }
        return false;
    }
    let compiler = build.get_compiler();
    match gomp_dir(compiler.path()) {
        Some(dir) => {
            println!("cargo:rustc-link-search=native={}", dir.display());
            println!("cargo:rustc-link-lib=gomp");
            true
        }
        None if required => panic!("OpenMP k-nearest region: finding libgomp failed"),
        None => false,
    }
}

fn gomp_dir(cc: &std::path::Path) -> Option<std::path::PathBuf> {
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

#[cfg(feature = "scotch")]
fn build_scotch() {
    println!("cargo:rerun-if-changed=src/scotch_part.c");
    let include = std::env::var("SCOTCH_INCLUDE").ok().or_else(|| {
        ["/usr/include/scotch", "/usr/include"]
            .into_iter()
            .find(|dir| std::path::Path::new(dir).join("scotch.h").is_file())
            .map(|dir| dir.to_string())
    });
    let Some(include) = include else {
        panic!("scotch feature needs scotch.h (set SCOTCH_INCLUDE)");
    };
    let mut build = cc::Build::new();
    build
        .file("src/scotch_part.c")
        .include(&include)
        .warnings(true);
    build.compile("linkcell_scotch");
    println!("cargo:rustc-link-lib=scotch");
    println!("cargo:rustc-link-lib=scotcherr");
    println!("cargo:rustc-link-lib=m");
}

#[cfg(feature = "ptscotch")]
fn build_ptscotch() {
    println!("cargo:rerun-if-changed=src/ptscotch_part.c");
    let include = std::env::var("SCOTCH_INCLUDE").unwrap_or_else(|_| "/usr/include/scotch".into());
    cc::Build::new()
        .file("src/ptscotch_part.c")
        .include(&include)
        .include("/usr/lib/x86_64-linux-gnu/openmpi/include")
        .warnings(true)
        .compile("linkcell_ptscotch");
    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu/openmpi/lib");
    println!("cargo:rustc-link-lib=ptscotch");
    println!("cargo:rustc-link-lib=scotch");
    println!("cargo:rustc-link-lib=ptscotcherr");
    println!("cargo:rustc-link-lib=scotcherr");
    println!("cargo:rustc-link-lib=mpi");
    println!("cargo:rustc-link-lib=m");
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
