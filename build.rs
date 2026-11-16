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
    // Clang, as `cc`, passes `--target=<rust-triple>` and then does not
    // see gcc's `omp.h`. Honour `CC` when it is set. A native build with
    // `CC` unset keeps gcc, the compiler this region already used. A
    // cross build leaves the choice to the `cc` crate.
    if std::env::var_os("CC").is_none()
        && std::env::var("HOST").ok() == std::env::var("TARGET").ok()
    {
        build.compiler("gcc");
    }
    build
        .file("src/knn_region.c")
        .flag("-fopenmp")
        .flag("-pthread")
        .opt_level(2)
        .warnings(true)
        .extra_warnings(true);
    // x86-64 baseline. The host gcc otherwise vectorises this file to
    // AVX2 and BMI2, which the ISA audit rejects. The region only
    // schedules; the wider code is not part of the build.
    if target_is("CARGO_CFG_TARGET_ARCH", "x86_64") {
        build
            .flag("-march=x86-64")
            .flag("-mtune=generic")
            .flag("-fno-tree-vectorize");
    }
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
