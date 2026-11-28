#![cfg(all(feature = "capi", target_os = "linux"))]
//! The row grid from a C host and a C++ host.
//!
//! `tests/row_grid.c` compiles against the headers alone and loads the
//! shared library at run time, which is how a host without a Rust
//! toolchain uses linkcell. `tests/row_grid.cpp` uses the C++ wrapper of
//! `linkcell.hpp` and links the library. Both compare every row with the
//! pair list and exit non-zero on a difference.

use std::path::{Path, PathBuf};
use std::process::Command;

fn run(cmd: &mut Command) -> String {
    let shown = format!("{cmd:?}");
    let output = cmd
        .output()
        .unwrap_or_else(|err| panic!("failed to start {shown}: {err}"));
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    if !output.status.success() {
        panic!(
            "command failed ({})\n{shown}\nstdout:\n{stdout}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr),
        );
    }
    stdout
}

fn host_triple() -> String {
    let output = Command::new("rustc")
        .args(["-vV"])
        .output()
        .expect("rustc -vV");
    let text = String::from_utf8_lossy(&output.stdout);
    text.lines()
        .find_map(|line| line.strip_prefix("host: "))
        .expect("rustc -vV host")
        .to_string()
}

fn compiler(cpp: bool) -> cc::Tool {
    let host = host_triple();
    // cc::Build reads Cargo build-script variables, which an integration
    // test does not have.
    static CLEAR_CARGO_BUILD_ENV: std::sync::Once = std::sync::Once::new();
    CLEAR_CARGO_BUILD_ENV.call_once(|| {
        std::env::remove_var("TARGET");
        std::env::remove_var("HOST");
        std::env::remove_var("OPT_LEVEL");
    });
    let mut build = cc::Build::new();
    build
        .cargo_metadata(false)
        .cpp(cpp)
        .opt_level(1)
        .debug(false)
        .host(&host)
        .target(&host);
    if cpp {
        build.std("c++17");
    } else {
        build.std("c11");
    }
    build.get_compiler()
}

/// The directory cargo wrote the shared library to: the parent of the
/// `deps` directory this test runs from.
fn library_dir() -> PathBuf {
    let exe = std::env::current_exe().expect("test executable path");
    let dir = exe
        .parent()
        .and_then(Path::parent)
        .expect("target profile directory")
        .to_path_buf();
    assert!(
        dir.join("liblinkcell.so").is_file(),
        "no liblinkcell.so in {}",
        dir.display()
    );
    dir
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("linkcell-row-grid-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");
    dir
}

#[test]
fn a_c_host_loads_the_library_and_reads_the_rows() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let library = library_dir().join("liblinkcell.so");
    let dir = scratch("c");
    let exe = dir.join("row_grid");
    // The headers and the system loader: no linkcell library on the line.
    let mut cmd = compiler(false).to_command();
    cmd.arg("-D_DEFAULT_SOURCE")
        .arg("-I")
        .arg(root.join("include"))
        .arg(root.join("tests/row_grid.c"))
        .arg("-o")
        .arg(&exe)
        .arg("-ldl");
    run(&mut cmd);
    let needed = run(Command::new("readelf").arg("-d").arg(&exe));
    assert!(
        !needed.contains("linkcell"),
        "the C host must not link linkcell:\n{needed}"
    );
    let out = run(Command::new(&exe).arg(&library));
    let stamp = format!(
        "abi library=linkcell header={0}.{1}.{2} loaded={0}.{1}.{2}",
        linkcell::LC_ABI_MAJOR,
        linkcell::LC_ABI_MINOR,
        linkcell::LC_ABI_LAYOUT_REVISION
    );
    assert!(out.contains(&stamp), "{out}");
    assert!(out.contains("status=verified"), "{out}");
    assert!(out.contains("row_grid: ok"), "{out}");
    // The same host through the environment variable.
    let out = run(Command::new(&exe).env("LINKCELL_LIBRARY", &library));
    assert!(out.contains("row_grid: ok"), "{out}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_cpp_host_reads_the_rows_through_the_wrapper() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let libs = library_dir();
    let dir = scratch("cpp");
    let exe = dir.join("row_grid_cpp");
    let mut cmd = compiler(true).to_command();
    cmd.arg("-DLINKCELL_EH_STATIC")
        .arg("-I")
        .arg(root.join("include"))
        .arg(root.join("tests/row_grid.cpp"))
        .arg(root.join("src/error.cpp"))
        .arg("-o")
        .arg(&exe)
        .arg("-L")
        .arg(&libs)
        .arg("-llinkcell")
        .arg(format!("-Wl,-rpath,{}", libs.display()))
        .arg("-pthread");
    run(&mut cmd);
    let out = run(Command::new(&exe).env("LD_LIBRARY_PATH", &libs));
    assert!(out.contains("row_grid_cpp: ok"), "{out}");
    let _ = std::fs::remove_dir_all(&dir);
}
