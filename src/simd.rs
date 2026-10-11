//! Runtime SIMD path for the mesh fold, the k-nearest scan, and cutoff pairs.
//!
//! `LINKCELL_SIMD` forces `scalar`, `avx2`, `avx512`, or `neon`.
//! An unknown name is ignored. A name this CPU cannot run selects
//! the scalar path. Unset, the widest usable path is used: AVX-512,
//! then AVX2 with FMA, then NEON, then scalar.
//!
//! AVX-512 is taken only when AVX-512F, AVX-512VL, POPCNT, and FMA are
//! all present. A CPU with AVX-512F but not VL uses AVX2. AVX2 is taken
//! only when FMA is present too, so Zen 2/3 and Haswell use it and
//! Sandy Bridge stays on the scalar path. FMA is a selection gate for
//! the fold and the scan: those kernels multiply and add separately.
//! Pair screening may use FMA inside the margin test. No fused
//! multiply-add reaches a stored value. The floor in the fold is
//! minimage's `floor_fast` for this architecture: truncation on x86_64,
//! `f64::floor` elsewhere.

use std::cell::Cell;
use std::sync::OnceLock;

/// Width of the mesh fold, the SoA distance scan, and the pair tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Path {
    /// One point at a time.
    Scalar,
    /// Four f64 lanes. Requires AVX2 and FMA.
    Avx2,
    /// Eight f64 lanes. Requires AVX-512F, AVX-512VL, POPCNT, and FMA.
    Avx512,
    /// Two f64 lanes. Requires NEON.
    Neon,
}

impl Path {
    /// Stable name for logs and [`crate::simd_path`].
    pub(crate) fn name(self) -> &'static str {
        match self {
            Path::Scalar => "scalar",
            Path::Avx2 => "avx2",
            Path::Avx512 => "avx512",
            Path::Neon => "neon",
        }
    }

    /// This build contains the kernel and this CPU can run it.
    pub(crate) fn usable(self) -> bool {
        match self {
            Path::Scalar => true,
            Path::Avx2 => avx2_ready(),
            Path::Avx512 => avx512_ready(),
            Path::Neon => neon_ready(),
        }
    }
}

thread_local! {
    // `const { ... }` is newer than this crate's 1.70 floor.
    #[allow(clippy::missing_const_for_thread_local)]
    static FORCE: Cell<Option<Path>> = Cell::new(None);
}

/// Path for the next fold or scan.
///
/// A test override wins, then `LINKCELL_SIMD`, then [`native`].
pub(crate) fn select() -> Path {
    if let Some(path) = FORCE.with(|slot| slot.get()) {
        return if path.usable() { path } else { Path::Scalar };
    }
    if let Some(path) = env_path() {
        return path;
    }
    native()
}

/// Widest path this CPU can run. Ignores `LINKCELL_SIMD`.
pub(crate) fn native() -> Path {
    if avx512_ready() {
        Path::Avx512
    } else if avx2_ready() {
        Path::Avx2
    } else if neon_ready() {
        Path::Neon
    } else {
        Path::Scalar
    }
}

fn env_path() -> Option<Path> {
    static CACHED: OnceLock<Option<Path>> = OnceLock::new();
    *CACHED.get_or_init(|| {
        let raw = std::env::var("LINKCELL_SIMD").ok()?;
        let path = parse(raw.trim())?;
        Some(if path.usable() { path } else { Path::Scalar })
    })
}

pub(crate) fn parse(name: &str) -> Option<Path> {
    match name {
        "scalar" | "0" => Some(Path::Scalar),
        "avx2" | "avx2fma" => Some(Path::Avx2),
        "avx512" | "avx512f" => Some(Path::Avx512),
        "neon" => Some(Path::Neon),
        _ => None,
    }
}

fn avx2_ready() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

fn avx512_ready() -> bool {
    #[cfg(all(target_arch = "x86_64", linkcell_avx512))]
    {
        // Pair kernels enable every one of these. Fold and scan need
        // only AVX-512F, and they follow this gate so one path is shared.
        std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512vl")
            && std::is_x86_feature_detected!("popcnt")
            && std::is_x86_feature_detected!("fma")
    }
    #[cfg(not(all(target_arch = "x86_64", linkcell_avx512)))]
    {
        false
    }
}

fn neon_ready() -> bool {
    // NEON is baseline on aarch64. A runtime check would not change the answer.
    cfg!(target_feature = "neon")
}

/// Name of the path [`select`] would use.
pub fn simd_path() -> &'static str {
    select().name()
}

#[cfg(feature = "timing")]
static SCAN_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(feature = "timing")]
static KNN_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(feature = "timing")]
static MESH_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `LINKCELL_SCAN_TIME` is set. Read once.
///
/// This arms a timer around each source's SoA scan.
#[cfg(feature = "timing")]
pub(crate) fn scan_timing_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("LINKCELL_SCAN_TIME").is_some())
}

/// `LINKCELL_COARSE_TIME` is set. Read once.
///
/// This arms one timer around [`crate::knearest_into`] and one around
/// the mesh build. It does not time each source.
#[cfg(feature = "timing")]
pub(crate) fn coarse_timing_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("LINKCELL_COARSE_TIME").is_some())
}

#[cfg(feature = "timing")]
pub(crate) fn add_scan_ns(ns: u128) {
    SCAN_NS.fetch_add(ns as u64, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(feature = "timing")]
pub(crate) fn add_knn_ns(ns: u128) {
    KNN_NS.fetch_add(ns as u64, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(feature = "timing")]
pub(crate) fn add_mesh_ns(ns: u128) {
    MESH_NS.fetch_add(ns as u64, std::sync::atomic::Ordering::Relaxed);
}

/// Nanoseconds accumulated since process start.
///
/// `mesh_ns` is the calling thread's wait for the mesh build inside
/// [`crate::knearest_into`]. `knn_ns` is that search after the output
/// buffers are filled, on the calling thread. `scan_ns` is the SoA
/// distance scan. With one thread the scan total is a slice of the
/// search total. With more threads the scan total sums every worker,
/// so it can exceed the wall time.
///
/// All three stay zero unless the matching variable is set before the
/// first search. `LINKCELL_SCAN_TIME` counts the scan.
/// `LINKCELL_COARSE_TIME` counts the search and the mesh build.
#[cfg(feature = "timing")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SimdTiming {
    /// Mesh build inside [`crate::knearest_into`], on the calling thread.
    pub mesh_ns: u64,
    /// SoA distance scan. Sums every worker when the search is parallel.
    pub scan_ns: u64,
    /// [`crate::knearest_into`] after the output buffers are filled.
    pub knn_ns: u64,
}

/// Current [`SimdTiming`] counters.
#[cfg(feature = "timing")]
pub fn simd_timing() -> SimdTiming {
    SimdTiming {
        mesh_ns: MESH_NS.load(std::sync::atomic::Ordering::Relaxed),
        scan_ns: SCAN_NS.load(std::sync::atomic::Ordering::Relaxed),
        knn_ns: KNN_NS.load(std::sync::atomic::Ordering::Relaxed),
    }
}

/// Restores the previous test override, including on panic.
#[cfg(test)]
pub(crate) struct ForceGuard {
    prev: Option<Path>,
}

/// Force [`select`] on this thread until the guard drops.
///
/// Rayon jobs must capture [`select`] on the calling thread. Workers
/// do not see this thread's override.
#[cfg(test)]
pub(crate) fn force(path: Path) -> ForceGuard {
    let prev = FORCE.with(|slot| slot.replace(Some(path)));
    ForceGuard { prev }
}

#[cfg(test)]
impl Drop for ForceGuard {
    fn drop(&mut self) {
        let prev = self.prev;
        FORCE.with(|slot| slot.set(prev));
    }
}

/// Scalar plus every wide path this CPU can run.
#[cfg(test)]
pub(crate) fn paths_to_compare() -> Vec<Path> {
    let mut out = vec![Path::Scalar];
    for path in [Path::Avx2, Path::Avx512, Path::Neon] {
        if path.usable() {
            out.push(path);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_names() {
        assert_eq!(parse("scalar"), Some(Path::Scalar));
        assert_eq!(parse("0"), Some(Path::Scalar));
        assert_eq!(parse("avx2"), Some(Path::Avx2));
        assert_eq!(parse("avx2fma"), Some(Path::Avx2));
        assert_eq!(parse("avx512"), Some(Path::Avx512));
        assert_eq!(parse("avx512f"), Some(Path::Avx512));
        assert_eq!(parse("neon"), Some(Path::Neon));
        assert_eq!(parse("avx"), None);
        assert_eq!(parse(""), None);
    }

    #[test]
    fn native_path_is_usable() {
        let path = native();
        assert!(path.usable());
        assert_eq!(path.name(), simd_path());
    }

    #[cfg(feature = "timing")]
    #[test]
    fn simd_timing_names_mesh_scan_and_search() {
        assert!(std::env::var_os("LINKCELL_SCAN_TIME").is_none());
        assert!(std::env::var_os("LINKCELL_COARSE_TIME").is_none());
        let t = simd_timing();
        assert_eq!(
            t,
            SimdTiming {
                mesh_ns: 0,
                scan_ns: 0,
                knn_ns: 0,
            }
        );
    }
}
