//! Runtime SIMD path for the mesh fold and the k-nearest scan.
//!
//! `LINKCELL_SIMD` forces `scalar`, `avx2`, `avx512`, or `neon`.
//! An unknown name is ignored. A name this CPU cannot run selects
//! the scalar path. Unset, the widest usable path is used: AVX-512F,
//! then AVX2 with FMA, then NEON, then scalar.
//!
//! AVX2 is taken only when FMA is present too, so Zen 2/3 and Haswell
//! use it and Sandy Bridge stays on the scalar path. The kernels still
//! multiply and add separately. A fused multiply-add would not match
//! [`crate::Cell::fractional`] bit for bit.

use std::cell::Cell;
use std::sync::OnceLock;

/// Width of the mesh fold and of the SoA distance scan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Path {
    /// One point at a time.
    Scalar,
    /// Four f64 lanes. Requires AVX2 and FMA.
    Avx2,
    /// Eight f64 lanes. Requires AVX-512F.
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
        std::is_x86_feature_detected!("avx512f")
    }
    #[cfg(not(all(target_arch = "x86_64", linkcell_avx512)))]
    {
        false
    }
}

fn neon_ready() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("neon")
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

/// Name of the path [`select`] would use.
pub fn simd_path() -> &'static str {
    select().name()
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
}
