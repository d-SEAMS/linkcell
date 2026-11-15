//! CPUs for Rayon workers.
//!
//! The caller records the cpus this process may use with `lc_set_places`
//! before it pins itself. Workers then take those cpus instead of
//! inheriting the caller's single cpu. A later call replaces the list;
//! a pool already built keeps the cpus it was started with.

use std::ffi::c_int;
#[cfg(feature = "parallel")]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

static PLACES: Mutex<Vec<i32>> = Mutex::new(Vec::new());

/// CPUs Rayon workers may use. Replaces any list recorded earlier.
/// A pool that has already started keeps the cpus it was built with.
/// `n <= 0` is ignored.
#[no_mangle]
pub extern "C" fn lc_set_places(cpus: *const c_int, n: c_int) {
    if cpus.is_null() || n <= 0 {
        return;
    }
    let incoming = unsafe { std::slice::from_raw_parts(cpus, n as usize) };
    let mut guard = PLACES.lock().unwrap_or_else(|err| err.into_inner());
    guard.clear();
    guard.extend_from_slice(incoming);
}

#[cfg(feature = "parallel")]
fn places_snapshot() -> Vec<i32> {
    PLACES.lock().unwrap_or_else(|err| err.into_inner()).clone()
}

#[cfg(feature = "parallel")]
fn bind_cpu(cpu: i32) {
    #[cfg(target_os = "linux")]
    {
        extern "C" {
            fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const u8) -> i32;
        }
        if cpu < 0 {
            return;
        }
        let cpu = cpu as usize;
        let mut mask = [0u8; 128];
        let byte = cpu / 8;
        if byte >= mask.len() {
            return;
        }
        mask[byte] = 1u8 << (cpu % 8);
        unsafe {
            sched_setaffinity(0, mask.len(), mask.as_ptr());
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = cpu;
    }
}

#[cfg(feature = "parallel")]
pub(crate) fn build_pool(n: usize) -> rayon::ThreadPool {
    let places = places_snapshot();
    let fallback = || {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("rayon")
    };
    if places.is_empty() {
        return rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .unwrap_or_else(|_| fallback());
    }
    let next = AtomicUsize::new(0);
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .spawn_handler(move |thread| {
            let cpu = places[next.fetch_add(1, Ordering::Relaxed) % places.len()];
            let mut builder = std::thread::Builder::new();
            if let Some(name) = thread.name() {
                builder = builder.name(name.to_string());
            }
            builder.spawn(move || {
                bind_cpu(cpu);
                thread.run();
            })?;
            Ok(())
        })
        .build()
        .unwrap_or_else(|_| fallback())
}
