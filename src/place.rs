//! CPUs for Rayon workers.
//!
//! The caller records the cpus this process may use with `lc_set_places`
//! before it pins itself. Each worker then runs on any of those cpus
//! instead of inheriting the caller's single cpu. A later call replaces
//! the list; a pool already built keeps the cpus it was started with.
//!
//! A worker is not tied to one cpu of the list. Work stealing evens out
//! the load only while the scheduler can move a worker off a cpu another
//! process is using: tied one to a cpu each beside an 8-thread spinner,
//! the 110592-molecule k-nearest search took 78 ms instead of 36. On an
//! idle machine the two placements time the same.

use std::ffi::c_int;
use std::sync::Mutex;

static PLACES: Mutex<Vec<i32>> = Mutex::new(Vec::new());

/// CPUs Rayon workers may use. Every worker may run on any of them.
/// Replaces any list recorded earlier. A pool that has already started
/// keeps the cpus it was built with. `n <= 0` is ignored.
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
fn bind_cpus(cpus: &[i32]) {
    #[cfg(target_os = "linux")]
    {
        extern "C" {
            fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const u8) -> i32;
        }
        let mut mask = [0u8; 128];
        let mut any = false;
        for &cpu in cpus {
            let Ok(cpu) = usize::try_from(cpu) else {
                continue;
            };
            if let Some(byte) = mask.get_mut(cpu / 8) {
                *byte |= 1u8 << (cpu % 8);
                any = true;
            }
        }
        if any {
            unsafe {
                sched_setaffinity(0, mask.len(), mask.as_ptr());
            }
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = cpus;
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
    let places = std::sync::Arc::new(places);
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .spawn_handler(move |thread| {
            let cpus = std::sync::Arc::clone(&places);
            let mut builder = std::thread::Builder::new();
            if let Some(name) = thread.name() {
                builder = builder.name(name.to_string());
            }
            builder.spawn(move || {
                bind_cpus(&cpus);
                thread.run();
            })?;
            Ok(())
        })
        .build()
        .unwrap_or_else(|_| fallback())
}

#[cfg(all(test, feature = "parallel", target_os = "linux"))]
mod tests {
    use super::*;

    extern "C" {
        fn sched_getaffinity(pid: i32, cpusetsize: usize, mask: *mut u8) -> i32;
    }

    fn allowed() -> Vec<i32> {
        let mut mask = [0u8; 128];
        if unsafe { sched_getaffinity(0, mask.len(), mask.as_mut_ptr()) } != 0 {
            return Vec::new();
        }
        (0..mask.len() * 8)
            .filter(|cpu| mask[cpu / 8] & (1u8 << (cpu % 8)) != 0)
            .map(|cpu| cpu as i32)
            .collect()
    }

    #[test]
    fn every_worker_may_use_every_listed_cpu() {
        let cpus = allowed();
        if cpus.len() < 2 {
            return;
        }
        let list: Vec<c_int> = cpus[..2].iter().map(|&cpu| cpu as c_int).collect();
        lc_set_places(list.as_ptr(), 2);
        let seen = build_pool(3).broadcast(|_| allowed());
        assert!(seen.iter().all(|mask| mask == &cpus[..2]));
    }
}
