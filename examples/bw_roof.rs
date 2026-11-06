//! Memory roof and the LLC counters this guest can open.
//!
//! Hardware events are missing on this kernel, so the roof is a timed
//! copy. Eight threads write disjoint slices of one buffer.

use std::time::Instant;

fn copy_gbps(threads: usize, bytes: usize) -> f64 {
    let n = bytes / 8;
    let chunk = (n + threads - 1) / threads;
    let mut srcs = Vec::new();
    let mut dsts = Vec::new();
    for t in 0..threads {
        let lo = t * chunk;
        let hi = ((t + 1) * chunk).min(n);
        srcs.push(vec![1u64; hi - lo]);
        dsts.push(vec![0u64; hi - lo]);
    }
    let once = |srcs: &mut [Vec<u64>], dsts: &mut [Vec<u64>]| {
        std::thread::scope(|scope| {
            for (src, dst) in srcs.iter().zip(dsts.iter_mut()) {
                scope.spawn(move || {
                    dst.copy_from_slice(src);
                });
            }
        });
    };
    once(&mut srcs, &mut dsts);
    let reps = 4usize;
    let t0 = Instant::now();
    for _ in 0..reps {
        once(&mut srcs, &mut dsts);
    }
    let sec = t0.elapsed().as_secs_f64() / reps as f64;
    // Read plus write.
    (2.0 * bytes as f64) / sec / 1e9
}

fn perf_open(typ: u32, config: u64) -> (i64, i32) {
    #[repr(C)]
    struct Attr {
        typ: u32,
        size: u32,
        config: u64,
        period: u64,
        sample_type: u64,
        read_format: u64,
        flags: u64,
        wakeup: u32,
        bp_type: u32,
        bp_addr: u64,
    }
    extern "C" {
        fn syscall(nr: i64, ...) -> i64;
    }
    let mut attr = Attr {
        typ,
        size: 64,
        config,
        period: 0,
        sample_type: 0,
        read_format: 0,
        flags: 0,
        wakeup: 0,
        bp_type: 0,
        bp_addr: 0,
    };
    let fd = unsafe { syscall(298, &mut attr as *mut Attr, 0i32, -1i32, -1i32, 0u64) };
    let err = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
    if fd >= 0 {
        unsafe { close(fd as i32) };
    }
    (fd, err)
}

extern "C" {
    fn close(fd: i32) -> i32;
}

fn main() {
    // PERF_TYPE_HARDWARE cache-misses is config 3. LLC read-miss is the
    // cache event (level 2, op read, result miss).
    let (fd_miss, err_miss) = perf_open(0, 3);
    let llc = 2u64 | (0u64 << 8) | (1u64 << 16);
    let (fd_llc, err_llc) = perf_open(3, llc);
    println!("perf hw_cache_misses fd={fd_miss} errno={err_miss}");
    println!("perf llc_read_miss fd={fd_llc} errno={err_llc}");
    let line = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index3/shared_cpu_list")
        .unwrap_or_default();
    println!("l3_shared_cpu_list {}", line.trim());
    let nodes = std::fs::read_dir("/sys/devices/system/node")
        .map(|d| {
            d.filter_map(|e| e.ok())
                .filter(|e| e.file_name().to_string_lossy().starts_with("node"))
                .count()
        })
        .unwrap_or(0);
    println!("numa_nodes {nodes}");
    // Larger than the 320 MiB last-level cache, so the copy leaves that cache.
    let bytes = 512usize << 20;
    for threads in [1usize, 2, 4, 8] {
        let gbps = copy_gbps(threads, bytes);
        println!("copy threads={threads} bytes={bytes} gbps={gbps:.3}");
    }
}
