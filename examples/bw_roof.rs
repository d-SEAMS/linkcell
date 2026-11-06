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
    // Payload bytes only. A temporal store of a cold line also reads it
    // (write allocate). A non-temporal store does not.
    for threads in [1usize, 2, 4, 8] {
        for kind in ["line64", "pair40", "soa"] {
            let temporal = stream_gbps(kind, threads, bytes, false);
            let nontemporal = stream_gbps(kind, threads, bytes, true);
            println!(
                "store kind={kind} threads={threads} temporal_gbps={temporal:.3} nontemporal_gbps={nontemporal:.3}"
            );
        }
    }
}

fn stream_gbps(kind: &str, threads: usize, bytes: usize, nontemporal: bool) -> f64 {
    let chunk = (bytes + threads - 1) / threads;
    let mut bufs = Vec::new();
    for _ in 0..threads {
        // 64 bytes of headroom so a slice can be aligned.
        let mut raw = vec![0u8; chunk + 128];
        let addr = raw.as_mut_ptr() as usize;
        let aligned = (addr + 63) & !63;
        let off = aligned - addr;
        bufs.push((raw, off, chunk));
    }
    let once = |bufs: &mut [(Vec<u8>, usize, usize)]| {
        std::thread::scope(|scope| {
            for (raw, off, len) in bufs.iter_mut() {
                let addr = unsafe { raw.as_mut_ptr().add(*off) } as usize;
                let len = *len;
                scope.spawn(move || {
                    fill_bytes(kind, addr as *mut u8, len, nontemporal);
                });
            }
        });
    };
    once(&mut bufs);
    let reps = 4usize;
    let t0 = Instant::now();
    for _ in 0..reps {
        once(&mut bufs);
    }
    let sec = t0.elapsed().as_secs_f64() / reps as f64;
    bytes as f64 / sec / 1e9
}

fn fill_bytes(kind: &str, ptr: *mut u8, len: usize, nontemporal: bool) {
    match kind {
        "line64" => fill_lines(ptr, len, nontemporal),
        "pair40" => fill_pairs(ptr, len, nontemporal),
        "soa" => fill_soa(ptr, len, nontemporal),
        _ => {}
    }
}

fn fill_lines(ptr: *mut u8, len: usize, nontemporal: bool) {
    let n = len / 64;
    if nontemporal {
        unsafe { stream_lines(ptr, n) };
    } else {
        unsafe {
            for i in 0..n {
                let p = ptr.add(i * 64) as *mut u64;
                let v = std::hint::black_box(i as u64);
                for lane in 0..8 {
                    std::ptr::write(p.add(lane), v);
                }
            }
        }
    }
}

fn fill_pairs(ptr: *mut u8, len: usize, nontemporal: bool) {
    let nbytes = (len / 40) * 40;
    unsafe {
        if nontemporal {
            // The packed records stay 40 bytes. Streaming stores still move
            // whole aligned lines of that byte range.
            stream_lines(ptr, nbytes / 64);
        } else {
            let n = nbytes / 40;
            for i in 0..n {
                let p = ptr.add(i * 40) as *mut u64;
                let v = std::hint::black_box(i as u64);
                std::ptr::write(p, v);
                std::ptr::write(p.add(1), v);
                std::ptr::write(p.add(2), v);
                std::ptr::write(p.add(3), v);
                std::ptr::write(p.add(4), v);
            }
        }
    }
}

fn fill_soa(ptr: *mut u8, len: usize, nontemporal: bool) {
    // Four columns of a 28-byte row: i, j, shift, dist2. Each column
    // starts on a 64-byte boundary so a streaming store is aligned.
    let align = |n: usize| (n + 63) & !63;
    let mut rows = len / 28;
    loop {
        let need = align(rows * 4) + align(rows * 4) + align(rows * 12) + align(rows * 8);
        if need <= len || rows == 0 {
            break;
        }
        rows -= 1;
    }
    if rows == 0 {
        return;
    }
    let widths = [rows * 4, rows * 4, rows * 12, rows * 8];
    let mut at = 0usize;
    for bytes in widths {
        unsafe {
            if nontemporal {
                stream_lines(ptr.add(at), bytes / 64);
            } else {
                let words = bytes / 8;
                let p = ptr.add(at) as *mut u64;
                for i in 0..words {
                    std::ptr::write(p.add(i), std::hint::black_box(i as u64));
                }
            }
        }
        at += align(bytes);
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn stream_lines(ptr: *mut u8, n: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::{_mm512_setzero_si512, _mm512_stream_si512, _mm_sfence};
        let z = _mm512_setzero_si512();
        let p = ptr as *mut std::arch::x86_64::__m512i;
        for i in 0..n {
            _mm512_stream_si512(p.add(i), z);
        }
        _mm_sfence();
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (ptr, n);
    }
}
